import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import subprocess
import sys
import platform
import os
import json
import concurrent.futures

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xrp_alert_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Parameters ---
PRODUCT_ID = "XRP-PERP-INTX"
MARGIN = 250
LEVERAGE = 20
ENTRY_ZONE_LOW = 2.93
ENTRY_ZONE_HIGH = 2.95
STOP_LOSS = 2.80
PROFIT_TARGET = 3.30
VOLUME_PERIOD = 20
TRIGGER_STATE_FILE = "xrp_breakout_trigger_state.json"

# --- Alert sound ---
def play_alert_sound(filename="alert_sound.wav"):
    try:
        system = platform.system()
        if system == "Darwin":
            cmd = ["afplay", filename]
        elif system == "Linux":
            cmd = ["aplay", filename]
        elif system == "Windows":
            cmd = ["start", "/min", "cmd", "/c", f"powershell -c \"(New-Object Media.SoundPlayer '{filename}').PlaySync()\""]
        else:
            logger.warning(f"Unknown operating system: {system}. Cannot play sound.")
            return False
        subprocess.run(cmd, check=True, timeout=5)
        logger.info("Alert sound played successfully")
        return True
    except Exception as e:
        logger.error(f"Error playing alert sound: {e}")
        return False

# --- Coinbase setup ---
def setup_coinbase():
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    return CoinbaseService(api_key, api_secret)

# --- Candle fetch with retry ---
def safe_get_candles(cb_service, product_id, start_ts, end_ts, granularity):
    def _get_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity=granularity
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    for _ in range(3):
        try:
            return _get_candles()
        except Exception as e:
            logger.warning(f"Retrying candle fetch: {e}")
            time.sleep(2)
    return []

# --- Trade execution ---
def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = MARGIN, leverage: int = LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
    try:
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.4f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        position_size_usd = margin * leverage
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', product,
            '--side', side,
            '--size', str(position_size_usd),
            '--leverage', str(leverage),
            '--tp', str(take_profit),
            '--sl', str(stop_loss),
            '--no-confirm'
        ]
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info(f"Crypto {trade_type} trade executed successfully!")
            logger.info(f"Trade output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"Crypto {trade_type} trade failed!")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logger.error(f"Error executing crypto {trade_type} trade: {e}")
        return False, str(e)

# --- Trigger state ---
def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None}
    return {"triggered": False, "trigger_ts": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

# --- Main alert logic ---
def xrp_breakout_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting XRP breakout alert ===")
    GRANULARITY = "ONE_HOUR"
    periods_needed = VOLUME_PERIOD + 2
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        # Get the last completed hour
        last_completed_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        start = last_completed_hour - timedelta(hours=periods_needed-1)
        end = last_completed_hour
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Time range: {start} to {end} (last completed hour)")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        logger.info(f"Candles fetched: {len(candles) if candles else 0} candles")
        if not candles or len(candles) < periods_needed:
            logger.warning("Not enough XRP 1h candle data for breakout alert.")
            logger.info("=== xrp_breakout_alert completed (insufficient data) ===")
            return last_alert_ts
        # Use the most recent completed candle
        last_candle = candles[0]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        low = float(last_candle['low'])
        v0 = float(last_candle['volume'])
        historical_candles = candles[1:VOLUME_PERIOD+1]
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles) if historical_candles else 0
        logger.info(f"Candle timestamp: {ts} (UTC)")
        logger.info(f"Candle data: close=${close:.4f}, high=${high:.4f}, low=${low:.4f}, volume={v0:,.0f}, avg20={avg20:,.0f}")
        # --- Reporting ---
        logger.info("")
        logger.info(f"Entry zone: ${ENTRY_ZONE_LOW:.2f} – ${ENTRY_ZONE_HIGH:.2f}")
        logger.info(f"Stop-loss: ${STOP_LOSS:.2f}")
        logger.info(f"Profit target 1: ${PROFIT_TARGET:.2f} (projected from triangle breakout height)")
        logger.info("Facts: Institutional interest trending, XRP testing resistance with edge into breakout zone")
        logger.info("Opinion: Higher beta—enter half-size, avoid if ETH and BTC occupied capital.")
        logger.info("")
        logger.info(f"Candle close: ${close:.4f}, High: ${high:.4f}, Low: ${low:.4f}, Volume: {v0:,.0f}, Avg(20): {avg20:,.0f}")
        # --- Entry logic ---
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        volume_ok = v0 >= 2 * avg20 if avg20 > 0 else False
        logger.info(f"Entry conditions: in_zone={in_entry_zone}, volume_ok={volume_ok} (v0={v0:.0f} >= 2*avg20={2*avg20:.0f})")
        if in_entry_zone and volume_ok and not trigger_state.get("triggered", False):
            logger.info("Entry conditions met - preparing to execute trade...")
            logger.info(f"Entry condition met: close (${close:.4f}) is within entry zone (${ENTRY_ZONE_LOW:.2f}-${ENTRY_ZONE_HIGH:.2f}) and volume > 2x 20-SMA. Taking trade.")
            logger.info("Playing alert sound...")
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            logger.info("Executing crypto trade...")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="XRP-USD breakout entry",
                entry_price=close,
                stop_loss=STOP_LOSS,
                take_profit=PROFIT_TARGET,
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            logger.info(f"Trade execution completed: success={trade_success}")
            if trade_success:
                logger.info(f"XRP-USD breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"XRP-USD breakout trade failed: {trade_result}")
            logger.info("Saving trigger state...")
            trigger_state = {"triggered": True, "trigger_ts": int(last_candle['start'])}
            save_trigger_state(trigger_state)
            logger.info("Trigger state saved")
            logger.info("=== xrp_breakout_alert completed (trade executed) ===")
            return ts
        # Reset trigger if price leaves entry zone or volume drops
        if trigger_state.get("triggered", False):
            if not in_entry_zone or not volume_ok:
                logger.info("Resetting trigger state...")
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(trigger_state)
                logger.info("Trigger state reset")
        logger.info("=== xrp_breakout_alert completed (no trade) ===")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in XRP breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== xrp_breakout_alert completed (with error) ===")
    return last_alert_ts

# --- Main loop ---
def main():
    logger.info("Starting custom XRP breakout alert script")
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    cb_service = setup_coinbase()
    last_alert_ts = None
    def poll_iteration():
        nonlocal last_alert_ts
        iteration_start_time = time.time()
        last_alert_ts = xrp_breakout_alert(cb_service, last_alert_ts)
        logger.info(f"✅ Alert cycle completed in {time.time() - iteration_start_time:.1f} seconds")
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)  # 2 minute max per poll
                    wait_seconds = 60  # 1 minute
                    logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll\n")
                    time.sleep(wait_seconds)
                except concurrent.futures.TimeoutError:
                    logger.error('Polling iteration timed out! Skipping to next.')
        except KeyboardInterrupt:
            logger.info("👋 Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in alert loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(60)

if __name__ == "__main__":
    main() 