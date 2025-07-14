import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
import pandas_ta as ta
import subprocess
import sys
import platform
import os
import json
import concurrent.futures

# Set up file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xrp_alert_debug.log'),
        logging.StreamHandler()  # Keep console output too
    ]
)
logger = logging.getLogger(__name__)

PRODUCT_ID = "XRP-PERP-INTX"
XRP_MARGIN = 300
XRP_LEVERAGE = 20
STOP_LOSS = 2.17
TP1 = 2.38
TP2 = 2.60
ENTRY_TRIGGER = 2.24
ENTRY_ZONE_LOW = 2.23
ENTRY_ZONE_HIGH = 2.26
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.15  # 15% uptick

TRIGGER_STATE_FILE = "xrp_breakout_trigger_state.json"


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

def setup_coinbase():
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    return CoinbaseService(api_key, api_secret)

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

def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = XRP_MARGIN, leverage: int = XRP_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
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

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}
    return {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")


# === NEW XRP BREAKOUT STRATEGY ALERT & TRADE LOGIC (BLOCKS) ===

XRP_STOP_LOSS = 2.58
XRP_TP1 = 3.00
XRP_TP2 = 3.325  # Midpoint of 3.25–3.40
XRP_ENTRY_TRIGGER = 2.73
XRP_ENTRY_ZONE_LOW = 2.73
XRP_ENTRY_ZONE_HIGH = 2.75
XRP_VOLUME_MULTIPLIER = 1.15
XRP_VOLUME_PERIOD = 20
XRP_MARGIN = 300
XRP_LEVERAGE = 20
XRP_HARD_EXIT = 2.70

TRIGGER_STATE_FILE = "xrp_breakout_blocks_state.json"

def load_blocks_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"breakout_seen": False, "retest_fired": False, "breakout_ts": None}
    return {"breakout_seen": False, "retest_fired": False, "breakout_ts": None}

def save_blocks_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def xrp_custom_breakout_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting xrp_custom_breakout_alert ===")
    PRODUCT_ID = "XRP-PERP-INTX"
    GRANULARITY = "ONE_HOUR"
    ENTRY_ZONE_LOW = 2.88
    ENTRY_ZONE_HIGH = 2.94
    STOP_LOSS = 2.73
    PROFIT_TARGET = 3.40
    MARGIN = 300
    LEVERAGE = 20
    VOLUME_PERIOD = 2  # For facts reporting
    periods_needed = VOLUME_PERIOD + 2
    trigger_state = load_blocks_state()  # Reuse state logic for simplicity
    try:
        logger.info("Setting up time parameters...")
        now = datetime.now(UTC)
        # Get the last completed hour, not the current hour
        last_completed_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        start = last_completed_hour - timedelta(hours=periods_needed-1)  # -1 because we want periods_needed including the last completed candle
        end = last_completed_hour
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Time range: {start} to {end} (last completed hour)")
        
        logger.info("Fetching candles from API...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        logger.info(f"Candles fetched: {len(candles) if candles else 0} candles")
        
        if not candles or len(candles) < periods_needed:
            log_emoji("Not enough XRP candle data for custom breakout alert.", "⚠️")
            logger.info("=== xrp_custom_breakout_alert completed (insufficient data) ===")
            return last_alert_ts
            
        logger.info("Processing candle data...")
        # Candles are in reverse chronological order (newest first), so use candles[0]
        last_candle = candles[0]  # Use the most recent completed candle
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        low = float(last_candle['low'])
        v0 = float(last_candle['volume'])
        historical_candles = candles[1:VOLUME_PERIOD+1]  # Use candles 1-3 for historical average
        avg2 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles) if historical_candles else 0
        logger.info(f"Candle timestamp: {ts} (UTC)")
        logger.info(f"Candle data processed: close=${close:.4f}, high=${high:.4f}, low=${low:.4f}")
        
        # Debug: Show all candles for verification
        logger.info("All candles in range:")
        for i, candle in enumerate(candles):
            candle_ts = datetime.fromtimestamp(int(candle['start']), UTC)
            logger.info(f"  Candle {i}: {candle_ts} - Close: ${float(candle['close']):.4f}")
        
        # Get current live price for comparison
        try:
            ticker = cb_service.get_product_ticker(product_id=PRODUCT_ID)
            current_price = float(ticker['price'])
            logger.info(f"Current live price: ${current_price:.4f}")
            logger.info(f"Price difference: ${current_price - close:.4f} (live - candle close)")
        except Exception as e:
            logger.warning(f"Could not get current price: {e}")
        
        # --- Reporting ---
        logger.info("Generating report...")
        log_emoji("", "")
        log_emoji(f"Entry zone: ${ENTRY_ZONE_LOW:.2f} – ${ENTRY_ZONE_HIGH:.2f}", "")
        log_emoji(f"Stop-loss: ${STOP_LOSS:.2f}", "")
        log_emoji(f"1st target: ${PROFIT_TARGET:.2f} (triangle height added)", "")
        log_emoji(f"Facts: clean thrust through $2.84 resistance with ~2x hourly average volume; institutional accumulation flagged. [CoinDesk]", "")
        log_emoji(f"Opinion: higher beta—consider half-size until daily close confirms above $2.95.", "")
        log_emoji("", "")
        log_emoji(f"Candle close: ${close:.4f}, High: ${high:.4f}, Low: ${low:.4f}, Volume: {v0:,.0f}, Avg(2): {avg2:,.0f}", "")
        
        # --- Entry logic ---
        logger.info("Checking entry conditions...")
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        volume_ok = v0 >= 2 * avg2 if avg2 > 0 else False
        logger.info(f"Entry conditions: in_zone={in_entry_zone}, volume_ok={volume_ok}")
        
        if in_entry_zone and volume_ok and not trigger_state.get("triggered", False):
            logger.info("Entry conditions met - preparing to execute trade...")
            log_emoji(f"Entry condition met: close (${close:.4f}) is within entry zone (${ENTRY_ZONE_LOW:.2f}-${ENTRY_ZONE_HIGH:.2f}) and volume > 2x avg. Taking trade.", "🚀")
            
            logger.info("Playing alert sound...")
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                log_emoji(f"Failed to play alert sound: {e}", "❌")
                logger.error(f"Alert sound error: {e}")
            
            logger.info("Executing crypto trade...")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="XRP-USD custom breakout entry",
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
                log_emoji(f"XRP-USD custom breakout trade executed successfully!", "🎉")
                log_emoji(f"Trade output: {trade_result}", "")
            else:
                log_emoji(f"XRP-USD custom breakout trade failed: {trade_result}", "❌")
            
            logger.info("Saving trigger state...")
            # Set trigger to avoid duplicate trades
            trigger_state = {"triggered": True, "trigger_ts": int(last_candle['start'])}
            save_blocks_state(trigger_state)
            logger.info("Trigger state saved")
            
            logger.info("=== xrp_custom_breakout_alert completed (trade executed) ===")
            return ts
            
        # Reset trigger if price leaves entry zone or volume drops
        logger.info("Checking if trigger should be reset...")
        if trigger_state.get("triggered", False):
            if not in_entry_zone or not volume_ok:
                logger.info("Resetting trigger state...")
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_blocks_state(trigger_state)
                logger.info("Trigger state reset")
        
        logger.info("=== xrp_custom_breakout_alert completed (no trade) ===")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in XRP custom breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== xrp_custom_breakout_alert completed (with error) ===")
    return last_alert_ts

# --- Helper: Format with emojis ---
def log_status(msg):
    logger.info(f"\n{'='*40}\n{msg}\n{'='*40}")

def log_emoji(msg, emoji):
    logger.info(f"{emoji} {msg}")

# --- Helper: Get equity for position sizing (stub, can be improved) ---
def get_equity(cb_service):
    try:
        equity, _ = cb_service.get_portfolio_info(portfolio_type="INTX")
        return equity
    except Exception:
        return 20000  # fallback

# --- Helper: Place bracket order for a given size ---
def place_bracket(cb_service, size, tp, sl, leverage, product=PRODUCT_ID):
    return cb_service.place_market_order_with_targets(
        product_id=product,
        side="BUY",
        size=size,
        take_profit_price=tp,
        stop_loss_price=sl,
        leverage=str(leverage)
    )

# --- Helper: Monitor for TP1 fill and place new bracket for TP2 ---
def monitor_tp1_and_rebracket(cb_service, product, original_size, tp1, tp2, sl, leverage):
    # Wait for TP1 fill (poll recent trades)
    log_emoji("Monitoring for TP1 fill (sell half)...", "🔎")
    half_size = original_size / 2
    filled_tp1 = False
    for _ in range(60):  # up to 1 hour (1 min polling)
        trades = cb_service.get_recent_trades()
        for t in trades:
            if t['product_id'] == product and float(t['price']) >= tp1 and float(t['size']) >= half_size * 0.95:
                filled_tp1 = True
                break
        if filled_tp1:
            break
        time.sleep(60)
    if filled_tp1:
        log_emoji(f"TP1 hit! Sold half at ${tp1}", "🎯")
        # Place new bracket for remainder
        log_emoji(f"Placing new bracket for remainder: TP2=${tp2}, SL=${sl}", "🔁")
        result = place_bracket(cb_service, half_size, tp2, sl, leverage, product)
        if result.get("error"):
            log_emoji(f"Error placing TP2 bracket: {result['error']}", "❌")
        else:
            log_emoji(f"TP2 bracket placed: TP2=${tp2}, SL=${sl}", "✅")
    else:
        log_emoji("TP1 not filled within monitoring window.", "⌛")

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
        last_alert_ts = xrp_custom_breakout_alert(cb_service, last_alert_ts)
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