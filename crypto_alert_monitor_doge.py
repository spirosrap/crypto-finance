import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
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
        logging.FileHandler('doge_alert_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 60
BACKOFF_MULTIPLIER = 2

CONNECTION_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
    Exception
)

def exponential_backoff_delay(attempt):
    import random
    delay = min(INITIAL_RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt), MAX_RETRY_DELAY)
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except CONNECTION_ERRORS as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"❌ Permanent failure after {MAX_RETRIES} attempts: {e}")
                return None
            delay = exponential_backoff_delay(attempt)
            logger.warning(f"⚠️ Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"❌ Non-recoverable error: {e}")
            return None
    return None

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
    return retry_with_backoff(_get_candles)

# DOGE double-bottom breakout parameters (from image)
PRODUCT_ID = "DOGE-PERP-INTX"
GRANULARITY = "ONE_HOUR"
VOLUME_PERIOD = 20
BREAKOUT_LEVEL = 0.26  # Buy breakout ≥ $0.26
STOP_LOSS = 0.21
PROFIT_TARGET = 0.46
ENTRY_ZONE_LOW = 0.26  # Confirmed breakout
ENTRY_ZONE_HIGH = 0.27  # Small buffer above breakout
DOGE_MARGIN = 250  # USD
DOGE_LEVERAGE = 10  # Lower leverage for DOGE
TRIGGER_STATE_FILE = "doge_doublebottom_trigger_state.json"
VOLUME_SURGE_FACTOR = 1.0  # Must be > 20-period average
EMA_PERIOD = 20


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
    except subprocess.TimeoutExpired:
        logger.error("Sound playback timed out")
        return False
    except Exception as e:
        logger.error(f"Error playing alert sound: {e}")
        return False

def setup_coinbase():
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    def _create_service():
        service = CoinbaseService(api_key, api_secret)
        try:
            test_response = service.client.get_public_candles(
                product_id=PRODUCT_ID,
                start=int((datetime.now(UTC) - timedelta(days=2)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity=GRANULARITY
            )
            logger.info("✅ Coinbase connection validated successfully")
            return service
        except Exception as e:
            logger.error(f"❌ Failed to validate Coinbase connection: {e}")
            raise
    service = retry_with_backoff(_create_service)
    if service is None:
        raise ConnectionError("Failed to establish Coinbase connection after retries")
    return service

def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = DOGE_MARGIN, leverage: int = DOGE_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
    def _execute_trade():
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
    try:
        result = retry_with_backoff(_execute_trade)
        if result is None:
            return False, "Failed after multiple retry attempts"
        return result
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error executing crypto {trade_type} trade: {e}")
        return False, str(e)

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

# --- DOGE Double-Bottom Breakout Alert Logic ---
def doge_doublebottom_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting DOGE Double-Bottom Breakout Alert ($0.26 breakout) ===")
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        # Get last 30 days of daily candles for double-bottom detection
        end = now
        start = now - timedelta(days=30)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Fetching daily candles for double-bottom detection...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, "ONE_DAY")
        if not candles or len(candles) < VOLUME_PERIOD + 2:
            logger.warning("Not enough daily candle data for double-bottom detection.")
            return last_alert_ts
        # Sort by timestamp ascending
        candles = sorted(candles, key=lambda x: int(x['start']))
        closes = [float(c['close']) for c in candles]
        lows = [float(c['low']) for c in candles]
        volumes = [float(c['volume']) for c in candles]
        # --- Double-bottom detection ---
        # Find two lowest points in last 20 days (approximate double-bottom)
        recent_lows = pd.Series(lows[-(VOLUME_PERIOD+2):])
        low_indices = recent_lows.nsmallest(2).index.tolist()
        if len(low_indices) < 2:
            logger.info("Not enough lows for double-bottom pattern.")
            return last_alert_ts
        low1, low2 = low_indices[0], low_indices[1]
        low1_val, low2_val = recent_lows[low1], recent_lows[low2]
        # Ensure lows are separated by at least 3 days
        if abs(low2 - low1) < 3:
            logger.info("Double-bottom lows too close together.")
            return last_alert_ts
        # Confirm both lows are near $0.21 (±0.01)
        if not (0.20 <= low1_val <= 0.22 and 0.20 <= low2_val <= 0.22):
            logger.info(f"Lows not in $0.21 region: {low1_val:.3f}, {low2_val:.3f}")
            return last_alert_ts
        # --- Breakout trigger ---
        last_candle = candles[-1]
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        volume = float(last_candle['volume'])
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        # Calculate average volume (excluding current candle)
        avg_volume = sum(volumes[-(VOLUME_PERIOD+1):-1]) / VOLUME_PERIOD
        # Calculate EMA
        ema = pd.Series(closes).ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1]
        # Breakout and volume triggers
        breakout_trigger = close >= BREAKOUT_LEVEL
        volume_trigger = volume > avg_volume
        ema_trend_trigger = close > ema
        logger.info(f"Double-bottom detected at ${low1_val:.3f} and ${low2_val:.3f} (indices {low1}, {low2})")
        logger.info(f"Breakout close: ${close:.3f} (trigger: ≥${BREAKOUT_LEVEL}) -> {'✅' if breakout_trigger else '❌'}")
        logger.info(f"Volume: {volume:,.0f} (avg: {avg_volume:,.0f}) -> {'✅' if volume_trigger else '❌'}")
        logger.info(f"EMA({EMA_PERIOD}): {ema:.3f} (close above EMA) -> {'✅' if ema_trend_trigger else '❌'}")
        # --- Report summary of current conditions ---
        logger.info(f"--- Current DOGE Double-Bottom Breakout Conditions ---")
        logger.info(f"Daily close: ${close:.3f} (trigger: ≥${BREAKOUT_LEVEL}) -> {'✅' if breakout_trigger else '❌'}")
        logger.info(f"Volume: {volume:,.0f} (avg: {avg_volume:,.0f}) -> {'✅' if volume_trigger else '❌'}")
        logger.info(f"EMA({EMA_PERIOD}): {ema:.3f} (close above EMA) -> {'✅' if ema_trend_trigger else '❌'}")
        logger.info(f"-----------------------------------------------------")
        entry_triggered = breakout_trigger and volume_trigger and ema_trend_trigger
        # --- Entry zone check (hourly candles) ---
        if entry_triggered:
            logger.info("Entry trigger met. Checking for entry zone on hourly candles...")
            h_start = now - timedelta(hours=48)
            h_start_ts = int(h_start.timestamp())
            h_end_ts = int(now.timestamp())
            h_candles = safe_get_candles(cb_service, PRODUCT_ID, h_start_ts, h_end_ts, "ONE_HOUR")
            if not h_candles:
                logger.warning("No hourly candles available.")
                return last_alert_ts
            h_last = max(h_candles, key=lambda x: int(x['start']))
            h_close = float(h_last['close'])
            h_ema = pd.Series([float(c['close']) for c in h_candles]).ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1]
            in_entry_zone = (h_close >= ENTRY_ZONE_LOW) and (h_close <= ENTRY_ZONE_HIGH)
            logger.info(f"Most recent hourly close: ${h_close:.3f}")
            logger.info(f"Entry zone: ${ENTRY_ZONE_LOW}-${ENTRY_ZONE_HIGH} -> {'✅' if in_entry_zone else '❌'}")
            stop_loss = max(STOP_LOSS, h_ema)
            logger.info(f"Stop-loss set at: ${stop_loss:.3f} (max of fixed: ${STOP_LOSS}, hourly EMA: ${h_ema:.3f})")
            all_conditions_met = in_entry_zone
            if all_conditions_met and not trigger_state.get("triggered", False):
                logger.info("🎯 ALL DOUBLE-BOTTOM BREAKOUT CONDITIONS MET - EXECUTING TRADE!")
                play_alert_sound()
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="DOGE Double-Bottom Breakout Trade",
                    entry_price=h_close,
                    stop_loss=stop_loss,
                    take_profit=PROFIT_TARGET,
                    margin=DOGE_MARGIN,
                    leverage=DOGE_LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                logger.info(f"Trade execution completed: success={trade_success}")
                if trade_success:
                    logger.info("🎉 DOGE Double-Bottom Breakout trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    logger.info(f"Profit target: ${PROFIT_TARGET}")
                else:
                    logger.error(f"❌ DOGE Double-Bottom Breakout trade failed: {trade_result}")
                trigger_state = {"triggered": True, "trigger_ts": int(h_last['start'])}
                save_trigger_state(trigger_state)
                logger.info("Trigger state saved")
                logger.info("=== DOGE Double-Bottom Breakout Alert completed (trade executed) ===")
                return ts
        # Reset trigger if price falls below stop loss
        if trigger_state.get("triggered", False):
            if close < STOP_LOSS:
                logger.info("🔄 Resetting trigger state - price fell below stop loss level")
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(trigger_state)
                logger.info("Trigger state reset")
        logger.info("=== DOGE Double-Bottom Breakout Alert completed (no trade) ===")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in DOGE Double-Bottom Breakout Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== DOGE Double-Bottom Breakout Alert completed (with error) ===")
    return last_alert_ts

def main():
    logger.info("Starting DOGE Double-Bottom Breakout Alert Monitor ($0.26 breakout)")
    logger.info("🎯 Monitoring for double-bottom breakout entry ≥ $0.26 with volume spike and EMA trend")
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    cb_service = setup_coinbase()
    last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    def poll_iteration():
        nonlocal last_alert_ts, consecutive_failures
        iteration_start_time = time.time()
        last_alert_ts = doge_doublebottom_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)
                    wait_seconds = 300
                    logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
                    logger.info("")
                    time.sleep(wait_seconds)
                except concurrent.futures.TimeoutError:
                    logger.error('Polling iteration timed out! Skipping to next.')
        except KeyboardInterrupt:
            logger.info("👋 Stopped by user.")
            break
        except CONNECTION_ERRORS as e:
            consecutive_failures += 1
            logger.error(f"🔗 Connection error (failure {consecutive_failures}/{max_consecutive_failures}): {e}")
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"❌ Too many consecutive connection failures. Attempting to reconnect...")
                try:
                    cb_service = setup_coinbase()
                    consecutive_failures = 0
                    logger.info("✅ Reconnection successful, resuming monitoring...")
                except Exception as reconnect_error:
                    logger.error(f"❌ Reconnection failed: {reconnect_error}")
                    logger.info("😴 Sleeping for 5 minutes before retry...")
                    time.sleep(300)
            else:
                delay = exponential_backoff_delay(consecutive_failures - 1)
                logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"❌ Unexpected error in alert loop (failure {consecutive_failures}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            delay = min(60 * consecutive_failures, 300)
            logger.info(f"😴 Sleeping for {delay} seconds before retry...")

if __name__ == "__main__":
    main() 