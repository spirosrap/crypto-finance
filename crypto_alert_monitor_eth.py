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
        logging.FileHandler('eth_alert_debug.log'),
        logging.StreamHandler()  # Keep console output too
    ]
)
logger = logging.getLogger(__name__)

# Connection retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
MAX_RETRY_DELAY = 60  # seconds
BACKOFF_MULTIPLIER = 2

# Connection error types to handle
CONNECTION_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Covers network-related OS errors
    Exception  # Catch-all for API-specific errors
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

# ETH bull-flag continuation parameters (based on image)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY = "ONE_HOUR"  # Using 1-hour candles for more precise entry
VOLUME_PERIOD = 20
BREAKOUT_LEVEL = 3600  # Breakout above $3,600 from bull-flag
ENTRY_ZONE_LOW = 3640  # Shallow retest post breakout
ENTRY_ZONE_HIGH = 3670
STOP_LOSS = 3500  # Below flag lower boundary & 20-EMA
PROFIT_TARGET = 4000  # Projection from flag height & psychological level
EXTENDED_TARGET = 4100  # Additional upside potential
ETH_MARGIN = 250  # USD
ETH_LEVERAGE = 20  # 20x leverage
TRIGGER_STATE_FILE = "eth_bullflag_trigger_state.json"
MARGIN = 250
LEVERAGE = 20

# === STRATEGY PARAMETERS (from image) ===
BREAKOUT_TRIGGER_LEVEL = 3830  # Daily close above this
ENTRY_ZONE_LOW = 3830
ENTRY_ZONE_HIGH = 3900
STOP_LOSS = 3720
PROFIT_TARGET_LOW = 4000
EXTENDED_TARGET_LOW = 4200
EXTENDED_TARGET_HIGH = 4250
VOLUME_LOOKBACK = 20  # For above-average volume
EMA_PERIOD = 20  # Key EMA trend
TRIGGER_STATE_FILE = "eth_breakout_trigger_state.json"
VOLUME_SURGE_FACTOR = 1.2  # 20% above average

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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = PRODUCT_ID):
    def _execute_trade():
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
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

# --- New alert logic for breakout past $3,830 resistance ---
def eth_breakout_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting ETH-USD Breakout Alert ($3,830 resistance) ===")
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        # Get daily candles for entry trigger
        end = now
        start = now - timedelta(days=VOLUME_LOOKBACK + 5)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Fetching daily candles for {VOLUME_LOOKBACK+5} days...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, "ONE_DAY")
        if not candles or len(candles) < VOLUME_LOOKBACK + 1:
            logger.warning("Not enough daily candle data for breakout alert.")
            return last_alert_ts
        # Sort by timestamp ascending
        candles = sorted(candles, key=lambda x: int(x['start']))
        last_candle = candles[-1]
        prev_candles = candles[-(VOLUME_LOOKBACK+1):-1]
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        low = float(last_candle['low'])
        volume = float(last_candle['volume'])
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        # Calculate average volume (excluding current candle)
        avg_volume = sum(float(c['volume']) for c in prev_candles) / len(prev_candles)
        # Calculate EMA (using pandas for simplicity)
        closes = [float(c['close']) for c in candles]
        ema = pd.Series(closes).ewm(span=EMA_PERIOD, adjust=False).mean().iloc[-1]
        # --- Entry trigger logic ---
        daily_close_trigger = close > BREAKOUT_TRIGGER_LEVEL
        volume_trigger = volume >= VOLUME_SURGE_FACTOR * avg_volume
        ema_trend_trigger = close > ema
        logger.info(f"Daily close: ${close:,.2f} (trigger: >${BREAKOUT_TRIGGER_LEVEL}) -> {'✅' if daily_close_trigger else '❌'}")
        logger.info(f"Volume: {volume:,.0f} (avg: {avg_volume:,.0f}, surge: {VOLUME_SURGE_FACTOR}x) -> {'✅' if volume_trigger else '❌'}")
        logger.info(f"EMA({EMA_PERIOD}): {ema:,.2f} (close above EMA) -> {'✅' if ema_trend_trigger else '❌'}")
        entry_triggered = daily_close_trigger and volume_trigger and ema_trend_trigger
        # --- If entry trigger is met, check for entry zone on hourly candles ---
        if entry_triggered:
            logger.info("Entry trigger met. Checking for entry zone on hourly candles...")
            # Get last 24 hours of hourly candles
            h_start = now - timedelta(hours=48)
            h_start_ts = int(h_start.timestamp())
            h_end_ts = int(now.timestamp())
            h_candles = safe_get_candles(cb_service, PRODUCT_ID, h_start_ts, h_end_ts, "ONE_HOUR")
            if not h_candles:
                logger.warning("No hourly candles available.")
                return last_alert_ts
            # Use most recent hourly close
            h_last = max(h_candles, key=lambda x: int(x['start']))
            h_close = float(h_last['close'])
            logger.info(f"Most recent hourly close: ${h_close:,.2f}")
            in_entry_zone = ENTRY_ZONE_LOW <= h_close <= ENTRY_ZONE_HIGH
            logger.info(f"Entry zone: ${ENTRY_ZONE_LOW}-${ENTRY_ZONE_HIGH} -> {'✅' if in_entry_zone else '❌'}")
            all_conditions_met = in_entry_zone
            if all_conditions_met and not trigger_state.get("triggered", False):
                logger.info("🎯 ALL BREAKOUT CONDITIONS MET - EXECUTING TRADE!")
                play_alert_sound()
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD Breakout Trade",
                    entry_price=h_close,
                    stop_loss=STOP_LOSS,
                    take_profit=PROFIT_TARGET_LOW,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                logger.info(f"Trade execution completed: success={trade_success}")
                if trade_success:
                    logger.info("🎉 ETH-USD Breakout trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                else:
                    logger.error(f"❌ ETH-USD Breakout trade failed: {trade_result}")
                trigger_state = {"triggered": True, "trigger_ts": int(h_last['start'])}
                save_trigger_state(trigger_state)
                logger.info("Trigger state saved")
                logger.info("=== ETH-USD Breakout Alert completed (trade executed) ===")
                return ts
        # Reset trigger if price falls below stop loss
        if trigger_state.get("triggered", False):
            if close < STOP_LOSS:
                logger.info("🔄 Resetting trigger state - price fell below stop loss level")
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(trigger_state)
                logger.info("Trigger state reset")
        logger.info("=== ETH-USD Breakout Alert completed (no trade) ===")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in ETH-USD Breakout Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH-USD Breakout Alert completed (with error) ===")
    return last_alert_ts

# Replace main loop to use new alert
def main():
    logger.info("Starting ETH-USD Breakout Alert Monitor ($3,830 resistance)")
    logger.info("🎯 Monitoring for breakout entry at $3,830–3,900 with above-average volume and EMA trend")
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
        last_alert_ts = eth_breakout_alert(cb_service, last_alert_ts)
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
            time.sleep(delay)

if __name__ == "__main__":
    main() 