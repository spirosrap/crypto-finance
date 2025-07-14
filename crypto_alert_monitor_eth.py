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

# ETH breakout parameters (updated)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY = "ONE_DAY"
VOLUME_PERIOD = 20
BREAKOUT_LEVEL = 2820
ENTRY_ZONE_LOW = 2820
ENTRY_ZONE_HIGH = 2900
STOP_LOSS = 2715
PROFIT_TARGET = 3000
EXTENDED_TARGET = 3525
ETH_MARGIN = 300  # USD
ETH_LEVERAGE = 20
TRIGGER_STATE_FILE = "eth_breakout_trigger_state.json"

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
                     margin: float = ETH_MARGIN, leverage: int = ETH_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
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

def eth_custom_breakout_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting eth_custom_breakout_alert ===")
    PRODUCT_ID = "ETH-PERP-INTX"
    GRANULARITY = "ONE_HOUR"
    ENTRY_ZONE_LOW = 3000
    ENTRY_ZONE_HIGH = 3050
    STOP_LOSS = 2940
    PROFIT_TARGET = 3380
    MARGIN = 300
    LEVERAGE = 20
    VOLUME_PERIOD = 2  # For facts reporting
    periods_needed = VOLUME_PERIOD + 2
    trigger_state = load_trigger_state()
    try:
        logger.info("Setting up time parameters...")
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=periods_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Time range: {start} to {end}")
        
        logger.info("Fetching candles from API...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        logger.info(f"Candles fetched: {len(candles) if candles else 0} candles")
        
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough ETH {GRANULARITY} candle data for custom breakout alert.")
            logger.info("=== eth_custom_breakout_alert completed (insufficient data) ===")
            return last_alert_ts
            
        logger.info("Processing candle data...")
        last_candle = candles[-2]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        low = float(last_candle['low'])
        v0 = float(last_candle['volume'])
        historical_candles = candles[-(VOLUME_PERIOD+2):-2]
        avg2 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles) if historical_candles else 0
        logger.info(f"Candle data processed: close=${close:,.2f}, high=${high:,.2f}, low=${low:,.2f}")
        
        # --- Reporting ---
        logger.info("Generating report...")
        logger.info("")
        logger.info(f"Entry zone: ${ENTRY_ZONE_LOW:,} – ${ENTRY_ZONE_HIGH:,}")
        logger.info(f"Stop-loss: ${STOP_LOSS:,}")
        logger.info(f"1st target: ${PROFIT_TARGET:,} (halfway to full flag objective ≈ $3,834)")
        logger.info(f"Facts: reclaimed the $3K barrier for first time since Feb; record ETF inflows; volume > 2-day average. [FX Leaders]")
        logger.info(f"Opinion: risk-to-reward ≈ 4 : 1—size accordingly.")
        logger.info("")
        logger.info(f"Candle close: ${close:,.2f}, High: ${high:,.2f}, Low: ${low:,.2f}, Volume: {v0:,.0f}, Avg(2): {avg2:,.0f}")
        
        # --- Entry logic ---
        logger.info("Checking entry conditions...")
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        logger.info(f"Entry conditions: in_zone={in_entry_zone}")
        
        if in_entry_zone and not trigger_state.get("triggered", False):
            logger.info("Entry conditions met - preparing to execute trade...")
            logger.info(f"Entry condition met: close (${close:,.2f}) is within entry zone (${ENTRY_ZONE_LOW:,}-${ENTRY_ZONE_HIGH:,}). Taking trade.")
            
            logger.info("Playing alert sound...")
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            
            logger.info("Executing crypto trade...")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD custom breakout entry",
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
                logger.info(f"ETH-USD custom breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"ETH-USD custom breakout trade failed: {trade_result}")
            
            logger.info("Saving trigger state...")
            # Set trigger to avoid duplicate trades
            trigger_state = {"triggered": True, "trigger_ts": int(last_candle['start'])}
            save_trigger_state(trigger_state)
            logger.info("Trigger state saved")
            
            logger.info("=== eth_custom_breakout_alert completed (trade executed) ===")
            return ts
            
        # Reset trigger if price leaves entry zone
        logger.info("Checking if trigger should be reset...")
        if trigger_state.get("triggered", False):
            if not in_entry_zone:
                logger.info("Resetting trigger state...")
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(trigger_state)
                logger.info("Trigger state reset")
        
        logger.info("=== eth_custom_breakout_alert completed (no trade) ===")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in ETH custom breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== eth_custom_breakout_alert completed (with error) ===")
    return last_alert_ts

def main():
    logger.info("Starting custom ETH breakout alert script")
    logger.info("")
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    logger.info("")
    cb_service = setup_coinbase()
    last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    def poll_iteration():
        nonlocal last_alert_ts, consecutive_failures
        iteration_start_time = time.time()
        last_alert_ts = eth_custom_breakout_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)  # 2 minute max per poll
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