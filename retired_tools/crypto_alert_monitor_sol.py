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

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
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

# SOL breakout parameters (NEW SETUP)
PRODUCT_ID = "SOL-PERP-INTX"
GRANULARITY = "ONE_DAY"  # Changed to daily
VOLUME_PERIOD = 20
ENTRY_TRIGGER = 180  # Break above $180
ENTRY_ZONE = 180     # Entry above $180
STOP_LOSS = 175      # Below $175
PROFIT_TARGET = 200  # $200
VOLUME_MULTIPLIER = 1.25  # 25% above average
SOL_MARGIN = 150  # USD (adjust as needed)
SOL_LEVERAGE = 20
TRIGGER_STATE_FILE = "sol_breakout_trigger_state.json"

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
                start=int((datetime.now(UTC) - timedelta(hours=8)).timestamp()),
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
                     margin: float = SOL_MARGIN, leverage: int = SOL_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
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

def sol_breakout_alert_new(cb_service, last_alert_ts=None):
    periods_needed = VOLUME_PERIOD + 2
    days_needed = periods_needed
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=days_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching SOL candles from {start} to {end}")
        logger.info(f"Product ID: {PRODUCT_ID}, Granularity: {GRANULARITY}")
        
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough SOL {GRANULARITY} candle data for daily breakout alert.")
            return last_alert_ts
        
         
        # Fix: Candles are returned in reverse chronological order (newest first)
        # So candles[0] is the current incomplete candle, candles[1] is the last closed candle
        last_candle = candles[1]  # Last closed daily candle (second in array)
        historical_candles = candles[2:VOLUME_PERIOD+2]  # Use candles 2 onwards for historical data
        
        
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        close = float(last_candle['close'])
        v0 = float(last_candle['volume'])
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
        
        
        trigger_ok = close > ENTRY_TRIGGER
        vol_ok = v0 >= VOLUME_MULTIPLIER * avg20
        logger.info(f"=== SOL DAILY BREAKOUT ALERT (NEW SETUP) ===")
        logger.info(f"Candle close: ${close:,.2f}, Volume: {v0:,.0f}, Avg(20): {avg20:,.0f}")
        logger.info(f"  - Close above trigger ${ENTRY_TRIGGER:,.1f}: {'✅ Met' if trigger_ok else '❌ Not Met'}")
        logger.info(f"  - Volume ≥ 1.25x avg: {'✅ Met' if vol_ok else '❌ Not Met'}")
        # Step 1: Set trigger if breakout and volume are high
        if trigger_ok and vol_ok and not trigger_state.get("triggered", False):
            logger.info(f"--- SOL breakout TRIGGERED: taking trade ---")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="SOL daily breakout long (new setup)",
                entry_price=close,
                stop_loss=STOP_LOSS,
                take_profit=PROFIT_TARGET,
                margin=SOL_MARGIN,
                leverage=SOL_LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            if trade_success:
                logger.info(f"SOL daily breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"SOL daily breakout trade failed: {trade_result}")
            # Reset trigger after trade
            trigger_state = {"triggered": False, "trigger_ts": None}
            save_trigger_state(trigger_state)
            return ts
        # Step 2: If previously triggered, do nothing (one-shot trade)
        if trigger_state.get("triggered", False):
            logger.info(f"SOL breakout previously triggered at candle {trigger_state.get('trigger_ts')}, waiting for reset...")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in SOL daily breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts

def get_current_sol_price(cb_service):
    """Get current SOL price to verify API is working correctly"""
    try:
        now = datetime.now(UTC)
        start = now - timedelta(hours=1)
        end = now
        
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=int(start.timestamp()),
            end=int(end.timestamp()),
            granularity="ONE_HOUR"
        )
        
        if hasattr(response, 'candles'):
            candles = response.candles
        else:
            candles = response.get('candles', [])
            
        if candles:
            latest_candle = candles[-1]
            current_price = float(latest_candle['close'])
            logger.info(f"Current SOL price: ${current_price:,.2f}")
            return current_price
        else:
            logger.error("No candles returned for current price check")
            return None
    except Exception as e:
        logger.error(f"Error getting current SOL price: {e}")
        return None

def main():
    logger.info("Starting SOL daily breakout alert script (NEW SETUP)")
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
    
    # Test current SOL price to verify API is working
    logger.info("Testing SOL price API...")
    current_price = get_current_sol_price(cb_service)
    if current_price is None:
        logger.error("❌ Failed to get current SOL price. Check API connection.")
        return
    elif current_price < 100:  # SOL should be well above $100
        logger.error(f"❌ Current SOL price seems wrong: ${current_price:,.2f}. Check product ID.")
        return
    else:
        logger.info(f"✅ Current SOL price verified: ${current_price:,.2f}")
    
    sol_breakout_last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    while True:
        try:
            iteration_start_time = time.time()
            sol_breakout_last_alert_ts = sol_breakout_alert_new(cb_service, sol_breakout_last_alert_ts)
            consecutive_failures = 0
            wait_seconds = 300  # 5 minutes
            logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
            logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
            logger.info("")
            time.sleep(wait_seconds)
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