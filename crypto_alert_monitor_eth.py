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

# ETH bull-flag breakout continuation parameters (from image)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY = "ONE_DAY"  # Using daily candles for entry trigger
VOLUME_PERIOD = 20
BREAKOUT_LEVEL = 3830  # Breakout above $3,830 from bull-flag
ENTRY_ZONE_LOW = 3830  # Entry zone as per image
ENTRY_ZONE_HIGH = 3900
STOP_LOSS = 3720  # EMA/trendline support zone as per image
PROFIT_TARGET = 4000  # Resistance + structural target as per image
EXTENDED_TARGET_LOW = 4500  # Stretch target per on-chain MVRV and flag projection
EXTENDED_TARGET_HIGH = 4900
ETH_MARGIN = 250  # USD
ETH_LEVERAGE = 20  # 20x leverage
TRIGGER_STATE_FILE = "eth_bullflag_trigger_state.json"
MARGIN = 250
LEVERAGE = 20

# === STRATEGY PARAMETERS (from image) ===
BREAKOUT_TRIGGER_LOW = 3830  # Daily close > $3,830
BREAKOUT_TRIGGER_HIGH = 3900  # Daily close upper bound
VOLUME_LOOKBACK = 20  # For above-average volume
EMA_PERIOD = 20  # Key EMA trend
VOLUME_SURGE_FACTOR = 1.2  # 20% above average volume pickup

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

# --- ETH-USD Bull-Flag Breakout Continuation Alert (from image) ---
def eth_breakout_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting ETH-USD Bull-Flag Breakout Continuation Alert ===")
    logger.info("🎯 Strategy: Daily close > $3,830 with ≥20% volume pickup")
    logger.info("📍 Entry zone: $3,830–3,900")
    logger.info("🛑 Stop-loss: $3,720 (EMA/trendline support)")
    logger.info("🎯 First target: $4,000 (resistance + structural)")
    logger.info("🚀 Stretch target: $4,500–4,900 (on-chain MVRV + flag projection)")
    
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
        
        # --- Entry trigger logic (from image) ---
        daily_close_trigger = close > BREAKOUT_TRIGGER_LOW  # Daily close > $3,830
        volume_trigger = volume >= VOLUME_SURGE_FACTOR * avg_volume  # ≥20% volume pickup
        ema_trend_trigger = close > ema  # Price above EMA for trend confirmation
        
        logger.info(f"📊 Daily Analysis:")
        logger.info(f"   Close: ${close:,.2f} (trigger: >${BREAKOUT_TRIGGER_LOW}) -> {'✅' if daily_close_trigger else '❌'}")
        logger.info(f"   Volume: {volume:,.0f} (avg: {avg_volume:,.0f}, surge: {VOLUME_SURGE_FACTOR}x) -> {'✅' if volume_trigger else '❌'}")
        logger.info(f"   EMA({EMA_PERIOD}): {ema:,.2f} (close above EMA) -> {'✅' if ema_trend_trigger else '❌'}")
        
        entry_triggered = daily_close_trigger and volume_trigger and ema_trend_trigger
        
        if entry_triggered:
            logger.info("🎯 ENTRY TRIGGER MET - Daily close > $3,830 with ≥20% volume pickup!")
            
            # Check if we're in the entry zone ($3,830–3,900)
            in_entry_zone = (close >= ENTRY_ZONE_LOW) and (close <= ENTRY_ZONE_HIGH)
            logger.info(f"📍 Entry zone check: ${ENTRY_ZONE_LOW}-${ENTRY_ZONE_HIGH} -> {'✅' if in_entry_zone else '❌'}")
            
            if in_entry_zone and not trigger_state.get("triggered", False):
                logger.info("🚀 ALL CONDITIONS MET - EXECUTING BULL-FLAG BREAKOUT TRADE!")
                play_alert_sound()
                
                # Execute the trade with parameters from image
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD Bull-Flag Breakout Trade",
                    entry_price=close,
                    stop_loss=STOP_LOSS,  # $3,720
                    take_profit=PROFIT_TARGET,  # $4,000
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                logger.info(f"Trade execution completed: success={trade_success}")
                if trade_success:
                    logger.info("🎉 ETH-USD Bull-Flag Breakout trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    logger.info(f"🎯 First profit target: ${PROFIT_TARGET:,.2f}")
                    logger.info(f"🚀 Stretch target: ${EXTENDED_TARGET_LOW:,.2f}-${EXTENDED_TARGET_HIGH:,.2f}")
                    logger.info("💡 Consider extending target if momentum holds")
                else:
                    logger.error(f"❌ ETH-USD Bull-Flag Breakout trade failed: {trade_result}")
                
                trigger_state = {"triggered": True, "trigger_ts": int(last_candle['start'])}
                save_trigger_state(trigger_state)
                logger.info("Trigger state saved")
                logger.info("=== ETH-USD Bull-Flag Breakout Alert completed (trade executed) ===")
                return ts
            elif not in_entry_zone:
                logger.info("⚠️ Entry trigger met but not in entry zone - monitoring for pullback")
        
        # Reset trigger if price falls below stop loss
        if trigger_state.get("triggered", False):
            if close < STOP_LOSS:
                logger.info("🔄 Resetting trigger state - price fell below stop loss level ($3,720)")
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(trigger_state)
                logger.info("Trigger state reset")
        
        logger.info("=== ETH-USD Bull-Flag Breakout Alert completed (no trade) ===")
        return last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in ETH-USD Bull-Flag Breakout Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH-USD Bull-Flag Breakout Alert completed (with error) ===")
    return last_alert_ts

# Replace main loop to use new alert
def main():
    logger.info("Starting ETH-USD Bull-Flag Breakout Continuation Monitor")
    logger.info("🎯 Strategy: Daily close > $3,830 with ≥20% volume pickup")
    logger.info("📍 Entry zone: $3,830–3,900")
    logger.info("🛑 Stop-loss: $3,720 (EMA/trendline support)")
    logger.info("🎯 First target: $4,000 (resistance + structural)")
    logger.info("🚀 Stretch target: $4,500–4,900 (on-chain MVRV + flag projection)")
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