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

# NEAR mini breakout parameters (cup & handle)
PRODUCT_ID = "NEAR-PERP-INTX"
GRANULARITY = "ONE_HOUR"
VOLUME_PERIOD = 20
ENTRY_TRIGGER_LOW = 2.20  # Above this level
ENTRY_ZONE_LOW = 2.15
ENTRY_ZONE_HIGH = 2.18
SUPPORT_LOW = 2.13
SUPPORT_HIGH = 2.15
STOP_LOSS = 2.05
PROFIT_TARGET = 2.30
VOLUME_MULTIPLIER = 1.20  # 20% above average for confirmation
RSI_THRESHOLD = 30
NEAR_MARGIN = 200  # USD
NEAR_LEVERAGE = 20

TRIGGER_STATE_FILE = "near_breakout_trigger_state.json"


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
                start=int((datetime.now(UTC) - timedelta(hours=4)).timestamp()),
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
                     margin: float = NEAR_MARGIN, leverage: int = NEAR_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
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

def near_bounce_support_alert(cb_service, last_alert_ts=None):
    periods_needed = 30  # Fetch last 30 candles for more recent data
    hours_needed = periods_needed
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Requesting candles from {start.strftime('%m-%d %H:%M')} (ts={start_ts}) to {end.strftime('%m-%d %H:%M')} (ts={end_ts})")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < 5:
            logger.warning(f"Not enough NEAR {GRANULARITY} candle data for support bounce alert.")
            return last_alert_ts
        candles = list(reversed(candles))
        logger.info(f"Fetched {len(candles)} candles from {datetime.fromtimestamp(int(candles[0]['start']), UTC).strftime('%m-%d %H:%M')} to {datetime.fromtimestamp(int(candles[-1]['start']), UTC).strftime('%m-%d %H:%M')}")
        historical_candles = candles[-(VOLUME_PERIOD+2):-2]
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
        # Calculate RSI for each candle
        closes = [float(c['close']) for c in candles]
        rsi_series = ta.rsi(pd.Series(closes), length=14)
        # Log the latest candle's info for reporting
        latest_candle = candles[-2]  # -2 is the last completed candle
        latest_close = float(latest_candle['close'])
        latest_low = float(latest_candle['low'])
        latest_high = float(latest_candle['high'])
        latest_vol = float(latest_candle['volume'])
        latest_rsi = rsi_series.iloc[-2] if not pd.isna(rsi_series.iloc[-2]) else None
        rsi_str = f"{latest_rsi:.2f}" if latest_rsi is not None else "N/A"
        logger.info(f"Latest Candle: Close=${latest_close:.4f}, Low=${latest_low:.4f}, High=${latest_high:.4f}, Volume={latest_vol:.0f}, RSI={rsi_str}")
        logger.info(f"20-period avg volume: {avg20:.0f}")
        # Debug: Print the times and closes of the last 5 fetched candles
        candle_info = []
        for c in candles[-5:]:
            t = datetime.fromtimestamp(int(c['start']), UTC)
            candle_info.append(f"{t.strftime('%m-%d %H:%M')}:${c['close']}")
        logger.info(f"Last 5 candles: {' | '.join(candle_info)}")
        # Only check the last 3 completed candles for support touch
        for i in range(len(candles)-2, len(candles)-5, -1):
            if i < 0:
                break
            c = candles[i]
            low = float(c['low'])
            if SUPPORT_LOW <= low <= SUPPORT_HIGH:
                logger.info(f"Candle at {datetime.fromtimestamp(int(c['start']), UTC).strftime('%m-%d %H:%M')} touched support zone (${SUPPORT_LOW}-${SUPPORT_HIGH})")
                # Check next candle for bounce
                if i+1 >= len(candles):
                    continue
                bounce = candles[i+1]
                bounce_close = float(bounce['close'])
                bounce_vol = float(bounce['volume'])
                bounce_rsi = rsi_series.iloc[i+1] if not pd.isna(rsi_series.iloc[i+1]) else 100
                logger.info(f"Bounce candle close: ${bounce_close:.2f}, volume: {bounce_vol:.0f}, RSI: {bounce_rsi:.2f}")
                if (ENTRY_ZONE_LOW <= bounce_close <= ENTRY_ZONE_HIGH and
                    bounce_vol >= VOLUME_MULTIPLIER * avg20 and
                    bounce_rsi < RSI_THRESHOLD):
                    logger.info(f"--- NEAR SUPPORT BOUNCE TRADE ALERT ---")
                    logger.info(f"Entry condition met: bounce close ${bounce_close:.2f} in entry zone, volume spike, RSI < 30. Taking trade.")
                    try:
                        play_alert_sound()
                    except Exception as e:
                        logger.error(f"Failed to play alert sound: {e}")
                    trade_success, trade_result = execute_crypto_trade(
                        cb_service=cb_service,
                        trade_type="NEAR support bounce long",
                        entry_price=bounce_close,
                        stop_loss=STOP_LOSS,
                        take_profit=PROFIT_TARGET,
                        margin=NEAR_MARGIN,
                        leverage=NEAR_LEVERAGE,
                        side="BUY",
                        product=PRODUCT_ID
                    )
                    if trade_success:
                        logger.info(f"NEAR support bounce trade executed successfully!")
                        logger.info(f"Trade output: {trade_result}")
                    else:
                        logger.error(f"NEAR support bounce trade failed: {trade_result}")
                    return datetime.fromtimestamp(int(bounce['start']), UTC)
                else:
                    logger.info(f"Bounce/volume/RSI confirmation absent. Skipping trade.")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in NEAR support bounce alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts

def main():
    logger.info("Starting NEAR cup & handle breakout alert script")
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
    near_breakout_last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    while True:
        try:
            iteration_start_time = time.time()
            near_breakout_last_alert_ts = near_bounce_support_alert(cb_service, near_breakout_last_alert_ts)
            consecutive_failures = 0
            wait_seconds = 300
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