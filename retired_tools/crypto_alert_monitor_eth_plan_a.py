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
        logging.FileHandler('eth_plan_a_alert_debug.log'),
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

# ETH Plan A Alert parameters (based on Pine Script)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY = "FOUR_HOUR"  # 4-hour candles as per Pine Script
VOLUME_PERIOD = 20  # For EMA calculation
BREAKOUT_LEVEL = 3835  # Breakout confirmation level
ABORT_LEVEL = 3750  # Abort to flat level
VOLUME_SURGE_FACTOR = 1.25  # 25% above average volume for breakout
VOLUME_WEAK_FACTOR = 0.90  # 90% below average volume for fakeout/abort
TRIGGER_STATE_FILE = "eth_plan_a_trigger_state.json"

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
            return {"breakout_confirmed": False, "fakeout_triggered": False, "abort_triggered": False, "last_trigger_ts": None}
    return {"breakout_confirmed": False, "fakeout_triggered": False, "abort_triggered": False, "last_trigger_ts": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def calculate_volume_ema(candles, period=20):
    """Calculate Exponential Moving Average of volume"""
    if len(candles) < period:
        return None
    
    volumes = [float(c['volume']) for c in candles[-period:]]
    # Simple EMA calculation
    alpha = 2.0 / (period + 1)
    ema = volumes[0]
    for volume in volumes[1:]:
        ema = alpha * volume + (1 - alpha) * ema
    return ema

# --- ETH Plan A Alert Logic (4-Hour Breakout Strategy) ---
def eth_plan_a_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting ETH Plan A 4-Hour Alert Monitor ===")
    logger.info(f"🎯 Monitoring for breakout ≥${BREAKOUT_LEVEL:,} with volume confirmation")
    logger.info(f"🛑 Abort level: <${ABORT_LEVEL:,} with weak volume (pre-breakout only)")
    trigger_state = load_trigger_state()
    
    try:
        now = datetime.now(UTC)
        
        # Get 4-hour candles for analysis
        end = now
        start = now - timedelta(hours=4 * (VOLUME_PERIOD + 10))  # Enough data for EMA calculation
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 4-hour candles for {VOLUME_PERIOD + 10} periods...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        
        if not candles or len(candles) < VOLUME_PERIOD + 1:
            logger.warning("Not enough 4-hour candle data for Plan A alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending
        candles = sorted(candles, key=lambda x: int(x['start']))
        
        # Use the last fully closed 4-hour candle (not the current forming candle)
        # 4-hour candles close at: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
        current_ts = datetime.fromtimestamp(int(candles[-1]['start']), UTC)
        now_ts = now.replace(minute=0, second=0, microsecond=0)
        
        # Determine if we should use the current candle or the previous one
        # If current time is within the first 30 minutes of a new 4-hour period, 
        # the previous candle might not be fully closed yet
        time_since_candle_start = (now - current_ts).total_seconds() / 3600  # hours
        
        if time_since_candle_start < 3.5:  # Less than 3.5 hours into the current candle
            # Use the previous fully closed candle
            if len(candles) < 2:
                logger.warning("Not enough candles to get previous closed candle")
                return last_alert_ts
            analysis_candle = candles[-2]  # Previous candle
            logger.info("Using previous fully closed 4-hour candle for analysis")
        else:
            # Current candle is mostly complete, safe to use
            analysis_candle = candles[-1]  # Current candle
            logger.info("Using current 4-hour candle for analysis (mostly complete)")
        
        # Get candle data for analysis
        candle_close = float(analysis_candle['close'])
        candle_high = float(analysis_candle['high'])
        candle_low = float(analysis_candle['low'])
        candle_volume = float(analysis_candle['volume'])
        candle_ts = datetime.fromtimestamp(int(analysis_candle['start']), UTC)
        
        # Calculate 20-period EMA of volume using data up to the analysis candle
        # Exclude the analysis candle itself from EMA calculation
        ema_candles = candles[:-1] if analysis_candle == candles[-1] else candles[:-2]
        volume_ema = calculate_volume_ema(ema_candles, VOLUME_PERIOD)
        if volume_ema is None:
            logger.warning("Cannot calculate volume EMA - insufficient data")
            return last_alert_ts
        
        # Pine Script logic implementation (evaluated on 4-hour CLOSE)
        # breakout = close >= 3835 and volume >= 1.25 * volE
        breakout_condition = (candle_close >= BREAKOUT_LEVEL and 
                             candle_volume >= VOLUME_SURGE_FACTOR * volume_ema)
        
        # fakeout = close < 3835 and volume < 0.90 * volE (ONLY after breakout_confirmed)
        # Explicit gating: fakeout can NEVER fire unless breakout_confirmed == True
        fakeout_price_volume_condition = (candle_close < BREAKOUT_LEVEL and 
                                         candle_volume < VOLUME_WEAK_FACTOR * volume_ema)
        fakeout_condition = (fakeout_price_volume_condition and 
                            trigger_state.get("breakout_confirmed", False))
        
        # abort = close < 3750 and volume < volE (pre-breakout only)
        abort_condition = (candle_close < ABORT_LEVEL and 
                          candle_volume < volume_ema and
                          not trigger_state.get("breakout_confirmed", False))
        
        logger.info(f"Analysis candle timestamp: {candle_ts}")
        logger.info(f"ETH close price: ${candle_close:,.2f}")
        logger.info(f"Volume: {candle_volume:,.0f}")
        logger.info(f"Volume EMA (20): {volume_ema:,.0f}")
        logger.info(f"Breakout level: ${BREAKOUT_LEVEL:,} -> {'TRIGGER' if candle_close >= BREAKOUT_LEVEL else 'ARMED'}")
        logger.info(f"Volume surge (≥{VOLUME_SURGE_FACTOR}x): {'TRIGGER' if candle_volume >= VOLUME_SURGE_FACTOR * volume_ema else 'ARMED'}")
        logger.info(f"Abort level: ${ABORT_LEVEL:,} -> {'TRIGGER' if candle_close < ABORT_LEVEL else 'ARMED'}")
        logger.info(f"Breakout confirmed flag: {'✅' if trigger_state.get('breakout_confirmed', False) else '❌'}")
        
        # Handle breakout condition
        if breakout_condition and not trigger_state.get("breakout_confirmed", False):
            logger.info("🎯 ETH 4H BREAKOUT CONFIRMED - EXECUTING TRADE!")
            play_alert_sound()
            
            # Calculate trade parameters
            entry_price = candle_close
            stop_loss = ABORT_LEVEL  # Use abort level as stop loss
            take_profit = entry_price + (entry_price - stop_loss) * 2  # 2:1 risk-reward
            
            # Execute the trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH Plan A 4H Breakout",
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                margin=250,
                leverage=20,
                side="BUY",
                product=PRODUCT_ID
            )
            
            if trade_success:
                logger.info("🎉 ETH Plan A Breakout trade executed successfully!")
                logger.info(f"Entry: ${entry_price:,.2f}")
                logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                logger.info(f"Take-profit: ${take_profit:,.2f}")
                logger.info(f"Volume confirmation: {candle_volume:,.0f} ≥ {VOLUME_SURGE_FACTOR * volume_ema:,.0f}")
            else:
                logger.error(f"❌ ETH Plan A Breakout trade failed: {trade_result}")
            
            # Set persistent breakout confirmed flag
            trigger_state["breakout_confirmed"] = True
            trigger_state["last_trigger_ts"] = int(analysis_candle['start'])
            save_trigger_state(trigger_state)
            logger.info("Breakout confirmed flag set - fakeout alerts now enabled")
            return candle_ts
        
        # Handle fakeout condition (only after breakout_confirmed)
        elif fakeout_condition and not trigger_state.get("fakeout_triggered", False):
            logger.info("⚠️ ETH 4H FAKEOUT DETECTED - EXIT SIGNAL!")
            play_alert_sound()
            
            logger.info(f"Fakeout detected: Close ${candle_close:,.2f} < ${BREAKOUT_LEVEL:,}")
            logger.info(f"Weak volume: {candle_volume:,.0f} < {VOLUME_WEAK_FACTOR * volume_ema:,.0f}")
            logger.info("Consider exiting long positions or waiting for stronger confirmation")
            
            # Update trigger state
            trigger_state["fakeout_triggered"] = True
            trigger_state["last_trigger_ts"] = int(analysis_candle['start'])
            save_trigger_state(trigger_state)
            logger.info("Fakeout trigger state saved")
            return candle_ts
        
        # Handle abort condition (pre-breakout only)
        elif abort_condition and not trigger_state.get("abort_triggered", False):
            logger.info("🛑 ETH 4H ABORT TO FLAT - EXIT ALL POSITIONS!")
            play_alert_sound()
            
            logger.info(f"Abort condition: Close ${candle_close:,.2f} < ${ABORT_LEVEL:,}")
            logger.info(f"Weak volume: {candle_volume:,.0f} < {volume_ema:,.0f}")
            logger.info("Exit all positions and wait for better setup")
            
            # Update trigger state
            trigger_state["abort_triggered"] = True
            trigger_state["last_trigger_ts"] = int(analysis_candle['start'])
            save_trigger_state(trigger_state)
            logger.info("Abort trigger state saved")
            return candle_ts
        
        # Reset triggers if price moves back into favorable territory
        if trigger_state.get("breakout_confirmed", False):
            if candle_close < BREAKOUT_LEVEL:
                logger.info("🔄 Resetting breakout confirmed flag - price back below breakout level")
                trigger_state["breakout_confirmed"] = False
                save_trigger_state(trigger_state)
        
        if trigger_state.get("fakeout_triggered", False):
            if candle_close >= BREAKOUT_LEVEL and candle_volume >= VOLUME_SURGE_FACTOR * volume_ema:
                logger.info("🔄 Resetting fakeout trigger - strong breakout confirmed")
                trigger_state["fakeout_triggered"] = False
                save_trigger_state(trigger_state)
        
        if trigger_state.get("abort_triggered", False):
            if candle_close >= ABORT_LEVEL:
                logger.info("🔄 Resetting abort trigger - price back above abort level")
                trigger_state["abort_triggered"] = False
                save_trigger_state(trigger_state)
        
        if not any([breakout_condition, fakeout_condition, abort_condition]):
            logger.info("⏳ Waiting for Plan A conditions...")
            logger.info(f"   Price: ${candle_close:,.2f} (breakout: ${BREAKOUT_LEVEL:,}, abort: ${ABORT_LEVEL:,})")
            logger.info(f"   Volume: {candle_volume:,.0f} (EMA: {volume_ema:,.0f})")
            if trigger_state.get("breakout_confirmed", False):
                logger.info("   ✅ Breakout confirmed - fakeout alerts enabled, abort alerts disabled")
                if fakeout_price_volume_condition:
                    logger.info("   ⚠️ Fakeout price/volume conditions met but already triggered")
            else:
                logger.info("   ❌ Breakout not confirmed - abort alerts enabled, fakeout alerts disabled")
                if fakeout_price_volume_condition:
                    logger.info("   ⚠️ Fakeout price/volume conditions met but breakout not confirmed yet")
        
        logger.info("=== ETH Plan A Alert completed ===")
        return last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in ETH Plan A Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Plan A Alert completed (with error) ===")
    return last_alert_ts

def main():
    logger.info("Starting ETH Plan A 4-Hour Alert Monitor")
    logger.info(f"🎯 Monitoring for breakout ≥${BREAKOUT_LEVEL:,} with volume confirmation")
    logger.info(f"🛑 Abort level: <${ABORT_LEVEL:,} with weak volume")
    logger.info("📊 Strategy: 4-hour breakout detection with volume EMA confirmation")
    logger.info("💡 Based on Pine Script: ETH Plan A Alerts")
    
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
        last_alert_ts = eth_plan_a_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)
                    wait_seconds = 600  # 10 minutes between checks for 4-hour candles
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