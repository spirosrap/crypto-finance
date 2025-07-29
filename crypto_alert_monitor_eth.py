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

# ETH Trading Plan Parameters (based on image)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY = "ONE_HOUR"  # 1-hour candles as specified in image
VOLUME_PERIOD = 20  # For volume confirmation

# Plan A - Momentum Breakout Parameters
BREAKOUT_RESISTANCE_LOW = 3920  # Buy on 1h close ≥ $3,920-3,930
BREAKOUT_RESISTANCE_HIGH = 3930
BREAKOUT_STOP_LOSS = 3860  # below the breakout candle or <$3,860 (whichever is lower)
BREAKOUT_TP1 = 4050  # $4,050
BREAKOUT_TP2_LOW = 4170  # $4,170-$4,300
BREAKOUT_TP2_HIGH = 4300

# Plan B - Pullback Buy Parameters
PULLBACK_BID_LOW = 3700  # Bid $3,700-$3,730 (first support zone)
PULLBACK_BID_HIGH = 3730
PULLBACK_STOP_LOSS = 3640  # <$3,640 (or 1h close < $3,680)
PULLBACK_STOP_LOSS_ALT = 3680  # 1h close < $3,680
PULLBACK_TP_LOW = 3880  # back into $3,880-$3,920
PULLBACK_TP_HIGH = 3920

# Volume confirmation requirement
VOLUME_SURGE_FACTOR = 1.25  # ≥1.25x 20-period 1h volume on the breakout bar

# Trade parameters
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage

# State files for each plan
PLAN_A_TRIGGER_FILE = "eth_plan_a_trigger_state.json"
PLAN_B_TRIGGER_FILE = "eth_plan_b_trigger_state.json"

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

def load_trigger_state(plan_file):
    if os.path.exists(plan_file):
        try:
            with open(plan_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None}
    return {"triggered": False, "trigger_ts": None}

def save_trigger_state(state, plan_file):
    try:
        with open(plan_file, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

# --- ETH Trading Plan Alert Logic ---
def eth_trading_plan_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting ETH-USD 1-Hour Trading Plan Alert ===")
    logger.info("🎯 Monitoring for Plan A (Momentum Breakout) and Plan B (Pullback Buy)")
    logger.info("📊 Plan A: Buy on 1h close ≥ $3,920-3,930 (key resistance cluster)")
    logger.info("📊 Plan B: Bid $3,700-$3,730 (first support zone)")
    
    # Load trigger states for both plans
    plan_a_state = load_trigger_state(PLAN_A_TRIGGER_FILE)
    plan_b_state = load_trigger_state(PLAN_B_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 1-hour candles for analysis
        end = now
        start = now - timedelta(hours=VOLUME_PERIOD + 24)  # Enough data for volume analysis
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 1-hour candles for {VOLUME_PERIOD + 24} hours...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        
        if not candles or len(candles) < VOLUME_PERIOD + 1:
            logger.warning("Not enough 1-hour candle data for trading plan alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending
        candles = sorted(candles, key=lambda x: int(x['start']))
        
        # Get current candle data
        current_candle = candles[-1]
        current_close = float(current_candle['close'])
        current_high = float(current_candle['high'])
        current_low = float(current_candle['low'])
        current_volume = float(current_candle['volume'])
        current_ts = datetime.fromtimestamp(int(current_candle['start']), UTC)
        
        # Calculate 20-period average volume (excluding current candle)
        volume_candles = candles[-(VOLUME_PERIOD+1):-1]
        avg_volume = sum(float(c['volume']) for c in volume_candles) / len(volume_candles)
        
        # Check volume confirmation (1.25x above average)
        volume_confirmed = current_volume >= (VOLUME_SURGE_FACTOR * avg_volume)
        
        logger.info(f"Current ETH price: ${current_close:,.2f}")
        logger.info(f"Volume: {current_volume:,.0f} (avg: {avg_volume:,.0f}, required: {VOLUME_SURGE_FACTOR}x) -> {'✅' if volume_confirmed else '❌'}")
        
        # Plan A - Momentum Breakout Conditions
        breakout_condition = (
            current_close >= BREAKOUT_RESISTANCE_LOW and 
            current_close <= BREAKOUT_RESISTANCE_HIGH and 
            volume_confirmed and 
            not plan_a_state.get("triggered", False)
        )
        
        # Plan B - Pullback Buy Conditions
        pullback_condition = (
            current_close >= PULLBACK_BID_LOW and 
            current_close <= PULLBACK_BID_HIGH and 
            volume_confirmed and 
            not plan_b_state.get("triggered", False)
        )
        
        logger.info(f"Plan A (Breakout): ${BREAKOUT_RESISTANCE_LOW}-${BREAKOUT_RESISTANCE_HIGH} -> {'✅' if breakout_condition else '❌'}")
        logger.info(f"Plan B (Pullback): ${PULLBACK_BID_LOW}-${PULLBACK_BID_HIGH} -> {'✅' if pullback_condition else '❌'}")
        
        # Execute Plan A - Momentum Breakout
        if breakout_condition:
            logger.info("🎯 PLAN A CONDITION MET - EXECUTING MOMENTUM BREAKOUT TRADE!")
            play_alert_sound()
            
            # Calculate stop loss (below breakout candle or <$3,860, whichever is lower)
            stop_loss = min(current_low, BREAKOUT_STOP_LOSS)
            
            # Execute the trade with first target
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD Plan A - Momentum Breakout",
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=BREAKOUT_TP1,  # First target at $4,050
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            
            logger.info(f"Plan A trade execution completed: success={trade_success}")
            
            if trade_success:
                logger.info(f"🎉 Plan A - Momentum Breakout trade executed successfully!")
                logger.info(f"Entry: ${current_close:,.2f}")
                logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                logger.info(f"First profit target: ${BREAKOUT_TP1:,.2f}")
                logger.info(f"Second profit target: ${BREAKOUT_TP2_LOW}-${BREAKOUT_TP2_HIGH:,.2f}")
                logger.info(f"Trade output: {trade_result}")
                logger.info("📊 Strategy: Key resistance cluster break opens $4,050-$4,170")
            else:
                logger.error(f"❌ Plan A trade failed: {trade_result}")
            
            # Save trigger state to prevent duplicate trades
            plan_a_state = {"triggered": True, "trigger_ts": int(current_candle['start'])}
            save_trigger_state(plan_a_state, PLAN_A_TRIGGER_FILE)
            logger.info("Plan A trigger state saved")
        
        # Execute Plan B - Pullback Buy
        elif pullback_condition:
            logger.info("🎯 PLAN B CONDITION MET - EXECUTING PULLBACK BUY TRADE!")
            play_alert_sound()
            
            # Calculate stop loss (<$3,640 or 1h close < $3,680)
            stop_loss = min(PULLBACK_STOP_LOSS, PULLBACK_STOP_LOSS_ALT)
            
            # Execute the trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD Plan B - Pullback Buy",
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=PULLBACK_TP_HIGH,  # Target back into $3,880-$3,920
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            
            logger.info(f"Plan B trade execution completed: success={trade_success}")
            
            if trade_success:
                logger.info(f"🎉 Plan B - Pullback Buy trade executed successfully!")
                logger.info(f"Entry: ${current_close:,.2f}")
                logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                logger.info(f"Profit target: ${PULLBACK_TP_LOW}-${PULLBACK_TP_HIGH:,.2f}")
                logger.info(f"Trade output: {trade_result}")
                logger.info("📊 Strategy: First support zone that's been holding recently")
            else:
                logger.error(f"❌ Plan B trade failed: {trade_result}")
            
            # Save trigger state to prevent duplicate trades
            plan_b_state = {"triggered": True, "trigger_ts": int(current_candle['start'])}
            save_trigger_state(plan_b_state, PLAN_B_TRIGGER_FILE)
            logger.info("Plan B trigger state saved")
        
        else:
            logger.info("⏳ Waiting for Plan A or Plan B conditions...")
            if not volume_confirmed:
                logger.info(f"   Volume {current_volume:,.0f} below required {VOLUME_SURGE_FACTOR}x average")
            if plan_a_state.get("triggered", False):
                logger.info("   Plan A already triggered")
            if plan_b_state.get("triggered", False):
                logger.info("   Plan B already triggered")
        
        # Reset triggers if price moves significantly away from entry zones
        if plan_a_state.get("triggered", False):
            if current_close < BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Plan A trigger state - price fell below stop loss")
                plan_a_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(plan_a_state, PLAN_A_TRIGGER_FILE)
                logger.info("Plan A trigger state reset")
        
        if plan_b_state.get("triggered", False):
            if current_close < PULLBACK_STOP_LOSS:
                logger.info("🔄 Resetting Plan B trigger state - price fell below stop loss")
                plan_b_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(plan_b_state, PLAN_B_TRIGGER_FILE)
                logger.info("Plan B trigger state reset")
        
        logger.info("=== ETH-USD Trading Plan Alert completed ===")
        return current_ts
        
    except Exception as e:
        logger.error(f"Error in ETH-USD Trading Plan Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH-USD Trading Plan Alert completed (with error) ===")
    return last_alert_ts

# Replace main loop to use new alert
def main():
    logger.info("Starting ETH-USD 1-Hour Trading Plan Monitor")
    logger.info("🎯 Monitoring for Plan A (Momentum Breakout) and Plan B (Pullback Buy)")
    logger.info("📊 Plan A: Buy on 1h close ≥ $3,920-3,930 (key resistance cluster)")
    logger.info("📊 Plan B: Bid $3,700-$3,730 (first support zone)")
    logger.info("💡 Volume confirmation: ≥1.25x 20-period 1h volume")
    logger.info("🛑 Plan A SL: <$3,860, Plan B SL: <$3,640")
    logger.info("🎯 Plan A TP: $4,050 / $4,170-$4,300, Plan B TP: $3,880-$3,920")
    logger.info("⏰ Timeframe: 1-hour")
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
        last_alert_ts = eth_trading_plan_alert(cb_service, last_alert_ts)
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