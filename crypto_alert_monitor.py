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
        logging.FileHandler('btc_intraday_alert.log'),
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
    """Calculate exponential backoff delay with jitter"""
    import random
    delay = min(INITIAL_RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt), MAX_RETRY_DELAY)
    # Add jitter to prevent thundering herd
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
    """
    Retry a function with exponential backoff on connection errors
    
    Args:
        func: Function to retry
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Function result on success, None on permanent failure
    """
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
            # For non-connection errors, don't retry
            logger.error(f"❌ Non-recoverable error: {e}")
            return None
    
    return None

def safe_get_candles(cb_service, product_id, start_ts, end_ts, granularity):
    """
    Safely get candles with retry logic
    
    Args:
        cb_service: Coinbase service instance
        product_id: Trading product ID
        start_ts: Start timestamp
        end_ts: End timestamp
        granularity: Candle granularity
    
    Returns:
        Candles list on success, None on failure
    """
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

def safe_get_5m_candles(cb_service, product_id, start_ts, end_ts):
    """
    Safely get 5-minute candles with retry logic
    """
    def _get_5m_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity="FIVE_MINUTE"
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_5m_candles)

# Constants for BTC intraday strategy
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global Rules from the plan
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
RISK_PERCENTAGE = 0.8  # 0.8-1.2% of price as 1R
VOLUME_THRESHOLD_1H = 1.25  # 1.25x 20-SMA volume on 1h
VOLUME_THRESHOLD_5M = 2.0   # 2x 20-SMA volume on 5m

# LONG - Breakout Strategy Parameters
BREAKOUT_ENTRY_LOW = 119050   # $119,050 (above HOD + buffer)
BREAKOUT_ENTRY_HIGH = 119250  # $119,250 (above HOD + buffer)
BREAKOUT_STOP_LOSS = 118450   # $118,450
BREAKOUT_TP1_LOW = 120000     # $120,000 (below ~ATH/52-wk high)
BREAKOUT_TP1_HIGH = 120200    # $120,200
BREAKOUT_TP2_LOW = 121800     # $121,800
BREAKOUT_TP2_HIGH = 122300    # $122,300

# LONG - Retest Strategy Parameters
RETEST_ENTRY_LOW = 117800     # $117,800
RETEST_ENTRY_HIGH = 118100    # $118,100
RETEST_SWEEP_LOW = 117300     # $117,300 (sweep into this zone)
RETEST_SWEEP_HIGH = 117600    # $117,600 (sweep into this zone)
RETEST_STOP_LOSS = 116900     # $116,900
RETEST_TP1 = 118900           # $118,900
RETEST_TP2_LOW = 119800       # $119,800
RETEST_TP2_HIGH = 120000      # $120,000

# Trade tracking
TRIGGER_STATE_FILE = "btc_intraday_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"breakout_triggered": False, "retest_triggered": False, "last_trigger_ts": None}
    return {"breakout_triggered": False, "retest_triggered": False, "last_trigger_ts": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def play_alert_sound(filename="alert_sound.wav"):
    """
    Play the alert sound using system commands
    """
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
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
    """Setup Coinbase service with connection validation"""
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    
    def _create_service():
        service = CoinbaseService(api_key, api_secret)
        # Test the connection with a simple API call
        try:
            # Try to get a small amount of candle data to validate connection
            test_response = service.client.get_public_candles(
                product_id="BTC-PERP-INTX",
                start=int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity="ONE_HOUR"
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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = "BTC-PERP-INTX"):
    """
    General crypto trade execution function using trade_btc_perp.py with retry logic
    
    Args:
        cb_service: Coinbase service instance
        trade_type: Description of the trade type for logging
        entry_price: Entry price for logging
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        margin: USD amount to risk (default: 250)
        leverage: Leverage multiplier (default: 20)
        side: Trade side - "BUY" or "SELL" (default: "BUY")
        product: Trading product (default: "BTC-PERP-INTX")
    """
    def _execute_trade():
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        
        # Calculate position size based on margin and leverage
        position_size_usd = margin * leverage
        
        # Use subprocess to call trade_btc_perp.py
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', product,
            '--side', side,
            '--size', str(position_size_usd),
            '--leverage', str(leverage),
            '--tp', str(take_profit),
            '--sl', str(stop_loss),
            '--no-confirm'  # Skip confirmation for automated trading
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute the trade command
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
        # Use retry logic for trade execution
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

def calculate_volume_sma(candles, period=20):
    """
    Calculate Simple Moving Average of volume
    
    Args:
        candles: List of candle data
        period: Period for SMA calculation
    
    Returns:
        Volume SMA value
    """
    if len(candles) < period:
        return 0
    
    volumes = []
    for candle in candles[1:period+1]:  # Skip current candle, use previous period candles
        if isinstance(candle, dict):
            volume = float(candle.get('volume', 0))
        else:
            volume = float(getattr(candle, 'volume', 0))
        volumes.append(volume)
    
    return sum(volumes) / len(volumes) if volumes else 0

def get_candle_value(candle, key):
    """Extract value from candle object (handles both dict and object formats)"""
    if isinstance(candle, dict):
        return candle.get(key)
    else:
        return getattr(candle, key, None)

def btc_intraday_alert(cb_service, last_alert_ts=None):
    """
    BTC Intraday Alert - Implements both Breakout and Retest strategies
    Based on the trading plan: "Spiros - BTC intraday plan (both sides) using live levels"
    """
    logger.info("=== BTC Intraday Alert (Breakout + Retest Strategies) ===")
    
    # Load trigger state
    trigger_state = load_trigger_state()
    
    try:
        # Get current time and calculate time ranges
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)  # Start of current hour
        
        # Get 1-hour candles for main analysis
        start_1h = now - timedelta(hours=25)  # Get 25 hours of data
        end_1h = now
        start_ts_1h = int(start_1h.timestamp())
        end_ts_1h = int(end_1h.timestamp())
        
        # Get 5-minute candles for volume confirmation
        start_5m = now - timedelta(hours=2)  # Get 2 hours of 5m data
        end_5m = now
        start_ts_5m = int(start_5m.timestamp())
        end_ts_5m = int(end_5m.timestamp())
        
        logger.info(f"Fetching 1-hour candles from {start_1h} to {end_1h}")
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts_1h, end_ts_1h, GRANULARITY_1H)
        
        logger.info(f"Fetching 5-minute candles from {start_5m} to {end_5m}")
        candles_5m = safe_get_5m_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts_5m)
        
        if not candles_1h or len(candles_1h) < 3:
            logger.warning("Not enough 1-hour candle data for analysis")
            return last_alert_ts
            
        if not candles_5m or len(candles_5m) < 24:  # Need at least 2 hours of 5m data
            logger.warning("Not enough 5-minute candle data for volume analysis")
            return last_alert_ts
        
        # Get current and previous 1-hour candles
        current_1h = candles_1h[0]  # Most recent candle (may be in progress)
        last_1h = candles_1h[1]     # Last completed 1-hour candle
        prev_1h = candles_1h[2]     # Previous completed 1-hour candle
        
        # Extract values from last completed 1-hour candle
        last_ts = datetime.fromtimestamp(int(get_candle_value(last_1h, 'start')), UTC)
        last_close = float(get_candle_value(last_1h, 'close'))
        last_high = float(get_candle_value(last_1h, 'high'))
        last_low = float(get_candle_value(last_1h, 'low'))
        last_volume = float(get_candle_value(last_1h, 'volume'))
        
        # Get current price from most recent 5-minute candle
        current_5m = candles_5m[0]
        current_price = float(get_candle_value(current_5m, 'close'))
        
        # Calculate volume SMAs
        volume_sma_1h = calculate_volume_sma(candles_1h, 20)
        volume_sma_5m = calculate_volume_sma(candles_5m, 24)  # 2 hours of 5m data
        
        # Calculate relative volumes
        relative_volume_1h = last_volume / volume_sma_1h if volume_sma_1h > 0 else 0
        current_5m_volume = float(get_candle_value(current_5m, 'volume'))
        relative_volume_5m = current_5m_volume / volume_sma_5m if volume_sma_5m > 0 else 0
        
        # Check for sweep into retest zone (look at recent 5m candles)
        sweep_detected = False
        for candle in candles_5m[1:13]:  # Check last hour of 5m candles
            low = float(get_candle_value(candle, 'low'))
            if RETEST_SWEEP_LOW <= low <= RETEST_SWEEP_HIGH:
                sweep_detected = True
                break
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 BTC Intraday Trading Plan Alert")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Trigger: 1h timeframe, execute on 5-15m timeframe")
        logger.info(f"   • Volume confirm: ≥{VOLUME_THRESHOLD_1H}x 20-SMA on 1h OR ≥{VOLUME_THRESHOLD_5M}x 20-SMA on 5m")
        logger.info(f"   • Risk: Size so 1R is ~{RISK_PERCENTAGE}% of price")
        logger.info(f"   • Context: ~$120k is near-term cap; expect whips if volume weak")
        logger.info("")
        logger.info("📊 LONG - Breakout Strategy:")
        logger.info(f"   • Entry: Buy-stop at ${BREAKOUT_ENTRY_LOW:,}-${BREAKOUT_ENTRY_HIGH:,} (above HOD + buffer)")
        logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,}")
        logger.info(f"   • TP1: ${BREAKOUT_TP1_LOW:,}-${BREAKOUT_TP1_HIGH:,}")
        logger.info(f"   • TP2: ${BREAKOUT_TP2_LOW:,}-${BREAKOUT_TP2_HIGH:,} (below ~ATH/52-wk high)")
        logger.info(f"   • Why: Range expansion through HOD with room before prior extremes")
        logger.info("")
        logger.info("📊 LONG - Retest Strategy:")
        logger.info(f"   • Entry: ${RETEST_ENTRY_LOW:,}-${RETEST_ENTRY_HIGH:,}")
        logger.info(f"   • Conditions: Only after sweep into ${RETEST_SWEEP_LOW:,}-${RETEST_SWEEP_HIGH:,} and 5-15m reclaim")
        logger.info(f"   • SL: ${RETEST_STOP_LOSS:,}")
        logger.info(f"   • TP1: ${RETEST_TP1:,}")
        logger.info(f"   • TP2: ${RETEST_TP2_LOW:,}-${RETEST_TP2_HIGH:,}")
        logger.info(f"   • Why: Higher low at mid-range without chasing")
        logger.info("")
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_close:,.2f}, High: ${last_high:,.2f}, Low: ${last_low:,.2f}")
        logger.info(f"1H Volume: {last_volume:,.0f}, 1H SMA: {volume_sma_1h:,.0f}, Rel_Vol: {relative_volume_1h:.2f}")
        logger.info(f"5M Volume: {current_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}")
        logger.info(f"Sweep Detected: {'✅' if sweep_detected else '❌'}")
        logger.info("")
        
        # --- Breakout Strategy Analysis ---
        breakout_conditions = []
        
        # Check if price is in breakout entry zone
        in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_price <= BREAKOUT_ENTRY_HIGH
        
        # Check volume confirmation (1h OR 5m)
        volume_confirmed_1h = relative_volume_1h >= VOLUME_THRESHOLD_1H
        volume_confirmed_5m = relative_volume_5m >= VOLUME_THRESHOLD_5M
        volume_confirmed = volume_confirmed_1h or volume_confirmed_5m
        
        # Check if not already triggered
        not_already_triggered = not trigger_state.get("breakout_triggered", False)
        
        breakout_conditions = [in_breakout_zone, volume_confirmed, not_already_triggered]
        breakout_ready = all(breakout_conditions)
        
        logger.info("🔍 Breakout Strategy Analysis:")
        logger.info(f"   • Price in entry zone (${BREAKOUT_ENTRY_LOW:,}-${BREAKOUT_ENTRY_HIGH:,}): {'✅' if in_breakout_zone else '❌'}")
        logger.info(f"   • Volume confirmed (1H: {relative_volume_1h:.2f}x, 5M: {relative_volume_5m:.2f}x): {'✅' if volume_confirmed else '❌'}")
        logger.info(f"   • Not already triggered: {'✅' if not_already_triggered else '❌'}")
        logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")
        
        # --- Retest Strategy Analysis ---
        retest_conditions = []
        
        # Check if price is in retest entry zone
        in_retest_zone = RETEST_ENTRY_LOW <= current_price <= RETEST_ENTRY_HIGH
        
        # Check if sweep was detected
        sweep_condition = sweep_detected
        
        # Check if price reclaimed above the sweep zone (5-15m reclaim)
        reclaim_condition = current_price > RETEST_SWEEP_HIGH
        
        # Check if not already triggered
        not_already_triggered_retest = not trigger_state.get("retest_triggered", False)
        
        retest_conditions = [in_retest_zone, sweep_condition, reclaim_condition, not_already_triggered_retest]
        retest_ready = all(retest_conditions)
        
        logger.info("")
        logger.info("🔍 Retest Strategy Analysis:")
        logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,}-${RETEST_ENTRY_HIGH:,}): {'✅' if in_retest_zone else '❌'}")
        logger.info(f"   • Sweep detected (${RETEST_SWEEP_LOW:,}-${RETEST_SWEEP_HIGH:,}): {'✅' if sweep_condition else '❌'}")
        logger.info(f"   • 5-15m reclaim (price > ${RETEST_SWEEP_HIGH:,}): {'✅' if reclaim_condition else '❌'}")
        logger.info(f"   • Not already triggered: {'✅' if not_already_triggered_retest else '❌'}")
        logger.info(f"   • Retest Ready: {'🎯 YES' if retest_ready else '⏳ NO'}")
        
        # --- Execute Trades ---
        trade_executed = False
        
        if breakout_ready:
            logger.info("")
            logger.info("🎯 Breakout Strategy conditions met - executing trade...")
            
            # Use current price as entry
            entry_price = current_price
            
            logger.info(f"Trade Setup: Entry=${entry_price:,.2f}, SL=${BREAKOUT_STOP_LOSS:,.2f}")
            logger.info(f"TP1: ${BREAKOUT_TP1_LOW:,.2f}-${BREAKOUT_TP1_HIGH:,.2f}")
            logger.info(f"TP2: ${BREAKOUT_TP2_LOW:,.2f}-${BREAKOUT_TP2_HIGH:,.2f}")
            logger.info(f"Risk: ${MARGIN}, Leverage: {LEVERAGE}x")
            
            # Play alert sound
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            
            # Execute Breakout trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="BTC Intraday Breakout Long",
                entry_price=entry_price,
                stop_loss=BREAKOUT_STOP_LOSS,
                take_profit=BREAKOUT_TP1_LOW,  # Use TP1 as primary target
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            
            if trade_success:
                logger.info(f"🎉 Breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
                trigger_state["breakout_triggered"] = True
                trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                save_trigger_state(trigger_state)
                trade_executed = True
            else:
                logger.error(f"❌ Breakout trade failed: {trade_result}")
        
        elif retest_ready:
            logger.info("")
            logger.info("🎯 Retest Strategy conditions met - executing trade...")
            
            # Use current price as entry
            entry_price = current_price
            
            logger.info(f"Trade Setup: Entry=${entry_price:,.2f}, SL=${RETEST_STOP_LOSS:,.2f}")
            logger.info(f"TP1: ${RETEST_TP1:,.2f}")
            logger.info(f"TP2: ${RETEST_TP2_LOW:,.2f}-${RETEST_TP2_HIGH:,.2f}")
            logger.info(f"Risk: ${MARGIN}, Leverage: {LEVERAGE}x")
            
            # Play alert sound
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            
            # Execute Retest trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="BTC Intraday Retest Long",
                entry_price=entry_price,
                stop_loss=RETEST_STOP_LOSS,
                take_profit=RETEST_TP1,  # Use TP1 as primary target
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            
            if trade_success:
                logger.info(f"🎉 Retest trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
                trigger_state["retest_triggered"] = True
                trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                save_trigger_state(trigger_state)
                trade_executed = True
            else:
                logger.error(f"❌ Retest trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for either strategy")
            logger.info(f"Breakout triggered: {trigger_state.get('breakout_triggered', False)}")
            logger.info(f"Retest triggered: {trigger_state.get('retest_triggered', False)}")
        
        logger.info("=== BTC Intraday Alert completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in BTC Intraday alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== BTC Intraday Alert completed (with error) ===")
    return last_alert_ts

def main():
    logger.info("Starting BTC Intraday Alert Monitor (Breakout + Retest Strategies)")
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
        last_alert_ts = btc_intraday_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        logger.info(f"✅ Intraday alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)  # 2 minute max per poll
                    wait_seconds = 300  # 5 minutes between polls
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