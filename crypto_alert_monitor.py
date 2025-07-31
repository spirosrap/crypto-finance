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
LEVERAGE = 20  # 20x leverage (margin x leverage = 5000 USD position size)
RISK_PERCENTAGE = 0.8  # 0.8-1.0% of price as 1R
VOLUME_THRESHOLD_1H = 1.25  # 1.25x 20-SMA volume on 1h
VOLUME_THRESHOLD_5M = 2.0   # 2x 20-SMA volume on 5m

# Today's session levels from the plan
HOD = 118891  # High of Day
LOD = 116040  # Low of Day  
MID = 117466  # Mid point of today's range

# LONG - Breakout continuation strategy
BREAKOUT_ENTRY_LOW = 118950   # Entry zone low (above HOD by ~0.05%)
BREAKOUT_ENTRY_HIGH = 119150  # Entry zone high (above HOD by ~0.22%)
BREAKOUT_STOP_LOSS = 118480   # SL back inside range (0.45-0.60% below entry)
BREAKOUT_TP1 = 120320         # TP1 ≈ 0.5× today's range
BREAKOUT_TP2 = 121740         # TP2 ≈ 1.0× today's range

# LONG - Failed break / LOD reclaim strategy
RECLAIM_SWEEP_LOW = 115900    # Brief sweep ≤ 0.15% below LOD (115900 = LOD * 0.9985)
RECLAIM_ENTRY_LOW = 116100    # Entry zone low on reclaim
RECLAIM_ENTRY_HIGH = 116250   # Entry zone high on reclaim
RECLAIM_STOP_LOSS = 115750    # SL below sweep
RECLAIM_TP1 = 117460          # TP1 range mid
RECLAIM_TP2 = 118800          # TP2 revisit HOD

# SHORT - Breakdown continuation strategy
BREAKDOWN_ENTRY_LOW = 115900   # Entry zone low (below LOD by ~0.12%)
BREAKDOWN_ENTRY_HIGH = 115700  # Entry zone high (below LOD by ~0.29%)
BREAKDOWN_STOP_LOSS = 116250   # SL back inside range
BREAKDOWN_TP1 = 114615         # TP1 ≈ 0.5× range extension
BREAKDOWN_TP2 = 113190         # TP2 ≈ 1.0× range extension

# SHORT - Fade under HOD strategy
FADE_ENTRY_LOW = 118700        # Entry zone low (stall ≤118850-118890)
FADE_ENTRY_HIGH = 118820       # Entry zone high
FADE_STOP_LOSS = 119200        # SL above HOD
FADE_TP1 = 118000              # TP1 VWAP/first flush
FADE_TP2 = 117460              # TP2 mid
FADE_TP3 = 116150              # TP3 final target

# Trade tracking
TRIGGER_STATE_FILE = "btc_intraday_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "breakout_triggered": False, 
                "reclaim_triggered": False, 
                "breakdown_triggered": False,
                "fade_triggered": False,
                "last_trigger_ts": None
            }
    return {
        "breakout_triggered": False, 
        "reclaim_triggered": False, 
        "breakdown_triggered": False,
        "fade_triggered": False,
        "last_trigger_ts": None
    }

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

def check_spike_rejection(candles_5m, resistance_level):
    """
    Check for spike and rejection pattern at resistance level
    
    Args:
        candles_5m: List of 5-minute candles
        resistance_level: Price level to check for rejection
    
    Returns:
        True if spike and rejection detected, False otherwise
    """
    if len(candles_5m) < 3:
        return False
    
    # Check last 3 candles for spike and rejection
    for i in range(min(3, len(candles_5m))):
        candle = candles_5m[i]
        high = float(get_candle_value(candle, 'high'))
        low = float(get_candle_value(candle, 'low'))
        close = float(get_candle_value(candle, 'close'))
        open_price = float(get_candle_value(candle, 'open'))
        
        # Check if candle spiked above resistance and closed below it
        if high > resistance_level and close < resistance_level:
            # Calculate upper wick (spike)
            upper_wick = high - max(open_price, close)
            body = abs(high - low)
            
            # Upper wick should be significant (at least 30% of body)
            if upper_wick > 0.3 * body:
                return True
    
    return False

def btc_intraday_alert(cb_service, last_alert_ts=None):
    """
    BTC Intraday Alert - Implements complete trading plan with both LONG and SHORT strategies
    Based on the trading plan: "Spiros — BTC plan for today (live levels)"
    """
    logger.info("=== BTC Intraday Alert (Complete Strategy - LONG & SHORT) ===")
    
    # Load trigger state
    trigger_state = load_trigger_state()
    
    try:
        # Get current time and calculate time ranges
        current_time = datetime.now(UTC)
        now_hour = current_time.replace(minute=0, second=0, microsecond=0)  # Start of current hour
        
        # Get 1-hour candles for main analysis
        start_1h = now_hour - timedelta(hours=25)  # Get 25 hours of data
        end_1h = now_hour
        start_ts_1h = int(start_1h.timestamp())
        end_ts_1h = int(end_1h.timestamp())
        
        # Get 5-minute candles for volume confirmation and pattern analysis
        start_5m = current_time - timedelta(hours=2)  # Get 2 hours of 5m data
        end_5m = current_time  # Use actual current time, not rounded hour
        start_ts_5m = int(start_5m.timestamp())
        end_ts_5m = int(end_5m.timestamp())
        
        logger.info(f"Fetching 1-hour candles from {start_1h} to {end_1h}")
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts_1h, end_ts_1h, GRANULARITY_1H)
        
        logger.info(f"Fetching 5-minute candles from {start_5m} to {end_5m} (current time: {current_time})")
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
            if RECLAIM_SWEEP_LOW <= low <= RECLAIM_ENTRY_HIGH: # Changed to RECLAIM_SWEEP_LOW and RECLAIM_ENTRY_HIGH
                sweep_detected = True
                break
        
        # Check for spike rejection at resistance level (HOD area)
        spike_rejection_detected = check_spike_rejection(candles_5m, HOD)
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 BTC Plan for Today (Live Levels) Alert")
        logger.info("")
        logger.info("📊 Today's Levels:")
        logger.info(f"   • BTC ≈ ${current_price:,.0f}")
        logger.info(f"   • HOD: ${HOD:,}")
        logger.info(f"   • LOD: ${LOD:,}")
        logger.info(f"   • MID: ${MID:,}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Timeframe: 1h trigger; execute on 5-15m")
        logger.info(f"   • Volume confirm: ≥{VOLUME_THRESHOLD_1H}x 20-SMA on 1h OR ≥{VOLUME_THRESHOLD_5M}x 20-SMA on 5m")
        logger.info(f"   • Risk: Size so 1R is ~{RISK_PERCENTAGE}% of price")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin x {LEVERAGE}x leverage)")
        logger.info("")
        logger.info("📊 LONG - Breakout Strategy:")
        logger.info(f"   • Entry: ${BREAKOUT_ENTRY_LOW:,}-${BREAKOUT_ENTRY_HIGH:,} (above HOD + buffer)")
        logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,} (back inside range)")
        logger.info(f"   • TP1: ${BREAKOUT_TP1:,}")
        logger.info(f"   • TP2: ${BREAKOUT_TP2:,}")
        logger.info(f"   • Why: Fresh expansion through HOD; continuation if volume confirms")
        logger.info("")
        logger.info("📊 LONG - Reclaim Strategy:")
        logger.info(f"   • Entry: ${RECLAIM_ENTRY_LOW:,}-${RECLAIM_ENTRY_HIGH:,}")
        logger.info(f"   • Conditions: Only after sweep of ${RECLAIM_SWEEP_LOW:,}-${RECLAIM_ENTRY_HIGH:,} and 5-15m reclaim")
        logger.info(f"   • SL: ${RECLAIM_STOP_LOSS:,} (below LOD)")
        logger.info(f"   • TP1: ${RECLAIM_TP1:,}")
        logger.info(f"   • TP2: ${RECLAIM_TP2:,}")
        logger.info(f"   • Why: Liquidity sweep of LOD then higher-low reclaim")
        logger.info("")
        logger.info("📊 SHORT - Breakdown Strategy:")
        logger.info(f"   • Entry: ${BREAKDOWN_ENTRY_LOW:,}-${BREAKDOWN_ENTRY_HIGH:,} (through LOD)")
        logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,}")
        logger.info(f"   • TP1: ${BREAKDOWN_TP1:,}")
        logger.info(f"   • TP2: ${BREAKDOWN_TP2:,}")
        logger.info(f"   • Why: Range failure + continuation if 1h closes below LOD on volume")
        logger.info("")
        logger.info("📊 SHORT - Fade under HOD Strategy:")
        logger.info(f"   • Entry: ${FADE_ENTRY_LOW:,}-${FADE_ENTRY_HIGH:,} only if spike + rejection")
        logger.info(f"   • SL: ${FADE_STOP_LOSS:,}")
        logger.info(f"   • TP1: ${FADE_TP1:,}")
        logger.info(f"   • TP2: ${FADE_TP2:,}")
        logger.info(f"   • TP3: ${FADE_TP3:,}")
        logger.info(f"   • Why: First test into round-number/overhead supply tends to mean-revert intraday")
        logger.info("")
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_close:,.2f}, High: ${last_high:,.2f}, Low: ${last_low:,.2f}")
        logger.info(f"1H Volume: {last_volume:,.0f}, 1H SMA: {volume_sma_1h:,.0f}, Rel_Vol: {relative_volume_1h:.2f}")
        logger.info(f"5M Volume: {current_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}")
        logger.info(f"Sweep Detected: {'✅' if sweep_detected else '❌'}")
        logger.info(f"Spike Rejection Detected: {'✅' if spike_rejection_detected else '❌'}")
        logger.info("")
        
        # --- Volume Confirmation Check ---
        volume_confirmed_1h = relative_volume_1h >= VOLUME_THRESHOLD_1H
        volume_confirmed_5m = relative_volume_5m >= VOLUME_THRESHOLD_5M
        volume_confirmed = volume_confirmed_1h or volume_confirmed_5m
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy
        if not trigger_state.get("breakout_triggered", False):
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_price <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and volume_confirmed
            
            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKOUT_ENTRY_LOW:,}-${BREAKOUT_ENTRY_HIGH:,}): {'✅' if in_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H: {relative_volume_1h:.2f}x, 5M: {relative_volume_5m:.2f}x): {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")
            
            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Strategy conditions met - executing trade...")
                
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
                    entry_price=current_price,
                    stop_loss=BREAKOUT_STOP_LOSS,
                    take_profit=BREAKOUT_TP1,  # Use TP1 as primary target
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
        
        # 2. LONG - Reclaim Strategy
        if not trade_executed and not trigger_state.get("reclaim_triggered", False):
            in_reclaim_zone = RECLAIM_ENTRY_LOW <= current_price <= RECLAIM_ENTRY_HIGH
            reclaim_condition = current_price > RECLAIM_ENTRY_HIGH # Changed to RECLAIM_ENTRY_HIGH
            reclaim_ready = in_reclaim_zone and sweep_detected and reclaim_condition and volume_confirmed
            
            logger.info("")
            logger.info("🔍 LONG - Reclaim Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${RECLAIM_ENTRY_LOW:,}-${RECLAIM_ENTRY_HIGH:,}): {'✅' if in_reclaim_zone else '❌'}")
            logger.info(f"   • Sweep detected (${RECLAIM_SWEEP_LOW:,}-${RECLAIM_ENTRY_HIGH:,}): {'✅' if sweep_detected else '❌'}")
            logger.info(f"   • 5-15m reclaim (price > ${RECLAIM_ENTRY_HIGH:,}): {'✅' if reclaim_condition else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Reclaim Ready: {'🎯 YES' if reclaim_ready else '⏳ NO'}")
            
            if reclaim_ready:
                logger.info("")
                logger.info("🎯 LONG - Reclaim Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Reclaim trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday Reclaim Long",
                    entry_price=current_price,
                    stop_loss=RECLAIM_STOP_LOSS,
                    take_profit=RECLAIM_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Reclaim trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["reclaim_triggered"] = True
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Reclaim trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if not trade_executed and not trigger_state.get("breakdown_triggered", False):
            in_breakdown_zone = BREAKDOWN_ENTRY_LOW <= current_price <= BREAKDOWN_ENTRY_HIGH
            breakdown_ready = in_breakdown_zone and volume_confirmed
            
            logger.info("")
            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKDOWN_ENTRY_LOW:,}-${BREAKDOWN_ENTRY_HIGH:,}): {'✅' if in_breakdown_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")
            
            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday Breakdown Short",
                    entry_price=current_price,
                    stop_loss=BREAKDOWN_STOP_LOSS,
                    take_profit=BREAKDOWN_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakdown trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakdown_triggered"] = True
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown trade failed: {trade_result}")
        
        # 4. SHORT - Fade into Resistance Strategy
        if not trade_executed and not trigger_state.get("fade_triggered", False):
            in_fade_zone = FADE_ENTRY_LOW <= current_price <= FADE_ENTRY_HIGH
            fade_ready = in_fade_zone and spike_rejection_detected and volume_confirmed
            
            logger.info("")
            logger.info("🔍 SHORT - Fade under HOD Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${FADE_ENTRY_LOW:,}-${FADE_ENTRY_HIGH:,}): {'✅' if in_fade_zone else '❌'}")
            logger.info(f"   • Spike rejection detected: {'✅' if spike_rejection_detected else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Fade Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")
            
            if fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fade under HOD Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Fade trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday Fade under HOD Short",
                    entry_price=current_price,
                    stop_loss=FADE_STOP_LOSS,
                    take_profit=FADE_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Fade trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["fade_triggered"] = True
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Fade trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout triggered: {trigger_state.get('breakout_triggered', False)}")
            logger.info(f"Reclaim triggered: {trigger_state.get('reclaim_triggered', False)}")
            logger.info(f"Breakdown triggered: {trigger_state.get('breakdown_triggered', False)}")
            logger.info(f"Fade triggered: {trigger_state.get('fade_triggered', False)}")
        
        logger.info("=== BTC Intraday Alert completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in BTC Intraday alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== BTC Intraday Alert completed (with error) ===")
    return last_alert_ts

def main():
    logger.info("Starting BTC Plan for Today Alert Monitor (Complete Strategy - LONG & SHORT)")
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