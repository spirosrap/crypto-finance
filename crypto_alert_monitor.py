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
import argparse

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

def safe_get_15m_candles(cb_service, product_id, start_ts, end_ts):
    """
    Safely get 15-minute candles with retry logic
    """
    def _get_15m_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity="FIFTEEN_MINUTE"
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_15m_candles)

# Constants for BTC intraday strategy
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global Rules from the new plan
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage (margin x leverage = 5000 USD position size)
RISK_PERCENTAGE = 0.8  # 0.8-1.2% of price as 1R
VOLUME_THRESHOLD_1H = 1.25  # 1.25x 20-period vol on 1h
VOLUME_THRESHOLD_5M = 2.0   # 2x 20-SMA vol on 5m

# Today's session levels from the new plan (BTC ≈ $114,458, H: $115,729, L: $114,134)
HOD = 115729  # High of Day
LOD = 114134  # Low of Day  
MID = 114931  # Mid point of today's range

# LONG - Breakout strategy
BREAKOUT_ENTRY_LOW = 115800   # Buy-stop 115,800 – 115,950 (HOD + 0.6 – 0.8 %)
BREAKOUT_ENTRY_HIGH = 115950  # Entry zone high
BREAKOUT_STOP_LOSS = 115400   # < 115,400 (back inside range)
BREAKOUT_TP1 = 116900         # TP1
BREAKOUT_TP2 = 118500         # TP2

# LONG - Range-retest strategy
RECLAIM_ENTRY_LOW = 114300    # Limit 114,300 – 114,450 (LOD + 0.15 – 0.25 %)
RECLAIM_ENTRY_HIGH = 114450   # Entry zone high
RECLAIM_STOP_LOSS = 113900    # < 113,900
RECLAIM_TP1 = 115200          # TP1
RECLAIM_TP2 = 115700          # TP2

# SHORT - Breakdown strategy
BREAKDOWN_ENTRY_LOW = 113800   # Sell-stop 113,950 – 113,800 (LOD − 0.15 – 0.3 %)
BREAKDOWN_ENTRY_HIGH = 113950  # Entry zone high
BREAKDOWN_STOP_LOSS = 114300   # > 114,300
BREAKDOWN_TP1 = 113000         # TP1
BREAKDOWN_TP2 = 111800         # TP2

# SHORT - Fail-retest strategy
FADE_ENTRY_LOW = 115600        # Limit 115,600 – 115,800 (near HOD supply)
FADE_ENTRY_HIGH = 115800       # Entry zone high
FADE_STOP_LOSS = 116100        # > 116,100
FADE_TP1 = 114800              # TP1
FADE_TP2 = 114000              # TP2

# Pivot bands for execution
PIVOT_BAND_HIGH = 115000  # Upper pivot band
PIVOT_BAND_LOW = 114000   # Lower pivot band

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
                "retest_reclaim_triggered": False, 
                "range_break_triggered": False,
                "retest_reject_triggered": False,
                "last_trigger_ts": None,
                "last_1h_structure": None,
                "active_trade_direction": None  # Track which direction is active
            }
    return {
        "breakout_triggered": False, 
        "retest_reclaim_triggered": False, 
        "range_break_triggered": False,
        "retest_reject_triggered": False,
        "last_trigger_ts": None,
        "last_1h_structure": None,
        "active_trade_direction": None  # Track which direction is active
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
    for candle in candles[1:period+1]:  # Use most recent period candles (skip current incomplete candle)
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



def btc_intraday_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    Spiros — clean two-sided intraday plan Alert
    Based on the trading plan: "Spiros — clean two-sided intraday plan, live at 05-Aug-2025 06:00 UTC+3"
    
    Rules (both directions):
    - Time-frame: 1 h trigger, execute on 5–15 m
    - Volume confirm: ≥ 1.25 × 20-period vol on the 1 h or ≥ 2 × 20-SMA vol on the 5 m at trigger
    - Risk: size so 1 R ≈ 0.8 – 1.2 % of price; scale out 50 % at +1.0–1.5 R, trail remainder
    - Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
    LONG SCENARIOS:
    - Breakout: Buy-stop 115,800 – 115,950 (HOD + 0.6 – 0.8 %)
    - Range-retest: Limit 114,300 – 114,450 (LOD + 0.15 – 0.25 %)
    
    SHORT SCENARIOS:
    - Breakdown: Sell-stop 113,950 – 113,800 (LOD − 0.15 – 0.3 %)
    - Fail-retest: Limit 115,600 – 115,800 (near HOD supply)
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== Spiros — clean two-sided intraday plan (Complete Strategy - LONG & SHORT) ===")
    else:
        logger.info(f"=== Spiros — clean two-sided intraday plan ({direction} Strategy Only) ===")
    
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
        end_5m = current_time
        start_ts_5m = int(start_5m.timestamp())
        end_ts_5m = int(end_5m.timestamp())
        
        # Get 15-minute candles for pattern analysis
        start_15m = current_time - timedelta(hours=2)  # Get 2 hours of 15m data
        end_15m = current_time
        start_ts_15m = int(start_15m.timestamp())
        end_ts_15m = int(end_15m.timestamp())
        
        logger.info(f"Fetching 1-hour candles from {start_1h} to {end_1h}")
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts_1h, end_ts_1h, GRANULARITY_1H)
        
        logger.info(f"Fetching 5-minute candles from {start_5m} to {end_5m}")
        candles_5m = safe_get_5m_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts_5m)
        
        logger.info(f"Fetching 15-minute candles from {start_15m} to {end_15m}")
        candles_15m = safe_get_15m_candles(cb_service, PRODUCT_ID, start_ts_15m, end_ts_15m)
        
        if not candles_1h or len(candles_1h) < 3:
            logger.warning("Not enough 1-hour candle data for analysis")
            return last_alert_ts
            
        if not candles_5m or len(candles_5m) < 24:  # Need at least 2 hours of 5m data
            logger.warning("Not enough 5-minute candle data for volume analysis")
            return last_alert_ts
        
        if not candles_15m or len(candles_15m) < 8:  # Need at least 2 hours of 15m data
            logger.warning("Not enough 15-minute candle data for pattern analysis")
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
        
        # Check if price is chopping in the no-trade zone
        in_chop_zone = PIVOT_BAND_LOW <= current_price <= PIVOT_BAND_HIGH
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — clean two-sided intraday plan, live at 05-Aug-2025 06:00 UTC+3")
        logger.info(f"(BTC ≈ ${current_price:,.0f}, HOD ${HOD:,}, LOD ${LOD:,})")
        logger.info("")
        logger.info("📊 Rules (both directions):")
        logger.info(f"   • Time-frame: 1 h trigger, execute on 5–15 m")
        logger.info(f"   • Volume confirm: ≥ {VOLUME_THRESHOLD_1H} × 20-period vol on the 1 h or ≥ {VOLUME_THRESHOLD_5M} × 20-SMA vol on the 5 m at trigger")
        logger.info(f"   • Risk: size so 1 R ≈ {RISK_PERCENTAGE} – 1.2 % of price; scale out 50 % at +1.0–1.5 R, trail remainder")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin × {LEVERAGE} leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SCENARIOS:")
            logger.info("")
            logger.info("📊 Type: Breakout")
            logger.info(f"   • Entry zone: Buy-stop ${BREAKOUT_ENTRY_LOW:,} – ${BREAKOUT_ENTRY_HIGH:,} (HOD + 0.6 – 0.8 %)")
            logger.info(f"   • Invalidation (SL): < ${BREAKOUT_STOP_LOSS:,} (back inside range)")
            logger.info(f"   • TP1 / TP2: ${BREAKOUT_TP1:,} / ${BREAKOUT_TP2:,}")
            logger.info(f"   • Why it's high-probability: Range expansion above HOD into thin 1-week supply gap; stops clustered above 115,750 could fuel momentum")
            logger.info("")
            logger.info("📊 Type: Range-retest")
            logger.info(f"   • Entry zone: Limit ${RECLAIM_ENTRY_LOW:,} – ${RECLAIM_ENTRY_HIGH:,} (LOD + 0.15 – 0.25 %)")
            logger.info(f"   • Invalidation (SL): < ${RECLAIM_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${RECLAIM_TP1:,} / ${RECLAIM_TP2:,}")
            logger.info(f"   • Why it's high-probability: Buyers twice defended 114,150 area overnight; sweep + reclaim signals spring and efficient risk")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SCENARIOS:")
            logger.info("")
            logger.info("📊 Type: Breakdown")
            logger.info(f"   • Entry zone: Sell-stop ${BREAKDOWN_ENTRY_LOW:,} – ${BREAKDOWN_ENTRY_HIGH:,} (LOD − 0.15 – 0.3 %)")
            logger.info(f"   • Invalidation (SL): > ${BREAKDOWN_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${BREAKDOWN_TP1:,} / ${BREAKDOWN_TP2:,}")
            logger.info(f"   • Why it's high-probability: Clean air below 114,000 toward late-July support; liquidity vacuum if day lows crack")
            logger.info("")
            logger.info("📊 Type: Fail-retest")
            logger.info(f"   • Entry zone: Limit ${FADE_ENTRY_LOW:,} – ${FADE_ENTRY_HIGH:,} (near HOD supply)")
            logger.info(f"   • Invalidation (SL): > ${FADE_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${FADE_TP1:,} / ${FADE_TP2:,}")
            logger.info(f"   • Why it's high-probability: Fade into prior resistance if breakout stalls; seeks trapped longs from aggressive bids above 115,500")
            logger.info("")
        logger.info("")
        logger.info("📊 Fact vs. opinion:")
        logger.info(f"   • Facts: Live price & 24 h high/low from CoinMarketCap data")
        logger.info(f"   • Opinion: These zones assume typical August liquidity; shrink size if vol < average or if funding flips strongly positive")
        logger.info("")
        logger.info("Execute mechanically, respect stops, and log every trade for post-session review.")
        logger.info("")
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_close:,.2f}, High: ${last_high:,.2f}, Low: ${last_low:,.2f}")
        logger.info(f"1H Volume: {last_volume:,.0f}, 1H SMA: {volume_sma_1h:,.0f}, Rel_Vol: {relative_volume_1h:.2f}")
        logger.info(f"5M Volume: {current_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}")
        logger.info(f"In Chop Zone (${PIVOT_BAND_LOW:,}-${PIVOT_BAND_HIGH:,}): {'✅' if in_chop_zone else '❌'}")
        logger.info("")
        
        # --- Volume Confirmation Check ---
        volume_confirmed_1h = relative_volume_1h >= VOLUME_THRESHOLD_1H
        volume_confirmed_5m = relative_volume_5m >= VOLUME_THRESHOLD_5M
        volume_confirmed = volume_confirmed_1h or volume_confirmed_5m
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check if we should skip trading due to chop zone
        if in_chop_zone and not volume_confirmed:
            logger.info("")
            logger.info("⏸️ Price chopping in no-trade zone without volume expansion - skipping all strategies")
            logger.info("")
        else:
            # 1. LONG - Breakout Strategy
            if long_strategies_enabled and not trigger_state.get("breakout_triggered", False):
                in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_price <= BREAKOUT_ENTRY_HIGH
                breakout_ready = in_breakout_zone and volume_confirmed
                
                logger.info("🔍 LONG - Breakout Analysis:")
                logger.info(f"   • Price in buy-stop zone (${BREAKOUT_ENTRY_LOW:,} – ${BREAKOUT_ENTRY_HIGH:,}): {'✅' if in_breakout_zone else '❌'}")
                logger.info(f"   • Volume confirmed (1H: {relative_volume_1h:.2f}x, 5M: {relative_volume_5m:.2f}x): {'✅' if volume_confirmed else '❌'}")
                logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")
                
                if breakout_ready:
                    logger.info("")
                    logger.info("🎯 LONG - Breakout conditions met - executing trade...")
                    
                    # Play alert sound
                    try:
                        play_alert_sound()
                        logger.info("Alert sound played successfully")
                    except Exception as e:
                        logger.error(f"Failed to play alert sound: {e}")
                    
                    # Execute Breakout trade
                    trade_success, trade_result = execute_crypto_trade(
                        cb_service=cb_service,
                        trade_type="BTC Two-Sided Plan Breakout Long",
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
                        trigger_state["active_trade_direction"] = "LONG"
                        trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                        save_trigger_state(trigger_state)
                        trade_executed = True
                    else:
                        logger.error(f"❌ Breakout trade failed: {trade_result}")
            
            # 2. LONG - Range-retest Strategy
            if long_strategies_enabled and not trade_executed and not trigger_state.get("retest_reclaim_triggered", False):
                in_reclaim_zone = RECLAIM_ENTRY_LOW <= current_price <= RECLAIM_ENTRY_HIGH
                reclaim_ready = in_reclaim_zone and volume_confirmed
                
                logger.info("")
                logger.info("🔍 LONG - Range-retest Analysis:")
                logger.info(f"   • Price in limit entry zone (${RECLAIM_ENTRY_LOW:,} – ${RECLAIM_ENTRY_HIGH:,}): {'✅' if in_reclaim_zone else '❌'}")
                logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
                logger.info(f"   • Range-retest Ready: {'🎯 YES' if reclaim_ready else '⏳ NO'}")
                
                if reclaim_ready:
                    logger.info("")
                    logger.info("🎯 LONG - Range-retest conditions met - executing trade...")
                    
                    # Play alert sound
                    try:
                        play_alert_sound()
                        logger.info("Alert sound played successfully")
                    except Exception as e:
                        logger.error(f"Failed to play alert sound: {e}")
                    
                    # Execute Range-retest trade
                    trade_success, trade_result = execute_crypto_trade(
                        cb_service=cb_service,
                        trade_type="BTC Two-Sided Plan Range-retest Long",
                        entry_price=current_price,
                        stop_loss=RECLAIM_STOP_LOSS,
                        take_profit=RECLAIM_TP1,  # Use TP1 as primary target
                        margin=MARGIN,
                        leverage=LEVERAGE,
                        side="BUY",
                        product=PRODUCT_ID
                    )
                    
                    if trade_success:
                        logger.info(f"🎉 Range-retest trade executed successfully!")
                        logger.info(f"Trade output: {trade_result}")
                        trigger_state["retest_reclaim_triggered"] = True
                        trigger_state["active_trade_direction"] = "LONG"
                        trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                        save_trigger_state(trigger_state)
                        trade_executed = True
                    else:
                        logger.error(f"❌ Range-retest trade failed: {trade_result}")
            
            # 3. SHORT - Breakdown Strategy
            if short_strategies_enabled and not trade_executed and not trigger_state.get("range_break_triggered", False):
                in_breakdown_zone = BREAKDOWN_ENTRY_LOW <= current_price <= BREAKDOWN_ENTRY_HIGH
                breakdown_ready = in_breakdown_zone and volume_confirmed
                
                logger.info("")
                logger.info("🔍 SHORT - Breakdown Analysis:")
                logger.info(f"   • Price in sell-stop zone (${BREAKDOWN_ENTRY_LOW:,} – ${BREAKDOWN_ENTRY_HIGH:,}): {'✅' if in_breakdown_zone else '❌'}")
                logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
                logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")
                
                if breakdown_ready:
                    logger.info("")
                    logger.info("🎯 SHORT - Breakdown conditions met - executing trade...")
                    
                    # Play alert sound
                    try:
                        play_alert_sound()
                        logger.info("Alert sound played successfully")
                    except Exception as e:
                        logger.error(f"Failed to play alert sound: {e}")
                    
                    # Execute Breakdown trade
                    trade_success, trade_result = execute_crypto_trade(
                        cb_service=cb_service,
                        trade_type="BTC Two-Sided Plan Breakdown Short",
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
                        trigger_state["range_break_triggered"] = True
                        trigger_state["active_trade_direction"] = "SHORT"
                        trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                        save_trigger_state(trigger_state)
                        trade_executed = True
                    else:
                        logger.error(f"❌ Breakdown trade failed: {trade_result}")
            
            # 4. SHORT - Fail-retest Strategy
            if short_strategies_enabled and not trade_executed and not trigger_state.get("retest_reject_triggered", False):
                in_fade_zone = FADE_ENTRY_LOW <= current_price <= FADE_ENTRY_HIGH
                fade_ready = in_fade_zone and volume_confirmed
                
                logger.info("")
                logger.info("🔍 SHORT - Fail-retest Analysis:")
                logger.info(f"   • Price in limit entry zone (${FADE_ENTRY_LOW:,} – ${FADE_ENTRY_HIGH:,}): {'✅' if in_fade_zone else '❌'}")
                logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
                logger.info(f"   • Fail-retest Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")
                
                if fade_ready:
                    logger.info("")
                    logger.info("🎯 SHORT - Fail-retest conditions met - executing trade...")
                    
                    # Play alert sound
                    try:
                        play_alert_sound()
                        logger.info("Alert sound played successfully")
                    except Exception as e:
                        logger.error(f"Failed to play alert sound: {e}")
                    
                    # Execute Fail-retest trade
                    trade_success, trade_result = execute_crypto_trade(
                        cb_service=cb_service,
                        trade_type="BTC Two-Sided Plan Fail-retest Short",
                        entry_price=current_price,
                        stop_loss=FADE_STOP_LOSS,
                        take_profit=FADE_TP1,  # Use TP1 as primary target
                        margin=MARGIN,
                        leverage=LEVERAGE,
                        side="SELL",
                        product=PRODUCT_ID
                    )
                    
                    if trade_success:
                        logger.info(f"🎉 Fail-retest trade executed successfully!")
                        logger.info(f"Trade output: {trade_result}")
                        trigger_state["retest_reject_triggered"] = True
                        trigger_state["active_trade_direction"] = "SHORT"
                        trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                        save_trigger_state(trigger_state)
                        trade_executed = True
                    else:
                        logger.error(f"❌ Fail-retest trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout triggered: {trigger_state.get('breakout_triggered', False)}")
            logger.info(f"Range-retest triggered: {trigger_state.get('retest_reclaim_triggered', False)}")
            logger.info(f"Breakdown triggered: {trigger_state.get('range_break_triggered', False)}")
            logger.info(f"Fail-retest triggered: {trigger_state.get('retest_reject_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — clean two-sided intraday plan completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — clean two-sided intraday plan logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — clean two-sided intraday plan completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BTC Intraday Alert Monitor with optional direction filter')
    parser.add_argument('--direction', choices=['LONG', 'SHORT', 'BOTH'], default='BOTH',
                       help='Trading direction to monitor: LONG, SHORT, or BOTH (default: BOTH)')
    args = parser.parse_args()
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    logger.info("Strategy Overview:")
    logger.info("  • LONG Breakout: Buy-stop $115,800-$115,950 (HOD + 0.6-0.8%)")
    logger.info("  • LONG Range-retest: Limit $114,300-$114,450 (LOD + 0.15-0.25%)")
    logger.info("  • SHORT Breakdown: Sell-stop $113,800-$113,950 (LOD - 0.15-0.3%)")
    logger.info("  • SHORT Fail-retest: Limit $115,600-$115,800 (near HOD supply)")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20 leverage)")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — clean two-sided intraday plan Alert Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: Complete Two-Sided BTC Plan - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
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
        last_alert_ts = btc_intraday_alert(cb_service, last_alert_ts, direction)
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