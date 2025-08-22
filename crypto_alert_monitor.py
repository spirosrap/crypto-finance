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

# Constants for BTC Jackson Hole Trading Setup
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global rules
MARGIN = 250  # USD
LEVERAGE = 20  # Always margin x leverage = 250 x 20 = $5,000 position size
RISK_PERCENTAGE = 0.5

# Session snapshot (for reporting only) - Updated intraday context
CURRENT_PRICE = 113296
TODAY_HOD = 113960
TODAY_LOD = 112021
TWENTY_FOUR_HOUR_HIGH = 113960
TWENTY_FOUR_HOUR_LOW = 112021

# Key levels from updated intraday setup
# Range today: 112,021–113,960
# Mid-range chop zone to avoid: 112,200–113,900

# 1) Breakout Continuation LONG - Updated Strategy
BREAKOUT_LONG_TRIGGER_LEVEL = 114000     # 1h close ≥ 114,000 trigger
BREAKOUT_LONG_ENTRY_LOW = 114050         # Entry: 114,050–114,300 on retest or next 1h open  
BREAKOUT_LONG_ENTRY_HIGH = 114300
BREAKOUT_LONG_INVALIDATION = 113400      # Invalidation: 113,400 (or below breakout bar low)
BREAKOUT_LONG_TP1 = 115000               # TP1: 115,000
BREAKOUT_LONG_TP2 = 116000               # TP2: 116,000 (range extension)
BREAKOUT_LONG_VOLUME_THRESHOLD_1H = 1.25 # ≥1.25× 20-MA on 1h timeframe

# 2) Range Low Reclaim LONG - Updated Strategy  
RANGE_LOW_SWEEP_LOW = 112000             # Sweep of 112,000–112,100
RANGE_LOW_SWEEP_HIGH = 112100
RANGE_LOW_RECLAIM_LEVEL = 112200         # 15m close back above 112,200
RANGE_LOW_ENTRY_LOW = 112220             # Entry: 112,220–112,350
RANGE_LOW_ENTRY_HIGH = 112350
RANGE_LOW_INVALIDATION = 111800          # Invalidation: 111,800
RANGE_LOW_TP1 = 113000                   # TP1: 113,000
RANGE_LOW_TP2 = 113800                   # TP2: 113,800–114,000
RANGE_LOW_TP2_HIGH = 114000

# 3) Breakdown Continuation SHORT - Updated Strategy
BREAKDOWN_SHORT_TRIGGER_LEVEL = 111950   # 1h close ≤ 111,950 trigger
BREAKDOWN_SHORT_ENTRY_LOW = 111700       # Entry: 111,900–111,700 on retest
BREAKDOWN_SHORT_ENTRY_HIGH = 111900
BREAKDOWN_SHORT_INVALIDATION = 112400    # Invalidation: 112,400
BREAKDOWN_SHORT_TP1 = 110800             # TP1: 110,800
BREAKDOWN_SHORT_TP2 = 110000             # TP2: 110,000
BREAKDOWN_SHORT_VOLUME_THRESHOLD_1H = 1.25 # ≥1.25× 20-MA on 1h timeframe

# 4) Range High Rejection SHORT - Updated Strategy
RANGE_HIGH_REJECTION_LOW = 113900        # Wick above 113,900–114,000
RANGE_HIGH_REJECTION_HIGH = 114000
RANGE_HIGH_ENTRY_LOW = 113600            # Entry: 113,780–113,600  
RANGE_HIGH_ENTRY_HIGH = 113780
RANGE_HIGH_INVALIDATION = 114200         # Invalidation: 114,200
RANGE_HIGH_TP1 = 113000                  # TP1: 113,000
RANGE_HIGH_TP2_LOW = 112200              # TP2: 112,400–112,200
RANGE_HIGH_TP2_HIGH = 112400

# Mid-range chop zone to avoid (no trades)
MID_RANGE_CHOP_LOW = 112200
MID_RANGE_CHOP_HIGH = 113900

# Risk guide levels - invalidation conditions remain similar but adjusted
# Note: Original strategy doesn't specify exact invalidation levels, keeping conservative ones

# Trade tracking
TRIGGER_STATE_FILE = "btc_intraday_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "breakout_continuation_long_triggered": False,
                "range_low_reclaim_long_triggered": False,
                "breakdown_continuation_short_triggered": False,
                "range_high_rejection_short_triggered": False,
                "last_trigger_ts": None,
                "active_trade_direction": None,
                "attempts_per_side": {"LONG": 0, "SHORT": 0},
                "last_breakout_bar_low": None,
                "last_breakdown_bar_high": None
            }
    return {
        "breakout_continuation_long_triggered": False,
        "range_low_reclaim_long_triggered": False,
        "breakdown_continuation_short_triggered": False,
        "range_high_rejection_short_triggered": False,
        "last_trigger_ts": None,
        "active_trade_direction": None,
        "attempts_per_side": {"LONG": 0, "SHORT": 0},
        "last_breakout_bar_low": None,
        "last_breakdown_bar_high": None
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

def calculate_1h_volume_sma(candles_1h, period=20):
    """
    Calculate Simple Moving Average of volume for 1-hour candles
    
    Args:
        candles_1h: List of 1-hour candle data
        period: Period for SMA calculation
    
    Returns:
        1-hour volume SMA value
    """
    if len(candles_1h) < period:
        return 0
    
    volumes = []
    for candle in candles_1h[1:period+1]:  # Use most recent period candles (skip current incomplete candle)
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
    Updated BTC Intraday Trading Setup with automated execution and fixed position size (margin 250 × leverage 20 = $5,000):
    
    Long setups:
    1) Breakout continuation: 1h close ≥ 114,000 with volume ≥ 1.25× 20-MA → Entry 114,050–114,300; Invalidation 113,400; TP1 115,000, TP2 116,000
    2) Range low reclaim: Sweep 112,000–112,100 then 15m close back above 112,200 → Entry 112,220–112,350; Invalidation 111,800; TP1 113,000, TP2 113,800–114,000
    
    Short setups:
    3) Breakdown continuation: 1h close ≤ 111,950 with volume ≥ 1.25× 20-MA → Entry 111,900–111,700; Invalidation 112,400; TP1 110,800, TP2 110,000
    4) Range high rejection: Wick above 113,900–114,000 with 5–15m bearish reversal → Entry 113,780–113,600; Invalidation 114,200; TP1 113,000, TP2 112,400–112,200
    
    Risk guide: Size for 0.6–0.9% SL, First TP ≥ +1.2–1.8% for ≥2:1 R:R. Avoid mid-range chop 112.2k–113.9k.
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== Spiros — BTC Jackson Hole Setup (LONG & SHORT enabled) ===")
    else:
        logger.info(f"=== Spiros — BTC Jackson Hole Setup ({direction} only) ===")
    
    # Load trigger state
    trigger_state = load_trigger_state()
    
    try:
        # Get current time and calculate time ranges
        current_time = datetime.now(UTC)
        
        # Get 1-hour candles for invalidation checks
        start_1h = current_time - timedelta(hours=25)  # Get 25 hours of data
        end_1h = current_time
        start_ts_1h = int(start_1h.timestamp())
        end_ts_1h = int(end_1h.timestamp())
        
        # Get 15-minute candles for main analysis (trigger timeframe)
        start_15m = current_time - timedelta(hours=8)  # Get 8 hours of 15m data (32 candles)
        end_15m = current_time
        start_ts_15m = int(start_15m.timestamp())
        end_ts_15m = int(end_15m.timestamp())
        
        logger.info(f"Fetching 1-hour candles from {start_1h} to {end_1h}")
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts_1h, end_ts_1h, GRANULARITY_1H)
        
        logger.info(f"Fetching 15-minute candles from {start_15m} to {end_15m}")
        candles_15m = safe_get_15m_candles(cb_service, PRODUCT_ID, start_ts_15m, end_ts_15m)
        
        if not candles_1h or len(candles_1h) < 3:
            logger.warning("Not enough 1-hour candle data for analysis")
            return last_alert_ts
        
        if not candles_15m or len(candles_15m) < 20:  # Need at least 4 hours of 15m data
            logger.warning("Not enough 15-minute candle data for pattern analysis")
            return last_alert_ts
        
        # Get current and previous candles
        current_15m = candles_15m[0]  # Most recent 15m candle (may be in progress)
        last_15m = candles_15m[1]     # Last completed 15m candle
        prev_15m = candles_15m[2]     # Previous completed 15m candle
        
        current_1h = candles_1h[0]    # Most recent 1h candle (may be in progress)
        last_1h = candles_1h[1]       # Last completed 1h candle
        
        # Extract values from last completed 15m candle
        last_15m_ts = datetime.fromtimestamp(int(get_candle_value(last_15m, 'start')), UTC)
        last_15m_close = float(get_candle_value(last_15m, 'close'))
        last_15m_high = float(get_candle_value(last_15m, 'high'))
        last_15m_low = float(get_candle_value(last_15m, 'low'))
        last_15m_volume = float(get_candle_value(last_15m, 'volume'))
        
        # Extract values from last completed 1h candle
        last_1h_close = float(get_candle_value(last_1h, 'close'))
        last_1h_volume = float(get_candle_value(last_1h, 'volume'))
        
        # Get current price from most recent 15m candle
        current_price = float(get_candle_value(current_15m, 'close'))
        
        # Calculate volume SMAs for volume confirmation
        volume_sma_15m = calculate_volume_sma(candles_15m, 20)
        relative_volume_15m = last_15m_volume / volume_sma_15m if volume_sma_15m > 0 else 0
        
        volume_sma_1h = calculate_1h_volume_sma(candles_1h, 20)
        relative_volume_1h = last_1h_volume / volume_sma_1h if volume_sma_1h > 0 else 0
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # Mid-range chop zone check - avoid trading in the chop zone without clear triggers
        in_chop_zone = (current_price >= MID_RANGE_CHOP_LOW and current_price <= MID_RANGE_CHOP_HIGH)
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, intraday context: BTC ~$113,296. 24h range 112,021–113,960. Funding mildly positive across majors.")
        logger.info(f"Live: BTC ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules:")
        logger.info("   • Timeframes: 1h for breakout/breakdown triggers, 15m for range reclaim, 5-15m for rejection")
        logger.info("   • Size for 0.6–0.9% SL. First TP ≥ +1.2–1.8% to keep ≥2:1 R:R")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} × {LEVERAGE}x)")
        logger.info("   • Volume filter mandatory. Favor continuation if OI expands; fade if OI contracts on the move")
        logger.info(f"   • Avoid mid-range chop {MID_RANGE_CHOP_LOW:,}k–{MID_RANGE_CHOP_HIGH:,}k without a trigger")
        logger.info("")
        
        # Show market state
        logger.info("📊 Market State:")
        logger.info(f"   • Mid-range chop zone ({MID_RANGE_CHOP_LOW:,}–{MID_RANGE_CHOP_HIGH:,}): {'🔒 YES - Use only clear triggers' if in_chop_zone else '✅ NO - Clear for setups'}")
        logger.info(f"   • Data refs: price and today's high/low from live feeds; funding snapshot positive on major venues")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1. Breakout continuation")
            logger.info(f"   • Trigger: 1h close ≥ {BREAKOUT_LONG_TRIGGER_LEVEL:,} with volume ≥ 1.25× 20-MA")
            logger.info(f"   • Entry: {BREAKOUT_LONG_ENTRY_LOW:,}–{BREAKOUT_LONG_ENTRY_HIGH:,} on retest or next 1h open")
            logger.info(f"   • Invalidation: {BREAKOUT_LONG_INVALIDATION:,} (or below breakout bar low)")
            logger.info(f"   • Targets: {BREAKOUT_LONG_TP1:,} then {BREAKOUT_LONG_TP2:,} (range extension of today's ~1.94k band)")
            logger.info("   • Notes: Prefer rising OI on break; avoid if funding spikes sharply")
            logger.info("")
            logger.info("2. Range low reclaim")
            logger.info(f"   • Trigger: Sweep of {RANGE_LOW_SWEEP_LOW:,}–{RANGE_LOW_SWEEP_HIGH:,} then 15m close back above {RANGE_LOW_RECLAIM_LEVEL:,}")
            logger.info(f"   • Entry: {RANGE_LOW_ENTRY_LOW:,}–{RANGE_LOW_ENTRY_HIGH:,}")
            logger.info(f"   • Invalidation: {RANGE_LOW_INVALIDATION:,}")
            logger.info(f"   • Targets: {RANGE_LOW_TP1:,} then {RANGE_LOW_TP2:,}–{RANGE_LOW_TP2_HIGH:,}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("1. Breakdown continuation")
            logger.info(f"   • Trigger: 1h close ≤ {BREAKDOWN_SHORT_TRIGGER_LEVEL:,} with volume ≥ 1.25× 20-MA")
            logger.info(f"   • Entry: {BREAKDOWN_SHORT_ENTRY_HIGH:,}–{BREAKDOWN_SHORT_ENTRY_LOW:,} on retest")
            logger.info(f"   • Invalidation: {BREAKDOWN_SHORT_INVALIDATION:,}")
            logger.info(f"   • Targets: {BREAKDOWN_SHORT_TP1:,} then {BREAKDOWN_SHORT_TP2:,}")
            logger.info("")
            logger.info("2. Range high rejection")
            logger.info(f"   • Trigger: Wick above {RANGE_HIGH_REJECTION_LOW:,}–{RANGE_HIGH_REJECTION_HIGH:,} with 5–15m bearish reversal and RSI cool-off")
            logger.info(f"   • Entry: {RANGE_HIGH_ENTRY_HIGH:,}–{RANGE_HIGH_ENTRY_LOW:,}")
            logger.info(f"   • Invalidation: {RANGE_HIGH_INVALIDATION:,}")
            logger.info(f"   • Targets: {RANGE_HIGH_TP1:,} then {RANGE_HIGH_TP2_HIGH:,}–{RANGE_HIGH_TP2_LOW:,}")
            logger.info("")
        
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_1h_close:,.2f}, Volume: {last_1h_volume:,.0f}, Rel_Vol_1H: {relative_volume_1h:.2f}x")
        logger.info(f"Last 15M Close: ${last_15m_close:,.2f}, High: ${last_15m_high:,.2f}, Low: ${last_15m_low:,.2f}")
        logger.info(f"15M Volume: {last_15m_volume:,.0f}, 15M SMA: {volume_sma_15m:,.0f}, Rel_Vol_15M: {relative_volume_15m:.2f}x")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check attempts per side (max 2 attempts per side)
        long_attempts = trigger_state.get("attempts_per_side", {}).get("LONG", 0)
        short_attempts = trigger_state.get("attempts_per_side", {}).get("SHORT", 0)
        
        logger.info("🔒 Trade attempts status:")
        logger.info(f"   • LONG attempts: {long_attempts}/2")
        logger.info(f"   • SHORT attempts: {short_attempts}/2")
        logger.info("")
        
        # 1) Breakout Continuation LONG - Updated Logic
        if (long_strategies_enabled and 
            not trigger_state.get("breakout_continuation_long_triggered", False) and long_attempts < 2):
            
            # Check if 1h close >= trigger level with volume confirmation
            breakout_trigger_condition = last_1h_close >= BREAKOUT_LONG_TRIGGER_LEVEL
            breakout_volume_condition = relative_volume_1h >= BREAKOUT_LONG_VOLUME_THRESHOLD_1H
            # Check if current price is in entry zone for retest or next 1h open
            breakout_entry_condition = (current_price >= BREAKOUT_LONG_ENTRY_LOW and 
                                      current_price <= BREAKOUT_LONG_ENTRY_HIGH)
            # Check we're not in mid-range chop (unless clear trigger)
            breakout_chop_ok = not in_chop_zone or breakout_trigger_condition
            breakout_ready = breakout_trigger_condition and breakout_volume_condition and breakout_entry_condition and breakout_chop_ok

            logger.info("🔍 LONG - Breakout Continuation Analysis:")
            logger.info(f"   • 1h close ≥ {BREAKOUT_LONG_TRIGGER_LEVEL:,}: {'✅' if breakout_trigger_condition else '❌'} (last 1h close: {last_1h_close:,.0f})")
            logger.info(f"   • 1h vol ≥ 1.25× 20-MA: {'✅' if breakout_volume_condition else '❌'} (rel: {relative_volume_1h:.2f}x)")
            logger.info(f"   • Entry zone {BREAKOUT_LONG_ENTRY_LOW:,}–{BREAKOUT_LONG_ENTRY_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • Chop zone check: {'✅' if breakout_chop_ok else '❌'} (in chop: {in_chop_zone}, but trigger: {breakout_trigger_condition})")
            logger.info(f"   • Breakout Continuation Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Continuation conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Breakout Continuation Long",
                    entry_price=current_price,
                    stop_loss=BREAKOUT_LONG_INVALIDATION,
                    take_profit=BREAKOUT_LONG_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakout Continuation LONG trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakout_continuation_long_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    # Save the breakout bar low for invalidation tracking
                    trigger_state["last_breakout_bar_low"] = float(get_candle_value(last_1h, 'low'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout Continuation LONG trade failed: {trade_result}")
        
        # 2) Range Low Reclaim LONG - Updated Logic
        if (long_strategies_enabled and not trade_executed and
            not trigger_state.get("range_low_reclaim_long_triggered", False) and long_attempts < 2):
            
            # Check if we've swept the range low and reclaimed above key level
            sweep_occurred = last_15m_low <= RANGE_LOW_SWEEP_HIGH and last_15m_low >= RANGE_LOW_SWEEP_LOW
            reclaim_condition = last_15m_close > RANGE_LOW_RECLAIM_LEVEL
            sweep_entry_condition = (current_price >= RANGE_LOW_ENTRY_LOW and 
                                   current_price <= RANGE_LOW_ENTRY_HIGH)
            # This setup can work in any zone since it's a clear reversal pattern
            sweep_ready = sweep_occurred and reclaim_condition and sweep_entry_condition

            logger.info("")
            logger.info("🔍 LONG - Range Low Reclaim Analysis:")
            logger.info(f"   • Sweep of {RANGE_LOW_SWEEP_LOW:,}–{RANGE_LOW_SWEEP_HIGH:,}: {'✅' if sweep_occurred else '❌'} (last low: {last_15m_low:,.0f})")
            logger.info(f"   • 15m close back above {RANGE_LOW_RECLAIM_LEVEL:,}: {'✅' if reclaim_condition else '❌'} (last close: {last_15m_close:,.0f})")
            logger.info(f"   • Entry zone {RANGE_LOW_ENTRY_LOW:,}–{RANGE_LOW_ENTRY_HIGH:,}: {'✅' if sweep_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • Range Low Reclaim Ready: {'🎯 YES' if sweep_ready else '⏳ NO'}")

            if sweep_ready:
                logger.info("")
                logger.info("🎯 LONG - Range Low Reclaim conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Range Low Reclaim Long",
                    entry_price=current_price,
                    stop_loss=RANGE_LOW_INVALIDATION,
                    take_profit=RANGE_LOW_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Range Low Reclaim LONG trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["range_low_reclaim_long_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_15m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range Low Reclaim LONG trade failed: {trade_result}")
        
        # 3) Breakdown Continuation SHORT - Updated Logic
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("breakdown_continuation_short_triggered", False) and short_attempts < 2):
            
            # Check if 1h close <= trigger level with volume confirmation
            breakdown_trigger_condition = last_1h_close <= BREAKDOWN_SHORT_TRIGGER_LEVEL
            breakdown_volume_condition = relative_volume_1h >= BREAKDOWN_SHORT_VOLUME_THRESHOLD_1H
            # Check if current price is in entry zone for retest
            breakdown_entry_condition = (current_price >= BREAKDOWN_SHORT_ENTRY_LOW and 
                                       current_price <= BREAKDOWN_SHORT_ENTRY_HIGH)
            # Check we're not in mid-range chop (unless clear trigger)
            breakdown_chop_ok = not in_chop_zone or breakdown_trigger_condition
            breakdown_ready = breakdown_trigger_condition and breakdown_volume_condition and breakdown_entry_condition and breakdown_chop_ok

            logger.info("")
            logger.info("🔍 SHORT - Breakdown Continuation Analysis:")
            logger.info(f"   • 1h close ≤ {BREAKDOWN_SHORT_TRIGGER_LEVEL:,}: {'✅' if breakdown_trigger_condition else '❌'} (last 1h close: {last_1h_close:,.0f})")
            logger.info(f"   • 1h vol ≥ 1.25× 20-MA: {'✅' if breakdown_volume_condition else '❌'} (rel: {relative_volume_1h:.2f}x)")
            logger.info(f"   • Entry zone {BREAKDOWN_SHORT_ENTRY_HIGH:,}–{BREAKDOWN_SHORT_ENTRY_LOW:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • Chop zone check: {'✅' if breakdown_chop_ok else '❌'} (in chop: {in_chop_zone}, but trigger: {breakdown_trigger_condition})")
            logger.info(f"   • Breakdown Continuation Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Continuation conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Breakdown Continuation Short",
                    entry_price=current_price,
                    stop_loss=BREAKDOWN_SHORT_INVALIDATION,
                    take_profit=BREAKDOWN_SHORT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown Continuation SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakdown_continuation_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    # Save the breakdown bar high for invalidation tracking
                    trigger_state["last_breakdown_bar_high"] = float(get_candle_value(last_1h, 'high'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown Continuation SHORT trade failed: {trade_result}")
        
        # 4) Range High Rejection SHORT - Updated Logic  
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("range_high_rejection_short_triggered", False) and short_attempts < 2):
            
            # Check for wick above range high with 5-15m bearish reversal
            wick_above_range = last_15m_high >= RANGE_HIGH_REJECTION_LOW and last_15m_high <= RANGE_HIGH_REJECTION_HIGH
            # Bearish reversal: high wicked above but closed lower (failed to sustain)
            bearish_reversal = (last_15m_high >= RANGE_HIGH_REJECTION_LOW and 
                              last_15m_close < (last_15m_high - (last_15m_high - last_15m_low) * 0.3))  # Closed in lower 70% of range
            # Check if current price is in entry zone
            rejection_entry_condition = (current_price >= RANGE_HIGH_ENTRY_LOW and 
                                       current_price <= RANGE_HIGH_ENTRY_HIGH)
            # For rejection, we want RSI cool-off - approximated by not being in extreme high
            rsi_cooloff = current_price < RANGE_HIGH_REJECTION_HIGH  # Simple proxy for RSI cool-off
            rejection_ready = wick_above_range and bearish_reversal and rejection_entry_condition and rsi_cooloff

            logger.info("")
            logger.info("🔍 SHORT - Range High Rejection Analysis:")
            logger.info(f"   • Wick above {RANGE_HIGH_REJECTION_LOW:,}–{RANGE_HIGH_REJECTION_HIGH:,}: {'✅' if wick_above_range else '❌'} (last high: {last_15m_high:,.0f})")
            logger.info(f"   • 5–15m bearish reversal: {'✅' if bearish_reversal else '❌'} (high: {last_15m_high:,.0f}, close: {last_15m_close:,.0f})")
            logger.info(f"   • Entry zone {RANGE_HIGH_ENTRY_HIGH:,}–{RANGE_HIGH_ENTRY_LOW:,}: {'✅' if rejection_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • RSI cool-off (price < {RANGE_HIGH_REJECTION_HIGH:,}): {'✅' if rsi_cooloff else '❌'}")
            logger.info(f"   • Range High Rejection Ready: {'🎯 YES' if rejection_ready else '⏳ NO'}")

            if rejection_ready:
                logger.info("")
                logger.info("🎯 SHORT - Range High Rejection conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Range High Rejection Short",
                    entry_price=current_price,
                    stop_loss=RANGE_HIGH_INVALIDATION,
                    take_profit=RANGE_HIGH_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Range High Rejection SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["range_high_rejection_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_15m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range High Rejection SHORT trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout Continuation LONG triggered: {trigger_state.get('breakout_continuation_long_triggered', False)}")
            logger.info(f"Range Low Reclaim LONG triggered: {trigger_state.get('range_low_reclaim_long_triggered', False)}")
            logger.info(f"Breakdown Continuation SHORT triggered: {trigger_state.get('breakdown_continuation_short_triggered', False)}")
            logger.info(f"Range High Rejection SHORT triggered: {trigger_state.get('range_high_rejection_short_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — BTC Intraday setup completed ===")
        return last_15m_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — BTC setups logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — BTC setups completed (with error) ===")
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
    logger.info("BTC Intraday Strategy Overview:")
    logger.info("LONG SETUPS:")
    logger.info(f"  • Breakout continuation: 1h close ≥ {BREAKOUT_LONG_TRIGGER_LEVEL:,} with volume ≥ 1.25× 20-MA → Entry {BREAKOUT_LONG_ENTRY_LOW:,}–{BREAKOUT_LONG_ENTRY_HIGH:,}; Invalidation {BREAKOUT_LONG_INVALIDATION:,}; TP1 {BREAKOUT_LONG_TP1:,}, TP2 {BREAKOUT_LONG_TP2:,}")
    logger.info(f"  • Range low reclaim: Sweep {RANGE_LOW_SWEEP_LOW:,}–{RANGE_LOW_SWEEP_HIGH:,} then 15m close back above {RANGE_LOW_RECLAIM_LEVEL:,} → Entry {RANGE_LOW_ENTRY_LOW:,}–{RANGE_LOW_ENTRY_HIGH:,}; Invalidation {RANGE_LOW_INVALIDATION:,}; TP1 {RANGE_LOW_TP1:,}, TP2 {RANGE_LOW_TP2:,}–{RANGE_LOW_TP2_HIGH:,}")
    logger.info("SHORT SETUPS:")
    logger.info(f"  • Breakdown continuation: 1h close ≤ {BREAKDOWN_SHORT_TRIGGER_LEVEL:,} with volume ≥ 1.25× 20-MA → Entry {BREAKDOWN_SHORT_ENTRY_HIGH:,}–{BREAKDOWN_SHORT_ENTRY_LOW:,}; Invalidation {BREAKDOWN_SHORT_INVALIDATION:,}; TP1 {BREAKDOWN_SHORT_TP1:,}, TP2 {BREAKDOWN_SHORT_TP2:,}")
    logger.info(f"  • Range high rejection: Wick above {RANGE_HIGH_REJECTION_LOW:,}–{RANGE_HIGH_REJECTION_HIGH:,} with 5–15m bearish reversal → Entry {RANGE_HIGH_ENTRY_HIGH:,}–{RANGE_HIGH_ENTRY_LOW:,}; Invalidation {RANGE_HIGH_INVALIDATION:,}; TP1 {RANGE_HIGH_TP1:,}, TP2 {RANGE_HIGH_TP2_HIGH:,}–{RANGE_HIGH_TP2_LOW:,}")
    logger.info(f"  • Position Size: ${MARGIN * LEVERAGE:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("  • Timeframes: 1h for breakout/breakdown triggers, 15m for range reclaim, 5-15m for rejection")
    logger.info("  • Risk guide: Size for 0.6–0.9% SL, First TP ≥ +1.2–1.8% for ≥2:1 R:R, avoid mid-range chop without triggers")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — BTC Intraday Alert Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: LONG & SHORT")
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
        logger.info(f"✅ BTC Intraday alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
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