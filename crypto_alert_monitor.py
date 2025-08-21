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

# Session snapshot (for reporting only)
CURRENT_PRICE = 113871
TODAY_HOD = 114726
TODAY_LOD = 112482
YESTERDAY_HIGH = 114616
YESTERDAY_LOW = 112409

# Key levels from Jackson Hole setup
# Range today: 112,482–114,726

# 1) Breakout Continuation LONG
BREAKOUT_LONG_TRIGGER_LOW = 114700    # Entry range: 114,700–114,900
BREAKOUT_LONG_TRIGGER_HIGH = 114900
BREAKOUT_LONG_RETEST_THRESHOLD = 114800  # After 15m close >114,800 and quick retest hold
BREAKOUT_LONG_STOP_LOSS = 114200         # SL: 114,200
BREAKOUT_LONG_TP1 = 116200               # TP1: 116,200
BREAKOUT_LONG_TP2 = 116700               # TP2: 116,700 (Aug 19 swing high zone)
BREAKOUT_LONG_VOLUME_THRESHOLD_15M = 1.25  # ≥1.25× 20-bar avg

# 2) Sweep-reclaim LONG
SWEEP_RECLAIM_LONG_TRIGGER = 112409      # Dip through yesterday's low
SWEEP_RECLAIM_LONG_ENTRY_LOW = 112600    # Entry: 112,600–112,900
SWEEP_RECLAIM_LONG_ENTRY_HIGH = 112900
SWEEP_RECLAIM_LONG_STOP_LOSS = 112250    # SL: 112,250
SWEEP_RECLAIM_LONG_TP1 = 113900          # TP1: 113,900
SWEEP_RECLAIM_LONG_TP2 = 114600          # TP2: 114,600 (yesterday high)
SWEEP_RECLAIM_LONG_VOLUME_THRESHOLD_15M = 1.25  # ≥1.25× 20-bar avg

# 3) Range Breakdown SHORT
RANGE_BREAKDOWN_SHORT_TRIGGER = 112400   # 15m close <112,400
RANGE_BREAKDOWN_SHORT_ENTRY_LOW = 112300 # Entry: 112,300–112,500
RANGE_BREAKDOWN_SHORT_ENTRY_HIGH = 112500
RANGE_BREAKDOWN_SHORT_STOP_LOSS = 112900 # SL: 112,900
RANGE_BREAKDOWN_SHORT_TP1 = 111600       # TP1: 111,600
RANGE_BREAKDOWN_SHORT_TP2 = 110200       # TP2: 110,200 (measured move)
RANGE_BREAKDOWN_SHORT_VOLUME_THRESHOLD_15M = 1.25  # ≥1.25× 20-bar avg

# 4) Lower-high Fade SHORT
LOWER_HIGH_FADE_SHORT_ENTRY_LOW = 114400   # Entry: 114,400–114,600
LOWER_HIGH_FADE_SHORT_ENTRY_HIGH = 114600
LOWER_HIGH_FADE_SHORT_RESISTANCE_LOW = 114600  # 114,6–114,8k supply zone
LOWER_HIGH_FADE_SHORT_RESISTANCE_HIGH = 114800
LOWER_HIGH_FADE_SHORT_STOP_LOSS = 114900       # SL: 114,900
LOWER_HIGH_FADE_SHORT_TP1 = 113700             # TP1: 113,700
LOWER_HIGH_FADE_SHORT_TP2 = 112900             # TP2: 112,900
LOWER_HIGH_FADE_SHORT_VOLUME_THRESHOLD_15M = 1.1  # ≥1.1× 20-bar avg

# Invalidation levels
LONG_INVALIDATION_LEVEL = 112000   # Sustained 1h closes <112k invalidate longs
SHORT_INVALIDATION_LEVEL = 115000  # Sustained 1h closes >115k invalidate shorts

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
                "sweep_reclaim_long_triggered": False,
                "range_breakdown_short_triggered": False,
                "lower_high_fade_short_triggered": False,
                "last_trigger_ts": None,
                "active_trade_direction": None,
                "attempts_per_side": {"LONG": 0, "SHORT": 0},
                "longs_invalidated": False,
                "shorts_invalidated": False
            }
    return {
        "breakout_continuation_long_triggered": False,
        "sweep_reclaim_long_triggered": False,
        "range_breakdown_short_triggered": False,
        "lower_high_fade_short_triggered": False,
        "last_trigger_ts": None,
        "active_trade_direction": None,
        "attempts_per_side": {"LONG": 0, "SHORT": 0},
        "longs_invalidated": False,
        "shorts_invalidated": False
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
    BTC Jackson Hole Trading Setup with automated execution and fixed position size (margin 250 × leverage 20 = $5,000):
    
    Long setups:
    1) Breakout continuation: Entry 114,700–114,900 after 15m close >114,800 and quick retest hold; SL 114,200; TP1 116,200, TP2 116,700
    2) Sweep-reclaim: Entry 112,600–112,900 after dip through 112,409 and 15m reclaim; SL 112,250; TP1 113,900, TP2 114,600
    
    Short setups:
    3) Range breakdown: Entry 112,300–112,500 after 15m close <112,400 and failed retest; SL 112,900; TP1 111,600, TP2 110,200
    4) Lower-high fade: Entry 114,400–114,600 on 15m bearish reversal in 114,6–114,8k supply; SL 114,900; TP1 113,700, TP2 112,900
    
    Invalidations: sustained 1h closes >115k invalidate shorts; sustained 1h closes <112k invalidate longs.
    
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
        
        # Get current price from most recent 15m candle
        current_price = float(get_candle_value(current_15m, 'close'))
        
        # Calculate 15m volume SMA for volume confirmation
        volume_sma_15m = calculate_volume_sma(candles_15m, 20)
        relative_volume_15m = last_15m_volume / volume_sma_15m if volume_sma_15m > 0 else 0
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # Check invalidation conditions (1h closes)
        longs_invalidated = trigger_state.get("longs_invalidated", False) or last_1h_close < LONG_INVALIDATION_LEVEL
        shorts_invalidated = trigger_state.get("shorts_invalidated", False) or last_1h_close > SHORT_INVALIDATION_LEVEL
        
        # Update invalidation state
        trigger_state["longs_invalidated"] = longs_invalidated
        trigger_state["shorts_invalidated"] = shorts_invalidated
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, BTC ~113,871. Today's range so far: 112,482–114,726.")
        logger.info(f"Key levels: yesterday's high {YESTERDAY_HIGH:,} and low {YESTERDAY_LOW:,}.")
        logger.info("Event risk today: Jackson Hole headlines can spike volatility.")
        logger.info(f"Live: BTC ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules:")
        logger.info("   • Timeframe: trigger 15m, manage on 1h")
        logger.info("   • If TP1 hits, move SL to breakeven; partials at TP1 (~50–60%)")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} × {LEVERAGE}x)")
        logger.info("   • Volume: 15m volume ≥1.25× 20-bar avg for main setups, ≥1.1× for rejection short")
        logger.info("")
        
        # Show invalidation status
        logger.info("⚠️ Invalidations:")
        logger.info(f"   • Longs invalidated (1h closes <112k): {'✅ YES' if longs_invalidated else '❌ NO'}")
        logger.info(f"   • Shorts invalidated (1h closes >115k): {'✅ YES' if shorts_invalidated else '❌ NO'}")
        logger.info("")
        
        # Show only relevant strategies based on direction and invalidation
        if long_strategies_enabled and not longs_invalidated:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1. Breakout continuation")
            logger.info(f"   • Entry: {BREAKOUT_LONG_TRIGGER_LOW:,}–{BREAKOUT_LONG_TRIGGER_HIGH:,} after a 15m close >{BREAKOUT_LONG_RETEST_THRESHOLD:,} and quick retest hold")
            logger.info(f"   • SL: {BREAKOUT_LONG_STOP_LOSS:,}")
            logger.info(f"   • TP1: {BREAKOUT_LONG_TP1:,}  TP2: {BREAKOUT_LONG_TP2:,} (Aug 19 swing high zone)")
            logger.info(f"   • Volume: 15m volume ≥1.25× 20-bar avg")
            logger.info(f"   • Type: breakout + retest")
            logger.info("")
            logger.info("2. Sweep-reclaim of yesterday's low")
            logger.info(f"   • Entry: {SWEEP_RECLAIM_LONG_ENTRY_LOW:,}–{SWEEP_RECLAIM_LONG_ENTRY_HIGH:,} only after a dip through {SWEEP_RECLAIM_LONG_TRIGGER:,} and 15m reclaim above it")
            logger.info(f"   • SL: {SWEEP_RECLAIM_LONG_STOP_LOSS:,}")
            logger.info(f"   • TP1: {SWEEP_RECLAIM_LONG_TP1:,}  TP2: {SWEEP_RECLAIM_LONG_TP2:,} (y-day high)")
            logger.info(f"   • Volume: 15m ≥1.25× 20-bar avg")
            logger.info(f"   • Type: liquidity sweep + reclaim")
            logger.info("")
        
        if short_strategies_enabled and not shorts_invalidated:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("1. Range breakdown")
            logger.info(f"   • Entry: {RANGE_BREAKDOWN_SHORT_ENTRY_LOW:,}–{RANGE_BREAKDOWN_SHORT_ENTRY_HIGH:,} after a 15m close <{RANGE_BREAKDOWN_SHORT_TRIGGER:,} and failed retest")
            logger.info(f"   • SL: {RANGE_BREAKDOWN_SHORT_STOP_LOSS:,}")
            logger.info(f"   • TP1: {RANGE_BREAKDOWN_SHORT_TP1:,}  TP2: {RANGE_BREAKDOWN_SHORT_TP2:,} (≈ measured move of recent 2.2k range)")
            logger.info(f"   • Volume: 15m ≥1.25× 20-bar avg")
            logger.info(f"   • Type: breakdown + retest")
            logger.info("")
            logger.info("2. Lower-high fade at resistance")
            logger.info(f"   • Entry: {LOWER_HIGH_FADE_SHORT_ENTRY_LOW:,}–{LOWER_HIGH_FADE_SHORT_ENTRY_HIGH:,} on 15m bearish reversal in the {LOWER_HIGH_FADE_SHORT_RESISTANCE_LOW:,}–{LOWER_HIGH_FADE_SHORT_RESISTANCE_HIGH:,}k supply")
            logger.info(f"   • SL: {LOWER_HIGH_FADE_SHORT_STOP_LOSS:,}")
            logger.info(f"   • TP1: {LOWER_HIGH_FADE_SHORT_TP1:,}  TP2: {LOWER_HIGH_FADE_SHORT_TP2:,}")
            logger.info(f"   • Volume: 15m ≥1.1× 20-bar avg")
            logger.info(f"   • Type: rejection short")
            logger.info("")
        
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 15M Close: ${last_15m_close:,.2f}, High: ${last_15m_high:,.2f}, Low: ${last_15m_low:,.2f}")
        logger.info(f"15M Volume: {last_15m_volume:,.0f}, 15M SMA: {volume_sma_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}")
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
        
        # 1) Breakout Continuation LONG
        if (long_strategies_enabled and not longs_invalidated and 
            not trigger_state.get("breakout_continuation_long_triggered", False) and long_attempts < 2):
            
            # Check if 15m close above threshold and price in entry zone
            breakout_trigger_condition = last_15m_close > BREAKOUT_LONG_RETEST_THRESHOLD
            breakout_entry_condition = (current_price >= BREAKOUT_LONG_TRIGGER_LOW and 
                                      current_price <= BREAKOUT_LONG_TRIGGER_HIGH)
            breakout_volume_condition = relative_volume_15m >= BREAKOUT_LONG_VOLUME_THRESHOLD_15M
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and breakout_volume_condition

            logger.info("🔍 LONG - Breakout Continuation Analysis:")
            logger.info(f"   • 15m close > {BREAKOUT_LONG_RETEST_THRESHOLD:,}: {'✅' if breakout_trigger_condition else '❌'} (last close: {last_15m_close:,.0f})")
            logger.info(f"   • Entry zone {BREAKOUT_LONG_TRIGGER_LOW:,}–{BREAKOUT_LONG_TRIGGER_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • 15m vol ≥ 1.25× 20-SMA: {'✅' if breakout_volume_condition else '❌'} (rel: {relative_volume_15m:.2f}x)")
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
                    trade_type="BTC Jackson Hole - Breakout Continuation Long",
                    entry_price=current_price,
                    stop_loss=BREAKOUT_LONG_STOP_LOSS,
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
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_15m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout Continuation LONG trade failed: {trade_result}")
        
        # 2) Sweep-reclaim LONG
        if (long_strategies_enabled and not longs_invalidated and not trade_executed and
            not trigger_state.get("sweep_reclaim_long_triggered", False) and long_attempts < 2):
            
            # Check if we've dipped through yesterday's low and reclaimed above it
            sweep_occurred = last_15m_low <= SWEEP_RECLAIM_LONG_TRIGGER
            reclaim_condition = last_15m_close > SWEEP_RECLAIM_LONG_TRIGGER
            sweep_entry_condition = (current_price >= SWEEP_RECLAIM_LONG_ENTRY_LOW and 
                                   current_price <= SWEEP_RECLAIM_LONG_ENTRY_HIGH)
            sweep_volume_condition = relative_volume_15m >= SWEEP_RECLAIM_LONG_VOLUME_THRESHOLD_15M
            sweep_ready = sweep_occurred and reclaim_condition and sweep_entry_condition and sweep_volume_condition

            logger.info("")
            logger.info("🔍 LONG - Sweep-Reclaim Analysis:")
            logger.info(f"   • Dip through {SWEEP_RECLAIM_LONG_TRIGGER:,}: {'✅' if sweep_occurred else '❌'} (last low: {last_15m_low:,.0f})")
            logger.info(f"   • 15m reclaim above {SWEEP_RECLAIM_LONG_TRIGGER:,}: {'✅' if reclaim_condition else '❌'} (last close: {last_15m_close:,.0f})")
            logger.info(f"   • Entry zone {SWEEP_RECLAIM_LONG_ENTRY_LOW:,}–{SWEEP_RECLAIM_LONG_ENTRY_HIGH:,}: {'✅' if sweep_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • 15m vol ≥ 1.25× 20-SMA: {'✅' if sweep_volume_condition else '❌'} (rel: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Sweep-Reclaim Ready: {'🎯 YES' if sweep_ready else '⏳ NO'}")

            if sweep_ready:
                logger.info("")
                logger.info("🎯 LONG - Sweep-Reclaim conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Jackson Hole - Sweep-Reclaim Long",
                    entry_price=current_price,
                    stop_loss=SWEEP_RECLAIM_LONG_STOP_LOSS,
                    take_profit=SWEEP_RECLAIM_LONG_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Sweep-Reclaim LONG trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["sweep_reclaim_long_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_15m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Sweep-Reclaim LONG trade failed: {trade_result}")
        
        # 3) Range Breakdown SHORT
        if (short_strategies_enabled and not shorts_invalidated and not trade_executed and
            not trigger_state.get("range_breakdown_short_triggered", False) and short_attempts < 2):
            
            # Check if 15m close below threshold and price in entry zone
            breakdown_trigger_condition = last_15m_close < RANGE_BREAKDOWN_SHORT_TRIGGER
            breakdown_entry_condition = (current_price >= RANGE_BREAKDOWN_SHORT_ENTRY_LOW and 
                                       current_price <= RANGE_BREAKDOWN_SHORT_ENTRY_HIGH)
            breakdown_volume_condition = relative_volume_15m >= RANGE_BREAKDOWN_SHORT_VOLUME_THRESHOLD_15M
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and breakdown_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Range Breakdown Analysis:")
            logger.info(f"   • 15m close < {RANGE_BREAKDOWN_SHORT_TRIGGER:,}: {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_15m_close:,.0f})")
            logger.info(f"   • Entry zone {RANGE_BREAKDOWN_SHORT_ENTRY_LOW:,}–{RANGE_BREAKDOWN_SHORT_ENTRY_HIGH:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • 15m vol ≥ 1.25× 20-SMA: {'✅' if breakdown_volume_condition else '❌'} (rel: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Range Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Range Breakdown conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Jackson Hole - Range Breakdown Short",
                    entry_price=current_price,
                    stop_loss=RANGE_BREAKDOWN_SHORT_STOP_LOSS,
                    take_profit=RANGE_BREAKDOWN_SHORT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Range Breakdown SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["range_breakdown_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_15m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range Breakdown SHORT trade failed: {trade_result}")
        
        # 4) Lower-high Fade SHORT
        if (short_strategies_enabled and not shorts_invalidated and not trade_executed and
            not trigger_state.get("lower_high_fade_short_triggered", False) and short_attempts < 2):
            
            # Check if we're in resistance zone with bearish reversal pattern
            in_resistance_zone = (current_price >= LOWER_HIGH_FADE_SHORT_RESISTANCE_LOW and 
                                current_price <= LOWER_HIGH_FADE_SHORT_RESISTANCE_HIGH)
            in_entry_zone = (current_price >= LOWER_HIGH_FADE_SHORT_ENTRY_LOW and 
                           current_price <= LOWER_HIGH_FADE_SHORT_ENTRY_HIGH)
            # Simple bearish reversal check: 15m high above resistance but closed lower
            bearish_reversal = (last_15m_high >= LOWER_HIGH_FADE_SHORT_RESISTANCE_LOW and 
                              last_15m_close < last_15m_high)
            fade_volume_condition = relative_volume_15m >= LOWER_HIGH_FADE_SHORT_VOLUME_THRESHOLD_15M
            fade_ready = in_resistance_zone and in_entry_zone and bearish_reversal and fade_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Lower-High Fade Analysis:")
            logger.info(f"   • In resistance zone {LOWER_HIGH_FADE_SHORT_RESISTANCE_LOW:,}–{LOWER_HIGH_FADE_SHORT_RESISTANCE_HIGH:,}: {'✅' if in_resistance_zone else '❌'} (current: {current_price:,.0f})")
            logger.info(f"   • Entry zone {LOWER_HIGH_FADE_SHORT_ENTRY_LOW:,}–{LOWER_HIGH_FADE_SHORT_ENTRY_HIGH:,}: {'✅' if in_entry_zone else '❌'}")
            logger.info(f"   • 15m bearish reversal: {'✅' if bearish_reversal else '❌'} (high: {last_15m_high:,.0f}, close: {last_15m_close:,.0f})")
            logger.info(f"   • 15m vol ≥ 1.1× 20-SMA: {'✅' if fade_volume_condition else '❌'} (rel: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Lower-High Fade Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")

            if fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Lower-High Fade conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Jackson Hole - Lower-High Fade Short",
                    entry_price=current_price,
                    stop_loss=LOWER_HIGH_FADE_SHORT_STOP_LOSS,
                    take_profit=LOWER_HIGH_FADE_SHORT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Lower-High Fade SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["lower_high_fade_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_15m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Lower-High Fade SHORT trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout Continuation LONG triggered: {trigger_state.get('breakout_continuation_long_triggered', False)}")
            logger.info(f"Sweep-Reclaim LONG triggered: {trigger_state.get('sweep_reclaim_long_triggered', False)}")
            logger.info(f"Range Breakdown SHORT triggered: {trigger_state.get('range_breakdown_short_triggered', False)}")
            logger.info(f"Lower-High Fade SHORT triggered: {trigger_state.get('lower_high_fade_short_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — BTC Jackson Hole setup completed ===")
        return last_15m_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — BTC setups logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — BTC setups completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BTC Jackson Hole Alert Monitor with optional direction filter')
    parser.add_argument('--direction', choices=['LONG', 'SHORT', 'BOTH'], default='BOTH',
                       help='Trading direction to monitor: LONG, SHORT, or BOTH (default: BOTH)')
    args = parser.parse_args()
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    logger.info("Jackson Hole Strategy Overview:")
    logger.info("LONG SETUPS:")
    logger.info(f"  • Breakout continuation: Entry {BREAKOUT_LONG_TRIGGER_LOW:,}–{BREAKOUT_LONG_TRIGGER_HIGH:,} after 15m close >{BREAKOUT_LONG_RETEST_THRESHOLD:,}; SL {BREAKOUT_LONG_STOP_LOSS:,}; TP1 {BREAKOUT_LONG_TP1:,}, TP2 {BREAKOUT_LONG_TP2:,}")
    logger.info(f"  • Sweep-reclaim: Entry {SWEEP_RECLAIM_LONG_ENTRY_LOW:,}–{SWEEP_RECLAIM_LONG_ENTRY_HIGH:,} after dip through {SWEEP_RECLAIM_LONG_TRIGGER:,} and reclaim; SL {SWEEP_RECLAIM_LONG_STOP_LOSS:,}; TP1 {SWEEP_RECLAIM_LONG_TP1:,}, TP2 {SWEEP_RECLAIM_LONG_TP2:,}")
    logger.info("SHORT SETUPS:")
    logger.info(f"  • Range breakdown: Entry {RANGE_BREAKDOWN_SHORT_ENTRY_LOW:,}–{RANGE_BREAKDOWN_SHORT_ENTRY_HIGH:,} after 15m close <{RANGE_BREAKDOWN_SHORT_TRIGGER:,}; SL {RANGE_BREAKDOWN_SHORT_STOP_LOSS:,}; TP1 {RANGE_BREAKDOWN_SHORT_TP1:,}, TP2 {RANGE_BREAKDOWN_SHORT_TP2:,}")
    logger.info(f"  • Lower-high fade: Entry {LOWER_HIGH_FADE_SHORT_ENTRY_LOW:,}–{LOWER_HIGH_FADE_SHORT_ENTRY_HIGH:,} on bearish reversal; SL {LOWER_HIGH_FADE_SHORT_STOP_LOSS:,}; TP1 {LOWER_HIGH_FADE_SHORT_TP1:,}, TP2 {LOWER_HIGH_FADE_SHORT_TP2:,}")
    logger.info(f"  • Position Size: ${MARGIN * LEVERAGE:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("  • Timeframe: trigger 15m, manage on 1h")
    logger.info("  • Invalidations: sustained 1h closes >115k invalidate shorts; sustained 1h closes <112k invalidate longs")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — BTC Jackson Hole Alert Monitor")
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
        logger.info(f"✅ Jackson Hole alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
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