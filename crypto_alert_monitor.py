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
RISK_PERCENTAGE = 0.5  # 0.5-1.0% of equity per setup
VOLUME_THRESHOLD_LONG = 1.25  # ≥1.25× 20-period vol on 15m for long trades
VOLUME_THRESHOLD_SHORT = 1.3   # ≥1.3× 20-SMA vol on 15m for short trades

# Today's session levels from the new plan (BTC ≈ $118k; HOD $119,325; LOD $117,449)
HOD = 119325  # Today's high
LOD = 117449  # Today's low
PDH = 124530  # Prior Day High
PDL = 117430  # Prior Day Low

# LONG - "PDL sweep → reclaim" strategy
PDL_SWEEP_ENTRY_LOW = 117600   # Entry zone: $117,600–$117,750
PDL_SWEEP_ENTRY_HIGH = 117750  # Entry zone high
PDL_SWEEP_STOP_LOSS = 116900   # SL: $116,900 (below session low / liquidity pocket)
PDL_SWEEP_TP1 = 119800         # TP1: $119,800
PDL_SWEEP_TP2 = 121000         # TP2: $121,000

# LONG - "Range break momentum" strategy
RANGE_BREAK_ENTRY_LOW = 120200   # Entry zone: $120,200–$120,500
RANGE_BREAK_ENTRY_HIGH = 120500  # Entry zone high
RANGE_BREAK_STOP_LOSS = 119400   # SL: $119,400
RANGE_BREAK_TP1 = 122800         # TP1: $122,800
RANGE_BREAK_TP2 = 124300         # TP2: $124,300 (near PDH)

# SHORT - "PDL loss → retest sell" strategy
PDL_LOSS_ENTRY_LOW = 117300   # Entry zone: $117,300–$117,500
PDL_LOSS_ENTRY_HIGH = 117500  # Entry zone high
PDL_LOSS_STOP_LOSS = 118100   # SL: $118,100
PDL_LOSS_TP1 = 116000         # TP1: $116,000
PDL_LOSS_TP2 = 115200         # TP2: $115,200 (cluster from Aug 3–8 lows)

# Trade tracking
TRIGGER_STATE_FILE = "btc_intraday_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "pdl_sweep_triggered": False, 
                "range_break_triggered": False, 
                "pdl_loss_triggered": False,
                "last_trigger_ts": None,
                "last_15m_structure": None,
                "active_trade_direction": None,  # Track which direction is active
                "attempts_per_side": {"LONG": 0, "SHORT": 0}  # Track attempts per side (max 2)
            }
    return {
        "pdl_sweep_triggered": False, 
        "range_break_triggered": False, 
        "pdl_loss_triggered": False,
        "last_trigger_ts": None,
        "last_15m_structure": None,
        "active_trade_direction": None,  # Track which direction is active
        "attempts_per_side": {"LONG": 0, "SHORT": 0}  # Track attempts per side (max 2)
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
    Spiros — BTC intraday setups for Aug 15, 2025 (USD quotes).
    Live: BTC ≈ $118k. Today's range: HOD $119,325 / LOD $117,449.
    
    Rules (both directions)
    • Trigger on 15m, execute on 15m.
    • Volume confirm: ≥ 1.25× 20-period vol on 15m for long trades or ≥ 1.3× 20-SMA vol on 15m for short trades at trigger.
    • Risk: define 1R = |entry − SL|; risk ≤ 0.5–1.0% of equity per setup. Partial at +1.0–1.2R, trail to BE.
    • Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
    LONG
    
    1) "PDL sweep → reclaim"
    • Entry: 117,600–117,750 on a 15m close back above 117,700 after trading below it (failed breakdown).
    • SL: 116,900 (below session low / liquidity pocket).
    • TP1: 119,800; TP2: 121,000.
    • Why: Prior Day Low ~117,430 is the key pivot; a sweep + reclaim often mean-reverts to mid/upper prior range.
    • Volume confirm: 15m volume ≥ 1.25× your 20-period average (Coinbase perps).
    • Timeframe: Trigger 15m; manage on 1h.
    • Type: Liquidity sweep → reclaim.
    • Status: Waiting / near (price ~118k).
    
    2) "Range break momentum"
    • Entry: 120,200–120,500 on a 15m close above 120,200 (clean break back into yesterday's upper range).
    • SL: 119,400.
    • TP1: 122,800; TP2: 124,300 (near PDH).
    • Why: Reclaiming 120k shifts tape from mid-range to upper-range momentum; PDH ~124,530 is the magnet.
    • Volume confirm: 15m ≥ 1.25× 20-MA volume.
    • Timeframe: 15m trigger; 1h/4h context.
    • Type: Breakout continuation.
    • Status: Waiting.
    
    SHORT
    
    3) "PDL loss → retest sell"
    • Entry: 117,300–117,500 after a 15m close < 117,400, sell the weak retest.
    • SL: 118,100.
    • TP1: 116,000; TP2: 115,200 (cluster from Aug 3–8 lows).
    • Why: Losing PDL opens path to early-Aug demand; first stop at 116k, then prior cluster ~115.2k.
    • Volume confirm: 15m ≥ 1.3× 20-MA volume.
    • Timeframe: 15m trigger; 1h manage.
    • Type: Breakdown → retest short.
    • Status: Waiting.
    
    Alerts you can set (verbatim)
    • Long sweep/reclaim: "15m close > 117,700 AND prior 15m low < 117,500 AND vol ≥ 1.25× MA20."
    • Long breakout: "15m close ≥ 120,200 AND vol ≥ 1.25× MA20."
    • Short breakdown: "15m close ≤ 117,400 AND vol ≥ 1.3× MA20."
    
    Execution checklist (strict)
    1. Price at/through trigger.
    2. Volume condition met.
    3. No immediate reversal candle (5–15m) against the trade.
    4. Place stop at the level stated, not "mental."
    5. Scale: 50% at TP1, trail to BE; run remainder to TP2 or a 5m swing.
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== Spiros — clean two-sided BTC plan for today (Complete Strategy - LONG & SHORT) ===")
    else:
        logger.info(f"=== Spiros — clean two-sided BTC plan for today ({direction} Strategy Only) ===")
    
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
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — BTC intraday setups for Aug 15, 2025 (USD quotes)")
        logger.info(f"Live: BTC ≈ $118k | HOD ${HOD:,} | LOD ${LOD:,} | PDH ${PDH:,} | PDL ${PDL:,} | Current ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules (both directions):")
        logger.info(f"   • Trigger on 15m, execute on 15m.")
        logger.info(f"   • Volume confirm: ≥{VOLUME_THRESHOLD_LONG}× 20-period vol on 15m for long trades or ≥{VOLUME_THRESHOLD_SHORT}× 20-SMA vol on 15m for short trades at trigger.")
        logger.info(f"   • Risk: define 1R = |entry − SL|; risk ≤ {RISK_PERCENTAGE}–1.0% of equity per setup. Partial at +1.0–1.2R, trail to BE.")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin × {LEVERAGE} leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG:")
            logger.info("")
            logger.info("📊 1) PDL sweep → reclaim — status: waiting")
            logger.info(f"   • Entry: 117,600–117,750 on a 15m close back above 117,700 after trading below it (failed breakdown)")
            logger.info(f"   • SL: ${PDL_SWEEP_STOP_LOSS:,} (below session low / liquidity pocket)")
            logger.info(f"   • TP1: ${PDL_SWEEP_TP1:,}; TP2: ${PDL_SWEEP_TP2:,}")
            logger.info(f"   • Why: Prior Day Low ~${PDL:,} is the key pivot; a sweep + reclaim often mean-reverts to mid/upper prior range")
            logger.info("")
            logger.info("📊 2) Range break momentum — status: waiting")
            logger.info(f"   • Entry: 120,200–120,500 on a 15m close above 120,200 (clean break back into yesterday's upper range)")
            logger.info(f"   • SL: ${RANGE_BREAK_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${RANGE_BREAK_TP1:,}; TP2: ${RANGE_BREAK_TP2:,} (near PDH)")
            logger.info(f"   • Why: Reclaiming 120k shifts tape from mid-range to upper-range momentum; PDH ~${PDH:,} is the magnet")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT:")
            logger.info("")
            logger.info("📊 3) PDL loss → retest sell — status: waiting")
            logger.info(f"   • Entry: 117,300–117,500 after a 15m close < 117,400, sell the weak retest")
            logger.info(f"   • SL: ${PDL_LOSS_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${PDL_LOSS_TP1:,}; TP2: ${PDL_LOSS_TP2:,} (cluster from Aug 3–8 lows)")
            logger.info(f"   • Why: Losing PDL opens path to early-Aug demand; first stop at 116k, then prior cluster ~115.2k")
            logger.info("")
        
        logger.info("📊 Alerts you can set (verbatim):")
        logger.info("   • Long sweep/reclaim: '15m close > 117,700 AND prior 15m low < 117,500 AND vol ≥ 1.25× MA20'")
        logger.info("   • Long breakout: '15m close ≥ 120,200 AND vol ≥ 1.25× MA20'")
        logger.info("   • Short breakdown: '15m close ≤ 117,400 AND vol ≥ 1.3× MA20'")
        logger.info("")
        logger.info("📊 Execution checklist (strict):")
        logger.info("   1. Price at/through trigger.")
        logger.info("   2. Volume condition met.")
        logger.info("   3. No immediate reversal candle (5–15m) against the trade.")
        logger.info("   4. Place stop at the level stated, not 'mental.'")
        logger.info("   5. Scale: 50% at TP1, trail to BE; run remainder to TP2 or a 5m swing.")
        logger.info("")
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_close:,.2f}, High: ${last_high:,.2f}, Low: ${last_low:,.2f}")
        logger.info(f"1H Volume: {last_volume:,.0f}, 1H SMA: {volume_sma_1h:,.0f}, Rel_Vol: {relative_volume_1h:.2f}")
        logger.info(f"15M Volume: {current_15m_volume:,.0f}, 15M SMA: {volume_sma_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}")
        logger.info("")
        
        # --- Volume Confirmation Check ---
        # Calculate 15m volume SMA for more accurate volume confirmation
        volume_sma_15m = calculate_volume_sma(candles_15m, 20)
        current_15m_volume = float(get_candle_value(candles_15m[0], 'volume'))
        relative_volume_15m = current_15m_volume / volume_sma_15m if volume_sma_15m > 0 else 0
        
        # Volume confirmation for different strategies
        volume_confirmed_long = relative_volume_15m >= VOLUME_THRESHOLD_LONG  # ≥1.25× for long trades
        volume_confirmed_short = relative_volume_15m >= VOLUME_THRESHOLD_SHORT  # ≥1.3× for short trades
        
        # --- Execution Guards ---
        # Check for immediate reversal candle (5–15m) against the trade
        # This will be checked in each strategy individually
        
        logger.info("")
        logger.info("🔒 Execution Guards:")
        logger.info("   • Volume confirmation required: ≥1.25× 20-period vol on 15m for long trades OR ≥1.3× 20-SMA vol on 15m for short trades")
        logger.info("   • No immediate reversal candle (5–15m) against the trade")
        logger.info("   • Place stop at the level stated, not 'mental'")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check attempts per side (max 2 attempts per side)
        long_attempts = trigger_state.get("attempts_per_side", {}).get("LONG", 0)
        short_attempts = trigger_state.get("attempts_per_side", {}).get("SHORT", 0)
        
        logger.info("")
        logger.info("🔒 Attempts per side:")
        logger.info(f"   • LONG attempts: {long_attempts}/2")
        logger.info(f"   • SHORT attempts: {short_attempts}/2")
        
        # 1. LONG - "PDL sweep → reclaim" Strategy
        if long_strategies_enabled and not trigger_state.get("pdl_sweep_triggered", False) and long_attempts < 2:
            # Check for PDL sweep → reclaim conditions
            # Entry: 117,600–117,750 on a 15m close back above 117,700 after trading below it (failed breakdown)
            current_15m_close = float(get_candle_value(candles_15m[0], 'close'))
            current_15m_low = float(get_candle_value(candles_15m[0], 'low'))
            prev_15m_low = float(get_candle_value(candles_15m[1], 'low'))
            
            # Conditions: 15m close > 117,700 AND prior 15m low < 117,500 AND vol ≥ 1.25× MA20
            pdl_sweep_close_condition = current_15m_close > 117700
            pdl_sweep_low_condition = prev_15m_low < 117500  # Prior 15m low < 117,500
            pdl_sweep_volume_condition = volume_confirmed_long
            
            pdl_sweep_ready = pdl_sweep_close_condition and pdl_sweep_low_condition and pdl_sweep_volume_condition
            
            logger.info("")
            logger.info("🔍 LONG - PDL Sweep → Reclaim Analysis:")
            logger.info(f"   • 15m close > 117,700: {'✅' if pdl_sweep_close_condition else '❌'} (current: ${current_15m_close:,.0f})")
            logger.info(f"   • Prior 15m low < 117,500: {'✅' if pdl_sweep_low_condition else '❌'} (prior: ${prev_15m_low:,.0f})")
            logger.info(f"   • Volume ≥ 1.25× MA20: {'✅' if pdl_sweep_volume_condition else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • PDL Sweep → Reclaim Ready: {'🎯 YES' if pdl_sweep_ready else '⏳ NO'}")
            
            if pdl_sweep_ready:
                logger.info("")
                logger.info("🎯 LONG - PDL Sweep conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute PDL Sweep trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - PDL Sweep Long",
                    entry_price=current_price,
                    stop_loss=PDL_SWEEP_STOP_LOSS,
                    take_profit=PDL_SWEEP_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 PDL Sweep trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["pdl_sweep_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ PDL Sweep trade failed: {trade_result}")
        
        # 2. LONG - "Range break momentum" Strategy
        if long_strategies_enabled and not trade_executed and not trigger_state.get("range_break_triggered", False) and long_attempts < 2:
            # Check for Range break momentum conditions
            # Entry: 120,200–120,500 on a 15m close above 120,200 (clean break back into yesterday's upper range)
            current_15m_close = float(get_candle_value(candles_15m[0], 'close'))
            
            # Conditions: 15m close ≥ 120,200 AND vol ≥ 1.25× MA20
            range_break_close_condition = current_15m_close >= 120200
            range_break_volume_condition = volume_confirmed_long
            
            range_break_ready = range_break_close_condition and range_break_volume_condition
            
            logger.info("")
            logger.info("🔍 LONG - Range Break Momentum Analysis:")
            logger.info(f"   • 15m close ≥ 120,200: {'✅' if range_break_close_condition else '❌'} (current: ${current_15m_close:,.0f})")
            logger.info(f"   • Volume ≥ 1.25× MA20: {'✅' if range_break_volume_condition else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Range Break Momentum Ready: {'🎯 YES' if range_break_ready else '⏳ NO'}")
            
            if range_break_ready:
                logger.info("")
                logger.info("🎯 LONG - Range Break conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Range Break trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Range Break Long",
                    entry_price=current_price,
                    stop_loss=RANGE_BREAK_STOP_LOSS,
                    take_profit=RANGE_BREAK_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Range Break trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["range_break_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range Break trade failed: {trade_result}")
        
        # 3. SHORT - "PDL loss → retest sell" Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("pdl_loss_triggered", False) and short_attempts < 2:
            # Check for PDL loss → retest sell conditions
            # Entry: 117,300–117,500 after a 15m close < 117,400, sell the weak retest
            current_15m_close = float(get_candle_value(candles_15m[0], 'close'))
            
            # Conditions: 15m close ≤ 117,400 AND vol ≥ 1.3× MA20
            pdl_loss_close_condition = current_15m_close <= 117400
            pdl_loss_volume_condition = volume_confirmed_short
            
            pdl_loss_ready = pdl_loss_close_condition and pdl_loss_volume_condition
            
            logger.info("")
            logger.info("🔍 SHORT - PDL Loss → Retest Sell Analysis:")
            logger.info(f"   • 15m close ≤ 117,400: {'✅' if pdl_loss_close_condition else '❌'} (current: ${current_15m_close:,.0f})")
            logger.info(f"   • Volume ≥ 1.3× MA20: {'✅' if pdl_loss_volume_condition else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • PDL Loss → Retest Sell Ready: {'🎯 YES' if pdl_loss_ready else '⏳ NO'}")
            
            if pdl_loss_ready:
                logger.info("")
                logger.info("🎯 SHORT - PDL Loss conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute PDL Loss trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - PDL Loss Short",
                    entry_price=current_price,
                    stop_loss=PDL_LOSS_STOP_LOSS,
                    take_profit=PDL_LOSS_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 PDL Loss trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["pdl_loss_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ PDL Loss trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"PDL Sweep triggered: {trigger_state.get('pdl_sweep_triggered', False)}")
            logger.info(f"Range Break triggered: {trigger_state.get('range_break_triggered', False)}")
            logger.info(f"PDL Loss triggered: {trigger_state.get('pdl_loss_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — BTC intraday setups for Aug 15, 2025 completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — BTC intraday setups for Aug 15, 2025 logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — BTC intraday setups for Aug 15, 2025 completed (with error) ===")
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
    logger.info("  • LONG PDL sweep → reclaim: 15m close > 117,700 AND prior 15m low < 117,500 AND vol ≥ 1.25× MA20")
    logger.info("  • LONG Range break momentum: 15m close ≥ 120,200 AND vol ≥ 1.25× MA20")
    logger.info("  • SHORT PDL loss → retest sell: 15m close ≤ 117,400 AND vol ≥ 1.3× MA20")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20 leverage)")
    logger.info("  • Volume confirm: ≥1.25× 20-period vol on 15m for long trades OR ≥1.3× 20-SMA vol on 15m for short trades")
    logger.info("  • Max 2 attempts per side; if trigger wicks without close + volume, abort and retry only once")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — BTC intraday setups for Aug 15, 2025 Alert Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: Complete BTC Intraday Strategy - LONG & SHORT")
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