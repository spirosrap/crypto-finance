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

# Constants for BTC intraday strategy (Aug 18, 2025 setups)
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global rules
MARGIN = 250  # USD
LEVERAGE = 20  # Always margin x leverage = 250 x 20 = $5,000 position size
RISK_PERCENTAGE = 0.5

# Session snapshot (for reporting only)
CURRENT_PRICE = 115446
TODAY_HOD = 118519
TODAY_LOD = 115008

# 1) Long — Breakout
BREAKOUT_LONG_ENTRY_LOW = 118600
BREAKOUT_LONG_ENTRY_HIGH = 118900
BREAKOUT_LONG_TRIGGER_CLOSE_1H = 118520  # 1h close must be > this
BREAKOUT_LONG_STOP_LOSS = 117650
BREAKOUT_LONG_TP1 = 120000
BREAKOUT_LONG_VOLUME_THRESHOLD_1H = 1.25  # ≥25% above 20-bar avg

# 2) Long — Failed breakdown reclaim
RECLAIM_LONG_WICK_LEVEL = 115000  # must wick below first
RECLAIM_LONG_ENTRY_LOW = 115800
RECLAIM_LONG_ENTRY_HIGH = 116100
RECLAIM_LONG_RECLAIM_CLOSE_1H = 115000  # 1h close must reclaim above this
RECLAIM_LONG_STOP_LOSS = 114700
RECLAIM_LONG_TP1 = 117200
RECLAIM_LONG_VOLUME_MIN_1H = 1.15  # ≥15–25% above 20-bar avg

# 3) Short — Range break
RANGEBREAK_SHORT_ENTRY_LOW = 114600
RANGEBREAK_SHORT_ENTRY_HIGH = 114900
RANGEBREAK_SHORT_TRIGGER_CLOSE_1H = 115000  # 1h close must be < this
RANGEBREAK_SHORT_STOP_LOSS = 115650
RANGEBREAK_SHORT_TP1 = 113500
RANGEBREAK_SHORT_VOLUME_THRESHOLD_1H = 1.25  # ≥25% above 20-bar avg

# 4) Short — Rejection at range highs
REJECTION_SHORT_ENTRY_LOW = 118000
REJECTION_SHORT_ENTRY_HIGH = 118400
REJECTION_SHORT_FAILS_TO_CLOSE_OVER = 118520  # 15–60m rejection that fails to close > this
REJECTION_SHORT_STOP_LOSS = 118900
REJECTION_SHORT_TP1 = 116500
# Volume: Normal ok; avoid if breakout volume spikes → skip if 1h rel vol ≥ breakout threshold

# Trade tracking
TRIGGER_STATE_FILE = "btc_intraday_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "breakout_long_triggered": False,
                "reclaim_long_triggered": False,
                "rangebreak_short_triggered": False,
                "rejection_short_triggered": False,
                "last_trigger_ts": None,
                "active_trade_direction": None,
                "attempts_per_side": {"LONG": 0, "SHORT": 0},
                "seen_failed_breakdown_wick": False
            }
    return {
        "breakout_long_triggered": False,
        "reclaim_long_triggered": False,
        "rangebreak_short_triggered": False,
        "rejection_short_triggered": False,
        "last_trigger_ts": None,
        "active_trade_direction": None,
        "attempts_per_side": {"LONG": 0, "SHORT": 0},
        "seen_failed_breakdown_wick": False
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
    Four BTC setups with automated execution and fixed position size (margin 250 × leverage 20 = $5,000):
    1) Long — Breakout: Entry 118,600–118,900 after a 1h close >118,520; SL 117,650; TP1 120,000; Volume ≥25% above 20-bar avg.
    2) Long — Failed breakdown reclaim: Entry 115,800–116,100 only if price wicks <115,000 then 1h close reclaims above; SL 114,700; TP1 117,200; Volume ≥15–25% above 20-bar avg.
    3) Short — Range break: Entry 114,900–114,600 after a 1h close <115,000; SL 115,650; TP1 113,500; Volume ≥25% above 20-bar avg.
    4) Short — Rejection at range highs: Entry 118,000–118,400 on 15–60m rejection that fails to close >118,520; SL 118,900; TP1 116,500; Avoid if breakout volume spikes.
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== Spiros — BTC Setups for Today (LONG & SHORT enabled) ===")
    else:
        logger.info(f"=== Spiros — BTC Setups for Today ({direction} only) ===")
    
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
        
        # Calculate volume (1h primary for gating)
        volume_sma_1h = calculate_volume_sma(candles_1h, 20)
        relative_volume_1h = last_volume / volume_sma_1h if volume_sma_1h > 0 else 0
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — BTC setups (USD)")
        logger.info(f"Live: BTC ≈ ${CURRENT_PRICE:,.0f} | Today's range: ${TODAY_LOD:,}–${TODAY_HOD:,} | Current ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules:")
        logger.info("   • Use acceptance rules for the 1h triggers (close, not wick).")
        logger.info("   • If TP1 hits, move SL to breakeven; partials at TP1 (~50–60%).")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} × {LEVERAGE}x)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG:")
            logger.info("")
            logger.info("📊 1) Long — Breakout — status: waiting")
            logger.info(f"   • Entry: ${BREAKOUT_LONG_ENTRY_LOW:,}–${BREAKOUT_LONG_ENTRY_HIGH:,} after a 1h close > ${BREAKOUT_LONG_TRIGGER_CLOSE_1H:,}")
            logger.info(f"   • SL: ${BREAKOUT_LONG_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${BREAKOUT_LONG_TP1:,}")
            logger.info(f"   • Volume: 1h ≥ 25% above 20-bar avg")
            logger.info("")
            logger.info("📊 2) Long — Failed breakdown reclaim — status: waiting")
            logger.info(f"   • Entry: ${RECLAIM_LONG_ENTRY_LOW:,}–${RECLAIM_LONG_ENTRY_HIGH:,} only if wick < ${RECLAIM_LONG_WICK_LEVEL:,} then 1h close reclaims above")
            logger.info(f"   • SL: ${RECLAIM_LONG_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${RECLAIM_LONG_TP1:,}")
            logger.info(f"   • Volume: 1h ≥ 15–25% above 20-bar avg")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT:")
            logger.info("")
            logger.info("📊 3) Short — Range break — status: waiting")
            logger.info(f"   • Entry: ${RANGEBREAK_SHORT_ENTRY_HIGH:,}–${RANGEBREAK_SHORT_ENTRY_LOW:,} after a 1h close < ${RANGEBREAK_SHORT_TRIGGER_CLOSE_1H:,}")
            logger.info(f"   • SL: ${RANGEBREAK_SHORT_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${RANGEBREAK_SHORT_TP1:,}")
            logger.info(f"   • Volume: 1h ≥ 25% above 20-bar avg")
            logger.info("")
            logger.info("📊 4) Short — Rejection at range highs — status: waiting")
            logger.info(f"   • Entry: ${REJECTION_SHORT_ENTRY_LOW:,}–${REJECTION_SHORT_ENTRY_HIGH:,} on 15–60m rejection that fails to close > ${REJECTION_SHORT_FAILS_TO_CLOSE_OVER:,}")
            logger.info(f"   • SL: ${REJECTION_SHORT_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${REJECTION_SHORT_TP1:,}")
            logger.info(f"   • Volume: normal ok; skip if breakout volume spikes")
            logger.info("")
        
        logger.info("📊 Alert text (copy/paste):")
        logger.info(f"   • 'BTC long if 1h closes > {BREAKOUT_LONG_TRIGGER_CLOSE_1H:,}; enter {BREAKOUT_LONG_ENTRY_LOW:,}–{BREAKOUT_LONG_ENTRY_HIGH:,}; SL {BREAKOUT_LONG_STOP_LOSS:,}; TP1 {BREAKOUT_LONG_TP1:,}; 1h vol ≥25% over 20-SMA.'")
        logger.info(f"   • 'BTC long failed breakdown: if wick < {RECLAIM_LONG_WICK_LEVEL:,} then 1h reclaims; enter {RECLAIM_LONG_ENTRY_LOW:,}–{RECLAIM_LONG_ENTRY_HIGH:,}; SL {RECLAIM_LONG_STOP_LOSS:,}; TP1 {RECLAIM_LONG_TP1:,}; 1h vol ≥15–25% over 20-SMA.'")
        logger.info(f"   • 'BTC short if 1h closes < {RANGEBREAK_SHORT_TRIGGER_CLOSE_1H:,}; enter {RANGEBREAK_SHORT_ENTRY_HIGH:,}–{RANGEBREAK_SHORT_ENTRY_LOW:,}; SL {RANGEBREAK_SHORT_STOP_LOSS:,}; TP1 {RANGEBREAK_SHORT_TP1:,}; 1h vol ≥25% over 20-SMA.'")
        logger.info(f"   • 'BTC short rejection: enter {REJECTION_SHORT_ENTRY_LOW:,}–{REJECTION_SHORT_ENTRY_HIGH:,} on 15–60m failure to close > {REJECTION_SHORT_FAILS_TO_CLOSE_OVER:,}; SL {REJECTION_SHORT_STOP_LOSS:,}; TP1 {REJECTION_SHORT_TP1:,}; skip if breakout vol spikes.'")
        logger.info("")
        logger.info("📊 Execution checklist (strict):")
        logger.info("   1. Use acceptance rules (multiple closes) rather than wick pokes.")
        logger.info("   2. Volume condition met.")
        logger.info("   3. If TP1 hits, move SL to breakeven; partials at TP1 (~50–60%) keep expectancy rational.")
        logger.info("   4. Skip trades if the volume conditions aren't met — no exceptions.")
        logger.info("   5. Position Size: Always margin × leverage = 250 × 20 = $5,000 USD")
        logger.info("")
        # --- Volume Confirmation Check ---
        # Calculate 15m volume SMA for more accurate volume confirmation
        volume_sma_15m = calculate_volume_sma(candles_15m, 20)
        current_15m_volume = float(get_candle_value(candles_15m[0], 'volume'))
        relative_volume_15m = current_15m_volume / volume_sma_15m if volume_sma_15m > 0 else 0
        
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_close:,.2f}, High: ${last_high:,.2f}, Low: ${last_low:,.2f}")
        logger.info(f"1H Volume: {last_volume:,.0f}, 1H SMA: {volume_sma_1h:,.0f}, Rel_Vol: {relative_volume_1h:.2f}")
        logger.info(f"15M Volume: {current_15m_volume:,.0f}, 15M SMA: {volume_sma_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}")
        logger.info("")
        
        # Volume confirmations (1h)
        volume_confirmed_breakout_long = relative_volume_1h >= BREAKOUT_LONG_VOLUME_THRESHOLD_1H
        volume_confirmed_breakdown_short = None  # deprecated path; replaced with rangebreak logic
        
        # --- Execution Guards ---
        # Check for immediate reversal candle (5–15m) against the trade
        # This will be checked in each strategy individually
        
        logger.info("")
        logger.info("🔒 Execution Guards:")
        logger.info("   • Use acceptance rules for 1h triggers; avoid wick pokes")
        logger.info("   • Respect volume gates per setup")
        logger.info("   • If TP1 hits, move SL to breakeven")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check attempts per side (max 2 attempts per side)
        long_attempts = trigger_state.get("attempts_per_side", {}).get("LONG", 0)
        short_attempts = trigger_state.get("attempts_per_side", {}).get("SHORT", 0)
        
        logger.info("")
        logger.info("🔒 Attempts per side:")
        logger.info(f"   • LONG attempts: {long_attempts}/2")
        logger.info(f"   • SHORT attempts: {short_attempts}/2")
        
        # Get current 15m candle data
        current_15m_close = float(get_candle_value(candles_15m[0], 'close'))
        current_15m_low = float(get_candle_value(candles_15m[0], 'low'))
        current_15m_high = float(get_candle_value(candles_15m[0], 'high'))
        current_15m_open = float(get_candle_value(candles_15m[0], 'open'))
        
        # Note: previous 15m close not required for these setups
        
        # 1) LONG — Breakout
        if long_strategies_enabled and not trigger_state.get("breakout_long_triggered", False) and long_attempts < 2:
            breakout_long_trigger_condition = last_close > BREAKOUT_LONG_TRIGGER_CLOSE_1H
            breakout_long_entry_condition = BREAKOUT_LONG_ENTRY_LOW <= current_price <= BREAKOUT_LONG_ENTRY_HIGH
            breakout_long_volume_condition = relative_volume_1h >= BREAKOUT_LONG_VOLUME_THRESHOLD_1H
            breakout_long_ready = breakout_long_trigger_condition and breakout_long_entry_condition and breakout_long_volume_condition

            logger.info("")
            logger.info("🔍 LONG - Breakout Analysis:")
            logger.info(f"   • 1h close > ${BREAKOUT_LONG_TRIGGER_CLOSE_1H:,}: {'✅' if breakout_long_trigger_condition else '❌'} (last close: ${last_close:,.0f})")
            logger.info(f"   • Entry zone ${BREAKOUT_LONG_ENTRY_LOW:,}–${BREAKOUT_LONG_ENTRY_HIGH:,}: {'✅' if breakout_long_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • 1h vol ≥ 1.25× 20-SMA: {'✅' if breakout_long_volume_condition else '❌'} (rel: {relative_volume_1h:.2f}x)")
            logger.info(f"   • Breakout LONG Ready: {'🎯 YES' if breakout_long_ready else '⏳ NO'}")

            if breakout_long_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setups - Breakout Long",
                    entry_price=current_price,
                    stop_loss=BREAKOUT_LONG_STOP_LOSS,
                    take_profit=BREAKOUT_LONG_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info(f"🎉 Breakout LONG trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakout_long_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout LONG trade failed: {trade_result}")
        
        # 2) LONG — Failed breakdown reclaim
        if long_strategies_enabled and not trade_executed and not trigger_state.get("reclaim_long_triggered", False) and long_attempts < 2:
            has_wicked_below = trigger_state.get("seen_failed_breakdown_wick", False) or current_15m_low < RECLAIM_LONG_WICK_LEVEL
            if current_15m_low < RECLAIM_LONG_WICK_LEVEL and not trigger_state.get("seen_failed_breakdown_wick", False):
                trigger_state["seen_failed_breakdown_wick"] = True
                save_trigger_state(trigger_state)
                logger.info(f"✅ Wick below {RECLAIM_LONG_WICK_LEVEL:,} detected (low: ${current_15m_low:,.0f})")

            reclaim_close_condition = last_close > RECLAIM_LONG_WICK_LEVEL
            reclaim_entry_condition = RECLAIM_LONG_ENTRY_LOW <= current_price <= RECLAIM_LONG_ENTRY_HIGH
            reclaim_volume_condition = relative_volume_1h >= RECLAIM_LONG_VOLUME_MIN_1H
            reclaim_long_ready = has_wicked_below and reclaim_close_condition and reclaim_entry_condition and reclaim_volume_condition

            logger.info("")
            logger.info("🔍 LONG - Failed breakdown reclaim Analysis:")
            logger.info(f"   • Wick < ${RECLAIM_LONG_WICK_LEVEL:,}: {'✅' if has_wicked_below else '❌'} (15m low: ${current_15m_low:,.0f})")
            logger.info(f"   • 1h close > ${RECLAIM_LONG_WICK_LEVEL:,}: {'✅' if reclaim_close_condition else '❌'} (last close: ${last_close:,.0f})")
            logger.info(f"   • Entry zone ${RECLAIM_LONG_ENTRY_LOW:,}–${RECLAIM_LONG_ENTRY_HIGH:,}: {'✅' if reclaim_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • 1h vol ≥ 1.15–1.25× 20-SMA: {'✅' if reclaim_volume_condition else '❌'} (rel: {relative_volume_1h:.2f}x)")
            logger.info(f"   • Reclaim LONG Ready: {'🎯 YES' if reclaim_long_ready else '⏳ NO'}")

            if reclaim_long_ready:
                logger.info("")
                logger.info("🎯 LONG - Failed breakdown reclaim conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setups - Failed Breakdown Reclaim Long",
                    entry_price=current_price,
                    stop_loss=RECLAIM_LONG_STOP_LOSS,
                    take_profit=RECLAIM_LONG_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info(f"🎉 Reclaim LONG trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["reclaim_long_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    trigger_state["seen_failed_breakdown_wick"] = False
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Reclaim LONG trade failed: {trade_result}")
        
        # 3) SHORT — Range break
        if short_strategies_enabled and not trade_executed and not trigger_state.get("rangebreak_short_triggered", False) and short_attempts < 2:
            rangebreak_trigger_condition = last_close < RANGEBREAK_SHORT_TRIGGER_CLOSE_1H
            rangebreak_entry_condition = RANGEBREAK_SHORT_ENTRY_LOW <= current_price <= RANGEBREAK_SHORT_ENTRY_HIGH or RANGEBREAK_SHORT_ENTRY_HIGH >= current_price >= RANGEBREAK_SHORT_ENTRY_LOW
            rangebreak_volume_condition = relative_volume_1h >= RANGEBREAK_SHORT_VOLUME_THRESHOLD_1H
            rangebreak_ready = rangebreak_trigger_condition and rangebreak_entry_condition and rangebreak_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Range break Analysis:")
            logger.info(f"   • 1h close < ${RANGEBREAK_SHORT_TRIGGER_CLOSE_1H:,}: {'✅' if rangebreak_trigger_condition else '❌'} (last close: ${last_close:,.0f})")
            logger.info(f"   • Entry zone ${RANGEBREAK_SHORT_ENTRY_HIGH:,}–${RANGEBREAK_SHORT_ENTRY_LOW:,}: {'✅' if rangebreak_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • 1h vol ≥ 1.25× 20-SMA: {'✅' if rangebreak_volume_condition else '❌'} (rel: {relative_volume_1h:.2f}x)")
            logger.info(f"   • Range break SHORT Ready: {'🎯 YES' if rangebreak_ready else '⏳ NO'}")

            if rangebreak_ready:
                logger.info("")
                logger.info("🎯 SHORT - Range break conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setups - Range Break Short",
                    entry_price=current_price,
                    stop_loss=RANGEBREAK_SHORT_STOP_LOSS,
                    take_profit=RANGEBREAK_SHORT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info(f"🎉 Range break SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["rangebreak_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range break SHORT trade failed: {trade_result}")

        # 4) SHORT — Rejection at range highs
        if short_strategies_enabled and not trade_executed and not trigger_state.get("rejection_short_triggered", False) and short_attempts < 2:
            # Define rejection as bearish 15m that fails to close above the threshold
            rejection_close_fail = current_15m_close <= REJECTION_SHORT_FAILS_TO_CLOSE_OVER
            bearish_15m = current_15m_close < current_15m_open
            rejection_entry_condition = REJECTION_SHORT_ENTRY_LOW <= current_price <= REJECTION_SHORT_ENTRY_HIGH
            volume_ok = relative_volume_1h < BREAKOUT_LONG_VOLUME_THRESHOLD_1H  # avoid if breakout volume spikes
            rejection_ready = rejection_close_fail and bearish_15m and rejection_entry_condition and volume_ok

            logger.info("")
            logger.info("🔍 SHORT - Rejection at range highs Analysis:")
            logger.info(f"   • 15–60m fails to close > ${REJECTION_SHORT_FAILS_TO_CLOSE_OVER:,}: {'✅' if rejection_close_fail else '❌'} (15m close: ${current_15m_close:,.0f})")
            logger.info(f"   • Bearish 15m candle: {'✅' if bearish_15m else '❌'}")
            logger.info(f"   • Entry zone ${REJECTION_SHORT_ENTRY_LOW:,}–${REJECTION_SHORT_ENTRY_HIGH:,}: {'✅' if rejection_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume spike absent (rel_1h < {BREAKOUT_LONG_VOLUME_THRESHOLD_1H:.2f}x): {'✅' if volume_ok else '❌'} (rel: {relative_volume_1h:.2f}x)")
            logger.info(f"   • Rejection SHORT Ready: {'🎯 YES' if rejection_ready else '⏳ NO'}")

            if rejection_ready:
                logger.info("")
                logger.info("🎯 SHORT - Rejection at range highs conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setups - Rejection at Highs Short",
                    entry_price=current_price,
                    stop_loss=REJECTION_SHORT_STOP_LOSS,
                    take_profit=REJECTION_SHORT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info(f"🎉 Rejection SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["rejection_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Rejection SHORT trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout LONG triggered: {trigger_state.get('breakout_long_triggered', False)}")
            logger.info(f"Reclaim LONG triggered: {trigger_state.get('reclaim_long_triggered', False)}")
            logger.info(f"Range break SHORT triggered: {trigger_state.get('rangebreak_short_triggered', False)}")
            logger.info(f"Rejection SHORT triggered: {trigger_state.get('rejection_short_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
            logger.info(f"Failed breakdown wick seen: {trigger_state.get('seen_failed_breakdown_wick', False)}")
        
        logger.info("=== Spiros — BTC setups completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — BTC setups logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — BTC setups completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BTC Setups Alert Monitor with optional direction filter')
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
    logger.info("  • Long — Breakout: Entry 118,600–118,900 after a 1h close >118,520; SL 117,650; TP1 120,000; 1h vol ≥25% over 20-SMA")
    logger.info("  • Long — Failed breakdown reclaim: Entry 115,800–116,100 only if wick <115,000 then 1h reclaims; SL 114,700; TP1 117,200; 1h vol ≥15–25% over 20-SMA")
    logger.info("  • Short — Range break: Entry 114,900–114,600 after a 1h close <115,000; SL 115,650; TP1 113,500; 1h vol ≥25% over 20-SMA")
    logger.info("  • Short — Rejection at range highs: Entry 118,000–118,400 on 15–60m rejection failing to close >118,520; SL 118,900; TP1 116,500; skip if breakout vol spikes")
    logger.info("  • Position Size: $5,000 (250 × 20x)")
    logger.info("  • Max 2 attempts per side; use acceptance rules for 1h triggers")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — BTC Setups Alert Monitor")
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