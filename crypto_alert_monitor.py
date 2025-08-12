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
RISK_PERCENTAGE = 0.7  # 0.6-0.8% of price as 1R
VOLUME_THRESHOLD_1H = 1.25  # 1.25x 20-period vol on 1h
VOLUME_THRESHOLD_5M = 2.0   # 2x 20-SMA vol on 5m

# Today's session levels from the new plan (BTC ≈ 118,915 | HOD 122,242 | LOD 118,355)
HOD = 122242  # 24h high
LOD = 118355  # 24h low
MID = 120298  # Mid point of 24h range

# LONG - Breakout continuation strategy
BREAKOUT_ENTRY_LOW = 122400   # Entry: 122,400–122,650 (clearance above HOD)
BREAKOUT_ENTRY_HIGH = 122650  # Entry zone high
BREAKOUT_STOP_LOSS = 121600   # SL: 121,600 (back inside range)
BREAKOUT_TP1 = 123500         # TP1: 123,350–123,500 (near ATH supply)
BREAKOUT_TP2 = 124800         # TP2: 124,600–124,800
BREAKOUT_RUNNER = 125800      # Runner: 125,800

# LONG - Failed-breakdown → reclaim strategy
RECLAIM_ENTRY_LOW = 118400    # Entry: After a 5m close back above 118,400–118,500
RECLAIM_ENTRY_HIGH = 118500   # Entry zone high
RECLAIM_STOP_LOSS = 117650    # SL: 117,650
RECLAIM_TP1 = 119300          # TP1: 119,300
RECLAIM_TP2 = 120300          # TP2: 120,300 (≈ range mid)

# SHORT - Breakdown continuation strategy
BREAKDOWN_ENTRY_LOW = 118100   # Entry: 118,200–118,100 (clean break of LOD)
BREAKDOWN_ENTRY_HIGH = 118200  # Entry zone high
BREAKDOWN_STOP_LOSS = 118950   # SL: 118,950
BREAKDOWN_TP1 = 117250         # TP1: 117,250
BREAKDOWN_TP2 = 116400         # TP2: 116,400

# SHORT - Failed-breakout → rejection strategy
REJECTION_ENTRY_LOW = 121850   # Entry: 121,850–121,950 on 5m close back inside the prior range
REJECTION_ENTRY_HIGH = 121950  # Entry zone high
REJECTION_STOP_LOSS = 122600   # SL: 122,600
REJECTION_TP1 = 120800         # TP1: 120,800
REJECTION_TP2 = 119800         # TP2: 119,800
REJECTION_RUNNER = 118600      # Runner: 118,600

# No-trade filter (chop avoidance)
CHOP_BAND_LOW = 119600         # Stand down while price is stuck 119,600–120,900
CHOP_BAND_HIGH = 120900
CHOP_VOLUME_THRESHOLD = 1.5    # and volume < thresholds

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
                "rejection_triggered": False,
                "last_trigger_ts": None,
                "last_1h_structure": None,
                "active_trade_direction": None  # Track which direction is active
            }
    return {
        "breakout_triggered": False, 
        "reclaim_triggered": False, 
        "breakdown_triggered": False,
        "rejection_triggered": False,
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

def detect_sweep_and_reclaim(candles_5m, candles_15m, sweep_low=120000, sweep_high=120000, reclaim_level=120000):
    """
    Detect if there was a sweep of the $120,000 level followed by a reclaim with higher low
    
    Args:
        candles_5m: List of 5-minute candles
        candles_15m: List of 15-minute candles
        sweep_low: Lower bound of sweep range (default: 120000)
        sweep_high: Upper bound of sweep range (default: 120000)
        reclaim_level: Level to reclaim above (default: 120000)
    
    Returns:
        Tuple of (sweep_detected, reclaim_detected)
    """
    if not candles_5m or len(candles_5m) < 12:  # Need at least 1 hour of 5m data
        return False, False
    
    if not candles_15m or len(candles_15m) < 4:  # Need at least 1 hour of 15m data
        return False, False
    
    # Check for sweep of $120,000 in recent 5m candles (last 12 candles = 1 hour)
    sweep_detected = False
    for i, candle in enumerate(candles_5m[1:13]):  # Skip current incomplete candle
        low = float(get_candle_value(candle, 'low'))
        if low <= sweep_low:
            sweep_detected = True
            break
    
    # Check for reclaim above $120,000 with higher low in recent 15m candles (last 4 candles = 1 hour)
    reclaim_detected = False
    if sweep_detected:
        # Find the lowest low after the sweep
        lowest_after_sweep = float('inf')
        for candle in candles_15m[1:5]:  # Skip current incomplete candle
            low = float(get_candle_value(candle, 'low'))
            if low < lowest_after_sweep:
                lowest_after_sweep = low
        
        # Check if we have a higher low and close above $120,000
        for i, candle in enumerate(candles_15m[1:5]):  # Skip current incomplete candle
            close = float(get_candle_value(candle, 'close'))
            low = float(get_candle_value(candle, 'low'))
            if close > reclaim_level and low > lowest_after_sweep:
                reclaim_detected = True
                break
    
    return sweep_detected, reclaim_detected

def detect_rejection_wick(candles_5m, candles_15m, entry_low=123000, entry_high=123500):
    """
    Detect if there was a rejection wick in the 123,000–123,500 zone with 5–15m wick + momentum stall
    
    Args:
        candles_5m: List of 5-minute candles
        candles_15m: List of 15-minute candles
        entry_low: Lower bound of entry range (default: 123000)
        entry_high: Upper bound of entry range (default: 123500)
    
    Returns:
        Boolean indicating if rejection wick was detected
    """
    if not candles_5m or len(candles_5m) < 12:  # Need at least 1 hour of 5m data
        return False
    
    if not candles_15m or len(candles_15m) < 4:  # Need at least 1 hour of 15m data
        return False
    
    # Check for rejection wick in recent 5m candles (last 12 candles = 1 hour)
    for i, candle in enumerate(candles_5m[1:13]):  # Skip current incomplete candle
        high = float(get_candle_value(candle, 'high'))
        low = float(get_candle_value(candle, 'low'))
        close = float(get_candle_value(candle, 'close'))
        volume = float(get_candle_value(candle, 'volume'))
        
        # Check if high touched the entry zone
        if entry_low <= high <= entry_high:
            # Check for rejection wick (high close to high, close below high)
            wick_size = high - max(close, low)
            body_size = abs(close - low)
            
            # Rejection wick if wick is significant compared to body and close back inside prior bar
            if wick_size > body_size * 0.5 and close < entry_high:
                # Check for momentum stall (volume spike)
                if i < len(candles_5m) - 2:
                    prev_volume = float(get_candle_value(candles_5m[i+2], 'volume'))
                    if volume > prev_volume * 1.2:  # 20% higher volume
                        return True
    
    # Also check 15m candles
    for i, candle in enumerate(candles_15m[1:5]):  # Skip current incomplete candle
        high = float(get_candle_value(candle, 'high'))
        low = float(get_candle_value(candle, 'low'))
        close = float(get_candle_value(candle, 'close'))
        
        # Check if high touched the entry zone
        if entry_low <= high <= entry_high:
            # Check for rejection wick and close back inside prior bar
            wick_size = high - max(close, low)
            body_size = abs(close - low)
            
            # Rejection wick if wick is significant compared to body and close back inside prior bar
            if wick_size > body_size * 0.5 and close < entry_high:
                return True
    
    return False



def btc_intraday_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    Spiros — here's a clean, two-sided BTC plan for today (UTC session, Aug 12, 2025) using Coinbase spot levels: BTC ≈ 118,915 | HOD 122,242 | LOD 118,355.
    
    Rules (both directions):
    - Trigger on 1h; execute on 5–15m with market brackets only.
    - Volume confirm: ≥1.25× 20-period vol on 1h OR ≥2× 20-SMA vol on 5m at trigger.
    - Risk: size so 1R ≈ 0.6–0.8% of price; partial at +1.0–1.5R; trail remainder.
    - Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
    LONGS:
    1) LONG (breakout continuation) — status: waiting
    - Entry (buy-stop): 122,400–122,650 (clearance above HOD).
    - SL (invalidation): 121,600 (back inside range).
    - TP1 / TP2 / runner: 123,350–123,500 (near ATH supply), 124,600–124,800, runner 125,800.
    - Why: Continuation above today's range; watch for supply near ATH ≈ 123,231.
    
    2) LONG (failed-breakdown → reclaim) — status: waiting
    - Setup: Sweep sub-LOD then reclaim.
    - Entry: After a 5m close back above 118,400–118,500.
    - SL: 117,650.
    - TP1 / TP2: 119,300 / 120,300 (≈ range mid).
    - Why: Liquidity grab below LOD, mean-revert toward mid-range (~120,298).
    
    SHORTS:
    3) SHORT (breakdown continuation) — status: waiting
    - Entry (sell-stop): 118,200–118,100 (clean break of LOD).
    - SL: 118,950.
    - TP1 / TP2: 117,250 / 116,400.
    - Why: Range expansion lower after taking out LOD; momentum continuation.
    
    4) SHORT (failed-breakout → rejection) — status: waiting
    - Setup: Spike above 122,2xx that fails.
    - Entry: 121,850–121,950 on 5m close back inside the prior range.
    - SL: 122,600.
    - TP1 / TP2 / runner: 120,800 / 119,800 / 118,600.
    - Why: Trap above HOD; fade back through mid into lower third.
    
    No-trade filter (chop avoidance):
    - Stand down while price is stuck 119,600–120,900 and volume < thresholds.
    
    Bot-ready trigger logic (concise):
    - Breakout long: if last_price >= 122400 AND volume rule true → market buy; SL 121600; TP1 123500; TP2 124800; Runner 125800.
    - Reclaim long: if 5m close back above 118400 after sweep sub-LOD AND volume rule → market buy; SL 117650; TP1 119300; TP2 120300.
    - Breakdown short: if last_price <= 118100 AND volume rule → market sell; SL 118950; TP1 117250; TP2 116400.
    - Rejection short: if 5m close back inside prior range after spike above 122,2xx AND volume rule → market sell; SL 122600; TP1 120800; TP2 119800; Runner 118600.
    
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
        logger.info("🚀 Spiros — here's a clean, two-sided BTC plan for today (UTC session, Aug 12, 2025)")
        logger.info(f"BTC ≈ 118,915 | HOD ${HOD:,} | LOD ${LOD:,} | Current ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules (both directions):")
        logger.info(f"   • Trigger on 1h; execute on 5–15m with market brackets only.")
        logger.info(f"   • Volume confirm: ≥{VOLUME_THRESHOLD_1H}× 20-period vol on 1h OR ≥{VOLUME_THRESHOLD_5M}× 20-SMA vol on 5m at trigger.")
        logger.info(f"   • Risk: size so 1R ≈ {RISK_PERCENTAGE}–0.8% of price; partial at +1.0–1.5R; trail remainder.")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin × {LEVERAGE} leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONGS:")
            logger.info("")
            logger.info("📊 1) LONG (breakout continuation) — status: waiting")
            logger.info(f"   • Entry (buy-stop): ${BREAKOUT_ENTRY_LOW:,}–${BREAKOUT_ENTRY_HIGH:,} (clearance above HOD)")
            logger.info(f"   • SL (invalidation): ${BREAKOUT_STOP_LOSS:,} (back inside range)")
            logger.info(f"   • TP1 / TP2 / runner: ${BREAKOUT_TP1:,}–${BREAKOUT_TP1:,} (near ATH supply), ${BREAKOUT_TP2:,}–${BREAKOUT_TP2:,}, runner ${BREAKOUT_RUNNER:,}")
            logger.info(f"   • Why: Continuation above today's range; watch for supply near ATH ≈ 123,231")
            logger.info("")
            logger.info("📊 2) LONG (failed-breakdown → reclaim) — status: waiting")
            logger.info(f"   • Setup: Sweep sub-LOD then reclaim.")
            logger.info(f"   • Entry: After a 5m close back above ${RECLAIM_ENTRY_LOW:,}–${RECLAIM_ENTRY_HIGH:,}")
            logger.info(f"   • SL: ${RECLAIM_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${RECLAIM_TP1:,} / ${RECLAIM_TP2:,} (≈ range mid)")
            logger.info(f"   • Why: Liquidity grab below LOD, mean-revert toward mid-range (~{MID:,})")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORTS:")
            logger.info("")
            logger.info("📊 3) SHORT (breakdown continuation) — status: waiting")
            logger.info(f"   • Entry (sell-stop): ${BREAKDOWN_ENTRY_LOW:,}–${BREAKDOWN_ENTRY_HIGH:,} (clean break of LOD)")
            logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${BREAKDOWN_TP1:,} / ${BREAKDOWN_TP2:,}")
            logger.info(f"   • Why: Range expansion lower after taking out LOD; momentum continuation")
            logger.info("")
            logger.info("📊 4) SHORT (failed-breakout → rejection) — status: waiting")
            logger.info(f"   • Setup: Spike above 122,2xx that fails.")
            logger.info(f"   • Entry: ${REJECTION_ENTRY_LOW:,}–${REJECTION_ENTRY_HIGH:,} on 5m close back inside the prior range")
            logger.info(f"   • SL: ${REJECTION_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2 / runner: ${REJECTION_TP1:,} / ${REJECTION_TP2:,} / ${REJECTION_RUNNER:,}")
            logger.info(f"   • Why: Trap above HOD; fade back through mid into lower third")
            logger.info("")
        logger.info("")
        logger.info("📊 Bot-ready trigger logic (concise):")
        logger.info(f"   • Breakout long: if last_price >= {BREAKOUT_ENTRY_LOW} AND volume rule true → market buy; SL {BREAKOUT_STOP_LOSS}; TP1 {BREAKOUT_TP1}; TP2 {BREAKOUT_TP2}; Runner {BREAKOUT_RUNNER}")
        logger.info(f"   • Reclaim long: if 5m close back above {RECLAIM_ENTRY_LOW} after sweep sub-LOD AND volume rule → market buy; SL {RECLAIM_STOP_LOSS}; TP1 {RECLAIM_TP1}; TP2 {RECLAIM_TP2}")
        logger.info(f"   • Breakdown short: if last_price <= {BREAKDOWN_ENTRY_LOW} AND volume rule → market sell; SL {BREAKDOWN_STOP_LOSS}; TP1 {BREAKDOWN_TP1}; TP2 {BREAKDOWN_TP2}")
        logger.info(f"   • Rejection short: if 5m close back inside prior range after spike above 122,2xx AND volume rule → market sell; SL {REJECTION_STOP_LOSS}; TP1 {REJECTION_TP1}; TP2 {REJECTION_TP2}; Runner {REJECTION_RUNNER}")
        logger.info("")
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 1H Close: ${last_close:,.2f}, High: ${last_high:,.2f}, Low: ${last_low:,.2f}")
        logger.info(f"1H Volume: {last_volume:,.0f}, 1H SMA: {volume_sma_1h:,.0f}, Rel_Vol: {relative_volume_1h:.2f}")
        logger.info(f"5M Volume: {current_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}")
        logger.info("")
        
        # --- Volume Confirmation Check ---
        volume_confirmed_1h = relative_volume_1h >= VOLUME_THRESHOLD_1H
        volume_confirmed_5m = relative_volume_5m >= VOLUME_THRESHOLD_5M
        volume_confirmed = volume_confirmed_1h or volume_confirmed_5m
        
        # --- Execution Guards ---
        # Check if price is in chop band (117,250–117,900)
        in_chop_band = CHOP_BAND_LOW <= current_price <= CHOP_BAND_HIGH
        chop_volume_ok = relative_volume_1h >= CHOP_VOLUME_THRESHOLD or relative_volume_5m >= CHOP_VOLUME_THRESHOLD
        
        # Disallow entries inside chop band unless volume ≥ 1.5×
        chop_band_allowed = not in_chop_band or chop_volume_ok
        
        logger.info("")
        logger.info("🔒 Execution Guards:")
        logger.info(f"   • Price in chop band (${CHOP_BAND_LOW:,}–${CHOP_BAND_HIGH:,}): {'✅' if in_chop_band else '❌'}")
        logger.info(f"   • Chop band volume OK (≥{CHOP_VOLUME_THRESHOLD}x): {'✅' if chop_volume_ok else '❌'}")
        logger.info(f"   • Chop band entries allowed: {'✅' if chop_band_allowed else '❌'}")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout continuation Strategy
        if long_strategies_enabled and not trigger_state.get("breakout_triggered", False):
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_price <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and volume_confirmed and chop_band_allowed
            
            logger.info("🔍 LONG - Breakout continuation Analysis:")
            logger.info(f"   • Price in buy-stop zone (${BREAKOUT_ENTRY_LOW:,}–${BREAKOUT_ENTRY_HIGH:,}): {'✅' if in_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H: {relative_volume_1h:.2f}x, 5M: {relative_volume_5m:.2f}x): {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")
            
            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout continuation conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakout trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Breakout continuation Long",
                    entry_price=current_price,
                    stop_loss=BREAKOUT_STOP_LOSS,
                    take_profit=BREAKOUT_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakout continuation trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakout_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout continuation trade failed: {trade_result}")
        
        # 2. LONG - Failed-breakdown → reclaim Strategy
        if long_strategies_enabled and not trade_executed and not trigger_state.get("reclaim_triggered", False):
            in_reclaim_zone = RECLAIM_ENTRY_LOW <= current_price <= RECLAIM_ENTRY_HIGH
            sweep_detected, reclaim_detected = detect_sweep_and_reclaim(candles_5m, candles_15m, sweep_low=LOD, sweep_high=LOD, reclaim_level=RECLAIM_ENTRY_LOW)
            reclaim_ready = in_reclaim_zone and volume_confirmed and sweep_detected and reclaim_detected and chop_band_allowed
            
            logger.info("")
            logger.info("🔍 LONG - Failed-breakdown → reclaim Analysis:")
            logger.info(f"   • Price in entry zone (${RECLAIM_ENTRY_LOW:,}–${RECLAIM_ENTRY_HIGH:,}): {'✅' if in_reclaim_zone else '❌'}")
            logger.info(f"   • Sweep of LOD (${LOD:,}) detected: {'✅' if sweep_detected else '❌'}")
            logger.info(f"   • 5m close back above ${RECLAIM_ENTRY_LOW:,} detected: {'✅' if reclaim_detected else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Reclaim Ready: {'🎯 YES' if reclaim_ready else '⏳ NO'}")
            
            if reclaim_ready:
                logger.info("")
                logger.info("🎯 LONG - Failed-breakdown → reclaim conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Reclaim trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Failed-breakdown → reclaim Long",
                    entry_price=current_price,
                    stop_loss=RECLAIM_STOP_LOSS,
                    take_profit=RECLAIM_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Failed-breakdown → reclaim trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["reclaim_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Failed-breakdown → reclaim trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown continuation Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("breakdown_triggered", False):
            in_breakdown_zone = BREAKDOWN_ENTRY_LOW <= current_price <= BREAKDOWN_ENTRY_HIGH
            breakdown_ready = in_breakdown_zone and volume_confirmed and chop_band_allowed
            
            logger.info("")
            logger.info("🔍 SHORT - Breakdown continuation Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKDOWN_ENTRY_LOW:,}–${BREAKDOWN_ENTRY_HIGH:,}): {'✅' if in_breakdown_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")
            
            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown continuation conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Breakdown continuation Short",
                    entry_price=current_price,
                    stop_loss=BREAKDOWN_STOP_LOSS,
                    take_profit=BREAKDOWN_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakdown continuation trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakdown_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown continuation trade failed: {trade_result}")
        
        # 4. SHORT - Failed-breakout → rejection Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("rejection_triggered", False):
            in_rejection_zone = REJECTION_ENTRY_LOW <= current_price <= REJECTION_ENTRY_HIGH
            rejection_detected = detect_rejection_wick(candles_5m, candles_15m, entry_low=122200, entry_high=122300)
            rejection_ready = in_rejection_zone and volume_confirmed and rejection_detected and chop_band_allowed
            
            logger.info("")
            logger.info("🔍 SHORT - Failed-breakout → rejection Analysis:")
            logger.info(f"   • Price in entry zone (${REJECTION_ENTRY_LOW:,}–${REJECTION_ENTRY_HIGH:,}): {'✅' if in_rejection_zone else '❌'}")
            logger.info(f"   • Spike above 122,2xx that fails detected: {'✅' if rejection_detected else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Rejection Ready: {'🎯 YES' if rejection_ready else '⏳ NO'}")
            
            if rejection_ready:
                logger.info("")
                logger.info("🎯 SHORT - Failed-breakout → rejection conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Rejection trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Failed-breakout → rejection Short",
                    entry_price=current_price,
                    stop_loss=REJECTION_STOP_LOSS,
                    take_profit=REJECTION_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Failed-breakout → rejection trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["rejection_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Failed-breakout → rejection trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout triggered: {trigger_state.get('breakout_triggered', False)}")
            logger.info(f"Reclaim triggered: {trigger_state.get('reclaim_triggered', False)}")
            logger.info(f"Breakdown triggered: {trigger_state.get('breakdown_triggered', False)}")
            logger.info(f"Rejection triggered: {trigger_state.get('rejection_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — clean two-sided BTC plan for today completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — clean two-sided BTC plan for today logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — clean two-sided BTC plan for today completed (with error) ===")
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
    logger.info("  • LONG Breakout continuation: Entry $122,400-$122,650 (clearance above HOD)")
    logger.info("  • LONG Failed-breakdown → reclaim: Entry $118,400-$118,500 after 5m close back above")
    logger.info("  • SHORT Breakdown continuation: Entry $118,100-$118,200 (clean break of LOD)")
    logger.info("  • SHORT Failed-breakout → rejection: Entry $121,850-$121,950 on 5m close back inside prior range")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20 leverage)")
    logger.info("  • Volume confirm: ≥1.25× 20-period vol on 1h OR ≥2× 20-SMA vol on 5m at trigger")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — clean two-sided BTC plan for today Alert Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: Complete Two-Sided BTC Strategy - LONG & SHORT")
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