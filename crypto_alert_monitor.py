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
RISK_PERCENTAGE = 0.5  # 0.5-1.0% of equity as 1R
VOLUME_THRESHOLD_1H = 1.25  # 1.25x 20-period vol on 1h
VOLUME_THRESHOLD_5M = 2.0   # 2x 20-SMA vol on 5m

# Today's session levels from the new plan (BTC ≈ 117,884, HOD 118,655, LOD 116,468)
HOD = 118655  # 24h high
LOD = 116468  # 24h low
MID = 117562  # Mid point of 24h range

# LONG - Breakout strategy (range expansion)
BREAKOUT_ENTRY_LOW = 118785   # Entry: 118,785–118,865 (≈ HOD + 0.11–0.18%)
BREAKOUT_ENTRY_HIGH = 118865  # Entry zone high
BREAKOUT_STOP_LOSS = 118300   # SL: 118,300 (back inside prior range by ~0.3%)
BREAKOUT_TP1 = 119400         # TP1: 119,400
BREAKOUT_TP2 = 119850         # TP2: 119,850–120,050

# LONG - LOD sweep → reclaim
RETEST_ENTRY_LOW = 116720     # Entry: 116,720–116,820 after a LOD sweep and 5–15m close back above 116,700 with volume
RETEST_ENTRY_HIGH = 116820    # Entry zone high
RETEST_STOP_LOSS = 116280     # SL: 116,280 (≈ LOD − 0.15%)
RETEST_TP1 = 117280           # TP1: 117,280
RETEST_TP2 = 118000           # TP2: 118,000

# SHORT - Breakdown strategy (range expansion)
BREAKDOWN_ENTRY_LOW = 116250   # Entry: 116,250–116,150 (≈ LOD − 0.17–0.27%)
BREAKDOWN_ENTRY_HIGH = 116150  # Entry zone high
BREAKDOWN_STOP_LOSS = 116720   # SL: 116,720 (back inside range)
BREAKDOWN_TP1 = 115680         # TP1: 115,680
BREAKDOWN_TP2 = 115160         # TP2: 115,160–114,900

# SHORT - Failed high → lower-high sell
FADE_ENTRY_LOW = 118050        # Entry: 118,050–118,150 on rejection (5–15m close back below 118,150 + rising 5m volume)
FADE_ENTRY_HIGH = 118150       # Entry zone high
FADE_STOP_LOSS = 118420        # SL: 118,420
FADE_TP1 = 117780              # TP1: 117,780
FADE_TP2 = 117460              # TP2: 117,460–117,300

# Execution guards
CHOP_BAND_LOW = 117250         # Disallow entries inside 117,250–117,900 unless volume ≥ 1.5×
CHOP_BAND_HIGH = 117900
CHOP_VOLUME_THRESHOLD = 1.5    # 1.5x volume required in chop band

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
                "retest_triggered": False, 
                "breakdown_triggered": False,
                "fade_triggered": False,
                "last_trigger_ts": None,
                "last_1h_structure": None,
                "active_trade_direction": None  # Track which direction is active
            }
    return {
        "breakout_triggered": False, 
        "retest_triggered": False, 
        "breakdown_triggered": False,
        "fade_triggered": False,
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

def detect_sweep_and_reclaim(candles_5m, candles_15m, sweep_low=116720, sweep_high=116820, reclaim_level=116700):
    """
    Detect if there was a sweep of the specified range followed by a reclaim above 116,700
    
    Args:
        candles_5m: List of 5-minute candles
        candles_15m: List of 15-minute candles
        sweep_low: Lower bound of sweep range
        sweep_high: Upper bound of sweep range
        reclaim_level: Level to reclaim above
    
    Returns:
        Tuple of (sweep_detected, reclaim_detected)
    """
    if not candles_5m or len(candles_5m) < 12:  # Need at least 1 hour of 5m data
        return False, False
    
    if not candles_15m or len(candles_15m) < 4:  # Need at least 1 hour of 15m data
        return False, False
    
    # Check for sweep in recent 5m candles (last 12 candles = 1 hour)
    sweep_detected = False
    for i, candle in enumerate(candles_5m[1:13]):  # Skip current incomplete candle
        low = float(get_candle_value(candle, 'low'))
        if sweep_low <= low <= sweep_high:
            sweep_detected = True
            break
    
    # Check for reclaim in recent 15m candles (last 4 candles = 1 hour)
    reclaim_detected = False
    if sweep_detected:
        for i, candle in enumerate(candles_15m[1:5]):  # Skip current incomplete candle
            close = float(get_candle_value(candle, 'close'))
            if close > reclaim_level:
                reclaim_detected = True
                break
    
    return sweep_detected, reclaim_detected

def detect_rejection_wick(candles_5m, candles_15m, entry_low=118050, entry_high=118150):
    """
    Detect if there was a rejection wick in the specified range with 5–15m close back below 118,150 + rising 5m volume
    
    Args:
        candles_5m: List of 5-minute candles
        candles_15m: List of 15-minute candles
        entry_low: Lower bound of entry range
        entry_high: Upper bound of entry range
    
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
            
            # Rejection wick if wick is significant compared to body and close below 118,150
            if wick_size > body_size * 0.5 and close < 118150:
                # Check for rising 5m volume (compare with previous candle)
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
            # Check for rejection wick and close below 118,150
            wick_size = high - max(close, low)
            body_size = abs(close - low)
            
            # Rejection wick if wick is significant compared to body and close below 118,150
            if wick_size > body_size * 0.5 and close < 118150:
                return True
    
    return False



def btc_intraday_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    Spiros — BTC plan for today (live now): HOD 118,655 / LOD 116,468 / last ≈ 117,884.
    
    Rules (both directions):
    - Venue/Exec: Coinbase perps. Bracket market orders only.
    - Trigger TF: Use 1h for context, execute on 5–15m (no 1h acceptance required).
    - Volume confirm: ≥ 1.25× 20-SMA volume on 1h or ≥ 2× 20-SMA volume on 5m at trigger.
    - Risk: Size so 1R (entry→SL) risks ≤0.5–1.0% of equity. Partial at +1.0–1.3R; move SL to BE at +1.0R.
    - Attempts: Max 2 total (one per side). If invalidated, stand down until a different setup type triggers.
    - Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
    LONGS:
    1) Breakout (range expansion) — status: waiting
    - Entry: 118,785–118,865 (≈ HOD + 0.11–0.18%).
    - SL: 118,300 (back inside prior range by ~0.3%).
    - TP1 / TP2: 119,400 / 119,850–120,050.
    - Why: Clean HOD break with today's range ~1.9% opens path toward 120k; volume must confirm expansion.
    
    2) LOD sweep → reclaim — status: waiting
    - Entry: 116,720–116,820 after a LOD sweep and 5–15m close back above 116,700 with volume.
    - SL: 116,280 (≈ LOD − 0.15%).
    - TP1 / TP2: 117,280 / 118,000.
    - Why: Stop-run reversal off lows; needs real reclaim, not just a wick.
    
    SHORTS:
    3) Breakdown (range expansion) — status: waiting
    - Entry: 116,250–116,150 (≈ LOD − 0.17–0.27%).
    - SL: 116,720 (back inside range).
    - TP1 / TP2: 115,680 / 115,160–114,900.
    - Why: Fresh lows with confirmation target the 115k liquidity pocket.
    
    4) Failed high → lower-high sell — status: waiting
    - Entry: 118,050–118,150 on rejection (5–15m close back below 118,150 + rising 5m volume).
    - SL: 118,420.
    - TP1 / TP2: 117,780 / 117,460–117,300.
    - Why: Rotational fade if breakouts fail and prior support flips to resistance.
    
    Execution guards (to code in):
    - Disallow entries inside 117,250–117,900 unless volume ≥ 1.5× (chop band).
    - Cancel trigger if post-entry 5–15m bar closes back inside the broken level and volume < trigger bar.
    - One-shot per setup type: don't re-arm the same setup within 60 minutes if stopped.
    
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
        logger.info("🚀 Spiros — BTC plan for today (live now)")
        logger.info(f"HOD ${HOD:,} / LOD ${LOD:,} / last ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules (both directions):")
        logger.info(f"   • Venue/Exec: Coinbase perps. Bracket market orders only.")
        logger.info(f"   • Trigger TF: Use 1h for context, execute on 5–15m (no 1h acceptance required).")
        logger.info(f"   • Volume confirm: ≥ {VOLUME_THRESHOLD_1H}× 20-SMA volume on 1h or ≥ {VOLUME_THRESHOLD_5M}× 20-SMA volume on 5m at trigger.")
        logger.info(f"   • Risk: Size so 1R (entry→SL) risks ≤{RISK_PERCENTAGE}–1.0% of equity. Partial at +1.0–1.3R; move SL to BE at +1.0R.")
        logger.info(f"   • Attempts: Max 2 total (one per side). If invalidated, stand down until a different setup type triggers.")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin × {LEVERAGE} leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONGS:")
            logger.info("")
            logger.info("📊 1) Breakout (range expansion) — status: waiting")
            logger.info(f"   • Entry: ${BREAKOUT_ENTRY_LOW:,}–${BREAKOUT_ENTRY_HIGH:,} (≈ HOD + 0.11–0.18%)")
            logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,} (back inside prior range by ~0.3%)")
            logger.info(f"   • TP1 / TP2: ${BREAKOUT_TP1:,} / ${BREAKOUT_TP2:,}")
            logger.info(f"   • Why: Clean HOD break with today's range ~1.9% opens path toward 120k; volume must confirm expansion")
            logger.info("")
            logger.info("📊 2) LOD sweep → reclaim — status: waiting")
            logger.info(f"   • Entry: ${RETEST_ENTRY_LOW:,}–${RETEST_ENTRY_HIGH:,} after a LOD sweep and 5–15m close back above 116,700 with volume")
            logger.info(f"   • SL: ${RETEST_STOP_LOSS:,} (≈ LOD − 0.15%)")
            logger.info(f"   • TP1 / TP2: ${RETEST_TP1:,} / ${RETEST_TP2:,}")
            logger.info(f"   • Why: Stop-run reversal off lows; needs real reclaim, not just a wick")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORTS:")
            logger.info("")
            logger.info("📊 3) Breakdown (range expansion) — status: waiting")
            logger.info(f"   • Entry: ${BREAKDOWN_ENTRY_LOW:,}–${BREAKDOWN_ENTRY_HIGH:,} (≈ LOD − 0.17–0.27%)")
            logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,} (back inside range)")
            logger.info(f"   • TP1 / TP2: ${BREAKDOWN_TP1:,} / ${BREAKDOWN_TP2:,}")
            logger.info(f"   • Why: Fresh lows with confirmation target the 115k liquidity pocket")
            logger.info("")
            logger.info("📊 4) Failed high → lower-high sell — status: waiting")
            logger.info(f"   • Entry: ${FADE_ENTRY_LOW:,}–${FADE_ENTRY_HIGH:,} on rejection (5–15m close back below 118,150 + rising 5m volume)")
            logger.info(f"   • SL: ${FADE_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${FADE_TP1:,} / ${FADE_TP2:,}")
            logger.info(f"   • Why: Rotational fade if breakouts fail and prior support flips to resistance")
            logger.info("")
        logger.info("")
        logger.info("📊 Execution guards (to code in):")
        logger.info("   • Disallow entries inside 117,250–117,900 unless volume ≥ 1.5× (chop band).")
        logger.info("   • Cancel trigger if post-entry 5–15m bar closes back inside the broken level and volume < trigger bar.")
        logger.info("   • One-shot per setup type: don't re-arm the same setup within 60 minutes if stopped.")
        logger.info("   • Max **2** total (one per side). If invalidated, stand down until a different setup type triggers.")
        logger.info("")
        logger.info("📊 Trigger logic (for your scanner; market brackets only):")
        logger.info("   • Breakout long: `crosses_up(price, 118785..118865) AND vol_ok AND not_in_chop_band`")
        logger.info("   • Retest long: `low_sweeps(116720..116820) AND reclaim(116700, 5–15m) AND vol_ok AND not_in_chop_band`")
        logger.info("   • Breakdown short: `crosses_down(price, 116250..116150) AND vol_ok AND not_in_chop_band`")
        logger.info("   • Fade short: `touches(118050..118150) AND rejection_wick(5–15m) AND vol_ok AND not_in_chop_band`")
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
        
        # 1. LONG - Breakout Strategy
        if long_strategies_enabled and not trigger_state.get("breakout_triggered", False):
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_price <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and volume_confirmed and chop_band_allowed
            
            logger.info("🔍 LONG - Breakout Analysis:")
            logger.info(f"   • Price in buy-stop zone (${BREAKOUT_ENTRY_LOW:,}–${BREAKOUT_ENTRY_HIGH:,}): {'✅' if in_breakout_zone else '❌'}")
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
                    trade_type="BTC Two-Sided Plan - Breakout Long",
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
        
        # 2. LONG - Retest Strategy
        if long_strategies_enabled and not trade_executed and not trigger_state.get("retest_triggered", False):
            in_retest_zone = RETEST_ENTRY_LOW <= current_price <= RETEST_ENTRY_HIGH
            sweep_detected, reclaim_detected = detect_sweep_and_reclaim(candles_5m, candles_15m)
            retest_ready = in_retest_zone and volume_confirmed and sweep_detected and reclaim_detected and chop_band_allowed
            
            logger.info("")
            logger.info("🔍 LONG - Retest Analysis:")
            logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,}–${RETEST_ENTRY_HIGH:,}): {'✅' if in_retest_zone else '❌'}")
            logger.info(f"   • Sweep of $116,720–$116,820 detected: {'✅' if sweep_detected else '❌'}")
            logger.info(f"   • 5–15m reclaim of $116,700 detected: {'✅' if reclaim_detected else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if retest_ready else '⏳ NO'}")
            
            if retest_ready:
                logger.info("")
                logger.info("🎯 LONG - Retest conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Retest Long",
                    entry_price=current_price,
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
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Retest trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("breakdown_triggered", False):
            in_breakdown_zone = BREAKDOWN_ENTRY_LOW <= current_price <= BREAKDOWN_ENTRY_HIGH
            breakdown_ready = in_breakdown_zone and volume_confirmed and chop_band_allowed
            
            logger.info("")
            logger.info("🔍 SHORT - Breakdown Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKDOWN_ENTRY_LOW:,}–${BREAKDOWN_ENTRY_HIGH:,}): {'✅' if in_breakdown_zone else '❌'}")
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
                    trade_type="BTC Two-Sided Plan - Breakdown Short",
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
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown trade failed: {trade_result}")
        
        # 4. SHORT - Fade Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("fade_triggered", False):
            in_fade_zone = FADE_ENTRY_LOW <= current_price <= FADE_ENTRY_HIGH
            rejection_detected = detect_rejection_wick(candles_5m, candles_15m)
            fade_ready = in_fade_zone and volume_confirmed and rejection_detected and chop_band_allowed
            
            logger.info("")
            logger.info("🔍 SHORT - Fade into resistance Analysis:")
            logger.info(f"   • Price in entry zone (${FADE_ENTRY_LOW:,}–${FADE_ENTRY_HIGH:,}): {'✅' if in_fade_zone else '❌'}")
            logger.info(f"   • Spike + rejection (upper wick 5–15m) detected: {'✅' if rejection_detected else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Fade into resistance Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")
            
            if fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fade into resistance conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Fade into resistance trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Fade into Resistance Short",
                    entry_price=current_price,
                    stop_loss=FADE_STOP_LOSS,
                    take_profit=FADE_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Fade into resistance trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["fade_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Fade into resistance trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout triggered: {trigger_state.get('breakout_triggered', False)}")
            logger.info(f"Retest triggered: {trigger_state.get('retest_triggered', False)}")
            logger.info(f"Breakdown triggered: {trigger_state.get('breakdown_triggered', False)}")
            logger.info(f"Fade triggered: {trigger_state.get('fade_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — BTC plan for today completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — BTC plan for today logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — BTC plan for today completed (with error) ===")
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
    logger.info("  • LONG Breakout (range expansion): Entry $118,785-$118,865 (≈ HOD + 0.11–0.18%)")
    logger.info("  • LONG LOD sweep → reclaim: Entry $116,720-$116,820 after a LOD sweep and 5–15m close back above 116,700 with volume")
    logger.info("  • SHORT Breakdown (range expansion): Entry $116,250-$116,150 (≈ LOD − 0.17–0.27%)")
    logger.info("  • SHORT Failed high → lower-high sell: Entry $118,050-$118,150 on rejection (5–15m close back below 118,150 + rising 5m volume)")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20 leverage)")
    logger.info("  • Execution Guards: Disallow entries inside 117,250–117,900 unless volume ≥ 1.5×")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — BTC plan for today Alert Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: Complete Two-Sided BTC Plan for today - LONG & SHORT")
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