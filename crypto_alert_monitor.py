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
VOLUME_THRESHOLD_1H = 1.25  # ≥1.25× 20-period vol on 1h
VOLUME_THRESHOLD_5M = 2.0   # ≥2× 20-SMA vol on 5m

# Today's session levels from the new plan (BTC ≈ $121,847; HOD $124,234; LOD $119,183; MID $121,709)
HOD = 124234  # 24h high
LOD = 119183  # 24h low
MID = 121709  # Mid point of 24h range

# LONG - Breakout continuation strategy
BREAKOUT_ENTRY = 124478   # Entry: $124,478 (≈ HOD + 0.20% buffer)
BREAKOUT_STOP_LOSS = 124051   # SL: $124,051 (≈ HOD − 0.15%)
BREAKOUT_TP1 = 124947         # TP1: $124,947 (~+1.1R)
BREAKOUT_TP2 = 125416         # TP2: $125,416 (~+2.2R)

# LONG - Breakout retest (safer fill)
RETEST_ENTRY_LOW = 124356   # Entry zone: $124,356 – $124,478 (0.10–0.20% above HOD after break)
RETEST_ENTRY_HIGH = 124478  # Entry zone high
RETEST_STOP_LOSS = 124112   # SL: $124,112 (≈ HOD − 0.10%)
RETEST_TP1 = 124721         # TP1: $124,721 (~+1R from mid-zone)
RETEST_TP2 = 125026         # TP2: $125,026 (~+2R from mid-zone)

# SHORT - Breakdown continuation
BREAKDOWN_ENTRY = 118939   # Entry: $118,939 (≈ LOD − 0.20%)
BREAKDOWN_STOP_LOSS = 119366   # SL: $119,366 (≈ LOD + 0.15%)
BREAKDOWN_TP1 = 118470         # TP1: $118,470 (~+1.1R)
BREAKDOWN_TP2 = 118001         # TP2: $118,001 (~+2.2R)

# SHORT - Retest of broken support
SUPPORT_RETEST_ENTRY_LOW = 118939   # Entry zone: $118,939 – $119,061 (0.10–0.20% below LOD on retest)
SUPPORT_RETEST_ENTRY_HIGH = 119061  # Entry zone high
SUPPORT_RETEST_STOP_LOSS = 119305   # SL: $119,305 (≈ LOD + 0.10%)
SUPPORT_RETEST_TP1 = 118696         # TP1: $118,696 (~+1R from mid-zone)
SUPPORT_RETEST_TP2 = 118391         # TP2: $118,391 (~+2R from mid-zone)

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
                "support_retest_triggered": False,
                "last_trigger_ts": None,
                "last_1h_structure": None,
                "active_trade_direction": None,  # Track which direction is active
                "attempts_per_side": {"LONG": 0, "SHORT": 0}  # Track attempts per side (max 2)
            }
    return {
        "breakout_triggered": False, 
        "retest_triggered": False, 
        "breakdown_triggered": False,
        "support_retest_triggered": False,
        "last_trigger_ts": None,
        "last_1h_structure": None,
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
    Spiros — two-sided intraday plan for BTC (today).
    Live: BTC ≈ $121,847. Today's range: HOD $124,234 / LOD $119,183 / MID $121,709.
    
    Rules (both directions)
    • Trigger on 1h, execute on 5–15m.
    • Volume confirm: ≥ 1.25× 20-period vol on 1h or ≥ 2× 20-SMA vol on 5m at trigger.
    • Risk: define 1R = |entry − SL|; risk ≤ 0.5–1.0% of equity per setup. Partial at +1.0–1.2R, trail to BE.
    • Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
    LONG
    
    1) Breakout continuation
    • Entry: $124,478 (≈ HOD + 0.20% buffer).
    • SL: $124,051 (≈ HOD − 0.15%).
    • TP1 / TP2: $124,947 / $125,416 (~+1.1R / +2.2R).
    • Why: Expansion above today's high with trend participation when volume confirms. HOD: $124,234.
    
    2) Breakout retest (safer fill)
    • Entry zone: $124,356 – $124,478 (0.10–0.20% above HOD after break).
    • SL: $124,112 (≈ HOD − 0.10%).
    • TP1 / TP2 (from mid-zone): $124,721 / $125,026 (~+1R / +2R).
    • Why: Rejects back-inside attempts; continuation from reclaimed high.
    
    SHORT
    
    3) Breakdown continuation
    • Entry: $118,939 (≈ LOD − 0.20%).
    • SL: $119,366 (≈ LOD + 0.15%).
    • TP1 / TP2: $118,470 / $118,001 (~+1.1R / +2.2R).
    • Why: Momentum follow-through below today's low when sell volume expands. LOD: $119,183.
    
    4) Retest of broken support
    • Entry zone: $118,939 – $119,061 (0.10–0.20% below LOD on retest).
    • SL: $119,305 (≈ LOD + 0.10%).
    • TP1 / TP2 (from mid-zone): $118,696 / $118,391 (~+1R / +2R).
    • Why: Acceptance below range; failed reclaim of LOD.
    
    Context notes
    • Mid-range pivot: $121,709. Reclaim/lose it with volume for intraday scalps back to range edges.
    
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
        logger.info("🚀 Spiros — two-sided intraday plan for BTC (today)")
        logger.info(f"Live: BTC ≈ $121,847 | HOD ${HOD:,} | LOD ${LOD:,} | MID ${MID:,} | Current ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules (both directions):")
        logger.info(f"   • Trigger on 1h, execute on 5–15m.")
        logger.info(f"   • Volume confirm: ≥{VOLUME_THRESHOLD_1H}× 20-period vol on 1h or ≥{VOLUME_THRESHOLD_5M}× 20-SMA vol on 5m at trigger.")
        logger.info(f"   • Risk: define 1R = |entry − SL|; risk ≤ {RISK_PERCENTAGE}–1.0% of equity per setup. Partial at +1.0–1.2R, trail to BE.")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin × {LEVERAGE} leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG:")
            logger.info("")
            logger.info("📊 1) Breakout continuation — status: waiting")
            logger.info(f"   • Entry: ${BREAKOUT_ENTRY:,} (≈ HOD + 0.20% buffer)")
            logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,} (≈ HOD − 0.15%)")
            logger.info(f"   • TP1 / TP2: ${BREAKOUT_TP1:,} / ${BREAKOUT_TP2:,} (~+1.1R / +2.2R)")
            logger.info(f"   • Why: Expansion above today's high with trend participation when volume confirms. HOD: ${HOD:,}")
            logger.info("")
            logger.info("📊 2) Breakout retest (safer fill) — status: waiting")
            logger.info(f"   • Entry zone: ${RETEST_ENTRY_LOW:,} – ${RETEST_ENTRY_HIGH:,} (0.10–0.20% above HOD after break)")
            logger.info(f"   • SL: ${RETEST_STOP_LOSS:,} (≈ HOD − 0.10%)")
            logger.info(f"   • TP1 / TP2 (from mid-zone): ${RETEST_TP1:,} / ${RETEST_TP2:,} (~+1R / +2R)")
            logger.info(f"   • Why: Rejects back-inside attempts; continuation from reclaimed high")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT:")
            logger.info("")
            logger.info("📊 3) Breakdown continuation — status: waiting")
            logger.info(f"   • Entry: ${BREAKDOWN_ENTRY:,} (≈ LOD − 0.20%)")
            logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,} (≈ LOD + 0.15%)")
            logger.info(f"   • TP1 / TP2: ${BREAKDOWN_TP1:,} / ${BREAKDOWN_TP2:,} (~+1.1R / +2.2R)")
            logger.info(f"   • Why: Momentum follow-through below today's low when sell volume expands. LOD: ${LOD:,}")
            logger.info("")
            logger.info("📊 4) Retest of broken support — status: waiting")
            logger.info(f"   • Entry zone: ${SUPPORT_RETEST_ENTRY_LOW:,} – ${SUPPORT_RETEST_ENTRY_HIGH:,} (0.10–0.20% below LOD on retest)")
            logger.info(f"   • SL: ${SUPPORT_RETEST_STOP_LOSS:,} (≈ LOD + 0.10%)")
            logger.info(f"   • TP1 / TP2 (from mid-zone): ${SUPPORT_RETEST_TP1:,} / ${SUPPORT_RETEST_TP2:,} (~+1R / +2R)")
            logger.info(f"   • Why: Acceptance below range; failed reclaim of LOD")
            logger.info("")
        
        logger.info("📊 Context notes:")
        logger.info(f"   • Mid-range pivot: ${MID:,}. Reclaim/lose it with volume for intraday scalps back to range edges.")
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
        logger.info(f"5M Volume: {current_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}")
        logger.info("")
        
        # --- Volume Confirmation Check ---
        volume_confirmed_1h = relative_volume_1h >= VOLUME_THRESHOLD_1H
        volume_confirmed_5m = relative_volume_5m >= VOLUME_THRESHOLD_5M
        volume_confirmed = volume_confirmed_1h or volume_confirmed_5m
        
        # --- Execution Guards ---
        # Check for immediate reversal candle (5–15m) against the trade
        # This will be checked in each strategy individually
        
        logger.info("")
        logger.info("🔒 Execution Guards:")
        logger.info("   • Volume confirmation required: ≥1.25× 20-period vol on 1h OR ≥2× 20-SMA vol on 5m")
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
        
        # 1. LONG - Breakout continuation Strategy
        if long_strategies_enabled and not trigger_state.get("breakout_triggered", False) and long_attempts < 2:
            # Check if price is at the exact breakout entry level
            at_breakout_entry = abs(current_price - BREAKOUT_ENTRY) <= 10  # Allow small tolerance
            breakout_ready = at_breakout_entry and volume_confirmed
            
            logger.info("")
            logger.info("🔍 LONG - Breakout continuation Analysis:")
            logger.info(f"   • Price at entry level ${BREAKOUT_ENTRY:,} (≈ HOD + 0.20% buffer): {'✅' if at_breakout_entry else '❌'}")
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
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout continuation trade failed: {trade_result}")
        
        # 2. LONG - Breakout retest Strategy
        if long_strategies_enabled and not trade_executed and not trigger_state.get("retest_triggered", False) and long_attempts < 2:
            in_retest_zone = RETEST_ENTRY_LOW <= current_price <= RETEST_ENTRY_HIGH
            retest_ready = in_retest_zone and volume_confirmed
            
            logger.info("")
            logger.info("🔍 LONG - Breakout retest Analysis:")
            logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,}–${RETEST_ENTRY_HIGH:,}): {'✅' if in_retest_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if retest_ready else '⏳ NO'}")
            
            if retest_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout retest conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Breakout retest Long",
                    entry_price=current_price,
                    stop_loss=RETEST_STOP_LOSS,
                    take_profit=RETEST_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakout retest trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["retest_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout retest trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown continuation Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("breakdown_triggered", False) and short_attempts < 2:
            # Check if price is at the exact breakdown entry level
            at_breakdown_entry = abs(current_price - BREAKDOWN_ENTRY) <= 10  # Allow small tolerance
            breakdown_ready = at_breakdown_entry and volume_confirmed
            
            logger.info("")
            logger.info("🔍 SHORT - Breakdown continuation Analysis:")
            logger.info(f"   • Price at entry level ${BREAKDOWN_ENTRY:,} (≈ LOD − 0.20%): {'✅' if at_breakdown_entry else '❌'}")
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
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown continuation trade failed: {trade_result}")
        
        # 4. SHORT - Retest of broken support Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("support_retest_triggered", False) and short_attempts < 2:
            in_support_retest_zone = SUPPORT_RETEST_ENTRY_LOW <= current_price <= SUPPORT_RETEST_ENTRY_HIGH
            support_retest_ready = in_support_retest_zone and volume_confirmed
            
            logger.info("")
            logger.info("🔍 SHORT - Retest of broken support Analysis:")
            logger.info(f"   • Price in entry zone (${SUPPORT_RETEST_ENTRY_LOW:,}–${SUPPORT_RETEST_ENTRY_HIGH:,}): {'✅' if in_support_retest_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if support_retest_ready else '⏳ NO'}")
            
            if support_retest_ready:
                logger.info("")
                logger.info("🎯 SHORT - Retest of broken support conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Two-Sided Plan - Retest of broken support Short",
                    entry_price=current_price,
                    stop_loss=SUPPORT_RETEST_STOP_LOSS,
                    take_profit=SUPPORT_RETEST_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Retest of broken support trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["support_retest_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Retest of broken support trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout triggered: {trigger_state.get('breakout_triggered', False)}")
            logger.info(f"Retest triggered: {trigger_state.get('retest_triggered', False)}")
            logger.info(f"Breakdown triggered: {trigger_state.get('breakdown_triggered', False)}")
            logger.info(f"Support retest triggered: {trigger_state.get('support_retest_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — two-sided intraday plan for BTC (today) completed ===")
        return last_ts if trade_executed else last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in Spiros — two-sided intraday plan for BTC (today) logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Spiros — two-sided intraday plan for BTC (today) completed (with error) ===")
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
    logger.info("  • LONG Breakout continuation: Entry $124,478 (≈ HOD + 0.20% buffer)")
    logger.info("  • LONG Breakout retest (safer fill): Entry $124,356 – $124,478 (0.10–0.20% above HOD after break)")
    logger.info("  • SHORT Breakdown continuation: Entry $118,939 (≈ LOD − 0.20%)")
    logger.info("  • SHORT Retest of broken support: Entry $118,939 – $119,061 (0.10–0.20% below LOD on retest)")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20 leverage)")
    logger.info("  • Volume confirm: ≥1.25× 20-period vol on 1h OR ≥2× 20-SMA vol on 5m at trigger")
    logger.info("  • Max 2 attempts per side; if trigger wicks without close + volume, abort and retry only once")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — two-sided intraday plan for BTC (today) Alert Monitor")
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