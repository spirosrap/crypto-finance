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

# Today's session levels from the new plan (BTC ≈ $118.1k; today's intraday range $117,268–$118,271)
CURRENT_PRICE = 118100  # Price now
TODAY_HOD = 118271  # Today's high
TODAY_LOD = 117268  # Today's low

# 1. Long — Breakout/hold above today's high
BREAKOUT_LONG_ENTRY_LOW = 118400   # Entry zone: $118,400–$118,700
BREAKOUT_LONG_ENTRY_HIGH = 118700  # Entry zone high
BREAKOUT_LONG_TRIGGER = 118300     # Trigger: 15–60m close above $118,300 (clears today's high)
BREAKOUT_LONG_STOP_LOSS = 117900   # SL: $117,900 (or ~0.8% below trigger, structure-based)
BREAKOUT_LONG_TP1 = 119800         # TP1: $119,800
BREAKOUT_LONG_TP2 = 121000         # TP2: $121,000
BREAKOUT_LONG_VOLUME_THRESHOLD = 1.25  # Volume confirm: 1h volume ≥ 125% of 20-period avg

# 2. Long — Liquidity sweep & reclaim
SWEEP_LONG_ENTRY_LOW = 116900      # Entry zone: $116,900–$117,100 (sweep zone)
SWEEP_LONG_ENTRY_HIGH = 117100     # Entry zone high
SWEEP_LONG_RECLAIM = 117200        # Reclaim: $117,200 with a 15m close
SWEEP_LONG_STOP_LOSS = 116500      # SL: $116,500
SWEEP_LONG_TP1 = 118200            # TP1: $118,200
SWEEP_LONG_TP2 = 119400            # TP2: $119,400
SWEEP_LONG_VOLUME_THRESHOLD = 1.0  # Volume: Spike on sweep (capitulation) and rising delta on reclaim

# 3. Short — Breakdown & acceptance
BREAKDOWN_SHORT_ENTRY_LOW = 116950   # Entry zone: $116,950–$117,150
BREAKDOWN_SHORT_ENTRY_HIGH = 117150  # Entry zone high
BREAKDOWN_SHORT_TRIGGER = 117200     # Trigger: two 15m closes below $117,200 (today's low lost + VWAP rejection)
BREAKDOWN_SHORT_STOP_LOSS = 117900   # SL: $117,900
BREAKDOWN_SHORT_TP1 = 116000         # TP1: $116,000
BREAKDOWN_SHORT_TP2 = 115200         # TP2: $115,200
BREAKDOWN_SHORT_VOLUME_THRESHOLD = 1.50  # Volume confirm: 1h sell volume ≥ 150% of 20-period avg on breakdown bar

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
                "sweep_long_triggered": False, 
                "breakdown_short_triggered": False,
                "last_trigger_ts": None,
                "last_15m_structure": None,
                "active_trade_direction": None,  # Track which direction is active
                "attempts_per_side": {"LONG": 0, "SHORT": 0},  # Track attempts per side (max 2)
                "sweep_low_reached": False,  # Track if sweep low was reached for reclaim strategy
                "breakdown_closes_count": 0  # Track consecutive closes below breakdown trigger
            }
    return {
        "breakout_long_triggered": False, 
        "sweep_long_triggered": False, 
        "breakdown_short_triggered": False,
        "last_trigger_ts": None,
        "last_15m_structure": None,
        "active_trade_direction": None,  # Track which direction is active
        "attempts_per_side": {"LONG": 0, "SHORT": 0},  # Track attempts per side (max 2)
        "sweep_low_reached": False,  # Track if sweep low was reached for reclaim strategy
        "breakdown_closes_count": 0  # Track consecutive closes below breakdown trigger
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
    Spiros — here are clean, conditional BTC setups for today. First, a quick snapshot, then entries.
    
    Context (facts)
    • Spot ~$118.1k; today's intraday range (so far) $117,268–$118,271.
    • Funding is mildly positive across majors (~+0.00–0.01%), i.e., slight long skew but not crowded.
    • Futures OI elevated (~$82.6B, +1.3% d/d) → moves can extend once triggered.
    
    Setups (objective triggers):
    1. Long — Breakout/hold above today's high
    • Entry: $118,400–$118,700 after a 15–60m close above $118,300 (clears today's high)
    • SL: $117,900 (or ~0.8% below trigger, structure-based)
    • TP1: $119,800 • TP2: $121,000
    • Why: Clears intraday supply; with positive (but tame) funding and firm OI, momentum continuation is favored once acceptance above the high prints.
    • Volume condition: 1h volume ≥ 125% of 20-period avg on your Coinbase perp chart.
    • Timeframe / Type: 1h–4h breakout continuation
    
    2. Long — Liquidity sweep & reclaim
    • Entry: On a fast flush into $116,900–$117,100, then reclaim $117,200 with a 15m close
    • SL: $116,500
    • TP1: $118,200 • TP2: $119,400
    • Why: Fades a stop-run below today's low ($117,268) into prior demand, then rides the reclaim. Works best if funding stays near flat or dips.
    • Volume condition: Spike on the sweep (capitulation) and rising delta on the reclaim bar.
    • Timeframe / Type: 15m–1h sweep-reclaim (SFP)
    
    3. Short — Breakdown & acceptance
    • Entry: $116,950–$117,150 after two 15m closes below $117,200 (today's low lost + VWAP rejection)
    • SL: $117,900
    • TP1: $116,000 • TP2: $115,200
    • Why: Losing the low with OI elevated often invites continuation as longs unwind. Confirmation improves if funding flips toward zero/negative.
    • Volume condition: 1h sell volume ≥ 150% of 20-period avg on breakdown bar.
    • Timeframe / Type: 15m–1h trend continuation (breakdown)
    
    Execution notes (objective):
    • Use acceptance rules (multiple closes) rather than wick pokes.
    • If TP1 hits, move SL to breakeven; partials at TP1 (~50–60%) keep expectancy rational.
    • Skip trades if the volume conditions aren't met — no exceptions.
    • Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
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
        logger.info("🚀 Spiros — Clean Conditional BTC Setups (USD quotes)")
        logger.info(f"Live: BTC ≈ ${CURRENT_PRICE:,.0f} | HOD ${TODAY_HOD:,} | LOD ${TODAY_LOD:,} | Current ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules (both directions):")
        logger.info(f"   • Use acceptance rules (multiple closes) rather than wick pokes.")
        logger.info(f"   • Volume confirm: ≥{BREAKOUT_LONG_VOLUME_THRESHOLD:.0%}× 20-period vol for breakout long, ≥{BREAKDOWN_SHORT_VOLUME_THRESHOLD:.0%}× for breakdown short.")
        logger.info(f"   • If TP1 hits, move SL to breakeven; partials at TP1 (~50–60%) keep expectancy rational.")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} margin × {LEVERAGE} leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG:")
            logger.info("")
            logger.info("📊 1) Long — Breakout/hold above today's high — status: waiting")
            logger.info(f"   • Entry: ${BREAKOUT_LONG_ENTRY_LOW:,}–${BREAKOUT_LONG_ENTRY_HIGH:,} after a 15–60m close above ${BREAKOUT_LONG_TRIGGER:,} (clears today's high)")
            logger.info(f"   • SL: ${BREAKOUT_LONG_STOP_LOSS:,} (or ~0.8% below trigger, structure-based)")
            logger.info(f"   • TP1: ${BREAKOUT_LONG_TP1:,} • TP2: ${BREAKOUT_LONG_TP2:,}")
            logger.info(f"   • Why: Clears intraday supply; with positive (but tame) funding and firm OI, momentum continuation is favored once acceptance above the high prints")
            logger.info(f"   • Volume condition: 1h volume ≥ {BREAKOUT_LONG_VOLUME_THRESHOLD:.0%} of 20-period avg on your Coinbase perp chart")
            logger.info(f"   • Timeframe / Type: 1h–4h breakout continuation")
            logger.info("")
            logger.info("📊 2) Long — Liquidity sweep & reclaim — status: waiting")
            logger.info(f"   • Entry: On a fast flush into ${SWEEP_LONG_ENTRY_LOW:,}–${SWEEP_LONG_ENTRY_HIGH:,}, then reclaim ${SWEEP_LONG_RECLAIM:,} with a 15m close")
            logger.info(f"   • SL: ${SWEEP_LONG_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${SWEEP_LONG_TP1:,} • TP2: ${SWEEP_LONG_TP2:,}")
            logger.info(f"   • Why: Fades a stop-run below today's low (${TODAY_LOD:,}) into prior demand, then rides the reclaim. Works best if funding stays near flat or dips")
            logger.info(f"   • Volume condition: Spike on the sweep (capitulation) and rising delta on the reclaim bar")
            logger.info(f"   • Timeframe / Type: 15m–1h sweep-reclaim (SFP)")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT:")
            logger.info("")
            logger.info("📊 3) Short — Breakdown & acceptance — status: waiting")
            logger.info(f"   • Entry: ${BREAKDOWN_SHORT_ENTRY_LOW:,}–${BREAKDOWN_SHORT_ENTRY_HIGH:,} after two 15m closes below ${BREAKDOWN_SHORT_TRIGGER:,} (today's low lost + VWAP rejection)")
            logger.info(f"   • SL: ${BREAKDOWN_SHORT_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${BREAKDOWN_SHORT_TP1:,} • TP2: ${BREAKDOWN_SHORT_TP2:,}")
            logger.info(f"   • Why: Losing the low with OI elevated often invites continuation as longs unwind. Confirmation improves if funding flips toward zero/negative")
            logger.info(f"   • Volume condition: 1h sell volume ≥ {BREAKDOWN_SHORT_VOLUME_THRESHOLD:.0%} of 20-period avg on breakdown bar")
            logger.info(f"   • Timeframe / Type: 15m–1h trend continuation (breakdown)")
            logger.info("")
        
        logger.info("📊 Alert text (copy/paste):")
        logger.info(f"   • 'BTC long if 15–60m closes > {BREAKOUT_LONG_TRIGGER:,}; enter {BREAKOUT_LONG_ENTRY_LOW:,}–{BREAKOUT_LONG_ENTRY_HIGH:,}; SL {BREAKOUT_LONG_STOP_LOSS:,}; TP1 {BREAKOUT_LONG_TP1:,}, TP2 {BREAKOUT_LONG_TP2:,}; 1h vol ≥{BREAKOUT_LONG_VOLUME_THRESHOLD:.0%} of 20-SMA.'")
        logger.info(f"   • 'BTC long on sweep to {SWEEP_LONG_ENTRY_LOW:,}–{SWEEP_LONG_ENTRY_HIGH:,}, then 15m close back above {SWEEP_LONG_RECLAIM:,}; SL {SWEEP_LONG_STOP_LOSS:,}; TP1 {SWEEP_LONG_TP1:,}, TP2 {SWEEP_LONG_TP2:,}; reclaim volume uptick.'")
        logger.info(f"   • 'BTC short if two 15m closes < {BREAKDOWN_SHORT_TRIGGER:,}; enter {BREAKDOWN_SHORT_ENTRY_LOW:,}–{BREAKDOWN_SHORT_ENTRY_HIGH:,}; SL {BREAKDOWN_SHORT_STOP_LOSS:,}; TP1 {BREAKDOWN_SHORT_TP1:,}, TP2 {BREAKDOWN_SHORT_TP2:,}; 1h vol ≥{BREAKDOWN_SHORT_VOLUME_THRESHOLD:.0%} of 20-SMA.'")
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
        
        # Volume confirmation for different strategies
        volume_confirmed_breakout_long = relative_volume_15m >= BREAKOUT_LONG_VOLUME_THRESHOLD  # ≥125% for breakout long
        volume_confirmed_breakdown_short = relative_volume_15m >= BREAKDOWN_SHORT_VOLUME_THRESHOLD  # ≥150% for breakdown short
        
        # --- Execution Guards ---
        # Check for immediate reversal candle (5–15m) against the trade
        # This will be checked in each strategy individually
        
        logger.info("")
        logger.info("🔒 Execution Guards:")
        logger.info(f"   • Volume confirmation required: ≥{BREAKOUT_LONG_VOLUME_THRESHOLD:.0%}× 20-period vol for breakout long, ≥{BREAKDOWN_SHORT_VOLUME_THRESHOLD:.0%}× for breakdown short")
        logger.info("   • Use acceptance rules (multiple closes) rather than wick pokes")
        logger.info("   • Skip trades if the volume conditions aren't met — no exceptions")
        logger.info("   • If TP1 hits, move SL to breakeven; partials at TP1 (~50–60%) keep expectancy rational")
        
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
        
        # Get previous 15m candle for consecutive close analysis
        prev_15m_close = float(get_candle_value(candles_15m[1], 'close'))
        
        # 1. LONG - "Breakout/hold above today's high" Strategy
        if long_strategies_enabled and not trigger_state.get("breakout_long_triggered", False) and long_attempts < 2:
            # Check for Breakout LONG conditions
            # Entry: $118,400–$118,700 after a 15–60m close above $118,300 (clears today's high)
            breakout_long_trigger_condition = current_15m_close > BREAKOUT_LONG_TRIGGER
            breakout_long_entry_condition = current_price >= BREAKOUT_LONG_ENTRY_LOW and current_price <= BREAKOUT_LONG_ENTRY_HIGH
            breakout_long_volume_condition = volume_confirmed_breakout_long
            
            breakout_long_ready = breakout_long_trigger_condition and breakout_long_entry_condition and breakout_long_volume_condition
            
            logger.info("")
            logger.info("🔍 LONG - Breakout/hold above today's high Analysis:")
            logger.info(f"   • 15–60m close above ${BREAKOUT_LONG_TRIGGER:,}: {'✅' if breakout_long_trigger_condition else '❌'} (current: ${current_15m_close:,.0f})")
            logger.info(f"   • Entry zone ${BREAKOUT_LONG_ENTRY_LOW:,}–${BREAKOUT_LONG_ENTRY_HIGH:,}: {'✅' if breakout_long_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume ≥ {BREAKOUT_LONG_VOLUME_THRESHOLD:.0%}× 20-period avg: {'✅' if breakout_long_volume_condition else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Breakout LONG Ready: {'🎯 YES' if breakout_long_ready else '⏳ NO'}")
            
            if breakout_long_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout/hold above today's high conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakout LONG trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Clean Conditional Setups - Breakout Long",
                    entry_price=current_price,
                    stop_loss=BREAKOUT_LONG_STOP_LOSS,
                    take_profit=BREAKOUT_LONG_TP1,  # Use TP1 as primary target
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
        
        # 2. LONG - "Liquidity sweep & reclaim" Strategy
        if long_strategies_enabled and not trade_executed and not trigger_state.get("sweep_long_triggered", False) and long_attempts < 2:
            # Check for sweep and reclaim conditions
            # Entry: On a fast flush into $116,900–$117,100, then reclaim $117,200 with a 15m close
            sweep_low_reached = trigger_state.get("sweep_low_reached", False)
            
            # Check if sweep low was reached
            if not sweep_low_reached and current_15m_low <= SWEEP_LONG_ENTRY_HIGH:
                trigger_state["sweep_low_reached"] = True
                save_trigger_state(trigger_state)
                logger.info(f"✅ Sweep low reached: ${current_15m_low:,.0f} (target: ${SWEEP_LONG_ENTRY_HIGH:,})")
            
            # Check for reclaim condition
            sweep_reclaim_condition = trigger_state.get("sweep_low_reached", False) and current_15m_close > SWEEP_LONG_RECLAIM
            sweep_entry_condition = current_price >= SWEEP_LONG_ENTRY_LOW and current_price <= SWEEP_LONG_ENTRY_HIGH
            
            sweep_long_ready = sweep_reclaim_condition and sweep_entry_condition
            
            logger.info("")
            logger.info("🔍 LONG - Liquidity sweep & reclaim Analysis:")
            logger.info(f"   • Sweep low reached: {'✅' if trigger_state.get('sweep_low_reached', False) else '❌'} (low: ${current_15m_low:,.0f})")
            logger.info(f"   • 15m close above ${SWEEP_LONG_RECLAIM:,}: {'✅' if sweep_reclaim_condition else '❌'} (current: ${current_15m_close:,.0f})")
            logger.info(f"   • Entry zone ${SWEEP_LONG_ENTRY_LOW:,}–${SWEEP_LONG_ENTRY_HIGH:,}: {'✅' if sweep_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Sweep & Reclaim LONG Ready: {'🎯 YES' if sweep_long_ready else '⏳ NO'}")
            
            if sweep_long_ready:
                logger.info("")
                logger.info("🎯 LONG - Liquidity sweep & reclaim conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Sweep & Reclaim LONG trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Clean Conditional Setups - Sweep & Reclaim Long",
                    entry_price=current_price,
                    stop_loss=SWEEP_LONG_STOP_LOSS,
                    take_profit=SWEEP_LONG_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Sweep & Reclaim LONG trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["sweep_long_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Sweep & Reclaim LONG trade failed: {trade_result}")
        
        # 3. SHORT - "Breakdown & acceptance" Strategy
        if short_strategies_enabled and not trade_executed and not trigger_state.get("breakdown_short_triggered", False) and short_attempts < 2:
            # Check for Breakdown SHORT conditions
            # Entry: $116,950–$117,150 after two 15m closes below $117,200 (today's low lost + VWAP rejection)
            
            # Check for consecutive closes below trigger
            breakdown_closes_count = trigger_state.get("breakdown_closes_count", 0)
            
            if current_15m_close < BREAKDOWN_SHORT_TRIGGER:
                breakdown_closes_count += 1
                trigger_state["breakdown_closes_count"] = breakdown_closes_count
                save_trigger_state(trigger_state)
                logger.info(f"✅ Breakdown close #{breakdown_closes_count}: ${current_15m_close:,.0f} < ${BREAKDOWN_SHORT_TRIGGER:,}")
            else:
                # Reset counter if price moves above trigger
                if breakdown_closes_count > 0:
                    trigger_state["breakdown_closes_count"] = 0
                    save_trigger_state(trigger_state)
                    logger.info(f"🔄 Breakdown closes reset: price ${current_15m_close:,.0f} > ${BREAKDOWN_SHORT_TRIGGER:,}")
            
            breakdown_trigger_condition = breakdown_closes_count >= 2  # Two consecutive closes below trigger
            breakdown_entry_condition = current_price >= BREAKDOWN_SHORT_ENTRY_LOW and current_price <= BREAKDOWN_SHORT_ENTRY_HIGH
            breakdown_volume_condition = volume_confirmed_breakdown_short
            
            breakdown_short_ready = breakdown_trigger_condition and breakdown_entry_condition and breakdown_volume_condition
            
            logger.info("")
            logger.info("🔍 SHORT - Breakdown & acceptance Analysis:")
            logger.info(f"   • Two 15m closes below ${BREAKDOWN_SHORT_TRIGGER:,}: {'✅' if breakdown_trigger_condition else '❌'} (count: {breakdown_closes_count}/2)")
            logger.info(f"   • Entry zone ${BREAKDOWN_SHORT_ENTRY_LOW:,}–${BREAKDOWN_SHORT_ENTRY_HIGH:,}: {'✅' if breakdown_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume ≥ {BREAKDOWN_SHORT_VOLUME_THRESHOLD:.0%}× 20-period avg: {'✅' if breakdown_volume_condition else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Breakdown SHORT Ready: {'🎯 YES' if breakdown_short_ready else '⏳ NO'}")
            
            if breakdown_short_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown & acceptance conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown SHORT trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Clean Conditional Setups - Breakdown Short",
                    entry_price=current_price,
                    stop_loss=BREAKDOWN_SHORT_STOP_LOSS,
                    take_profit=BREAKDOWN_SHORT_TP1,  # Use TP1 as primary target
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakdown SHORT trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakdown_short_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_1h, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown SHORT trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout LONG triggered: {trigger_state.get('breakout_long_triggered', False)}")
            logger.info(f"Sweep LONG triggered: {trigger_state.get('sweep_long_triggered', False)}")
            logger.info(f"Breakdown SHORT triggered: {trigger_state.get('breakdown_short_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
            logger.info(f"Sweep low reached: {trigger_state.get('sweep_low_reached', False)}")
            logger.info(f"Breakdown closes count: {trigger_state.get('breakdown_closes_count', 0)}")
        
        logger.info("=== Spiros — Clean Conditional BTC Setups completed ===")
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
    logger.info("  • Breakout LONG: 15–60m close above $118,300 AND vol ≥ 125% of 20-period avg")
    logger.info("  • Sweep & Reclaim LONG: Fast flush into $116,900–$117,100, then 15m close above $117,200")
    logger.info("  • Breakdown SHORT: Two 15m closes below $117,200 AND vol ≥ 150% of 20-period avg")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20 leverage)")
    logger.info("  • Volume confirm: ≥125% of 20-period vol for breakout long, ≥150% for breakdown short")
    logger.info("  • Max 2 attempts per side; use acceptance rules (multiple closes) rather than wick pokes")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting Spiros — Clean Conditional BTC Setups Alert Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: Clean Conditional BTC Setups - LONG & SHORT")
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