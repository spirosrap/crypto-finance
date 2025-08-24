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

# Constants for BTC Intraday Trading Setup
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global rules
MARGIN = 250  # USD
LEVERAGE = 20  # Always margin x leverage = 250 x 20 = $5,000 position size
RISK_PERCENTAGE = 0.5

# Session snapshot (for reporting only) - New intraday context
# Spot ~$114,951. 24h range: $114,583–$115,952. Key levels: $114,560 support, $115,995–$116,034 resistance
TWENTY_FOUR_HOUR_LOW = 114583
TWENTY_FOUR_HOUR_HIGH = 115952
CURRENT_SPOT = 114951
SUPPORT_LEVEL = 114560
RESISTANCE_LOW = 115995
RESISTANCE_HIGH = 116034
WIDER_S1 = 113303
WIDER_R1 = 119035

# 1) Long breakout - New Strategy
LONG_BREAKOUT_TRIGGER_LEVEL = 116050     # 5-min close > $116,050 then hold above
LONG_BREAKOUT_ENTRY = 116100             # Entry: $116,100
LONG_BREAKOUT_STOP_LOSS = 115650         # Stop: $115,650
LONG_BREAKOUT_TP1 = 117000               # TP1: $117,000
LONG_BREAKOUT_TP2 = 118800               # TP2: $118,800–$119,000
LONG_BREAKOUT_RVOL_THRESHOLD = 2.0       # 5-min vol ≥ 2× 20-SMA

# 2) Long bounce - New Strategy  
LONG_BOUNCE_LOW = 114600                 # Wick hold at $114,600–$114,750
LONG_BOUNCE_HIGH = 114750
LONG_BOUNCE_ENTRY = 114680               # Entry: $114,680±
LONG_BOUNCE_STOP_LOSS = 114450           # Stop: $114,450
LONG_BOUNCE_TP1 = 115400                 # TP1: $115,400
LONG_BOUNCE_TP2 = 116000                 # TP2: $116,000
LONG_BOUNCE_VOLUME_RULE = "5m vol uptick, ΔOI not dumping"

# 3) Short breakdown - New Strategy
SHORT_BREAKDOWN_TRIGGER_LEVEL = 114500   # 5-min close < $114,500 then hold below
SHORT_BREAKDOWN_ENTRY = 114450           # Entry: $114,450
SHORT_BREAKDOWN_STOP_LOSS = 114900       # Stop: $114,900
SHORT_BREAKDOWN_TP1 = 113700             # TP1: $113,700
SHORT_BREAKDOWN_TP2 = 113300             # TP2: $113,300
SHORT_BREAKDOWN_RVOL_THRESHOLD = 2.0     # 5-min vol ≥ 2× 20-SMA

# 4) Short rejection - New Strategy
SHORT_REJECTION_LOW = 115995             # Fail at $115,995–$116,050
SHORT_REJECTION_HIGH = 116050
SHORT_REJECTION_ENTRY = 115950           # Entry: $115,950
SHORT_REJECTION_STOP_LOSS = 116350       # Stop: $116,350
SHORT_REJECTION_TP1 = 115200             # TP1: $115,200
SHORT_REJECTION_TP2 = 114700             # TP2: $114,700
SHORT_REJECTION_VOLUME_RULE = "Bearish delta/absorption at the top"

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
                "long_breakout_triggered": False,
                "long_bounce_triggered": False,
                "short_breakdown_triggered": False,
                "short_rejection_triggered": False,
                "last_trigger_ts": None,
                "active_trade_direction": None,
                "attempts_per_side": {"LONG": 0, "SHORT": 0},
                "last_breakout_bar_low": None,
                "last_breakdown_bar_high": None
            }
    return {
        "long_breakout_triggered": False,
        "long_bounce_triggered": False,
        "short_breakdown_triggered": False,
        "short_rejection_triggered": False,
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

def calculate_5m_volume_average_today(candles_5m):
    """
    Calculate average 5-minute volume for today's session
    
    Args:
        candles_5m: List of 5-minute candle data
    
    Returns:
        Today's 5-minute volume average
    """
    if len(candles_5m) < 2:
        return 0
    
    volumes = []
    # Calculate from completed candles only (skip current incomplete candle)
    for candle in candles_5m[1:]:  
        if isinstance(candle, dict):
            volume = float(candle.get('volume', 0))
        else:
            volume = float(getattr(candle, 'volume', 0))
        volumes.append(volume)
    
    return sum(volumes) / len(volumes) if volumes else 0

def calculate_rvol_5m(current_volume, volume_sma_20, volume_avg_today):
    """
    Calculate RVOL (Relative Volume) for 5-minute timeframe
    Uses the higher of 20-SMA or today's average as specified in setup
    
    Args:
        current_volume: Current 5-minute candle volume
        volume_sma_20: 20-period SMA of 5-minute volumes
        volume_avg_today: Today's 5-minute volume average
    
    Returns:
        RVOL ratio
    """
    # Use 20-SMA for primary comparison, today's average as secondary reference
    if volume_sma_20 > 0:
        rvol_vs_sma = current_volume / volume_sma_20
    else:
        rvol_vs_sma = 0
        
    if volume_avg_today > 0:
        rvol_vs_today = current_volume / volume_avg_today
    else:
        rvol_vs_today = 0
    
    # Return both for decision making - setup uses "≥1.25× 20-SMA or ≥2× today's 5-min avg"
    return rvol_vs_sma, rvol_vs_today

def get_candle_value(candle, key):
    """Extract value from candle object (handles both dict and object formats)"""
    if isinstance(candle, dict):
        return candle.get(key)
    else:
        return getattr(candle, key, None)





def btc_intraday_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    BTC Trading Setup for Aug 24, 2025 with automated execution and fixed position size (margin 250 × leverage 20 = $5,000):
    
    Long setups:
    1) Long-B/O: 5-min close > $116,050 holds → Entry $116,100; Invalidation $115,650; TP1 $117,000, TP2 $118,800–$119,000
    2) Long-Bounce: Wick hold at $114,600–$114,750 → Entry $114,680±; Invalidation $114,450; TP1 $115,400, TP2 $116,000
    
    Short setups:
    3) Short-B/D: 5-min close < $114,500 holds → Entry $114,450; Invalidation $114,900; TP1 $113,700, TP2 $113,300
    4) Short-Rejection: Fail at $115,995–$116,050 → Entry $115,950; Invalidation $116,350; TP1 $115,200, TP2 $114,700
    
    Volume rules: 5-min vol ≥ 2× 20-SMA for breakouts/breakdowns. Risk ≤0.5–1.0% per idea. One contract set, no add-ons.
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== Spiros — BTC Intraday Setup (LONG & SHORT enabled) ===")
    else:
        logger.info(f"=== Spiros — BTC Intraday Setup ({direction} only) ===")
    
    # Load trigger state
    trigger_state = load_trigger_state()
    
    try:
        # Get current time and calculate time ranges
        current_time = datetime.now(UTC)
        
        # Get 5-minute candles for main analysis (primary timeframe)
        start_5m = current_time - timedelta(hours=8)  # Get 8 hours of 5m data (96 candles)
        end_5m = current_time
        start_ts_5m = int(start_5m.timestamp())
        end_ts_5m = int(end_5m.timestamp())
        
        # Get 15-minute candles for management and confirmation
        start_15m = current_time - timedelta(hours=8)  # Get 8 hours of 15m data (32 candles)
        end_15m = current_time
        start_ts_15m = int(start_15m.timestamp())
        end_ts_15m = int(end_15m.timestamp())
        
        logger.info(f"Fetching 5-minute candles from {start_5m} to {end_5m}")
        candles_5m = safe_get_5m_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts_5m)
        
        logger.info(f"Fetching 15-minute candles from {start_15m} to {end_15m}")
        candles_15m = safe_get_15m_candles(cb_service, PRODUCT_ID, start_ts_15m, end_ts_15m)
        
        if not candles_5m or len(candles_5m) < 20:  # Need at least 20 5m candles for volume SMA
            logger.warning("Not enough 5-minute candle data for analysis")
            return last_alert_ts
        
        if not candles_15m or len(candles_15m) < 10:  # Need at least 10 15m candles for management
            logger.warning("Not enough 15-minute candle data for management analysis")
            return last_alert_ts
        
        # Get current and previous candles (5m primary, 15m for management)
        current_5m = candles_5m[0]    # Most recent 5m candle (may be in progress)
        last_5m = candles_5m[1]       # Last completed 5m candle
        prev_5m = candles_5m[2]       # Previous completed 5m candle
        
        current_15m = candles_15m[0]  # Most recent 15m candle (may be in progress)
        last_15m = candles_15m[1]     # Last completed 15m candle
        
        # Extract values from last completed 5m candle (primary for triggers)
        last_5m_ts = datetime.fromtimestamp(int(get_candle_value(last_5m, 'start')), UTC)
        last_5m_close = float(get_candle_value(last_5m, 'close'))
        last_5m_high = float(get_candle_value(last_5m, 'high'))
        last_5m_low = float(get_candle_value(last_5m, 'low'))
        last_5m_volume = float(get_candle_value(last_5m, 'volume'))
        
        # Extract values from last completed 15m candle (for management)
        last_15m_ts = datetime.fromtimestamp(int(get_candle_value(last_15m, 'start')), UTC)
        last_15m_close = float(get_candle_value(last_15m, 'close'))
        last_15m_high = float(get_candle_value(last_15m, 'high'))
        last_15m_low = float(get_candle_value(last_15m, 'low'))
        last_15m_volume = float(get_candle_value(last_15m, 'volume'))
        
        # Get current price from most recent 5m candle
        current_price = float(get_candle_value(current_5m, 'close'))
        
        # Calculate volume SMAs and RVOL for new setup
        volume_sma_5m = calculate_volume_sma(candles_5m, 20)  # 20-period SMA for 5m
        volume_avg_today_5m = calculate_5m_volume_average_today(candles_5m)  # Today's 5m average
        
        # Calculate RVOL for current 5m candle
        rvol_vs_sma, rvol_vs_today = calculate_rvol_5m(last_5m_volume, volume_sma_5m, volume_avg_today_5m)
        
        # Volume confirmation logic: ≥2× 20-SMA for breakout/breakdown
        volume_confirmed_breakout = rvol_vs_sma >= LONG_BREAKOUT_RVOL_THRESHOLD
        volume_confirmed_breakdown = rvol_vs_sma >= SHORT_BREAKDOWN_RVOL_THRESHOLD
        
        # For bounce setup: 5m vol uptick, ΔOI not dumping (simplified as volume increase)
        volume_confirmed_bounce = last_5m_volume > volume_sma_5m * 1.2  # 20% above SMA
        
        # For rejection setup: Bearish delta/absorption at the top (simplified as lower volume on rejection)
        volume_confirmed_rejection = last_5m_volume < volume_sma_5m * 0.8  # 20% below SMA
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, here are BTC setups for Aug 24, 2025. Spot ~$114,951. 24h range: $114,583–$115,952. Key levels: $114,560 support, $115,995–$116,034 resistance. Wider refs: S1 $113,303, R1 $119,035.")
        logger.info(f"Live: BTC ≈ ${current_price:,.0f}")
        logger.info("")
        logger.info("📊 Rules:")
        logger.info("   • Use stop-market. Risk ≤0.5–1.0% per idea. One contract set, no add-ons.")
        logger.info("   • Confirm with 5m structure + volume; skip if churn near midpoint of range.")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} × {LEVERAGE}x)")
        logger.info("   • Volume confirm: 5-min vol ≥ 2× 20-SMA for breakouts/breakdowns")
        logger.info("   • If 116,050 flips to support with volume, bias long toward 117,000 → 118,800/119,035.")
        logger.info("   • If 114,500 breaks on volume, bias short toward 113,700 → 113,303.")
        logger.info("")
        
        # Show market state
        logger.info("📊 Market State:")
        logger.info(f"   • 24h Range: ${TWENTY_FOUR_HOUR_LOW:,}–${TWENTY_FOUR_HOUR_HIGH:,}")
        logger.info(f"   • Current Price: ${current_price:,.0f}")
        logger.info(f"   • Support: ${SUPPORT_LEVEL:,}, Resistance: ${RESISTANCE_LOW:,}–${RESISTANCE_HIGH:,}")
        logger.info(f"   • Wider Levels: S1 ${WIDER_S1:,}, R1 ${WIDER_R1:,}")
        logger.info(f"   • RVOL Analysis: {rvol_vs_sma:.2f}× vs 20-SMA, {rvol_vs_today:.2f}× vs today avg")
        logger.info("   • Levels reference BTC/USD; align to Coinbase PERP mark before execution")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1. Long-B/O")
            logger.info(f"   • Trigger: 5-min close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,} holds")
            logger.info(f"   • Entry: ${LONG_BREAKOUT_ENTRY:,}")
            logger.info(f"   • Invalidation: ${LONG_BREAKOUT_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${LONG_BREAKOUT_TP1:,}, TP2: ${LONG_BREAKOUT_TP2:,}–${WIDER_R1:,}")
            logger.info(f"   • Volume rule: 5m vol ≥ {LONG_BREAKOUT_RVOL_THRESHOLD}× 20-SMA")
            logger.info("")
            logger.info("2. Long-Bounce")
            logger.info(f"   • Trigger: Wick hold at ${LONG_BOUNCE_LOW:,}–${LONG_BOUNCE_HIGH:,}")
            logger.info(f"   • Entry: ${LONG_BOUNCE_ENTRY:,}±")
            logger.info(f"   • Invalidation: ${LONG_BOUNCE_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${LONG_BOUNCE_TP1:,}, TP2: ${LONG_BOUNCE_TP2:,}")
            logger.info(f"   • Volume rule: {LONG_BOUNCE_VOLUME_RULE}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("1. Short-B/D")
            logger.info(f"   • Trigger: 5-min close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,} holds")
            logger.info(f"   • Entry: ${SHORT_BREAKDOWN_ENTRY:,}")
            logger.info(f"   • Invalidation: ${SHORT_BREAKDOWN_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${SHORT_BREAKDOWN_TP1:,}, TP2: ${SHORT_BREAKDOWN_TP2:,}")
            logger.info(f"   • Volume rule: 5m vol ≥ {SHORT_BREAKDOWN_RVOL_THRESHOLD}× 20-SMA")
            logger.info("")
            logger.info("2. Short-Rejection")
            logger.info(f"   • Trigger: Fail at ${SHORT_REJECTION_LOW:,}–${SHORT_REJECTION_HIGH:,}")
            logger.info(f"   • Entry: ${SHORT_REJECTION_ENTRY:,}")
            logger.info(f"   • Invalidation: ${SHORT_REJECTION_STOP_LOSS:,}")
            logger.info(f"   • TP1: ${SHORT_REJECTION_TP1:,}, TP2: ${SHORT_REJECTION_TP2:,}")
            logger.info(f"   • Volume rule: {SHORT_REJECTION_VOLUME_RULE}")
            logger.info("")
        
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 5M Close: ${last_5m_close:,.2f}, High: ${last_5m_high:,.2f}, Low: ${last_5m_low:,.2f}")
        logger.info(f"5M Volume: {last_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}")
        logger.info(f"Last 15M Close: ${last_15m_close:,.2f}, High: ${last_15m_high:,.2f}, Low: ${last_15m_low:,.2f}")
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
        
        # 1) Long Breakout - New Strategy
        if (long_strategies_enabled and 
            not trigger_state.get("long_breakout_triggered", False) and long_attempts < 2):
            
            # Check if 5-min close > trigger level and holding above
            breakout_trigger_condition = last_5m_close > LONG_BREAKOUT_TRIGGER_LEVEL
            # Check if current price is in entry zone
            breakout_entry_condition = (current_price >= LONG_BREAKOUT_ENTRY and 
                                      current_price <= LONG_BREAKOUT_ENTRY + 50) # Allow +/- 50 for entry
            # Volume confirmation: 5-min RVOL ≥1.25× 20-SMA or ≥2× today's 5-min avg
            breakout_volume_condition = volume_confirmed_breakout
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and breakout_volume_condition

            logger.info("🔍 LONG - Breakout Analysis:")
            logger.info(f"   • 5-min close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,}: {'✅' if breakout_trigger_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • Entry zone ${LONG_BREAKOUT_ENTRY:,}–${LONG_BREAKOUT_ENTRY + 50:,}: {'✅' if breakout_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume confirm (5m vol ≥ {LONG_BREAKOUT_RVOL_THRESHOLD}× 20-SMA): {'✅' if breakout_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Long Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Long Breakout",
                    entry_price=current_price,
                    stop_loss=LONG_BREAKOUT_STOP_LOSS,
                    take_profit=LONG_BREAKOUT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Long Breakout trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["long_breakout_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    # Save the breakout bar low for invalidation tracking
                    trigger_state["last_breakout_bar_low"] = float(get_candle_value(last_5m, 'low'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Breakout trade failed: {trade_result}")
        
        # 2) Long Bounce - New Strategy
        if (long_strategies_enabled and not trade_executed and
            not trigger_state.get("long_bounce_triggered", False) and long_attempts < 2):
            
            # Check if we've had a wick hold at the bounce zone
            wick_hold_condition = (last_5m_low >= LONG_BOUNCE_LOW and 
                                 last_5m_low <= LONG_BOUNCE_HIGH and
                                 last_5m_close > last_5m_low)  # Closed above the wick
            # Check if current price is near entry zone
            bounce_entry_condition = abs(current_price - LONG_BOUNCE_ENTRY) <= 50  # Allow ±50 for entry
            # Volume confirmation: 5m vol uptick, ΔOI not dumping
            bounce_volume_condition = volume_confirmed_bounce
            
            bounce_ready = wick_hold_condition and bounce_entry_condition and bounce_volume_condition

            logger.info("")
            logger.info("🔍 LONG - Bounce Analysis:")
            logger.info(f"   • Wick hold at ${LONG_BOUNCE_LOW:,}–${LONG_BOUNCE_HIGH:,}: {'✅' if wick_hold_condition else '❌'} (last 5m low: ${last_5m_low:,.0f}, close: ${last_5m_close:,.0f})")
            logger.info(f"   • Entry near ${LONG_BOUNCE_ENTRY:,}±50: {'✅' if bounce_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume confirm (5m vol uptick): {'✅' if bounce_volume_condition else '❌'} (volume: {last_5m_volume:,.0f} vs SMA: {volume_sma_5m:,.0f})")
            logger.info(f"   • Long Bounce Ready: {'🎯 YES' if bounce_ready else '⏳ NO'}")

            if bounce_ready:
                logger.info("")
                logger.info("🎯 LONG - Bounce conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Long Bounce",
                    entry_price=current_price,
                    stop_loss=LONG_BOUNCE_STOP_LOSS,
                    take_profit=LONG_BOUNCE_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Long Bounce trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["long_bounce_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Bounce trade failed: {trade_result}")
        
        # 3) Short Breakdown - New Strategy
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("short_breakdown_triggered", False) and short_attempts < 2):
            
            # Check if 5-min close < trigger level then fail retest
            breakdown_trigger_condition = last_5m_close < SHORT_BREAKDOWN_TRIGGER_LEVEL
            # Check if current price is in entry zone for retest failure
            breakdown_entry_condition = (current_price >= SHORT_BREAKDOWN_ENTRY and 
                                       current_price <= SHORT_BREAKDOWN_ENTRY + 50) # Allow +/- 50 for entry
            # Volume confirmation: 5-min RVOL ≥1.25× 20-SMA or ≥2× 5-min avg
            breakdown_volume_condition = volume_confirmed_breakdown
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and breakdown_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Breakdown Analysis:")
            logger.info(f"   • 5-min close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,}: {'✅' if breakdown_trigger_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • Entry zone ${SHORT_BREAKDOWN_ENTRY:,}–${SHORT_BREAKDOWN_ENTRY + 50:,}: {'✅' if breakdown_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume confirm (5m vol ≥ {SHORT_BREAKDOWN_RVOL_THRESHOLD}× 20-SMA): {'✅' if breakdown_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Short Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Short Breakdown",
                    entry_price=current_price,
                    stop_loss=SHORT_BREAKDOWN_STOP_LOSS,
                    take_profit=SHORT_BREAKDOWN_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Short Breakdown trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["short_breakdown_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    # Save the breakdown bar high for invalidation tracking
                    trigger_state["last_breakdown_bar_high"] = float(get_candle_value(last_5m, 'high'))
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Breakdown trade failed: {trade_result}")
        
        # 4) Short Rejection - New Strategy  
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("short_rejection_triggered", False) and short_attempts < 2):
            
            # Check for failure at resistance zone
            failure_in_zone = (last_5m_high >= SHORT_REJECTION_LOW and 
                             last_5m_high <= SHORT_REJECTION_HIGH and
                             last_5m_close < last_5m_high * 0.995)  # Failed to hold above resistance
            # Check if current price is near entry zone
            rejection_entry_condition = abs(current_price - SHORT_REJECTION_ENTRY) <= 50  # Allow ±50 for entry
            # Volume confirmation: Bearish delta/absorption at the top
            rejection_volume_condition = volume_confirmed_rejection
            
            rejection_ready = failure_in_zone and rejection_entry_condition and rejection_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Rejection Analysis:")
            logger.info(f"   • Fail at ${SHORT_REJECTION_LOW:,}–${SHORT_REJECTION_HIGH:,}: {'✅' if failure_in_zone else '❌'} (last 5m high: ${last_5m_high:,.0f}, close: ${last_5m_close:,.0f})")
            logger.info(f"   • Entry near ${SHORT_REJECTION_ENTRY:,}±50: {'✅' if rejection_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume confirm (Bearish delta/absorption): {'✅' if rejection_volume_condition else '❌'} (volume: {last_5m_volume:,.0f} vs SMA: {volume_sma_5m:,.0f})")
            logger.info(f"   • Short Rejection Ready: {'🎯 YES' if rejection_ready else '⏳ NO'}")

            if rejection_ready:
                logger.info("")
                logger.info("🎯 SHORT - Rejection conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Short Rejection",
                    entry_price=current_price,
                    stop_loss=SHORT_REJECTION_STOP_LOSS,
                    take_profit=SHORT_REJECTION_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Short Rejection trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["short_rejection_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Rejection trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Long Breakout triggered: {trigger_state.get('long_breakout_triggered', False)}")
            logger.info(f"Long Bounce triggered: {trigger_state.get('long_bounce_triggered', False)}")
            logger.info(f"Short Breakdown triggered: {trigger_state.get('short_breakdown_triggered', False)}")
            logger.info(f"Short Rejection triggered: {trigger_state.get('short_rejection_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
        
        logger.info("=== Spiros — BTC Intraday setup completed ===")
        return last_5m_ts if trade_executed else last_alert_ts
        
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
    logger.info(f"  • Long-B/O: 5-min close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,} holds → Entry ${LONG_BREAKOUT_ENTRY:,}; Invalidation ${LONG_BREAKOUT_STOP_LOSS:,}; TP1 ${LONG_BREAKOUT_TP1:,}, TP2 ${LONG_BREAKOUT_TP2:,}–${WIDER_R1:,}")
    logger.info(f"  • Long-Bounce: Wick hold at ${LONG_BOUNCE_LOW:,}–${LONG_BOUNCE_HIGH:,} → Entry ${LONG_BOUNCE_ENTRY:,}±; Invalidation ${LONG_BOUNCE_STOP_LOSS:,}; TP1 ${LONG_BOUNCE_TP1:,}, TP2 ${LONG_BOUNCE_TP2:,}")
    logger.info("SHORT SETUPS:")
    logger.info(f"  • Short-B/D: 5-min close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,} holds → Entry ${SHORT_BREAKDOWN_ENTRY:,}; Invalidation ${SHORT_BREAKDOWN_STOP_LOSS:,}; TP1 ${SHORT_BREAKDOWN_TP1:,}, TP2 ${SHORT_BREAKDOWN_TP2:,}")
    logger.info(f"  • Short-Rejection: Fail at ${SHORT_REJECTION_LOW:,}–${SHORT_REJECTION_HIGH:,} → Entry ${SHORT_REJECTION_ENTRY:,}; Invalidation ${SHORT_REJECTION_STOP_LOSS:,}; TP1 ${SHORT_REJECTION_TP1:,}, TP2 ${SHORT_REJECTION_TP2:,}")
    logger.info(f"  • Position Size: ${MARGIN * LEVERAGE:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("  • Use stop-market. Risk ≤0.5–1.0% per idea. One contract set, no add-ons.")
    logger.info("  • Volume confirm: 5-min vol ≥ 2× 20-SMA for breakouts/breakdowns")
    logger.info("  • If 116,050 flips to support with volume, bias long toward 117,000 → 118,800/119,035.")
    logger.info("  • If 114,500 breaks on volume, bias short toward 113,700 → 113,303.")
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