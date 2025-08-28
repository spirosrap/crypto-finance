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
import csv

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
# Updated for current BTC setup from Spiros
TWENTY_FOUR_HOUR_LOW = 110465  # Day low: $110,465
TWENTY_FOUR_HOUR_HIGH = 113079  # Day high: $113,079
CURRENT_SPOT = 113034  # Current spot ≈ $113,034

# Reference levels from yesterday OHLC pivots
PIVOT = 111430  # Pivot: $111,430
R1 = 112500     # R1: $112,500
S1 = 110170     # S1: $110,170
R2 = 113760     # R2: $113,760
S2 = 109100     # S2: $109,100

# 1) Long breakout - New Strategy from Spiros
LONG_BREAKOUT_TRIGGER_LEVEL = 113100     # 5-min close > $113,100
LONG_BREAKOUT_ENTRY_LOW = 113100         # Entry zone low: $113,100
LONG_BREAKOUT_ENTRY_HIGH = 113200        # Entry zone high: $113,200
LONG_BREAKOUT_ENTRY = 113150             # Entry: $113,150 (middle of entry zone)
LONG_BREAKOUT_STOP_LOSS = 112110         # Stop: $112,110
LONG_BREAKOUT_TP1 = 114410               # TP1: $114,410
LONG_BREAKOUT_TP2 = 115710               # TP2: $115,710
LONG_BREAKOUT_RVOL_THRESHOLD = 1.5       # Volume confirmation threshold (RVOL ≥1.5× vs 20-SMA vol)

# 2) Short breakdown - New Strategy from Spiros
SHORT_BREAKDOWN_TRIGGER_LEVEL = 110500   # 5-min close < $110,500
SHORT_BREAKDOWN_ENTRY_LOW = 110500       # Entry zone low: $110,500
SHORT_BREAKDOWN_ENTRY_HIGH = 110400      # Entry zone high: $110,400
SHORT_BREAKDOWN_ENTRY = 110450           # Entry: $110,450 (middle of entry zone)
SHORT_BREAKDOWN_STOP_LOSS = 111490       # Stop: $111,490
SHORT_BREAKDOWN_TP1 = 109190             # TP1: $109,190
SHORT_BREAKDOWN_TP2 = 107890             # TP2: $107,890
SHORT_BREAKDOWN_RVOL_THRESHOLD = 1.5     # Volume confirmation threshold (RVOL ≥1.5×)

# 3) Mean-revert scalp (rejection near R2) - New Strategy from Spiros
MEAN_REVERT_TRIGGER_LOW = 113650         # Short trigger low: $113,650
MEAN_REVERT_TRIGGER_HIGH = 113750        # Short trigger high: $113,750
MEAN_REVERT_ENTRY = 113700               # Entry: $113,700 (middle of trigger zone)
MEAN_REVERT_STOP_LOSS = 114050           # Stop: $114,050
MEAN_REVERT_TP1 = 111770                 # TP1: $111,770 (mid)
MEAN_REVERT_TP2 = 111430                 # TP2: $111,430 (P)
MEAN_REVERT_RVOL_THRESHOLD = 0.8         # Weak RVOL condition (< 1.0× for mean reversion)

# Risk rules
MAX_RISK_PER_PROBE = 0.5                 # ≤0.5R per probe
MAX_PROBES_PER_SIDE = 2                  # max 2 probes/side
INVALIDATION_TIME_MINUTES = 15           # Invalidate if price re-enters prior range within 15 minutes

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
                "short_breakdown_triggered": False,
                "mean_revert_triggered": False,
                "last_trigger_ts": None,
                "active_trade_direction": None,
                "attempts_per_side": {"LONG": 0, "SHORT": 0},
                "last_breakout_bar_low": None,
                "last_breakdown_bar_high": None,
                "breakout_invalidation_time": None,
                "breakdown_invalidation_time": None,
                "prior_range_low": None,
                "prior_range_high": None
            }
    return {
        "long_breakout_triggered": False,
        "short_breakdown_triggered": False,
        "mean_revert_triggered": False,
        "last_trigger_ts": None,
        "active_trade_direction": None,
        "attempts_per_side": {"LONG": 0, "SHORT": 0},
        "last_breakout_bar_low": None,
        "last_breakdown_bar_high": None,
        "breakout_invalidation_time": None,
        "breakdown_invalidation_time": None,
        "prior_range_low": None,
        "prior_range_high": None
    }

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def log_trade_to_csv(trade_data):
    """
    Log trade details to CSV file
    
    Args:
        trade_data: Dictionary containing trade information
    """
    csv_file = "chatgpt_trades.csv"
    
    # Define CSV headers
    headers = [
        'timestamp', 'strategy', 'symbol', 'side', 'entry_price', 'stop_loss', 
        'take_profit', 'position_size_usd', 'margin', 'leverage', 'volume_sma', 
        'volume_ratio', 'current_price', 'market_conditions', 'trade_status', 
        'execution_time', 'notes'
    ]
    
    try:
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            # Write headers if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write trade data
            writer.writerow(trade_data)
            
        logger.info(f"✅ Trade logged to {csv_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log trade to CSV: {e}")

def test_csv_logging():
    """
    Test function to verify CSV logging is working correctly
    """
    logger.info("🧪 Testing CSV logging functionality...")
    
    # Test BTC trade data
    btc_trade_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-Long-Breakout',
        'symbol': 'BTC-PERP-INTX',
        'side': 'BUY',
        'entry_price': 112501.0,
        'stop_loss': 112107.0,
        'take_profit': 112683.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 1500.0,
        'volume_ratio': 1.5,
        'current_price': 112501.0,
        'market_conditions': '24h Range: $111,000-$113,000',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - BTC Long Breakout (Aug 25, 2025)'
    }
    
    # Test ETH trade data
    eth_trade_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-Short-Breakdown',
        'symbol': 'BTC-PERP-INTX',
        'side': 'SELL',
        'entry_price': 112143.0,
        'stop_loss': 112534.0,
        'take_profit': 111961.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 1200.0,
        'volume_ratio': 1.3,
        'current_price': 112143.0,
        'market_conditions': '24h Range: $111,000-$113,000',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - BTC Short Breakdown (Aug 25, 2025)'
    }
    
    # Log test trades
    log_trade_to_csv(btc_trade_data)
    log_trade_to_csv(eth_trade_data)
    
    logger.info("✅ CSV logging test completed!")
    logger.info("📊 Check chatgpt_trades.csv to verify test trades were added correctly")

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
    BTC Trading Setup for today with automated execution and fixed position size (margin 250 × leverage 20 = $5,000):
    
    Setup	Trigger (5-min close + retest)	Entry zone	Stop	TP1	TP2	Confirmations
    Breakout LONG	> 113,100	113,100–113,200	112,110	114,410	115,710	RVOL≥1.5× vs 20-SMA vol, no bearish div on 5-min
    Breakdown SHORT	< 110,500	110,500–110,400	111,490	109,190	107,890	RVOL≥1.5×, no bullish div on 5-min
    
    Mean-revert scalp only if rejection near R2 113,760 with weak RVOL: 
    short 113,650–113,750 → TP 111,770 (mid) → 111,430 (P), SL 114,050.
    
    Notes:
    • Invalidate a breakout/breakdown if price re-enters the prior range within 15 minutes. 
    • Target partial at TP1, trail to prior swing lows/highs for TP2.
    
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
        last_5m_open = float(get_candle_value(last_5m, 'open'))
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
        logger.info("🚀 Spiros—here are clean intraday BTC setups for today.")
        logger.info(f"Current ~${current_price:,.0f}. Day range ${TWENTY_FOUR_HOUR_LOW:,}–${TWENTY_FOUR_HOUR_HIGH:,} so far.")
        logger.info("")
        logger.info("📊 Rules:")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} × {LEVERAGE}x)")
        logger.info("   • Trade only on confirmation, not limits")
        logger.info("   • Invalidate if price re-enters prior range within 15 minutes")
        logger.info("")
        
        # Show market state
        logger.info("📊 Market State:")
        logger.info(f"   • Day Range: ${TWENTY_FOUR_HOUR_LOW:,}–${TWENTY_FOUR_HOUR_HIGH:,}")
        logger.info(f"   • Current Price: ${current_price:,.0f}")
        logger.info(f"   • Reference Levels: P=${PIVOT:,} • R1=${R1:,} • S1=${S1:,} • R2=${R2:,} • S2=${S2:,}")
        logger.info(f"   • RVOL Analysis: {rvol_vs_sma:.2f}× vs 20-SMA, {rvol_vs_today:.2f}× vs today avg")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1. Long breakout")
            logger.info(f"   • Trigger: 5-min close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,}")
            logger.info(f"   • Entry zone: ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}")
            logger.info(f"   • Stop: ${LONG_BREAKOUT_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${LONG_BREAKOUT_TP1:,} / ${LONG_BREAKOUT_TP2:,}")
            logger.info(f"   • Confirmation: RVOL≥1.5× vs 20-SMA vol, no bearish div on 5-min")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("1. Short breakdown")
            logger.info(f"   • Trigger: 5-min close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,}")
            logger.info(f"   • Entry zone: ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,}")
            logger.info(f"   • Stop: ${SHORT_BREAKDOWN_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${SHORT_BREAKDOWN_TP1:,} / ${SHORT_BREAKDOWN_TP2:,}")
            logger.info(f"   • Confirmation: RVOL≥1.5×, no bullish div on 5-min")
            logger.info("")
            logger.info("2. Mean-revert scalp (rejection near R2)")
            logger.info(f"   • Trigger: ${MEAN_REVERT_TRIGGER_LOW:,}–${MEAN_REVERT_TRIGGER_HIGH:,} rejection")
            logger.info(f"   • Entry: ${MEAN_REVERT_ENTRY:,}")
            logger.info(f"   • Stop: ${MEAN_REVERT_STOP_LOSS:,}")
            logger.info(f"   • TP1 / TP2: ${MEAN_REVERT_TP1:,} / ${MEAN_REVERT_TP2:,}")
            logger.info(f"   • Confirmation: Weak RVOL (< 1.0×) for mean reversion")
            logger.info("")
        
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 5M Close: ${last_5m_close:,.2f}, High: ${last_5m_high:,.2f}, Low: ${last_5m_low:,.2f}")
        logger.info(f"5M Volume: {last_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}")
        logger.info("")
        logger.info("Notes: Invalidate a breakout/breakdown if price re-enters the prior range within 15 minutes. Target partial at TP1, trail to prior swing lows/highs for TP2.")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check attempts per side (max 2 attempts per side)
        long_attempts = trigger_state.get("attempts_per_side", {}).get("LONG", 0)
        short_attempts = trigger_state.get("attempts_per_side", {}).get("SHORT", 0)
        
        logger.info("🔒 Trade attempts status:")
        logger.info(f"   • LONG attempts: {long_attempts}/{MAX_PROBES_PER_SIDE}")
        logger.info(f"   • SHORT attempts: {short_attempts}/{MAX_PROBES_PER_SIDE}")
        logger.info("")
        
        # Check for invalidation conditions (price re-enters prior range within 15 minutes)
        current_time = datetime.now(UTC)
        
        # Check if we need to invalidate any active trades
        if trigger_state.get("breakout_invalidation_time"):
            invalidation_time = datetime.fromtimestamp(trigger_state["breakout_invalidation_time"], UTC)
            if current_time > invalidation_time:
                logger.info("⚠️ Breakout invalidation time expired - clearing breakout trigger")
                trigger_state["long_breakout_triggered"] = False
                trigger_state["breakout_invalidation_time"] = None
                save_trigger_state(trigger_state)
        
        if trigger_state.get("breakdown_invalidation_time"):
            invalidation_time = datetime.fromtimestamp(trigger_state["breakdown_invalidation_time"], UTC)
            if current_time > invalidation_time:
                logger.info("⚠️ Breakdown invalidation time expired - clearing breakdown trigger")
                trigger_state["short_breakdown_triggered"] = False
                trigger_state["breakdown_invalidation_time"] = None
                save_trigger_state(trigger_state)
        
        # 1) Long breakout - New Strategy from Spiros
        if (long_strategies_enabled and 
            not trigger_state.get("long_breakout_triggered", False) and long_attempts < MAX_PROBES_PER_SIDE):
            
            # Check if 5-min close > $113,100
            breakout_trigger_condition = last_5m_close > LONG_BREAKOUT_TRIGGER_LEVEL
            # Check if current price is in entry zone (113,100–113,200)
            breakout_entry_condition = LONG_BREAKOUT_ENTRY_LOW <= current_price <= LONG_BREAKOUT_ENTRY_HIGH
            # Volume confirmation: RVOL ≥1.5× vs 20-SMA vol
            breakout_volume_condition = rvol_vs_sma >= LONG_BREAKOUT_RVOL_THRESHOLD
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and breakout_volume_condition

            logger.info("🔍 LONG - Breakout Analysis:")
            logger.info(f"   • 5-min close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,}: {'✅' if breakout_trigger_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • Entry in zone ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume confirm (RVOL ≥ {LONG_BREAKOUT_RVOL_THRESHOLD}× 20-SMA): {'✅' if breakout_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
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
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Long-Breakout',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': current_price,
                        'stop_loss': LONG_BREAKOUT_STOP_LOSS,
                        'take_profit': LONG_BREAKOUT_TP1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"Day Range: ${TWENTY_FOUR_HOUR_LOW:,}-${TWENTY_FOUR_HOUR_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 5m close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,}, Volume: {rvol_vs_sma:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Set invalidation time (15 minutes from now)
                    invalidation_time = current_time + timedelta(minutes=INVALIDATION_TIME_MINUTES)
                    trigger_state["long_breakout_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    trigger_state["breakout_invalidation_time"] = int(invalidation_time.timestamp())
                    trigger_state["prior_range_low"] = TWENTY_FOUR_HOUR_LOW
                    trigger_state["prior_range_high"] = LONG_BREAKOUT_TRIGGER_LEVEL
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Breakout trade failed: {trade_result}")
        

        
        # 2) Short breakdown - New Strategy from Spiros
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("short_breakdown_triggered", False) and short_attempts < MAX_PROBES_PER_SIDE):
            
            # Check if 5-min close < $110,500
            breakdown_trigger_condition = last_5m_close < SHORT_BREAKDOWN_TRIGGER_LEVEL
            # Check if current price is in entry zone (110,500–110,400)
            breakdown_entry_condition = SHORT_BREAKDOWN_ENTRY_HIGH <= current_price <= SHORT_BREAKDOWN_ENTRY_LOW
            # Volume confirmation: RVOL ≥1.5×
            breakdown_volume_condition = rvol_vs_sma >= SHORT_BREAKDOWN_RVOL_THRESHOLD
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and breakdown_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Breakdown Analysis:")
            logger.info(f"   • 5-min close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,}: {'✅' if breakdown_trigger_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • Entry in zone ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,}: {'✅' if breakdown_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Volume confirm (RVOL ≥ {SHORT_BREAKDOWN_RVOL_THRESHOLD}× 20-SMA): {'✅' if breakdown_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
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
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Short-Breakdown',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': SHORT_BREAKDOWN_STOP_LOSS,
                        'take_profit': SHORT_BREAKDOWN_TP1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"Day Range: ${TWENTY_FOUR_HOUR_LOW:,}-${TWENTY_FOUR_HOUR_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 5m close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,}, Volume: {rvol_vs_sma:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Set invalidation time (15 minutes from now)
                    invalidation_time = current_time + timedelta(minutes=INVALIDATION_TIME_MINUTES)
                    trigger_state["short_breakdown_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    trigger_state["breakdown_invalidation_time"] = int(invalidation_time.timestamp())
                    trigger_state["prior_range_low"] = SHORT_BREAKDOWN_TRIGGER_LEVEL
                    trigger_state["prior_range_high"] = TWENTY_FOUR_HOUR_HIGH
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Breakdown trade failed: {trade_result}")
        
        # 3) Mean-revert scalp (rejection near R2) - New Strategy from Spiros
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("mean_revert_triggered", False) and short_attempts < MAX_PROBES_PER_SIDE):
            
            # Check for rejection near R2 (113,760) with weak RVOL
            r2_rejection_condition = MEAN_REVERT_TRIGGER_LOW <= current_price <= MEAN_REVERT_TRIGGER_HIGH
            # Check for weak RVOL condition (< 1.0× for mean reversion)
            weak_volume_condition = rvol_vs_sma < MEAN_REVERT_RVOL_THRESHOLD
            # Check if current price is near entry
            mean_revert_entry_condition = abs(current_price - MEAN_REVERT_ENTRY) <= 50  # Allow ±50 for entry
            
            mean_revert_ready = r2_rejection_condition and weak_volume_condition and mean_revert_entry_condition

            logger.info("")
            logger.info("🔍 SHORT - Mean-Revert Scalp Analysis:")
            logger.info(f"   • Rejection near R2 ${MEAN_REVERT_TRIGGER_LOW:,}-${MEAN_REVERT_TRIGGER_HIGH:,}: {'✅' if r2_rejection_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Weak RVOL (< {MEAN_REVERT_RVOL_THRESHOLD}×): {'✅' if weak_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Entry near ${MEAN_REVERT_ENTRY:,}±50: {'✅' if mean_revert_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Mean-Revert Ready: {'🎯 YES' if mean_revert_ready else '⏳ NO'}")

            if mean_revert_ready:
                logger.info("")
                logger.info("🎯 SHORT - Mean-Revert conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Mean-Revert Scalp",
                    entry_price=current_price,
                    stop_loss=MEAN_REVERT_STOP_LOSS,
                    take_profit=MEAN_REVERT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Mean-Revert Scalp trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Mean-Revert-Scalp',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': MEAN_REVERT_STOP_LOSS,
                        'take_profit': MEAN_REVERT_TP1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"Day Range: ${TWENTY_FOUR_HOUR_LOW:,}-${TWENTY_FOUR_HOUR_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Rejection near R2 ${R2:,}, weak RVOL: {rvol_vs_sma:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    trigger_state["mean_revert_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Mean-Revert Scalp trade failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Long Breakout triggered: {trigger_state.get('long_breakout_triggered', False)}")
            logger.info(f"Short Breakdown triggered: {trigger_state.get('short_breakdown_triggered', False)}")
            logger.info(f"Mean-Revert triggered: {trigger_state.get('mean_revert_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
            logger.info(f"Breakout invalidation time: {trigger_state.get('breakout_invalidation_time', 'None')}")
            logger.info(f"Breakdown invalidation time: {trigger_state.get('breakdown_invalidation_time', 'None')}")
        
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
    parser.add_argument('--test-csv', action='store_true', help='Test CSV logging functionality')
    args = parser.parse_args()
    
    # Test CSV logging if requested
    if args.test_csv:
        test_csv_logging()
        return
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    logger.info("BTC Intraday Strategy Overview (Current Setup):")
    logger.info("LONG SETUPS:")
    logger.info(f"  • Long Breakout: 5-min close > ${LONG_BREAKOUT_TRIGGER_LEVEL:,} → Entry zone ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}; Stop ${LONG_BREAKOUT_STOP_LOSS:,}; Targets ${LONG_BREAKOUT_TP1:,} / ${LONG_BREAKOUT_TP2:,}")
    logger.info("SHORT SETUPS:")
    logger.info(f"  • Short Breakdown: 5-min close < ${SHORT_BREAKDOWN_TRIGGER_LEVEL:,} → Entry zone ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,}; Stop ${SHORT_BREAKDOWN_STOP_LOSS:,}; Targets ${SHORT_BREAKDOWN_TP1:,} / ${SHORT_BREAKDOWN_TP2:,}")
    logger.info(f"  • Mean-Revert Scalp: Rejection near R2 ${R2:,} with weak RVOL → Entry ${MEAN_REVERT_ENTRY:,}; Stop ${MEAN_REVERT_STOP_LOSS:,}; Targets ${MEAN_REVERT_TP1:,} / ${MEAN_REVERT_TP2:,}")
    logger.info(f"  • Position Size: ${MARGIN * LEVERAGE:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("  • Trade only on confirmation, not limits")
    logger.info("  • Invalidate if price re-enters prior range within 15 minutes")
    logger.info("  • RVOL≥1.5× vs 20-SMA vol for breakout/breakdown; weak RVOL for mean reversion")
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