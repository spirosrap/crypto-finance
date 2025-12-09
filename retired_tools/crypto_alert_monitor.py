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

def safe_get_1m_candles(cb_service, product_id, start_ts, end_ts):
    """
    Safely get 1-minute candles with retry logic
    """
    def _get_1m_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity="ONE_MINUTE"
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_1m_candles)

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
# Updated for Aug 31, 2025 BTC setup from Spiros
TWENTY_FOUR_HOUR_LOW = 108100  # Day low: $108,100
TWENTY_FOUR_HOUR_HIGH = 109500  # Day high: $109,500
CURRENT_SPOT = 108800  # Current spot ≈ $108,800

# Reference levels from 24h range
ID_MID = (TWENTY_FOUR_HOUR_HIGH + TWENTY_FOUR_HOUR_LOW) / 2  # ID mid = (IDH+IDL)/2

# Strategy thresholds and filters
ATR_PERCENT_MIN = 0.002  # ATR% < 0.2% minimum for trading
RVOL_BREAKOUT_THRESHOLD = 1.5  # RVOL5 ≥ 1.5 for breakout
RVOL_REJECTION_THRESHOLD = 1.0  # RVOL5 falling for rejection
RVOL_DIP_BUY_THRESHOLD = 1.2  # RVOL5 ≥ 1.2 for dip buy
RVOL_BREAKDOWN_THRESHOLD = 1.7  # RVOL5 ≥ 1.7 for breakdown

# Risk rules
MAX_RISK_PER_PROBE = 0.5                 # ≤0.5R per probe
MAX_PROBES_PER_SIDE = 2                  # max 2 probes/side
INVALIDATION_TIME_MINUTES = 15           # Invalidate if price re-enters prior range within 15 minutes

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
                "breakout_above_h_triggered": False,
                "rejection_at_1095k_triggered": False,
                "dip_buy_1083k_triggered": False,
                "breakdown_below_l_triggered": False,
                "last_trigger_ts": None,
                "active_trade_direction": None,
                "attempts_per_side": {"LONG": 0, "SHORT": 0},
                "breakout_high": None,
                "rejection_lower_high": None,
                "dip_buy_low": None,
                "breakdown_low": None,
                "idh": None,
                "idl": None,
                "atr5": None
            }
    return {
        "breakout_above_h_triggered": False,
        "rejection_at_1095k_triggered": False,
        "dip_buy_1083k_triggered": False,
        "breakdown_below_l_triggered": False,
        "last_trigger_ts": None,
        "active_trade_direction": None,
        "attempts_per_side": {"LONG": 0, "SHORT": 0},
        "breakout_high": None,
        "rejection_lower_high": None,
        "dip_buy_low": None,
        "breakdown_low": None,
        "idh": None,
        "idl": None,
        "atr5": None
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

def calculate_atr(candles, period=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        candles: List of candle data
        period: Period for ATR calculation
    
    Returns:
        ATR value
    """
    if len(candles) < period + 1:
        return 0
    
    true_ranges = []
    for i in range(1, period + 1):  # Skip current incomplete candle
        candle = candles[i]
        prev_candle = candles[i + 1] if i + 1 < len(candles) else candles[i]
        
        high = float(get_candle_value(candle, 'high'))
        low = float(get_candle_value(candle, 'low'))
        prev_close = float(get_candle_value(prev_candle, 'close'))
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    return sum(true_ranges) / len(true_ranges) if true_ranges else 0

def calculate_vwap(candles):
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        candles: List of candle data
    
    Returns:
        VWAP value
    """
    if len(candles) < 2:
        return 0
    
    cumulative_pv = 0
    cumulative_volume = 0
    
    # Calculate from completed candles only (skip current incomplete candle)
    for candle in candles[1:]:
        high = float(get_candle_value(candle, 'high'))
        low = float(get_candle_value(candle, 'low'))
        close = float(get_candle_value(candle, 'close'))
        volume = float(get_candle_value(candle, 'volume'))
        
        typical_price = (high + low + close) / 3
        cumulative_pv += typical_price * volume
        cumulative_volume += volume
    
    return cumulative_pv / cumulative_volume if cumulative_volume > 0 else 0

def calculate_ema(candles, period=20):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        candles: List of candle data
        period: Period for EMA calculation
    
    Returns:
        EMA value
    """
    if len(candles) < period:
        return 0
    
    # Use close prices for EMA calculation
    closes = []
    for candle in candles[1:period+1]:  # Skip current incomplete candle
        close = float(get_candle_value(candle, 'close'))
        closes.append(close)
    
    if not closes:
        return 0
    
    # Calculate EMA
    multiplier = 2 / (period + 1)
    ema = closes[0]  # Start with first value
    
    for close in closes[1:]:
        ema = (close * multiplier) + (ema * (1 - multiplier))
    
    return ema

def get_intraday_high_low(candles):
    """
    Get today's intraday high and low
    
    Args:
        candles: List of candle data
    
    Returns:
        Tuple of (intraday_high, intraday_low)
    """
    if len(candles) < 2:
        return TWENTY_FOUR_HOUR_HIGH, TWENTY_FOUR_HOUR_LOW
    
    intraday_high = float('-inf')
    intraday_low = float('inf')
    
    # Check completed candles only (skip current incomplete candle)
    for candle in candles[1:]:
        high = float(get_candle_value(candle, 'high'))
        low = float(get_candle_value(candle, 'low'))
        
        intraday_high = max(intraday_high, high)
        intraday_low = min(intraday_low, low)
    
    # Use default values if no data
    if intraday_high == float('-inf'):
        intraday_high = TWENTY_FOUR_HOUR_HIGH
    if intraday_low == float('inf'):
        intraday_low = TWENTY_FOUR_HOUR_LOW
    
    return intraday_high, intraday_low

def check_bear_engulfing(candles):
    """
    Check for bearish engulfing pattern
    
    Args:
        candles: List of candle data (need at least 2 candles)
    
    Returns:
        True if bearish engulfing pattern detected
    """
    if len(candles) < 3:
        return False
    
    current = candles[1]  # Last completed candle
    previous = candles[2]  # Previous completed candle
    
    current_open = float(get_candle_value(current, 'open'))
    current_close = float(get_candle_value(current, 'close'))
    current_high = float(get_candle_value(current, 'high'))
    current_low = float(get_candle_value(current, 'low'))
    
    prev_open = float(get_candle_value(previous, 'open'))
    prev_close = float(get_candle_value(previous, 'close'))
    prev_high = float(get_candle_value(previous, 'high'))
    prev_low = float(get_candle_value(previous, 'low'))
    
    # Bearish engulfing: current candle completely engulfs previous bullish candle
    is_bearish_engulfing = (
        current_close < current_open and  # Current candle is bearish
        prev_close > prev_open and       # Previous candle is bullish
        current_open > prev_close and    # Current open above previous close
        current_close < prev_open        # Current close below previous open
    )
    
    return is_bearish_engulfing

def check_bull_reversal(candles):
    """
    Check for bullish reversal pattern (simplified as higher low after lower low)
    
    Args:
        candles: List of candle data (need at least 3 candles)
    
    Returns:
        True if bullish reversal pattern detected
    """
    if len(candles) < 4:
        return False
    
    current = candles[1]   # Last completed candle
    previous = candles[2]  # Previous completed candle
    prev2 = candles[3]     # Two candles ago
    
    current_low = float(get_candle_value(current, 'low'))
    previous_low = float(get_candle_value(previous, 'low'))
    prev2_low = float(get_candle_value(prev2, 'low'))
    
    # Bullish reversal: higher low after lower low
    is_bull_reversal = (
        prev2_low > previous_low and  # Lower low
        previous_low < current_low    # Higher low (reversal)
    )
    
    return is_bull_reversal







def btc_intraday_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    BTC Trading Setup for Aug 31, 2025 with automated execution and fixed position size (margin 250 × leverage 20 = $5,000):
    
    Setup	Side	Trigger (M5)	Entry	Stop	TP1	TP2	Confirm
    Breakout > today's H	Long	Close > 109,500	109,520–109,650	109,150	110,800	112,000	RVOL ≥1.5× 20-SMA; no immediate sell delta spike
    Rejection at 109.5k	Short	Wick + lower high under 109,500	109,450–109,520	109,800	108,700	108,100	RVOL↓ on the LH; tape slows
    Dip buy near low	Long	Seller exhaust at 108,300 zone	108,250–108,350	107,950	109,000	109,450	Delta flips +; RVOL ≥1.2×
    Breakdown < today's L	Short	Close < 108,100	108,050–107,950	108,350	107,200	106,400	RVOL ≥1.7×; momentum holds 3 bars
    
    Execution rules:
    • Timeframes: trigger on M5, manage on M1–M5.
    • Invalidate any long if M5 closes back below 109,500 after breakout; invalidate any short if M5 closes back above 109,500 after rejection.
    • Skip longs if funding spikes > +0.02% across top venues; skip shorts if funding < −0.02%.
    
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
        
        # Calculate technical indicators
        volume_sma_5m = calculate_volume_sma(candles_5m, 20)  # 20-period SMA for 5m
        atr5 = calculate_atr(candles_5m, 14)  # 14-period ATR on 5m
        
        # Get intraday high and low
        idh, idl = get_intraday_high_low(candles_5m)
        id_mid = (idh + idl) / 2  # ID mid = (IDH+IDL)/2
        
        # Calculate RVOL for current 5m candle
        rvol_vs_sma, rvol_vs_today = calculate_rvol_5m(last_5m_volume, volume_sma_5m, 0)
        
        # Update trigger state with current levels
        trigger_state["idh"] = idh
        trigger_state["idl"] = idl
        trigger_state["atr5"] = atr5
        
        # Check ATR filter (no trade if ATR5/close < 0.002)
        atr_percent = atr5 / current_price if current_price > 0 else 0
        atr_filter_passed = atr_percent >= ATR_PERCENT_MIN
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros—here are actionable BTC plays for Aug 31, 2025.")
        logger.info(f"Current ~${current_price:,.0f}. Day range ${idl:,.0f}–${idh:,.0f}.")
        logger.info("")
        logger.info("📊 Rules:")
        logger.info(f"   • Position Size: ${MARGIN * LEVERAGE:,.0f} USD (${MARGIN} × {LEVERAGE}x)")
        logger.info("   • Trade only on confirmation, not limits")
        logger.info("   • ATR Filter: ATR% ≥ 0.2% (current: {:.3f}%)".format(atr_percent * 100))
        logger.info("")
        
        # Show market state
        logger.info("📊 Market State:")
        logger.info(f"   • IDH: ${idh:,.0f}, IDL: ${idl:,.0f}, ID Mid: ${id_mid:,.0f}")
        logger.info(f"   • ATR5: ${atr5:,.0f} ({atr_percent*100:.2f}%)")
        logger.info(f"   • RVOL5: {rvol_vs_sma:.2f}× vs 5m SMA")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1. Breakout > today's H")
            logger.info(f"   • Trigger: M5 close > ${idh:,.0f} and RVOL5 ≥ {RVOL_BREAKOUT_THRESHOLD}")
            logger.info(f"   • Entry: ${idh+20:,.0f}–${idh+150:,.0f}")
            logger.info(f"   • Stop: ${idh-350:,.0f}")
            logger.info(f"   • Targets: TP1 = ${idh+1300:,.0f}, TP2 = ${idh+2500:,.0f}")
            logger.info("")
            logger.info("2. Dip buy near low")
            logger.info(f"   • Trigger: Seller exhaust at ${idl+200:,.0f} zone")
            logger.info(f"   • Entry: ${idl+150:,.0f}–${idl+250:,.0f}")
            logger.info(f"   • Stop: ${idl-150:,.0f}")
            logger.info(f"   • Targets: TP1 = ${idl+900:,.0f}, TP2 = ${idl+1350:,.0f}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("1. Rejection at 109.5k")
            logger.info(f"   • Trigger: Wick + lower high under ${idh:,.0f}")
            logger.info(f"   • Entry: ${idh-50:,.0f}–${idh+20:,.0f}")
            logger.info(f"   • Stop: ${idh+300:,.0f}")
            logger.info(f"   • Targets: TP1 = ${idh-800:,.0f}, TP2 = ${idh-1400:,.0f}")
            logger.info("")
            logger.info("2. Breakdown < today's L")
            logger.info(f"   • Trigger: M5 close < ${idl:,.0f}")
            logger.info(f"   • Entry: ${idl-50:,.0f}–${idl-150:,.0f}")
            logger.info(f"   • Stop: ${idl+250:,.0f}")
            logger.info(f"   • Targets: TP1 = ${idl-900:,.0f}, TP2 = ${idl-1700:,.0f}")
            logger.info("")
        
        logger.info(f"Current Price: ${current_price:,.2f}")
        logger.info(f"Last 5M Close: ${last_5m_close:,.2f}, High: ${last_5m_high:,.2f}, Low: ${last_5m_low:,.2f}")
        logger.info(f"5M Volume: {last_5m_volume:,.0f}, 5M SMA: {volume_sma_5m:,.0f}")
        logger.info("")
        logger.info("Filters: No trade if ATR% < 0.2%. Invalidate longs if M5 closes back below 109,500 after breakout; invalidate shorts if M5 closes back above 109,500 after rejection.")
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
        
        # Check ATR filter first
        if not atr_filter_passed:
            logger.info("⚠️ ATR filter not passed - no trades allowed")
            logger.info(f"   • ATR%: {atr_percent*100:.3f}% (minimum: {ATR_PERCENT_MIN*100:.1f}%)")
            return last_alert_ts
        
        # 1) Breakout > today's H - Long Strategy
        if (long_strategies_enabled and 
            not trigger_state.get("breakout_above_h_triggered", False) and long_attempts < MAX_PROBES_PER_SIDE):
            
            # Check if M5 close > 109,500 and RVOL5 ≥ 1.5
            breakout_trigger_condition = last_5m_close > 109500
            breakout_volume_condition = rvol_vs_sma >= RVOL_BREAKOUT_THRESHOLD
            
            # Check if current price is in entry zone (109,520–109,650)
            entry_zone_low = 109520
            entry_zone_high = 109650
            breakout_entry_condition = entry_zone_low <= current_price <= entry_zone_high
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and breakout_volume_condition

            logger.info("🔍 LONG - Breakout > today's H Analysis:")
            logger.info(f"   • M5 close > $109,500: {'✅' if breakout_trigger_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • RVOL5 ≥ {RVOL_BREAKOUT_THRESHOLD}: {'✅' if breakout_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Entry in zone ${entry_zone_low:,.0f}–${entry_zone_high:,.0f}: {'✅' if breakout_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Breakout > today's H Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout > today's H conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate stop loss and take profits
                stop_loss = 109150
                tp1 = 110800
                tp2 = 112000

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Breakout > today's H",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakout > today's H trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Breakout-AboveH',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"IDH: ${idh:,.0f}, IDL: ${idl:,.0f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"M5 close >109500; avoid if sell delta spike, RVOL: {rvol_vs_sma:.2f}x"
                    }
                    log_trade_to_csv(trade_data)
                    
                    trigger_state["breakout_above_h_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout > today's H trade failed: {trade_result}")
        
        # 2) Rejection at 109.5k - Short Strategy
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("rejection_at_1095k_triggered", False) and short_attempts < MAX_PROBES_PER_SIDE):
            
            # Check for wick + lower high under 109,500
            rejection_wick_condition = last_5m_high > 109500  # Wick above 109.5k
            rejection_lower_high_condition = last_5m_close < 109500  # Lower high under 109.5k
            rejection_volume_condition = rvol_vs_sma < RVOL_REJECTION_THRESHOLD  # RVOL falling
            
            # Check if current price is in entry zone (109,450–109,520)
            entry_zone_low = 109450
            entry_zone_high = 109520
            rejection_entry_condition = entry_zone_low <= current_price <= entry_zone_high
            
            rejection_ready = rejection_wick_condition and rejection_lower_high_condition and rejection_entry_condition and rejection_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Rejection at 109.5k Analysis:")
            logger.info(f"   • Wick above $109,500: {'✅' if rejection_wick_condition else '❌'} (last 5m high: ${last_5m_high:,.0f})")
            logger.info(f"   • Lower high under 109.5k: {'✅' if rejection_lower_high_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • RVOL falling: {'✅' if rejection_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Entry in zone ${entry_zone_low:,.0f}–${entry_zone_high:,.0f}: {'✅' if rejection_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Rejection at 109.5k Ready: {'🎯 YES' if rejection_ready else '⏳ NO'}")

            if rejection_ready:
                logger.info("")
                logger.info("🎯 SHORT - Rejection at 109.5k conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate stop loss and take profits
                stop_loss = 109800
                tp1 = 108700
                tp2 = 108100

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Rejection at 109.5k",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Rejection at 109.5k trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Range-Reject-1095',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"IDH: ${idh:,.0f}, IDL: ${idl:,.0f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Lower high under 109500, RVOL: {rvol_vs_sma:.2f}x"
                    }
                    log_trade_to_csv(trade_data)
                    
                    trigger_state["rejection_at_1095k_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    trigger_state["rejection_lower_high"] = last_5m_high
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Rejection at 109.5k trade failed: {trade_result}")
        
        # 3) Dip buy near low - Long Strategy
        if (long_strategies_enabled and not trade_executed and
            not trigger_state.get("dip_buy_1083k_triggered", False) and long_attempts < MAX_PROBES_PER_SIDE):
            
            # Check for seller exhaust at 108,300 zone
            dip_buy_exhaust_condition = last_5m_low <= 108300  # Seller exhaust at 108.3k zone
            dip_buy_volume_condition = rvol_vs_sma >= RVOL_DIP_BUY_THRESHOLD  # RVOL ≥ 1.2×
            
            # Check if current price is in entry zone (108,250–108,350)
            entry_zone_low = 108250
            entry_zone_high = 108350
            dip_buy_entry_condition = entry_zone_low <= current_price <= entry_zone_high
            
            dip_buy_ready = dip_buy_exhaust_condition and dip_buy_entry_condition and dip_buy_volume_condition

            logger.info("")
            logger.info("🔍 LONG - Dip buy near low Analysis:")
            logger.info(f"   • Seller exhaust at 108.3k zone: {'✅' if dip_buy_exhaust_condition else '❌'} (last 5m low: ${last_5m_low:,.0f})")
            logger.info(f"   • RVOL5 ≥ {RVOL_DIP_BUY_THRESHOLD}: {'✅' if dip_buy_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Entry in zone ${entry_zone_low:,.0f}–${entry_zone_high:,.0f}: {'✅' if dip_buy_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Dip buy near low Ready: {'🎯 YES' if dip_buy_ready else '⏳ NO'}")

            if dip_buy_ready:
                logger.info("")
                logger.info("🎯 LONG - Dip buy near low conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate stop loss and take profits
                stop_loss = 107950
                tp1 = 109000
                tp2 = 109450

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Dip buy near low",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Dip buy near low trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Dip-Buy-1083',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"IDH: ${idh:,.0f}, IDL: ${idl:,.0f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Buyer absorption at 108.3k, RVOL: {rvol_vs_sma:.2f}x"
                    }
                    log_trade_to_csv(trade_data)
                    
                    trigger_state["dip_buy_1083k_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["LONG"] = long_attempts + 1
                    trigger_state["dip_buy_low"] = last_5m_low
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Dip buy near low trade failed: {trade_result}")
        
        # 4) Breakdown < today's L - Short Strategy
        if (short_strategies_enabled and not trade_executed and
            not trigger_state.get("breakdown_below_l_triggered", False) and short_attempts < MAX_PROBES_PER_SIDE):
            
            # Check for M5 close < 108,100 and RVOL5 ≥ 1.7
            breakdown_trigger_condition = last_5m_close < 108100
            breakdown_volume_condition = rvol_vs_sma >= RVOL_BREAKDOWN_THRESHOLD
            
            # Check if current price is in entry zone (108,050–107,950)
            entry_zone_low = 107950
            entry_zone_high = 108050
            breakdown_entry_condition = entry_zone_low <= current_price <= entry_zone_high
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and breakdown_volume_condition

            logger.info("")
            logger.info("🔍 SHORT - Breakdown < today's L Analysis:")
            logger.info(f"   • M5 close < $108,100: {'✅' if breakdown_trigger_condition else '❌'} (last 5m close: ${last_5m_close:,.0f})")
            logger.info(f"   • RVOL5 ≥ {RVOL_BREAKDOWN_THRESHOLD}: {'✅' if breakdown_volume_condition else '❌'} (RVOL: {rvol_vs_sma:.2f}×)")
            logger.info(f"   • Entry in zone ${entry_zone_low:,.0f}–${entry_zone_high:,.0f}: {'✅' if breakdown_entry_condition else '❌'} (current: ${current_price:,.0f})")
            logger.info(f"   • Breakdown < today's L Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown < today's L conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate stop loss and take profits
                stop_loss = 108350
                tp1 = 107200
                tp2 = 106400

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Intraday - Breakdown < today's L",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown < today's L trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'Breakdown-BelowL',
                        'symbol': 'BTC-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': MARGIN * LEVERAGE,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': volume_sma_5m,
                        'volume_ratio': rvol_vs_sma,
                        'current_price': current_price,
                        'market_conditions': f"IDH: ${idh:,.0f}, IDL: ${idl:,.0f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"M5 close <108100; hold momentum, RVOL: {rvol_vs_sma:.2f}x"
                    }
                    log_trade_to_csv(trade_data)
                    
                    trigger_state["breakdown_below_l_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(get_candle_value(last_5m, 'start'))
                    trigger_state["attempts_per_side"]["SHORT"] = short_attempts + 1
                    trigger_state["breakdown_low"] = last_5m_low
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown < today's L trade failed: {trade_result}")
        

        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any strategy")
            logger.info(f"Breakout > today's H triggered: {trigger_state.get('breakout_above_h_triggered', False)}")
            logger.info(f"Rejection at 109.5k triggered: {trigger_state.get('rejection_at_1095k_triggered', False)}")
            logger.info(f"Dip buy near low triggered: {trigger_state.get('dip_buy_1083k_triggered', False)}")
            logger.info(f"Breakdown < today's L triggered: {trigger_state.get('breakdown_below_l_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")
            logger.info(f"Current IDH: ${idh:,.0f}, IDL: ${idl:,.0f}")
            logger.info(f"ATR5: ${atr5:.0f} ({atr_percent*100:.2f}%), RVOL5: {rvol_vs_sma:.2f}×")
        
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
    logger.info("BTC Intraday Strategy Overview (Aug 31, 2025 Setup):")
    logger.info("LONG SETUPS:")
    logger.info(f"  • Breakout > today's H: M5 close > 109,500 and RVOL5 ≥ {RVOL_BREAKOUT_THRESHOLD} → Entry 109,520–109,650; Stop 109,150; Targets TP1 = 110,800, TP2 = 112,000")
    logger.info(f"  • Dip buy near low: Seller exhaust at 108,300 zone and RVOL5 ≥ {RVOL_DIP_BUY_THRESHOLD} → Entry 108,250–108,350; Stop 107,950; Targets TP1 = 109,000, TP2 = 109,450")
    logger.info("SHORT SETUPS:")
    logger.info(f"  • Rejection at 109.5k: Wick + lower high under 109,500 and RVOL falling → Entry 109,450–109,520; Stop 109,800; Targets TP1 = 108,700, TP2 = 108,100")
    logger.info(f"  • Breakdown < today's L: M5 close < 108,100 and RVOL5 ≥ {RVOL_BREAKDOWN_THRESHOLD} → Entry 108,050–107,950; Stop 108,350; Targets TP1 = 107,200, TP2 = 106,400")
    logger.info(f"  • Position Size: ${MARGIN * LEVERAGE:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("  • Trade only on confirmation, not limits")
    logger.info("  • ATR Filter: ATR% ≥ 0.2% minimum for trading")
    logger.info("  • Invalidate longs if M5 closes back below 109,500 after breakout; invalidate shorts if M5 closes back above 109,500 after rejection")
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