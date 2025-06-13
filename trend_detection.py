import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_coinbase_data(product_id: str = 'BTC-USDC', 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch historical data from Coinbase.
    
    Args:
        product_id: Trading pair (e.g., 'BTC-USDC')
        start_date: Optional start date in 'YYYY-MM-DD' format
        end_date: Optional end date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Fetching data for {product_id}")
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    
    if start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=UTC)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=UTC)
    else:
        # Default to last 8000 1-hour candles
        now = datetime.now(UTC)
        end = now
        start = now - timedelta(hours=8000)
    
    logger.info(f"Fetching data from {start} to {end}")
    
    # Initialize empty list to store all candles
    all_candles = []
    current_end = end
    max_retries = 3
    
    # Fetch data in chunks of 350 candles (maximum allowed by API)
    while current_end > start:
        retries = 0
        while retries < max_retries:
            try:
                # Calculate start time for this chunk (350 candles * 1 hour = 350 hours)
                chunk_start = current_end - timedelta(hours=350)
                if chunk_start < start:
                    chunk_start = start
                    
                # Get raw data for this chunk
                response = cb.client.get_public_candles(
                    product_id=product_id,
                    start=int(chunk_start.timestamp()),
                    end=int(current_end.timestamp()),
                    granularity='ONE_HOUR'
                )
                
                # Handle the Coinbase API response
                if hasattr(response, 'candles'):
                    candles = response.candles
                elif isinstance(response, dict) and 'candles' in response:
                    candles = response['candles']
                elif isinstance(response, list):
                    candles = response
                else:
                    logger.error(f"Unexpected API response format: {type(response)}")
                    retries += 1
                    time.sleep(1)
                    continue
                
                if not candles:
                    logger.warning(f"No candles in chunk from {chunk_start} to {current_end}")
                    break
                
                # Convert each candle to a dictionary
                for candle in candles:
                    if hasattr(candle, 'to_dict'):
                        candle_dict = candle.to_dict()
                    elif isinstance(candle, dict):
                        candle_dict = candle
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                        candle_dict = {
                            'start': candle[0],
                            'open': candle[1],
                            'high': candle[2],
                            'low': candle[3],
                            'close': candle[4],
                            'volume': candle[5]
                        }
                    else:
                        logger.error(f"Unexpected candle format: {type(candle)}")
                        continue
                    
                    all_candles.append(candle_dict)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Error fetching data chunk (attempt {retries + 1}/{max_retries}): {e}")
                retries += 1
                if retries < max_retries:
                    time.sleep(2 ** retries)  # Exponential backoff
                continue
        
        if retries == max_retries:
            logger.error(f"Failed to fetch data chunk after {max_retries} attempts")
            break
            
        # Update end time for next chunk
        current_end = chunk_start
        
        # If we've reached the start time, break
        if chunk_start <= start:
            break
    
    if not all_candles:
        logger.error("No candles fetched from API")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Create DataFrame from all candles
    df = pd.DataFrame(all_candles)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp conversion - convert to numeric first to avoid warning
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['start'] = pd.to_datetime(df['start'], unit='s', utc=True)
    df.set_index('start', inplace=True)
    # Ensure DataFrame is sorted by index (datetime)
    df.sort_index(inplace=True)
    
    logger.info(f"Fetched {len(df)} candles")
    # Print the date range of the DataFrame after fetching data
    logger.info(f"Data covers from {df.index.min()} to {df.index.max()}")
    return df

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
        
    Returns:
        Series containing ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        close: Close prices
        period: RSI period (default: 14)
        
    Returns:
        Series containing RSI values
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_clear_downtrend(df: pd.DataFrame) -> Tuple[bool, dict]:
    """
    Detect if there is a clear downtrend in the market.
    
    A clear downtrend is defined as:
    1. Price consistently below 50-period EMA
    2. Lower lows and lower highs over the last 5 bars
    3. ATR (14) > 0.7% of close price (volatility filter)
    
    Args:
        df: DataFrame containing OHLCV data with columns:
            - open, high, low, close, volume
            
    Returns:
        Tuple[bool, dict]: (is_downtrend, metrics)
    """
    df = df.copy()
    # Calculate technical indicators
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    df['ATR%'] = df['ATR'] / df['close'] * 100
    
    # Check if last 5 closes are below EMA50
    ema_check = (df['close'].iloc[-5:] < df['EMA50'].iloc[-5:]).all()
    
    # Check lower lows and lower highs in last 5 bars
    lows = df['low'].iloc[-5:]
    highs = df['high'].iloc[-5:]
    lower_lows = all(lows.iloc[i] < lows.iloc[i-1] for i in range(1, len(lows)))
    lower_highs = all(highs.iloc[i] < highs.iloc[i-1] for i in range(1, len(highs)))
    
    # ATR filter
    atr_check = df['ATR%'].iloc[-1] > 0.7
    
    # Collect metrics
    metrics = {
        'ema_check': ema_check,
        'lower_lows': lower_lows,
        'lower_highs': lower_highs,
        'atr_check': atr_check,
        'current_atr%': df['ATR%'].iloc[-1],
        'current_close': df['close'].iloc[-1],
        'current_ema50': df['EMA50'].iloc[-1]
    }
    
    return ema_check and lower_lows and lower_highs and atr_check, metrics

def detect_oversold_reversal(df: pd.DataFrame) -> Tuple[bool, dict]:
    """
    Detect potential oversold reversal conditions.
    
    An oversold reversal is defined as:
    1. RSI(14) < 25 (stricter than before)
    2. Price at least 0.5% below the lower Bollinger Band (20, 2)
    
    Args:
        df: DataFrame containing OHLCV data with columns:
            - open, high, low, close, volume
            
    Returns:
        Tuple[bool, dict]: (is_oversold, metrics)
    """
    df = df.copy()
    # Calculate technical indicators
    df['RSI'] = calculate_rsi(df['close'])
    
    # Calculate Bollinger Bands
    df['BB_middle'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # Check conditions
    rsi_check = df['RSI'].iloc[-1] < 25  # Stricter RSI threshold
    bb_check = df['close'].iloc[-1] <= df['BB_lower'].iloc[-1] * 0.995  # Price must be at least 0.5% below BB_lower
    
    # Calculate how far price is below BB_lower
    bb_distance = ((df['close'].iloc[-1] - df['BB_lower'].iloc[-1]) / df['BB_lower'].iloc[-1]) * 100
    
    # Collect metrics
    metrics = {
        'rsi_check': rsi_check,
        'bb_check': bb_check,
        'current_rsi': df['RSI'].iloc[-1],
        'current_close': df['close'].iloc[-1],
        'current_bb_lower': df['BB_lower'].iloc[-1],
        'bb_distance_pct': bb_distance  # Added to show how far price is below BB_lower
    }
    
    return rsi_check and bb_check, metrics

def analyze_market_conditions(df: pd.DataFrame) -> Tuple[bool, bool, dict, dict]:
    """
    Analyze market conditions for both downtrend and oversold reversal.
    
    Args:
        df: DataFrame containing OHLCV data with columns:
            - open, high, low, close, volume
            
    Returns:
        Tuple[bool, bool, dict, dict]: (is_downtrend, is_oversold_reversal, downtrend_metrics, oversold_metrics)
    """
    is_downtrend, downtrend_metrics = detect_clear_downtrend(df)
    is_oversold_reversal, oversold_metrics = detect_oversold_reversal(df)
    return is_downtrend, is_oversold_reversal, downtrend_metrics, oversold_metrics

def find_last_downtrend_signal(df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, dict]]:
    """
    Find the last time a clear downtrend was detected in the DataFrame.
    Returns the timestamp and metrics if found, else None.
    """
    min_history = 50  # for EMA50 and ATR
    window_size = 5
    if len(df) < min_history + window_size - 1:
        return None
    last_signal = None
    for i in range(min_history + window_size - 1, len(df)):
        window = df.iloc[:i+1]
        is_downtrend, metrics = detect_clear_downtrend(window)
        if is_downtrend:
            last_signal = (window.index[-1], metrics)
    return last_signal

def find_last_oversold_signal(df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, dict]]:
    """
    Find the last time an oversold reversal was detected in the DataFrame.
    Returns the timestamp and metrics if found, else None.
    """
    min_history = 20  # for Bollinger Bands
    rsi_period = 14
    if len(df) < min_history + rsi_period - 1:
        return None
    last_signal = None
    for i in range(min_history + rsi_period - 1, len(df)):
        window = df.iloc[:i+1]
        is_oversold, metrics = detect_oversold_reversal(window)
        if is_oversold:
            last_signal = (window.index[-1], metrics)
    return last_signal

def find_all_oversold_signals(df: pd.DataFrame) -> list:
    """
    Find all timestamps where an oversold reversal was detected in the DataFrame.
    Returns a list of (timestamp, metrics) tuples.
    """
    min_history = 20  # for Bollinger Bands
    rsi_period = 14
    signals = []
    if len(df) < min_history + rsi_period - 1:
        return signals
    for i in range(min_history + rsi_period - 1, len(df)):
        window = df.iloc[:i+1]
        is_oversold, metrics = detect_oversold_reversal(window)
        if is_oversold:
            signals.append((window.index[-1], metrics))
    return signals

# Example usage:
if __name__ == "__main__":
    # Fetch data from Coinbase for the last two months
    from datetime import datetime, timedelta
    today = datetime.now().date()
    two_months_ago = (today - timedelta(days=60)).strftime('%Y-%m-%d')
    today_str = today.strftime('%Y-%m-%d')
    df = fetch_coinbase_data('BTC-USDC', start_date=two_months_ago, end_date=today_str)
    
    # Analyze market conditions
    is_downtrend, is_oversold_reversal, downtrend_metrics, oversold_metrics = analyze_market_conditions(df)
    
    # Print results
    logger.info("\nMarket Analysis Results:")
    logger.info("=" * 50)
    
    if is_downtrend:
        logger.info("\u2713 Clear downtrend detected")
        logger.info("\nDowntrend Metrics:")
        for key, value in downtrend_metrics.items():
            logger.info(f"{key}: {value}")
    else:
        logger.info("\u2717 No clear downtrend detected")
    
    logger.info("\n" + "=" * 50)
    
    if is_oversold_reversal:
        logger.info("\u2713 Potential oversold reversal detected")
        logger.info("\nOversold Metrics:")
        for key, value in oversold_metrics.items():
            logger.info(f"{key}: {value}")
    else:
        logger.info("\u2717 No oversold reversal detected")
        logger.info(f"Current RSI: {oversold_metrics['current_rsi']:.2f}")

    # Find last time downtrend and oversold reversal conditions were met
    last_downtrend = find_last_downtrend_signal(df)
    last_oversold = find_last_oversold_signal(df)

    logger.info("\n" + "=" * 50)
    if last_downtrend:
        ts, metrics = last_downtrend
        logger.info(f"Last clear downtrend detected at: {ts}")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    else:
        logger.info("No clear downtrend detected in the past.")

    logger.info("\n" + "=" * 50)
    if last_oversold:
        ts, metrics = last_oversold
        logger.info(f"Last oversold reversal detected at: {ts}")
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
    else:
        logger.info("No oversold reversal detected in the past.")

    # Print the last 3 oversold reversal signals for clarity (newest first)
    all_oversold_signals = find_all_oversold_signals(df)
    logger.info("\nLast 3 oversold reversal signals in the period (newest first):")
    if all_oversold_signals:
        for ts, metrics in reversed(all_oversold_signals[-3:]):
            logger.info(f"Oversold at: {ts}")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
    else:
        logger.info("No oversold reversal signals found in the period.")

    # Debug: Print latest RSI, close, and lower Bollinger Band values for last 10 candles
    logger.info("\nLatest 10 candles RSI, close, BB_lower:")
    df_debug = df.copy()
    df_debug['RSI'] = calculate_rsi(df_debug['close'])
    df_debug['BB_middle'] = df_debug['close'].rolling(20).mean()
    df_debug['BB_std'] = df_debug['close'].rolling(20).std()
    df_debug['BB_lower'] = df_debug['BB_middle'] - 2 * df_debug['BB_std']
    for idx, row in df_debug.iloc[-10:].iterrows():
        logger.info(f"{idx}: close={row['close']}, RSI={row['RSI']}, BB_lower={row['BB_lower']}") 