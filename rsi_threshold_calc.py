import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

def fetch_historical_data(product_id: str = 'BTC-USDC', days: int = 30):
    """Fetch historical 5-minute candle data for the specified product."""
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    
    # Calculate time range
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=days)
    
    # Fetch historical data
    candles = cb.historical_data.get_historical_data(
        product_id=product_id,
        start_date=start_time,
        end_date=end_time,
        granularity="FIVE_MINUTE"
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(candles)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'start']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['start'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    return df

def calculate_rsi_threshold(df: pd.DataFrame, period: int = 14, percentile: float = 10.0, floor: float = 25.0):
    """Calculate RSI threshold based on historical data."""
    # Calculate RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=period)
    
    # Calculate percentile threshold
    raw_threshold = np.percentile(df['rsi'].dropna(), percentile)
    
    # Apply floor if specified
    floored_threshold = max(raw_threshold, floor)
    
    return raw_threshold, floored_threshold

def main():
    # Fetch historical data
    print("Fetching historical data...")
    df = fetch_historical_data()
    
    # Calculate RSI threshold
    print("Calculating RSI threshold...")
    raw_threshold, floored_threshold = calculate_rsi_threshold(df)
    
    # Print results
    print(f"\nDynamic RSI Threshold (10th percentile): {raw_threshold:.2f}")
    print(f"Floored RSI Threshold (min 25): {floored_threshold:.2f}")

if __name__ == "__main__":
    main() 