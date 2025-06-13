import pandas as pd
from typing import Tuple
from trend_detection import fetch_coinbase_data, calculate_atr

def check_atr_expansion_coinbase(product_id: str = "BTC-USDC", lookback: int = 5) -> Tuple[bool, float, float]:
    """
    Check if the ATR (Average True Range) is expanding by comparing current value with historical value using Coinbase data.
    
    Args:
        product_id (str): Trading pair symbol (default: "BTC-USDC")
        lookback (int): Number of periods to look back for comparison (default: 5)
    
    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - Boolean indicating if ATR is expanding
            - Current ATR value
            - Historical ATR value (lookback periods ago)
    """
    # Fetch 1-hour OHLC data from Coinbase
    df = fetch_coinbase_data(product_id=product_id)
    if df.empty or not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("Failed to fetch data or missing columns from Coinbase.")
    
    # Calculate 14-period ATR
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # Get current and historical ATR values
    current_atr = df['ATR'].iloc[-1]
    historical_atr = df['ATR'].iloc[-(lookback + 1)]
    
    # Check if ATR is expanding
    is_expanding = current_atr > historical_atr
    
    return is_expanding, current_atr, historical_atr

if __name__ == "__main__":
    # Example usage
    try:
        is_expanding, current_atr, historical_atr = check_atr_expansion_coinbase()
        print(f"BTC/USD ATR Analysis (Coinbase):")
        print(f"Current ATR: {current_atr:.2f}")
        print(f"ATR 5 periods ago: {historical_atr:.2f}")
        print(f"ATR is expanding: {is_expanding}")
    except Exception as e:
        print(f"Error: {e}") 