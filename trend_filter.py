import pandas as pd
import numpy as np
import talib
from typing import Optional, Tuple, Dict, Union

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average for a given period.
    
    Args:
        data: Price series (pandas Series)
        period: EMA period
        
    Returns:
        Series containing EMA values
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """
    Calculate Average True Range (ATR) for a given period.
    
    Args:
        high: High prices (pandas Series)
        low: Low prices (pandas Series)
        close: Close prices (pandas Series)
        period: ATR period
        
    Returns:
        Current ATR value
    """
    atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
    return atr[-1]

def classify_market_regime(
    df: pd.DataFrame,
    ema_short_period: int = 50,
    ema_long_period: int = 200,
    atr_period: int = 14,
    volatility_threshold: float = 0.005,
    use_volatility_filter: bool = True
) -> str:
    """
    Classify the current market regime as 'Bullish', 'Bearish', or 'Sideways'
    based on EMA crossovers and price position relative to EMAs.
    
    Args:
        df: DataFrame with columns ['timestamp', 'close', 'high', 'low']
        ema_short_period: Period for short-term EMA (default: 50)
        ema_long_period: Period for long-term EMA (default: 200)
        atr_period: Period for ATR calculation (default: 14)
        volatility_threshold: Minimum ATR percentage to confirm trend (default: 0.5%)
        use_volatility_filter: Whether to apply volatility filter (default: True)
        
    Returns:
        String indicating market regime: 'Bullish', 'Bearish', or 'Sideways'
    """
    # Ensure we have enough data
    if len(df) < max(ema_short_period, ema_long_period, atr_period):
        return "Sideways"  # Default to sideways if not enough data
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Calculate EMAs
    df_copy['ema_short'] = calculate_ema(df_copy['close'], ema_short_period)
    df_copy['ema_long'] = calculate_ema(df_copy['close'], ema_long_period)
    
    # Get current values
    current_price = df_copy['close'].iloc[-1]
    current_ema_short = df_copy['ema_short'].iloc[-1]
    current_ema_long = df_copy['ema_long'].iloc[-1]
    
    # Calculate ATR if volatility filter is enabled
    volatility_ok = True
    if use_volatility_filter:
        atr = calculate_atr(df_copy['high'], df_copy['low'], df_copy['close'], atr_period)
        atr_percent = atr / current_price
        volatility_ok = atr_percent >= volatility_threshold
    
    # Classify market regime
    if current_ema_short > current_ema_long and current_price > current_ema_short:
        if use_volatility_filter and not volatility_ok:
            return "Sideways"  # Not enough volatility to confirm trend
        return "Bullish"
    elif current_ema_short < current_ema_long and current_price < current_ema_short:
        if use_volatility_filter and not volatility_ok:
            return "Sideways"  # Not enough volatility to confirm trend
        return "Bearish"
    else:
        return "Sideways"

def get_trend_filter(
    df: pd.DataFrame,
    ema_short_period: int = 50,
    ema_long_period: int = 200,
    atr_period: int = 14,
    volatility_threshold: float = 0.005,
    use_volatility_filter: bool = True
) -> Dict[str, Union[str, float]]:
    """
    Get trend filter results including market regime and key metrics.
    
    Args:
        df: DataFrame with columns ['timestamp', 'close', 'high', 'low']
        ema_short_period: Period for short-term EMA (default: 50)
        ema_long_period: Period for long-term EMA (default: 200)
        atr_period: Period for ATR calculation (default: 14)
        volatility_threshold: Minimum ATR percentage to confirm trend (default: 0.5%)
        use_volatility_filter: Whether to apply volatility filter (default: True)
        
    Returns:
        Dictionary with market regime and key metrics
    """
    # Ensure we have enough data
    if len(df) < max(ema_short_period, ema_long_period, atr_period):
        return {
            "regime": "Sideways",
            "ema_short": None,
            "ema_long": None,
            "atr_percent": None,
            "volatility_ok": False
        }
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Calculate EMAs
    df_copy['ema_short'] = calculate_ema(df_copy['close'], ema_short_period)
    df_copy['ema_long'] = calculate_ema(df_copy['close'], ema_long_period)
    
    # Get current values
    current_price = df_copy['close'].iloc[-1]
    current_ema_short = df_copy['ema_short'].iloc[-1]
    current_ema_long = df_copy['ema_long'].iloc[-1]
    
    # Calculate ATR
    atr = calculate_atr(df_copy['high'], df_copy['low'], df_copy['close'], atr_period)
    atr_percent = atr / current_price
    volatility_ok = atr_percent >= volatility_threshold if use_volatility_filter else True
    
    # Determine regime
    if current_ema_short > current_ema_long and current_price > current_ema_short:
        regime = "Bullish" if volatility_ok else "Sideways"
    elif current_ema_short < current_ema_long and current_price < current_ema_short:
        regime = "Bearish" if volatility_ok else "Sideways"
    else:
        regime = "Sideways"
    
    return {
        "regime": regime,
        "ema_short": current_ema_short,
        "ema_long": current_ema_long,
        "atr_percent": atr_percent,
        "volatility_ok": volatility_ok
    }

# Example usage in a backtest loop
def example_backtest_usage():
    """
    Example of how to use the trend filter in a backtest loop.
    """
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # For demonstration, create sample data
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'timestamp': dates,
        'close': np.random.normal(100, 10, 300).cumsum() + 1000,
        'high': np.random.normal(101, 10, 300).cumsum() + 1000,
        'low': np.random.normal(99, 10, 300).cumsum() + 1000
    })
    
    # Example backtest loop
    for i in range(200, len(df)):
        # Get data up to current point - create a copy to avoid SettingWithCopyWarning
        current_data = df.iloc[:i+1].copy()
        
        # Get trend filter results
        trend_info = get_trend_filter(
            current_data,
            ema_short_period=50,
            ema_long_period=200,
            atr_period=14,
            volatility_threshold=0.005,
            use_volatility_filter=True
        )
        
        # Use the trend information in your trading logic
        regime = trend_info["regime"]
        print(f"Date: {current_data['timestamp'].iloc[-1]}, Regime: {regime}")
        
        # Example trading logic based on regime
        if regime == "Bullish":
            # Implement bullish strategy
            pass
        elif regime == "Bearish":
            # Implement bearish strategy
            pass
        else:
            # Implement sideways strategy
            pass

if __name__ == "__main__":
    example_backtest_usage() 