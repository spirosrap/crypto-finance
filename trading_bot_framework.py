import pandas as pd
import numpy as np
from datetime import datetime, UTC, timedelta
import logging
from typing import Dict, Tuple, Optional, List
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY_PERPS, API_SECRET_PERPS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration dictionary
CONFIG = {
    # Market Regime Parameters
    'ATR_PERIOD': 14,
    'EMA_PERIOD': 20,
    'TREND_SLOPE_PERIOD': 5,
    'TREND_THRESHOLD': 0.001,  # 0.1% per bar
    
    # RSI Parameters
    'RSI_PERIOD': 14,
    'RSI_OVERSOLD': 30,
    'RSI_CONFIRMATION': 35,
    
    # Volume Parameters
    'VOLUME_LOOKBACK': 20,
    'MIN_RELATIVE_VOLUME': 1.2,
    
    # Volatility Parameters
    'MIN_ATR_PERCENT': 0.2,  # 0.2% ATR
    'MAX_ATR_PERCENT': 2.0,  # 2.0% ATR
    
    # Risk Parameters
    'MAX_POSITION_SIZE': 250,  # USD
    'MAX_RISK_PER_TRADE': 1,  # 100% of account
    'LEVERAGE': 5,
    
    # Take Profit & Stop Loss
    'FIXED_TP_PERCENT': 0.015,  # 1.5%
    'FIXED_TP_PERCENT_UNCERTAIN_CHOP': 0.011,  # 1.1%
    'FIXED_SL_PERCENT': 0.007,  # 0.7%
    'ADAPTIVE_TP_MULTIPLIER': 2.5,  # 2.5x ATR
    'ADAPTIVE_SL_MULTIPLIER': 1.5,  # 1.5x ATR
}

def calculate_trend_slope(df: pd.DataFrame) -> float:
    """
    Calculate normalized trend slope over last 5 closes.
    Returns: slope as % change per bar.
    """
    if len(df) < 5:
        return 0.0

    close_prices = df['close'].tail(5).values
    x = np.arange(len(close_prices))
    y = close_prices

    slope, _ = np.polyfit(x, y, 1)

    # Normalize: slope as % of price per bar
    mean_price = np.mean(close_prices)
    normalized_slope = slope / mean_price

    return normalized_slope

def detect_regime(df: pd.DataFrame) -> str:
    """
    Classify market regime using ATR and EMA slope.
    Returns: 'TRENDING', 'CHOP', or 'UNCERTAIN'
    """
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(CONFIG['ATR_PERIOD']).mean()
    atr_percent = (atr / df['close']) * 100
    
    # Calculate EMA slope
    ema = df['close'].ewm(span=CONFIG['EMA_PERIOD']).mean()
    slope = ema.diff(CONFIG['TREND_SLOPE_PERIOD']) / CONFIG['TREND_SLOPE_PERIOD']
    slope_percent = (slope / ema) * 100
    trend_slope = calculate_trend_slope(df)
    
    # Determine regime
    if abs(slope_percent.iloc[-1]) > CONFIG['TREND_THRESHOLD']:
        return 'TRENDING'
    elif atr_percent.iloc[-1] > CONFIG['MIN_ATR_PERCENT']:
        return 'CHOP'
    return 'UNCERTAIN'

def detect_rsi_dip(df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
    """
    Detect RSI dip with confirmation.
    Returns: (signal, entry_price)
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(CONFIG['RSI_PERIOD']).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(CONFIG['RSI_PERIOD']).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Check for RSI dip with confirmation
    current_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]
    
    if (prev_rsi < CONFIG['RSI_OVERSOLD'] and 
        current_rsi < CONFIG['RSI_CONFIRMATION']):
        return True, df['close'].iloc[-1]
    return False, None

def detect_breakout(df: pd.DataFrame) -> Tuple[bool, Optional[float]]:
    """
    Detect price breakout above resistance with volume confirmation.
    Returns: (signal, entry_price)
    """
    # Calculate resistance (20-period high)
    resistance = df['high'].rolling(20).max()
    current_price = df['close'].iloc[-1]
    prev_resistance = resistance.iloc[-2]
    
    # Check for breakout
    if current_price > prev_resistance:
        # Calculate relative volume
        avg_volume = df['volume'].rolling(CONFIG['VOLUME_LOOKBACK']).mean()
        relative_volume = df['volume'].iloc[-1] / avg_volume.iloc[-1]
        
        if relative_volume > CONFIG['MIN_RELATIVE_VOLUME']:
            return True, current_price
    return False, None

def filter_by_volume(relative_volume: float) -> bool:
    """Filter out low volume setups"""
    return relative_volume >= CONFIG['MIN_RELATIVE_VOLUME']

def filter_by_volatility(atr_percent: float) -> bool:
    """Filter out low volatility setups"""
    return (CONFIG['MIN_ATR_PERCENT'] <= atr_percent <= CONFIG['MAX_ATR_PERCENT'])

def compute_tp_sl(entry_price: float, regime: str, atr: float) -> Tuple[float, float]:
    """
    Compute take profit and stop loss based on regime.
    Returns: (take_profit, stop_loss)
    """
    if regime == 'TRENDING':
        # Use adaptive TP/SL in trending markets
        tp = entry_price + (CONFIG['ADAPTIVE_TP_MULTIPLIER'] * atr)
        sl = entry_price - (CONFIG['ADAPTIVE_SL_MULTIPLIER'] * atr)
    else:
        # Use fixed TP/SL in other regimes
        if regime == "TRENDING":
            tp = entry_price * (1 + CONFIG['FIXED_TP_PERCENT'])
        else:
            tp = entry_price * (1 + CONFIG['FIXED_TP_PERCENT_UNCERTAIN_CHOP'])

        sl = entry_price * (1 - CONFIG['FIXED_SL_PERCENT'])
    
    return tp, sl

def risk_check(entry_price: float, stop_loss: float, position_size: float) -> bool:
    """Check if trade meets risk management rules"""
    # Calculate risk in USD
    risk_per_share = entry_price - stop_loss
    total_risk = risk_per_share * (position_size / entry_price)
    
    # Check position size
    if position_size > CONFIG['MAX_POSITION_SIZE']:
        logger.warning("Position size exceeds maximum")
        return False
    
    # Check risk per trade
    if total_risk > (position_size * CONFIG['MAX_RISK_PER_TRADE']):
        logger.warning("Risk per trade exceeds maximum")
        return False
    
    return True

def execute_trade(signal_data: Dict) -> None:
    """Log and simulate trade execution"""
    logger.info(f"Executing trade: {signal_data}")
    # In a real implementation, this would place orders with an exchange
    # For now, we just log the trade details

# Strategy dictionary mapping names to detection functions
STRATEGIES = {
    "RSI Dip": detect_rsi_dip,
    "Breakout": detect_breakout
}

def run_strategy(df: pd.DataFrame) -> None:
    """Main strategy loop"""
    # Detect market regime
    regime = detect_regime(df)
    logger.info(f"Current market regime: {regime}")
    
    # Calculate ATR for volatility check
    atr = df['high'].rolling(CONFIG['ATR_PERIOD']).max() - df['low'].rolling(CONFIG['ATR_PERIOD']).min()
    atr_percent = (atr / df['close']) * 100
    
    # Calculate relative volume
    avg_volume = df['volume'].rolling(CONFIG['VOLUME_LOOKBACK']).mean()
    relative_volume = df['volume'].iloc[-1] / avg_volume.iloc[-1]
    
    # Run each strategy
    for strategy_name, detection_fn in STRATEGIES.items():
        signal, entry_price = detection_fn(df)
        
        if signal:
            # Apply filters
            if not filter_by_volume(relative_volume):
                logger.info(f"{strategy_name} filtered by volume")
                continue
                
            if not filter_by_volatility(atr_percent.iloc[-1]):
                logger.info(f"{strategy_name} filtered by volatility")
                continue
            
            # Compute TP/SL
            tp, sl = compute_tp_sl(entry_price, regime, atr.iloc[-1])
            
            # Calculate position size
            position_size = CONFIG['MAX_POSITION_SIZE'] * CONFIG['LEVERAGE']
            
            # Risk check
            if not risk_check(entry_price, sl, position_size):
                logger.info(f"{strategy_name} failed risk check")
                continue
            
            # Prepare trade data
            trade_data = {
                'timestamp': datetime.now(UTC).isoformat(),
                'strategy': strategy_name,
                'regime': regime,
                'entry_price': entry_price,
                'take_profit': tp,
                'stop_loss': sl,
                'position_size': position_size,
                'leverage': CONFIG['LEVERAGE'],
                'rsi': None,  # Would be populated in real implementation
                'relative_volume': relative_volume,
                'atr_percent': atr_percent.iloc[-1]
            }
            
            # Execute trade
            execute_trade(trade_data)
            break  # Only execute one strategy per run

def fetch_candles(cb: CoinbaseService, product_id: str = 'BTC-USDC') -> pd.DataFrame:
    """
    Fetch historical candles from Coinbase.
    Returns DataFrame with OHLCV data.
    """
    # Default to last 8000 5-minute candles
    now = datetime.now(UTC)
    start = now - timedelta(minutes=5 * 8000)
    end = now
    
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, "FIVE_MINUTE")
    df = pd.DataFrame(raw_data)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp - convert Unix timestamp to datetime
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s', utc=True)        
        df.set_index('start', inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    
    return df

def main():
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data
        df = fetch_candles(cb)
        
        # Run strategy
        run_strategy(df)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 