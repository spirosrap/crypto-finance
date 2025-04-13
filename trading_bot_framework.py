import pandas as pd
import numpy as np
from datetime import datetime, UTC, timedelta
import logging
from typing import Dict, Tuple, Optional, List
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY_PERPS, API_SECRET_PERPS
import os
import subprocess

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

    # VOLATILITY CALIBRATION
    # Recalculate these values every 50 trades using plot_atr_histogram.py    
    'MEAN_ATR_PERCENT': 0.284,
    'STD_ATR_PERCENT': 0.148,
    
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

def detect_regime(df: pd.DataFrame, ta: TechnicalAnalysis, candles: pd.DataFrame) -> str:
    """
    Classify market regime using ATR and EMA slope.
    Returns: 'TRENDING', 'CHOP', or 'UNCERTAIN'
    """
    # Calculate ATR using TechnicalAnalysis
    atr = ta.compute_atr(candles)
    atr_percent = (atr / df['close'].iloc[-1]) * 100
        
    # Calculate EMA slope
    trend_slope = calculate_trend_slope(df)
    
    # Determine regime
    if abs(trend_slope) > CONFIG['TREND_THRESHOLD']:
        return 'TRENDING'
    elif atr_percent > CONFIG['MIN_ATR_PERCENT']:
        return 'CHOP'
    return 'UNCERTAIN'

def detect_rsi_dip(df: pd.DataFrame, ta: TechnicalAnalysis, candles: pd.DataFrame, product_id: str) -> Tuple[bool, Optional[float]]:
    """
    Detect RSI dip with confirmation.
    Args:
        df: DataFrame with price data
        ta: TechnicalAnalysis instance
        candles: DataFrame with candle data
        product_id: Trading pair identifier (e.g. 'BTC-USDC')
    Returns: (signal, entry_price)
    """
    # Calculate RSI for current and previous candles
    current_rsi = ta.compute_rsi(product_id, candles, CONFIG['RSI_PERIOD'])
    
    # Get previous RSI by removing the last candle
    prev_candles = candles[:-1]
    prev_rsi = ta.compute_rsi(product_id, prev_candles, CONFIG['RSI_PERIOD'])
    
    if (prev_rsi < CONFIG['RSI_OVERSOLD'] and 
        current_rsi < CONFIG['RSI_CONFIRMATION']):
        return True, df['close'].iloc[-1]
    return False, None

def detect_breakout(df: pd.DataFrame, ta: TechnicalAnalysis, candles: pd.DataFrame, product_id: str) -> Tuple[bool, Optional[float]]:
    """
    Detect price breakout above resistance with volume confirmation.
    Args:
        df: DataFrame with price data
        ta: TechnicalAnalysis instance
        candles: DataFrame with candle data
        product_id: Trading pair identifier (e.g. 'BTC-USDC')
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
    """Execute trade using trade_btc_perp.py and log to CSV"""
    try:
        # Read existing trades to get the next trade number
        try:
            trades_df = pd.read_csv('automated_trades.csv')
            next_trade_no = len(trades_df) + 1
        except FileNotFoundError:
            next_trade_no = 1
            trades_df = pd.DataFrame()
        
        # Calculate R/R ratio
        rr_ratio = (signal_data['take_profit'] - signal_data['entry_price']) / (signal_data['entry_price'] - signal_data['stop_loss'])
        
        # Determine volatility level based on ATR percentage
        if signal_data['atr_percent'] > CONFIG['MEAN_ATR_PERCENT'] + CONFIG['STD_ATR_PERCENT']:
            volatility_level = "Very Strong"
        elif signal_data['atr_percent'] > CONFIG['MEAN_ATR_PERCENT']:
            volatility_level = "Strong"
        elif signal_data['atr_percent'] > CONFIG['MEAN_ATR_PERCENT'] - CONFIG['STD_ATR_PERCENT']:
            volatility_level = "Moderate"
        else:
            volatility_level = "Weak"                
            
        
        # Determine trading session based on current UTC time
        current_hour = datetime.now(UTC).hour
        if 0 <= current_hour < 9:
            session = "Asia"
        elif 9 <= current_hour < 17:
            session = "EU"
        else:
            session = "US"

        # Create new trade entry
        new_trade = pd.DataFrame([{
            'No.': next_trade_no,
            'Timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'SIDE': 'LONG',
            'ENTRY': signal_data['entry_price'],
            'Take Profit': signal_data['take_profit'],
            'Stop Loss': signal_data['stop_loss'],
            'R/R Ratio': round(rr_ratio, 2),
            'Volatility Level': volatility_level,
            'Outcome': 'PENDING',
            'Outcome %': 0.0,
            'Leverage': f"{signal_data['leverage']}x",
            'Margin': signal_data['position_size'] / signal_data['leverage'],
            'Session': session,
            'TP Mode': 'ADAPTIVE' if signal_data['regime'] == 'TRENDING' else 'FIXED',
            'ATR %': round(signal_data['atr_percent'], 2),
            'Setup Type': signal_data['strategy'],
            'MAE': 0.0,
            'MFE': 0.0,
            'Exit Trade': 0.0,
            'Trend Regime': signal_data['regime'],
            'RSI at Entry': 0.0,  # Would be populated in real implementation
            'Relative Volume': round(signal_data['relative_volume'], 2),
            'Trend Slope': round(signal_data['trend_slope'], 4),
            'Exit Reason': 'PENDING',
            'Duration': 0.0,
            'Market Trend': 'PENDING'
        }])
        
        # Append new trade to CSV
        new_trade.to_csv('automated_trades.csv', mode='a', header=not os.path.exists('automated_trades.csv'), index=False)
        logger.info("Trade logged to automated_trades.csv")

        # Prepare command for trade_btc_perp.py
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', signal_data['product_id'],
            '--side', 'BUY',
            '--size', str(signal_data['position_size']),
            '--leverage', str(signal_data['leverage']),
            '--tp', str(signal_data['take_profit']),
            '--sl', str(signal_data['stop_loss']),
            '--no-confirm'
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error placing order: {result.stderr}")
            return False
            
        logger.info("Order placed successfully!")
        logger.info(f"Command output: {result.stdout}")
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

# Strategy dictionary mapping names to detection functions
STRATEGIES = {
    "RSI Dip": detect_rsi_dip,
    "Breakout": detect_breakout
}

def run_strategy(df: pd.DataFrame) -> None:
    """Main strategy loop"""
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    # Fetch candles once for all calculations
    candles = cb.historical_data.get_historical_data('BTC-USDC', datetime.now(UTC) - timedelta(minutes=5 * 100), datetime.now(UTC), "FIVE_MINUTE")
    
    # Detect market regime
    regime = detect_regime(df, ta, candles)
    logger.info(f"Current market regime: {regime}")
    
    # Calculate ATR for volatility check using TechnicalAnalysis
    atr = ta.compute_atr(candles)
    atr_percent = (atr / df['close'].iloc[-1]) * 100
    
    # Calculate relative volume
    avg_volume = df['volume'].rolling(CONFIG['VOLUME_LOOKBACK']).mean()
    relative_volume = df['volume'].iloc[-1] / avg_volume.iloc[-1]
    
    # Calculate trend slope
    trend_slope = calculate_trend_slope(df)
    
    # Flag to track if any signal was detected
    signal_detected = False
    
    # Run each strategy
    for strategy_name, detection_fn in STRATEGIES.items():
        signal, entry_price = detection_fn(df, ta, candles, 'BTC-USDC')
        
        if signal:
            signal_detected = True
            # Apply filters
            if not filter_by_volume(relative_volume):
                logger.info(f"{strategy_name} filtered by volume")
                continue
                
            if not filter_by_volatility(atr_percent):
                logger.info(f"{strategy_name} filtered by volatility")
                continue
            
            # Compute TP/SL
            tp, sl = compute_tp_sl(entry_price, regime, atr)
            
            # Calculate position size
            position_size = CONFIG['MAX_POSITION_SIZE'] * CONFIG['LEVERAGE']
            
            # Risk check
            if not risk_check(entry_price, sl, position_size):
                logger.info(f"{strategy_name} failed risk check")
                continue
            
            # Prepare trade data
            trade_data = {
                'strategy': strategy_name,
                'regime': regime,
                'entry_price': entry_price,
                'take_profit': tp,
                'stop_loss': sl,
                'position_size': position_size,
                'leverage': CONFIG['LEVERAGE'],
                'relative_volume': relative_volume,
                'atr_percent': atr_percent,
                'trend_slope': trend_slope,
                'product_id': 'BTC-USDC'
            }
            
            # Execute trade
            execute_trade(trade_data)
            break  # Only execute one strategy per run
    
    if not signal_detected:
        logger.info("Signal RSI Dip or Signal breakout not detected")

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