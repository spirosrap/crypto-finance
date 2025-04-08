# Simplified Trading Bot
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes

from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from datetime import datetime, timedelta, UTC
import pandas as pd
from config import API_KEY_PERPS, API_SECRET_PERPS
import logging
import argparse
import subprocess
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress logs from other modules
logging.getLogger('technicalanalysis').setLevel(logging.WARNING)
logging.getLogger('historicaldata').setLevel(logging.WARNING)
logging.getLogger('bitcoinpredictionmodel').setLevel(logging.WARNING)
logging.getLogger('ml_model').setLevel(logging.WARNING)

# Parameters
GRANULARITY = "FIVE_MINUTE"
RSI_THRESHOLD = 30
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD
EMA_SLOPE_THRESHOLD = -0.001

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified Trading Bot')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Product ID to trade (e.g., BTC-USDC)')
    parser.add_argument('--margin', type=float, default=100,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage')
    parser.add_argument('--test', action='store_true',
                      help='Run execute_trade tests')
    parser.add_argument('--ema-slope-threshold', type=float, default=0.0,
                      help='Minimum EMA50 slope threshold for trend filter (default: 0.0)')
    return parser.parse_args()

def get_perp_product(product_id):
    """Convert spot product ID to perpetual futures product ID"""
    perp_map = {
        'BTC-USDC': 'BTC-PERP-INTX',
        'ETH-USDC': 'ETH-PERP-INTX',
        'DOGE-USDC': 'DOGE-PERP-INTX',
        'SOL-USDC': 'SOL-PERP-INTX',
        'SHIB-USDC': '1000SHIB-PERP-INTX'
    }
    return perp_map.get(product_id, 'BTC-PERP-INTX')

def get_price_precision(product_id):
    """Get price precision for a product"""
    precision_map = {
        'BTC-PERP-INTX': 1,      # $1 precision for BTC
        'ETH-PERP-INTX': 0.1,    # $0.1 precision for ETH
        'DOGE-PERP-INTX': 0.0001, # $0.0001 precision for DOGE
        'SOL-PERP-INTX': 0.01,   # $0.01 precision for SOL
        '1000SHIB-PERP-INTX': 0.000001  # $0.000001 precision for SHIB
    }
    return precision_map.get(product_id, 1)

def fetch_candles(cb, product_id):
    # Default to last 8000 5-minute candles
    now = datetime.now(UTC)
    start = now - timedelta(minutes=5 * 8000)
    end = now
    
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, GRANULARITY)
    # print(raw_data[:2])
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

def analyze(df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str):
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    # Calculate RSI
    rsi = ta.compute_rsi(product_id, candles, period=14)
    
    # Calculate EMA for current and past points to determine slope
    # We need at least 10 candles to calculate the slope
    if len(candles) >= 10:
        # Get current EMA
        current_ema = ta.get_moving_average(candles, period=50, ma_type='ema')
        
        # Get past EMA (10 bars ago)
        past_candles = candles[:-10]  # Remove the last 10 candles
        past_ema = ta.get_moving_average(past_candles, period=50, ma_type='ema')
        
        # Calculate slope as percentage change
        ema_slope = (current_ema - past_ema) / past_ema
    else:
        # Not enough data to calculate slope
        ema_slope = 0
        logger.warning("Not enough candles to calculate EMA slope. Using default value of 0.")
    
    # Calculate ATR for volatility regime detection
    try:
        atr = ta.compute_atr(candles, period=14)
        current_price = df.iloc[-1]['close']
        atr_percent = (atr / current_price) * 100
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}. Defaulting to moderate volatility.")
        atr_percent = 0.75  # Default to moderate volatility
    
    # Classify volatility regime
    if atr_percent < 0.5:
        volatility_regime = "Weak"
    elif atr_percent <= 1.0:
        volatility_regime = "Moderate"
    else:
        volatility_regime = "Strong"
    
    # Classify trend regime
    if abs(ema_slope) < 0.001:  # 0.1%
        trend_regime = "Sideways"
    else:
        trend_regime = "Trending"
    
    # Determine market regime
    market_regime = f"{volatility_regime}/{trend_regime}"
    
    # Get current values
    current = df.iloc[-1]
    avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()

    # Log analysis details with regime information
    logger.info(f"Analysis: RSI={rsi:.2f}, Volume={current['volume']:.2f} > Avg={avg_volume:.2f}, "
                f"EMA Slope={ema_slope:.4f}, ATR%={atr_percent:.2f}%, Regime={market_regime}")

    # Initialize entry mode as None
    entry_mode = None
    
    # Check for Mean Reversion Entry (Reversal Mode)
    # Only in Weak/Sideways or Moderate/Sideways regimes
    if (volatility_regime in ["Weak", "Moderate"] and trend_regime == "Sideways" and
        rsi < RSI_THRESHOLD and current["volume"] > avg_volume):
        entry_mode = "mean_reversion"
        logger.info(f"[SIGNAL] Mean Reversion Entry for {product_id} at {current['close']:.2f} in {market_regime} regime")
        return True, current["close"], market_regime, entry_mode
    
    # Check for Momentum Entry (Trend Mode)
    # Only in Strong/Trending or Moderate/Trending regimes
    if (volatility_regime in ["Strong", "Moderate"] and trend_regime == "Trending" and
        rsi > 50 and current["close"] > current_ema and current["volume"] > avg_volume):
        entry_mode = "momentum"
        logger.info(f"[SIGNAL] Momentum Entry for {product_id} at {current['close']:.2f} in {market_regime} regime")
        return True, current["close"], market_regime, entry_mode
    
    # Log when conditions are met but regime doesn't allow entry
    if rsi < RSI_THRESHOLD and current["volume"] > avg_volume and volatility_regime in ["Weak", "Moderate"] and trend_regime == "Sideways":
        logger.info(f"[BLOCKED] Mean Reversion signal blocked due to unfavorable regime: {market_regime}")
    
    if rsi > 50 and current["close"] > current_ema and current["volume"] > avg_volume and volatility_regime in ["Strong", "Moderate"] and trend_regime == "Trending":
        logger.info(f"[BLOCKED] Momentum signal blocked due to unfavorable regime: {market_regime}")
    
    return False, None, market_regime, None

def determine_tp_mode(entry_price: float, atr: float, price_precision: float = None) -> tuple[str, float]:
    """
    Determine take profit mode and price based on ATR volatility.
    Returns a tuple of (tp_mode, tp_price)
    """
    atr_percent = (atr / entry_price) * 100
    if atr_percent > 0.7:
        # High volatility → Use adaptive TP (2.5x ATR handles volatility better)
        tp_mode = "ADAPTIVE"
        tp_price = entry_price + (2.5 * atr)  # 2.5×ATR adaptive TP
    else:
        # Low volatility → Use fixed TP (market less likely to run, so %-based makes sense)
        tp_mode = "FIXED"
        tp_price = entry_price * (1 + TP_PERCENT)  # 1.5% fixed TP
    
    # Round the price if precision is provided
    if price_precision is not None:
        tp_price = round(tp_price, price_precision)
    
    return tp_mode, tp_price

def execute_trade(cb, entry_price: float, product_id: str, margin: float, leverage: int, market_regime: str = "Unknown/Unknown", entry_mode: str = None):
    """Execute the trade using trade_btc_perp.py functions"""
    try:
        # Convert to perpetual futures product ID
        perp_product = get_perp_product(product_id)
        price_precision = get_price_precision(perp_product)
        
        # Calculate ATR for volatility check
        candles = cb.historical_data.get_historical_data(product_id, datetime.now(UTC) - timedelta(minutes=5 * 100), datetime.now(UTC), GRANULARITY)
        ta = TechnicalAnalysis(cb)
        atr = ta.compute_atr(candles)
        
        # Determine TP mode and price using centralized function
        tp_mode, tp_price = determine_tp_mode(entry_price, atr, price_precision)
        
        # Fixed stop loss
        sl_price = round(entry_price * (1 - SL_PERCENT), price_precision)              
        # Calculate size in USD
        size_usd = margin * leverage
        
        # Determine trading session based on current UTC time
        current_hour = datetime.now(UTC).hour
        if 0 <= current_hour < 9:
            session = "Asia"
        elif 9 <= current_hour < 17:
            session = "EU"
        else:
            session = "US"            
        
        # Calculate ATR percentage
        atr_percent = (atr / entry_price) * 100
        
        # Determine setup type based on entry mode
        if entry_mode == "mean_reversion":
            setup_type = "RSI Dip"
        elif entry_mode == "momentum":
            setup_type = "Momentum"
        else:
            setup_type = "Unknown"  # Fallback
        
        # Log trade setup information
        logger.info(f"TP Mode: {tp_mode}")
        logger.info(f"Take Profit: ${tp_price:.2f}")
        logger.info(f"Stop Loss: ${sl_price:.2f}")
        logger.info(f"Size: ${size_usd:.2f}")
        logger.info(f"Session: {session}")
        logger.info(f"ATR %: {atr_percent:.2f}")
        logger.info(f"Setup Type: {setup_type}")
        logger.info(f"Market Regime: {market_regime}")
        logger.info(f"Entry Mode: {entry_mode}")
        
        # Prepare command for trade_btc_perp.py
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', perp_product,
            '--side', 'BUY',
            '--size', str(size_usd),
            '--leverage', str(leverage),
            '--tp', str(tp_price),
            '--sl', str(sl_price),
            '--no-confirm'
        ]
        
        # Execute the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error placing order: {result.stderr}")
            return False
            
        logger.info("Order placed successfully!")
        logger.info(f"Command output: {result.stdout}")

        # Log trade to automated_trades.csv
        try:
            # Read existing trades to get the next trade number
            trades_df = pd.read_csv('automated_trades.csv')
            next_trade_no = len(trades_df) + 1
            
            # Calculate R/R ratio
            rr_ratio = (tp_price - entry_price) / (entry_price - sl_price)
            
            # Determine volume strength based on ATR percentage - THIS IS VOLATILITY LEVEL
            if atr_percent > 1.0:
                volatility_level = "Strong"
            elif atr_percent > 0.5:
                volatility_level = "Moderate"
            else:
                volatility_level = "Weak"
            
            # Create new trade entry with additional metrics
            new_trade = pd.DataFrame([{
                'No.': next_trade_no,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'SIDE': 'LONG',
                'ENTRY': entry_price,
                'Take Profit': tp_price,
                'Stop Loss': sl_price,
                'R/R Ratio': round(rr_ratio, 2),
                'Volatility Level': volatility_level,
                'Outcome': 'PENDING',
                'Outcome %': 0.0,
                'Leverage': f"{leverage}x",
                'Margin': margin,
                'Session': session,
                'TP Mode': tp_mode,
                'ATR %': round(atr_percent, 2),
                'Setup Type': setup_type,
                'MAE': 0.0,
                'MFE': 0.0,
                'Trend Regime': market_regime,
                'Entry Mode': entry_mode
            }])
            
            # Append new trade to CSV
            new_trade.to_csv('automated_trades.csv', mode='a', header=False, index=False)
            logger.info("Trade logged to automated_trades.csv")
            
        except Exception as e:
            logger.error(f"Error logging trade to automated_trades.csv: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def main():
    args = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data
        df = fetch_candles(cb, args.product_id)
        
        # Update global EMA_SLOPE_THRESHOLD with command line argument
        global EMA_SLOPE_THRESHOLD
        EMA_SLOPE_THRESHOLD = args.ema_slope_threshold
        
        # Live trading mode
        signal, entry, market_regime, entry_mode = analyze(df, ta, args.product_id)
        
        if signal:
            logger.info(f"[SIGNAL] BUY {args.product_id} at {entry:.2f}")
            if execute_trade(cb, entry, args.product_id, args.margin, args.leverage, market_regime, entry_mode):
                logger.info("Trade executed successfully!")
            else:
                logger.error("Failed to execute trade")
        else:
            logger.info("[NO SIGNAL] Conditions not met.")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
