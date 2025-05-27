# Simplified Trading Bot v1.1 (Paper Trading Version)
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes
# Added 1-bar confirmation delay for RSI entries
# PAPER TRADING VERSION - No real trades executed

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from datetime import datetime, timedelta, UTC
import pandas as pd
from config import API_KEY_PERPS, API_SECRET_PERPS
import logging
import argparse
import numpy as np
import json

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
RSI_CONFIRMATION_THRESHOLD = 35  # New parameter for confirmation bar
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD

# Recalculate these values every 50 trades using plot_atr_histogram.py
mean_atr_percent = 0.284
std_atr_percent = 0.148

# Paper trading state file
PAPER_TRADING_STATE_FILE = 'paper_trading_state.json'

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified Trading Bot (Paper Trading)')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Product ID to trade (e.g., BTC-USDC)')
    parser.add_argument('--margin', type=float, default=100,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage')
    parser.add_argument('--initial_balance', type=float, default=1000,
                      help='Initial paper trading balance in USD')
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

def analyze(df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str):
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    # Calculate RSI for the last two bars
    rsi_current = ta.compute_rsi(product_id, candles, period=14)
    rsi_previous = ta.compute_rsi(product_id, candles[:-1], period=14)
    
    # Calculate EMA
    ema_50 = ta.get_moving_average(candles, period=50, ma_type='ema')
    
    # Get current and previous values
    current = df.iloc[-1]
    previous = df.iloc[-2]
    avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()
    
    # Calculate relative volume
    relative_volume = current['volume'] / avg_volume if avg_volume > 0 else 0
    
    # Calculate trend slope
    trend_slope = calculate_trend_slope(df)

    # Log analysis details
    logger.info(f"Analysis: Current RSI={rsi_current:.2f}, Previous RSI={rsi_previous:.2f}, "
                f"Current Close={current['close']:.2f}, Previous Close={previous['close']:.2f}, "
                f"Volume={current['volume']:.2f} > Avg={avg_volume:.2f}, "
                f"Relative Volume={relative_volume:.2f}, Trend Slope={trend_slope:.4f}")

    # RSI 1-bar confirmation logic
    rsi_triggered = rsi_previous < RSI_THRESHOLD
    rsi_confirmed = rsi_current < RSI_THRESHOLD

    if rsi_triggered and rsi_confirmed:
        logger.info(f"[SIGNAL] BUY {product_id} at {current['close']:.2f} "
                    f"(RSI triggered at {rsi_previous:.2f}, confirmed at {rsi_current:.2f})")
        return True, current["close"], rsi_current, relative_volume, trend_slope    
    
    if rsi_triggered and not rsi_confirmed:
        logger.info(f"[SKIPPED] RSI triggered at {rsi_previous:.2f} but not confirmed at {rsi_current:.2f}")
    
    return False, None, None, None, None

def determine_tp_mode(entry_price: float, atr: float, price_precision: float = None, 
                     df: pd.DataFrame = None, trend_slope: float = None) -> tuple[str, float, str]:
    """
    Determine take profit mode and price based on ATR volatility and market regime.
    """
    atr_percent = (atr / entry_price) * 100
    market_regime = "UNCERTAIN"
    
    if df is not None and trend_slope is None:
        trend_slope = calculate_trend_slope(df)
    
    if trend_slope is not None:
        TREND_THRESHOLD = 0.001
        VOLATILITY_THRESHOLD = 0.25
        
        if abs(trend_slope) > TREND_THRESHOLD:
            market_regime = "TRENDING"
        elif atr_percent > VOLATILITY_THRESHOLD:
            market_regime = "CHOP"
    
    adaptive_trigger = mean_atr_percent + std_atr_percent

    if atr_percent > adaptive_trigger and market_regime == "TRENDING":
        tp_mode = "ADAPTIVE"
        tp_price = entry_price + (2.5 * atr)
    else:
        tp_mode = "FIXED"                
        if market_regime == "TRENDING":
            tp_price = entry_price * (1 + 0.015)
        elif market_regime == "CHOP":
            tp_price = entry_price * (1 + 0.011)
        else:
            tp_price = entry_price * (1 + 0.011)
    
    if price_precision is not None:
        tp_price = round(tp_price, price_precision)
    
    return tp_mode, tp_price, market_regime

def load_paper_trading_state():
    """Load paper trading state from file"""
    if os.path.exists(PAPER_TRADING_STATE_FILE):
        with open(PAPER_TRADING_STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'balance': 1000.0,
        'positions': [],
        'trade_history': []
    }

def save_paper_trading_state(state):
    """Save paper trading state to file"""
    with open(PAPER_TRADING_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

def execute_paper_trade(cb, entry_price: float, product_id: str, margin: float, leverage: int, trend_slope: float = None):
    """Execute a paper trade"""
    try:
        # Load current state
        state = load_paper_trading_state()
        
        # Convert to perpetual futures product ID
        perp_product = get_perp_product(product_id)
        price_precision = get_price_precision(perp_product)
        
        # Calculate ATR for volatility check
        candles = cb.historical_data.get_historical_data(product_id, datetime.now(UTC) - timedelta(minutes=5 * 100), datetime.now(UTC), GRANULARITY)
        ta = TechnicalAnalysis(cb)
        atr = ta.compute_atr(candles)
        
        # Calculate ATR percentage
        atr_percent = (atr / entry_price) * 100
        
        # Calculate trend slope if not provided
        if trend_slope is None:
            candles_df = pd.DataFrame(candles)
            if 'start' in candles_df.columns:
                candles_df['start'] = pd.to_datetime(pd.to_numeric(candles_df['start']), unit='s', utc=True)
                candles_df.set_index('start', inplace=True)
            elif 'timestamp' in candles_df.columns:
                candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'], unit='s', utc=True)
                candles_df.set_index('timestamp', inplace=True)
            elif 'time' in candles_df.columns:
                candles_df['time'] = pd.to_datetime(candles_df['time'], unit='s', utc=True)
                candles_df.set_index('time', inplace=True)
            trend_slope = calculate_trend_slope(candles_df)
        
        # Determine TP mode and price
        tp_mode, tp_price, market_regime = determine_tp_mode(entry_price, atr, price_precision, pd.DataFrame(candles), trend_slope)
        
        # Fixed stop loss
        sl_price = round(entry_price * (1 - SL_PERCENT), price_precision)
        
        # Calculate size in USD
        size_usd = margin * leverage
        
        # Determine trading session
        current_hour = datetime.now(UTC).hour
        if 0 <= current_hour < 9:
            session = "Asia"
        elif 9 <= current_hour < 17:
            session = "EU"
        else:
            session = "US"
        
        # Create trade record
        trade = {
            'timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'product': perp_product,
            'side': 'BUY',
            'entry_price': entry_price,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'size_usd': size_usd,
            'leverage': leverage,
            'session': session,
            'tp_mode': tp_mode,
            'market_regime': market_regime,
            'atr_percent': atr_percent,
            'trend_slope': trend_slope,
            'status': 'OPEN'
        }
        
        # Add to positions
        state['positions'].append(trade)
        
        # Save state
        save_paper_trading_state(state)
        
        # Write to CSV file
        try:
            # Read existing trades to get the next trade number
            if os.path.exists('automated_trades_paper.csv'):
                trades_df = pd.read_csv('automated_trades_paper.csv')
                next_trade_no = len(trades_df) + 1
            else:
                next_trade_no = 1
                trades_df = pd.DataFrame()
            
            # Calculate R/R ratio
            rr_ratio = (tp_price - entry_price) / (entry_price - sl_price)
            
            # Determine volatility level
            if atr_percent > mean_atr_percent + std_atr_percent:
                volatility_level = "Very Strong"
            elif atr_percent > mean_atr_percent:
                volatility_level = "Strong"
            elif atr_percent > mean_atr_percent - std_atr_percent:
                volatility_level = "Moderate"
            else:
                volatility_level = "Weak"
            
            # Create new trade entry
            new_trade = pd.DataFrame([{
                'No.': next_trade_no,
                'Timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
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
                'Setup Type': 'RSI Dip',
                'MAE': 0.0,
                'MFE': 0.0,
                'Trend Regime': market_regime,
                'RSI at Entry': 0.0,  # Will be updated with actual value
                'Relative Volume': 0.0,  # Will be updated with actual value
                'Trend Slope': trend_slope,
                'Exit Reason': 'PENDING',
                'Duration': 0.0
            }])
            
            # Append new trade to CSV
            new_trade.to_csv('automated_trades_paper.csv', mode='a', header=not os.path.exists('automated_trades_paper.csv'), index=False)
            logger.info("Trade logged to automated_trades_paper.csv")
            
        except Exception as e:
            logger.error(f"Error logging trade to automated_trades_paper.csv: {e}")
        
        # Log trade details
        logger.info(f"Paper Trade Executed:")
        logger.info(f"Product: {perp_product}")
        logger.info(f"Entry Price: ${entry_price:.2f}")
        logger.info(f"Take Profit: ${tp_price:.2f}")
        logger.info(f"Stop Loss: ${sl_price:.2f}")
        logger.info(f"Size: ${size_usd:.2f}")
        logger.info(f"Leverage: {leverage}x")
        logger.info(f"Session: {session}")
        logger.info(f"TP Mode: {tp_mode}")
        logger.info(f"Market Regime: {market_regime}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error executing paper trade: {e}")
        return False

def check_open_positions():
    """Check if there are any open positions in the paper trading state"""
    try:
        state = load_paper_trading_state()
        return len(state['positions']) > 0
    except Exception as e:
        logger.error(f"Error checking open positions: {e}")
        return False

def main():
    args = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Check for open positions first
        if check_open_positions():
            logger.info("[SKIP] Open position exists - waiting for it to close")
            return
            
        # Fetch historical data
        df = fetch_candles(cb, args.product_id)
        
        # Live trading mode
        signal, entry, rsi_value, relative_volume, trend_slope = analyze(df, ta, args.product_id)
        
        if signal:
            logger.info(f"[SIGNAL] BUY {args.product_id} at {entry:.2f}")
            if execute_paper_trade(cb, entry, args.product_id, args.margin, args.leverage, trend_slope):
                logger.info("Paper trade executed successfully!")
            else:
                logger.error("Failed to execute paper trade")
        else:
            logger.info("[NO SIGNAL] Conditions not met.")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 