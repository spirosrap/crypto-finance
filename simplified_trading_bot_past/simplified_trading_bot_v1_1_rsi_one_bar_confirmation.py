# Simplified Trading Bot v1.1
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes
# Added 1-bar confirmation delay for RSI entries

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
RSI_CONFIRMATION_THRESHOLD = 35  # New parameter for confirmation bar
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD

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
    
    # Calculate RSI for the last two bars
    # First, get RSI for the current bar
    rsi_current = ta.compute_rsi(product_id, candles, period=14)
    
    # Get RSI for the previous bar by using all candles except the last one
    rsi_previous = ta.compute_rsi(product_id, candles[:-1], period=14)
    
    # Calculate EMA
    ema_50 = ta.get_moving_average(candles, period=50, ma_type='ema')
    
    # Get current and previous values
    current = df.iloc[-1]
    previous = df.iloc[-2]
    avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()
    
    # Calculate relative volume (current volume / average volume)
    relative_volume = current['volume'] / avg_volume if avg_volume > 0 else 0
    
    # Calculate trend slope (using EMA) with robust error handling
    trend_slope = 0.0
    try:
        # Use a simpler approach - calculate slope between current and previous close prices
        if len(df) >= 2:
            # Get the last 5 close prices for a short-term trend
            close_prices = df['close'].tail(5).values
            if len(close_prices) >= 2:
                # Simple linear regression slope calculation
                x = np.arange(len(close_prices))
                y = close_prices
                slope, _ = np.polyfit(x, y, 1)
                trend_slope = slope
    except Exception as e:
        logger.warning(f"Error calculating trend slope: {e}")
        trend_slope = 0.0

    # Log analysis details
    logger.info(f"Analysis: Current RSI={rsi_current:.2f}, Previous RSI={rsi_previous:.2f}, "
                f"Current Close={current['close']:.2f}, Previous Close={previous['close']:.2f}, "
                f"Volume={current['volume']:.2f} > Avg={avg_volume:.2f}, "
                f"Relative Volume={relative_volume:.2f}, Trend Slope={trend_slope:.4f}")

    # Check if RSI was below threshold on previous bar and is now below confirmation threshold
    rsi_triggered = rsi_previous < RSI_THRESHOLD
    rsi_confirmed = rsi_current < RSI_CONFIRMATION_THRESHOLD
    
    # Pure 1-bar delay logic
    rsi_triggered = rsi_previous < RSI_THRESHOLD
    rsi_confirmed = rsi_current < RSI_THRESHOLD

    if rsi_triggered and rsi_confirmed:
        logger.info(f"[SIGNAL] BUY {product_id} at {current['close']:.2f} "
                    f"(RSI triggered at {rsi_previous:.2f}, confirmed at {rsi_current:.2f})")
        return True, current["close"], rsi_current, relative_volume, trend_slope    
    
    if rsi_triggered and not rsi_confirmed:
        logger.info(f"[SKIPPED] RSI triggered at {rsi_previous:.2f} but not confirmed at {rsi_current:.2f}")
    
    return False, None, None, None, None

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

def execute_trade(cb, entry_price: float, product_id: str, margin: float, leverage: int):
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
        
        # Determine setup type based on entry conditions
        setup_type = "RSI Dip"  # Default since we're using RSI strategy
        
        # Log trade setup information
        logger.info(f"TP Mode: {tp_mode}")
        logger.info(f"Take Profit: ${tp_price:.2f}")
        logger.info(f"Stop Loss: ${sl_price:.2f}")
        logger.info(f"Size: ${size_usd:.2f}")
        logger.info(f"Session: {session}")
        logger.info(f"ATR %: {atr_percent:.2f}")
        logger.info(f"Setup Type: {setup_type}")

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
                'Setup Type': setup_type,
                'MAE': 0.0,
                'MFE': 0.0,
                'Trend Regime': 'PENDING',
                'RSI at Entry': 0.0,  # Will be updated with actual value
                'Relative Volume': 0.0,  # Will be updated with actual value
                'Trend Slope': 0.0  # Will be updated with actual value
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
        
        # Live trading mode
        signal, entry, rsi_value, relative_volume, trend_slope = analyze(df, ta, args.product_id)
        
        if signal:
            logger.info(f"[SIGNAL] BUY {args.product_id} at {entry:.2f}")
            if execute_trade(cb, entry, args.product_id, args.margin, args.leverage):
                logger.info("Trade executed successfully!")
                
                # Update the last trade with the additional metrics
                try:
                    trades_df = pd.read_csv('automated_trades.csv')
                    if not trades_df.empty:
                        # Update the last row with the additional metrics
                        trades_df.iloc[-1, trades_df.columns.get_loc('RSI at Entry')] = round(rsi_value, 2)
                        trades_df.iloc[-1, trades_df.columns.get_loc('Relative Volume')] = round(relative_volume, 2)
                        trades_df.iloc[-1, trades_df.columns.get_loc('Trend Slope')] = round(trend_slope, 4)
                        trades_df.to_csv('automated_trades.csv', index=False)
                        logger.info(f"Updated trade with RSI: {rsi_value:.2f}, Rel Volume: {relative_volume:.2f}, Trend Slope: {trend_slope:.4f}")
                except Exception as e:
                    logger.error(f"Error updating trade with additional metrics: {e}")
            else:
                logger.error("Failed to execute trade")
        else:
            logger.info("[NO SIGNAL] Conditions not met.")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 