# Simplified Trading Bot v1.2
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes
# Added 1-bar confirmation delay for RSI entries
# Added trend-following and momentum short capabilities

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

# Shared helpers
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def slope(series, window):
    return (series.diff(window) / window)

# Parameters
GRANULARITY = "FIVE_MINUTE"
RSI_THRESHOLD = 30
RSI_CONFIRMATION_THRESHOLD = 35  # New parameter for confirmation bar
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD

# EMA and slope parameters
EMA_WINDOW = 50
SLOPE_THRESHOLD_TREND = 0        # ≥ 0 = up-slope
SLOPE_THRESHOLD_MOM = 0         # < 0 = down-slope

# Recalculate these values every 50 trades using plot_atr_histogram.py
mean_atr_percent = 0.284
std_atr_percent = 0.148
# Recalculate this value every 50 trades using analyze_volume_thresholds.py
VOLUME_THRESHOLD = 1.4
# Recalculate this value every 50 trades using volatility_threshold.py
VOLATILITY_THRESHOLD = 0.25 
# Use trend_threshold.py to inform about this value
TREND_THRESHOLD = 0.001  # 1% per bar thresholid

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
    # First, get RSI for the current bar
    rsi_current = ta.compute_rsi(product_id, candles, period=14)
    
    # Get RSI for the previous bar by using all candles except the last one
    rsi_previous = ta.compute_rsi(product_id, candles[:-1], period=14)
    
    # Calculate EMA and EMA slope
    close_series = df['close']
    ema50 = ema(close_series, EMA_WINDOW)
    ema_slope = slope(ema50, 1).iloc[-1]
    
    # Get current and previous values
    current = df.iloc[-1]
    previous = df.iloc[-2]
    avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()
    
    # Calculate relative volume (current volume / average volume)
    relative_volume = current['volume'] / avg_volume if avg_volume > 0 else 0
    
    # Calculate trend slope using the new function
    trend_slope = calculate_trend_slope(df)

    # Log analysis details
    logger.info(f"Analysis: Current RSI={rsi_current:.2f}, Previous RSI={rsi_previous:.2f}, "
                f"Current Close={current['close']:.2f}, Previous Close={previous['close']:.2f}, "
                f"Volume={current['volume']:.2f} > Avg={avg_volume:.2f}, "
                f"Relative Volume={relative_volume:.2f}, Trend Slope={trend_slope:.4f}, "
                f"EMA Slope={ema_slope:.4f}")

    # Calculate ATR for the new filter
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta_obj = TechnicalAnalysis(cb)
    atr = ta_obj.compute_atr(candles)
    atr_percent = (atr / current['close']) * 100
    
    # Determine market regime based on trend slope
    market_regime = "UNCERTAIN"
    if abs(trend_slope) > TREND_THRESHOLD:
        market_regime = "TRENDING"
    elif atr_percent > VOLATILITY_THRESHOLD:
        market_regime = "CHOP"
    
    # Determine trade side based on EMA slope
    if ema_slope < SLOPE_THRESHOLD_MOM:
        trade_side = "SHORT"
    else:
        trade_side = "LONG"
    
    # Option A - Trend-follow filter for longs
    if trade_side == "LONG" and ema_slope < SLOPE_THRESHOLD_TREND:
        logger.info(f"[FILTERED] Down-slope blocked long (slope={ema_slope:.4f})")
        return False, None, None, None, None, None, None
        
    # New trade entry filters
    # Filter 1: Reject trades if Trend Regime == "UNCERTAIN" and RSI at Entry > 27
    if market_regime == "UNCERTAIN" and rsi_current > 27:
        logger.info(f"[FILTERED] Rejected trade: RSI > 27 in Uncertain regime (RSI={rsi_current:.2f})")
        return False, None, None, None, None, None, None
        
    # Filter 2: Reject trades if Trend Regime == "UNCERTAIN" and Relative Volume < 1.5
    if market_regime == "UNCERTAIN" and relative_volume < 1.5:
        logger.info(f"[FILTERED] Rejected trade: Low volume in Uncertain regime (Rel Vol={relative_volume:.2f})")
        return False, None, None, None, None, None, None
        
    # Filter 3: Reject trades if ATR % < 0.2 and abs(Trend Slope) < 0.001 and RSI at Entry > 25
    if atr_percent < 0.2 and abs(trend_slope) < 0.001 and rsi_current > 25:
        logger.info(f"[FILTERED] Rejected trade: Low volatility, flat trend, RSI > 25 (ATR %={atr_percent:.2f}, Trend Slope={trend_slope:.4f}, RSI={rsi_current:.2f})")
        return False, None, None, None, None, None, None

    # Option B - Momentum short module
    if trade_side == "SHORT":
        logger.info(f"[SIGNAL] {trade_side} {product_id} at {current['close']:.2f} (EMA slope={ema_slope:.4f})")
        entry_price = current["close"]
        tp_price = entry_price * (1 - TP_PERCENT)  # mirror TP
        sl_price = entry_price * (1 + SL_PERCENT)  # mirror SL
        tp_mode = "FIXED_SHORT"
        # Make sure we're returning proper values for all components
        if None in (trade_side, entry_price, tp_price, sl_price, tp_mode, POSITION_SIZE_USD):
            logger.error(f"SHORT signal has None values: {trade_side}, {entry_price}, {tp_price}, {sl_price}, {tp_mode}, {POSITION_SIZE_USD}")
        return True, trade_side, entry_price, tp_price, sl_price, tp_mode, POSITION_SIZE_USD
    
    # For LONG positions, use the existing RSI confirmation logic
    rsi_triggered = rsi_previous < RSI_THRESHOLD
    rsi_confirmed = rsi_current < RSI_THRESHOLD

    if rsi_triggered and rsi_confirmed:
        logger.info(f"[SIGNAL] {trade_side} {product_id} at {current['close']:.2f} "
                   f"(RSI triggered at {rsi_previous:.2f}, confirmed at {rsi_current:.2f})")
        entry_price = current["close"]
        tp_price = entry_price * (1 + TP_PERCENT)
        sl_price = entry_price * (1 - SL_PERCENT)
        tp_mode = "FIXED"
        return True, trade_side, entry_price, tp_price, sl_price, tp_mode, POSITION_SIZE_USD
    
    if rsi_triggered and not rsi_confirmed:
        logger.info(f"[SKIPPED] RSI triggered at {rsi_previous:.2f} but not confirmed at {rsi_current:.2f}")
    
    return False, None, None, None, None, None, None

def determine_tp_mode(entry_price: float, atr: float, price_precision: float = None, 
                     df: pd.DataFrame = None, trend_slope: float = None, 
                     trade_side: str = "LONG") -> tuple[str, float, str]:
    """
    Determine take profit mode and price based on ATR volatility and market regime.
    Also determines the market regime internally.
    
    Args:
        entry_price: Current entry price
        atr: Average True Range value
        price_precision: Precision for rounding the price
        df: DataFrame containing price data (optional)
        trend_slope: Pre-calculated trend slope (optional)
        trade_side: Trade direction, "LONG" or "SHORT" (default: "LONG")
        
    Returns:
        tuple: (tp_mode, tp_price, market_regime)
    """
    atr_percent = (atr / entry_price) * 100
    # Determine market regime internally
    market_regime = "UNCERTAIN"  # Default
    
    # If we have a DataFrame, calculate trend slope if not provided
    if df is not None and trend_slope is None:
        trend_slope = calculate_trend_slope(df)
    
    # If we have a trend slope, determine market regime
    if trend_slope is not None:
        # Check if we have a strong trend
        if abs(trend_slope) > TREND_THRESHOLD:
            market_regime = "TRENDING"
        
        # Check if we're in a choppy market (low trend, moderate volatility)
        elif atr_percent > VOLATILITY_THRESHOLD:
            market_regime = "CHOP"
    
    # For SHORT trades, use fixed TP/SL with mirrored values
    if trade_side == "SHORT":
        tp_mode = "FIXED_SHORT"
        tp_price = entry_price * (1 - TP_PERCENT)
    else:
        # High volatility → Use adaptive TP (2.5x ATR handles volatility better)
        adaptive_trigger = mean_atr_percent + std_atr_percent

        if atr_percent > adaptive_trigger and market_regime == "TRENDING":
            tp_mode = "ADAPTIVE"
            tp_price = entry_price + (2.5 * atr)  # 2×ATR adaptive TP
        else:
            # Low volatility → Use fixed TP based on market regime
            tp_mode = "FIXED"
                    
            # Regime-based TP logic
            if market_regime == "TRENDING":
                # Trending regime → TP = 1.5% or higher
                tp_price = entry_price * (1 + TP_PERCENT)  # 1.5% fixed TP
            elif market_regime == "CHOP":
                # Chop/news/noise → TP = 1.1%
                tp_price = entry_price * (1 + 0.011)  # 1.1% fixed TP
            else:
                # Default fallback → TP = 1.1% if uncertain
                tp_price = entry_price * (1 + 0.011)  # 1.1% fixed TP
    
    # Round the price if precision is provided
    if price_precision is not None:
        tp_price = round(tp_price, price_precision)
    
    return tp_mode, tp_price, market_regime

def execute_trade(cb, entry_price: float, product_id: str, margin: float, leverage: int, 
                 trend_slope: float = None, trade_side: str = "LONG", 
                 tp_price: float = None, sl_price: float = None, tp_mode: str = None):
    """Execute the trade using trade_btc_perp.py functions"""
    try:
        # Convert to perpetual futures product ID
        perp_product = get_perp_product(product_id)
        price_precision = get_price_precision(perp_product)
        
        # Calculate ATR for volatility check if we need to determine TP/SL
        candles = cb.historical_data.get_historical_data(product_id, datetime.now(UTC) - timedelta(minutes=5 * 100), datetime.now(UTC), GRANULARITY)
        ta = TechnicalAnalysis(cb)
        atr = ta.compute_atr(candles)
        # Calculate ATR percentage
        atr_percent = (atr / entry_price) * 100
        
        # If TP/SL not provided, determine them
        if tp_price is None or sl_price is None:
            # Determine TP mode, price, and market regime using centralized function
            tp_mode, tp_price, market_regime = determine_tp_mode(entry_price, atr, price_precision, 
                                                              pd.DataFrame(candles), trend_slope, trade_side)
            
            # Fixed stop loss based on trade side
            if trade_side == "SHORT":
                sl_price = round(entry_price * (1 + SL_PERCENT), price_precision)
            else:
                sl_price = round(entry_price * (1 - SL_PERCENT), price_precision)
        else:
            # Calculate market regime for logging
            if abs(trend_slope) > TREND_THRESHOLD:
                market_regime = "TRENDING"
            elif atr_percent > VOLATILITY_THRESHOLD:
                market_regime = "CHOP"
            else:
                market_regime = "UNCERTAIN"
                
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
        
        # Determine setup type based on entry conditions
        setup_type = "RSI Dip" if trade_side == "LONG" else "EMA Short"
        
        # Log trade setup information
        logger.info(f"Trade Side: {trade_side}")
        logger.info(f"TP Mode: {tp_mode}")
        logger.info(f"Take Profit: ${tp_price:.2f}")
        logger.info(f"Stop Loss: ${sl_price:.2f}")
        logger.info(f"Size: ${size_usd:.2f}")
        logger.info(f"Session: {session}")
        logger.info(f"ATR %: {atr_percent:.2f}")
        logger.info(f"Setup Type: {setup_type}")
        logger.info(f"Market Regime: {market_regime}")
        logger.info(f"Trend Slope: {trend_slope:.4f}")

        # Prepare command for trade_btc_perp.py
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', perp_product,
            '--side', 'BUY' if trade_side == "LONG" else 'SELL',
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
            if trade_side == "LONG":
                rr_ratio = (tp_price - entry_price) / (entry_price - sl_price)
            else:
                rr_ratio = (entry_price - tp_price) / (sl_price - entry_price)
            
            if atr_percent > mean_atr_percent + std_atr_percent:
                volatility_level = "Very Strong"
            elif atr_percent > mean_atr_percent:
                volatility_level = "Strong"
            elif atr_percent > mean_atr_percent - std_atr_percent:
                volatility_level = "Moderate"
            else:
                volatility_level = "Weak"                
                       
            # Create new trade entry with additional metrics
            new_trade = pd.DataFrame([{
                'No.': next_trade_no,
                'Timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
                'SIDE': trade_side,
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
                'Exit Trade': 0.0,
                'Trend Regime': market_regime,
                'RSI at Entry': 0.0,  # Will be updated with actual value
                'Relative Volume': 0.0,  # Will be updated with actual value
                'Trend Slope': trend_slope,  # Will be updated with actual value
                'Exit Reason': 'PENDING',
                'Duration': 0.0,
                'Market Trend': 'PENDING'
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
        perp_product = get_perp_product(args.product_id)
        df = fetch_candles(cb, perp_product)
        
        # Live trading mode
        signal, trade_side, entry_price, tp_price, sl_price, tp_mode, position_size = analyze(df, ta, args.product_id)
        
        if signal:
            logger.info(f"[SIGNAL] {trade_side} {args.product_id} at {entry_price:.2f}")
            if execute_trade(cb, entry_price, args.product_id, args.margin, args.leverage, 
                           calculate_trend_slope(df), trade_side, tp_price, sl_price, tp_mode):
                logger.info("Trade executed successfully!")
                
                # Get metrics from analysis for logging
                rsi_current = ta.compute_rsi(args.product_id, df.to_dict('records'), period=14)
                avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()
                relative_volume = df.iloc[-1]['volume'] / avg_volume if avg_volume > 0 else 0
                trend_slope = calculate_trend_slope(df)
                
                # Update the last trade with the additional metrics
                try:
                    trades_df = pd.read_csv('automated_trades.csv')
                    if not trades_df.empty:
                        # Update the last row with the additional metrics
                        trades_df.iloc[-1, trades_df.columns.get_loc('RSI at Entry')] = round(rsi_current, 2)
                        trades_df.iloc[-1, trades_df.columns.get_loc('Relative Volume')] = round(relative_volume, 2)
                        trades_df.iloc[-1, trades_df.columns.get_loc('Trend Slope')] = round(trend_slope, 4)
                        trades_df.to_csv('automated_trades.csv', index=False)
                        logger.info(f"Updated trade with RSI: {rsi_current:.2f}, Rel Volume: {relative_volume:.2f}, Trend Slope: {trend_slope:.4f}")
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