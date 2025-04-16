#!/usr/bin/env python3
"""
Time Stop Monitor for Trading Bot
Monitors open positions and implements a time-based stop loss strategy
for the RSI Dip trading setup.

Time Stop Logic:
- After a trade is entered, monitor the next 3 bars
- If price doesn't increase by at least 0.5% from entry during those 3 bars,
  exit the trade with reason: 'TIME STOP'

Usage:
    python time_stop_monitor.py
"""

import logging
import time
from datetime import datetime, timedelta, UTC
import pandas as pd
import subprocess
import argparse
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

# Import the time stop parameters from simplified_trading_bot
from simplified_trading_bot import TIME_STOP_BARS, TIME_STOP_THRESHOLD, GRANULARITY

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_latest_candles(cb, product_id, bars=10):
    """Fetch the latest N candles for a product."""
    now = datetime.now(UTC)
    # Calculate start time based on the granularity
    if GRANULARITY == "FIVE_MINUTE":
        start = now - timedelta(minutes=5 * (bars + 1))  # Add one extra bar to ensure we get enough
    elif GRANULARITY == "FIFTEEN_MINUTE":
        start = now - timedelta(minutes=15 * (bars + 1))
    elif GRANULARITY == "ONE_HOUR":
        start = now - timedelta(hours=(bars + 1))
    else:
        start = now - timedelta(minutes=5 * (bars + 1))  # Default to 5-minute

    raw_data = cb.historical_data.get_historical_data(product_id, start, now, GRANULARITY)
    df = pd.DataFrame(raw_data)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp conversion
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s', utc=True)        
        df.set_index('start', inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    
    return df.tail(bars)  # Return only the requested number of bars

def get_open_positions(cb):
    """Get all open positions from the exchange."""
    try:
        # Get the INTX portfolio UUID first
        ports = cb.client.get_portfolios()
        portfolio_uuid = None
        for p in ports['portfolios']:
            if p['type'] == "INTX":
                portfolio_uuid = p['uuid']
                break
        
        if not portfolio_uuid:
            logger.error("Could not find INTX portfolio")
            return []
            
        # Get portfolio positions
        portfolio = cb.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
        
        # Access the perp_positions from the response object
        positions = []
        if hasattr(portfolio, 'breakdown'):
            if hasattr(portfolio.breakdown, 'perp_positions'):
                positions = portfolio.breakdown.perp_positions
        
        # Format positions
        formatted_positions = []
        for position in positions:
            try:
                symbol = position['symbol']
                size = float(position['net_size'])
                entry_price = float(position['entry_price']['rawCurrency']['value'])
                mark_price = float(position['mark_price']['rawCurrency']['value'])
                unrealized_pnl = float(position['unrealized_pnl']['rawCurrency']['value'])
                position_side = position['position_side']
                
                # Only include non-zero positions
                if size != 0:
                    formatted_positions.append({
                        'symbol': symbol,
                        'size': size,
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'unrealized_pnl': unrealized_pnl,
                        'position_side': position_side
                    })
            except Exception as e:
                logger.error(f"Error processing position data: {e}")
                continue
                
        return formatted_positions
                
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        return []

def get_entry_time_from_csv(product_symbol):
    """Get the entry time of a position from the automated_trades.csv file."""
    try:
        trades_df = pd.read_csv('automated_trades.csv')
        # Find the most recent trade with PENDING outcome for this symbol
        pending_trades = trades_df[
            (trades_df['Outcome'] == 'PENDING') & 
            (trades_df['SIDE'] == 'LONG')
        ]
        
        if pending_trades.empty:
            return None
            
        # Get the most recent entry (last row)
        latest_trade = pending_trades.iloc[-1]
        
        # Convert timestamp to datetime
        try:
            entry_time = datetime.strptime(latest_trade['Timestamp'], '%Y-%m-%d %H:%M:%S UTC')
            entry_time = entry_time.replace(tzinfo=UTC)
        except:
            # Try alternative timestamp format
            entry_time = datetime.strptime(latest_trade['Timestamp'], '%Y-%m-%d %H:%M:%S')
            entry_time = entry_time.replace(tzinfo=UTC)
        
        return {
            'entry_time': entry_time,
            'entry_price': latest_trade['ENTRY']
        }
    except Exception as e:
        logger.error(f"Error reading trade entry from CSV: {e}")
        return None

def should_apply_time_stop(cb, position, trade_info):
    """Check if time stop should be applied based on price action."""
    if not trade_info or 'entry_time' not in trade_info:
        logger.warning("No valid trade entry information found")
        return False
        
    # Get the entry time and current time
    entry_time = trade_info['entry_time']
    current_time = datetime.now(UTC)
    
    # Get entry price from position data
    entry_price = position['entry_price']
    
    # Skip time stop check if position is not LONG or if it's not an RSI Dip setup
    if position['position_side'] != 'LONG':
        logger.info(f"Skipping time stop check for {position['symbol']} - not a LONG position")
        return False
        
    # Fetch candles since entry
    product_id = position['symbol'].replace('-PERP-INTX', '-USDC')  # Convert perp symbol to spot for historical data
    df = fetch_latest_candles(cb, product_id, TIME_STOP_BARS + 1)  # +1 to include current bar
    
    if df.empty or len(df) < TIME_STOP_BARS:
        logger.warning(f"Not enough candles to evaluate time stop for {position['symbol']}")
        return False
    
    # Calculate time elapsed since entry
    elapsed_time = current_time - entry_time
    elapsed_bars = elapsed_time.total_seconds() / (5 * 60)  # Convert to 5-minute bars
    
    # Debug info
    logger.info(f"Position: {position['symbol']}, Entry time: {entry_time}, Current time: {current_time}")
    logger.info(f"Elapsed time: {elapsed_time}, Elapsed bars: {elapsed_bars:.2f}")
    
    # Only apply time stop if we've completed the specified number of bars
    if elapsed_bars < TIME_STOP_BARS:
        logger.info(f"Not enough bars elapsed for time stop ({elapsed_bars:.2f}/{TIME_STOP_BARS})")
        return False
    
    # Check if we've already checked this position for time stop
    # (To prevent checking the same position multiple times after the TIME_STOP_BARS period)
    if elapsed_bars > TIME_STOP_BARS + 1:
        logger.info(f"Already passed time stop window for {position['symbol']}")
        return False
    
    # Check the highest price reached during the bars since entry
    # Use the last TIME_STOP_BARS bars from the dataframe
    recent_bars = df.tail(TIME_STOP_BARS)
    highest_price = recent_bars['high'].max()
    
    # Calculate the percent change from entry
    price_change_percent = (highest_price - entry_price) / entry_price
    
    logger.info(f"Entry price: {entry_price}, Highest price: {highest_price}")
    logger.info(f"Price change: {price_change_percent:.4f} ({price_change_percent*100:.2f}%), Threshold: {TIME_STOP_THRESHOLD:.4f} ({TIME_STOP_THRESHOLD*100:.2f}%)")
    
    # Apply time stop if price didn't reach the threshold
    if price_change_percent < TIME_STOP_THRESHOLD:
        logger.info(f"Time stop triggered for {position['symbol']} - Price didn't reach +{TIME_STOP_THRESHOLD*100:.2f}% within {TIME_STOP_BARS} bars")
        return True
    else:
        logger.info(f"Time stop not triggered - Price reached +{price_change_percent*100:.2f}% (> {TIME_STOP_THRESHOLD*100:.2f}% threshold)")
        return False

def close_position(position, reason="TIME STOP"):
    """Close a position using the close_positions.py script."""
    try:
        logger.info(f"Closing position {position['symbol']} - Reason: {reason}")
        
        # Execute the close_positions.py script
        cmd = ['python', 'close_positions.py']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error closing position: {result.stderr}")
            return False
            
        logger.info("Position closed successfully!")
        logger.info(f"Command output: {result.stdout}")
        
        # Update the trade in automated_trades.csv
        try:
            trades_df = pd.read_csv('automated_trades.csv')
            # Find the most recent trade with PENDING outcome
            for idx in reversed(range(len(trades_df))):
                if trades_df.loc[idx, 'Outcome'] == 'PENDING':
                    # Update the trade
                    current_time = datetime.now(UTC)
                    
                    # Calculate trade duration in hours
                    entry_time_str = trades_df.loc[idx, 'Timestamp']
                    try:
                        entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S UTC')
                    except:
                        entry_time = datetime.strptime(entry_time_str, '%Y-%m-%d %H:%M:%S')
                    
                    entry_time = entry_time.replace(tzinfo=UTC)
                    duration_hours = (current_time - entry_time).total_seconds() / 3600
                    
                    # Update the trade
                    trades_df.loc[idx, 'Outcome'] = 'LOSS'
                    trades_df.loc[idx, 'Exit Trade'] = position['mark_price']
                    trades_df.loc[idx, 'Outcome %'] = (position['mark_price'] - trades_df.loc[idx, 'ENTRY']) / trades_df.loc[idx, 'ENTRY'] * 100
                    trades_df.loc[idx, 'Exit Reason'] = reason
                    trades_df.loc[idx, 'Duration'] = round(duration_hours, 2)
                    
                    # Save the updated dataframe
                    trades_df.to_csv('automated_trades.csv', index=False)
                    logger.info(f"Updated trade in automated_trades.csv - Exit reason: {reason}")
                    break
        except Exception as e:
            logger.error(f"Error updating trade in automated_trades.csv: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error executing close_position: {e}")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Time Stop Monitor')
    parser.add_argument('--check-interval', type=int, default=60,
                      help='Check interval in seconds (default: 60)')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    check_interval = args.check_interval
    
    logger.info("Starting Time Stop Monitor...")
    logger.info(f"Time stop parameters: {TIME_STOP_BARS} bars, {TIME_STOP_THRESHOLD*100:.2f}% threshold")
    logger.info(f"Check interval: {check_interval} seconds")
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    
    while True:
        try:
            # Get open positions
            positions = get_open_positions(cb)
            
            if not positions:
                logger.info("No open positions found. Waiting...")
                time.sleep(check_interval)
                continue
                
            # Check each position for time stop
            for position in positions:
                logger.info(f"Checking position: {position['symbol']}, Size: {position['size']}, Entry: {position['entry_price']}")
                
                # Get trade information from CSV
                trade_info = get_entry_time_from_csv(position['symbol'])
                
                # Check if time stop should be applied
                if should_apply_time_stop(cb, position, trade_info):
                    # Close the position
                    if close_position(position, "TIME STOP"):
                        logger.info(f"Successfully closed position {position['symbol']} due to TIME STOP")
                    else:
                        logger.error(f"Failed to close position {position['symbol']}")
                else:
                    logger.info(f"No time stop applied for {position['symbol']}")
            
            # Sleep before next check
            time.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main() 