from simplified_trading_bot import (
    CoinbaseService, TechnicalAnalysis, GRANULARITY, RSI_THRESHOLD,
    VOLUME_LOOKBACK, TP_PERCENT, SL_PERCENT, get_perp_product,
    get_price_precision, analyze, determine_tp_mode
)
from datetime import datetime, timedelta, UTC
import pandas as pd
import logging
import argparse
from config import API_KEY_PERPS, API_SECRET_PERPS

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress logs from other modules
logging.getLogger('technicalanalysis').setLevel(logging.WARNING)
logging.getLogger('historicaldata').setLevel(logging.WARNING)
logging.getLogger('bitcoinpredictionmodel').setLevel(logging.WARNING)
logging.getLogger('ml_model').setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest Trading Bot')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Product ID to trade (e.g., BTC-USDC)')
    parser.add_argument('--initial_balance', type=float, default=10000,
                      help='Initial balance for backtesting')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage')
    parser.add_argument('--start_date', type=str, default=None,
                      help='Start date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for backtest (format: YYYY-MM-DD)')
    return parser.parse_args()

def fetch_candles(cb, product_id, start_date=None, end_date=None):
    if start_date and end_date:
        # Convert string dates to datetime objects
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=UTC)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=UTC)
    else:
        # Default to last 8000 5-minute candles if no dates specified
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

def backtest(df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str, initial_balance: float = 10000, leverage: int = 5):
    """
    Backtest the trading strategy on historical data
    Returns performance metrics and trade history
    """
    balance = initial_balance
    position = 0
    trades = []
    current_trade = None
    
    # Track balance history for drawdown calculation
    balance_history = [initial_balance]
    peak_balance = initial_balance
    max_drawdown = 0
    max_drawdown_duration = 0
    current_drawdown_start = None
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"backtest_trades_{product_id}_{timestamp}.csv"
    
    # Debug log DataFrame info at start of backtest
    logger.info(f"DataFrame index type: {type(df.index)}")
    logger.info(f"First few timestamps in backtest: {df.index[:3]}")
    
    # Ensure df index is datetime and in UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    for i in range(50, len(df)):  # Start after EMA period
        # Get historical data up to current point
        historical_df = df.iloc[:i+1]
        signal, entry = analyze(historical_df, ta, product_id)
        
        # If we have an open position, check if TP or SL is hit
        if current_trade:
            current_price = df.iloc[i]['close']
            
            # Calculate ATR for volatility check
            atr = ta.compute_atr(historical_df.to_dict('records'))
            
            # Determine TP mode and price using centralized function
            tp_mode, tp_price = determine_tp_mode(current_trade['entry_price'], atr)
            
            # Check if take profit or stop loss is hit
            if current_price >= tp_price:
                # Take profit hit
                profit = (tp_price - current_trade['entry_price']) * current_trade['size'] * leverage
                balance += profit
                trades.append({
                    'entry_time': current_trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': current_trade['entry_price'],
                    'exit_price': tp_price,
                    'profit': profit,
                    'type': 'TP',
                    'atr': atr,
                    'atr_percent': (atr / current_trade['entry_price']) * 100,
                    'tp_mode': tp_mode
                })
                if len(trades) >= 200:
                    break                
                current_trade = None
                position = 0
                
            elif current_price <= current_trade['sl_price']:
                # Stop loss hit
                loss = (current_trade['sl_price'] - current_trade['entry_price']) * current_trade['size'] * leverage
                balance += loss
                trades.append({
                    'entry_time': current_trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': current_trade['entry_price'],
                    'exit_price': current_trade['sl_price'],
                    'profit': loss,
                    'type': 'SL',
                    'atr': atr,
                    'atr_percent': (atr / current_trade['entry_price']) * 100,
                    'tp_mode': tp_mode
                })
                if len(trades) >= 200:
                    break                
                current_trade = None
                position = 0
        
        # If no position and signal, enter trade
        elif signal and not current_trade:
            current_price = df.iloc[i]['close']
            
            # Calculate ATR for volatility check
            atr = ta.compute_atr(historical_df.to_dict('records'))
            
            # Determine TP mode and price using centralized function
            tp_mode, tp_price = determine_tp_mode(current_price, atr, get_price_precision(get_perp_product(product_id)))
            
            # Fixed stop loss
            sl_price = round(current_price * (1 - SL_PERCENT), get_price_precision(get_perp_product(product_id)))
            
            # Calculate position size
            position_size = balance * 0.1  # Use 10% of balance per trade
            size = position_size / current_price
            
            # Log trade setup information
            atr_percent = (atr / current_price) * 100
            logger.info(f"ATR: {atr:.2f} ({atr_percent:.2f}% of entry price)")
            logger.info(f"TP Mode: {tp_mode}")
            logger.info(f"Take Profit: ${tp_price:.2f}")
            logger.info(f"Stop Loss: ${sl_price:.2f}")
            
            current_trade = {
                'entry_time': df.index[i],
                'entry_price': current_price,
                'size': size,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_mode': tp_mode
            }
            position = size
        
        # Update balance history and calculate drawdown
        balance_history.append(balance)
        if balance > peak_balance:
            peak_balance = balance
            # If we were in a drawdown, calculate its duration
            if current_drawdown_start is not None:
                drawdown_duration = (df.index[i] - current_drawdown_start).total_seconds() / 3600  # Convert to hours
                max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                current_drawdown_start = None
        else:
            # Calculate current drawdown
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
            # If this is the start of a new drawdown
            if current_drawdown_start is None:
                current_drawdown_start = df.index[i]
    
    # Calculate performance metrics
    if trades:
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = len([t for t in trades if t['profit'] < 0])
        total_profit = sum(t['profit'] for t in trades)
        win_rate = (winning_trades / total_trades) * 100
        profit_factor = abs(sum(t['profit'] for t in trades if t['profit'] > 0) / 
                          sum(t['profit'] for t in trades if t['profit'] < 0)) if losing_trades > 0 else float('inf')
        
        # Calculate metrics by TP mode
        fixed_tp_trades = [t for t in trades if t['tp_mode'] == "FIXED"]
        adaptive_tp_trades = [t for t in trades if t['tp_mode'] == "ADAPTIVE"]
        
        fixed_tp_win_rate = (len([t for t in fixed_tp_trades if t['profit'] > 0]) / len(fixed_tp_trades) * 100) if fixed_tp_trades else 0
        adaptive_tp_win_rate = (len([t for t in adaptive_tp_trades if t['profit'] > 0]) / len(adaptive_tp_trades) * 100) if adaptive_tp_trades else 0
        
        # Calculate average ATR for winning vs losing trades
        winning_trades_atr = [t['atr'] for t in trades if t['profit'] > 0]
        losing_trades_atr = [t['atr'] for t in trades if t['profit'] < 0]
        avg_winning_atr = sum(winning_trades_atr) / len(winning_trades_atr) if winning_trades_atr else 0
        avg_losing_atr = sum(losing_trades_atr) / len(losing_trades_atr) if losing_trades_atr else 0
        
        # Export trades to CSV
        try:
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades)
            
            # Reorder columns for better readability
            columns = [
                'entry_time', 'exit_time', 'entry_price', 'exit_price',
                'profit', 'type', 'tp_mode', 'atr', 'atr_percent'
            ]
            trades_df = trades_df[columns]
            
            # Rename columns for clarity
            trades_df.columns = [
                'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price',
                'Profit', 'Exit Type', 'TP Mode', 'ATR', 'ATR %'
            ]
            
            # Add additional columns
            trades_df['Session'] = trades_df['Entry Time'].apply(lambda x: 
                'Asia' if 0 <= pd.to_datetime(x).hour < 9 else 
                'EU' if 9 <= pd.to_datetime(x).hour < 17 else 'US')
            trades_df['Setup Type'] = 'RSI Dip'  # Since we're using RSI strategy
            
            # Add Outcome column based on Exit Type
            trades_df['Outcome'] = trades_df['Exit Type'].apply(lambda x: 'SUCCESS' if x == 'TP' else 'FAILURE')
            
            # Format numeric columns
            trades_df['Entry Price'] = trades_df['Entry Price'].round(2)
            trades_df['Exit Price'] = trades_df['Exit Price'].round(2)
            trades_df['Profit'] = trades_df['Profit'].round(2)
            trades_df['ATR'] = trades_df['ATR'].round(2)
            trades_df['ATR %'] = trades_df['ATR %'].round(2)
            
            # Reorder columns to match the desired order
            trades_df = trades_df[[
                'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price',
                'Profit', 'Exit Type', 'Session', 'TP Mode', 'ATR %', 'Setup Type', 'Outcome'
            ]]
            
            # Save to CSV
            trades_df.to_csv(csv_filename, index=False)
            logger.info(f"\nTrade history exported to: {csv_filename}")
            
        except Exception as e:
            logger.error(f"Error exporting trades to CSV: {e}")
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'avg_winning_atr': avg_winning_atr,
            'avg_losing_atr': avg_losing_atr,
            'fixed_tp_trades': len(fixed_tp_trades),
            'adaptive_tp_trades': len(adaptive_tp_trades),
            'fixed_tp_win_rate': fixed_tp_win_rate,
            'adaptive_tp_win_rate': adaptive_tp_win_rate,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'max_drawdown_duration': max_drawdown_duration,  # In hours
            'trades': trades,
            'csv_filename': csv_filename
        }
    else:
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'profit_factor': 0,
            'avg_winning_atr': 0,
            'avg_losing_atr': 0,
            'fixed_tp_trades': 0,
            'adaptive_tp_trades': 0,
            'fixed_tp_win_rate': 0,
            'adaptive_tp_win_rate': 0,
            'max_drawdown': 0,
            'max_drawdown_duration': 0,
            'trades': [],
            'csv_filename': None
        }

def main():
    args = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data with optional date range
        df = fetch_candles(cb, args.product_id, args.start_date, args.end_date)
        
        logger.info("Starting backtest...")
        if args.start_date and args.end_date:
            logger.info(f"Backtest period: {args.start_date} to {args.end_date}")
        else:
            logger.info("Using default period (last 8000 5-minute candles)")
        
        results = backtest(df, ta, args.product_id, args.initial_balance, args.leverage)
        
        # Print backtest results
        logger.info("\nBacktest Results:")
        logger.info(f"Initial Balance: ${results['initial_balance']:.2f}")
        logger.info(f"Final Balance: ${results['final_balance']:.2f}")
        logger.info(f"Total Profit: ${results['total_profit']:.2f}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Winning Trades: {results['winning_trades']}")
        logger.info(f"Losing Trades: {results['losing_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"\nTP Mode Statistics:")
        logger.info(f"Fixed TP Trades: {results['fixed_tp_trades']}")
        logger.info(f"Fixed TP Win Rate: {results['fixed_tp_win_rate']:.2f}%")
        logger.info(f"Adaptive TP Trades: {results['adaptive_tp_trades']}")
        logger.info(f"Adaptive TP Win Rate: {results['adaptive_tp_win_rate']:.2f}%")
        logger.info(f"\nRisk Metrics:")
        logger.info(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Maximum Drawdown Duration: {results['max_drawdown_duration']:.2f} hours")
        
        if results['csv_filename']:
            logger.info(f"\nDetailed trade history saved to: {results['csv_filename']}")
        
        # Print trade history
        logger.info("\nTrade History:")
        for trade in results['trades']:
            logger.info(f"Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
            logger.info(f"Exit: {trade['exit_time']} @ ${trade['exit_price']:.2f}")
            logger.info(f"Profit: ${trade['profit']:.2f} ({trade['type']})")
            logger.info(f"TP Mode: {trade['tp_mode']} (ATR: {trade['atr']:.2f}, {trade['atr_percent']:.2f}%)")
            logger.info("---")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 