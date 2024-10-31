import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from technical_analysis.backtest import Backtester
from technical_analysis.ethereum_analysis import EthereumAnalysis
from technical_analysis.backtest_analysis import BacktestAnalysis
from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
import pandas as pd
from typing import List, Dict
from config import API_KEY, API_SECRET, NEWS_API_KEY
import numpy as np

def validate_candle_data(candle: Dict) -> bool:
    """Validate that a candle has all required fields with valid values."""
    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Check all fields exist
    if not all(field in candle for field in required_fields):
        print(f"Missing required fields. Found: {list(candle.keys())}")
        return False
    
    # Check numeric fields are valid
    try:
        # Validate timestamp
        if not candle['timestamp']:
            print("Invalid timestamp")
            return False
            
        # Validate price data
        open_price = float(candle['open'])
        high_price = float(candle['high'])
        low_price = float(candle['low'])
        close_price = float(candle['close'])
        volume = float(candle['volume'])
        
        if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
            print("Found non-positive price value")
            return False
            
        if high_price < low_price:
            print("High price is less than low price")
            return False
            
        if volume < 0:
            print("Negative volume")
            return False
            
        return True
        
    except (ValueError, TypeError) as e:
        print(f"Validation error: {str(e)}")
        return False

def prepare_historical_data(historical_data: HistoricalData, product_id: str, start_date: str, end_date: str, granularity: str) -> List[Dict]:
    """
    Prepare historical data from HistoricalData class for backtesting.
    """
    # Convert string dates to datetime objects
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Get historical data
    print("Fetching data from HistoricalData...")
    data = historical_data.get_historical_data(
        product_id=product_id,
        start_date=start_datetime,
        end_date=end_datetime,
        granularity=granularity
    )
    
    print(f"Raw data type: {type(data)}")
    print("Sample of raw data:")
    print(data[:2] if isinstance(data, list) else data.head(2))
    
    # Convert to list of dictionaries based on data type
    candles = []
    
    try:
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                timestamp = row.get('timestamp') or row.get('start')
                # Convert timestamp to datetime if it's a string
                if isinstance(timestamp, str):
                    try:
                        # Try parsing as Unix timestamp first
                        timestamp = int(timestamp)
                    except ValueError:
                        # If not a Unix timestamp, try parsing as ISO format
                        timestamp = pd.to_datetime(timestamp).timestamp()
                
                candle = {
                    'timestamp': timestamp,  # Store as Unix timestamp
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                candles.append(candle)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            # If data is a list of dictionaries
            for item in data:
                timestamp = item.get('timestamp') or item.get('start')
                if isinstance(timestamp, str):
                    try:
                        timestamp = int(timestamp)
                    except ValueError:
                        timestamp = pd.to_datetime(timestamp).timestamp()
                
                candle = {
                    'timestamp': timestamp,
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': float(item['volume'])
                }
                candles.append(candle)
        elif isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
            # If data is a list of lists/tuples
            for item in data:
                candle = {
                    'timestamp': item[0],
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                }
                candles.append(candle)
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
            
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("Data sample for debugging:")
        print(data[:2] if isinstance(data, list) else data)
        raise
    
    if not candles:
        raise ValueError("No valid candle data could be processed")
    
    # Convert timestamp strings to integers for sorting
    for candle in candles:
        if isinstance(candle['timestamp'], str):
            candle['timestamp'] = int(candle['timestamp'])
    
    # Sort candles by timestamp
    candles.sort(key=lambda x: x['timestamp'])
    
    # Validate candles
    valid_candles = []
    for candle in candles:
        if validate_candle_data(candle):
            valid_candles.append(candle)
        else:
            print(f"Warning: Invalid candle data found and skipped: {candle}")
    
    if len(valid_candles) < len(candles):
        print(f"Warning: {len(candles) - len(valid_candles)} invalid candles were removed")
    
    if not valid_candles:
        raise ValueError("No valid candle data found")
    
    print("\nProcessed data sample:")
    print(valid_candles[0] if valid_candles else "No valid candles")
    
    return valid_candles

def print_trade_summary(trades_history: List[Dict]):
    """Print a summary of all trades."""
    print("\nTrade History:")
    print("=" * 100)
    print(f"{'Entry Time':<20} {'Exit Time':<20} {'Type':<6} {'Entry':<10} {'Exit':<10} {'P&L %':<8} {'P&L $':<10} {'Exit Type':<10}")
    print("-" * 100)
    
    for trade in trades_history:
        entry_time = pd.to_datetime(trade['entry_time']).strftime('%Y-%m-%d %H:%M')
        exit_time = pd.to_datetime(trade['exit_time']).strftime('%Y-%m-%d %H:%M')
        
        print(
            f"{entry_time:<20} "
            f"{exit_time:<20} "
            f"{trade['position_type']:<6} "
            f"${trade['entry_price']:<9.2f} "
            f"${trade['exit_price']:<9.2f} "
            f"{trade['pnl_pct']*100:>6.2f}% "
            f"${trade['pnl']:>9.2f} "
            f"{trade['exit_type']:<10}"
        )
    
    # Calculate and print trade statistics
    winning_trades = [t for t in trades_history if t['pnl'] > 0]
    losing_trades = [t for t in trades_history if t['pnl'] <= 0]
    
    print("\nTrade Statistics:")
    print("-" * 50)
    print(f"Total Trades: {len(trades_history)}")
    print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades_history)*100:.1f}%)")
    print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades_history)*100:.1f}%)")
    print(f"Average Win: ${np.mean([t['pnl'] for t in winning_trades]):.2f}")
    print(f"Average Loss: ${np.mean([t['pnl'] for t in losing_trades]):.2f}")
    print(f"Largest Win: ${max([t['pnl'] for t in winning_trades]):.2f}")
    print(f"Largest Loss: ${min([t['pnl'] for t in losing_trades]):.2f}")
    print(f"Average Trade Duration: {calculate_average_duration(trades_history)}")
    
def calculate_average_duration(trades_history: List[Dict]) -> str:
    """Calculate the average duration of trades."""
    durations = []
    for trade in trades_history:
        entry = pd.to_datetime(trade['entry_time'])
        exit = pd.to_datetime(trade['exit_time'])
        duration = exit - entry
        durations.append(duration)
    
    if not durations:
        return "N/A"
    
    avg_duration = sum(durations, pd.Timedelta(0)) / len(durations)
    hours = avg_duration.total_seconds() / 3600
    
    if hours < 1:
        return f"{hours*60:.1f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        return f"{hours/24:.1f} days"

def monitor_trades(backtester: Backtester, candles: List[Dict]):
    """Monitor and potentially modify trades in real-time."""
    for i in range(len(candles) - 1):
        current_candles = candles[:i+1]
        current_candle = current_candles[-1]
        
        # Get current position info
        position = backtester.get_current_position()
        if position:
            # Get market status
            market_status = backtester.get_market_status(current_candle)
            
            # Example: Trail stop loss if in profit
            if market_status['unrealized_pnl'] > 0.02:  # 2% profit
                new_stop_loss = position['entry_price']  # Move stop loss to break even
                backtester.modify_position(new_stop_loss=new_stop_loss)
            
            # Example: Take partial profits
            if market_status['unrealized_pnl'] > 0.03:  # 3% profit
                backtester.modify_position(reduce_size_pct=0.5)  # Reduce position by 50%
            
            # Example: Force close if conditions change
            latest_signal = backtester.get_latest_signal()
            if latest_signal and latest_signal.signal_type != position['position_type']:
                backtester.close_position(current_candle)

def print_trade_opportunities(trades: List[Dict]):
    """Print potential trade opportunities."""
    print("\nPotential Trade Opportunities:")
    print("=" * 100)
    print(f"{'Time':<20} {'Signal':<12} {'Price':<10} {'Stop Loss':<10} {'Take Profit':<12} {'Confidence':<10} {'R/R':<6}")
    print("-" * 100)
    
    for trade in trades:
        print(
            f"{trade['timestamp']:<20} "
            f"{trade['signal_type']:<12} "
            f"${trade['current_price']:<9.2f} "
            f"${trade['potential_stop_loss']:<9.2f} "
            f"${trade['potential_take_profit']:<11.2f} "
            f"{trade['confidence']*100:>7.1f}% "
            f"{trade['risk_reward_ratio']:>5.2f}"
        )

def main():
    try:
        # Initialize services
        print("Initializing services...")
        coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        historical_data = HistoricalData(coinbase_service.client)
        
        # Set backtest parameters
        product_id = 'ETH-USDC'
        start_date = '2024-01-01'
        end_date = '2024-12-31'
        granularity = 'ONE_HOUR'
        initial_capital = 10000.0
        
        print(f"\nBacktest Parameters:")
        print(f"Product ID: {product_id}")
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Granularity: {granularity}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        
        # Get historical data
        print(f"\nFetching historical data for {product_id}...")
        candles = prepare_historical_data(
            historical_data=historical_data,
            product_id=product_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity
        )
        
        print(f"\nRetrieved {len(candles)} valid candles")
        
        if len(candles) < 50:  # Minimum required for analysis
            raise ValueError(f"Insufficient data: only {len(candles)} candles retrieved. Need at least 50.")
        
        # Initialize backtester with Ethereum strategy
        backtester = Backtester(
            strategy_class=EthereumAnalysis,
            initial_capital=initial_capital,
            position_size=0.1,
            stop_loss_pct=0.02,
            take_profit_pct=0.04
        )
        
        # Run backtest
        print("Running backtest...")
        results = backtester.run(candles)
        
        # Get unresolved trade opportunities
        print("\nAnalyzing potential trade opportunities...")
        opportunities = backtester.simulate_unresolved_trades(candles)
        
        if opportunities:
            print_trade_opportunities(opportunities)
            
            # Print detailed analysis of last opportunity
            last_opportunity = opportunities[-1]
            print("\nDetailed Analysis of Latest Opportunity:")
            print("=" * 50)
            print(f"Time: {last_opportunity['timestamp']}")
            print(f"Signal: {last_opportunity['signal_type']}")
            print(f"Market Regime: {last_opportunity['market_regime']}")
            print(f"Current Price: ${last_opportunity['current_price']:.2f}")
            print(f"Suggested Entry: ${last_opportunity['potential_entry']:.2f}")
            print(f"Stop Loss: ${last_opportunity['potential_stop_loss']:.2f}")
            print(f"Take Profit: ${last_opportunity['potential_take_profit']:.2f}")
            print(f"Position Size: ${last_opportunity['potential_position_size']:.2f}")
            print(f"Risk/Reward Ratio: {last_opportunity['risk_reward_ratio']:.2f}")
            print(f"Confidence: {last_opportunity['confidence']*100:.1f}%")
            print("\nIndicator Values:")
            for indicator, value in last_opportunity['indicators'].items():
                print(f"{indicator}: {value:.2f}")
        else:
            print("\nNo trade opportunities found at current market conditions.")
        
        # Print trade summary if there are any trades
        if results.trades_history:
            print_trade_summary(results.trades_history)
        else:
            print("\nNo trades were executed during the backtest period.")
        
        # Print overall results
        print("\nBacktest Results:")
        print("=" * 50)
        print(f"Period: {start_date} to {end_date}")
        print(f"Product: {product_id}")
        print(f"Granularity: {granularity}")
        print("-" * 50)
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${initial_capital * (1 + results.total_return):,.2f}")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        
        # Create analysis object
        print("\nCreating analysis report...")
        try:
            analysis = BacktestAnalysis(results.trades_history, initial_capital=initial_capital)
            
            # Generate comprehensive report
            print("Generating analysis report...")
            metrics = analysis.generate_report(output_dir='backtest_results')
            
            # Save detailed trade history to CSV
            trades_df = pd.DataFrame(results.trades_history)
            print("\nTrade history columns:", trades_df.columns.tolist())  # Debug info
            
            output_file = f'backtest_results/trades_{product_id}_{start_date}_{end_date}.csv'
            os.makedirs('backtest_results', exist_ok=True)
            trades_df.to_csv(output_file, index=False)
            print(f"\nDetailed trade history saved to: {output_file}")
            
        except Exception as e:
            print(f"\nWarning: Could not generate analysis report: {str(e)}")
            print("Continuing with basic results...")
        
    except ValueError as e:
        print(f"\nValue Error: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()