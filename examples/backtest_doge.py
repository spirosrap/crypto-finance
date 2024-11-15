import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from technical_analysis.backtest import Backtester, BacktestResults
from technical_analysis.doge_analysis import DogeAnalysis
from technical_analysis.backtest_analysis import BacktestAnalysis
from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
import pandas as pd
from typing import List, Dict
from config import API_KEY, API_SECRET
import numpy as np

# Import the helper functions from backtest_ethereum.py
from backtest_ethereum import (
    validate_candle_data,
    prepare_historical_data,
    calculate_average_duration,
    monitor_trades,
    print_trade_opportunities
)

def print_trade_summary(trades_history: List[Dict]):
    """Print a summary of all trades."""
    if not trades_history:
        print("\nNo trades were executed during the backtest period.")
        return

    print("\nTrade History:")
    print("=" * 100)
    print(f"{'Entry Time':<20} {'Exit Time':<20} {'Type':<6} {'Entry':<10} {'Exit':<10} {'P&L %':<8} {'P&L $':<10} {'Exit Type':<10}")
    print("-" * 100)
    
    for trade in trades_history:
        # Convert timestamps to datetime
        entry_time = pd.Timestamp.fromtimestamp(trade['entry_time']/1000).strftime('%Y-%m-%d %H:%M')
        exit_time = pd.Timestamp.fromtimestamp(trade['exit_time']/1000).strftime('%Y-%m-%d %H:%M')
        
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

def print_backtest_results(results: BacktestResults, initial_capital: float):
    """Print backtest results summary."""
    print("\nBacktest Results:")
    print("=" * 50)
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${initial_capital * (1 + results.total_return):,.2f}")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")

def main():
    try:
        # Initialize services
        print("Initializing services...")
        coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        historical_data = HistoricalData(coinbase_service.client)
        
        # Set backtest parameters
        product_id = 'DOGE-USDC'
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
        
        if len(candles) < 50:
            raise ValueError(f"Insufficient data: only {len(candles)} candles retrieved. Need at least 50.")
        
        # Initialize backtester with DOGE strategy
        backtester = Backtester(
            strategy_class=DogeAnalysis,
            initial_capital=initial_capital,
            position_size=0.05,  # Reduced position size for better risk management
            stop_loss_pct=0.02,  # Tighter stop loss
            take_profit_pct=0.04  # 2:1 reward-to-risk ratio
        )
        
        # Run backtest
        print("Running backtest...")
        results = backtester.run(candles)
        
        # Print results
        if results.trades_history:
            print_trade_summary(results.trades_history)
            print_backtest_results(results, initial_capital)
            
            # Generate analysis report only if there are trades
            try:
                analysis = BacktestAnalysis(results.trades_history, initial_capital=initial_capital)
                metrics = analysis.generate_report(output_dir='backtest_results')
                
                # Save detailed trade history
                trades_df = pd.DataFrame(results.trades_history)
                output_file = f'backtest_results/trades_{product_id}_{start_date}_{end_date}.csv'
                os.makedirs('backtest_results', exist_ok=True)
                trades_df.to_csv(output_file, index=False)
                print(f"\nDetailed trade history saved to: {output_file}")
                
            except Exception as e:
                print(f"\nWarning: Could not generate analysis report: {str(e)}")
        else:
            print("\nNo trades were executed during the backtest period.")
            print("\nThis could be due to:")
            print("1. Strategy conditions were never met")
            print("2. Risk management rules prevented trade entries")
            print("3. Insufficient price movement for signal generation")
            
        # Get unresolved trade opportunities
        print("\nAnalyzing potential trade opportunities...")
        opportunities = backtester.simulate_unresolved_trades(candles)
        
        if opportunities:
            print_trade_opportunities(opportunities)
        else:
            print("\nNo trade opportunities found at current market conditions.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 