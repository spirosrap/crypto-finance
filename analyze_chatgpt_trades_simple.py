#!/usr/bin/env python3
"""
Simple ChatGPT Trades Analyzer

This program analyzes trades from chatgpt_trades.csv to provide basic statistics
and analysis without requiring API access for historical data.

Key Features:
- Loads and analyzes trades from CSV
- Calculates basic statistics
- Provides trade outcome analysis framework
- Handles both long and short positions
- Exports results to CSV
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatgpt_trades_simple_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimpleChatGPTTradesAnalyzer:
    def __init__(self, csv_file_path: str = "chatgpt_trades.csv"):
        self.csv_file_path = csv_file_path
        self.trades_df = None
        self.analyzed_trades = []
        
    def load_trades(self) -> bool:
        """Load trades from CSV file."""
        try:
            if not os.path.exists(self.csv_file_path):
                logger.error(f"CSV file not found: {self.csv_file_path}")
                return False
                
            self.trades_df = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(self.trades_df)} trades from {self.csv_file_path}")
            
            # Convert timestamp to datetime
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            
            # Convert numeric columns
            numeric_columns = ['entry_price', 'stop_loss', 'take_profit', 'position_size_usd', 'margin', 'leverage']
            for col in numeric_columns:
                if col in self.trades_df.columns:
                    self.trades_df[col] = pd.to_numeric(self.trades_df[col], errors='coerce')
            
            # Display basic information about the data
            self.print_data_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return False
    
    def print_data_info(self):
        """Print information about the loaded data."""
        print("\n" + "="*60)
        print("LOADED TRADES INFORMATION")
        print("="*60)
        
        print(f"Total trades: {len(self.trades_df)}")
        print(f"Date range: {self.trades_df['timestamp'].min()} to {self.trades_df['timestamp'].max()}")
        
        print(f"\nSymbols traded:")
        symbol_counts = self.trades_df['symbol'].value_counts()
        for symbol, count in symbol_counts.items():
            print(f"  {symbol}: {count} trades")
        
        print(f"\nTrade sides:")
        side_counts = self.trades_df['side'].value_counts()
        for side, count in side_counts.items():
            print(f"  {side}: {count} trades")
        
        print(f"\nStrategies used:")
        strategy_counts = self.trades_df['strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} trades")
        
        print(f"\nLeverage distribution:")
        leverage_counts = self.trades_df['leverage'].value_counts().sort_index()
        for leverage, count in leverage_counts.items():
            print(f"  {leverage}x: {count} trades")
        
        print("="*60)
    
    def calculate_risk_reward_ratios(self) -> pd.DataFrame:
        """Calculate risk/reward ratios for each trade."""
        if self.trades_df is None:
            logger.error("No trades loaded. Call load_trades() first.")
            return pd.DataFrame()
        
        # Create a copy for analysis
        analysis_df = self.trades_df.copy()
        
        # Calculate risk and reward in price terms
        analysis_df['risk_price'] = abs(analysis_df['entry_price'] - analysis_df['stop_loss'])
        analysis_df['reward_price'] = abs(analysis_df['take_profit'] - analysis_df['entry_price'])
        
        # Calculate risk/reward ratio
        analysis_df['risk_reward_ratio'] = analysis_df['reward_price'] / analysis_df['risk_price']
        
        # Calculate percentage risk and reward
        analysis_df['risk_pct'] = (analysis_df['risk_price'] / analysis_df['entry_price']) * 100
        analysis_df['reward_pct'] = (analysis_df['reward_price'] / analysis_df['entry_price']) * 100
        
        # Calculate leveraged risk and reward
        analysis_df['leveraged_risk_pct'] = analysis_df['risk_pct'] * analysis_df['leverage']
        analysis_df['leveraged_reward_pct'] = analysis_df['reward_pct'] * analysis_df['leverage']
        
        # Calculate potential profit/loss in USD
        analysis_df['potential_profit_usd'] = analysis_df['margin'] * (analysis_df['leveraged_reward_pct'] / 100)
        analysis_df['potential_loss_usd'] = analysis_df['margin'] * (analysis_df['leveraged_risk_pct'] / 100)
        
        return analysis_df
    
    def analyze_trade_characteristics(self) -> Dict:
        """Analyze characteristics of the trades."""
        if self.trades_df is None:
            logger.error("No trades loaded. Call load_trades() first.")
            return {}
        
        analysis_df = self.calculate_risk_reward_ratios()
        
        # Basic statistics
        total_trades = len(analysis_df)
        
        # Risk/Reward statistics
        avg_risk_reward_ratio = analysis_df['risk_reward_ratio'].mean()
        median_risk_reward_ratio = analysis_df['risk_reward_ratio'].median()
        min_risk_reward_ratio = analysis_df['risk_reward_ratio'].min()
        max_risk_reward_ratio = analysis_df['risk_reward_ratio'].max()
        
        # Leverage statistics
        avg_leverage = analysis_df['leverage'].mean()
        median_leverage = analysis_df['leverage'].median()
        max_leverage = analysis_df['leverage'].max()
        min_leverage = analysis_df['leverage'].min()
        
        # Position size statistics
        avg_position_size = analysis_df['position_size_usd'].mean()
        total_position_value = analysis_df['position_size_usd'].sum()
        
        # Risk statistics
        avg_risk_pct = analysis_df['risk_pct'].mean()
        avg_reward_pct = analysis_df['reward_pct'].mean()
        avg_leveraged_risk_pct = analysis_df['leveraged_risk_pct'].mean()
        avg_leveraged_reward_pct = analysis_df['leveraged_reward_pct'].mean()
        
        # Potential P&L statistics
        total_potential_profit = analysis_df['potential_profit_usd'].sum()
        total_potential_loss = analysis_df['potential_loss_usd'].sum()
        avg_potential_profit = analysis_df['potential_profit_usd'].mean()
        avg_potential_loss = analysis_df['potential_loss_usd'].mean()
        
        # Strategy analysis - simplified
        strategy_stats = {}
        for strategy in analysis_df['strategy'].unique():
            strategy_data = analysis_df[analysis_df['strategy'] == strategy]
            strategy_stats[strategy] = {
                'count': len(strategy_data),
                'avg_risk_reward': strategy_data['risk_reward_ratio'].mean(),
                'avg_leverage': strategy_data['leverage'].mean(),
                'total_profit': strategy_data['potential_profit_usd'].sum(),
                'total_loss': strategy_data['potential_loss_usd'].sum()
            }
        
        # Symbol analysis - simplified
        symbol_stats = {}
        for symbol in analysis_df['symbol'].unique():
            symbol_data = analysis_df[analysis_df['symbol'] == symbol]
            symbol_stats[symbol] = {
                'count': len(symbol_data),
                'avg_risk_reward': symbol_data['risk_reward_ratio'].mean(),
                'avg_leverage': symbol_data['leverage'].mean(),
                'total_profit': symbol_data['potential_profit_usd'].sum(),
                'total_loss': symbol_data['potential_loss_usd'].sum()
            }
        
        return {
            'total_trades': total_trades,
            'avg_risk_reward_ratio': round(avg_risk_reward_ratio, 2),
            'median_risk_reward_ratio': round(median_risk_reward_ratio, 2),
            'min_risk_reward_ratio': round(min_risk_reward_ratio, 2),
            'max_risk_reward_ratio': round(max_risk_reward_ratio, 2),
            'avg_leverage': round(avg_leverage, 2),
            'median_leverage': round(median_leverage, 2),
            'max_leverage': max_leverage,
            'min_leverage': min_leverage,
            'avg_position_size': round(avg_position_size, 2),
            'total_position_value': round(total_position_value, 2),
            'avg_risk_pct': round(avg_risk_pct, 2),
            'avg_reward_pct': round(avg_reward_pct, 2),
            'avg_leveraged_risk_pct': round(avg_leveraged_risk_pct, 2),
            'avg_leveraged_reward_pct': round(avg_leveraged_reward_pct, 2),
            'total_potential_profit': round(total_potential_profit, 2),
            'total_potential_loss': round(total_potential_loss, 2),
            'avg_potential_profit': round(avg_potential_profit, 2),
            'avg_potential_loss': round(avg_potential_loss, 2),
            'strategy_statistics': strategy_stats,
            'symbol_statistics': symbol_stats
        }
    
    def simulate_trade_outcomes(self, win_rate: float = 0.5) -> Dict:
        """Simulate trade outcomes based on a given win rate."""
        if self.trades_df is None:
            logger.error("No trades loaded. Call load_trades() first.")
            return {}
        
        analysis_df = self.calculate_risk_reward_ratios()
        
        # Simulate outcomes
        np.random.seed(42)  # For reproducible results
        simulated_outcomes = np.random.random(len(analysis_df)) < win_rate
        
        # Calculate simulated P&L
        simulated_pnl = []
        for i, outcome in enumerate(simulated_outcomes):
            if outcome:  # Win
                pnl = analysis_df.iloc[i]['potential_profit_usd']
            else:  # Loss
                pnl = -analysis_df.iloc[i]['potential_loss_usd']
            simulated_pnl.append(pnl)
        
        analysis_df['simulated_outcome'] = simulated_outcomes
        analysis_df['simulated_pnl'] = simulated_pnl
        
        # Calculate statistics
        total_simulated_pnl = sum(simulated_pnl)
        winning_trades = sum(simulated_outcomes)
        losing_trades = len(simulated_outcomes) - winning_trades
        actual_win_rate = winning_trades / len(simulated_outcomes) * 100
        
        winning_pnl = [pnl for pnl, outcome in zip(simulated_pnl, simulated_outcomes) if outcome]
        losing_pnl = [pnl for pnl, outcome in zip(simulated_pnl, simulated_outcomes) if not outcome]
        
        avg_win = np.mean(winning_pnl) if winning_pnl else 0
        avg_loss = np.mean(losing_pnl) if losing_pnl else 0
        
        profit_factor = abs(sum(winning_pnl) / sum(losing_pnl)) if sum(losing_pnl) != 0 else float('inf')
        
        return {
            'assumed_win_rate': win_rate * 100,
            'actual_win_rate': round(actual_win_rate, 2),
            'total_simulated_pnl': round(total_simulated_pnl, 2),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'simulated_trades': analysis_df[['symbol', 'side', 'entry_price', 'stop_loss', 'take_profit', 
                                           'leverage', 'simulated_outcome', 'simulated_pnl']].copy()
        }
    
    def save_analysis(self, output_file: str = "chatgpt_trades_simple_analysis.csv"):
        """Save the analysis results to CSV file."""
        if self.trades_df is None:
            logger.error("No trades loaded. Call load_trades() first.")
            return
        
        analysis_df = self.calculate_risk_reward_ratios()
        analysis_df.to_csv(output_file, index=False)
        logger.info(f"Analysis results saved to {output_file}")
    
    def print_analysis_summary(self):
        """Print a comprehensive analysis summary."""
        if self.trades_df is None:
            logger.error("No trades loaded. Call load_trades() first.")
            return
        
        # Get analysis
        characteristics = self.analyze_trade_characteristics()
        
        print("\n" + "="*80)
        print("CHATGPT TRADES ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nTRADE CHARACTERISTICS:")
        print(f"  Total Trades: {characteristics['total_trades']}")
        print(f"  Average Position Size: ${characteristics['avg_position_size']}")
        print(f"  Total Position Value: ${characteristics['total_position_value']}")
        
        print(f"\nRISK/REWARD ANALYSIS:")
        print(f"  Average Risk/Reward Ratio: {characteristics['avg_risk_reward_ratio']}")
        print(f"  Median Risk/Reward Ratio: {characteristics['median_risk_reward_ratio']}")
        print(f"  Min Risk/Reward Ratio: {characteristics['min_risk_reward_ratio']}")
        print(f"  Max Risk/Reward Ratio: {characteristics['max_risk_reward_ratio']}")
        
        print(f"\nLEVERAGE ANALYSIS:")
        print(f"  Average Leverage: {characteristics['avg_leverage']}x")
        print(f"  Median Leverage: {characteristics['median_leverage']}x")
        print(f"  Min Leverage: {characteristics['min_leverage']}x")
        print(f"  Max Leverage: {characteristics['max_leverage']}x")
        
        print(f"\nRISK METRICS:")
        print(f"  Average Risk: {characteristics['avg_risk_pct']}%")
        print(f"  Average Reward: {characteristics['avg_reward_pct']}%")
        print(f"  Average Leveraged Risk: {characteristics['avg_leveraged_risk_pct']}%")
        print(f"  Average Leveraged Reward: {characteristics['avg_leveraged_reward_pct']}%")
        
        print(f"\nPOTENTIAL P&L (if all trades hit TP/SL):")
        print(f"  Total Potential Profit: ${characteristics['total_potential_profit']}")
        print(f"  Total Potential Loss: ${characteristics['total_potential_loss']}")
        print(f"  Average Potential Profit: ${characteristics['avg_potential_profit']}")
        print(f"  Average Potential Loss: ${characteristics['avg_potential_loss']}")
        
        print(f"\nSTRATEGY BREAKDOWN:")
        for strategy, stats in characteristics['strategy_statistics'].items():
            count = stats['count']
            avg_rr = round(stats['avg_risk_reward'], 2)
            avg_lev = round(stats['avg_leverage'], 2)
            total_profit = round(stats['total_profit'], 2)
            total_loss = round(stats['total_loss'], 2)
            print(f"  {strategy}: {count} trades, R/R: {avg_rr}, Leverage: {avg_lev}x, P&L: ${total_profit}/${total_loss}")
        
        print(f"\nSYMBOL BREAKDOWN:")
        for symbol, stats in characteristics['symbol_statistics'].items():
            count = stats['count']
            avg_rr = round(stats['avg_risk_reward'], 2)
            avg_lev = round(stats['avg_leverage'], 2)
            total_profit = round(stats['total_profit'], 2)
            total_loss = round(stats['total_loss'], 2)
            print(f"  {symbol}: {count} trades, R/R: {avg_rr}, Leverage: {avg_lev}x, P&L: ${total_profit}/${total_loss}")
        
        print("\n" + "="*80)
    
    def print_simulation_results(self, win_rates: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]):
        """Print simulation results for different win rates."""
        print("\n" + "="*80)
        print("TRADE OUTCOME SIMULATIONS")
        print("="*80)
        
        for win_rate in win_rates:
            simulation = self.simulate_trade_outcomes(win_rate)
            
            print(f"\nSimulation with {simulation['assumed_win_rate']}% win rate:")
            print(f"  Actual Win Rate: {simulation['actual_win_rate']}%")
            print(f"  Total P&L: ${simulation['total_simulated_pnl']}")
            print(f"  Winning Trades: {simulation['winning_trades']}")
            print(f"  Losing Trades: {simulation['losing_trades']}")
            print(f"  Average Win: ${simulation['avg_win']}")
            print(f"  Average Loss: ${simulation['avg_loss']}")
            print(f"  Profit Factor: {simulation['profit_factor']}")
        
        print("\n" + "="*80)

def main():
    """Main function to run the analysis."""
    analyzer = SimpleChatGPTTradesAnalyzer()
    
    # Load trades
    if not analyzer.load_trades():
        logger.error("Failed to load trades. Exiting.")
        return
    
    # Print analysis summary
    analyzer.print_analysis_summary()
    
    # Print simulation results
    analyzer.print_simulation_results()
    
    # Save analysis
    analyzer.save_analysis()
    
    logger.info("Simple analysis completed successfully!")
    print("\nNote: This is a basic analysis without historical price data.")
    print("To get actual trade outcomes, run the full analyzer with API access.")

if __name__ == "__main__":
    main()
