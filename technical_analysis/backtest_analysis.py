import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np
import logging
import os

class BacktestAnalysis:
    """Class for analyzing and visualizing backtest results."""
    
    def __init__(self, trades_history: List[Dict], initial_capital: float):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        
        # Convert trades history to DataFrame
        self.trades_df = pd.DataFrame(trades_history)
        
        # Check if DataFrame is empty
        if self.trades_df.empty:
            self.logger.warning("No trades found in history")
            return
            
        # Print column names for debugging
        self.logger.debug(f"Available columns: {self.trades_df.columns.tolist()}")
        
        try:
            # Map expected columns to actual columns
            self._map_columns()
            
            # Convert timestamps
            self._convert_timestamps()
            
            # Calculate cumulative metrics
            self._calculate_cumulative_metrics()
            
        except Exception as e:
            self.logger.error(f"Error initializing BacktestAnalysis: {str(e)}")
            raise
    
    def _map_columns(self):
        """Map expected column names to actual column names in the DataFrame."""
        # Define column mappings (expected_name: possible_names)
        column_mappings = {
            'entry_time': ['entry_time', 'timestamp', 'time', 'start_time'],
            'exit_time': ['exit_time', 'end_time', 'close_time'],
            'entry_price': ['entry_price', 'open_price', 'open'],
            'exit_price': ['exit_price', 'close_price', 'close'],
            'position_type': ['position_type', 'type', 'side'],
            'position_size': ['position_size', 'size', 'amount'],
            'pnl': ['pnl', 'profit_loss', 'pl'],
            'pnl_pct': ['pnl_pct', 'return', 'returns']
        }
        
        # Create new columns with standardized names
        for expected_name, possible_names in column_mappings.items():
            found = False
            for name in possible_names:
                if name in self.trades_df.columns:
                    self.trades_df[expected_name] = self.trades_df[name]
                    found = True
                    break
            if not found:
                self.logger.warning(f"Could not find column for {expected_name}")
                # Set default values if column not found
                if expected_name in ['pnl', 'pnl_pct', 'position_size']:
                    self.trades_df[expected_name] = 0.0
                elif expected_name in ['position_type']:
                    self.trades_df[expected_name] = 'unknown'
                else:
                    self.trades_df[expected_name] = pd.NaT
    
    def _convert_timestamps(self):
        """Convert timestamp columns to datetime."""
        timestamp_columns = ['entry_time', 'exit_time']
        
        for col in timestamp_columns:
            if col in self.trades_df.columns:
                try:
                    # Handle different timestamp formats
                    self.trades_df[col] = pd.to_datetime(self.trades_df[col], unit='s', errors='coerce')
                    if self.trades_df[col].isna().all():
                        # Try parsing as regular datetime if unix timestamp fails
                        self.trades_df[col] = pd.to_datetime(self.trades_df[col], errors='coerce')
                except Exception as e:
                    self.logger.error(f"Error converting {col} to datetime: {str(e)}")
                    self.trades_df[col] = pd.NaT
    
    def _calculate_cumulative_metrics(self):
        """Calculate cumulative returns and drawdown."""
        try:
            if 'pnl_pct' not in self.trades_df.columns:
                if 'pnl' in self.trades_df.columns and 'position_size' in self.trades_df.columns:
                    # Calculate percentage returns if we have PnL and position size
                    self.trades_df['pnl_pct'] = self.trades_df['pnl'] / self.trades_df['position_size']
                else:
                    self.logger.error("Cannot calculate returns: missing required columns")
                    return
            
            # Calculate cumulative metrics
            self.trades_df['cumulative_returns'] = (1 + self.trades_df['pnl_pct']).cumprod()
            self.trades_df['cumulative_capital'] = self.initial_capital * self.trades_df['cumulative_returns']
            
            # Calculate drawdown
            rolling_max = self.trades_df['cumulative_capital'].expanding().max()
            self.trades_df['drawdown'] = (self.trades_df['cumulative_capital'] - rolling_max) / rolling_max
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def plot_equity_curve(self):
        """Plot equity curve with drawdown."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(self.trades_df['exit_time'], self.trades_df['cumulative_capital'])
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Capital')
        ax1.grid(True)
        
        # Plot drawdown
        ax2.fill_between(self.trades_df['exit_time'], 0, self.trades_df['drawdown'], color='red', alpha=0.3)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_monthly_returns(self):
        """Plot monthly returns heatmap."""
        try:
            # Calculate monthly returns
            self.trades_df['month'] = self.trades_df['exit_time'].dt.to_period('M')
            monthly_returns = self.trades_df.groupby('month')['pnl_pct'].sum()
            
            if len(monthly_returns) < 2:
                self.logger.warning("Not enough monthly data for heatmap")
                # Create a simple bar plot instead
                fig, ax = plt.subplots(figsize=(10, 6))
                monthly_returns.plot(kind='bar', ax=ax)
                ax.set_title('Monthly Returns')
                ax.set_xlabel('Month')
                ax.set_ylabel('Return')
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                return fig
            
            # Calculate number of years and months
            num_months = len(monthly_returns)
            num_years = (num_months + 11) // 12  # Round up to nearest year
            
            # Pad the data to fill complete years
            padding_months = num_years * 12 - num_months
            if padding_months > 0:
                padding = pd.Series([0] * padding_months)
                monthly_returns = pd.concat([monthly_returns, padding])
            
            # Reshape data for heatmap
            monthly_returns_matrix = monthly_returns.values.reshape(num_years, 12)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(monthly_returns_matrix, 
                       cmap='RdYlGn',
                       center=0,
                       annot=True,
                       fmt='.2%',
                       cbar_kws={'label': 'Monthly Return'})
            
            ax.set_title('Monthly Returns Heatmap')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating monthly returns plot: {str(e)}")
            # Create a simple line plot as fallback
            fig, ax = plt.subplots(figsize=(10, 6))
            self.trades_df.set_index('exit_time')['pnl_pct'].cumsum().plot(ax=ax)
            ax.set_title('Cumulative Returns')
            ax.grid(True)
            return fig
    
    def plot_trade_distribution(self):
        """Plot trade profit/loss distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(data=self.trades_df, x='pnl', bins=50, ax=ax)
        ax.axvline(x=0, color='r', linestyle='--')
        
        ax.set_title('Trade P&L Distribution')
        ax.set_xlabel('Profit/Loss')
        ax.set_ylabel('Frequency')
        
        return fig
    
    def generate_report(self, output_dir: str = '.'):
        """Generate and save comprehensive analysis report."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create plots with error handling
            try:
                equity_fig = self.plot_equity_curve()
                equity_fig.savefig(f'{output_dir}/equity_curve.png')
            except Exception as e:
                self.logger.error(f"Could not generate equity curve: {str(e)}")
                
            try:
                monthly_fig = self.plot_monthly_returns()
                monthly_fig.savefig(f'{output_dir}/monthly_returns.png')
            except Exception as e:
                self.logger.error(f"Could not generate monthly returns plot: {str(e)}")
                
            try:
                dist_fig = self.plot_trade_distribution()
                dist_fig.savefig(f'{output_dir}/trade_distribution.png')
            except Exception as e:
                self.logger.error(f"Could not generate trade distribution plot: {str(e)}")
            
            # Calculate additional metrics with error handling
            metrics = {}
            try:
                metrics = {
                    'Total Trades': len(self.trades_df),
                    'Win Rate': len(self.trades_df[self.trades_df['pnl'] > 0]) / len(self.trades_df) if len(self.trades_df) > 0 else 0,
                    'Average Win': self.trades_df[self.trades_df['pnl'] > 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] > 0]) > 0 else 0,
                    'Average Loss': self.trades_df[self.trades_df['pnl'] < 0]['pnl'].mean() if len(self.trades_df[self.trades_df['pnl'] < 0]) > 0 else 0,
                    'Largest Win': self.trades_df['pnl'].max() if len(self.trades_df) > 0 else 0,
                    'Largest Loss': self.trades_df['pnl'].min() if len(self.trades_df) > 0 else 0,
                    'Max Drawdown': self.trades_df['drawdown'].min() if 'drawdown' in self.trades_df else 0,
                    'Final Capital': self.trades_df['cumulative_capital'].iloc[-1] if 'cumulative_capital' in self.trades_df else self.initial_capital,
                    'Total Return': ((self.trades_df['cumulative_capital'].iloc[-1] / self.initial_capital) - 1) if 'cumulative_capital' in self.trades_df else 0
                }
            except Exception as e:
                self.logger.error(f"Error calculating metrics: {str(e)}")
            
            # Save metrics to CSV
            try:
                pd.Series(metrics).to_csv(f'{output_dir}/backtest_metrics.csv')
            except Exception as e:
                self.logger.error(f"Could not save metrics to CSV: {str(e)}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {}