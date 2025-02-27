import pandas as pd
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

class TradingAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = self._load_data()
        self.risk_free_rate = 0.0525  # 5.25% annual risk-free rate
        
    def _load_data(self) -> pd.DataFrame:
        """Load trading data from markdown file."""
        # Read all lines from the file
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the table header line (contains '|')
        header_idx = 0
        for i, line in enumerate(lines):
            if '|' in line and '---' not in line and 'No.' in line:
                header_idx = i
                break
        
        # Skip the header and separator lines, keep only data rows
        data_lines = [line.strip() for line in lines[header_idx:] if '|' in line and '---' not in line]
        
        # Parse the header
        headers = [col.strip() for col in data_lines[0].split('|')[1:-1]]
        
        # Parse the data rows
        data = []
        for line in data_lines[1:]:
            row = [col.strip() for col in line.split('|')[1:-1]]
            if len(row) == len(headers):  # Only include rows that match header length
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        
        # Convert timestamp to datetime with explicit format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
        
        # Convert numeric columns with percentage values
        percentage_columns = ['Probability', 'Outcome %']
        for col in percentage_columns:
            df[col] = pd.to_numeric(df[col].str.replace('%', '').str.strip(), errors='coerce')
        
        # Convert other numeric columns
        numeric_columns = ['ENTRY', 'Take Profit', 'Stop Loss', 'R/R Ratio']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col].str.strip(), errors='coerce')
        
        # Handle Leverage column specifically (remove 'x' if present)
        df['Leverage'] = pd.to_numeric(df['Leverage'].str.replace('x', '').str.strip(), errors='coerce')
        
        # Handle Margin column
        df['Margin'] = pd.to_numeric(df['Margin'].str.strip(), errors='coerce')
        
        return df
    
    def calculate_basic_metrics(self) -> dict:
        """Calculate basic trading metrics."""
        total_trades = len(self.df)
        winning_trades = len(self.df[self.df['Outcome'] == 'SUCCESS'])
        losing_trades = len(self.df[self.df['Outcome'] == 'STOP LOSS'])
        
        win_rate = winning_trades / (winning_trades + losing_trades)
        win_loss_ratio = winning_trades / losing_trades if losing_trades > 0 else float('inf')
        
        avg_profit = self.df['Outcome %'].mean()
        total_profit = self.df['Outcome %'].sum()
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': win_rate,
            'Win/Loss Ratio': win_loss_ratio,
            'Average Profit per Trade (%)': avg_profit,
            'Total Profit (%)': total_profit
        }
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate annualized Sharpe ratio."""
        returns = self.df['Outcome %'] / 100  # Convert percentage to decimal
        excess_returns = returns - (self.risk_free_rate / 365)  # Daily excess returns
        
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(365) * (returns.mean() / returns.std())
            return sharpe_ratio
        return 0
    
    def calculate_max_drawdown(self) -> Tuple[float, List[dict]]:
        """Calculate maximum drawdown and drawdown periods."""
        cumulative_returns = (1 + self.df['Outcome %'] / 100).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max * 100
        
        max_drawdown = drawdowns.min()
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i in range(len(drawdowns)):
            if not in_drawdown and drawdowns.iloc[i] < 0:
                in_drawdown = True
                start_idx = i
            elif in_drawdown and drawdowns.iloc[i] >= 0:
                in_drawdown = False
                drawdown_periods.append({
                    'start_date': self.df['Timestamp'].iloc[start_idx],
                    'end_date': self.df['Timestamp'].iloc[i],
                    'drawdown': drawdowns.iloc[start_idx:i].min()
                })
        
        return max_drawdown, drawdown_periods
    
    def calculate_risk_metrics(self) -> dict:
        """Calculate various risk metrics."""
        returns = self.df['Outcome %'] / 100
        
        # Add error handling for leverage calculation
        avg_leverage = self.df['Leverage'].mean()
        if pd.isna(avg_leverage):
            avg_leverage = 0  # or another appropriate default value
        
        return {
            'Standard Deviation': returns.std() * 100,
            'Average R/R Ratio': self.df['R/R Ratio'].mean(),
            'Average Probability': self.df['Probability'].mean(),
            'Average Leverage': avg_leverage
        }
    
    def generate_report(self) -> None:
        """Generate and print a comprehensive trading report."""
        basic_metrics = self.calculate_basic_metrics()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown, drawdown_periods = self.calculate_max_drawdown()
        risk_metrics = self.calculate_risk_metrics()
        
        print("\n=== Trading Performance Report ===\n")
        
        print("Basic Metrics:")
        print(f"Total Trades: {basic_metrics['Total Trades']}")
        print(f"Win Rate: {basic_metrics['Win Rate']:.2%}")
        print(f"Win/Loss Ratio: {basic_metrics['Win/Loss Ratio']:.2f}")
        print(f"Average Profit per Trade: {basic_metrics['Average Profit per Trade (%)']:.2f}%")
        print(f"Total Profit: {basic_metrics['Total Profit (%)']:.2f}%")
        
        print("\nRisk Metrics:")
        print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Standard Deviation: {risk_metrics['Standard Deviation']:.2f}%")
        print(f"Average R/R Ratio: {risk_metrics['Average R/R Ratio']:.2f}")
        print(f"Average Trade Probability: {risk_metrics['Average Probability']:.2f}%")
        if risk_metrics['Average Leverage'] > 0:
            print(f"Average Leverage: {risk_metrics['Average Leverage']:.2f}x")
        else:
            print("Average Leverage: N/A")
        
        print("\nLargest Drawdown Periods:")
        for period in drawdown_periods:
            print(f"From {period['start_date'].date()} to {period['end_date'].date()}: {period['drawdown']:.2f}%")

if __name__ == "__main__":
    analyzer = TradingAnalyzer("automated_trades.md")
    analyzer.generate_report() 