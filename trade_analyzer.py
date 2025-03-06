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
        """Load trading data from CSV file."""
        # Read the CSV file
        df = pd.read_csv(self.file_path)
        
        # Convert timestamp to datetime with explicit format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
        
        # Convert percentage columns
        percentage_columns = ['Probability', 'Outcome %']
        for col in percentage_columns:
            if df[col].dtype == 'object':  # Only apply string operations if column contains strings
                df[col] = pd.to_numeric(df[col].str.replace('%', '').str.strip(), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = ['ENTRY', 'Take Profit', 'Stop Loss', 'R/R Ratio']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle Leverage column specifically (remove 'x' if present)
        if df['Leverage'].dtype == 'object':  # Only apply string operations if column contains strings
            df['Leverage'] = pd.to_numeric(df['Leverage'].str.replace('x', '').str.strip(), errors='coerce')
        else:
            df['Leverage'] = pd.to_numeric(df['Leverage'], errors='coerce')
        
        # Handle Margin column
        df['Margin'] = pd.to_numeric(df['Margin'], errors='coerce')
        
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
    
    def calculate_dollar_profits(self) -> dict:
        """Calculate profits in dollar terms using margin, leverage, and outcome percentage."""
        # Calculate dollar profit for each trade
        self.df['Dollar Profit'] = self.df.apply(
            lambda row: row['Margin'] * (row['Outcome %'] / 100), axis=1
        )
        
        # Calculate total dollar profit
        total_dollar_profit = self.df['Dollar Profit'].sum()
        
        # Calculate average dollar profit per trade
        avg_dollar_profit = self.df['Dollar Profit'].mean()
        
        # Calculate dollar profit for winning trades
        winning_trades_profit = self.df[self.df['Outcome'] == 'SUCCESS']['Dollar Profit'].sum()
        
        # Calculate dollar loss for losing trades
        losing_trades_loss = self.df[self.df['Outcome'] == 'STOP LOSS']['Dollar Profit'].sum()
        
        # Calculate profit factor (total profit / total loss)
        profit_factor = abs(winning_trades_profit / losing_trades_loss) if losing_trades_loss != 0 else float('inf')
        
        return {
            'Total Dollar Profit': total_dollar_profit,
            'Average Dollar Profit per Trade': avg_dollar_profit,
            'Winning Trades Dollar Profit': winning_trades_profit,
            'Losing Trades Dollar Loss': losing_trades_loss,
            'Profit Factor': profit_factor
        }
    
    def calculate_leveraged_dollar_profits(self) -> dict:
        """Calculate profits in dollar terms considering margin, leverage, and outcome percentage."""
        # Calculate leveraged dollar profit for each trade
        self.df['Leveraged Dollar Profit'] = self.df.apply(
            lambda row: row['Margin'] * row['Leverage'] * (row['Outcome %'] / 100), axis=1
        )
        
        # Calculate total leveraged dollar profit
        total_leveraged_profit = self.df['Leveraged Dollar Profit'].sum()
        
        # Calculate average leveraged dollar profit per trade
        avg_leveraged_profit = self.df['Leveraged Dollar Profit'].mean()
        
        # Calculate leveraged dollar profit for winning trades
        winning_leveraged_profit = self.df[self.df['Outcome'] == 'SUCCESS']['Leveraged Dollar Profit'].sum()
        
        # Calculate leveraged dollar loss for losing trades
        losing_leveraged_loss = self.df[self.df['Outcome'] == 'STOP LOSS']['Leveraged Dollar Profit'].sum()
        
        # Calculate leveraged profit factor
        leveraged_profit_factor = abs(winning_leveraged_profit / losing_leveraged_loss) if losing_leveraged_loss != 0 else float('inf')
        
        return {
            'Total Leveraged Dollar Profit': total_leveraged_profit,
            'Average Leveraged Dollar Profit per Trade': avg_leveraged_profit,
            'Winning Trades Leveraged Profit': winning_leveraged_profit,
            'Losing Trades Leveraged Loss': losing_leveraged_loss,
            'Leveraged Profit Factor': leveraged_profit_factor
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
            elif in_drawdown and (drawdowns.iloc[i] >= 0 or i == len(drawdowns) - 1):
                # End the drawdown period if we recover or reach the end of data
                in_drawdown = False
                end_idx = i if drawdowns.iloc[i] >= 0 else i + 1
                period_drawdown = drawdowns.iloc[start_idx:end_idx].min()
                
                # Only add significant drawdowns (e.g., more than 1%)
                if period_drawdown < -1:
                    drawdown_periods.append({
                        'start_date': self.df['Timestamp'].iloc[start_idx],
                        'end_date': self.df['Timestamp'].iloc[end_idx - 1],
                        'drawdown': period_drawdown
                    })
        
        # Sort drawdown periods by magnitude
        drawdown_periods.sort(key=lambda x: x['drawdown'])
        
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
    
    def get_current_drawdown(self) -> float:
        """Calculate the current drawdown from the peak."""
        cumulative_returns = (1 + self.df['Outcome %'] / 100).cumprod()
        peak = cumulative_returns.max()
        current_value = cumulative_returns.iloc[-1]
        current_drawdown = ((current_value - peak) / peak) * 100
        return current_drawdown
    
    def generate_report(self) -> None:
        """Generate and print a comprehensive trading report."""
        basic_metrics = self.calculate_basic_metrics()
        dollar_profits = self.calculate_dollar_profits()
        leveraged_profits = self.calculate_leveraged_dollar_profits()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown, drawdown_periods = self.calculate_max_drawdown()
        risk_metrics = self.calculate_risk_metrics()
        current_drawdown = self.get_current_drawdown()
        
        print("\n=== Trading Performance Report ===\n")
        
        print("Basic Metrics:")
        print(f"Total Trades: {basic_metrics['Total Trades']}")
        print(f"Win Rate: {basic_metrics['Win Rate']:.2%}")
        print(f"Win/Loss Ratio: {basic_metrics['Win/Loss Ratio']:.2f}")
        print(f"Average Profit per Trade: {basic_metrics['Average Profit per Trade (%)']:.2f}%")
        print(f"Total Profit: {basic_metrics['Total Profit (%)']:.2f}%")
        
        print("\nDollar Profit Metrics (Without Leverage):")
        print(f"Total Dollar Profit: ${dollar_profits['Total Dollar Profit']:.2f}")
        print(f"Average Dollar Profit per Trade: ${dollar_profits['Average Dollar Profit per Trade']:.2f}")
        print(f"Winning Trades Dollar Profit: ${dollar_profits['Winning Trades Dollar Profit']:.2f}")
        print(f"Losing Trades Dollar Loss: ${dollar_profits['Losing Trades Dollar Loss']:.2f}")
        print(f"Profit Factor (Gross Profit / Gross Loss): {dollar_profits['Profit Factor']:.2f}")
        
        print("\nLeveraged Dollar Profit Metrics:")
        print(f"Total Leveraged Dollar Profit: ${leveraged_profits['Total Leveraged Dollar Profit']:.2f}")
        print(f"Average Leveraged Dollar Profit per Trade: ${leveraged_profits['Average Leveraged Dollar Profit per Trade']:.2f}")
        print(f"Winning Trades Leveraged Profit: ${leveraged_profits['Winning Trades Leveraged Profit']:.2f}")
        print(f"Losing Trades Leveraged Loss: ${leveraged_profits['Losing Trades Leveraged Loss']:.2f}")
        print(f"Leveraged Profit Factor: {leveraged_profits['Leveraged Profit Factor']:.2f}")
        
        print("\nRisk Metrics:")
        print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Current Drawdown: {current_drawdown:.2f}%")
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
    analyzer = TradingAnalyzer("automated_trades.csv")
    analyzer.generate_report()
    current_drawdown = analyzer.get_current_drawdown()
    print(f"Current Drawdown: {current_drawdown:.2f}%") 