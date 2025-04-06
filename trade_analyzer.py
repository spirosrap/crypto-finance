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
        
        # Convert timestamp to datetime with format that includes UTC
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S UTC')
        
        # Convert percentage columns
        percentage_columns = ['Outcome %']
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
        
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': win_rate,
            'Win/Loss Ratio': win_loss_ratio,
            'Average Profit per Trade (%)': avg_profit
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
            'Average Leverage': avg_leverage
        }
    
    def calculate_trade_duration_stats(self) -> dict:
        """Calculate detailed statistics about trade durations."""
        # Calculate duration between trades
        self.df['Duration'] = pd.to_datetime(self.df['Timestamp']).diff()
        
        # Convert duration to minutes
        durations_minutes = self.df['Duration'].dt.total_seconds() / 60
        
        # Calculate statistics
        stats = {
            'Average Duration (minutes)': durations_minutes.mean(),
            'Median Duration (minutes)': durations_minutes.median(),
            'Min Duration (minutes)': durations_minutes.min(),
            'Max Duration (minutes)': durations_minutes.max(),
            'Duration Std Dev (minutes)': durations_minutes.std(),
            'Trades per Day': len(self.df) / ((self.df['Timestamp'].max() - self.df['Timestamp'].min()).days + 1)
        }
        
        return stats
    
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
        duration_stats = self.calculate_trade_duration_stats()
        
        # ANSI color codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
        
        # Calculate additional metrics
        total_trades = basic_metrics['Total Trades']
        win_rate = basic_metrics['Win Rate']
        avg_profit = basic_metrics['Average Profit per Trade (%)']
        
        # Calculate expectancy
        avg_win = self.df[self.df['Outcome'] == 'SUCCESS']['Outcome %'].mean()
        avg_loss = self.df[self.df['Outcome'] == 'STOP LOSS']['Outcome %'].mean()
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        print(f"\n{BOLD}{UNDERLINE}📊 Trading Performance Report{END}\n")
        
        # Summary Section
        print(f"{BOLD}📈 Summary{END}")
        print(f"{'=' * 50}")
        print(f"Period: {self.df['Timestamp'].min().date()} to {self.df['Timestamp'].max().date()}")
        print(f"Total Trades: {GREEN}{total_trades}{END}")
        print(f"Win Rate: {GREEN if win_rate >= 0.5 else RED}{win_rate:.2%}{END}")
        print(f"Total Dollar Profit: {GREEN if dollar_profits['Total Dollar Profit'] >= 0 else RED}${dollar_profits['Total Dollar Profit']:.2f}{END}")
        print(f"Average Trade Duration: {duration_stats['Average Duration (minutes)']:.0f} minutes")
        print(f"Trades per Day: {BLUE}{duration_stats['Trades per Day']:.1f}{END}")
        print(f"{'=' * 50}\n")
        
        # Performance Metrics
        print(f"{BOLD}🎯 Performance Metrics{END}")
        print(f"{'=' * 50}")
        print(f"Average % Return per Trade: {GREEN if avg_profit >= 0 else RED}{avg_profit:.2f}%{END} (Not Cumulative)")
        print(f"Sum of Outcome %s: {GREEN if self.df['Outcome %'].sum() >= 0 else RED}{self.df['Outcome %'].sum():.2f}%{END} (Not Actual Profit)")
        print(f"Win/Loss Ratio: {GREEN}{basic_metrics['Win/Loss Ratio']:.2f}{END}")
        print(f"Expectancy: {GREEN if expectancy >= 0 else RED}{expectancy:.2f}%{END}")
        print(f"Sharpe Ratio: {GREEN if sharpe_ratio >= 1 else YELLOW if sharpe_ratio >= 0 else RED}{sharpe_ratio:.2f}{END}")
        print(f"{'=' * 50}\n")
        
        # Risk Metrics
        print(f"{BOLD}⚠️ Risk Metrics{END}")
        print(f"{'=' * 50}")
        print(f"Maximum Drawdown: {RED}{max_drawdown:.2f}%{END}")
        print(f"Current Drawdown: {RED if current_drawdown < 0 else GREEN}{current_drawdown:.2f}%{END}")
        print(f"Standard Deviation: {YELLOW}{risk_metrics['Standard Deviation']:.2f}%{END}")
        print(f"Average R/R Ratio: {BLUE}{risk_metrics['Average R/R Ratio']:.2f}{END}")
        if risk_metrics['Average Leverage'] > 0:
            print(f"Average Leverage: {YELLOW}{risk_metrics['Average Leverage']:.2f}x{END}")
        print(f"{'=' * 50}\n")
        
        # Dollar Performance
        print(f"{BOLD}💰 Dollar Performance{END}")
        print(f"{'=' * 50}")
        print(f"Total Dollar Profit: {GREEN if dollar_profits['Total Dollar Profit'] >= 0 else RED}${dollar_profits['Total Dollar Profit']:.2f}{END}")
        print(f"Average Dollar Profit per Trade: {GREEN if dollar_profits['Average Dollar Profit per Trade'] >= 0 else RED}${dollar_profits['Average Dollar Profit per Trade']:.2f}{END}")
        print(f"Profit Factor: {GREEN if dollar_profits['Profit Factor'] >= 1.5 else YELLOW if dollar_profits['Profit Factor'] >= 1 else RED}{dollar_profits['Profit Factor']:.2f}{END}")
        print(f"{'=' * 50}\n")
        
        # Leveraged Performance
        print(f"{BOLD}⚡ Leveraged Performance{END}")
        print(f"{'=' * 50}")
        print(f"Total Leveraged Profit: {GREEN if leveraged_profits['Total Leveraged Dollar Profit'] >= 0 else RED}${leveraged_profits['Total Leveraged Dollar Profit']:.2f}{END}")
        print(f"Average Leveraged Profit per Trade: {GREEN if leveraged_profits['Average Leveraged Dollar Profit per Trade'] >= 0 else RED}${leveraged_profits['Average Leveraged Dollar Profit per Trade']:.2f}{END}")
        print(f"Leveraged Profit Factor: {GREEN if leveraged_profits['Leveraged Profit Factor'] >= 1.5 else YELLOW if leveraged_profits['Leveraged Profit Factor'] >= 1 else RED}{leveraged_profits['Leveraged Profit Factor']:.2f}{END}")
        print(f"{'=' * 50}\n")
        
        # Drawdown Analysis
        if drawdown_periods:
            print(f"{BOLD}📉 Largest Drawdown Periods{END}")
            print(f"{'=' * 50}")
            for period in drawdown_periods[:3]:  # Show top 3 drawdowns
                print(f"From {period['start_date'].date()} to {period['end_date'].date()}: {RED}{period['drawdown']:.2f}%{END}")
            print(f"{'=' * 50}\n")
        
        # Trading Style Analysis
        print(f"{BOLD}🎨 Trading Style Analysis{END}")
        print(f"{'=' * 50}")
        avg_rr = risk_metrics['Average R/R Ratio']
        print(f"Average Risk/Reward: {BLUE}{avg_rr:.2f}{END}")
        print(f"Trading Style: {BLUE}{'Conservative' if avg_rr > 2 else 'Moderate' if avg_rr > 1.5 else 'Aggressive'}{END}")
        print(f"{'=' * 50}\n")

if __name__ == "__main__":
    analyzer = TradingAnalyzer("automated_trades.csv")
    analyzer.generate_report()
    current_drawdown = analyzer.get_current_drawdown()
    print(f"Current Drawdown: {current_drawdown:.2f}%") 