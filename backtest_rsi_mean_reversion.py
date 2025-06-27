

import time
from datetime import datetime, timedelta, UTC
import logging
import pandas as pd
import pandas_ta as ta
import argparse
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tabulate import tabulate
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants from rsi_mean_reversion_bot
PRODUCT_ID = "BTC-PERP-INTX"
GRANULARITY = "FIFTEEN_MINUTE"
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RISK_REWARD_RATIO = 1.5

@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""
    product_id: str = PRODUCT_ID
    initial_balance: float = 10000
    leverage: int = 10
    start_date: Optional[str] = None
    end_date: Optional[str] = None

@dataclass
class Trade:
    """Represents a single trade in the backtest."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    profit: float
    side: str  # 'BUY' or 'SELL'
    rsi_at_entry: float
    exit_reason: str  # 'TP' or 'SL'

@dataclass
class BacktestResults:
    """Results from running the backtest."""
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    profit_factor: float
    trades: List[Trade]
    csv_filename: Optional[str]

def parse_args() -> BacktestConfig:
    """Parse command line arguments and return a BacktestConfig object."""
    parser = argparse.ArgumentParser(description='Backtest RSI Mean Reversion Bot')
    parser.add_argument('--product_id', type=str, default=PRODUCT_ID,
                      help=f'Product ID to trade (default: {PRODUCT_ID})')
    parser.add_argument('--initial_balance', type=float, default=10000,
                      help='Initial balance for backtesting')
    parser.add_argument('--leverage', type=int, default=10,
                      help='Trading leverage')
    parser.add_argument('--start_date', type=str, default=None,
                      help='Start date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                      help='End date for backtest (format: YYYY-MM-DD)')
    
    args = parser.parse_args()
    return BacktestConfig(**vars(args))

def fetch_candles(cb: CoinbaseService, product_id: str, 
                 start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical candle data for the specified period."""
    if start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=UTC)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=UTC)
    else:
        # Default to last 30 days
        now = datetime.now(UTC)
        start = now - timedelta(days=30)
        end = now
    
    granularity_map = {
        "FIFTEEN_MINUTE": 900
    }
    
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, GRANULARITY)
    df = pd.DataFrame(raw_data)
    
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(df['start'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df

def backtest(df: pd.DataFrame, config: BacktestConfig) -> BacktestResults:
    """Run backtest on historical data and return results."""
    balance = config.initial_balance
    trades = []
    current_trade = None

    df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
    
    for i in range(RSI_PERIOD, len(df)):
        rsi_value = df['rsi'].iloc[i-1]
        current_price = df['close'].iloc[i]
        high_price = df['high'].iloc[i]
        low_price = df['low'].iloc[i]
        signal_candle_high = df['high'].iloc[i-1]
        signal_candle_low = df['low'].iloc[i-1]

        # Exit condition check
        if current_trade:
            if current_trade['side'] == 'BUY':
                if high_price >= current_trade['tp_price']:
                    profit = (current_trade['tp_price'] - current_trade['entry_price']) * (config.initial_balance * 0.1 / current_trade['entry_price']) * config.leverage
                    balance += profit
                    trades.append(Trade(
                        entry_time=current_trade['entry_time'], exit_time=df.index[i],
                        entry_price=current_trade['entry_price'], exit_price=current_trade['tp_price'],
                        profit=profit, side='BUY', rsi_at_entry=current_trade['rsi_at_entry'], exit_reason='TP'
                    ))
                    current_trade = None
                elif low_price <= current_trade['sl_price']:
                    loss = (current_trade['sl_price'] - current_trade['entry_price']) * (config.initial_balance * 0.1 / current_trade['entry_price']) * config.leverage
                    balance += loss
                    trades.append(Trade(
                        entry_time=current_trade['entry_time'], exit_time=df.index[i],
                        entry_price=current_trade['entry_price'], exit_price=current_trade['sl_price'],
                        profit=loss, side='BUY', rsi_at_entry=current_trade['rsi_at_entry'], exit_reason='SL'
                    ))
                    current_trade = None
            elif current_trade['side'] == 'SELL':
                if low_price <= current_trade['tp_price']:
                    profit = (current_trade['entry_price'] - current_trade['tp_price']) * (config.initial_balance * 0.1 / current_trade['entry_price']) * config.leverage
                    balance += profit
                    trades.append(Trade(
                        entry_time=current_trade['entry_time'], exit_time=df.index[i],
                        entry_price=current_trade['entry_price'], exit_price=current_trade['tp_price'],
                        profit=profit, side='SELL', rsi_at_entry=current_trade['rsi_at_entry'], exit_reason='TP'
                    ))
                    current_trade = None
                elif high_price >= current_trade['sl_price']:
                    loss = (current_trade['entry_price'] - current_trade['sl_price']) * (config.initial_balance * 0.1 / current_trade['entry_price']) * config.leverage
                    balance += loss
                    trades.append(Trade(
                        entry_time=current_trade['entry_time'], exit_time=df.index[i],
                        entry_price=current_trade['entry_price'], exit_price=current_trade['sl_price'],
                        profit=loss, side='SELL', rsi_at_entry=current_trade['rsi_at_entry'], exit_reason='SL'
                    ))
                    current_trade = None

        # Entry condition check
        if not current_trade:
            side = None
            stop_loss = None
            take_profit = None

            if rsi_value < RSI_OVERSOLD:
                side = "BUY"
                risk_amount = current_price - signal_candle_low
                stop_loss = current_price - risk_amount * 1.1
                take_profit = current_price + risk_amount * RISK_REWARD_RATIO
            elif rsi_value > RSI_OVERBOUGHT:
                side = "SELL"
                risk_amount = signal_candle_high - current_price
                stop_loss = current_price + risk_amount * 1.1
                take_profit = current_price - risk_amount * RISK_REWARD_RATIO

            if side:
                current_trade = {
                    'entry_time': df.index[i],
                    'entry_price': current_price,
                    'sl_price': stop_loss,
                    'tp_price': take_profit,
                    'side': side,
                    'rsi_at_entry': rsi_value
                }

    metrics = calculate_trade_metrics(trades)
    csv_filename = export_trades_to_csv(trades, config.product_id)

    return BacktestResults(
        initial_balance=config.initial_balance,
        final_balance=balance,
        trades=trades,
        csv_filename=csv_filename,
        **metrics
    )

def calculate_trade_metrics(trades: List[Trade]) -> Dict:
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_profit': 0, 'profit_factor': 0
        }
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.profit > 0])
    losing_trades = len([t for t in trades if t.profit < 0])
    total_profit = sum(t.profit for t in trades)
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    gross_profit = sum(t.profit for t in trades if t.profit > 0)
    gross_loss = abs(sum(t.profit for t in trades if t.profit < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'profit_factor': profit_factor
    }

def export_trades_to_csv(trades: List[Trade], product_id: str) -> str:
    """Export trade history to CSV file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_filename = f"backtest_rsi_mean_reversion_{product_id}_{timestamp}.csv"
    
    try:
        trades_df = pd.DataFrame([vars(t) for t in trades])
        trades_df.to_csv(csv_filename, index=False)
        logger.info(f"\nTrade history exported to: {csv_filename}")
        return csv_filename
    except Exception as e:
        logger.error(f"Error exporting trades to CSV: {e}")
        return None

def print_results(results: BacktestResults):
    """Print backtest results in a formatted table."""
    summary_data = [
        ["Initial Balance", f"${results.initial_balance:,.2f}"],
        ["Final Balance", f"${results.final_balance:,.2f}"],
        ["Total Profit", f"${results.total_profit:,.2f}"],
        ["Total Trades", results.total_trades],
        ["Winning Trades", results.winning_trades],
        ["Losing Trades", results.losing_trades],
        ["Win Rate", f"{results.win_rate:.2f}%"],
        ["Profit Factor", f"{results.profit_factor:.2f}"]
    ]
    
    print("\n=== RSI MEAN REVERSION BACKTEST RESULTS ===")
    print(tabulate(summary_data, tablefmt="grid"))
    
    if results.csv_filename:
        print(f"\nCSV file saved as: {results.csv_filename}")

def main():
    """Main entry point for the backtest script."""
    config = parse_args()
    
    try:
        cb_service = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        logger.info("Fetching historical data...")
        df = fetch_candles(cb_service, config.product_id, config.start_date, config.end_date)
        
        if df.empty:
            logger.error("No data fetched. Exiting.")
            return
            
        logger.info(f"Data fetched: {len(df)} candles from {df.index.min()} to {df.index.max()}")
        logger.info("Starting backtest...")
        
        results = backtest(df, config)
        print_results(results)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
