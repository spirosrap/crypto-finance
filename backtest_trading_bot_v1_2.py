from simplified_trading_bot_v1_2_regime_filters_only import (
    CoinbaseService, TechnicalAnalysis, GRANULARITY, RSI_THRESHOLD,
    VOLUME_LOOKBACK, TP_PERCENT, SL_PERCENT, get_perp_product,
    get_price_precision, analyze, determine_tp_mode, classify_market_regime
)
from datetime import datetime, timedelta, UTC
import pandas as pd
import logging
import argparse
from config import API_KEY_PERPS, API_SECRET_PERPS
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tabulate import tabulate
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress logs from other modules
logging.getLogger('technicalanalysis').setLevel(logging.WARNING)
logging.getLogger('historicaldata').setLevel(logging.WARNING)
logging.getLogger('bitcoinpredictionmodel').setLevel(logging.WARNING)
logging.getLogger('ml_model').setLevel(logging.WARNING)

# Global counter for regime distribution
regime_counter = Counter()

@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting."""
    product_id: str = 'BTC-USDC'
    initial_balance: float = 10000
    leverage: int = 5
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
    type: str
    atr: float
    atr_percent: float
    tp_mode: str

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
    avg_winning_atr: float
    avg_losing_atr: float
    fixed_tp_trades: int
    adaptive_tp_trades: int
    fixed_tp_win_rate: float
    adaptive_tp_win_rate: float
    max_drawdown: float
    max_drawdown_duration: float
    trades: List[Trade]
    csv_filename: Optional[str]

def parse_args() -> BacktestConfig:
    """Parse command line arguments and return a BacktestConfig object."""
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
        now = datetime.now(UTC)
        start = now - timedelta(minutes=5 * 8000)
        end = now
    
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, GRANULARITY)
    df = pd.DataFrame(raw_data)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp conversion
    timestamp_columns = ['start', 'timestamp', 'time']
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit='s', utc=True)
            df.set_index(col, inplace=True)
            break
    
    return df

def calculate_trade_metrics(trades: List[Trade]) -> Dict:
    """Calculate performance metrics from trade history."""
    if not trades:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_profit': 0, 'profit_factor': 0,
            'avg_winning_atr': 0, 'avg_losing_atr': 0,
            'fixed_tp_trades': 0, 'adaptive_tp_trades': 0,
            'fixed_tp_win_rate': 0, 'adaptive_tp_win_rate': 0
        }
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.profit > 0])
    losing_trades = len([t for t in trades if t.profit < 0])
    total_profit = sum(t.profit for t in trades)
    win_rate = (winning_trades / total_trades) * 100
    profit_factor = abs(sum(t.profit for t in trades if t.profit > 0) / 
                       sum(t.profit for t in trades if t.profit < 0)) if losing_trades > 0 else float('inf')
    
    # Calculate metrics by TP mode
    fixed_tp_trades = [t for t in trades if t.tp_mode == "FIXED"]
    adaptive_tp_trades = [t for t in trades if t.tp_mode == "ADAPTIVE"]
    
    fixed_tp_win_rate = (len([t for t in fixed_tp_trades if t.profit > 0]) / len(fixed_tp_trades) * 100) if fixed_tp_trades else 0
    adaptive_tp_win_rate = (len([t for t in adaptive_tp_trades if t.profit > 0]) / len(adaptive_tp_trades) * 100) if adaptive_tp_trades else 0
    
    # Calculate average ATR for winning vs losing trades
    winning_trades_atr = [t.atr for t in trades if t.profit > 0]
    losing_trades_atr = [t.atr for t in trades if t.profit < 0]
    avg_winning_atr = sum(winning_trades_atr) / len(winning_trades_atr) if winning_trades_atr else 0
    avg_losing_atr = sum(losing_trades_atr) / len(losing_trades_atr) if losing_trades_atr else 0
    
    return {
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
        'adaptive_tp_win_rate': adaptive_tp_win_rate
    }

def export_trades_to_csv(trades: List[Trade], product_id: str) -> str:
    """Export trade history to CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"backtest_trades_{product_id}_{timestamp}.csv"
    
    try:
        trades_df = pd.DataFrame([vars(t) for t in trades])
        trades_df['Session'] = trades_df['entry_time'].apply(lambda x: 
            'Asia' if 0 <= x.hour < 9 else 
            'EU' if 9 <= x.hour < 17 else 'US')
        trades_df['Setup Type'] = 'RSI Dip'
        trades_df['Outcome'] = trades_df['type'].apply(lambda x: 'SUCCESS' if x == 'TP' else 'FAILURE')
        
        # Format numeric columns
        numeric_columns = ['entry_price', 'exit_price', 'profit', 'atr', 'atr_percent']
        for col in numeric_columns:
            trades_df[col] = trades_df[col].round(2)
        
        # Reorder and rename columns
        columns = [
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'profit', 'type', 'Session', 'tp_mode', 'atr_percent',
            'Setup Type', 'Outcome'
        ]
        trades_df = trades_df[columns]
        trades_df.columns = [
            'Entry Time', 'Exit Time', 'Entry Price', 'Exit Price',
            'Profit', 'Exit Type', 'Session', 'TP Mode', 'ATR %',
            'Setup Type', 'Outcome'
        ]
        
        trades_df.to_csv(csv_filename, index=False)
        logger.info(f"\nTrade history exported to: {csv_filename}")
        return csv_filename
        
    except Exception as e:
        logger.error(f"Error exporting trades to CSV: {e}")
        return None

def backtest(df: pd.DataFrame, ta: TechnicalAnalysis, config: BacktestConfig) -> BacktestResults:
    """Run backtest on historical data and return results."""
    balance = config.initial_balance
    position = 0
    trades = []
    current_trade = None
    
    # Track balance history for drawdown calculation
    balance_history = [config.initial_balance]
    peak_balance = config.initial_balance
    max_drawdown = 0
    max_drawdown_duration = 0
    current_drawdown_start = None
    
    # Ensure df index is datetime and in UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    for i in range(50, len(df)):  # Start after EMA period
        # Get historical data up to current point
        historical_df = df.iloc[:i+1]
        
        # Calculate ATR for regime classification
        atr = ta.compute_atr(historical_df.to_dict('records'))
        current_price = df.iloc[i]['close']
        regime = classify_market_regime(atr, current_price)
        
        # Track regime distribution
        regime_counter[regime] += 1
        
        signal, entry = analyze(historical_df, ta, config.product_id)
        
        # Handle open position
        if current_trade:
            current_price = df.iloc[i]['close']
            atr = ta.compute_atr(historical_df.to_dict('records'))
            tp_mode, tp_price = determine_tp_mode(current_trade['entry_price'], atr)
            
            # Check for TP or SL
            if current_price >= tp_price:
                profit = (tp_price - current_trade['entry_price']) * current_trade['size'] * config.leverage
                balance += profit
                trades.append(Trade(
                    entry_time=current_trade['entry_time'],
                    exit_time=df.index[i],
                    entry_price=current_trade['entry_price'],
                    exit_price=tp_price,
                    profit=profit,
                    type='TP',
                    atr=atr,
                    atr_percent=(atr / current_trade['entry_price']) * 100,
                    tp_mode=tp_mode
                ))
                if len(trades) >= 200:
                    break
                
                # Update drawdown metrics
                balance_history.append(balance)
                if balance > peak_balance:
                    peak_balance = balance
                    current_drawdown_start = None
                else:
                    if current_drawdown_start is None:
                        current_drawdown_start = df.index[i]
                    current_drawdown = (peak_balance - balance) / peak_balance * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                        max_drawdown_duration = (df.index[i] - current_drawdown_start).total_seconds() / 3600
                
                current_trade = None
                position = 0
                
            elif current_price <= current_trade['sl_price']:
                profit = (current_trade['sl_price'] - current_trade['entry_price']) * current_trade['size'] * config.leverage
                balance += profit
                trades.append(Trade(
                    entry_time=current_trade['entry_time'],
                    exit_time=df.index[i],
                    entry_price=current_trade['entry_price'],
                    exit_price=current_trade['sl_price'],
                    profit=profit,
                    type='SL',
                    atr=atr,
                    atr_percent=(atr / current_trade['entry_price']) * 100,
                    tp_mode=tp_mode
                ))
                if len(trades) >= 200:
                    break
                
                # Update drawdown metrics
                balance_history.append(balance)
                if balance > peak_balance:
                    peak_balance = balance
                    current_drawdown_start = None
                else:
                    if current_drawdown_start is None:
                        current_drawdown_start = df.index[i]
                    current_drawdown = (peak_balance - balance) / peak_balance * 100
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                        max_drawdown_duration = (df.index[i] - current_drawdown_start).total_seconds() / 3600
                
                current_trade = None
                position = 0
        
        # Handle new trade entry
        elif signal and entry:
            # Calculate position size
            position_size = balance * 0.1  # Use 10% of balance per trade
            
            # Calculate ATR for TP/SL
            atr = ta.compute_atr(historical_df.to_dict('records'))
            tp_mode, tp_price = determine_tp_mode(entry, atr)
            
            # Set stop loss
            sl_price = entry * (1 - SL_PERCENT)
            
            # Record trade
            current_trade = {
                'entry_time': df.index[i],
                'entry_price': entry,
                'size': position_size / entry,
                'tp_price': tp_price,
                'sl_price': sl_price
            }
            position = 1
    
    # Calculate final metrics
    metrics = calculate_trade_metrics(trades)
    
    # Export trades to CSV
    csv_filename = export_trades_to_csv(trades, config.product_id)
    
    # Print regime distribution
    print("\n=== ATR Regime Distribution ===")
    total = sum(regime_counter.values())
    for regime, count in regime_counter.items():
        pct = (count / total) * 100
        print(f"{regime.capitalize():<10} : {count:>6} bars ({pct:.2f}%)")
    
    return BacktestResults(
        initial_balance=config.initial_balance,
        final_balance=balance,
        trades=trades,
        csv_filename=csv_filename,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        **metrics
    )

def print_results(results: BacktestResults):
    """Print backtest results in a formatted table."""
    # Prepare data for tabulate
    summary_data = [
        ["Initial Balance", f"${results.initial_balance:.2f}"],
        ["Final Balance", f"${results.final_balance:.2f}"],
        ["Total Profit", f"${results.total_profit:.2f}"],
        ["Total Trades", results.total_trades],
        ["Winning Trades", results.winning_trades],
        ["Losing Trades", results.losing_trades],
        ["Win Rate", f"{results.win_rate:.2f}%"],
        ["Profit Factor", f"{results.profit_factor:.2f}"]
    ]
    
    tp_mode_data = [
        ["Fixed TP Trades", results.fixed_tp_trades],
        ["Fixed TP Win Rate", f"{results.fixed_tp_win_rate:.2f}%"],
        ["Adaptive TP Trades", results.adaptive_tp_trades],
        ["Adaptive TP Win Rate", f"{results.adaptive_tp_win_rate:.2f}%"]
    ]
    
    risk_data = [
        ["Maximum Drawdown", f"{results.max_drawdown:.2f}%"],
        ["Maximum Drawdown Duration", f"{results.max_drawdown_duration:.2f} hours"]
    ]
    
    # Print results using tabulate with single logger calls
    logger.info("\nBacktest Results:\n" + tabulate(summary_data, tablefmt="grid"))
    logger.info("\nTP Mode Statistics:\n" + tabulate(tp_mode_data, tablefmt="grid"))
    logger.info("\nRisk Metrics:\n" + tabulate(risk_data, tablefmt="grid"))
    
    if results.csv_filename:
        logger.info(f"\nDetailed trade history saved to: {results.csv_filename}")
    
    # Print trade history
    logger.info("\nTrade History:")
    for trade in results.trades:
        logger.info(f"Entry: {trade.entry_time} @ ${trade.entry_price:.2f}")
        logger.info(f"Exit: {trade.exit_time} @ ${trade.exit_price:.2f}")
        logger.info(f"Profit: ${trade.profit:.2f} ({trade.type})")
        logger.info(f"TP Mode: {trade.tp_mode} (ATR: {trade.atr:.2f}, {trade.atr_percent:.2f}%)")
        logger.info("---")

def main():
    """Main entry point for the backtest script."""
    config = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data
        df = fetch_candles(cb, config.product_id, config.start_date, config.end_date)
        
        logger.info("Starting backtest...")
        if config.start_date and config.end_date:
            logger.info(f"Backtest period: {config.start_date} to {config.end_date}")
        else:
            logger.info("Using default period (last 8000 5-minute candles)")
        
        # Run backtest
        results = backtest(df, ta, config)
        
        # Print results
        print_results(results)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 