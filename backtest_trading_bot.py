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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tabulate import tabulate
import numpy as np

# Recalculate these values every 50 trades using plot_atr_histogram.py
mean_atr_percent = 0.284
std_atr_percent = 0.148


# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress logs from other modules
logging.getLogger('technicalanalysis').setLevel(logging.WARNING)
logging.getLogger('historicaldata').setLevel(logging.WARNING)
logging.getLogger('bitcoinpredictionmodel').setLevel(logging.WARNING)
logging.getLogger('ml_model').setLevel(logging.WARNING)

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
    rsi_at_entry: float  # New field for RSI value at entry
    relative_volume: float  # New field for relative volume
    trend_slope: float  # New field for trend slope
    market_regime: str  # New field for market regime
    mae: float = 0.0  # Maximum Adverse Excursion
    mfe: float = 0.0  # Maximum Favorable Excursion
    exit_reason: str = ""  # Detailed exit reason (TP WICK HIT, SL WICK HIT, etc.)

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
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_filename = f"backtest_trades_{product_id}_{timestamp}.csv"
    
    try:
        trades_df = pd.DataFrame([vars(t) for t in trades])
        
        # Add columns to match automated_trades.csv format
        trades_df['No.'] = range(1, len(trades_df) + 1)
        trades_df['SIDE'] = 'LONG'  # All trades are long in this backtest
        trades_df['ENTRY'] = trades_df['entry_price'].round(2)
        trades_df['Take Profit'] = trades_df.apply(
            lambda row: determine_tp_mode(row['entry_price'], row['atr'], None, None, row['trend_slope'])[1], 
            axis=1
        ).round(2)
        trades_df['Stop Loss'] = (trades_df['entry_price'] * (1 - SL_PERCENT)).round(2)
        trades_df['R/R Ratio'] = ((trades_df['Take Profit'] - trades_df['ENTRY']) / 
                                 (trades_df['ENTRY'] - trades_df['Stop Loss'])).round(2)

        trades_df['Volatility Level'] = trades_df['atr_percent'].apply(
            lambda atr_percent: "Very Strong" if atr_percent > mean_atr_percent + std_atr_percent else 
                                "Strong" if atr_percent > mean_atr_percent else 
                                "Moderate" if atr_percent > mean_atr_percent - std_atr_percent else 
                                "Weak"
        )
        trades_df['Outcome'] = trades_df['type'].apply(lambda x: 'SUCCESS' if x == 'TP' else 'STOP LOSS')
        trades_df['Outcome %'] = trades_df['profit'].apply(lambda x: 7.5 if x > 0 else -3.5)
        trades_df['Leverage'] = '5x'
        trades_df['Margin'] = 50.0
        trades_df['Session'] = trades_df['entry_time'].apply(lambda x: 
            'Asia' if 0 <= x.hour < 9 else 
            'EU' if 9 <= x.hour < 17 else 'US')
        trades_df['TP Mode'] = trades_df['tp_mode']
        trades_df['ATR %'] = trades_df['atr_percent'].round(2)
        trades_df['Setup Type'] = 'RSI Dip'
        trades_df['MAE'] = trades_df['mae'].round(2)
        trades_df['MFE'] = trades_df['mfe'].round(2)
        trades_df['Exit Trade'] = trades_df['exit_price'].round(2)
        trades_df['Trend Regime'] = trades_df['market_regime']
        trades_df['RSI at Entry'] = trades_df['rsi_at_entry'].round(2)
        trades_df['Relative Volume'] = trades_df['relative_volume'].round(2)
        trades_df['Trend Slope'] = trades_df['trend_slope'].round(4)
        # Use the new exit_reason field if available, otherwise fall back to type
        trades_df['Exit Reason'] = trades_df.apply(
            lambda row: row['exit_reason'] if row['exit_reason'] else 
                       ('TP HIT' if row['type'] == 'TP' else 'SL HIT'), 
            axis=1
        )
        trades_df['Duration'] = ((trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 3600).round(2)
        
        # Reorder columns to match automated_trades.csv
        columns = [
            'No.', 'Timestamp', 'SIDE', 'ENTRY', 'Take Profit', 'Stop Loss', 
            'R/R Ratio', 'Volatility Level', 'Outcome', 'Outcome %', 
            'Leverage', 'Margin', 'Session', 'TP Mode', 'ATR %', 
            'Setup Type', 'MAE', 'MFE', 'Exit Trade', 'Trend Regime',
            'RSI at Entry', 'Relative Volume', 'Trend Slope', 'Exit Reason', 'Duration'
        ]
        
        # Rename entry_time to Timestamp
        trades_df['Timestamp'] = trades_df['entry_time']
        
        # Select and reorder columns
        trades_df = trades_df[columns]
        
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
        signal, entry, rsi_value, relative_volume, trend_slope = analyze(historical_df, ta, config.product_id)
        
        # Handle open position
        if current_trade:
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            atr = ta.compute_atr(historical_df.to_dict('records'))
            tp_mode, tp_price, market_regime = determine_tp_mode(current_trade['entry_price'], atr, None, historical_df)
            
            # Calculate MAE and MFE for the current trade
            # Check high for maximum favorable excursion
            high_change_percent = (current_high - current_trade['entry_price']) / current_trade['entry_price'] * 100
            # Check low for maximum adverse excursion
            low_change_percent = (current_low - current_trade['entry_price']) / current_trade['entry_price'] * 100
            
            # Update MFE (maximum favorable excursion using high price)
            current_trade['mfe'] = max(current_trade['mfe'], high_change_percent)
            
            # Update MAE (maximum adverse excursion using low price)
            if low_change_percent < 0:
                current_trade['mae'] = min(current_trade['mae'], low_change_percent)
            
            # Check if high price reached or exceeded take profit
            if current_high >= tp_price:
                profit = (tp_price - current_trade['entry_price']) * current_trade['size'] * config.leverage
                balance += profit
                # Determine if TP was hit by wick or close
                exit_reason = "TP WICK HIT" if current_price < tp_price else "TP HIT"
                trades.append(Trade(
                    entry_time=current_trade['entry_time'],
                    exit_time=df.index[i],
                    entry_price=current_trade['entry_price'],
                    exit_price=tp_price,
                    profit=profit,
                    type='TP',
                    atr=atr,
                    atr_percent=(atr / current_trade['entry_price']) * 100,
                    tp_mode=tp_mode,
                    rsi_at_entry=current_trade['rsi_at_entry'],
                    relative_volume=current_trade['relative_volume'],
                    trend_slope=current_trade['trend_slope'],
                    market_regime=current_trade['market_regime'],
                    mae=current_trade['mae'],
                    mfe=current_trade['mfe'],
                    exit_reason=exit_reason
                ))

                current_trade = None
                position = 0
                
            # Check if low price reached or exceeded stop loss
            elif current_low <= current_trade['sl_price']:
                loss = (current_trade['sl_price'] - current_trade['entry_price']) * current_trade['size'] * config.leverage
                balance += loss
                # Determine if SL was hit by wick or close
                exit_reason = "SL WICK HIT" if current_price > current_trade['sl_price'] else "SL HIT"
                trades.append(Trade(
                    entry_time=current_trade['entry_time'],
                    exit_time=df.index[i],
                    entry_price=current_trade['entry_price'],
                    exit_price=current_trade['sl_price'],
                    profit=loss,
                    type='SL',
                    atr=atr,
                    atr_percent=(atr / current_trade['entry_price']) * 100,
                    tp_mode=tp_mode,
                    rsi_at_entry=current_trade['rsi_at_entry'],
                    relative_volume=current_trade['relative_volume'],
                    trend_slope=current_trade['trend_slope'],
                    market_regime=current_trade['market_regime'],
                    mae=current_trade['mae'],
                    mfe=current_trade['mfe'],
                    exit_reason=exit_reason
                ))

                current_trade = None
                position = 0
        
        # Enter new trade if signal and no position
        elif signal and not current_trade:
            current_price = df.iloc[i]['close']
            atr = ta.compute_atr(historical_df.to_dict('records'))
            
            # Get dynamic TP based on market conditions
            tp_mode, tp_price, market_regime = determine_tp_mode(
                current_price, 
                atr, 
                get_price_precision(get_perp_product(config.product_id)),
                historical_df,
                trend_slope
            )
            
            # Keep SL fixed at 0.7% for now
            sl_price = round(current_price * (1 - SL_PERCENT), get_price_precision(get_perp_product(config.product_id)))
            
            position_size = balance * 0.1  # Use 10% of balance per trade
            size = position_size / current_price
            
            current_trade = {
                'entry_time': df.index[i],
                'entry_price': current_price,
                'size': size,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'tp_mode': tp_mode,
                'atr': atr,
                'atr_percent': (atr / current_price) * 100,
                'rsi_at_entry': rsi_value,
                'relative_volume': relative_volume,
                'trend_slope': trend_slope,
                'market_regime': market_regime,
                'mae': 0.0,
                'mfe': 0.0
            }
            position = size
        
        # Update balance history and calculate drawdown
        balance_history.append(balance)
        if balance > peak_balance:
            peak_balance = balance
            if current_drawdown_start is not None:
                drawdown_duration = (df.index[i] - current_drawdown_start).total_seconds() / 3600
                max_drawdown_duration = max(max_drawdown_duration, drawdown_duration)
                current_drawdown_start = None
        else:
            current_drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, current_drawdown)
            if current_drawdown_start is None:
                current_drawdown_start = df.index[i]
    
    # Calculate metrics and export results
    metrics = calculate_trade_metrics(trades)
    csv_filename = export_trades_to_csv(trades, config.product_id)
    
    # Add drawdown metrics
    metrics['max_drawdown'] = max_drawdown * 100  # Convert to percentage
    metrics['max_drawdown_duration'] = max_drawdown_duration
    
    return BacktestResults(
        initial_balance=config.initial_balance,
        final_balance=balance,
        trades=trades,
        csv_filename=csv_filename,
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
        ["Profit Factor", f"{results.profit_factor:.2f}"],
        ["Max Drawdown", f"{results.max_drawdown:.2f}%"],
        ["Max Drawdown Duration", f"{results.max_drawdown_duration:.1f} hours"]
    ]
    
    tp_mode_data = [
        ["Fixed TP Trades", results.fixed_tp_trades],
        ["Fixed TP Win Rate", f"{results.fixed_tp_win_rate:.2f}%"],
        ["Adaptive TP Trades", results.adaptive_tp_trades],
        ["Adaptive TP Win Rate", f"{results.adaptive_tp_win_rate:.2f}%"]
    ]
    
    # Calculate market regime statistics
    trending_trades = [t for t in results.trades if t.market_regime == "TRENDING"]
    chop_trades = [t for t in results.trades if t.market_regime == "CHOP"]
    uncertain_trades = [t for t in results.trades if t.market_regime == "UNCERTAIN"]
    
    trending_wins = sum(1 for t in trending_trades if t.profit > 0)
    chop_wins = sum(1 for t in chop_trades if t.profit > 0)
    uncertain_wins = sum(1 for t in uncertain_trades if t.profit > 0)
    
    trending_win_rate = (trending_wins / len(trending_trades) * 100) if trending_trades else 0
    chop_win_rate = (chop_wins / len(chop_trades) * 100) if chop_trades else 0
    uncertain_win_rate = (uncertain_wins / len(uncertain_trades) * 100) if uncertain_trades else 0
    
    regime_data = [
        ["Trending Trades", len(trending_trades)],
        ["Trending Win Rate", f"{trending_win_rate:.2f}%"],
        ["Chop Trades", len(chop_trades)],
        ["Chop Win Rate", f"{chop_win_rate:.2f}%"],
        ["Uncertain Trades", len(uncertain_trades)],
        ["Uncertain Win Rate", f"{uncertain_win_rate:.2f}%"]
    ]
    
    # Print tables
    print("\n=== BACKTEST RESULTS ===")
    print(tabulate(summary_data, tablefmt="grid"))
    
    print("\n=== TP MODE STATISTICS ===")
    print(tabulate(tp_mode_data, tablefmt="grid"))
    
    print("\n=== MARKET REGIME STATISTICS ===")
    print(tabulate(regime_data, tablefmt="grid"))
    
    print(f"\nCSV file saved as: {results.csv_filename}")

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