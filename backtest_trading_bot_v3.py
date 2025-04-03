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
    market_regime: str  # Added market regime field
    entry_mode: str  # Added entry mode field

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
    # Added regime-specific metrics
    weak_sideways_trades: int
    weak_sideways_win_rate: float
    moderate_sideways_trades: int
    moderate_sideways_win_rate: float
    strong_trending_trades: int
    strong_trending_win_rate: float
    # Added entry mode metrics
    mean_reversion_trades: int
    mean_reversion_win_rate: float
    momentum_trades: int
    momentum_win_rate: float

def parse_args() -> BacktestConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Backtest the trading bot')
    parser.add_argument('--product', type=str, default='BTC-USDC',
                      help='Product ID to backtest (e.g., BTC-USDC)')
    parser.add_argument('--balance', type=float, default=10000,
                      help='Initial balance for backtesting')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage')
    parser.add_argument('--start-date', '--start_date', type=str,
                      help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', '--end_date', type=str,
                      help='End date for backtesting (YYYY-MM-DD)')
    args = parser.parse_args()
    
    return BacktestConfig(
        product_id=args.product,
        initial_balance=args.balance,
        leverage=args.leverage,
        start_date=args.start_date,
        end_date=args.end_date
    )

def fetch_candles(cb: CoinbaseService, product_id: str, 
                 start_date: Optional[str] = None, 
                 end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical candles for backtesting."""
    # Default to last 30 days if no dates provided
    end = datetime.now(UTC)
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=UTC)
    
    start = end - timedelta(days=30)
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=UTC)
    
    # Fetch data
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, GRANULARITY)
    df = pd.DataFrame(raw_data)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp - convert Unix timestamp to datetime
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s', utc=True)        
        df.set_index('start', inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    
    return df

def calculate_trade_metrics(trades: List[Trade]) -> Dict:
    """Calculate metrics from trades."""
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'profit_factor': 0,
            'avg_winning_atr': 0,
            'avg_losing_atr': 0,
            'fixed_tp_trades': 0,
            'adaptive_tp_trades': 0,
            'fixed_tp_win_rate': 0,
            'adaptive_tp_win_rate': 0,
            # Added regime-specific metrics
            'weak_sideways_trades': 0,
            'weak_sideways_win_rate': 0,
            'moderate_sideways_trades': 0,
            'moderate_sideways_win_rate': 0,
            'strong_trending_trades': 0,
            'strong_trending_win_rate': 0,
            # Added entry mode metrics
            'mean_reversion_trades': 0,
            'mean_reversion_win_rate': 0,
            'momentum_trades': 0,
            'momentum_win_rate': 0
        }
    
    winning_trades = [t for t in trades if t.profit > 0]
    losing_trades = [t for t in trades if t.profit <= 0]
    
    fixed_tp_trades = [t for t in trades if t.tp_mode == 'FIXED']
    adaptive_tp_trades = [t for t in trades if t.tp_mode == 'ADAPTIVE']
    
    winning_fixed_tp = [t for t in fixed_tp_trades if t.profit > 0]
    winning_adaptive_tp = [t for t in adaptive_tp_trades if t.profit > 0]
    
    # Calculate regime-specific metrics
    weak_sideways_trades = [t for t in trades if t.market_regime == 'Weak/Sideways']
    moderate_sideways_trades = [t for t in trades if t.market_regime == 'Moderate/Sideways']
    strong_trending_trades = [t for t in trades if t.market_regime == 'Strong/Trending']
    
    winning_weak_sideways = [t for t in weak_sideways_trades if t.profit > 0]
    winning_moderate_sideways = [t for t in moderate_sideways_trades if t.profit > 0]
    winning_strong_trending = [t for t in strong_trending_trades if t.profit > 0]
    
    # Calculate entry mode metrics
    mean_reversion_trades = [t for t in trades if t.entry_mode == 'mean_reversion']
    momentum_trades = [t for t in trades if t.entry_mode == 'momentum']
    
    winning_mean_reversion = [t for t in mean_reversion_trades if t.profit > 0]
    winning_momentum = [t for t in momentum_trades if t.profit > 0]
    
    total_profit = sum(t.profit for t in trades)
    winning_profit = sum(t.profit for t in winning_trades)
    losing_profit = abs(sum(t.profit for t in losing_trades))
    
    return {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'total_profit': total_profit,
        'profit_factor': winning_profit / losing_profit if losing_profit > 0 else float('inf'),
        'avg_winning_atr': sum(t.atr for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        'avg_losing_atr': sum(t.atr for t in losing_trades) / len(losing_trades) if losing_trades else 0,
        'fixed_tp_trades': len(fixed_tp_trades),
        'adaptive_tp_trades': len(adaptive_tp_trades),
        'fixed_tp_win_rate': len(winning_fixed_tp) / len(fixed_tp_trades) if fixed_tp_trades else 0,
        'adaptive_tp_win_rate': len(winning_adaptive_tp) / len(adaptive_tp_trades) if adaptive_tp_trades else 0,
        # Added regime-specific metrics
        'weak_sideways_trades': len(weak_sideways_trades),
        'weak_sideways_win_rate': len(winning_weak_sideways) / len(weak_sideways_trades) if weak_sideways_trades else 0,
        'moderate_sideways_trades': len(moderate_sideways_trades),
        'moderate_sideways_win_rate': len(winning_moderate_sideways) / len(moderate_sideways_trades) if moderate_sideways_trades else 0,
        'strong_trending_trades': len(strong_trending_trades),
        'strong_trending_win_rate': len(winning_strong_trending) / len(strong_trending_trades) if strong_trending_trades else 0,
        # Added entry mode metrics
        'mean_reversion_trades': len(mean_reversion_trades),
        'mean_reversion_win_rate': len(winning_mean_reversion) / len(mean_reversion_trades) if mean_reversion_trades else 0,
        'momentum_trades': len(momentum_trades),
        'momentum_win_rate': len(winning_momentum) / len(momentum_trades) if momentum_trades else 0
    }

def export_trades_to_csv(trades: List[Trade], product_id: str) -> str:
    """Export trades to CSV file."""
    if not trades:
        return None
    
    # Create DataFrame from trades
    trades_data = []
    for trade in trades:
        trades_data.append({
            'Entry Time': trade.entry_time,
            'Exit Time': trade.exit_time,
            'Entry Price': trade.entry_price,
            'Exit Price': trade.exit_price,
            'Profit': trade.profit,
            'Type': trade.type,
            'ATR': trade.atr,
            'ATR %': trade.atr_percent,
            'TP Mode': trade.tp_mode,
            'Market Regime': trade.market_regime,  # Added market regime
            'Entry Mode': trade.entry_mode  # Added entry mode
        })
    
    trades_df = pd.DataFrame(trades_data)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'backtest_trades_{product_id}_{timestamp}.csv'
    
    # Save to CSV
    trades_df.to_csv(filename, index=False)
    logger.info(f"Exported {len(trades)} trades to {filename}")
    
    return filename

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
        signal, entry, market_regime, entry_mode = analyze(historical_df, ta, config.product_id)  # Updated to handle four return values
        
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
                    tp_mode=tp_mode,
                    market_regime=current_trade['market_regime'],  # Added market regime
                    entry_mode=current_trade['entry_mode']  # Added entry mode
                ))
                if len(trades) >= 200:
                    break
                current_trade = None
                position = 0
                
            elif current_price <= current_trade['sl_price']:
                loss = (current_trade['sl_price'] - current_trade['entry_price']) * current_trade['size'] * config.leverage
                balance += loss
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
                    market_regime=current_trade['market_regime'],  # Added market regime
                    entry_mode=current_trade['entry_mode']  # Added entry mode
                ))
                if len(trades) >= 200:
                    break
                current_trade = None
                position = 0
        
        # Enter new trade if signal and no position
        elif signal and not current_trade:
            current_price = df.iloc[i]['close']
            atr = ta.compute_atr(historical_df.to_dict('records'))
            tp_mode, tp_price = determine_tp_mode(current_price, atr)
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
                'market_regime': market_regime,  # Added market regime
                'entry_mode': entry_mode  # Added entry mode
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
    
    # Calculate metrics
    metrics = calculate_trade_metrics(trades)
    
    # Export trades to CSV
    csv_filename = export_trades_to_csv(trades, config.product_id)
    
    return BacktestResults(
        initial_balance=config.initial_balance,
        final_balance=balance,
        total_trades=metrics['total_trades'],
        winning_trades=metrics['winning_trades'],
        losing_trades=metrics['losing_trades'],
        win_rate=metrics['win_rate'],
        total_profit=metrics['total_profit'],
        profit_factor=metrics['profit_factor'],
        avg_winning_atr=metrics['avg_winning_atr'],
        avg_losing_atr=metrics['avg_losing_atr'],
        fixed_tp_trades=metrics['fixed_tp_trades'],
        adaptive_tp_trades=metrics['adaptive_tp_trades'],
        fixed_tp_win_rate=metrics['fixed_tp_win_rate'],
        adaptive_tp_win_rate=metrics['adaptive_tp_win_rate'],
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        trades=trades,
        csv_filename=csv_filename,
        # Added regime-specific metrics
        weak_sideways_trades=metrics['weak_sideways_trades'],
        weak_sideways_win_rate=metrics['weak_sideways_win_rate'],
        moderate_sideways_trades=metrics['moderate_sideways_trades'],
        moderate_sideways_win_rate=metrics['moderate_sideways_win_rate'],
        strong_trending_trades=metrics['strong_trending_trades'],
        strong_trending_win_rate=metrics['strong_trending_win_rate'],
        # Added entry mode metrics
        mean_reversion_trades=metrics['mean_reversion_trades'],
        mean_reversion_win_rate=metrics['mean_reversion_win_rate'],
        momentum_trades=metrics['momentum_trades'],
        momentum_win_rate=metrics['momentum_win_rate']
    )

def print_results(results: BacktestResults):
    """Print backtest results in a formatted table."""
    # Basic metrics
    basic_metrics = [
        ['Initial Balance', f"${results.initial_balance:.2f}"],
        ['Final Balance', f"${results.final_balance:.2f}"],
        ['Total Profit', f"${results.total_profit:.2f}"],
        ['Profit Factor', f"{results.profit_factor:.2f}"],
        ['Total Trades', results.total_trades],
        ['Winning Trades', results.winning_trades],
        ['Losing Trades', results.losing_trades],
        ['Win Rate', f"{results.win_rate:.2%}"],
        ['Max Drawdown', f"{results.max_drawdown:.2%}"],
        ['Max Drawdown Duration', f"{results.max_drawdown_duration:.1f} hours"]
    ]
    
    # TP mode metrics
    tp_metrics = [
        ['Fixed TP Trades', results.fixed_tp_trades],
        ['Adaptive TP Trades', results.adaptive_tp_trades],
        ['Fixed TP Win Rate', f"{results.fixed_tp_win_rate:.2%}"],
        ['Adaptive TP Win Rate', f"{results.adaptive_tp_win_rate:.2%}"]
    ]
    
    # Regime-specific metrics
    regime_metrics = [
        ['Weak/Sideways Trades', results.weak_sideways_trades],
        ['Weak/Sideways Win Rate', f"{results.weak_sideways_win_rate:.2%}"],
        ['Moderate/Sideways Trades', results.moderate_sideways_trades],
        ['Moderate/Sideways Win Rate', f"{results.moderate_sideways_win_rate:.2%}"],
        ['Strong/Trending Trades', results.strong_trending_trades],
        ['Strong/Trending Win Rate', f"{results.strong_trending_win_rate:.2%}"]
    ]
    
    # Entry mode metrics
    entry_mode_metrics = [
        ['Mean Reversion Trades', results.mean_reversion_trades],
        ['Mean Reversion Win Rate', f"{results.mean_reversion_win_rate:.2%}"],
        ['Momentum Trades', results.momentum_trades],
        ['Momentum Win Rate', f"{results.momentum_win_rate:.2%}"]
    ]
    
    # Print tables
    print("\n=== BASIC METRICS ===")
    print(tabulate(basic_metrics, tablefmt='grid'))
    
    print("\n=== TP MODE METRICS ===")
    print(tabulate(tp_metrics, tablefmt='grid'))
    
    print("\n=== REGIME-SPECIFIC METRICS ===")
    print(tabulate(regime_metrics, tablefmt='grid'))
    
    print("\n=== ENTRY MODE METRICS ===")
    print(tabulate(entry_mode_metrics, tablefmt='grid'))
    
    if results.csv_filename:
        print(f"\nTrades exported to {results.csv_filename}")

def main():
    """Main function to run the backtest."""
    args = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data
        df = fetch_candles(cb, args.product_id, args.start_date, args.end_date)
        
        # Run backtest
        results = backtest(df, ta, args)
        
        # Print results
        print_results(results)
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 