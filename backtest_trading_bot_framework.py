from trading_bot_framework import (
    CoinbaseService, TechnicalAnalysis, CONFIG, get_perp_product,
    get_price_precision, detect_regime, detect_rsi_dip, detect_breakout,
    filter_by_volume, filter_by_volatility, compute_tp_sl, risk_check,fetch_candles,calculate_trend_slope
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

# Constants
GRANULARITY = "FIVE_MINUTE"  # Default candle interval

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
    regime: str
    atr: float
    atr_percent: float
    tp_mode: str
    strategy: str
    relative_volume: float
    trend_slope: float
    rsi: float
    take_profit: float
    stop_loss: float
    rr_ratio: float
    volatility_level: str
    outcome: str
    outcome_percent: float
    leverage: str
    margin: float
    session: str
    mae: float
    mfe: float
    exit_reason: str
    duration: float

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
    parser = argparse.ArgumentParser(description='Backtest Trading Bot Framework')
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

def run_backtest(config: BacktestConfig) -> BacktestResults:
    """Run the backtest with the given configuration."""
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    # Validate date range
    if config.start_date and config.end_date:
        try:
            start = datetime.strptime(config.start_date, '%Y-%m-%d')
            end = datetime.strptime(config.end_date, '%Y-%m-%d')
            if end < start:
                logger.error("End date cannot be before start date")
                return None
            if end > datetime.now():
                logger.error("End date cannot be in the future")
                return None
        except ValueError as e:
            logger.error(f"Invalid date format: {str(e)}. Use YYYY-MM-DD format")
            return None
    
    # Fetch historical data
    raw_data = cb.historical_data.get_historical_data(config.product_id, start, end, GRANULARITY)
    logger.info(f"Raw data type: {type(raw_data)}")
    if raw_data:
        logger.info(f"First candle structure: {raw_data[0]}")
    df = pd.DataFrame(raw_data)
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert timestamp to datetime
    df['start'] = pd.to_datetime(df['start'], unit='s', utc=True)
    df.set_index('start', inplace=True)
    
    if df.empty:
        logger.error("No historical data available for backtesting")
        return None
    
    # Check minimum data requirements
    min_candles = max(
        CONFIG['ATR_PERIOD'],
        CONFIG['EMA_PERIOD'],
        CONFIG['TREND_SLOPE_PERIOD'],
        CONFIG['RSI_PERIOD'],
        CONFIG['VOLUME_LOOKBACK']
    )
    
    if len(df) < min_candles:
        logger.error(f"Not enough historical data. Need at least {min_candles} candles, got {len(df)}")
        return None
    
    # Initialize backtest variables
    balance = config.initial_balance
    position = 0
    trades: List[Trade] = []
    balance_history = []
    max_balance = balance
    max_drawdown = 0
    max_drawdown_duration = 0
    current_drawdown_duration = 0
    
    # Track trade statistics
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    fixed_tp_trades = 0
    adaptive_tp_trades = 0
    fixed_tp_wins = 0
    adaptive_tp_wins = 0
    
    # Main backtest loop
    for i in range(50, len(df)):  # Start after EMA period
        current_time = df.index[i]
        current_price = df['close'].iloc[i]
        
        # Update balance history
        balance_history.append(balance + (position * current_price))
        
        # Calculate drawdown
        current_balance = balance + (position * current_price)
        if current_balance > max_balance:
            max_balance = current_balance
            current_drawdown_duration = 0
        else:
            drawdown = (max_balance - current_balance) / max_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
            current_drawdown_duration += 1
            if current_drawdown_duration > max_drawdown_duration:
                max_drawdown_duration = current_drawdown_duration
        
        # If we have an open position, check for exit conditions
        if position > 0:
            if not trades:  # Add check for empty trades list
                logger.warning("Position is > 0 but trades list is empty")
                position = 0
                continue
            
            current_trade = trades[-1]
            
            # Update MAE and MFE
            current_mae = (current_price - current_trade.entry_price) / current_trade.entry_price
            current_mfe = (current_price - current_trade.entry_price) / current_trade.entry_price
            current_trade.mae = min(current_trade.mae, current_mae)
            current_trade.mfe = max(current_trade.mfe, current_mfe)
            
            # Check for take profit
            if current_price >= current_trade.take_profit:
                # Calculate profit with leverage using take profit price
                profit = (current_trade.take_profit - current_trade.entry_price) * position
                balance += profit
                current_trade.exit_time = current_time
                current_trade.exit_price = current_trade.take_profit
                current_trade.profit = profit
                current_trade.outcome = 'SUCCESS'
                current_trade.outcome_percent = 7.5  # Hardcoded for now
                current_trade.exit_reason = 'TP HIT'
                current_trade.duration = (current_trade.exit_time - current_trade.entry_time).total_seconds() / 3600
                position = 0
                total_trades += 1
                winning_trades += 1
                if current_trade.tp_mode == 'FIXED':
                    fixed_tp_trades += 1
                    fixed_tp_wins += 1
                else:
                    adaptive_tp_trades += 1
                    adaptive_tp_wins += 1
                continue
            
            # Check for stop loss
            if current_price <= current_trade.stop_loss:
                # Calculate loss with leverage
                loss = (current_trade.stop_loss - current_trade.entry_price) * position
                balance += loss
                current_trade.exit_time = current_time
                current_trade.exit_price = current_trade.stop_loss
                current_trade.profit = loss
                current_trade.outcome = 'STOP LOSS'
                current_trade.outcome_percent = -3.5  # Hardcoded for now
                current_trade.exit_reason = 'SL HIT'
                current_trade.duration = (current_trade.exit_time - current_trade.entry_time).total_seconds() / 3600
                position = 0
                total_trades += 1
                losing_trades += 1
                if current_trade.tp_mode == 'FIXED':
                    fixed_tp_trades += 1
                else:
                    adaptive_tp_trades += 1
                continue
        
        # If we don't have a position, check for entry signals
        if position == 0 and balance > 0:
            # Get current market data
            current_df = df.iloc[:i+1]
            
            # Ensure we have enough data for calculations
            if len(current_df) < min_candles:
                continue
            
            try:
                # Detect market regime
                regime = detect_regime(current_df, ta, current_df.to_dict('records'))
                
                # Calculate ATR and other indicators
                atr = ta.compute_atr(current_df.to_dict('records'))
                atr_percent = (atr / current_price) * 100
                
                # Calculate relative volume
                avg_volume = current_df['volume'].rolling(CONFIG['VOLUME_LOOKBACK']).mean()
                relative_volume = current_df['volume'].iloc[-1] / avg_volume.iloc[-1]
                
                # Calculate trend slope
                trend_slope = calculate_trend_slope(current_df)
                
                # Check RSI dip strategy
                rsi_signal, rsi_entry_price = detect_rsi_dip(current_df, ta, current_df.to_dict('records'), config.product_id)
                if rsi_signal:# and filter_by_volume(relative_volume) and filter_by_volatility(atr_percent):
                    # Calculate position size and risk
                    tp, sl = compute_tp_sl(rsi_entry_price, regime, atr)
                    position_size = min(
                        CONFIG['MAX_POSITION_SIZE'],
                        balance * CONFIG['MAX_RISK_PER_TRADE']
                    )
                    position_size = 200 # Hardcoded for testing
                    # Calculate position with leverage
                    leveraged_position = position_size * config.leverage
                    
                    if risk_check(rsi_entry_price, sl, leveraged_position):
                        # Calculate final position size in units of the asset
                        position = leveraged_position / rsi_entry_price
                        
                        # Create new trade
                        trade = Trade(
                            entry_time=current_time,
                            exit_time=None,
                            entry_price=rsi_entry_price,
                            exit_price=None,
                            profit=None,
                            type='LONG',
                            regime=regime,
                            atr=atr,
                            atr_percent=atr_percent,
                            tp_mode='ADAPTIVE' if atr_percent > (CONFIG['MEAN_ATR_PERCENT'] + CONFIG['STD_ATR_PERCENT']) and regime == 'TRENDING' else 'FIXED',
                            strategy='RSI Dip',
                            relative_volume=relative_volume,
                            trend_slope=trend_slope,
                            rsi=ta.compute_rsi(config.product_id, current_df.to_dict('records'), CONFIG['RSI_PERIOD']),
                            take_profit=tp,
                            stop_loss=sl,
                            rr_ratio=round((tp - rsi_entry_price) / (rsi_entry_price - sl), 2),
                            volatility_level='Moderate' if atr_percent > CONFIG['MEAN_ATR_PERCENT'] else 'Weak',
                            outcome=None,
                            outcome_percent=None,
                            leverage=f"{config.leverage}x",
                            margin=50.0,  # Hardcoded for now
                            session='Asia' if 0 <= current_time.hour < 8 else 'EU' if 8 <= current_time.hour < 16 else 'US',
                            mae=0.0,
                            mfe=0.0,
                            exit_reason=None,
                            duration=0.0
                        )
                        trades.append(trade)
                        continue
                
                # Check breakout strategy
                breakout_signal, breakout_entry_price = detect_breakout(current_df, ta, current_df.to_dict('records'), config.product_id)
                if False and breakout_signal and filter_by_volume(relative_volume) and filter_by_volatility(atr_percent):
                    # Calculate position size and risk
                    tp, sl = compute_tp_sl(breakout_entry_price, regime, atr)
                    position_size = min(
                        CONFIG['MAX_POSITION_SIZE'],
                        balance * CONFIG['MAX_RISK_PER_TRADE']
                    )
                    
                    if risk_check(breakout_entry_price, sl, position_size):
                        # Calculate position with leverage
                        position = (position_size * config.leverage) / breakout_entry_price
                        
                        # Create new trade
                        trade = Trade(
                            entry_time=current_time,
                            exit_time=None,
                            entry_price=breakout_entry_price,
                            exit_price=None,
                            profit=None,
                            type='LONG',
                            regime=regime,
                            atr=atr,
                            atr_percent=atr_percent,
                            tp_mode='ADAPTIVE' if atr_percent > (CONFIG['MEAN_ATR_PERCENT'] + CONFIG['STD_ATR_PERCENT']) and regime == 'TRENDING' else 'FIXED',
                            strategy='Breakout',
                            relative_volume=relative_volume,
                            trend_slope=trend_slope,
                            rsi=ta.compute_rsi(config.product_id, current_df.to_dict('records'), CONFIG['RSI_PERIOD'])
                        )
                        trade.take_profit = tp
                        trade.stop_loss = sl
                        trades.append(trade)
            except Exception as e:
                logger.warning(f"Error processing signals at {current_time}: {str(e)}")
                continue

    # Calculate final results
    final_balance = balance
    total_profit = final_balance - config.initial_balance
    
    # Calculate win rates
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    fixed_tp_win_rate = fixed_tp_wins / fixed_tp_trades if fixed_tp_trades > 0 else 0
    adaptive_tp_win_rate = adaptive_tp_wins / adaptive_tp_trades if adaptive_tp_trades > 0 else 0
    
    # Calculate profit factor
    winning_profits = sum(t.profit for t in trades if t.profit is not None and t.profit > 0)
    losing_profits = abs(sum(t.profit for t in trades if t.profit is not None and t.profit < 0))
    profit_factor = winning_profits / losing_profits if losing_profits > 0 else float('inf')
    
    # Calculate average ATR for winning and losing trades
    winning_atrs = [t.atr for t in trades if t.profit is not None and t.profit > 0]
    losing_atrs = [t.atr for t in trades if t.profit is not None and t.profit < 0]
    avg_winning_atr = sum(winning_atrs) / len(winning_atrs) if winning_atrs else 0
    avg_losing_atr = sum(losing_atrs) / len(losing_atrs) if losing_atrs else 0
    
    # Save trades to CSV
    csv_filename = f"backtest_trades_{config.product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    trades_df = pd.DataFrame([{
        'No.': i + 1,
        'Timestamp': t.entry_time,
        'SIDE': t.type,
        'ENTRY': round(t.entry_price, 2),
        'Take Profit': round(t.take_profit, 2),
        'Stop Loss': round(t.stop_loss, 2),
        'R/R Ratio': round(t.rr_ratio, 2),
        'Volatility Level': t.volatility_level,
        'Outcome': t.outcome,
        'Outcome %': round(t.outcome_percent, 1) if t.outcome_percent is not None else None,
        'Leverage': t.leverage,
        'Margin': round(t.margin, 1),
        'Session': t.session,
        'TP Mode': t.tp_mode,
        'ATR %': round(t.atr_percent, 2),
        'Setup Type': t.strategy,
        'MAE': round(t.mae, 2),
        'MFE': round(t.mfe, 2),
        'Exit Trade': round(t.exit_price, 2) if t.exit_price is not None else None,
        'Trend Regime': t.regime,
        'RSI at Entry': round(t.rsi, 2),
        'Relative Volume': round(t.relative_volume, 2),
        'Trend Slope': round(t.trend_slope, 4),
        'Exit Reason': t.exit_reason,
        'Duration': round(t.duration, 2)
    } for i, t in enumerate(trades)])
    trades_df.to_csv(csv_filename, index=False)
    
    return BacktestResults(
        initial_balance=config.initial_balance,
        final_balance=final_balance,
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_profit=total_profit,
        profit_factor=profit_factor,
        avg_winning_atr=avg_winning_atr,
        avg_losing_atr=avg_losing_atr,
        fixed_tp_trades=fixed_tp_trades,
        adaptive_tp_trades=adaptive_tp_trades,
        fixed_tp_win_rate=fixed_tp_win_rate,
        adaptive_tp_win_rate=adaptive_tp_win_rate,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        trades=trades,
        csv_filename=csv_filename
    )

def display_results(results: BacktestResults) -> None:
    """Display backtest results in a formatted table."""
    if not results:
        logger.error("No results to display")
        return
    
    # Create summary table
    summary_data = [
        ["Initial Balance", f"${results.initial_balance:,.2f}"],
        ["Final Balance", f"${results.final_balance:,.2f}"],
        ["Total Profit", f"${results.total_profit:,.2f}"],
        ["Profit %", f"{(results.total_profit / results.initial_balance * 100):.2f}%"],
        ["Total Trades", results.total_trades],
        ["Winning Trades", results.winning_trades],
        ["Losing Trades", results.losing_trades],
        ["Win Rate", f"{results.win_rate * 100:.2f}%"],
        ["Profit Factor", f"{results.profit_factor:.2f}"],
        ["Max Drawdown", f"{results.max_drawdown * 100:.2f}%"],
        ["Max Drawdown Duration", f"{results.max_drawdown_duration} periods"],
        ["Fixed TP Trades", results.fixed_tp_trades],
        ["Adaptive TP Trades", results.adaptive_tp_trades],
        ["Fixed TP Win Rate", f"{results.fixed_tp_win_rate * 100:.2f}%"],
        ["Adaptive TP Win Rate", f"{results.adaptive_tp_win_rate * 100:.2f}%"],
        ["Avg Winning ATR", f"{results.avg_winning_atr:.2f}"],
        ["Avg Losing ATR", f"{results.avg_losing_atr:.2f}"]
    ]
    
    print("\n=== BACKTEST RESULTS ===:")
    print(tabulate(summary_data, tablefmt="grid"))
    
    if results.csv_filename:
        print(f"\nDetailed trade history saved to: {results.csv_filename}")

def main():
    """Main function to run the backtest."""
    config = parse_args()
    results = run_backtest(config)
    display_results(results)

if __name__ == "__main__":
    main() 