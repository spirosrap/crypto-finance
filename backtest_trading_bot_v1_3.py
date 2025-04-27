from simplified_trading_bot_v1_3 import (
    CoinbaseService, TechnicalAnalysis, GRANULARITY, RSI_THRESHOLD,
    VOLUME_LOOKBACK, TP_PERCENT, SL_PERCENT, get_perp_product,
    get_price_precision, analyze, determine_tp_mode, EMA_WINDOW, ema, slope
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
import traceback  # Add for detailed error tracking

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

def export_trades_to_csv(trades: List[Trade], product_id: str, leverage: int = 5) -> str:
    """Export trade history to CSV file."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_filename = f"backtest_trades_{product_id}_{timestamp}.csv"
    
    try:
        trades_df = pd.DataFrame([vars(t) for t in trades])
        
        # Add columns to match automated_trades.csv format
        trades_df['No.'] = range(1, len(trades_df) + 1)
        
        # Determine trade side from exit reason
        trades_df['SIDE'] = trades_df['exit_reason'].apply(
            lambda x: 'SHORT' if 'SHORT' in x else 'LONG'
        )
        
        trades_df['ENTRY'] = trades_df['entry_price'].round(2)
        
        # Calculate Take Profit and Stop Loss based on trade side
        def calculate_tp_sl(row):
            if 'SHORT' in row['exit_reason']:
                tp = row['entry_price'] * (1 - TP_PERCENT)
                sl = row['entry_price'] * (1 + SL_PERCENT)
            else:
                tp = row['entry_price'] * (1 + TP_PERCENT)
                sl = row['entry_price'] * (1 - SL_PERCENT)
            return pd.Series([tp, sl])
        
        trades_df[['Take Profit', 'Stop Loss']] = trades_df.apply(calculate_tp_sl, axis=1)
        trades_df['Take Profit'] = trades_df['Take Profit'].round(2)
        trades_df['Stop Loss'] = trades_df['Stop Loss'].round(2)
        
        # Calculate R/R Ratio based on trade side
        def calculate_rr_ratio(row):
            if row['SIDE'] == 'SHORT':
                return abs((row['entry_price'] - row['Take Profit']) / (row['Stop Loss'] - row['entry_price']))
            else:
                return abs((row['Take Profit'] - row['entry_price']) / (row['entry_price'] - row['Stop Loss']))
        
        trades_df['R/R Ratio'] = trades_df.apply(calculate_rr_ratio, axis=1).round(2)

        trades_df['Volatility Level'] = trades_df['atr_percent'].apply(
            lambda atr_percent: "Very Strong" if atr_percent > mean_atr_percent + std_atr_percent else 
                                "Strong" if atr_percent > mean_atr_percent else 
                                "Moderate" if atr_percent > mean_atr_percent - std_atr_percent else 
                                "Weak"
        )
        
        # Set outcome based on the type of exit (TP = success, SL = stop loss)
        trades_df['Outcome'] = trades_df['type'].apply(lambda x: 'SUCCESS' if x == 'TP' else 'STOP LOSS')
        
        # Fix for outcome percentages - use Take Profit or Stop Loss prices directly
        trades_df['Exit Trade'] = trades_df.apply(
            lambda row: row['Take Profit'] if row['type'] == 'TP' else row['Stop Loss'], 
            axis=1
        )
        
        # Calculate Outcome % based on the correct exit price and trade direction
        def calculate_outcome_percentage(row):
            if row['SIDE'] == 'SHORT':
                # For SHORT trades, the profit is entry_price - exit_price
                # Positive percentage means the price went down (success)
                return round((row['ENTRY'] - row['Exit Trade']) / row['ENTRY'] * 100 * leverage, 2)
            else:
                # For LONG trades, profit is exit_price - entry_price
                return round((row['Exit Trade'] - row['ENTRY']) / row['ENTRY'] * 100 * leverage, 2)
                
        trades_df['Outcome %'] = trades_df.apply(calculate_outcome_percentage, axis=1)
        
        # Make sure success trades have positive outcomes and failed trades have negative
        def fix_outcome_sign(row):
            if (row['type'] == 'TP' and row['Outcome %'] < 0) or (row['type'] == 'SL' and row['Outcome %'] > 0):
                return -row['Outcome %']
            return row['Outcome %']
            
        trades_df['Outcome %'] = trades_df.apply(fix_outcome_sign, axis=1)
        
        trades_df['Leverage'] = f"{leverage}x"
        trades_df['Margin'] = 50.0
        trades_df['Session'] = trades_df['entry_time'].apply(lambda x: 
            'Asia' if 0 <= x.hour < 9 else 
            'EU' if 9 <= x.hour < 17 else 'US')
        trades_df['TP Mode'] = trades_df['tp_mode']
        trades_df['ATR %'] = trades_df['atr_percent'].round(2)
        trades_df['Setup Type'] = trades_df['SIDE'].apply(lambda x: 'EMA Short' if x == 'SHORT' else 'RSI Dip')
        trades_df['MAE'] = trades_df['mae'].round(2)
        trades_df['MFE'] = trades_df['mfe'].round(2)
        trades_df['Trend Regime'] = trades_df['market_regime']
        trades_df['RSI at Entry'] = trades_df['rsi_at_entry'].round(2)
        trades_df['Relative Volume'] = trades_df['relative_volume'].round(2)
        trades_df['Trend Slope'] = trades_df['trend_slope'].round(4)
        trades_df['Exit Reason'] = trades_df['exit_reason']
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
    
    # Debug info
    logger.info(f"Starting backtest with {len(df)} candles")
    
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
    
    # Precompute trend_slope for each candle
    logger.info("Precomputing trend slope for each candle")
    df['trend_slope'] = 0.0  # Initialize column
    for i in range(5, len(df)):
        if i % 1000 == 0:
            logger.info(f"Computing trend slope for candle {i}/{len(df)}")
        close_prices = df['close'].iloc[i-5:i].values
        x = np.arange(len(close_prices))
        y = close_prices
        if len(close_prices) >= 5:
            slope, _ = np.polyfit(x, y, 1)
            mean_price = np.mean(close_prices)
            df.iloc[i, df.columns.get_loc('trend_slope')] = slope / mean_price
    
    for i in range(50, len(df)):  # Start after EMA period
        # Get historical data up to current point
        historical_df = df.iloc[:i+1]
        
        try:
            # Call analyze directly and debug the result
            logger.info(f"Analyzing candle at index {i}, date: {df.index[i]}")
            result = analyze(historical_df, ta, config.product_id)
            
            # Debug the result
            if result is None:
                logger.error("Analyze returned None")
                signal = False
                trade_side = None
                entry_price = None
                tp_price = None
                sl_price = None
                tp_mode = None
                position_size = None
            else:
                logger.info(f"Analyze result type: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'not a sequence'}")
                try:
                    signal, trade_side, entry_price, tp_price, sl_price, tp_mode, position_size = result
                    logger.info(f"Unpacked values: signal={signal}, side={trade_side}, tp={tp_price}, sl={sl_price}")
                except Exception as e:
                    logger.error(f"Error unpacking analyze result: {e}")
                    logger.error(traceback.format_exc())
                    signal = False
                    trade_side = None
                    entry_price = None
                    tp_price = None
                    sl_price = None
                    tp_mode = None
                    position_size = None
        except Exception as e:
            logger.error(f"Error calling analyze: {e}")
            logger.error(traceback.format_exc())
            signal = False
            trade_side = None
            entry_price = None
            tp_price = None
            sl_price = None
            tp_mode = None
            position_size = None
        
        # Calculate RSI for logging
        try:
            candles = historical_df.to_dict('records')
            rsi_value = ta.compute_rsi(config.product_id, candles, period=14)
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            rsi_value = 0
        
        # Calculate relative volume
        try:
            avg_volume = historical_df["volume"].tail(VOLUME_LOOKBACK).mean()
            relative_volume = historical_df.iloc[-1]['volume'] / avg_volume if avg_volume > 0 else 0
        except Exception as e:
            logger.warning(f"Error calculating relative volume: {e}")
            relative_volume = 0
        
        # Get trend slope
        trend_slope = df.iloc[i]['trend_slope']
        
        # Debug for signal
        if signal:
            logger.info(f"Got a {trade_side} trade signal at {df.index[i]}")
        
        # Handle open position
        if current_trade:
            # Ensure current_trade has all required fields
            required_fields = ['tp_price', 'sl_price', 'entry_price', 'size', 'trade_side']
            missing_fields = [field for field in required_fields if field not in current_trade or current_trade[field] is None]
            
            if missing_fields:
                logger.error(f"Current trade is missing required fields: {missing_fields}")
                logger.error(f"Resetting invalid trade")
                current_trade = None
                position = 0
                continue  # Skip to next iteration
                
            current_price = df.iloc[i]['close']
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            try:
                atr = ta.compute_atr(historical_df.to_dict('records'))
            except Exception as e:
                logger.warning(f"Error calculating ATR: {e}")
                atr = current_trade.get('atr', 0)  # Reuse last ATR if calculation fails
            
            # Get trade side with fallback
            trade_side = current_trade.get('trade_side', 'LONG')
            logger.info(f"Managing open {trade_side} position at price {current_price:.2f}, TP: {current_trade['tp_price']:.2f}, SL: {current_trade['sl_price']:.2f}")
            
            try:
                # Determine TP mode based on trade side
                tp_mode_result, tp_price_result, market_regime = determine_tp_mode(
                    current_trade['entry_price'], 
                    atr, 
                    None, 
                    historical_df, 
                    trend_slope, 
                    trade_side
                )
            except Exception as e:
                logger.warning(f"Error determining TP mode: {e}")
                tp_mode_result = current_trade.get('tp_mode', 'FIXED')
                tp_price_result = current_trade.get('tp_price', current_trade['entry_price'] * 1.015)
                market_regime = "UNCERTAIN"
            
            # Calculate MAE and MFE for the current trade
            # Check high for maximum favorable excursion
            if trade_side == 'LONG':
                high_change_percent = (current_high - current_trade['entry_price']) / current_trade['entry_price'] * 100
                low_change_percent = (current_low - current_trade['entry_price']) / current_trade['entry_price'] * 100
            else:  # SHORT
                high_change_percent = (current_trade['entry_price'] - current_low) / current_trade['entry_price'] * 100
                low_change_percent = (current_trade['entry_price'] - current_high) / current_trade['entry_price'] * 100
            
            # Update MFE (maximum favorable excursion)
            current_trade['mfe'] = max(current_trade['mfe'], high_change_percent)
            
            # Update MAE (maximum adverse excursion)
            if low_change_percent < 0:
                current_trade['mae'] = min(current_trade['mae'], low_change_percent)
            
            # Check if take profit hit
            tp_hit = False
            try:
                if trade_side == 'LONG' and current_high >= current_trade['tp_price']:
                    tp_hit = True
                elif trade_side == 'SHORT' and current_low <= current_trade['tp_price']:
                    tp_hit = True
                    
                if tp_hit:
                    profit = 0
                    if trade_side == 'LONG':
                        profit = (current_trade['tp_price'] - current_trade['entry_price']) * current_trade['size'] * config.leverage
                    else:  # SHORT
                        profit = (current_trade['entry_price'] - current_trade['tp_price']) * current_trade['size'] * config.leverage
                    
                    balance += profit
                    # Determine if TP was hit by wick or close
                    if trade_side == 'LONG':
                        exit_reason = "TP WICK HIT" if current_price < current_trade['tp_price'] else "TP HIT"
                    else:  # SHORT
                        exit_reason = "TP WICK HIT SHORT" if current_price > current_trade['tp_price'] else "TP HIT SHORT"
                    
                    trades.append(Trade(
                        entry_time=current_trade['entry_time'],
                        exit_time=df.index[i],
                        entry_price=current_trade['entry_price'],
                        exit_price=current_trade['tp_price'],
                        profit=profit,
                        type='TP',
                        atr=atr,
                        atr_percent=(atr / current_trade['entry_price']) * 100,
                        tp_mode=current_trade['tp_mode'],
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
            except (TypeError, KeyError) as e:
                logger.error(f"Error checking take profit: {e}")
                if current_trade is None:
                    logger.error("current_trade is None during TP check")
                else:
                    logger.error(f"current_trade keys: {current_trade.keys()}")
                    logger.error(f"tp_price missing from current_trade")
                # Reset trade if there's an error to prevent repeated errors
                current_trade = None
                position = 0
        
        # Enter new trade if signal and no position
        elif signal and not current_trade and entry_price is not None:
            try:
                logger.info(f"Opening new {trade_side} trade at {df.index[i]}")
                
                if trade_side is None:
                    logger.error("Trade side is None, defaulting to LONG")
                    trade_side = "LONG"
                    
                # Make sure all values are properly set
                if tp_price is None or sl_price is None or tp_mode is None:
                    logger.warning(f"Some trade parameters are None: tp={tp_price}, sl={sl_price}, mode={tp_mode}")
                    # Recalculate if needed
                    current_price = df.iloc[i]['close']
                    atr = ta.compute_atr(historical_df.to_dict('records'))
                    
                    if tp_price is None or sl_price is None:
                        if trade_side == "LONG":
                            tp_price = entry_price * (1 + TP_PERCENT) if tp_price is None else tp_price
                            sl_price = entry_price * (1 - SL_PERCENT) if sl_price is None else sl_price
                        else:  # SHORT
                            tp_price = entry_price * (1 - TP_PERCENT) if tp_price is None else tp_price
                            sl_price = entry_price * (1 + SL_PERCENT) if sl_price is None else sl_price
                    
                    if tp_mode is None:
                        tp_mode = "FIXED" if trade_side == "LONG" else "FIXED_SHORT"
                
                current_price = df.iloc[i]['close']
                atr = ta.compute_atr(historical_df.to_dict('records'))
                
                # Process TP/SL values - ensure we have values
                final_tp_price = tp_price
                final_sl_price = sl_price
                final_tp_mode = tp_mode
                market_regime = "UNCERTAIN"  # Default
                
                # Get market regime
                try:
                    _, _, market_regime = determine_tp_mode(
                        entry_price, atr, None, historical_df, trend_slope, trade_side
                    )
                except Exception as e:
                    logger.warning(f"Error getting market regime: {e}")
                    # Use default values if calculation fails
                    market_regime = "TRENDING"
                
                # Ensure precision for prices
                price_precision = get_price_precision(get_perp_product(config.product_id))
                final_tp_price = round(final_tp_price, price_precision) if final_tp_price else None
                final_sl_price = round(final_sl_price, price_precision) if final_sl_price else None
                
                # Calculate position size
                position_size_usd = config.initial_balance * 0.1  # Use 10% of balance per trade
                size = position_size_usd / entry_price
                
                # Create trade entry
                current_trade = {
                    'entry_time': df.index[i],
                    'entry_price': entry_price,
                    'size': size,
                    'tp_price': final_tp_price,
                    'sl_price': final_sl_price,
                    'tp_mode': final_tp_mode,
                    'atr': atr,
                    'atr_percent': (atr / entry_price) * 100,
                    'rsi_at_entry': rsi_value,
                    'relative_volume': relative_volume,
                    'trend_slope': trend_slope,
                    'market_regime': market_regime,
                    'mae': 0.0,
                    'mfe': 0.0,
                    'trade_side': trade_side
                }
                position = size
                
                logger.info(f"Opened {trade_side} trade at {entry_price:.2f}, TP: {final_tp_price:.2f}, SL: {final_sl_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error opening new trade: {e}")
                logger.error(traceback.format_exc())
                current_trade = None
        
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
    csv_filename = export_trades_to_csv(trades, config.product_id, config.leverage)
    
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
        try:
            results = backtest(df, ta, config)
            
            # Print results
            print_results(results)
        except Exception as e:
            logger.error(f"Error in backtest function: {e}")
            print(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 