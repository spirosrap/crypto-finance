from trading_bot_framework import (
    CoinbaseService, TechnicalAnalysis, CONFIG, get_perp_product,
    get_price_precision, detect_regime, detect_rsi_dip, detect_breakout,
    filter_by_volume, filter_by_volatility, compute_tp_sl, risk_check,
    fetch_candles, calculate_trend_slope
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

# Constants from CONFIG
GRANULARITY = CONFIG['GRANULARITY']
RSI_THRESHOLD = CONFIG['RSI_CONFIRMATION'] 
VOLUME_LOOKBACK = CONFIG['VOLUME_LOOKBACK']
TP_PERCENT = CONFIG['FIXED_TP_PERCENT']
SL_PERCENT = CONFIG['FIXED_SL_PERCENT']

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
            lambda row: compute_tp_sl(row['entry_price'], row['market_regime'], row['atr'])[0], 
            axis=1
        ).round(2)
        trades_df['Stop Loss'] = (trades_df['entry_price'] * (1 - SL_PERCENT)).round(2)
        trades_df['R/R Ratio'] = ((trades_df['Take Profit'] - trades_df['ENTRY']) / 
                                 (trades_df['ENTRY'] - trades_df['Stop Loss'])).round(2)

        trades_df['Volatility Level'] = trades_df['atr_percent'].apply(
            lambda atr_percent: "Very Strong" if atr_percent > CONFIG['MEAN_ATR_PERCENT'] + CONFIG['STD_ATR_PERCENT'] else 
                                "Strong" if atr_percent > CONFIG['MEAN_ATR_PERCENT'] else 
                                "Moderate" if atr_percent > CONFIG['MEAN_ATR_PERCENT'] - CONFIG['STD_ATR_PERCENT'] else 
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
        trades_df['Exit Reason'] = trades_df['type'].apply(lambda x: 'TP HIT' if x == 'TP' else 'SL HIT')
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

def analyze(historical_df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str) -> Tuple[bool, float, float, float, float]:
    """Analyze market data and generate signals."""
    signal = False
    entry_price = None
    rsi_value = 0
    relative_volume = 0
    trend_slope = 0
    
    try:
        # Check if DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in historical_df.columns for col in required_columns):
            logger.error(f"Missing required columns in DataFrame. Available columns: {historical_df.columns.tolist()}")
            return signal, entry_price, rsi_value, relative_volume, trend_slope
            
        # Convert DataFrame to candles format for ta functions
        candles = historical_df.to_dict('records')
        
        # Detect RSI dip
        rsi_signal, rsi_entry = detect_rsi_dip(historical_df, ta, candles, product_id)
        
        # Calculate RSI
        rsi_value = ta.compute_rsi(product_id, candles, CONFIG['RSI_PERIOD'])
        
        # Calculate relative volume
        avg_volume = historical_df['volume'].rolling(VOLUME_LOOKBACK).mean()
        relative_volume = historical_df['volume'].iloc[-1] / avg_volume.iloc[-1]
        
        # Calculate trend slope
        trend_slope = calculate_trend_slope(historical_df)
        
        # If RSI signal and volume is above threshold
        if rsi_signal:
            signal = True
            entry_price = rsi_entry
    
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
    
    return signal, entry_price, rsi_value, relative_volume, trend_slope

def determine_tp_mode(entry_price: float, atr: float, price_precision: Optional[int], historical_df=None, trend_slope=None) -> Tuple[str, float, str]:
    """Determine the TP mode (FIXED or ADAPTIVE) and calculate TP price."""
    atr_percent = (atr / entry_price) * 100
    
    # Get market regime from historical data if available
    regime = "UNCERTAIN"
    if historical_df is not None:
        # Initialize services for regime detection if not passed
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        ta = TechnicalAnalysis(cb)
        
        # Detect market regime
        regime = detect_regime(historical_df, ta, historical_df.to_dict('records'))
    
    # Calculate TP price based on regime and ATR
    tp, _ = compute_tp_sl(entry_price, regime, atr)
    
    # Determine TP mode
    adaptive_trigger = CONFIG['MEAN_ATR_PERCENT'] + CONFIG['STD_ATR_PERCENT']
    tp_mode = "ADAPTIVE" if atr_percent > adaptive_trigger and regime == "TRENDING" else "FIXED"
    
    # Round TP price if precision is specified
    if price_precision is not None:
        tp = round(tp, price_precision)
    
    return tp_mode, tp, regime

def prepare_dataframe(raw_df, raw_candles):
    """
    Prepare and standardize the DataFrame for backtesting.
    """
    # Check if DataFrame has required columns 
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # If using raw_candles to prepare the DataFrame
    if raw_df.empty or not all(col in raw_df.columns for col in required_columns):
        logger.info("Creating DataFrame from raw candles data")
        
        # Create DataFrame from raw candles
        df = pd.DataFrame(raw_candles)
        
        # Convert string columns to numeric
        for col in required_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle timestamp conversion
        timestamp_cols = ['start', 'timestamp', 'time']
        for col in timestamp_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(pd.to_numeric(df[col]), unit='s', utc=True)
                df.set_index(col, inplace=True)
                break
                
        return df
    
    return raw_df

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
    
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns in DataFrame. Available columns: {df.columns.tolist()}")
        return None
    
    for i in range(50, len(df)):  # Start after EMA period
        # Get historical data up to current point
        historical_df = df.iloc[:i+1]
        signal, entry, rsi_value, relative_volume, trend_slope = analyze(historical_df, ta, config.product_id)
        
        # Handle open position
        if current_trade:
            current_price = df.iloc[i]['close']
            atr = ta.compute_atr(historical_df.to_dict('records'))
            tp_mode, tp_price, market_regime = determine_tp_mode(current_trade['entry_price'], atr, None, historical_df)
            
            # Calculate MAE and MFE for the current trade
            price_change_percent = (current_price - current_trade['entry_price']) / current_trade['entry_price'] * 100
            
            # Update MAE (negative price change)
            if price_change_percent < 0:
                current_trade['mae'] = min(current_trade['mae'], price_change_percent)
            
            # Update MFE (positive price change)
            if price_change_percent > 0:
                current_trade['mfe'] = max(current_trade['mfe'], price_change_percent)
            
            # Check for TP or SL
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check if high price reached take profit
            if current_high >= tp_price:
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
                    rsi_at_entry=current_trade['rsi_at_entry'],
                    relative_volume=current_trade['relative_volume'],
                    trend_slope=current_trade['trend_slope'],
                    market_regime=current_trade['market_regime'],
                    mae=current_trade['mae'],
                    mfe=current_trade['mfe']
                ))
                if len(trades) >= 200:
                    break
                current_trade = None
                position = 0
                
            # Check if low price reached stop loss
            elif current_low <= current_trade['sl_price']:
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
                    rsi_at_entry=current_trade['rsi_at_entry'],
                    relative_volume=current_trade['relative_volume'],
                    trend_slope=current_trade['trend_slope'],
                    market_regime=current_trade['market_regime'],
                    mae=current_trade['mae'],
                    mfe=current_trade['mfe']
                ))
                if len(trades) >= 200:
                    break
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
            
            # Keep SL fixed at percentage from CONFIG
            sl_price = round(current_price * (1 - CONFIG['FIXED_SL_PERCENT']), get_price_precision(get_perp_product(config.product_id)))
            
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
    if results is None:
        logger.error("No results to print")
        return
        
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

def fetch_candles_custom(cb: CoinbaseService, product_id: str = 'BTC-USDC', 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> Tuple[pd.DataFrame, List]:
    """
    Custom version of fetch_candles that handles the data more carefully to avoid KeyErrors.
    Falls back to spot product if perpetual data is not available.
    """
    # Convert dates to datetime objects
    if start_date and end_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=UTC)
        end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=UTC)
    else:
        # Default to last 8000 5-minute candles
        now = datetime.now(UTC)
        start = now - timedelta(minutes=5 * 8000)
        end = now
    
    logger.info(f"Fetching data for {product_id} from {start} to {end}")
    
    # First try with perpetual product
    perp_product = get_perp_product(product_id)
    logger.info(f"Using perpetual product: {perp_product}")
    
    # Fetch historical data for perpetual
    raw_candles = cb.historical_data.get_historical_data(perp_product, start, end, CONFIG['GRANULARITY'])
    logger.info(f"Fetched {len(raw_candles)} candles for perpetual product")
    
    # If no data for perpetual, fall back to spot product
    if not raw_candles:
        logger.warning(f"No candles returned for {perp_product}. Falling back to spot product {product_id}")
        raw_candles = cb.historical_data.get_historical_data(product_id, start, end, CONFIG['GRANULARITY'])
        logger.info(f"Fetched {len(raw_candles)} candles for spot product")
    
    if not raw_candles:
        logger.error("No candles returned from API for either perpetual or spot product")
        return pd.DataFrame(), []
        
    # Log first candle to see structure
    logger.info(f"First candle structure: {raw_candles[0]}")
    
    # Create DataFrame from candles
    df = pd.DataFrame(raw_candles)
    logger.info(f"Initial DataFrame columns: {df.columns.tolist()}")
    
    # Check which columns exist and convert to numeric where appropriate
    expected_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in expected_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Try lowercase version if not found
            lowercase_col = col.lower()
            if lowercase_col in df.columns:
                df[col] = pd.to_numeric(df[lowercase_col], errors='coerce')
                df = df.rename(columns={lowercase_col: col})
            else:
                logger.warning(f"Column {col} not found in DataFrame")
    
    # Handle timestamp conversion
    timestamp_cols = ['start', 'timestamp', 'time']
    timestamp_found = False
    
    for col in timestamp_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(pd.to_numeric(df[col]), unit='s', utc=True)
                df.set_index(col, inplace=True)
                timestamp_found = True
                logger.info(f"Using {col} as timestamp column")
                break
            except:
                logger.warning(f"Failed to convert {col} to timestamp")
    
    if not timestamp_found and df.shape[0] > 0:
        logger.warning("No timestamp column found, creating index")
        df.index = pd.date_range(
            start=start, 
            periods=len(df), 
            freq='5T'  # 5 minute intervals
        )
    
    return df, raw_candles

def main():
    """Main entry point for the backtest script."""
    config = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data with custom function instead of the one from the framework
        df, raw_candles = fetch_candles_custom(cb, config.product_id, config.start_date, config.end_date)
        
        # Process the data and make sure it has the necessary format
        df = prepare_dataframe(df, raw_candles)
        
        # Check if DataFrame has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if df.empty or not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns in DataFrame. Available columns: {df.columns.tolist() if not df.empty else 'DataFrame is empty'}")
            return
        
        logger.info("Starting backtest...")
        if config.start_date and config.end_date:
            logger.info(f"Backtest period: {config.start_date} to {config.end_date}")
        else:
            logger.info("Using default period (last 8000 5-minute candles)")
        
        # Log data info
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"First 5 rows: \n{df.head()}")
        
        # Run backtest
        results = backtest(df, ta, config)
        
        # Print results
        print_results(results)
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 