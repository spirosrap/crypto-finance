# Simplified Trading Bot
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes

from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from datetime import datetime, timedelta, UTC
import pandas as pd
from config import API_KEY_PERPS, API_SECRET_PERPS
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress logs from other modules
logging.getLogger('technicalanalysis').setLevel(logging.WARNING)
logging.getLogger('historicaldata').setLevel(logging.WARNING)
logging.getLogger('bitcoinpredictionmodel').setLevel(logging.WARNING)
logging.getLogger('ml_model').setLevel(logging.WARNING)

# Parameters
GRANULARITY = "FIVE_MINUTE"
RSI_THRESHOLD = 30
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD

def parse_args():
    parser = argparse.ArgumentParser(description='Simplified Trading Bot')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Product ID to trade (e.g., BTC-USDC)')
    parser.add_argument('--margin', type=float, default=100,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage')
    parser.add_argument('--backtest', action='store_true',
                      help='Run in backtest mode')
    parser.add_argument('--initial_balance', type=float, default=10000,
                      help='Initial balance for backtesting')
    return parser.parse_args()

def get_perp_product(product_id):
    """Convert spot product ID to perpetual futures product ID"""
    perp_map = {
        'BTC-USDC': 'BTC-PERP-INTX',
        'ETH-USDC': 'ETH-PERP-INTX',
        'DOGE-USDC': 'DOGE-PERP-INTX',
        'SOL-USDC': 'SOL-PERP-INTX',
        'SHIB-USDC': '1000SHIB-PERP-INTX'
    }
    return perp_map.get(product_id, 'BTC-PERP-INTX')

def get_price_precision(product_id):
    """Get price precision for a product"""
    precision_map = {
        'BTC-PERP-INTX': 1,      # $1 precision for BTC
        'ETH-PERP-INTX': 0.1,    # $0.1 precision for ETH
        'DOGE-PERP-INTX': 0.0001, # $0.0001 precision for DOGE
        'SOL-PERP-INTX': 0.01,   # $0.01 precision for SOL
        '1000SHIB-PERP-INTX': 0.000001  # $0.000001 precision for SHIB
    }
    return precision_map.get(product_id, 1)

def fetch_candles(cb, product_id):
    now = datetime.now(UTC)
    start = now - timedelta(minutes=5 * 8000)
    df = cb.historical_data.get_historical_data(product_id, start, now, GRANULARITY)
    df = pd.DataFrame(df)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def analyze(df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str):
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    # Calculate RSI
    rsi = ta.compute_rsi(product_id, candles, period=14)
    
    # Calculate EMA
    ema_50 = ta.get_moving_average(candles, period=50, ma_type='ema')
    
    # Get current values
    current = df.iloc[-1]
    avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()

    if (
        rsi < RSI_THRESHOLD
        # and current["close"] > ema_50
        and current["volume"] > avg_volume
    ):
        return True, current["close"]
    return False, None

def execute_trade(cb, entry_price: float, product_id: str, position_size: float, leverage: int):
    """Execute the trade using trade_btc_perp.py functions"""
    try:
        # Convert to perpetual futures product ID
        perp_product = get_perp_product(product_id)
        price_precision = get_price_precision(perp_product)
        
        # Calculate take profit and stop loss prices
        tp_price = round(entry_price * (1 + TP_PERCENT), price_precision)
        sl_price = round(entry_price * (1 - SL_PERCENT), price_precision)
        
        # Get current market price
        trades = cb.client.get_market_trades(product_id=perp_product, limit=1)
        current_price = float(trades['trades'][0]['price'])
        
        # Calculate base size
        size = position_size / current_price
        
        # Place market order with targets
        result = cb.place_market_order_with_targets(
            product_id=perp_product,
            side="BUY",
            size=size,
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
            leverage=str(leverage)
        )
        
        if "error" in result:
            logger.error(f"Error placing order: {result['error']}")
            return False
            
        logger.info("Order placed successfully!")
        logger.info(f"Order ID: {result['order_id']}")
        logger.info(f"Take Profit Price: ${result['tp_price']}")
        logger.info(f"Stop Loss Price: ${result['sl_price']}")
        return True
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return False

def backtest(df: pd.DataFrame, ta: TechnicalAnalysis, product_id: str, initial_balance: float = 10000, leverage: int = 5):
    """
    Backtest the trading strategy on historical data
    Returns performance metrics and trade history
    """
    balance = initial_balance
    position = 0
    trades = []
    current_trade = None
    
    for i in range(50, len(df)):  # Start after EMA period
        # Get historical data up to current point
        historical_df = df.iloc[:i+1]
        signal, entry = analyze(historical_df, ta, product_id)
        
        # If we have an open position, check if TP or SL is hit
        if current_trade:
            current_price = df.iloc[i]['close']
            
            # Check if take profit or stop loss is hit
            if current_price >= current_trade['tp_price']:
                # Take profit hit
                profit = (current_trade['tp_price'] - current_trade['entry_price']) * current_trade['size'] * leverage
                balance += profit
                trades.append({
                    'entry_time': current_trade['entry_time'],
                    'exit_time': df.index[i],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': current_trade['tp_price'],
                    'profit': profit,
                    'type': 'TP'
                })
                if len(trades) >= 50:
                    break                
                current_trade = None
                position = 0
                
            elif current_price <= current_trade['sl_price']:
                # Stop loss hit
                loss = (current_trade['sl_price'] - current_trade['entry_price']) * current_trade['size'] * leverage
                balance += loss
                trades.append({
                    'entry_time': current_trade['entry_time'],
                    'exit_time': df.index[i],
                    'entry_price': current_trade['entry_price'],
                    'exit_price': current_trade['sl_price'],
                    'profit': loss,
                    'type': 'SL'
                })
                if len(trades) >= 50:
                    break                
                current_trade = None
                position = 0
        
        # If no position and signal, enter trade
        elif signal and not current_trade:
            current_price = df.iloc[i]['close']
            tp_price = round(current_price * (1 + TP_PERCENT), get_price_precision(get_perp_product(product_id)))
            sl_price = round(current_price * (1 - SL_PERCENT), get_price_precision(get_perp_product(product_id)))
            
            # Calculate position size
            position_size = balance * 0.1  # Use 10% of balance per trade
            size = position_size / current_price
            
            current_trade = {
                'entry_time': df.index[i],
                'entry_price': current_price,
                'size': size,
                'tp_price': tp_price,
                'sl_price': sl_price
            }
            position = size
    
    # Calculate performance metrics
    if trades:
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['profit'] > 0])
        losing_trades = len([t for t in trades if t['profit'] < 0])
        total_profit = sum(t['profit'] for t in trades)
        win_rate = (winning_trades / total_trades) * 100
        profit_factor = abs(sum(t['profit'] for t in trades if t['profit'] > 0) / 
                          sum(t['profit'] for t in trades if t['profit'] < 0)) if losing_trades > 0 else float('inf')
        
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'profit_factor': profit_factor,
            'trades': trades
        }
    else:
        return {
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_profit': 0,
            'profit_factor': 0,
            'trades': []
        }

def main():
    args = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch historical data
        df = fetch_candles(cb, args.product_id)
        
        if args.backtest:
            logger.info("Starting backtest...")
            results = backtest(df, ta, args.product_id, args.initial_balance, args.leverage)
            
            # Print backtest results
            logger.info("\nBacktest Results:")
            logger.info(f"Initial Balance: ${results['initial_balance']:.2f}")
            logger.info(f"Final Balance: ${results['final_balance']:.2f}")
            logger.info(f"Total Profit: ${results['total_profit']:.2f}")
            logger.info(f"Total Trades: {results['total_trades']}")
            logger.info(f"Winning Trades: {results['winning_trades']}")
            logger.info(f"Losing Trades: {results['losing_trades']}")
            logger.info(f"Win Rate: {results['win_rate']:.2f}%")
            logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
            
            # Print trade history
            logger.info("\nTrade History:")
            for trade in results['trades']:
                logger.info(f"Entry: {trade['entry_time']} @ ${trade['entry_price']:.2f}")
                logger.info(f"Exit: {trade['exit_time']} @ ${trade['exit_price']:.2f}")
                logger.info(f"Profit: ${trade['profit']:.2f} ({trade['type']})")
                logger.info("---")
        else:
            # Live trading mode
            signal, entry = analyze(df, ta, args.product_id)
            
            if signal:
                logger.info(f"[SIGNAL] BUY {args.product_id} at {entry:.2f}")
                if execute_trade(cb, entry, args.product_id, args.margin, args.leverage):
                    logger.info("Trade executed successfully!")
                else:
                    logger.error("Failed to execute trade")
            else:
                logger.info("[NO SIGNAL] Conditions not met.")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
