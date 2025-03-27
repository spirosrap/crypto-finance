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
    start = now - timedelta(minutes=5 * 200)
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
        and current["close"] > ema_50
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

def main():
    args = parse_args()
    
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    try:
        # Fetch and analyze candles
        df = fetch_candles(cb, args.product_id)
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
