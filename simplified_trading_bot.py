# Simplified Trading Bot
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes

from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from datetime import datetime, timedelta, UTC
import pandas as pd
from config import API_KEY_PERPS, API_SECRET_PERPS
import logging

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
PAIR = "BTC-PERP-INTX"  # Changed to perpetual futures pair
GRANULARITY = "FIVE_MINUTE"
RSI_THRESHOLD = 30
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007
LEVERAGE = 5  # Conservative leverage
POSITION_SIZE_USD = 100  # Position size in USD

# Services
cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
ta = TechnicalAnalysis(cb)

# Fetch candles
def fetch_candles():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=5 * 200)
    df = cb.historical_data.get_historical_data(PAIR, start, now, GRANULARITY)
    df = pd.DataFrame(df)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Analyze market
def analyze(df: pd.DataFrame):
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    # Calculate RSI
    rsi = ta.compute_rsi(PAIR, candles, period=14)
    
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

def execute_trade(entry_price: float):
    """Execute the trade using trade_btc_perp.py functions"""
    try:
        # Calculate take profit and stop loss prices
        tp_price = entry_price * (1 + TP_PERCENT)
        sl_price = entry_price * (1 - SL_PERCENT)
        
        # Get current market price
        trades = cb.client.get_market_trades(product_id=PAIR, limit=1)
        current_price = float(trades['trades'][0]['price'])
        
        # Calculate base size
        size = POSITION_SIZE_USD / current_price
        
        # Place market order with targets
        result = cb.place_market_order_with_targets(
            product_id=PAIR,
            side="BUY",
            size=size,
            take_profit_price=tp_price,
            stop_loss_price=sl_price,
            leverage=str(LEVERAGE)
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

# Execute logic
if __name__ == "__main__":
    try:
        candles = fetch_candles()
        signal, entry = analyze(candles)
        if signal:
            logger.info(f"[SIGNAL] BUY {PAIR} at {entry:.2f}")
            if execute_trade(entry):
                logger.info("Trade executed successfully!")
            else:
                logger.error("Failed to execute trade")
        else:
            logger.info("[NO SIGNAL] Conditions not met.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
