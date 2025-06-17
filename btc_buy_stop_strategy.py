import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import subprocess
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameters
GRANULARITY = "ONE_HOUR"  # Match the format used in Coinbase API
PRODUCT_ID = "BTC-PERP-INTX"

def setup_coinbase():
    """Initialize CoinbaseService with API credentials."""
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    
    return CoinbaseService(api_key, api_secret)

def get_hourly_candle(cb_service):
    """Get the current hourly candle data."""
    try:
        # Get the last 2 hours of data
        now = datetime.now(UTC)
        # Round down to the last completed hour
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=2)
        end = now
        
        # Convert to Unix timestamps
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching candles from {start} to {end}")
        logger.info(f"Unix timestamps: {start_ts} to {end_ts}")
        
        # Get candles directly from the client
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=start_ts,
            end=end_ts,
            granularity='ONE_HOUR'
        )
        
        logger.info(f"Received response: {response}")
        
        if not response or 'candles' not in response or len(response['candles']) < 2:
            logger.error("Not enough candle data available")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(response['candles'])
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame head:\n{df.head()}")
        
        # Convert string columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime - explicitly convert start to numeric first
        df['start'] = pd.to_numeric(df['start'])
        df['timestamp'] = pd.to_datetime(df['start'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        
        # Get the most recent complete candle
        last_candle = df.iloc[-2]
        last_candle_time = last_candle.name
        
        # Check if the candle is from the previous hour
        if last_candle_time.hour == (now - timedelta(hours=1)).hour:
            logger.info(f"Using completed candle from {last_candle_time}")
            return last_candle.to_dict()
        else:
            logger.info(f"Most recent candle is not complete yet (from {last_candle_time})")
            return None
        
    except Exception as e:
        logger.error(f"Error getting hourly candle: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def place_buy_stop_order(size_usd=4000, leverage=20, tp_price=112000, sl_price=104900):
    """Place a buy-stop order using trade_btc_perp.py."""
    try:
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', PRODUCT_ID,
            '--side', 'BUY',
            '--size', str(size_usd),
            '--leverage', str(leverage),
            '--tp', str(tp_price),
            '--sl', str(sl_price),
            '--limit', '107000'
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Buy-stop order placed successfully")
            logger.info(f"Output: {result.stdout}")
        else:
            logger.error(f"Error placing buy-stop order: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error executing trade_btc_perp.py: {e}")

def main():
    logger.info("Starting BTC buy-stop strategy")
    logger.info("Monitoring for hourly close above $107,000")
    logger.info("Strategy will abort if price closes below $106,000")
    
    cb_service = setup_coinbase()
    order_placed = False
    
    while not order_placed:
        try:
            # Get current time
            current_time = datetime.now(UTC)
            
            # Get hourly candle
            candle = get_hourly_candle(cb_service)
            
            if candle:
                close_price = float(candle['close'])
                logger.info(f"Current hourly close: ${close_price:,.2f}")
                
                # Check if price closed below $106,000 (cancel condition)
                if close_price < 106000:
                    logger.info("Price closed below $106,000 - aborting strategy")
                    return
                
                # Check if price closed above $107,000
                if close_price > 107000:
                    logger.info("Hourly close above $107,000 detected!")
                    place_buy_stop_order()
                    order_placed = True
                else:
                    logger.info("Waiting for price to close above $107,000...")
            else:
                logger.info("Waiting for current hour to complete...")
            
            # Wait for 5 minutes before next check
            time.sleep(300)
            
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    main() 