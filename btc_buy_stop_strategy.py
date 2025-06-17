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
        logger.info("Making API call...")
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=start_ts,
            end=end_ts,
            granularity='ONE_HOUR'
        )
        
        logger.info("API call completed")
        
        # Convert response to dict if it's a Coinbase response type
        if hasattr(response, 'candles'):
            candles = response.candles
        else:
            candles = response.get('candles', [])
            
        if not candles or len(candles) < 2:
            logger.error(f"Not enough candles in response: {len(candles) if candles else 0}")
            return None
            
        # Get the most recent complete candle (second to last in the list)
        last_candle = candles[-2]  # Get the second to last candle
        logger.info(f"Selected candle: {last_candle}")
        
        # Convert the candle data to the required format
        try:
            # Access dictionary values directly
            candle_dict = {
                'timestamp': datetime.fromtimestamp(int(last_candle['start']), UTC),
                'open': float(last_candle['open']),
                'high': float(last_candle['high']),
                'low': float(last_candle['low']),
                'close': float(last_candle['close']),
                'volume': float(last_candle['volume'])
            }
            logger.info(f"Processed candle data: {candle_dict}")
            return candle_dict
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing candle data: {e}")
            logger.error(f"Problematic candle data: {last_candle}")
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

def get_next_poll_time():
    """Calculate the next time to poll based on the current time."""
    now = datetime.now(UTC)
    # If we're in the first minute of the hour, wait until next hour
    if now.minute == 0:
        next_time = now.replace(minute=1, second=0, microsecond=0)
    else:
        next_time = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)        
    
    # Calculate seconds until next poll
    wait_seconds = (next_time - now).total_seconds()
    return wait_seconds

def main():
    logger.info("Starting BTC buy-stop strategy")
    logger.info("Monitoring for hourly close above $107,000")
    logger.info("Strategy requires price to first close above $106,000 (guard level)")
    
    cb_service = setup_coinbase()
    order_placed = False
    guard_armed = False  # Track if price has broken above guard level
    last_candle_ts = None  # Track last processed candle timestamp
    while not order_placed:
        try:
            # Get hourly candle
            candle = get_hourly_candle(cb_service)

            if candle:
                current_ts = candle['timestamp']
                
                # Skip if we've already processed this candle
                if current_ts == last_candle_ts:
                    logger.info("Skipping already processed candle")
                    time.sleep(get_next_poll_time())
                    continue
                
                close_price = float(candle['close'])
                logger.info(f"Processing new candle at {current_ts} with close: ${close_price:,.2f}")
                
                # 1. Entry test first (using previous guard state)
                if guard_armed and close_price >= 107000:
                    logger.info("Entry condition met! Price closed above $107,000 with armed guard from previous candle")
                    place_buy_stop_order()
                    order_placed = True
                    guard_armed = False  # Reset guard after entry
                else:
                    logger.info("Waiting for price to close above $107,000 (guard armed: {})".format(guard_armed))
                
                # 2. Guard update second (for next candle)
                guard_armed = close_price >= 106000
                if guard_armed:
                    logger.info("Guard armed for next candle (price closed above $106,000)")
                else:
                    logger.info("Guard disarmed for next candle (price closed below $106,000)")
                
                # Update last processed timestamp
                last_candle_ts = current_ts
            else:
                logger.info("Waiting for current hour to complete...")
            
            # Calculate and wait until next optimal poll time
            wait_time = get_next_poll_time()
            logger.info(f"Waiting {wait_time:.0f} seconds until next poll")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            logger.info("Strategy stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    main() 