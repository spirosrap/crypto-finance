import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
import pandas_ta as ta
import subprocess
import sys
import platform
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connection retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
MAX_RETRY_DELAY = 60  # seconds
BACKOFF_MULTIPLIER = 2

# Connection error types to handle
CONNECTION_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,  # Covers network-related OS errors
    Exception  # Catch-all for API-specific errors
)

def exponential_backoff_delay(attempt):
    """Calculate exponential backoff delay with jitter"""
    import random
    delay = min(INITIAL_RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt), MAX_RETRY_DELAY)
    # Add jitter to prevent thundering herd
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
    """
    Retry a function with exponential backoff on connection errors
    
    Args:
        func: Function to retry
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Function result on success, None on permanent failure
    """
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except CONNECTION_ERRORS as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"❌ Permanent failure after {MAX_RETRIES} attempts: {e}")
                return None
            
            delay = exponential_backoff_delay(attempt)
            logger.warning(f"⚠️ Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
            time.sleep(delay)
        except Exception as e:
            # For non-connection errors, don't retry
            logger.error(f"❌ Non-recoverable error: {e}")
            return None
    
    return None

def safe_get_candles(cb_service, product_id, start_ts, end_ts, granularity):
    """
    Safely get candles with retry logic
    
    Args:
        cb_service: Coinbase service instance
        product_id: Trading product ID
        start_ts: Start timestamp
        end_ts: End timestamp
        granularity: Candle granularity
    
    Returns:
        Candles list on success, None on failure
    """
    def _get_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity=granularity
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_candles)

# Constants for get_recent_hourly_candles
GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"

# Trade parameters for BTC horizontal resistance breakout
BTC_HORIZONTAL_MARGIN = 300  # USD
BTC_HORIZONTAL_LEVERAGE = 20  # 20x leverage

# Trade parameters for ETH EMA cluster breakout
ETH_EMA_MARGIN = 200  # USD
ETH_EMA_LEVERAGE = 10  # 10x leverage
ETH_EMA_STOP_LOSS = 2500  # Stop-loss at $2500 (consolidation base)
ETH_EMA_TAKE_PROFIT = 3000  # First profit target at $3000 (measured move)

# Trade tracking
btc_continuation_trade_taken = False

def play_alert_sound(filename="alert_sound.wav"):
    """
    Play the alert sound using system commands
    """
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
            cmd = ["afplay", filename]
        elif system == "Linux":
            cmd = ["aplay", filename]
        elif system == "Windows":
            cmd = ["start", "/min", "cmd", "/c", f"powershell -c \"(New-Object Media.SoundPlayer '{filename}').PlaySync()\""]
        else:
            logger.warning(f"Unknown operating system: {system}. Cannot play sound.")
            return False
        
        subprocess.run(cmd, check=True, timeout=5)
        logger.info("Alert sound played successfully")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Sound playback timed out")
        return False
    except Exception as e:
        logger.error(f"Error playing alert sound: {e}")
        return False

def setup_coinbase():
    """Setup Coinbase service with connection validation"""
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    
    def _create_service():
        service = CoinbaseService(api_key, api_secret)
        # Test the connection with a simple API call
        try:
            # Try to get a small amount of candle data to validate connection
            test_response = service.client.get_public_candles(
                product_id="BTC-PERP-INTX",
                start=int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity="ONE_HOUR"
            )
            logger.info("✅ Coinbase connection validated successfully")
            return service
        except Exception as e:
            logger.error(f"❌ Failed to validate Coinbase connection: {e}")
            raise
    
    service = retry_with_backoff(_create_service)
    if service is None:
        raise ConnectionError("Failed to establish Coinbase connection after retries")
    return service


def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = 300, leverage: int = 20, side: str = "BUY", product: str = "BTC-PERP-INTX"):
    """
    General crypto trade execution function using trade_btc_perp.py with retry logic
    
    Args:
        cb_service: Coinbase service instance
        trade_type: Description of the trade type for logging
        entry_price: Entry price for logging
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        margin: USD amount to risk (default: 300)
        leverage: Leverage multiplier (default: 20)
        side: Trade side - "BUY" or "SELL" (default: "BUY")
        product: Trading product (default: "BTC-PERP-INTX")
    """
    def _execute_trade():
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        
        # Calculate position size based on margin and leverage
        position_size_usd = margin * leverage
        
        # Use subprocess to call trade_btc_perp.py
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', product,
            '--side', side,
            '--size', str(position_size_usd),
            '--leverage', str(leverage),
            '--tp', str(take_profit),
            '--sl', str(stop_loss),
            '--no-confirm'  # Skip confirmation for automated trading
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute the trade command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info(f"Crypto {trade_type} trade executed successfully!")
            logger.info(f"Trade output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"Crypto {trade_type} trade failed!")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
    
    try:
        # Use retry logic for trade execution
        result = retry_with_backoff(_execute_trade)
        if result is None:
            return False, "Failed after multiple retry attempts"
        return result
            
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error executing crypto {trade_type} trade: {e}")
        return False, str(e)


class BreakoutState:
    def __init__(self):
        self.state = "WATCH"  # WATCH, TRIGGERED, ENTER
        self.entry_price = None

    def to_dict(self):
        return {"state": self.state, "entry_price": self.entry_price}

    def from_dict(self, d):
        self.state = d.get("state", "WATCH")
        self.entry_price = d.get("entry_price")


def btc_4h_breakout_vol_spike_alert_stateful(cb_service, breakout_state, last_alert_ts=None):
    """
    BTC 4h breakout alert with 1-bar confirmation and state machine:
    - WATCH: If close >= 110,100 and vol spike, set TRIGGERED
    - TRIGGERED: On next bar, if close >= 110,000, enter trade; else reset to WATCH
    - ENTER: If close < 110,000, exit trade; if high >= 111,500, move stop to breakeven
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    GRANULARITY = "FOUR_HOUR"
    VOL_LOOKBACK = 20
    PRICE_LVL = 110000
    PRICE_BUFFER = 100
    VOL_FACTOR = 1.20
    BREAKEVEN_HIGH = 111500
    STOP_LOSS = 108600
    PROFIT_TARGET = 113000
    periods_needed = VOL_LOOKBACK + 3  # 20 for avg, 2 for current/prev, 1 for safety
    hours_needed = periods_needed * 4

    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())

        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough BTC {GRANULARITY} candle data for 4h breakout alert.")
            return last_alert_ts

        # Determine order: if first candle is newer than last, it's newest-first
        first_ts = int(candles[0]['start'])
        last_ts = int(candles[-1]['start'])
        if first_ts > last_ts:
            # Newest first: use candles[1] as just-closed, [2] as prev, [3:23] as history
            just_closed = candles[1]
            prev_candle = candles[2]
            historical_candles = candles[3:VOL_LOOKBACK+3]
        else:
            # Oldest first: use candles[-2] as just-closed, [-3] as prev, [-(VOL_LOOKBACK+3):-3] as history
            just_closed = candles[-2]
            prev_candle = candles[-3]
            historical_candles = candles[-(VOL_LOOKBACK+3):-3]

        ts = datetime.fromtimestamp(int(just_closed['start']), UTC)
        if ts == last_alert_ts:
            return last_alert_ts

        close = float(just_closed['close'])
        high = float(just_closed['high'])
        v0 = float(just_closed['volume'])
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
        close_ok = close >= PRICE_LVL + PRICE_BUFFER
        vol_ok = v0 >= VOL_FACTOR * avg20

        logger.info(f"=== BTC 4H BREAKOUT STATE: {breakout_state.state} ===")
        logger.info(f"Candle close: ${close:,.2f}, High: ${high:,.2f}, Volume: {v0:,.0f}, Avg(20): {avg20:,.0f}")
        logger.info(f"  - Close >= ${PRICE_LVL + PRICE_BUFFER:,.0f}: {'✅ Met' if close_ok else '❌ Not Met'}")
        logger.info(f"  - Volume >= {VOL_FACTOR}x avg: {'✅ Met' if vol_ok else '❌ Not Met'}")

        # State machine
        if breakout_state.state == "WATCH":
            if close_ok and vol_ok:
                breakout_state.state = "TRIGGERED"
                logger.info("State change: WATCH → TRIGGERED (waiting for confirmation)")
            # else: remain in WATCH
        elif breakout_state.state == "TRIGGERED":
            # On confirmation bar (just-closed):
            prev_close = float(prev_candle['close'])
            if prev_close >= PRICE_LVL:
                breakout_state.state = "ENTER"
                breakout_state.entry_price = prev_close
                logger.info(f"State change: TRIGGERED → ENTER (trade entry at ${prev_close:,.2f})")
                try:
                    play_alert_sound()
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="4h breakout 1-bar confirm",
                    entry_price=prev_close,
                    stop_loss=STOP_LOSS,
                    take_profit=PROFIT_TARGET,
                    margin=BTC_HORIZONTAL_MARGIN,
                    leverage=BTC_HORIZONTAL_LEVERAGE
                )
                if trade_success:
                    logger.info(f"BTC 4h breakout 1-bar confirm trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                else:
                    logger.error(f"BTC 4h breakout 1-bar confirm trade failed: {trade_result}")
            else:
                breakout_state.state = "WATCH"
                breakout_state.entry_price = None
                logger.info("State change: TRIGGERED → WATCH (breakout failed)")
        elif breakout_state.state == "ENTER":
            # Exit if failed follow-through
            if close < PRICE_LVL:
                logger.info(f"Exit: close < ${PRICE_LVL:,.0f} (breakout nullified)")
                try:
                    subprocess.run([sys.executable, 'close_positions.py'], check=True, timeout=60)
                    logger.info("Position closed via close_positions.py")
                except Exception as e:
                    logger.error(f"Failed to close position: {e}")
                breakout_state.state = "WATCH"
                breakout_state.entry_price = None
                logger.info("State change: ENTER → WATCH (position closed)")
            elif high >= BREAKEVEN_HIGH and breakout_state.entry_price:
                # Move stop to breakeven (entry price)
                logger.info(f"High >= ${BREAKEVEN_HIGH:,.0f}: Move stop to breakeven (${breakout_state.entry_price:,.2f})")
                try:
                    # Cancel all open orders for BTC-PERP-INTX
                    cb_service.cancel_all_orders(product_id=PRODUCT_ID)
                    logger.info("Cancelled all open orders for BTC-PERP-INTX before updating stop loss.")
                    # Get position size (assume full margin * leverage for now)
                    position_size_usd = BTC_HORIZONTAL_MARGIN * BTC_HORIZONTAL_LEVERAGE
                    # Place new bracket order with stop at entry price
                    bracket_result = cb_service.place_bracket_order(
                        product_id=PRODUCT_ID,
                        side="SELL",
                        size=position_size_usd,
                        entry_price=breakout_state.entry_price,
                        take_profit_price=PROFIT_TARGET,
                        stop_loss_price=breakout_state.entry_price
                    )
                    logger.info(f"Placed new bracket order with stop at breakeven: {bracket_result}")
                except Exception as e:
                    logger.error(f"Failed to update stop to breakeven: {e}")
        return ts
    except Exception as e:
        logger.error(f"Error in BTC 4h breakout vol spike alert stateful logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts




def main():
    global btc_continuation_trade_taken
    logger.info("Starting multi-asset alert script")
    logger.info("")  # Empty line for visual separation
    # Show trade status
    logger.info("✅ Ready to take BTC 4h breakout vol spike trades (4H, 1-bar confirm)")
    logger.info("")  # Empty line for visual separation
    # Check if alert sound file exists
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    logger.info("")  # Empty line for visual separation
    cb_service = setup_coinbase()
    btc_4h_breakout_last_alert_ts = None
    breakout_state = BreakoutState()
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            # Reset failure counter on successful iteration start
            iteration_start_time = time.time()
            
            # BTC 4h breakout vol spike alert (stateful)
            btc_4h_breakout_last_alert_ts = btc_4h_breakout_vol_spike_alert_stateful(cb_service, breakout_state, btc_4h_breakout_last_alert_ts)
            
            # Reset consecutive failures on successful completion
            consecutive_failures = 0
            
            # Wait 5 minutes until next poll
            wait_seconds = 300  # 5 minutes
            logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
            logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
            logger.info("")  # Empty line for visual separation
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            logger.info("👋 Stopped by user.")
            break
        except CONNECTION_ERRORS as e:
            consecutive_failures += 1
            logger.error(f"🔗 Connection error (failure {consecutive_failures}/{max_consecutive_failures}): {e}")
            
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"❌ Too many consecutive connection failures. Attempting to reconnect...")
                try:
                    # Try to reconnect
                    cb_service = setup_coinbase()
                    consecutive_failures = 0
                    logger.info("✅ Reconnection successful, resuming monitoring...")
                except Exception as reconnect_error:
                    logger.error(f"❌ Reconnection failed: {reconnect_error}")
                    logger.info("😴 Sleeping for 5 minutes before retry...")
                    time.sleep(300)
            else:
                # Exponential backoff for connection errors
                delay = exponential_backoff_delay(consecutive_failures - 1)
                logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"❌ Unexpected error in alert loop (failure {consecutive_failures}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # For non-connection errors, use a shorter delay
            delay = min(60 * consecutive_failures, 300)  # Max 5 minutes
            logger.info(f"😴 Sleeping for {delay} seconds before retry...")
            time.sleep(delay)

if __name__ == "__main__":
    main() 