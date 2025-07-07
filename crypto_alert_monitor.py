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


def get_btc_perp_position_size(cb_service):
    """
    Returns the open position size for BTC-PERP-INTX (absolute value, in base currency units).
    Returns 0.0 if no open position.
    Handles both dict and SDK object responses.
    """
    try:
        # Get the INTX portfolio UUID and breakdown
        ports = cb_service.client.get_portfolios()
        portfolio_uuid = None
        for p in ports['portfolios']:
            if p['type'] == "INTX":
                portfolio_uuid = p['uuid']
                break
        if not portfolio_uuid:
            logger.error("Could not find INTX portfolio")
            return 0.0
        portfolio = cb_service.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
        # Convert to dict if needed
        if not isinstance(portfolio, dict):
            if hasattr(portfolio, '__dict__'):
                portfolio = vars(portfolio)
            else:
                logger.error("Portfolio breakdown is not a dict and has no __dict__")
                return 0.0
        breakdown = portfolio.get('breakdown', {})
        # Convert breakdown to dict if needed
        if not isinstance(breakdown, dict):
            if hasattr(breakdown, '__dict__'):
                breakdown = vars(breakdown)
            else:
                logger.error("Breakdown is not a dict and has no __dict__")
                return 0.0
        positions = breakdown.get('perp_positions', [])
        for pos in positions:
            # Convert pos to dict if needed
            if not isinstance(pos, dict):
                if hasattr(pos, '__dict__'):
                    pos = vars(pos)
                else:
                    continue
            if pos.get('symbol') == "BTC-PERP-INTX":
                return abs(float(pos.get('net_size', 0)))
        return 0.0
    except Exception as e:
        logger.error(f"Error getting BTC-PERP-INTX position size: {e}")
        return 0.0


def btc_1h_rejection_wick_short_alert(cb_service, last_alert_ts=None):
    """
    Bearish 1h rejection wick short setup:
    - 1h high >= $108,800 and close < $108,000
    - Enter short at close
    - Stop-loss: $110,600
    - Take profit: $104,000
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    GRANULARITY = "ONE_HOUR"
    periods_needed = 2  # 1 for just-closed, 1 for safety
    hours_needed = periods_needed
    WICK_ZONE = 108800
    CLOSE_TRIGGER = 108000
    STOP_LOSS = 110600
    PROFIT_TARGET = 104000
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough BTC {GRANULARITY} candle data for 1h rejection wick short alert.")
            return last_alert_ts
        first_ts = int(candles[0]['start'])
        last_ts = int(candles[-1]['start'])
        if first_ts > last_ts:
            last_candle = candles[1]
        else:
            last_candle = candles[-2]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        if ts == last_alert_ts:
            return last_alert_ts
        high = float(last_candle['high'])
        close = float(last_candle['close'])
        wick_ok = high >= WICK_ZONE
        close_ok = close < CLOSE_TRIGGER
        logger.info(f"=== BTC 1H REJECTION WICK SHORT ===")
        logger.info(f"Candle high: ${high:,.2f}, close: ${close:,.2f}")
        logger.info(f"  - High >= ${WICK_ZONE:,.0f}: {'✅ Met' if wick_ok else '❌ Not Met'}")
        logger.info(f"  - Close < ${CLOSE_TRIGGER:,.0f}: {'✅ Met' if close_ok else '❌ Not Met'}")
        if wick_ok and close_ok:
            logger.info(f"--- BTC 1H REJECTION WICK SHORT ALERT ---")
            logger.info(f"Entry condition met: 1h high >= ${WICK_ZONE:,.0f} and close < ${CLOSE_TRIGGER:,.0f}.")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="1h rejection wick short",
                entry_price=close,
                stop_loss=STOP_LOSS,
                take_profit=PROFIT_TARGET,
                margin=BTC_HORIZONTAL_MARGIN,
                leverage=BTC_HORIZONTAL_LEVERAGE,
                side="SELL",
                product=PRODUCT_ID
            )
            if trade_success:
                logger.info(f"BTC 1h rejection wick short trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"BTC 1h rejection wick short trade failed: {trade_result}")
            return ts
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in BTC 1h rejection wick short alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts


def btc_4h_breakout_momentum_alert(cb_service, last_alert_ts=None):
    """
    BTC-USD breakout above $109K with momentum confirmation:
    - Entry trigger: 4-hr/daily close >109,600–110,000 on ≥20% above-average volume
    - Entry zone: 109,600–110,200
    - Stop-loss: 107,000
    - First profit target: 115,000
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    GRANULARITY = "FOUR_HOUR"
    VOLUME_PERIOD = 20
    periods_needed = VOLUME_PERIOD + 2
    hours_needed = periods_needed * 4
    ENTRY_TRIGGER_LOW = 109600
    ENTRY_TRIGGER_HIGH = 110000
    ENTRY_ZONE_LOW = 109600
    ENTRY_ZONE_HIGH = 110200
    STOP_LOSS = 107000
    PROFIT_TARGET = 115000
    VOLUME_MULTIPLIER = 1.20  # 20% above average
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
        first_ts = int(candles[0]['start'])
        last_ts = int(candles[-1]['start'])
        if first_ts > last_ts:
            last_candle = candles[1]
            historical_candles = candles[2:VOLUME_PERIOD+2]
        else:
            last_candle = candles[-2]
            historical_candles = candles[-(VOLUME_PERIOD+2):-2]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        if ts == last_alert_ts:
            return last_alert_ts
        close = float(last_candle['close'])
        v0 = float(last_candle['volume'])
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
        trigger_ok = ENTRY_TRIGGER_LOW < close <= ENTRY_TRIGGER_HIGH
        vol_ok = v0 >= VOLUME_MULTIPLIER * avg20
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        logger.info(f"=== BTC 4H BREAKOUT MOMENTUM ALERT ===")
        logger.info(f"Candle close: ${close:,.2f}, Volume: {v0:,.0f}, Avg(20): {avg20:,.0f}")
        logger.info(f"  - Close in trigger zone ${ENTRY_TRIGGER_LOW:,.0f}-${ENTRY_TRIGGER_HIGH:,.0f}: {'✅ Met' if trigger_ok else '❌ Not Met'}")
        logger.info(f"  - Volume ≥ 1.20x avg: {'✅ Met' if vol_ok else '❌ Not Met'}")
        logger.info(f"  - Close in entry zone ${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}: {'✅ Met' if in_entry_zone else '❌ Not Met'}")
        if trigger_ok and vol_ok and in_entry_zone:
            logger.info(f"--- BTC 4H BREAKOUT MOMENTUM TRADE ALERT ---")
            logger.info(f"Entry condition met: 4h close in ${ENTRY_TRIGGER_LOW:,.0f}-${ENTRY_TRIGGER_HIGH:,.0f}, vol spike, entry zone.")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="4h breakout momentum long",
                entry_price=close,
                stop_loss=STOP_LOSS,
                take_profit=PROFIT_TARGET,
                margin=BTC_HORIZONTAL_MARGIN,
                leverage=BTC_HORIZONTAL_LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            if trade_success:
                logger.info(f"BTC 4h breakout momentum trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"BTC 4h breakout momentum trade failed: {trade_result}")
            return ts
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in BTC 4h breakout momentum alert logic: {e}")
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
            iteration_start_time = time.time()
            # BTC 4h breakout momentum alert
            btc_4h_breakout_last_alert_ts = btc_4h_breakout_momentum_alert(cb_service, btc_4h_breakout_last_alert_ts)
            # BTC 1h rejection wick short alert
            btc_4h_breakout_last_alert_ts = btc_1h_rejection_wick_short_alert(cb_service, btc_4h_breakout_last_alert_ts)
            consecutive_failures = 0
            wait_seconds = 300
            logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
            logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
            logger.info("")
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
                    cb_service = setup_coinbase()
                    consecutive_failures = 0
                    logger.info("✅ Reconnection successful, resuming monitoring...")
                except Exception as reconnect_error:
                    logger.error(f"❌ Reconnection failed: {reconnect_error}")
                    logger.info("😴 Sleeping for 5 minutes before retry...")
                    time.sleep(300)
            else:
                delay = exponential_backoff_delay(consecutive_failures - 1)
                logger.info(f"🔄 Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        except Exception as e:
            consecutive_failures += 1
            logger.error(f"❌ Unexpected error in alert loop (failure {consecutive_failures}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            delay = min(60 * consecutive_failures, 300)
            logger.info(f"😴 Sleeping for {delay} seconds before retry...")
            time.sleep(delay)

if __name__ == "__main__":
    main()