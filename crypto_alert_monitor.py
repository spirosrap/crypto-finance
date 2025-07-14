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
import concurrent.futures

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


# Trade tracking
btc_continuation_trade_taken = False

TRIGGER_STATE_FILE = "btc_breakout_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None}
    return {"triggered": False, "trigger_ts": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

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


def btc_custom_breakout_alert(cb_service, last_alert_ts=None):
    PRODUCT_ID = "BTC-PERP-INTX"
    GRANULARITY = "ONE_HOUR"
    ENTRY_ZONE_LOW = 120000
    ENTRY_ZONE_HIGH = 121500
    STOP_LOSS = 117500
    PROFIT_TARGET = 125000
    MARGIN = 300
    LEVERAGE = 20
    VOLUME_PERIOD = 20
    periods_needed = VOLUME_PERIOD + 2
    
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=periods_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough BTC {GRANULARITY} candle data for custom breakout alert.")
            return last_alert_ts
        # Use the most recent closed candle
        last_candle = candles[-2]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        close = float(last_candle['close'])
        high = float(last_candle['high'])
        low = float(last_candle['low'])
        v0 = float(last_candle['volume'])
        historical_candles = candles[-(VOLUME_PERIOD+2):-2]
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
        
        # --- Reporting ---
        logger.info("")
        logger.info(f"Entry zone: ${ENTRY_ZONE_LOW:,} – ${ENTRY_ZONE_HIGH:,} (pull-back to prior ceiling)")
        logger.info(f"Stop-loss: ${STOP_LOSS:,} (below today’s intraday low)")
        logger.info(f"1st target: ${PROFIT_TARGET:,} (psychological + flag projection)")
        logger.info(f"Facts: fresh all-time high above $120K on heavy flows from ETFs/corporate treasuries. [markets.businessinsider.com] [investing.com]")
        logger.info(f"Opinion: momentum intact; buy dips only—chase breakouts as last resort.")
        logger.info("")
        logger.info(f"Candle close: ${close:,.2f}, High: ${high:,.2f}, Low: ${low:,.2f}, Volume: {v0:,.0f}, Avg(20): {avg20:,.0f}")
        # --- Entry logic ---
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        if in_entry_zone and not trigger_state.get("triggered", False):
            logger.info(f"Entry condition met: close (${close:,.2f}) is within entry zone (${ENTRY_ZONE_LOW:,}-${ENTRY_ZONE_HIGH:,}). Taking trade.")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="BTC-USD custom breakout entry",
                entry_price=close,
                stop_loss=STOP_LOSS,
                take_profit=PROFIT_TARGET,
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            if trade_success:
                logger.info(f"BTC-USD custom breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"BTC-USD custom breakout trade failed: {trade_result}")
            # Set trigger to avoid duplicate trades
            trigger_state = {"triggered": True, "trigger_ts": int(last_candle['start'])}
            save_trigger_state(trigger_state)
            return ts
        # Reset trigger if price leaves entry zone
        if trigger_state.get("triggered", False):
            if not in_entry_zone:
                trigger_state = {"triggered": False, "trigger_ts": None}
                save_trigger_state(trigger_state)
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in BTC custom breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts

# Remove old alert functions
def main():
    logger.info("Starting custom BTC breakout alert script")
    logger.info("")
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    logger.info("")
    cb_service = setup_coinbase()
    last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    def poll_iteration():
        nonlocal last_alert_ts, consecutive_failures
        iteration_start_time = time.time()
        last_alert_ts = btc_custom_breakout_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        wait_seconds = 300
        logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
        logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
        logger.info("")
        time.sleep(wait_seconds)
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)  # 2 minute max per poll
                except concurrent.futures.TimeoutError:
                    logger.error('Polling iteration timed out! Skipping to next.')
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