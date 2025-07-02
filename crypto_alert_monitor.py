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




# Convenience wrapper for ETH EMA cluster breakout with default parameters
def execute_eth_ema_breakout_trade(cb_service, breakout_type: str, entry_price: float, stop_loss: float, take_profit: float):
    """
    Execute ETH EMA cluster breakout trade using default parameters
    """
    return execute_crypto_trade(
        cb_service=cb_service,
        trade_type=f"EMA cluster breakout ({breakout_type})",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        margin=ETH_EMA_MARGIN,
        leverage=ETH_EMA_LEVERAGE,
        product="ETH-PERP-INTX"
    )


def btc_descending_triangle_breakout_alert(cb_service, last_alert_ts=None):
    """
    BTC-USD descending triangle breakout alert
    Entry trigger: Close above 108,000-108,500 on ≥20% volume surge
    Entry zone: 108,000-108,500
    Stop-loss: 106,000 (below triangle support base)
    First profit target: 112,000-114,000 (triangle measured move / supply zone)
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    GRANULARITY = "ONE_HOUR"  # Using hourly candles for more responsive detection
    periods_needed = 24 + 2  # 24 periods for volume baseline + 2 for analysis
    hours_needed = periods_needed  # Hourly candles
    
    ENTRY_TRIGGER_LOW = 108000
    ENTRY_TRIGGER_HIGH = 108500
    ENTRY_ZONE_LOW = 108000
    ENTRY_ZONE_HIGH = 108500
    STOP_LOSS = 106000  # Below triangle support base
    PROFIT_TARGET_LOW = 112000  # Conservative target
    PROFIT_TARGET_HIGH = 114000  # Measured move target
    VOLUME_PERIOD = 24  # 24-hour volume baseline
    VOLUME_MULTIPLIER = 1.2  # ≥20% volume surge
    
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles:
            logger.warning(f"Failed to fetch BTC {GRANULARITY} candle data for descending triangle analysis.")
            return last_alert_ts
        if len(candles) < periods_needed:
            logger.warning(f"Not enough BTC {GRANULARITY} candle data for descending triangle analysis.")
            return last_alert_ts
        
        # Determine order: if first candle is newer than last, it's newest-first
        first_ts = int(candles[0]['start'])
        last_ts = int(candles[-1]['start'])
        if first_ts > last_ts:
            # Newest first: use candles[1] as most recent completed
            last_candle = candles[1]
        else:
            # Oldest first: use candles[-2]
            last_candle = candles[-2]
        
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        if ts == last_alert_ts:
            return last_alert_ts
        
        close = float(last_candle['close'])
        current_volume = float(last_candle['volume'])
        
        # Historical candles for volume baseline (24 periods before the last closed candle)
        if first_ts > last_ts:
            historical_candles = candles[2:VOLUME_PERIOD+2]
        else:
            historical_candles = candles[-(VOLUME_PERIOD+2):-2]
        volumes = [float(c['volume']) for c in historical_candles]
        baseline_volume = sum(volumes) / len(volumes)
        
        # Calculate RSI using all available candles
        if len(candles) >= 16:  # 14 for RSI + 2 for analysis
            # Create DataFrame for RSI calculation
            df_data = []
            if first_ts > last_ts:
                # Newest first - reverse for chronological order
                rsi_candles = candles[::-1]
            else:
                # Already in chronological order
                rsi_candles = candles
            
            for candle in rsi_candles:
                df_data.append({
                    'close': float(candle['close']),
                    'timestamp': int(candle['start'])
                })
            
            df = pd.DataFrame(df_data)
            df['rsi'] = ta.rsi(df['close'], length=14)
            current_rsi = df['rsi'].iloc[-2] if len(df) >= 2 else None  # Most recent completed candle
        else:
            current_rsi = None
        
        # Alert logic
        is_breakout_price = close >= ENTRY_TRIGGER_LOW and close <= ENTRY_TRIGGER_HIGH
        is_volume_surge = current_volume >= (baseline_volume * VOLUME_MULTIPLIER)
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        
        logger.info(f"=== BTC DESCENDING TRIANGLE BREAKOUT (HOURLY) ===")
        logger.info(f"Candle close: ${close:,.2f}, Volume: {current_volume:,.0f}, Baseline Vol({VOLUME_PERIOD}h): {baseline_volume:,.0f}")
        rsi_text = f"RSI: {current_rsi:.1f}" if current_rsi is not None else "RSI: N/A"
        logger.info(f"  - {rsi_text}")
        logger.info(f"  - Close in breakout range ${ENTRY_TRIGGER_LOW:,.0f}-${ENTRY_TRIGGER_HIGH:,.0f}: {'✅ Met' if is_breakout_price else '❌ Not Met'}")
        logger.info(f"  - Volume ≥ {VOLUME_MULTIPLIER}x Baseline: {'✅ Met' if is_volume_surge else '❌ Not Met'}")
        logger.info(f"  - Entry zone ${ENTRY_ZONE_LOW:,.0f}-{ENTRY_ZONE_HIGH:,.0f}: {'✅ Met' if in_entry_zone else '❌ Not Met'}")
        logger.info(f"  - Target range: ${PROFIT_TARGET_LOW:,.0f}-${PROFIT_TARGET_HIGH:,.0f} (triangle measured move)")
        
        if is_breakout_price and is_volume_surge and in_entry_zone:
            logger.info(f"--- BTC DESCENDING TRIANGLE BREAKOUT ALERT ---")
            logger.info(f"Entry condition met: Close above ${ENTRY_TRIGGER_LOW:,.0f}-${ENTRY_TRIGGER_HIGH:,.0f} with ≥ {VOLUME_MULTIPLIER}x volume surge.")
            rsi_alert_text = f" (RSI: {current_rsi:.1f})" if current_rsi is not None else " (RSI: N/A)"
            logger.info(f"Stop-loss: ${STOP_LOSS:,.0f}, Target: ${PROFIT_TARGET_LOW:,.0f}-${PROFIT_TARGET_HIGH:,.0f}{rsi_alert_text}")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            
            # Execute the trade using the mid-point of the target range
            profit_target = (PROFIT_TARGET_LOW + PROFIT_TARGET_HIGH) / 2  # Use mid-point: $113,000
            breakout_type = f"descending_triangle_{ENTRY_TRIGGER_LOW}_{ENTRY_TRIGGER_HIGH}"
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type=f"descending triangle breakout ({breakout_type})",
                entry_price=close,
                stop_loss=STOP_LOSS,
                take_profit=profit_target,
                margin=BTC_HORIZONTAL_MARGIN,
                leverage=BTC_HORIZONTAL_LEVERAGE
            )
            if trade_success:
                logger.info(f"BTC descending triangle breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"BTC descending triangle breakout trade failed: {trade_result}")
            return ts
            
    except Exception as e:
        logger.error(f"Error in BTC descending triangle breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts


def eth_ema_cluster_breakout_alert(cb_service, last_alert_ts=None):
    """
    ETH-USD EMA cluster breakout alert (daily candle)
    Entry trigger: Close > 2600 daily with confirmed volume spike
    Entry zone: 2600–2650
    Stop-loss: 2500
    First profit target: 3000
    """
    PRODUCT_ID = "ETH-PERP-INTX"
    GRANULARITY = "ONE_DAY"
    periods_needed = 20 + 2  # 20 periods for volume average + 2 for analysis
    hours_needed = periods_needed * 24  # Daily candles
    ENTRY_TRIGGER = 2600
    ENTRY_ZONE_LOW = 2600
    ENTRY_ZONE_HIGH = 2650
    STOP_LOSS = 2500
    PROFIT_TARGET = 3000
    VOLUME_PERIOD = 20
    VOLUME_MULTIPLIER = 1.2  # Confirmed volume spike (≥20% above average)
    
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles:
            logger.warning(f"Failed to fetch ETH {GRANULARITY} candle data for analysis.")
            return last_alert_ts
        if len(candles) < periods_needed:
            logger.warning(f"Not enough ETH {GRANULARITY} candle data for analysis.")
            return last_alert_ts
        
        # Determine order: if first candle is newer than last, it's newest-first
        first_ts = int(candles[0]['start'])
        last_ts = int(candles[-1]['start'])
        if first_ts > last_ts:
            # Newest first: use candles[1] as most recent completed
            last_candle = candles[1]
        else:
            # Oldest first: use candles[-2]
            last_candle = candles[-2]
        
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        if ts == last_alert_ts:
            return last_alert_ts
        
        close = float(last_candle['close'])
        current_volume = float(last_candle['volume'])
        
        # Historical candles for volume average (20 periods before the last closed candle)
        if first_ts > last_ts:
            historical_candles = candles[2:VOLUME_PERIOD+2]
        else:
            historical_candles = candles[-(VOLUME_PERIOD+2):-2]
        volumes = [float(c['volume']) for c in historical_candles]
        avg_volume = sum(volumes) / len(volumes)
        
        # Alert logic
        is_breakout_price = close > ENTRY_TRIGGER
        is_high_volume = current_volume >= (avg_volume * VOLUME_MULTIPLIER)
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        
        logger.info(f"=== ETH EMA CLUSTER BREAKOUT (DAILY) ===")
        logger.info(f"Candle close: ${close:,.2f}, Volume: {current_volume:,.0f}, Avg Vol({VOLUME_PERIOD}): {avg_volume:,.0f}")
        logger.info(f"  - Close > ${ENTRY_TRIGGER:,.0f}: {'✅ Met' if is_breakout_price else '❌ Not Met'}")
        logger.info(f"  - Volume ≥ {VOLUME_MULTIPLIER}x Avg: {'✅ Met' if is_high_volume else '❌ Not Met'}")
        logger.info(f"  - Entry zone ${ENTRY_ZONE_LOW}-{ENTRY_ZONE_HIGH}: {'✅ Met' if in_entry_zone else '❌ Not Met'}")
        
        if is_breakout_price and is_high_volume and in_entry_zone:
            logger.info(f"--- ETH EMA CLUSTER BREAKOUT ALERT (DAILY) ---")
            logger.info(f"Entry condition met: Daily close > ${ENTRY_TRIGGER:,.0f} with confirmed volume spike ≥ {VOLUME_MULTIPLIER}x 20-period avg.")
            logger.info(f"Stop-loss: ${STOP_LOSS:,.0f}, First profit target: ${PROFIT_TARGET:,.0f}")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            
            # Execute the trade
            breakout_type = f"ema_cluster_{ENTRY_TRIGGER}"
            trade_success, trade_result = execute_eth_ema_breakout_trade(cb_service, breakout_type, close, STOP_LOSS, PROFIT_TARGET)
            if trade_success:
                logger.info(f"ETH EMA cluster breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"ETH EMA cluster breakout trade failed: {trade_result}")
            return ts
            
    except Exception as e:
        logger.error(f"Error in ETH EMA cluster breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts


def main():
    global btc_continuation_trade_taken
    logger.info("Starting multi-asset alert script")
    logger.info("")  # Empty line for visual separation
    # Show trade status
    logger.info("✅ Ready to take BTC descending triangle breakout trades (1H)")
    logger.info("✅ Ready to take ETH EMA cluster breakout trades (1D)")
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
    btc_horizontal_last_alert_ts_4h = None
    btc_horizontal_last_alert_ts_1d = None
    btc_triangle_last_alert_ts = None
    eth_ema_last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        try:
            # Reset failure counter on successful iteration start
            iteration_start_time = time.time()
            
            # BTC descending triangle breakout alert (1h)
            btc_triangle_last_alert_ts = btc_descending_triangle_breakout_alert(cb_service, btc_triangle_last_alert_ts)
            # ETH EMA cluster breakout alert (1d)
            # eth_ema_last_alert_ts = eth_ema_cluster_breakout_alert(cb_service, eth_ema_last_alert_ts)
            
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