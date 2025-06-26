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

# Constants for get_recent_hourly_candles
GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"


# Trade parameters for BTC mid-range breakout
BTC_MIDRANGE_MARGIN = 300  # USD
BTC_MIDRANGE_LEVERAGE = 20  # 15x leverage
BTC_MIDRANGE_STOP_LOSS = 105500  # Stop-loss at $105,500 (recent swing low structure)
BTC_MIDRANGE_TAKE_PROFIT = 112000  # First profit target at $112,000 (prior ATH cluster)

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
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    return CoinbaseService(api_key, api_secret)


def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = 300, leverage: int = 20, side: str = "BUY", product: str = "BTC-PERP-INTX"):
    """
    General crypto trade execution function using trade_btc_perp.py
    
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
    try:
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
            
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error executing crypto {trade_type} trade: {e}")
        return False, str(e)


# Convenience wrapper for mid-range breakout with default parameters
def execute_btc_midrange_breakout_trade(cb_service, breakout_type: str, entry_price: float, stop_loss: float, take_profit: float):
    """
    Execute BTC mid-range breakout trade using default parameters
    """
    return execute_crypto_trade(
        cb_service=cb_service,
        trade_type=f"mid-range breakout ({breakout_type})",
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        margin=BTC_MIDRANGE_MARGIN,
        leverage=BTC_MIDRANGE_LEVERAGE
    )


def btc_breakout_midrange_alert(cb_service, last_alert_ts=None, timeframe='4h'):
    """
    BTC-USD breakout above mid-range resistance alert (4h or daily candle)
    Entry trigger: 4-hr or daily close above 107,500 with volume ≥20% above 20-period avg
    Entry zone: 107,500–108,200
    Stop-loss: 105,500
    First profit target: 112,000
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    if timeframe == '4h':
        GRANULARITY = "FOUR_HOUR"
        periods_needed = 20 + 2
        hours_needed = periods_needed * 4
    elif timeframe == '1d':
        GRANULARITY = "ONE_DAY"
        periods_needed = 20 + 2
        hours_needed = periods_needed * 24
    else:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return last_alert_ts
    ENTRY_TRIGGER = 107500
    ENTRY_ZONE_LOW = 107500
    ENTRY_ZONE_HIGH = 108200
    STOP_LOSS = 105500
    PROFIT_TARGET = 112000
    VOLUME_PERIOD = 20
    VOLUME_MULTIPLIER = 1.2
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=start_ts,
            end=end_ts,
            granularity=GRANULARITY
        )
        if hasattr(response, 'candles'):
            candles = response.candles
        else:
            candles = response.get('candles', [])
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough BTC {GRANULARITY} candle data for {timeframe} analysis.")
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
        logger.info(f"=== BTC MID-RANGE BREAKOUT ({timeframe.upper()}) ===")
        logger.info(f"Candle close: ${close:,.2f}, Volume: {current_volume:,.0f}, Avg Vol({VOLUME_PERIOD}): {avg_volume:,.0f}")
        logger.info(f"  - Close > ${ENTRY_TRIGGER:,.0f}: {'✅ Met' if is_breakout_price else '❌ Not Met'}")
        logger.info(f"  - Volume ≥ {VOLUME_MULTIPLIER}x Avg: {'✅ Met' if is_high_volume else '❌ Not Met'}")
        logger.info(f"  - Entry zone ${ENTRY_ZONE_LOW}-{ENTRY_ZONE_HIGH}: {'✅ Met' if in_entry_zone else '❌ Not Met'}")
        if is_breakout_price and is_high_volume and in_entry_zone:
            logger.info(f"--- BTC MID-RANGE BREAKOUT ALERT ({timeframe.upper()}) ---")
            logger.info(f"Entry condition met: {timeframe} close > ${ENTRY_TRIGGER:,.0f} with volume ≥ {VOLUME_MULTIPLIER}x 20-period avg.")
            logger.info(f"Stop-loss: ${STOP_LOSS:,.0f}, First profit target: ${PROFIT_TARGET:,.0f}")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            # Execute the trade
            breakout_type = f"midrange_breakout_{ENTRY_TRIGGER}_{timeframe}"
            trade_success, trade_result = execute_btc_midrange_breakout_trade(cb_service, breakout_type, close, STOP_LOSS, PROFIT_TARGET)
            if trade_success:
                logger.info(f"BTC {timeframe.upper()} mid-range breakout trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"BTC {timeframe.upper()} mid-range breakout trade failed: {trade_result}")
            return ts
    except Exception as e:
        logger.error(f"Error in BTC mid-range breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts


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
        
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=start_ts,
            end=end_ts,
            granularity=GRANULARITY
        )
        if hasattr(response, 'candles'):
            candles = response.candles
        else:
            candles = response.get('candles', [])
        if not candles or len(candles) < periods_needed:
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
    logger.info("✅ Ready to take BTC mid-range breakout trades (4H/1D)")
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
    btc_midrange_last_alert_ts_4h = None
    btc_midrange_last_alert_ts_1d = None
    eth_ema_last_alert_ts = None
    while True:
        try:
            # BTC mid-range breakout alert (4h)
            btc_midrange_last_alert_ts_4h = btc_breakout_midrange_alert(cb_service, btc_midrange_last_alert_ts_4h, timeframe='4h')
            # BTC mid-range breakout alert (1d)
            btc_midrange_last_alert_ts_1d = btc_breakout_midrange_alert(cb_service, btc_midrange_last_alert_ts_1d, timeframe='1d')
            # ETH EMA cluster breakout alert (1d)
            eth_ema_last_alert_ts = eth_ema_cluster_breakout_alert(cb_service, eth_ema_last_alert_ts)
            # Wait 5 minutes until next poll
            wait_seconds = 300  # 5 minutes
            logger.info(f"Waiting {wait_seconds} seconds until next poll")
            logger.info("")  # Empty line for visual separation
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in alert loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(60)

if __name__ == "__main__":
    main() 