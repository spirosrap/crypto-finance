import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
import pandas_ta as ta
import subprocess
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for get_recent_hourly_candles
GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"

# Trade parameters for BTC breakout
BTC_BREAKOUT_MARGIN = 300  # USD
BTC_BREAKOUT_LEVERAGE = 20  # 20x leverage
BTC_BREAKOUT_STOP_LOSS = 102000  # Stop-loss at $102,000
BTC_BREAKOUT_TAKE_PROFIT = 108000  # First profit target at $108,000

def setup_coinbase():
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    return CoinbaseService(api_key, api_secret)


def get_recent_hourly_candles(cb_service, num_candles=24):
    now = datetime.now(UTC)
    now = now.replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=num_candles)
    end = now
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    logger.info(f"Fetching last {num_candles} hourly candles from {start} to {end}")
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
    return candles


def btc_triangle_breakout_alert(cb_service, last_alert_ts=None):
    """
    Alerts on an intraday triangle breakout for BTC.
    Entry trigger: 1-hour close > 104,200 with >=2x 20-period volume.
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    ENTRY_PRICE_THRESHOLD = 104200
    VOLUME_PERIOD = 20
    VOLUME_MULTIPLIER = 2.0
    ENTRY_ZONE_LOW = 104200
    ENTRY_ZONE_HIGH = 104800

    try:
        # 1. Get candles for analysis (volume period + 2 for current and last closed)
        candles_raw = get_recent_hourly_candles(cb_service, num_candles=VOLUME_PERIOD + 2)
        if not candles_raw or len(candles_raw) < VOLUME_PERIOD + 2:
            logger.warning(f"Not enough BTC hourly candle data for triangle breakout alert. Need {VOLUME_PERIOD + 2}, got {len(candles_raw)}.")
            return last_alert_ts

        # 2. Prepare data for analysis
        last_closed_candle_raw = candles_raw[1]
        ts = datetime.fromtimestamp(int(last_closed_candle_raw['start']), UTC)

        # Avoid re-alerting for the same candle
        if ts == last_alert_ts:
            return last_alert_ts
        
        # Historical candles for volume average (20 periods before the last closed candle)
        historical_candles = candles_raw[2:VOLUME_PERIOD + 2]
        if len(historical_candles) < VOLUME_PERIOD:
            logger.warning(f"Not enough historical BTC hourly candle data for volume average. Need {VOLUME_PERIOD}, got {len(historical_candles)}.")
            return last_alert_ts

        volumes = [float(c['volume']) for c in historical_candles]
        avg_volume = sum(volumes) / len(volumes)

        last_close = float(last_closed_candle_raw['close'])
        last_volume = float(last_closed_candle_raw['volume'])

        logger.info(f"BTC Triangle Breakout Check: Last Close=${last_close:,.2f}, Last Volume={last_volume:,.0f}, Avg Volume({VOLUME_PERIOD})={avg_volume:,.0f}")

        # 4. Check alert conditions
        is_breakout_price = last_close > ENTRY_PRICE_THRESHOLD
        is_high_volume = last_volume >= (avg_volume * VOLUME_MULTIPLIER)
        
        if is_breakout_price and is_high_volume:
            logger.info(f"--- BTC TRIANGLE BREAKOUT ALERT ---")
            logger.info(f"Entry condition met: 1-hour close > ${ENTRY_PRICE_THRESHOLD:,.0f} with volume >= {VOLUME_MULTIPLIER}x 20-period average.")
            
            if ENTRY_ZONE_LOW <= last_close <= ENTRY_ZONE_HIGH:
                logger.info(f"Price ${last_close:,.2f} is within designated entry zone (${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}).")
            else:
                logger.warning(f"Price ${last_close:,.2f} is outside designated entry zone (${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}).")

            logger.info(f"Details: Timestamp={ts}, Close=${last_close:,.2f}, Volume={last_volume:,.0f}, Avg Volume={avg_volume:,.0f}")

            # Execute the trade
            logger.info("Executing BTC breakout trade...")
            breakout_type = f"triangle_breakout_{ENTRY_PRICE_THRESHOLD}"
            trade_success, trade_result = execute_btc_breakout_trade(cb_service, breakout_type, last_close)

            if trade_success:
                logger.info("BTC breakout trade executed successfully!")
                logger.info(f"Trade parameters: Margin=${BTC_BREAKOUT_MARGIN}, Leverage={BTC_BREAKOUT_LEVERAGE}x")
                logger.info(f"Stop Loss: ${BTC_BREAKOUT_STOP_LOSS:,.0f}, Take Profit: ${BTC_BREAKOUT_TAKE_PROFIT:,.0f}")
            else:
                logger.error(f"BTC breakout trade failed: {trade_result}")

            logger.info("")
            return ts

    except Exception as e:
        logger.error(f"Error in BTC triangle breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return last_alert_ts


def fartcoin_daily_alert(cb_service, last_alert_ts=None):
    PRODUCT_ID = "FARTCOIN-PERP-INTX"
    GRANULARITY = "ONE_DAY"
    BREAKOUT_THRESHOLD = 1.10
    BREAKOUT_VOLUME_MULTIPLIER = 1.2  # 20% above average

    try:
        now = datetime.now(UTC)
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=21)
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
        if not candles or len(candles) < 2:
            logger.warning("Not enough FARTCOIN daily candle data.")
            return last_alert_ts
        
        last_candle = candles[1]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)

        if ts == last_alert_ts:
            return last_alert_ts

        session_candles = candles[1:21]
        session_volumes = [float(c['volume']) for c in session_candles]
        avg_volume = sum(session_volumes) / 20 if len(session_volumes) == 20 else sum(session_volumes) / len(session_volumes)
        close = float(last_candle['close'])
        volume = float(last_candle['volume'])
        
        logger.info(f"FARTCOIN Check: Last Daily Close=${close:.5f}, Volume={volume:,.0f}, Avg Volume={avg_volume:,.0f}")
        
        if close > BREAKOUT_THRESHOLD and volume >= BREAKOUT_VOLUME_MULTIPLIER * avg_volume:
            logger.info(f"--- FARTCOIN ALERT ---")
            logger.info(f"Daily close > {BREAKOUT_THRESHOLD} and volume ≥ 20% above 20-day average!")
            logger.info(f"Timestamp: {ts}, Close: {close}, Volume: {volume:,.0f}, Avg Volume: {avg_volume:,.0f}")
            logger.info("")  # Empty line for visual separation
            return ts

    except Exception as e:
        logger.error(f"Error in FARTCOIN alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return last_alert_ts


def execute_btc_breakout_trade(cb_service, breakout_type: str, entry_price: float):
    """
    Execute BTC breakout trade using trade_btc_perp.py functionality
    """
    try:
        logger.info(f"Executing BTC breakout trade: {breakout_type} at ${entry_price:,.2f}")
        
        # Calculate position size based on margin and leverage
        position_size_usd = BTC_BREAKOUT_MARGIN * BTC_BREAKOUT_LEVERAGE
        
        # Use subprocess to call trade_btc_perp.py
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', 'BTC-PERP-INTX',
            '--side', 'BUY',
            '--size', str(position_size_usd),
            '--leverage', str(BTC_BREAKOUT_LEVERAGE),
            '--tp', str(BTC_BREAKOUT_TAKE_PROFIT),
            '--sl', str(BTC_BREAKOUT_STOP_LOSS),
            '--no-confirm'  # Skip confirmation for automated trading
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute the trade command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("BTC breakout trade executed successfully!")
            logger.info(f"Trade output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"BTC breakout trade failed!")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error executing BTC breakout trade: {e}")
        return False, str(e)


def main():
    logger.info("Starting multi-asset alert script")
    logger.info("")  # Empty line for visual separation
    cb_service = setup_coinbase()
    btc_triangle_breakout_last_alert_ts = None
    fartcoin_last_alert_ts = None
    
    while True:
        try:
            # BTC triangle breakout alert
            btc_triangle_breakout_last_alert_ts = btc_triangle_breakout_alert(cb_service, btc_triangle_breakout_last_alert_ts)

            # FARTCOIN daily alert (runs hourly but condition only changes daily)
            fartcoin_last_alert_ts = fartcoin_daily_alert(cb_service, fartcoin_last_alert_ts)

            # Wait until next hour + 1 minute
            now = datetime.now(UTC)
            next_poll = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
            wait_seconds = (next_poll - now).total_seconds()
            logger.info(f"Waiting {wait_seconds:.0f} seconds until next poll")
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