import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
import pandas_ta as ta

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for get_recent_hourly_candles
GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"

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


def process_candle(candle):
    return {
        'timestamp': datetime.fromtimestamp(int(candle['start']), UTC),
        'open': float(candle['open']),
        'high': float(candle['high']),
        'low': float(candle['low']),
        'close': float(candle['close']),
        'volume': float(candle['volume'])
    }

def btc_intraday_rsi_alert(cb_service, last_alert_ts=None):
    """
    Alerts when BTC price is near the intraday low with an oversold RSI.
    Entry: ~$104,000 (near current intraday low with RSI oversold conditions)
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    TARGET_PRICE = 104000
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    INTRADAY_LOW_PROXIMITY_PERCENT = 0.5  # Within 0.5% of intraday low

    try:
        # 1. Get candles for analysis (RSI period + 24h for intraday low)
        candles_raw = get_recent_hourly_candles(cb_service, num_candles=RSI_PERIOD + 24)
        if not candles_raw or len(candles_raw) < RSI_PERIOD + 1:
            logger.warning("Not enough BTC hourly candle data for RSI alert.")
            return last_alert_ts

        # 2. Prepare data for analysis
        last_closed_candle_raw = candles_raw[1]
        ts = datetime.fromtimestamp(int(last_closed_candle_raw['start']), UTC)

        # Avoid re-alerting for the same candle
        if ts == last_alert_ts:
            return last_alert_ts
        
        # Reverse historical candles for correct indicator calculation order
        historical_candles = candles_raw[1:]
        historical_candles.reverse()
        
        # Convert list of Candle objects to list of dicts for DataFrame compatibility
        historical_candles_dicts = [
            {'start': c['start'], 'open': c['open'], 'high': c['high'], 'low': c['low'], 'close': c['close'], 'volume': c['volume']} 
            for c in historical_candles
        ]

        df = pd.DataFrame(historical_candles_dicts)
        df['close'] = pd.to_numeric(df['close'])
        df['low'] = pd.to_numeric(df['low'])

        # 3. Calculate indicators and intraday low
        df.ta.rsi(length=RSI_PERIOD, append=True)
        
        latest_stats = df.iloc[-1]
        last_close = latest_stats['close']
        last_rsi = latest_stats[f'RSI_{RSI_PERIOD}']
        
        # Intraday low is the min of the last 24 closed candles
        intraday_low = df['low'][-24:].min()
        
        logger.info(f"BTC Check: Last Close=${last_close:,.2f}, Intraday Low=${intraday_low:,.2f}, RSI({RSI_PERIOD})={last_rsi:.2f}")

        # 4. Check alert conditions
        is_near_target = last_close <= TARGET_PRICE
        is_near_low = last_close <= (intraday_low * (1 + INTRADAY_LOW_PROXIMITY_PERCENT / 100))
        is_rsi_oversold = last_rsi < RSI_OVERSOLD
        
        if is_near_target and is_near_low and is_rsi_oversold:
            logger.info(f"--- BTC ALERT ---")
            logger.info(f"Entry condition met: Price near ${TARGET_PRICE:,.0f}, near intraday low, and RSI oversold.")
            logger.info(f"Details: Timestamp={ts}, Close=${last_close:,.2f}, Intraday Low=${intraday_low:,.2f}, RSI={last_rsi:.2f}")
            return ts  # Update last alert timestamp

    except Exception as e:
        logger.error(f"Error in BTC alert logic: {e}")
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
            return ts

    except Exception as e:
        logger.error(f"Error in FARTCOIN alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return last_alert_ts


def main():
    logger.info("Starting multi-asset alert script")
    cb_service = setup_coinbase()
    btc_last_alert_ts = None
    fartcoin_last_alert_ts = None
    
    while True:
        try:
            # BTC hourly alert logic
            btc_last_alert_ts = btc_intraday_rsi_alert(cb_service, btc_last_alert_ts)

            # FARTCOIN daily alert (runs hourly but condition only changes daily)
            fartcoin_last_alert_ts = fartcoin_daily_alert(cb_service, fartcoin_last_alert_ts)

            # Wait until next hour + 1 minute
            now = datetime.now(UTC)
            next_poll = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
            wait_seconds = (next_poll - now).total_seconds()
            logger.info(f"Waiting {wait_seconds:.0f} seconds until next poll")
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