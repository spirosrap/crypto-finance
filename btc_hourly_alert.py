import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"
BREAKOUT_THRESHOLD = 105500
BREAKDOWN_THRESHOLD = 104000
BREAKOUT_VOLUME_MULTIPLIER = 1.0  # Can be set higher for extra confirmation
BREAKDOWN_VOLUME_MULTIPLIER = 1.2  # 20% above average, adjust per risk tolerance


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


def fartcoin_daily_alert(cb_service):
    PRODUCT_ID = "FARTCOIN-PERP-INTX"
    GRANULARITY = "ONE_DAY"
    BREAKOUT_THRESHOLD = 1.10
    BREAKOUT_VOLUME_MULTIPLIER = 1.2  # 20% above average

    now = datetime.now(UTC)
    now = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = now - timedelta(days=21)
    end = now
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    logger.info(f"Fetching last 21 daily candles for FARTCOIN from {start} to {end}")
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
        return
    # Assume most recent is ongoing, so use candles[1] as last closed
    last_candle = candles[1]
    session_candles = candles[1:21]  # last 20 closed daily candles
    session_volumes = [float(c['volume']) for c in session_candles]
    avg_volume = sum(session_volumes) / 20 if len(session_volumes) == 20 else sum(session_volumes) / len(session_volumes)
    close = float(last_candle['close'])
    volume = float(last_candle['volume'])
    ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
    logger.info(f"FARTCOIN last daily close: {close}, volume: {volume}, avg_volume: {avg_volume}")
    if close > BREAKOUT_THRESHOLD and volume >= BREAKOUT_VOLUME_MULTIPLIER * avg_volume:
        logger.info(f"FARTCOIN ALERT: Daily close > {BREAKOUT_THRESHOLD} and volume ≥ 20% above 20-day average!")
        logger.info(f"Timestamp: {ts}, Close: {close}, Volume: {volume}, Avg Volume: {avg_volume}")


def main():
    logger.info("Starting BTC hourly close alert script")
    logger.info("Monitoring for hourly close breakout/breakdown and FARTCOIN daily breakout")
    cb_service = setup_coinbase()
    last_alert_ts = None
    while True:
        try:
            # BTC hourly alert logic
            candles = get_recent_hourly_candles(cb_service, num_candles=25)
            if not candles or len(candles) < 2:
                logger.warning("Not enough candle data.")
                time.sleep(60)
                continue

            # Debug: Print all candle timestamps and closes
            logger.info("--- Candle Debug Info ---")
            for i, c in enumerate(candles):
                ts = datetime.fromtimestamp(int(c['start']), UTC)
                close = c['close']
                logger.info(f"Candle {i}: {ts} close={close}")
            logger.info("-------------------------")

            # The most recent complete candle is the second in the list (index 1)
            last_candle = process_candle(candles[1])
            session_candles = [process_candle(c) for c in candles[1:21]]  # last 20 closed candles
            session_volumes = [c['volume'] for c in session_candles]
            avg_volume = sum(session_volumes) / 20 if len(session_volumes) == 20 else sum(session_volumes) / len(session_volumes)
            close = last_candle['close']
            volume = last_candle['volume']
            ts = last_candle['timestamp']
            logger.info(f"Last hourly close: {close}, volume: {volume}, avg_volume: {avg_volume}")
            if ts == last_alert_ts:
                logger.info("No new candle. Waiting...")
                time.sleep(60)
                continue
            if close >= BREAKOUT_THRESHOLD and volume >= BREAKOUT_VOLUME_MULTIPLIER * avg_volume:
                logger.info(f"ALERT: Hourly close ≥ {BREAKOUT_THRESHOLD} and volume ≥ session average!")
                logger.info(f"Timestamp: {ts}, Close: {close}, Volume: {volume}, Avg Volume: {avg_volume}")
                last_alert_ts = ts
            elif close <= BREAKDOWN_THRESHOLD and volume >= BREAKDOWN_VOLUME_MULTIPLIER * avg_volume:
                logger.info(f"BREAKDOWN ALERT: Hourly close ≤ {BREAKDOWN_THRESHOLD} and volume ≥ {(BREAKDOWN_VOLUME_MULTIPLIER-1)*100:.0f}% above avg!")
                logger.info(f"Timestamp: {ts}, Close: {close}, Volume: {volume}, Avg Volume: {avg_volume}")
                last_alert_ts = ts
            else:
                logger.info("No alert condition met.")
                last_alert_ts = ts

            # FARTOIN daily alert
            fartcoin_daily_alert(cb_service)

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