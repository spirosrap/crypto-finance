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

PRODUCT_ID = "XRP-PERP-INTX"
XRP_MARGIN = 200
XRP_LEVERAGE = 10
STOP_LOSS = 2.17
TP1 = 2.38
TP2 = 2.60
ENTRY_TRIGGER = 2.24
ENTRY_ZONE_LOW = 2.23
ENTRY_ZONE_HIGH = 2.26
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.15  # 15% uptick


def play_alert_sound(filename="alert_sound.wav"):
    try:
        system = platform.system()
        if system == "Darwin":
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
    except Exception as e:
        logger.error(f"Error playing alert sound: {e}")
        return False

def setup_coinbase():
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    return CoinbaseService(api_key, api_secret)

def safe_get_candles(cb_service, product_id, start_ts, end_ts, granularity):
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
    for _ in range(3):
        try:
            return _get_candles()
        except Exception as e:
            logger.warning(f"Retrying candle fetch: {e}")
            time.sleep(2)
    return []

def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = XRP_MARGIN, leverage: int = XRP_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
    try:
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.4f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        position_size_usd = margin * leverage
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', product,
            '--side', side,
            '--size', str(position_size_usd),
            '--leverage', str(leverage),
            '--tp', str(take_profit),
            '--sl', str(stop_loss),
            '--no-confirm'
        ]
        logger.info(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info(f"Crypto {trade_type} trade executed successfully!")
            logger.info(f"Trade output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"Crypto {trade_type} trade failed!")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logger.error(f"Error executing crypto {trade_type} trade: {e}")
        return False, str(e)

def xrp_trendline_bounce_alert(cb_service, last_alert_ts_1h=None, last_alert_ts_2h=None):
    results = {}
    for timeframe in ['1h', '2h']:
        if timeframe == '1h':
            GRANULARITY = "ONE_HOUR"
            periods_needed = VOLUME_PERIOD + 2
            hours_needed = periods_needed
            last_alert_ts = last_alert_ts_1h
        else:
            GRANULARITY = "TWO_HOUR"
            periods_needed = VOLUME_PERIOD + 2
            hours_needed = periods_needed * 2
            last_alert_ts = last_alert_ts_2h
        try:
            now = datetime.now(UTC)
            now = now.replace(minute=0, second=0, microsecond=0)
            start = now - timedelta(hours=hours_needed)
            end = now
            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())
            candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
            if not candles or len(candles) < periods_needed:
                logger.warning(f"Not enough XRP {GRANULARITY} candle data for trendline bounce alert.")
                results[timeframe] = last_alert_ts
                continue
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
                results[timeframe] = last_alert_ts
                continue
            close = float(last_candle['close'])
            v0 = float(last_candle['volume'])
            avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
            reclaim = close >= ENTRY_TRIGGER
            in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
            vol_ok = v0 >= VOLUME_MULTIPLIER * avg20
            logger.info(f"=== XRP TRENDLINE BOUNCE ({timeframe.upper()}) ===")
            logger.info(f"Candle close: ${close:,.4f}, Volume: {v0:,.0f}, Avg(20): {avg20:,.0f}")
            logger.info(f"  - Reclaim >= ${ENTRY_TRIGGER:,.2f}: {'✅ Met' if reclaim else '❌ Not Met'}")
            logger.info(f"  - Entry zone ${ENTRY_ZONE_LOW:,.2f}-${ENTRY_ZONE_HIGH:,.2f}: {'✅ Met' if in_entry_zone else '❌ Not Met'}")
            logger.info(f"  - Volume ≥ {VOLUME_MULTIPLIER}x avg: {'✅ Met' if vol_ok else '❌ Not Met'}")
            if reclaim and in_entry_zone and vol_ok:
                logger.info(f"--- XRP TRENDLINE BOUNCE ALERT ({timeframe.upper()}) ---")
                logger.info(f"Entry condition met: {timeframe.upper()} close >= ${ENTRY_TRIGGER:,.2f} in zone ${ENTRY_ZONE_LOW:,.2f}-${ENTRY_ZONE_HIGH:,.2f} with volume spike.")
                try:
                    play_alert_sound()
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type=f"XRP trendline bounce ({timeframe})",
                    entry_price=close,
                    stop_loss=STOP_LOSS,
                    take_profit=TP1,
                    margin=XRP_MARGIN,
                    leverage=XRP_LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                if trade_success:
                    logger.info(f"XRP {timeframe.upper()} trendline bounce trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                else:
                    logger.error(f"XRP {timeframe.upper()} trendline bounce trade failed: {trade_result}")
                results[timeframe] = ts
            else:
                results[timeframe] = last_alert_ts
        except Exception as e:
            logger.error(f"Error in XRP trendline bounce alert logic ({timeframe}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[timeframe] = last_alert_ts
    return results.get('1h', last_alert_ts_1h), results.get('2h', last_alert_ts_2h)

def main():
    logger.info("Starting XRP trendline bounce alert script")
    logger.info("")
    logger.info("✅ Ready to take XRP trendline bounce trades (1H/2H)")
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
    xrp_last_alert_ts_1h = None
    xrp_last_alert_ts_2h = None
    while True:
        try:
            iteration_start_time = time.time()
            xrp_last_alert_ts_1h, xrp_last_alert_ts_2h = xrp_trendline_bounce_alert(cb_service, xrp_last_alert_ts_1h, xrp_last_alert_ts_2h)
            wait_seconds = 300
            logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
            logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
            logger.info("")
            time.sleep(wait_seconds)
        except KeyboardInterrupt:
            logger.info("👋 Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in alert loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(60)

if __name__ == "__main__":
    main() 