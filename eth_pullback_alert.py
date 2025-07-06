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

PRODUCT_ID = "ETH-PERP-INTX"
ETH_MARGIN = 300
ETH_LEVERAGE = 20
ENTRY_ZONE_LOW = 2450
ENTRY_ZONE_HIGH = 2510
TP1 = 2800
FALLBACK_STOP_LOSS = 2400
VWAP_LOOKBACK_DAYS = 30


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

def safe_get_candles(cb_service, product_id, start_ts, end_ts, granularity, max_candles_per_req=300):
    """
    Fetch candles in chunks to avoid API limit (max 350 per request).
    Returns a list of candles sorted by time ascending.
    """
    all_candles = []
    current_start = start_ts
    granularity_sec = 3600 if granularity == "ONE_HOUR" else 60 * 60 * 4 if granularity == "FOUR_HOUR" else 60
    while current_start < end_ts:
        current_end = min(current_start + granularity_sec * max_candles_per_req, end_ts)
        for _ in range(3):
            try:
                response = cb_service.client.get_public_candles(
                    product_id=product_id,
                    start=current_start,
                    end=current_end,
                    granularity=granularity
                )
                if hasattr(response, 'candles'):
                    candles = response.candles
                else:
                    candles = response.get('candles', [])
                all_candles.extend(candles)
                break
            except Exception as e:
                logger.warning(f"Retrying candle fetch: {e}")
                time.sleep(2)
        current_start = current_end
    # Remove duplicates and sort by start time ascending
    seen = set()
    unique_candles = []
    for c in all_candles:
        ts = c['start'] if isinstance(c, dict) else getattr(c, 'start', None)
        if ts not in seen:
            unique_candles.append(c)
            seen.add(ts)
    unique_candles.sort(key=lambda c: c['start'] if isinstance(c, dict) else getattr(c, 'start', 0))
    return unique_candles

def execute_crypto_trade(cb_service, trade_type: str, entry_price: float, stop_loss: float, take_profit: float, 
                     margin: float = ETH_MARGIN, leverage: int = ETH_LEVERAGE, side: str = "BUY", product: str = PRODUCT_ID):
    try:
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
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

def eth_pullback_alert(cb_service, last_alert_ts=None):
    GRANULARITY = "ONE_HOUR"
    periods_needed = VWAP_LOOKBACK_DAYS * 24 + 2  # 30 days of 1h candles + 2
    hours_needed = periods_needed
    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=hours_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough ETH {GRANULARITY} candle data for pull-back alert.")
            return last_alert_ts
        first_ts = int(candles[0]['start'])
        last_ts = int(candles[-1]['start'])
        if first_ts > last_ts:
            last_candle = candles[1]
            historical_candles = candles[2:]
        else:
            last_candle = candles[-2]
            historical_candles = candles[:-2]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)
        if ts == last_alert_ts:
            return last_alert_ts
        close = float(last_candle['close'])
        in_entry_zone = ENTRY_ZONE_LOW <= close <= ENTRY_ZONE_HIGH
        # VWAP calculation (optional)
        vwap_stop = None
        try:
            # logger.info(f"Candles type: {type(candles)}, length: {len(candles)}")
            if not candles:
                logger.warning("VWAP calculation skipped: No candle data returned.")
                df = pd.DataFrame()
            elif isinstance(candles[0], (list, tuple)):
                # logger.info("Candles format: list of lists/tuples")
                df = pd.DataFrame(candles, columns=['start', 'low', 'high', 'open', 'close', 'volume'])
            elif isinstance(candles[0], dict):
                # logger.info("Candles format: list of dicts")
                df = pd.DataFrame(candles)
            elif hasattr(candles[0], '__dict__'):
                # logger.info("Candles format: list of SDK objects; converting to dicts")
                dict_candles = [vars(c) for c in candles]
                df = pd.DataFrame(dict_candles)
            else:
                logger.warning(f"VWAP calculation skipped: Unexpected candle element type: {type(candles[0])}, value: {candles[0]}")
                df = pd.DataFrame()
            if not df.empty and all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
                # Fix: convert 'start' to int, then to datetime, remove tz, sort, and set as index
                df['start'] = df['start'].astype(int)
                df['start'] = pd.to_datetime(df['start'], unit='s', utc=True)
                df['start'] = df['start'].dt.tz_localize(None)  # Remove timezone info
                df = df.sort_values('start')
                df = df.set_index('start')
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
                vwap_30d = vwap.iloc[-2] if vwap is not None and len(vwap) >= 2 else None
                if vwap_30d is not None:
                    vwap_stop = float(vwap_30d)
            else:
                logger.warning(f"VWAP calculation skipped: DataFrame empty or missing columns. Columns: {df.columns.tolist()}")
        except Exception as e:
            logger.warning(f"VWAP calculation failed, using fallback stop-loss: {e}. Columns: {df.columns.tolist() if 'df' in locals() else 'N/A'}")
        stop_loss = vwap_stop if vwap_stop is not None else FALLBACK_STOP_LOSS
        logger.info(f"=== ETH PULL-BACK LONG ALERT ===")
        logger.info(f"Candle close: ${close:,.2f}, Entry zone: ${ENTRY_ZONE_LOW}-{ENTRY_ZONE_HIGH}, VWAP(30d): {vwap_stop if vwap_stop is not None else 'N/A'}")
        logger.info(f"  - Close in entry zone: {'✅ Met' if in_entry_zone else '❌ Not Met'}")
        if in_entry_zone:
            logger.info(f"--- ETH PULL-BACK LONG ALERT ---")
            logger.info(f"Entry condition met: 1h close in ${ENTRY_ZONE_LOW}-${ENTRY_ZONE_HIGH}.")
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH pull-back long",
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=TP1,
                margin=ETH_MARGIN,
                leverage=ETH_LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            if trade_success:
                logger.info(f"ETH pull-back long trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
            else:
                logger.error(f"ETH pull-back long trade failed: {trade_result}")
            return ts
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in ETH pull-back alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts

def main():
    logger.info("Starting ETH pull-back alert script")
    logger.info("")
    logger.info("✅ Ready to take ETH pull-back long trades (1H)")
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
    eth_last_alert_ts = None
    while True:
        try:
            iteration_start_time = time.time()
            eth_last_alert_ts = eth_pullback_alert(cb_service, eth_last_alert_ts)
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