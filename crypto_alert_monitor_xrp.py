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

TRIGGER_STATE_FILE = "xrp_breakout_trigger_state.json"


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

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}
    return {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")


def xrp_breakout_pullback_alert(cb_service, last_alert_ts=None):
    GRANULARITY = "ONE_DAY"
    periods_needed = 30
    days_needed = periods_needed
    trigger_state = load_trigger_state()
    try:
        now = datetime.now(UTC)
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=days_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < 5:
            logger.warning(f"Not enough XRP {GRANULARITY} candle data for breakout alert.")
            return last_alert_ts
        # Ensure candles are in correct order (oldest to newest)
        if int(candles[0]['start']) > int(candles[-1]['start']):
            candles = list(reversed(candles))
        logger.info(f"Fetched {len(candles)} daily candles from {datetime.fromtimestamp(int(candles[0]['start']), UTC).strftime('%m-%d')} to {datetime.fromtimestamp(int(candles[-1]['start']), UTC).strftime('%m-%d')}")
        # Log the latest completed candle's info
        latest_candle = candles[-2]
        latest_time = datetime.fromtimestamp(int(latest_candle['start']), UTC)
        latest_close = float(latest_candle['close'])
        latest_low = float(latest_candle['low'])
        latest_high = float(latest_candle['high'])
        latest_vol = float(latest_candle['volume'])
        
        # Calculate average volume
        recent_volumes = [float(c['volume']) for c in candles[-VOLUME_PERIOD-1:-1]]  # Exclude current incomplete candle
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else 0
        
        logger.info(f"=== XRP DAILY BREAKOUT PULLBACK ALERT ===")
        logger.info(f"Candle close: ${latest_close:.2f}, Volume: {latest_vol:,.0f}, Avg({VOLUME_PERIOD}): {avg_volume:,.0f}")
        
        # Check conditions
        close_above_240 = latest_close > 2.40
        volume_above_threshold = latest_vol >= avg_volume * VOLUME_MULTIPLIER if avg_volume > 0 else False
        
        logger.info(f"  - Close > $2.40: {'✅ Met' if close_above_240 else '❌ Not Met'}")
        logger.info(f"  - Volume ≥ {VOLUME_MULTIPLIER}x avg: {'✅ Met' if volume_above_threshold else '❌ Not Met'}")
        
        # Show trigger status
        trigger_state_str = trigger_state.get("triggered", False)
        if not trigger_state_str:
            logger.info("  - Breakout trigger: ❌ Waiting for close > $2.40")
        else:
            logger.info("  - Breakout trigger: ✅ Active - waiting for pullback to $2.38-2.40")
        # Step 1: If not triggered, look for daily close > 2.40
        if not trigger_state.get("triggered", False):
            for i in range(len(candles)-2, -1, -1):  # Only check completed candles
                c = candles[i]
                close = float(c['close'])
                low = float(c['low'])
                ts = int(c['start'])
                if close > 2.40:
                    logger.info(f"✅ Breakout trigger set: Daily close ${close:.2f} > $2.40 at {datetime.fromtimestamp(ts, UTC).strftime('%m-%d')}")
                    trigger_state = {"triggered": True, "trigger_ts": ts, "min_price_since_trigger": close}
                    save_trigger_state(trigger_state)
                    return last_alert_ts
                if low < 2.24:
                    logger.info(f"❌ Price slipped below $2.24 (${low:.2f}) before trigger. Setup invalidated.")
                    trigger_state = {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}
                    save_trigger_state(trigger_state)
                    return last_alert_ts
            logger.info("⏳ No daily close > $2.40 found yet. Waiting...")
            return last_alert_ts
        # Step 2: If triggered, look for first pullback into 2.38–2.40, and check for slip < 2.24
        triggered_ts = trigger_state.get("trigger_ts")
        min_price = trigger_state.get("min_price_since_trigger", 9999)
        for i in range(len(candles)-2, -1, -1):
            c = candles[i]
            ts = int(c['start'])
            if ts <= triggered_ts:
                continue
            close = float(c['close'])
            low = float(c['low'])
            # Track min price since trigger
            if low < min_price:
                min_price = low
                trigger_state["min_price_since_trigger"] = min_price
                save_trigger_state(trigger_state)
            if low < 2.24:
                logger.info(f"❌ Price slipped below $2.24 (${low:.2f}) after trigger. Setup invalidated.")
                trigger_state = {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}
                save_trigger_state(trigger_state)
                return last_alert_ts
            if 2.38 <= close <= 2.40:
                logger.info(f"=== XRP BREAKOUT PULLBACK TRADE EXECUTED ===")
                logger.info(f"Entry condition met: pullback close ${close:.2f} in $2.38–2.40 after breakout trigger.")
                logger.info(f"Trade params: Stop Loss: $2.24, Take Profit: $2.60, Margin: ${XRP_MARGIN}, Leverage: {XRP_LEVERAGE}x")
                try:
                    play_alert_sound()
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="XRP breakout pullback long",
                    entry_price=close,
                    stop_loss=2.24,
                    take_profit=2.60,
                    margin=XRP_MARGIN,
                    leverage=XRP_LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                if trade_success:
                    logger.info(f"✅ XRP breakout pullback trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                else:
                    logger.error(f"❌ XRP breakout pullback trade failed: {trade_result}")
                # Reset trigger after trade
                trigger_state = {"triggered": False, "trigger_ts": None, "min_price_since_trigger": None}
                save_trigger_state(trigger_state)
                return datetime.fromtimestamp(ts, UTC)
        logger.info("⏳ Breakout triggered, but no valid pullback entry found yet.")
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in XRP breakout pullback alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts

def main():
    logger.info("Starting XRP breakout pullback alert script")
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
    xrp_last_alert_ts = None
    while True:
        try:
            iteration_start_time = time.time()
            xrp_last_alert_ts = xrp_breakout_pullback_alert(cb_service, xrp_last_alert_ts)
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