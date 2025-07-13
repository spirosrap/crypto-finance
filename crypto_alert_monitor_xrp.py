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
XRP_MARGIN = 300
XRP_LEVERAGE = 20
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


# === NEW XRP BREAKOUT STRATEGY ALERT & TRADE LOGIC (BLOCKS) ===

XRP_STOP_LOSS = 2.58
XRP_TP1 = 3.00
XRP_TP2 = 3.325  # Midpoint of 3.25–3.40
XRP_ENTRY_TRIGGER = 2.73
XRP_ENTRY_ZONE_LOW = 2.73
XRP_ENTRY_ZONE_HIGH = 2.75
XRP_VOLUME_MULTIPLIER = 1.15
XRP_VOLUME_PERIOD = 20
XRP_MARGIN = 300
XRP_LEVERAGE = 20
XRP_HARD_EXIT = 2.70

TRIGGER_STATE_FILE = "xrp_breakout_blocks_state.json"

def load_blocks_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"breakout_seen": False, "retest_fired": False, "breakout_ts": None}
    return {"breakout_seen": False, "retest_fired": False, "breakout_ts": None}

def save_blocks_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def xrp_breakout_blocks_alert(cb_service, last_alert_ts=None):
    PRODUCT_ID = "XRP-PERP-INTX"
    GRANULARITY = "ONE_DAY"
    periods_needed = XRP_VOLUME_PERIOD + 2
    state = load_blocks_state()
    try:
        now = datetime.now(UTC)
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=periods_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        if not candles or len(candles) < periods_needed:
            log_emoji("Not enough daily candle data.", "⚠️")
            return last_alert_ts
        # Ensure candles are in correct order (oldest to newest)
        if int(candles[0]['start']) > int(candles[-1]['start']):
            candles = list(reversed(candles))
        latest_candle = candles[-2]  # Last completed daily candle
        prev_candle = candles[-3] if len(candles) >= 3 else None
        historical_candles = candles[-(XRP_VOLUME_PERIOD+2):-2]
        close = float(latest_candle['close'])
        prev_close = float(prev_candle['close']) if prev_candle else None
        low = float(latest_candle['low'])
        v0 = float(latest_candle['volume'])
        avg20 = sum(float(c['volume']) for c in historical_candles) / len(historical_candles)
        ts = datetime.fromtimestamp(int(latest_candle['start']), UTC)
        # --- Reporting ---
        log_status(f"XRP Breakout Blocks Monitor\nClose: ${close:.2f} | Low: ${low:.2f} | Vol: {v0:,.0f} | AvgVol({XRP_VOLUME_PERIOD}): {avg20:,.0f}")
        # --- Block 1: Breakout continuation ---
        first_cross = close > XRP_ENTRY_TRIGGER and prev_close is not None and prev_close <= XRP_ENTRY_TRIGGER
        volume_ok = v0 >= XRP_VOLUME_MULTIPLIER * avg20
        clean_candle = low >= XRP_ENTRY_TRIGGER  # Optional: omit for less strict
        trigger = first_cross and volume_ok and clean_candle
        log_emoji(f"Breakout trigger: close > {XRP_ENTRY_TRIGGER} and prev_close <= {XRP_ENTRY_TRIGGER}: {'✅' if first_cross else '❌'}", "📈")
        log_emoji(f"Volume filter ≥ {XRP_VOLUME_MULTIPLIER}x: {'✅' if volume_ok else '❌'}", "🔊")
        log_emoji(f"Clean candle (low >= {XRP_ENTRY_TRIGGER}): {'✅' if clean_candle else '❌'}", "🕯️")
        # Reset breakout_seen if hard exit condition
        if close < XRP_HARD_EXIT:
            state = {"breakout_seen": False, "retest_fired": False, "breakout_ts": None}
            save_blocks_state(state)
            log_emoji(f"Breakout reset: close < {XRP_HARD_EXIT}", "🔄")
        # Block 1: Fire trigger
        if trigger and not state.get("breakout_seen", False):
            log_emoji(f"Breakout continuation detected! Buying at close ${close:.2f}", "🚀")
            play_alert_sound()
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="XRP Breakout Continuation",
                entry_price=close,
                stop_loss=XRP_STOP_LOSS,
                take_profit=XRP_TP1,
                margin=XRP_MARGIN,
                leverage=XRP_LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )
            if trade_success:
                log_emoji(f"Trade executed! TP1=${XRP_TP1}, SL=${XRP_STOP_LOSS}", "🎉")
            else:
                log_emoji(f"Trade failed: {trade_result}", "❌")
            state = {"breakout_seen": True, "retest_fired": False, "breakout_ts": int(latest_candle['start'])}
            save_blocks_state(state)
            return ts
        # --- Block 2: Pull-back retest ---
        if state.get("breakout_seen", False) and not state.get("retest_fired", False):
            # Look for first daily candle after breakout where low tags 2.73–2.75 and close > 2.73
            breakout_ts = state.get("breakout_ts")
            for c in candles:
                if int(c['start']) <= breakout_ts:
                    continue
                c_low = float(c['low'])
                c_close = float(c['close'])
                if XRP_ENTRY_ZONE_LOW <= c_low <= XRP_ENTRY_ZONE_HIGH and c_close > XRP_ENTRY_TRIGGER:
                    log_emoji(f"Pull-back retest detected! Limit buy in zone {XRP_ENTRY_ZONE_LOW}-{XRP_ENTRY_ZONE_HIGH} at close ${c_close:.2f}", "🟢")
                    play_alert_sound()
                    trade_success, trade_result = execute_crypto_trade(
                        cb_service=cb_service,
                        trade_type="XRP Pull-back Retest",
                        entry_price=c_close,
                        stop_loss=XRP_STOP_LOSS,
                        take_profit=XRP_TP1,
                        margin=XRP_MARGIN,
                        leverage=XRP_LEVERAGE,
                        side="BUY",
                        product=PRODUCT_ID
                    )
                    if trade_success:
                        log_emoji(f"Retest trade executed! TP1=${XRP_TP1}, SL=${XRP_STOP_LOSS}", "🎉")
                    else:
                        log_emoji(f"Retest trade failed: {trade_result}", "❌")
                    state["retest_fired"] = True
                    save_blocks_state(state)
                    return datetime.fromtimestamp(int(c['start']), UTC)
        return last_alert_ts
    except Exception as e:
        logger.error(f"Error in XRP breakout blocks alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    return last_alert_ts

# --- Helper: Format with emojis ---
def log_status(msg):
    logger.info(f"\n{'='*40}\n{msg}\n{'='*40}")

def log_emoji(msg, emoji):
    logger.info(f"{emoji} {msg}")

# --- Helper: Get equity for position sizing (stub, can be improved) ---
def get_equity(cb_service):
    try:
        equity, _ = cb_service.get_portfolio_info(portfolio_type="INTX")
        return equity
    except Exception:
        return 20000  # fallback

# --- Helper: Place bracket order for a given size ---
def place_bracket(cb_service, size, tp, sl, leverage, product=PRODUCT_ID):
    return cb_service.place_market_order_with_targets(
        product_id=product,
        side="BUY",
        size=size,
        take_profit_price=tp,
        stop_loss_price=sl,
        leverage=str(leverage)
    )

# --- Helper: Monitor for TP1 fill and place new bracket for TP2 ---
def monitor_tp1_and_rebracket(cb_service, product, original_size, tp1, tp2, sl, leverage):
    # Wait for TP1 fill (poll recent trades)
    log_emoji("Monitoring for TP1 fill (sell half)...", "🔎")
    half_size = original_size / 2
    filled_tp1 = False
    for _ in range(60):  # up to 1 hour (1 min polling)
        trades = cb_service.get_recent_trades()
        for t in trades:
            if t['product_id'] == product and float(t['price']) >= tp1 and float(t['size']) >= half_size * 0.95:
                filled_tp1 = True
                break
        if filled_tp1:
            break
        time.sleep(60)
    if filled_tp1:
        log_emoji(f"TP1 hit! Sold half at ${tp1}", "🎯")
        # Place new bracket for remainder
        log_emoji(f"Placing new bracket for remainder: TP2=${tp2}, SL=${sl}", "🔁")
        result = place_bracket(cb_service, half_size, tp2, sl, leverage, product)
        if result.get("error"):
            log_emoji(f"Error placing TP2 bracket: {result['error']}", "❌")
        else:
            log_emoji(f"TP2 bracket placed: TP2=${tp2}, SL=${sl}", "✅")
    else:
        log_emoji("TP1 not filled within monitoring window.", "⌛")

# --- Main loop ---
def main():
    logger.info("Starting XRP breakout blocks alert script (daily)")
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    cb_service = setup_coinbase()
    xrp_last_alert_ts = None
    while True:
        try:
            iteration_start_time = time.time()
            xrp_last_alert_ts = xrp_breakout_blocks_alert(cb_service, xrp_last_alert_ts)
            wait_seconds = 60  # 1 minute
            logger.info(f"✅ Alert cycle completed in {time.time() - iteration_start_time:.1f} seconds")
            logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll\n")
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