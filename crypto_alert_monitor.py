import time
from datetime import datetime, timedelta, UTC
from zoneinfo import ZoneInfo
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
import argparse

# Set up file logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_intraday_alert.log'),
        logging.StreamHandler()  # Keep console output too
    ]
)
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

def safe_get_5m_candles(cb_service, product_id, start_ts, end_ts):
    """
    Safely get 5-minute candles with retry logic
    """
    def _get_5m_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity="FIVE_MINUTE"
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_5m_candles)

def safe_get_15m_candles(cb_service, product_id, start_ts, end_ts):
    """
    Safely get 15-minute candles with retry logic
    """
    def _get_15m_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity="FIFTEEN_MINUTE"
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_15m_candles)

def safe_get_1h_candles(cb_service, product_id, start_ts, end_ts):
    """
    Safely get 1-hour candles with retry logic
    """
    def _get_1h_candles():
        response = cb_service.client.get_public_candles(
            product_id=product_id,
            start=start_ts,
            end=end_ts,
            granularity="ONE_HOUR"
        )
        if hasattr(response, 'candles'):
            return response.candles
        else:
            return response.get('candles', [])
    
    return retry_with_backoff(_get_1h_candles)

# Constants for rule-based BTC plan
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global execution settings (fixed position sizing)
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage (margin x leverage = $5,000 notional)

# Rule levels and parameters
LONG_TRIGGER = 117360.0
SHORT_TRIGGER = 115780.0
RVOL20_THRESHOLD = 1.25
RVOL20_FADE = 1.30
ATR_MIN_PCT = 0.35
ATR_MAX_PCT = 1.20
FUNDING_MIN_PCT = -0.03
FUNDING_MAX_PCT = 0.03
STOP_PCT_BUFFER = 0.60  # percent
LIMIT_SLIPPAGE_PCT = 0.10  # percent
TP1_PCT = 1.20
TP2_PCT = 2.40

# Trade tracking
TRIGGER_STATE_FILE = "btc_intraday_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "breakout_triggered": False, 
                "retest_triggered": False, 
                "breakdown_triggered": False,
                "fade_triggered": False,
                "last_trigger_ts": None,
                "last_1h_structure": None,
                "active_trade_direction": None  # Track which direction is active
            }
    return {
        "breakout_triggered": False, 
        "retest_triggered": False, 
        "breakdown_triggered": False,
        "fade_triggered": False,
        "last_trigger_ts": None,
        "last_1h_structure": None,
        "active_trade_direction": None  # Track which direction is active
    }

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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = "BTC-PERP-INTX",
                     limit_price: float | None = None):
    """
    General crypto trade execution function using trade_btc_perp.py with retry logic
    
    Args:
        cb_service: Coinbase service instance
        trade_type: Description of the trade type for logging
        entry_price: Entry price for logging
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        margin: USD amount to risk (default: 250)
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
        # If a limit price is provided, treat this as a stop-limit style entry with a slippage cap
        if limit_price is not None:
            cmd.extend(['--limit', str(limit_price)])
        
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

def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR value using the last completed bar set."""
    if df is None or df.empty or len(df) < period + 1:
        return 0.0
    highs = df['high']
    lows = df['low']
    closes = df['close']
    tr1 = highs - lows
    tr2 = (highs - closes.shift(1)).abs()
    tr3 = (lows - closes.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(window=period).mean()
    # Use last completed bar ATR
    return float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

def get_funding_rate_pct() -> float | None:
    """
    Placeholder for funding rate fetch. Returns None if unavailable.
    Implement integration with your preferred data source if needed.
    """
    return None

def calculate_volume_sma(candles, period=20):
    """
    Calculate Simple Moving Average of volume
    
    Args:
        candles: List of candle data
        period: Period for SMA calculation
    
    Returns:
        Volume SMA value
    """
    if len(candles) < period:
        return 0
    
    volumes = []
    for candle in candles[1:period+1]:  # Use most recent period candles (skip current incomplete candle)
        if isinstance(candle, dict):
            volume = float(candle.get('volume', 0))
        else:
            volume = float(getattr(candle, 'volume', 0))
        volumes.append(volume)
    
    return sum(volumes) / len(volumes) if volumes else 0

def candles_to_df(candles):
    """Convert candles (dicts or objects) to a pandas DataFrame sorted ascending by start time."""
    if not candles:
        return pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume"])
    records = []
    for c in candles:
        try:
            start = int(get_candle_value(c, 'start'))
            records.append({
                'start': start,
                'open': float(get_candle_value(c, 'open')),
                'high': float(get_candle_value(c, 'high')),
                'low': float(get_candle_value(c, 'low')),
                'close': float(get_candle_value(c, 'close')),
                'volume': float(get_candle_value(c, 'volume'))
            })
        except Exception:
            continue
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values('start').reset_index(drop=True)

def compute_session_vwap(df_5m_session):
    """Compute session VWAP series from 5m candles using pandas_ta."""
    if df_5m_session is None or df_5m_session.empty:
        return None
    try:
        vwap_series = ta.vwap(df_5m_session['high'], df_5m_session['low'], df_5m_session['close'], df_5m_session['volume'])
        return vwap_series
    except Exception as e:
        logger.warning(f"VWAP calculation failed: {e}")
        return None

def get_candle_value(candle, key):
    """Extract value from candle object (handles both dict and object formats)"""
    if isinstance(candle, dict):
        return candle.get(key)
    else:
        return getattr(candle, key, None)




def btc_intraday_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    Rule-based BTC plan: triggers, acceptance, ATR% filter, RVOL20 filter,
    funding bounds, fixed position sizing (250 x 20), max 2 trades/day,
    18:00 CDT cutoff, and optional fade.
    """
    # Load state and reset daily counters
    state = load_trigger_state()
    now_utc = datetime.now(UTC)
    now_cdt = datetime.now(ZoneInfo("America/Chicago"))
    today_str = now_cdt.strftime('%Y-%m-%d')
    if state.get('last_trade_date') != today_str:
        state['trades_today'] = 0
        state['last_trade_date'] = today_str
        state['active_trade_direction'] = None
        state['long_taken'] = False
        state['short_taken'] = False
        state['fade_taken'] = False
        save_trigger_state(state)

    # Respect cutoff time 18:00 CDT
    if now_cdt.hour > 18 or (now_cdt.hour == 18 and (now_cdt.minute > 0 or now_cdt.second > 0)):
        logger.info("⏹ After 18:00 CDT — pausing new orders per plan.")
        return last_alert_ts

    if int(state.get('trades_today', 0)) >= 2:
        logger.info("✅ Max trades for the day reached (2). Skipping new entries.")
        return last_alert_ts

    try:
        # Time ranges
        start_15m = now_utc - timedelta(hours=12)
        start_1h = now_utc - timedelta(hours=48)
        start_ts_15m = int(start_15m.timestamp())
        start_ts_1h = int(start_1h.timestamp())
        end_ts = int(now_utc.timestamp())

        # Fetch candles
        logger.info("Fetching 15m and 1h candles for acceptance and filters…")
        candles_15m = safe_get_15m_candles(cb_service, PRODUCT_ID, start_ts_15m, end_ts)
        candles_1h = safe_get_1h_candles(cb_service, PRODUCT_ID, start_ts_1h, end_ts)

        if not candles_15m or len(candles_15m) < 22:
            logger.warning("Not enough 15m data for RVOL/ATR filters")
            return last_alert_ts
        if not candles_1h or len(candles_1h) < 3:
            logger.warning("Not enough 1h data for acceptance checks")
            return last_alert_ts

        df_15m = candles_to_df(candles_15m)
        df_1h = candles_to_df(candles_1h)

        current_price = float(df_15m['close'].iloc[-1])

        # RVOL20 on last completed 15m bar
        df_15m['vol_sma20'] = df_15m['volume'].rolling(window=20).mean()
        last_vol_15m = float(df_15m['volume'].iloc[-1])
        vol_sma20_prev = float(df_15m['vol_sma20'].iloc[-2]) if len(df_15m) >= 22 else 0.0
        rvol20 = (last_vol_15m / vol_sma20_prev) if vol_sma20_prev > 0 else 0.0

        # ATR filters (15m)
        atr_value = compute_atr(df_15m)
        atr_pct = (atr_value / current_price) * 100 if current_price > 0 else 0.0
        atr_ok = (ATR_MIN_PCT <= atr_pct <= ATR_MAX_PCT)

        # Funding filter (placeholder)
        funding_pct = get_funding_rate_pct()
        funding_str = f"{funding_pct:.4f}%" if funding_pct is not None else "n/a"

        # Acceptance checks
        last_close_1h = float(df_1h['close'].iloc[-1])
        last_close_15m = float(df_15m['close'].iloc[-1])
        accept_long = (last_close_1h >= LONG_TRIGGER) or (last_close_15m >= LONG_TRIGGER and rvol20 >= RVOL20_THRESHOLD)
        accept_short = (last_close_1h <= SHORT_TRIGGER) or (last_close_15m <= SHORT_TRIGGER and rvol20 >= RVOL20_THRESHOLD)

        logger.info("")
        logger.info("— Plan status —")
        logger.info(f"Price=${current_price:,.0f} | ATR% (15m)={atr_pct:.2f}% in [{ATR_MIN_PCT}%, {ATR_MAX_PCT}%] -> {'OK' if atr_ok else 'SKIP'} | RVOL20={rvol20:.2f}")
        logger.info(f"Funding={funding_str} (bounds {FUNDING_MIN_PCT}%..{FUNDING_MAX_PCT}%)")
        logger.info(f"Acceptance: LONG={'YES' if accept_long else 'NO'} | SHORT={'YES' if accept_short else 'NO'}")
        logger.info(f"Trades today: {state.get('trades_today', 0)}/2; Active: {state.get('active_trade_direction', 'None')}")

        def place_rule_trade(side: str) -> bool:
            # Funding guardrails
            if funding_pct is not None:
                if side == 'BUY' and funding_pct > FUNDING_MAX_PCT:
                    logger.info("⛔ Funding too positive for LONG. Skipping.")
                    return False
                if side == 'SELL' and funding_pct < FUNDING_MIN_PCT:
                    logger.info("⛔ Funding too negative for SHORT. Skipping.")
                    return False

            if not atr_ok:
                logger.info("⛔ ATR% filter not satisfied. Skipping.")
                return False

            entry = current_price
            pct_buffer = (STOP_PCT_BUFFER / 100.0) * entry
            sl_offset = max(pct_buffer, atr_value)
            if side == 'BUY':
                sl_price = round(entry - sl_offset, 2)
                tp_price = round(entry * (1 + TP1_PCT / 100.0), 2)
                limit_price = round(entry * (1 + LIMIT_SLIPPAGE_PCT / 100.0), 2)
            else:
                sl_price = round(entry + sl_offset, 2)
                tp_price = round(entry * (1 - TP1_PCT / 100.0), 2)
                limit_price = round(entry * (1 - LIMIT_SLIPPAGE_PCT / 100.0), 2)

            logger.info("")
            logger.info(f"🎯 {side} setup met — placing stop-limit style entry (cap {LIMIT_SLIPPAGE_PCT:.2f}%)")
            logger.info(f"Entry≈${entry:,.2f} | SL=${sl_price:,.2f} | TP1=${tp_price:,.2f} | ATR=${atr_value:,.2f} ({atr_pct:.2f}%)")

            try:
                play_alert_sound()
            except Exception:
                pass

            ok, out = execute_crypto_trade(
                cb_service=cb_service,
                trade_type=f"BTC Rule Plan {'LONG' if side=='BUY' else 'SHORT'}",
                entry_price=entry,
                stop_loss=sl_price,
                take_profit=tp_price,
                margin=MARGIN,
                leverage=LEVERAGE,
                side=side,
                product=PRODUCT_ID,
                limit_price=limit_price
            )
            if ok:
                logger.info("✅ Trade placed")
                state['trades_today'] = int(state.get('trades_today', 0)) + 1
                state['active_trade_direction'] = 'LONG' if side == 'BUY' else 'SHORT'
                if side == 'BUY':
                    state['long_taken'] = True
                else:
                    state['short_taken'] = True
                save_trigger_state(state)
                return True
            else:
                logger.error(f"❌ Trade placement failed: {out}")
                return False

        trade_done = False
        if direction in ['LONG', 'BOTH'] and not state.get('long_taken', False) and accept_long:
            trade_done = place_rule_trade('BUY')

        if not trade_done and direction in ['SHORT', 'BOTH'] and not state.get('short_taken', False) and accept_short:
            trade_done = place_rule_trade('SELL')

        # Optional fade — failed breakout (one shot only)
        if not trade_done and direction in ['SHORT', 'BOTH'] and not state.get('fade_taken', False):
            wick_above = (df_15m['high'].tail(6) > LONG_TRIGGER).any()
            close_back_below = last_close_15m < 117200.0
            if wick_above and close_back_below and rvol20 >= RVOL20_FADE:
                logger.info("⚠️ Failed breakout fade condition detected")
                if place_rule_trade('SELL'):
                    state['fade_taken'] = True
                    save_trigger_state(state)

        logger.info("=== Rule-based plan check completed ===")
        return last_alert_ts

    except Exception as e:
        logger.error(f"Error in rule-based plan: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== Rule-based plan check completed (with error) ===")
        return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BTC Actionable Setups Monitor with optional direction filter')
    parser.add_argument('--direction', choices=['LONG', 'SHORT', 'BOTH'], default='BOTH',
                       help='Trading direction to monitor: LONG, SHORT, or BOTH (default: BOTH)')
    args = parser.parse_args()
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    logger.info("Strategy Overview (Rule-based Plan):")
    logger.info(f"  • LONG: Stop {LONG_TRIGGER:,.0f} with acceptance (1h close ≥ trigger OR 15m close ≥ trigger & RVOL20 ≥ {RVOL20_THRESHOLD})")
    logger.info(f"  • SHORT: Stop {SHORT_TRIGGER:,.0f} with acceptance (1h close ≤ trigger OR 15m close ≤ trigger & RVOL20 ≥ {RVOL20_THRESHOLD})")
    logger.info(f"  • Filters: ATR% (15m) in [{ATR_MIN_PCT}%, {ATR_MAX_PCT}%], Funding in [{FUNDING_MIN_PCT}%, {FUNDING_MAX_PCT}%]")
    logger.info(f"  • SL: max({STOP_PCT_BUFFER:.2f}%, 1×ATR15m)")
    logger.info(f"  • TP: +{TP1_PCT:.1f}% (TP1), +{TP2_PCT:.1f}% (TP2; trailing not automated)")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20x)")
    logger.info("  • Max trades/day: 2; Cutoff: 18:00 CDT")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting BTC actionable setups monitor")
    logger.info(f"Strategy: {direction} only" if direction != 'BOTH' else "Strategy: LONG & SHORT")
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
        last_alert_ts = btc_intraday_alert(cb_service, last_alert_ts, direction)
        consecutive_failures = 0
        logger.info(f"✅ Intraday alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)  # 2 minute max per poll
                    wait_seconds = 300  # 5 minutes between polls
                    logger.info(f"⏰ Waiting {wait_seconds} seconds until next poll")
                    logger.info("")
                    time.sleep(wait_seconds)
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
