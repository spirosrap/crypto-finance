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

# Constants for BTC intraday strategy
GRANULARITY_1H = "ONE_HOUR"
GRANULARITY_5M = "FIVE_MINUTE"
GRANULARITY_15M = "FIFTEEN_MINUTE"
PRODUCT_ID = "BTC-PERP-INTX"

# Global execution settings
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage (margin x leverage = $5,000 notional)

# Intraday reference (from brief):
INTRADAY_HIGH = 117596.0
INTRADAY_LOW = 114314.0

# Setups and levels (from brief)
# 1) Breakout-continuation (intraday high breach)
BREAKOUT_CONFIRM_LEVEL_15M = 117600.0  # 15m close above this
BREAKOUT_BUY_ZONE_LOW = 117700.0       # buy the strength zone
BREAKOUT_BUY_ZONE_HIGH = 118000.0
BREAKOUT_SL = 116950.0
BREAKOUT_TP1 = 118800.0
BREAKOUT_TP2 = 120400.0

# 2) VWAP reclaim after sweep (fade the low)
SWEEP_ZONE_LOW = 114300.0
SWEEP_ZONE_HIGH = 114400.0
VWAP_SL_BUFFER_PCT = 0.003  # 0.3% below sweep low
RECLAIM_TP1 = 116400.0
RECLAIM_TP2 = 117200.0
RECLAIM_TP3 = 118000.0

# 3) Failed-breakout short
FAILED_BO_PUSH_LEVEL = 117600.0  # push above
FAILED_BO_REJECT_CLOSE_5M = 117200.0  # then close back below on 5m
FAILED_BO_SELL_ZONE_LOW = 116900.0
FAILED_BO_SELL_ZONE_HIGH = 117100.0
FAILED_BO_SL = 117700.0
FAILED_BO_TP1 = 116000.0
FAILED_BO_TP2 = 115000.0

# 4) Breakdown-retest
BREAKDOWN_CONFIRM_LEVEL_15M = 114300.0  # 15m close below this
BREAKDOWN_RETEST_ZONE_LOW = 114200.0
BREAKDOWN_RETEST_ZONE_HIGH = 114400.0
BREAKDOWN_SL = 114900.0
BREAKDOWN_TP1 = 113200.0
BREAKDOWN_TP2 = 112300.0

# Volume confirmation thresholds
VOLUME_FACTOR_15M_BREAKOUT = 1.25
VOLUME_FACTOR_5M_RECLAIM = 1.30
VOLUME_FACTOR_5M_FAILED_BO = 1.40
VOLUME_FACTOR_15M_BREAKDOWN = 1.30

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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = "BTC-PERP-INTX"):
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
    Spiros — clean two-sided BTC plan for 07 Aug 2025 (live on Coinbase: ≈ $114,540, HOD $115,220, LOD $114,270)
    
    Rules (both directions):
    - Trigger: 1h; execute on 5–15m.
    - Volume confirm: ≥ 1.25 × 20-period vol (1h) or ≥ 2 × 20-SMA vol (5m).
    - Risk: size so 1 R ≈ 0.8–1.2 % of price; scale out ≥ +1.0–1.5 R.
    - Position Size: Always margin × leverage = 250 × 20 = $5,000 USD
    
    LONGS:
    - Breakout: Buy-stop 115,400–115,600 (above HOD + ~0.15%)
    - Retest: Limit 114,300–114,450 (prior LOD / demand shelf)
    
    SHORTS:
    - Breakdown: Sell-stop 113,950–113,750 (sub-LOD)
    - Lower-high retest: Limit 115,000–115,150 (failed push, bearish 1h close)
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== BTC Setups: Breakout, VWAP Reclaim, Failed Breakout, Breakdown-Retest ===")
    else:
        logger.info(f"=== BTC Setups ({direction} only): Breakout, VWAP Reclaim, Failed Breakout, Breakdown-Retest ===")
    
    # Load trigger state
    trigger_state = load_trigger_state()
    
    try:
        # Time ranges
        current_time = datetime.now(UTC)
        start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        now_hour = current_time.replace(minute=0, second=0, microsecond=0)

        # Fetch 5m candles for entire session (for VWAP) and last ~2 hours implicitly included
        start_ts_5m_session = int(start_of_day.timestamp())
        end_ts_5m_session = int(current_time.timestamp())

        # Fetch 15m candles for at least 20 bars history (≥5 hours). Use 12 hours for safety
        start_15m = current_time - timedelta(hours=12)
        end_15m = current_time
        start_ts_15m = int(start_15m.timestamp())
        end_ts_15m = int(end_15m.timestamp())

        logger.info(f"Fetching 5-minute candles (session) from {start_of_day} to {current_time}")
        candles_5m_session = safe_get_5m_candles(cb_service, PRODUCT_ID, start_ts_5m_session, end_ts_5m_session)

        logger.info(f"Fetching 15-minute candles from {start_15m} to {end_15m}")
        candles_15m = safe_get_15m_candles(cb_service, PRODUCT_ID, start_ts_15m, end_ts_15m)
        
        if not candles_5m_session or len(candles_5m_session) < 30:
            logger.warning("Not enough 5-minute session data for VWAP/volume analysis")
            return last_alert_ts

        if not candles_15m or len(candles_15m) < 22:  # Need at least 20+ bars
            logger.warning("Not enough 15-minute candle data for 20-period volume analysis")
            return last_alert_ts
        
        # DataFrames
        df_5m = candles_to_df(candles_5m_session)
        df_15m = candles_to_df(candles_15m)

        # Current price from latest 5m candle
        last_5m = candles_5m_session[0]
        current_price = float(get_candle_value(last_5m, 'close'))

        # Compute session VWAP
        vwap_series = compute_session_vwap(df_5m)

        # 5m volume SMA(20) excluding the latest in-progress bar
        if len(df_5m) >= 22:
            df_5m['vol_sma20'] = df_5m['volume'].rolling(window=20).mean()
        else:
            df_5m['vol_sma20'] = None

        # 15m volume SMA(20) excluding the latest completed bar when comparing
        df_15m['vol_sma20'] = df_15m['volume'].rolling(window=20).mean()
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 BTC actionable setups for today")
        logger.info(f"Live on Coinbase: ≈ ${current_price:,.0f}, Intraday High ${INTRADAY_HIGH:,.0f}, Low ${INTRADAY_LOW:,.0f}")
        logger.info("")
        logger.info("Set alerts: 117,700; 117,200; 114,300")
        logger.info(f"Position Size: ${MARGIN * LEVERAGE:,.0f} USD ({MARGIN} x {LEVERAGE}x)")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1) Breakout-continuation (15m close above 117,600; buy 117,700–118,000; vol: 15m ≥ 1.25x 20 SMA)
        if long_strategies_enabled and not trigger_state.get("breakout_continuation_triggered", False):
            # Last completed 15m candle
            last_15m_close = df_15m['close'].iloc[-1]
            last_15m_vol = df_15m['volume'].iloc[-1]
            # 20 SMA of previous 20 completed bars (exclude the last by shifting)
            sma20_prev = df_15m['vol_sma20'].iloc[-2] if len(df_15m) >= 22 else None
            vol_ok = (sma20_prev is not None) and (last_15m_vol >= VOLUME_FACTOR_15M_BREAKOUT * sma20_prev)
            price_close_ok = last_15m_close > BREAKOUT_CONFIRM_LEVEL_15M
            in_buy_zone = BREAKOUT_BUY_ZONE_LOW <= current_price <= BREAKOUT_BUY_ZONE_HIGH

            logger.info("🔍 LONG - Breakout-continuation:")
            logger.info(f"   • 15m close > {BREAKOUT_CONFIRM_LEVEL_15M:,.0f}: {'✅' if price_close_ok else '❌'} (close={last_15m_close:,.0f})")
            logger.info(f"   • 15m volume ≥ {VOLUME_FACTOR_15M_BREAKOUT:.2f}x 20-SMA: {'✅' if vol_ok else '❌'}")
            logger.info(f"   • Price in buy zone {BREAKOUT_BUY_ZONE_LOW:,.0f}-{BREAKOUT_BUY_ZONE_HIGH:,.0f}: {'✅' if in_buy_zone else '❌'}")

            if price_close_ok and vol_ok and in_buy_zone:
                logger.info("")
                logger.info("🎯 Breakout-continuation conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakout trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setup - Breakout-continuation Long",
                    entry_price=current_price,
                    stop_loss=BREAKOUT_SL,
                    take_profit=BREAKOUT_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakout-continuation trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakout_continuation_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(df_15m['start'].iloc[-1])
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout-continuation trade failed: {trade_result}")
        
        # 2) VWAP reclaim after sweep (5m): wick into 114,4xx-114,3xx, reclaim VWAP, then 5m higher-low above VWAP; vol ≥ 1.3x 20-SMA
        if long_strategies_enabled and not trade_executed and not trigger_state.get("vwap_reclaim_triggered", False):
            logger.info("")
            logger.info("🔍 LONG - VWAP reclaim after sweep:")

            vwap_ok = False
            hl_ok = False
            vol_ok = False
            sweep_low = None

            if vwap_series is not None and not vwap_series.empty and len(df_5m) >= 25:
                # Identify recent sweep (in last ~12 bars)
                recent_window = df_5m.tail(12).copy()
                sweeps = recent_window[(recent_window['low'] <= SWEEP_ZONE_HIGH) & (recent_window['low'] >= SWEEP_ZONE_LOW)]
                if sweeps.shape[0] == 0:
                    # allow slightly deeper sweep below zone
                    sweeps = recent_window[recent_window['low'] < SWEEP_ZONE_LOW]
                if sweeps.shape[0] > 0:
                    last_sweep_idx = sweeps.index.max()
                    sweep_low = float(df_5m.loc[last_sweep_idx, 'low'])
                    # Last two completed bars
                    last_idx = df_5m.index.max()
                    # Ensure we use completed bars (exclude very last row if still forming). We'll use last two rows.
                    if last_idx - last_sweep_idx >= 2:
                        c1 = df_5m.iloc[-1]  # last completed
                        c0 = df_5m.iloc[-2]  # prior completed
                        # VWAP at those bars
                        vwap_c1 = float(vwap_series.iloc[-1]) if not pd.isna(vwap_series.iloc[-1]) else None
                        vwap_c0 = float(vwap_series.iloc[-2]) if not pd.isna(vwap_series.iloc[-2]) else None
                        if vwap_c1 is not None and vwap_c0 is not None:
                            vwap_ok = (c1['close'] > vwap_c1)
                            hl_ok = (c1['low'] > c0['low']) and (c0['close'] > vwap_c0)
                            # Volume check on the reclaim bar (c1) vs SMA20 up to previous bar
                            sma20_prev = df_5m['vol_sma20'].iloc[-2]
                            vol_ok = (sma20_prev is not None) and (c1['volume'] >= VOLUME_FACTOR_5M_RECLAIM * sma20_prev)

            logger.info(f"   • Recent sweep into {SWEEP_ZONE_LOW:,.0f}-{SWEEP_ZONE_HIGH:,.0f}: {'✅' if sweep_low is not None else '❌'}")
            logger.info(f"   • Reclaim above VWAP: {'✅' if vwap_ok else '❌'}")
            logger.info(f"   • 5m higher-low above VWAP: {'✅' if hl_ok else '❌'}")
            logger.info(f"   • 5m volume ≥ {VOLUME_FACTOR_5M_RECLAIM:.2f}x 20-SMA: {'✅' if vol_ok else '❌'}")

            if sweep_low is not None and vwap_ok and hl_ok and vol_ok:
                logger.info("")
                logger.info("🎯 VWAP reclaim after sweep - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Reclaim trade
                reclaim_sl = round(sweep_low * (1 - VWAP_SL_BUFFER_PCT), 2)
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setup - VWAP reclaim Long",
                    entry_price=current_price,
                    stop_loss=reclaim_sl,
                    take_profit=RECLAIM_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="BUY",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 VWAP reclaim trade executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["vwap_reclaim_triggered"] = True
                    trigger_state["active_trade_direction"] = "LONG"
                    trigger_state["last_trigger_ts"] = int(df_5m['start'].iloc[-1])
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ VWAP reclaim trade failed: {trade_result}")
        
        # 3) Failed-breakout short: push above 117,600 then 5m close < 117,200; sell 117,100–116,900; vol ≥ 1.4x 20-SMA
        if short_strategies_enabled and not trade_executed and not trigger_state.get("failed_breakout_triggered", False):
            logger.info("")
            logger.info("🔍 SHORT - Failed-breakout:")

            # Check that a recent 5m candle pushed above 117,600
            recent_5m = df_5m.tail(24)
            had_push_above = (recent_5m['high'] > FAILED_BO_PUSH_LEVEL).any()
            # Last completed 5m close below 117,200
            last_close_5m = df_5m['close'].iloc[-1]
            reject_close_ok = last_close_5m < FAILED_BO_REJECT_CLOSE_5M
            # Volume check on that last bar vs SMA20 up to previous bar
            sma20_prev_5m = df_5m['vol_sma20'].iloc[-2] if len(df_5m) >= 22 else None
            last_vol_5m = df_5m['volume'].iloc[-1]
            vol_ok = (sma20_prev_5m is not None) and (last_vol_5m >= VOLUME_FACTOR_5M_FAILED_BO * sma20_prev_5m)
            # Entry zone
            in_sell_zone = FAILED_BO_SELL_ZONE_LOW <= current_price <= FAILED_BO_SELL_ZONE_HIGH

            logger.info(f"   • Recent push above {FAILED_BO_PUSH_LEVEL:,.0f}: {'✅' if had_push_above else '❌'}")
            logger.info(f"   • 5m close < {FAILED_BO_REJECT_CLOSE_5M:,.0f}: {'✅' if reject_close_ok else '❌'} (close={last_close_5m:,.0f})")
            logger.info(f"   • 5m volume ≥ {VOLUME_FACTOR_5M_FAILED_BO:.2f}x 20-SMA: {'✅' if vol_ok else '❌'}")
            logger.info(f"   • Price in sell zone {FAILED_BO_SELL_ZONE_HIGH:,.0f}-{FAILED_BO_SELL_ZONE_LOW:,.0f}: {'✅' if in_sell_zone else '❌'}")

            if had_push_above and reject_close_ok and vol_ok and in_sell_zone:
                logger.info("")
                logger.info("🎯 Failed-breakout conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Failed-breakout short
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setup - Failed-breakout Short",
                    entry_price=current_price,
                    stop_loss=FAILED_BO_SL,
                    take_profit=FAILED_BO_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Failed-breakout short executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["failed_breakout_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(df_5m['start'].iloc[-1])
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Failed-breakout short failed: {trade_result}")
        
        # 4) Breakdown-retest (15m close < 114,300, then retest 114,200–114,400; vol ≥ 1.3x 20-SMA)
        if short_strategies_enabled and not trade_executed and not trigger_state.get("breakdown_retest_triggered", False):
            logger.info("")
            logger.info("🔍 SHORT - Breakdown-retest:")

            # Last completed 15m candle close and volume
            last_15m_close = df_15m['close'].iloc[-1]
            last_15m_vol = df_15m['volume'].iloc[-1]
            sma20_prev_15m = df_15m['vol_sma20'].iloc[-2] if len(df_15m) >= 22 else None
            confirm_breakdown = last_15m_close < BREAKDOWN_CONFIRM_LEVEL_15M
            vol_ok = (sma20_prev_15m is not None) and (last_15m_vol >= VOLUME_FACTOR_15M_BREAKDOWN * sma20_prev_15m)
            in_retest_zone = BREAKDOWN_RETEST_ZONE_LOW <= current_price <= BREAKDOWN_RETEST_ZONE_HIGH

            logger.info(f"   • 15m close < {BREAKDOWN_CONFIRM_LEVEL_15M:,.0f}: {'✅' if confirm_breakdown else '❌'} (close={last_15m_close:,.0f})")
            logger.info(f"   • 15m volume ≥ {VOLUME_FACTOR_15M_BREAKDOWN:.2f}x 20-SMA: {'✅' if vol_ok else '❌'}")
            logger.info(f"   • Retest zone {BREAKDOWN_RETEST_ZONE_LOW:,.0f}-{BREAKDOWN_RETEST_ZONE_HIGH:,.0f}: {'✅' if in_retest_zone else '❌'}")

            if confirm_breakdown and vol_ok and in_retest_zone:
                logger.info("")
                logger.info("🎯 Breakdown-retest conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown-retest short
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="BTC Setup - Breakdown-retest Short",
                    entry_price=current_price,
                    stop_loss=BREAKDOWN_SL,
                    take_profit=BREAKDOWN_TP1,
                    margin=MARGIN,
                    leverage=LEVERAGE,
                    side="SELL",
                    product=PRODUCT_ID
                )
                
                if trade_success:
                    logger.info(f"🎉 Breakdown-retest short executed successfully!")
                    logger.info(f"Trade output: {trade_result}")
                    trigger_state["breakdown_retest_triggered"] = True
                    trigger_state["active_trade_direction"] = "SHORT"
                    trigger_state["last_trigger_ts"] = int(df_15m['start'].iloc[-1])
                    save_trigger_state(trigger_state)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown-retest short failed: {trade_result}")
        
        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for any setup")
            logger.info(f"Breakout-continuation triggered: {trigger_state.get('breakout_continuation_triggered', False)}")
            logger.info(f"VWAP reclaim triggered: {trigger_state.get('vwap_reclaim_triggered', False)}")
            logger.info(f"Failed-breakout triggered: {trigger_state.get('failed_breakout_triggered', False)}")
            logger.info(f"Breakdown-retest triggered: {trigger_state.get('breakdown_retest_triggered', False)}")
            logger.info(f"Active trade direction: {trigger_state.get('active_trade_direction', 'None')}")

        logger.info("=== BTC setups check completed ===")
        return last_alert_ts
        
    except Exception as e:
        logger.error(f"Error in BTC setups logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== BTC setups check completed (with error) ===")
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
    logger.info("Strategy Overview:")
    logger.info(f"  • Breakout-continuation: 15m close > {BREAKOUT_CONFIRM_LEVEL_15M:,.0f}; buy {BREAKOUT_BUY_ZONE_LOW:,.0f}–{BREAKOUT_BUY_ZONE_HIGH:,.0f}; SL {BREAKOUT_SL:,.0f}; TP1 {BREAKOUT_TP1:,.0f}")
    logger.info(f"  • VWAP reclaim: Sweep {SWEEP_ZONE_LOW:,.0f}–{SWEEP_ZONE_HIGH:,.0f}; reclaim VWAP + 5m HL; SL sweep_low-0.3%; TP1 {RECLAIM_TP1:,.0f}")
    logger.info(f"  • Failed-breakout short: push > {FAILED_BO_PUSH_LEVEL:,.0f} then 5m close < {FAILED_BO_REJECT_CLOSE_5M:,.0f}; sell {FAILED_BO_SELL_ZONE_LOW:,.0f}–{FAILED_BO_SELL_ZONE_HIGH:,.0f}; SL {FAILED_BO_SL:,.0f}; TP1 {FAILED_BO_TP1:,.0f}")
    logger.info(f"  • Breakdown-retest: 15m close < {BREAKDOWN_CONFIRM_LEVEL_15M:,.0f}; retest {BREAKDOWN_RETEST_ZONE_LOW:,.0f}–{BREAKDOWN_RETEST_ZONE_HIGH:,.0f}; SL {BREAKDOWN_SL:,.0f}; TP1 {BREAKDOWN_TP1:,.0f}")
    logger.info("  • Position Size: $5,000 USD (250 margin × 20x)")
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