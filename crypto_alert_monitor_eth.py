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
        logging.FileHandler('eth_alert_debug.log'),
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
    import random
    delay = min(INITIAL_RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt), MAX_RETRY_DELAY)
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
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
            logger.error(f"❌ Non-recoverable error: {e}")
            return None
    return None

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
    return retry_with_backoff(_get_candles)

# ETH Trading Strategy Parameters (based on new ETH plan - current levels)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_1H = "ONE_HOUR"  # 1-hour chart for trigger
GRANULARITY_5M = "FIVE_MINUTE"  # 5-minute chart for execution
GRANULARITY_15M = "FIFTEEN_MINUTE"  # 15-minute chart for confirmation
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (ETH ≈ $3,686, HOD $3,715.95, LOD $3,578.18)
CURRENT_ETH_PRICE = 3686.00
HOD = 3715.95  # High of day
LOD = 3578.18  # Low of day
TODAYS_RANGE_WIDTH = HOD - LOD  # 137.77 points
MID_RANGE_PIVOT = (HOD + LOD) / 2  # 3647.065

# LONG (breakout) Strategy Parameters
BREAKOUT_ENTRY_LOW = 3955  # Retest zone per plan
BREAKOUT_ENTRY_HIGH = 3965
BREAKOUT_STOP_LOSS = 3930  # Invalidation
BREAKOUT_TP1 = 4000  # TP1
BREAKOUT_TP2_LOW = 4065  # TP2 range low
BREAKOUT_TP2_HIGH = 4100  # TP2 range high

# LONG (retest) Strategy Parameters
RETEST_ENTRY_LOW = 3915  # PDH flip zone
RETEST_ENTRY_HIGH = 3925
RETEST_STOP_LOSS = 3895  # Invalidation
RETEST_TP1 = 3965  # TP1
RETEST_TP2_LOW = 4020  # TP2
RETEST_TP2_HIGH = 4020

# SHORT (breakdown) Strategy Parameters
BREAKDOWN_ENTRY_LOW = 3820
BREAKDOWN_ENTRY_HIGH = 3900
BREAKDOWN_STOP_LOSS = 3920
BREAKDOWN_TP1 = 3860
BREAKDOWN_TP2_LOW = 3820
BREAKDOWN_TP2_HIGH = 3820

# SHORT (retest) Strategy Parameters
FADE_ENTRY_LOW = 3955
FADE_ENTRY_HIGH = 3970
FADE_STOP_LOSS = 3975
FADE_TP1 = 3915
FADE_TP2_LOW = 3890
FADE_TP2_HIGH = 3890

# Volume confirmation requirements
VOLUME_SURGE_FACTOR_1H = 1.25  # ≥1.25× 20-period vol on 1h
VOLUME_SURGE_FACTOR_5M = 2.0   # ≥2× 20-SMA vol on 5m at trigger

# Risk management
RISK_PERCENTAGE_LOW = 0.8  # 1R ≈ 0.8–1.2% of price
RISK_PERCENTAGE_HIGH = 1.2
PARTIAL_PROFIT_RANGE_LOW = 1.0  # Partial at +1.0R
PARTIAL_PROFIT_RANGE_HIGH = 1.5  # Partial at +1.5R

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for each strategy
BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
RETEST_TRIGGER_FILE = "eth_retest_trigger_state.json"
BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"
FADE_TRIGGER_FILE = "eth_fade_trigger_state.json"

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
    def _create_service():
        service = CoinbaseService(api_key, api_secret)
        try:
            test_response = service.client.get_public_candles(
                product_id=PRODUCT_ID,
                start=int((datetime.now(UTC) - timedelta(days=2)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity=GRANULARITY_1H
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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = PRODUCT_ID, 
                     volume_confirmed: bool = True):
    def _execute_trade():
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        
        # Always use full position size per requirement (margin x leverage = $5,000)
        position_size_usd = POSITION_SIZE_USD
        
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
    try:
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

def load_trigger_state(strategy_file):
    if os.path.exists(strategy_file):
        try:
            with open(strategy_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None, "entry_price": None}
    return {"triggered": False, "trigger_ts": None, "entry_price": None}

def save_trigger_state(state, strategy_file):
    try:
        with open(strategy_file, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def check_volume_confirmation(cb_service, current_volume_1h, current_volume_5m, avg_volume_1h, avg_volume_5m):
    """Check volume confirmation on both 1h and 5m timeframes"""
    volume_1h_confirmed = current_volume_1h >= (VOLUME_SURGE_FACTOR_1H * avg_volume_1h)
    volume_5m_confirmed = current_volume_5m >= (VOLUME_SURGE_FACTOR_5M * avg_volume_5m)
    
    # Volume must be confirmed on either 1h OR 5m timeframe
    volume_confirmed = volume_1h_confirmed or volume_5m_confirmed
    
    logger.info("Volume confirmation check:")
    logger.info(f"  1H: {current_volume_1h:,.0f} vs {VOLUME_SURGE_FACTOR_1H}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_1h_confirmed else '❌'}")
    logger.info(f"  5M: {current_volume_5m:,.0f} vs {VOLUME_SURGE_FACTOR_5M}x avg ({avg_volume_5m:,.0f}) -> {'✅' if volume_5m_confirmed else '❌'}")
    logger.info(f"  Overall: {'✅' if volume_confirmed else '❌'}")
    
    return volume_confirmed



def get_current_day_hod_lod(cb_service, current_ts):
    """Get current day's HOD and LOD"""
    try:
        # Get today's candles starting from market open (UTC)
        today_start = current_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        start_ts = int(today_start.timestamp())
        end_ts = int(current_ts.timestamp())
        
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_1H)
        
        if not candles_1h:
            return HOD, LOD  # Use default values if no data
        
        # Calculate HOD and LOD from today's candles
        hod = max(float(candle['high']) for candle in candles_1h)
        lod = min(float(candle['low']) for candle in candles_1h)
        
        logger.info(f"Current day HOD: ${hod:,.2f}, LOD: ${lod:,.2f}")
        return hod, lod
        
    except Exception as e:
        logger.error(f"Error getting current day HOD/LOD: {e}")
        return HOD, LOD  # Use default values

def check_new_structure_formation(cb_service, current_ts, previous_hod, previous_lod):
    """Check if a new 1h structure has formed (new HOD or LOD)"""
    try:
        current_hod, current_lod = get_current_day_hod_lod(cb_service, current_ts)
        
        # Check if we have new highs or lows
        new_hod = current_hod > previous_hod
        new_lod = current_lod < previous_lod
        
        if new_hod or new_lod:
            logger.info(f"🔄 New 1h structure detected: {'HOD' if new_hod else ''}{' and ' if new_hod and new_lod else ''}{'LOD' if new_lod else ''}")
            logger.info(f"Previous HOD: ${previous_hod:,.2f} -> Current HOD: ${current_hod:,.2f}")
            logger.info(f"Previous LOD: ${previous_lod:,.2f} -> Current LOD: ${current_lod:,.2f}")
            return True, current_hod, current_lod
        
        return False, current_hod, current_lod
        
    except Exception as e:
        logger.error(f"Error checking new structure formation: {e}")
        return False, previous_hod, previous_lod



# --- ETH Trading Strategy Alert Logic ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH-USD Trading Strategy Alert - Implements clean two-sided ETH plan with current levels
    Based on the trading plan: "Spiros — here's a clean, two-sided plan for today based on live levels"
    
    Rules (both directions):
    - Time-frame: 1h trigger, execute on 5–15m
    - Volume confirm: ≥ 1.25× 20-period vol on 1h OR ≥ 2× 20-SMA vol on 5m at trigger
    - Risk: size for 1R = 0.8–1.2% of price; take 30-50% off at +1.0–1.5R; trail remainder
    - Position Size: Always margin x leverage = 250 x 20 = $5,000 USD
    
    LONG SETUPS:
    - Breakout: Buy-stop $3,723–3,730 (HOD + ~0.2%), SL $3,698, TP1 $3,780, TP2 $3,820–3,850
    - Retest: $3,660–3,670 (prior 1h pivot/VWAP), SL $3,630, TP1 $3,710, TP2 $3,750
    
    SHORT SETUPS:
    - Breakdown: Sell-stop $3,570–3,560 (LOD – ~0.2%), SL $3,598, TP1 $3,500, TP2 $3,450–3,420
    - Retest: $3,620–3,635 (retest broken 1h support), SL $3,660, TP1 $3,585, TP2 $3,520
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== ETH-USD Trading Strategy Alert (Clean Two-Sided ETH Plan - LONG & SHORT) ===")
    else:
        logger.info(f"=== ETH-USD Trading Strategy Alert ({direction} Strategy Only) ===")
    
    # Load trigger states for all strategies
    breakout_state = load_trigger_state(BREAKOUT_TRIGGER_FILE)
    retest_state = load_trigger_state(RETEST_TRIGGER_FILE)
    breakdown_state = load_trigger_state(BREAKDOWN_TRIGGER_FILE)
    fade_state = load_trigger_state(FADE_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 1-hour candles for analysis
        end = now
        start = now - timedelta(hours=VOLUME_PERIOD + 24)  # Enough data for volume analysis
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 1-hour candles for {VOLUME_PERIOD + 24} hours...")
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_1H)
        
        if not candles_1h or len(candles_1h) < VOLUME_PERIOD + 1:
            logger.warning("Not enough 1-hour candle data for trading strategy alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending
        candles_1h = sorted(candles_1h, key=lambda x: int(x['start']))
        
        # Get current 1-hour candle data
        current_candle_1h = candles_1h[-1]
        current_close_1h = float(current_candle_1h['close'])
        current_high_1h = float(current_candle_1h['high'])
        current_low_1h = float(current_candle_1h['low'])
        current_volume_1h = float(current_candle_1h['volume'])
        current_ts_1h = datetime.fromtimestamp(int(current_candle_1h['start']), UTC)
        
        # Get current day's HOD and LOD
        current_hod, current_lod = get_current_day_hod_lod(cb_service, current_ts_1h)
        
        # Calculate current range context
        current_range_width = current_hod - current_lod
        current_mid_range = (current_hod + current_lod) / 2
        
        logger.info(f"📊 Current range context: ${current_lod:,.2f}-${current_hod:,.2f} (width ≈ {current_range_width:.0f})")
        logger.info(f"📊 Current mid-range pivot: ${current_mid_range:,.2f}")
        logger.info(f"📊 Current HOD: ${current_hod:,.2f}, LOD: ${current_lod:,.2f}")
        
        # Check for new structure formation and reset stopped_out flags if needed
        # Get previous HOD/LOD from state files or use current values
        previous_hod = max(
            breakout_state.get("last_hod", current_hod),
            retest_state.get("last_hod", current_hod),
            breakdown_state.get("last_hod", current_hod),
            fade_state.get("last_hod", current_hod)
        )
        previous_lod = min(
            breakout_state.get("last_lod", current_lod),
            retest_state.get("last_lod", current_lod),
            breakdown_state.get("last_lod", current_lod),
            fade_state.get("last_lod", current_lod)
        )
        
        new_structure_formed, updated_hod, updated_lod = check_new_structure_formation(
            cb_service, current_ts_1h, previous_hod, previous_lod
        )
        
        if new_structure_formed:
            logger.info("🔄 New 1h structure formed - resetting stopped_out flags for all strategies")
            # Reset stopped_out flags for all strategies
            for strategy_name, state, state_file in [
                ("Breakout", breakout_state, BREAKOUT_TRIGGER_FILE),
                ("Retest", retest_state, RETEST_TRIGGER_FILE),
                ("Breakdown", breakdown_state, BREAKDOWN_TRIGGER_FILE),
                ("Fade", fade_state, FADE_TRIGGER_FILE)
            ]:
                if state.get("stopped_out", False):
                    state["stopped_out"] = False
                    state["last_hod"] = updated_hod
                    state["last_lod"] = updated_lod
                    save_trigger_state(state, state_file)
                    logger.info(f"✅ Reset stopped_out flag for {strategy_name} strategy")
        else:
            # Update last HOD/LOD in all states
            for state, state_file in [(breakout_state, BREAKOUT_TRIGGER_FILE), 
                                     (retest_state, RETEST_TRIGGER_FILE), 
                                     (breakdown_state, BREAKDOWN_TRIGGER_FILE), 
                                     (fade_state, FADE_TRIGGER_FILE)]:
                state["last_hod"] = current_hod
                state["last_lod"] = current_lod
                save_trigger_state(state, state_file)
        
        # Calculate 20-period average volume for 1h (excluding current candle)
        volume_candles_1h = candles_1h[-(VOLUME_PERIOD+1):-1]
        avg_volume_1h = sum(float(c['volume']) for c in volume_candles_1h) / len(volume_candles_1h)
        
        # Get 5-minute candles for volume confirmation and retest analysis
        start_5m = now - timedelta(hours=2)
        start_ts_5m = int(start_5m.timestamp())
        end_ts_5m = int(now.timestamp())
        
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts_5m, GRANULARITY_5M)
        
        if candles_5m and len(candles_5m) >= VOLUME_PERIOD + 1:
            candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
            current_candle_5m = candles_5m[-1]
            current_volume_5m = float(current_candle_5m['volume'])
            
            # Calculate 20-period average volume for 5m (excluding current candle)
            volume_candles_5m = candles_5m[-(VOLUME_PERIOD+1):-1]
            avg_volume_5m = sum(float(c['volume']) for c in volume_candles_5m) / len(volume_candles_5m)
        else:
            current_volume_5m = 0
            avg_volume_5m = 0
        
        # Check volume confirmation
        volume_confirmed = check_volume_confirmation(cb_service, current_volume_1h, current_volume_5m, avg_volume_1h, avg_volume_5m)
        # === Additional signals per Aug 8, 2025 plan ===
        volume_1h_confirmed = current_volume_1h >= (VOLUME_SURGE_FACTOR_1H * avg_volume_1h) if avg_volume_1h > 0 else False

        # 15m candles for confirmation / acceptance
        start_15m = now - timedelta(hours=12)
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, int(start_15m.timestamp()), int(now.timestamp()), "FIFTEEN_MINUTE")
        if candles_15m and len(candles_15m) >= 3:
            candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
            last_15m_close = float(candles_15m[-1]['close'])
            prev_15m_close = float(candles_15m[-2]['close'])
        else:
            last_15m_close = current_close_1h
            prev_15m_close = current_close_1h

        # 5m latest and previous
        if candles_5m and len(candles_5m) >= 3:
            prev_candle_5m = candles_5m[-2]
            last_5m_close = float(current_candle_5m['close'])
            last_5m_open = float(current_candle_5m['open'])
            prev_5m_low = float(prev_candle_5m['low'])
            last_5m_low = float(current_candle_5m['low'])
            last_5m_vol = float(current_candle_5m['volume'])
        else:
            last_5m_close = float(current_close_1h)
            last_5m_open = last_5m_close
            prev_5m_low = last_5m_close
            last_5m_low = last_5m_close
            last_5m_vol = 0.0

        # Derived conditions
        bo_breakout_confirmed = last_15m_close > 3955.0
        wick_guard = last_5m_close > 3955.0
        hl_ok = (last_5m_low > prev_5m_low) and (last_5m_close > last_5m_open)

        # Spike into 3955-3970 in recent 5m bars
        had_spike = False
        if candles_5m:
            for c in candles_5m[-18:]:
                try:
                    h = float(c['high'])
                except Exception:
                    continue
                if 3955.0 <= h <= 3970.0:
                    had_spike = True
                    break
        reject_close_ok = (last_5m_close < 3950.0) or (last_15m_close < 3950.0)

        # Selling pressure approx: red 5m candle + volume >= 1.3x 5m SMA20 (approx using avg_volume_5m)
        sma20_prev_5m = avg_volume_5m if avg_volume_5m > 0 else None
        sell_pressure = (last_5m_close < last_5m_open) and (sma20_prev_5m is not None and last_5m_vol >= 1.3 * sma20_prev_5m)

        # Acceptance below 3900: two consecutive 15m closes below 3900
        last_two_15m_below = (last_15m_close < 3900.0) and (prev_15m_close < 3900.0)

        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — Clean Two-Sided ETH Plan for Today (Live Levels) Alert")
        logger.info("")
        logger.info("📊 Today's Levels:")
        logger.info(f"   • ETH ≈ ${current_close_1h:,.0f}")
        logger.info(f"   • HOD: ${current_hod:,.0f}")
        logger.info(f"   • LOD: ${current_lod:,.0f}")
        logger.info(f"   • MID: ${current_mid_range:,.0f}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info("   • Time-frame: 1h trigger, execute on 5–15m")
        logger.info(f"   • Volume confirm: ≥{VOLUME_SURGE_FACTOR_1H}x 20-period vol on 1h OR ≥{VOLUME_SURGE_FACTOR_5M}x 20-SMA vol on 5m")
        logger.info(f"   • Risk: size for 1R = {RISK_PERCENTAGE_LOW}-{RISK_PERCENTAGE_HIGH}% of price")
        logger.info(f"   • Take 30-50% off at +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R; trail remainder")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} margin x {LEVERAGE}x leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("Type: Breakout")
            logger.info(f"   • Entry: Buy-stop ${BREAKOUT_ENTRY_LOW:,.0f}–${BREAKOUT_ENTRY_HIGH:,.0f} (HOD + ~0.2%)")
            logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,.0f} (back inside range)")
            logger.info(f"   • TP1: ${BREAKOUT_TP1:,.0f}")
            logger.info(f"   • TP2: ${BREAKOUT_TP2_LOW:,.0f}–${BREAKOUT_TP2_HIGH:,.0f}")
            logger.info("   • Rationale: Fresh expansion beyond HOD with momentum & confirmation")
            logger.info("")
            logger.info("Type: Retest")
            logger.info(f"   • Entry: ${RETEST_ENTRY_LOW:,.0f}–${RETEST_ENTRY_HIGH:,.0f} (prior 1h pivot / VWAP)")
            logger.info(f"   • SL: ${RETEST_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${RETEST_TP1:,.0f}")
            logger.info(f"   • TP2: ${RETEST_TP2_LOW:,.0f}")
            logger.info("   • Rationale: Pullback to intraday support; hold > 20 EMA keeps bullish structure")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("Type: Breakdown")
            logger.info(f"   • Entry: Sell-stop ${BREAKDOWN_ENTRY_LOW:,.0f}–${BREAKDOWN_ENTRY_HIGH:,.0f} (LOD – ~0.2%)")
            logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${BREAKDOWN_TP1:,.0f}")
            logger.info(f"   • TP2: ${BREAKDOWN_TP2_LOW:,.0f}–${BREAKDOWN_TP2_HIGH:,.0f}")
            logger.info("   • Rationale: Range failure + fresh low; room to prior 4h demand")
            logger.info("")
            logger.info("Type: Retest")
            logger.info(f"   • Entry: ${FADE_ENTRY_LOW:,.0f}–${FADE_ENTRY_HIGH:,.0f} (retest broken 1h support)")
            logger.info(f"   • SL: ${FADE_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${FADE_TP1:,.0f}")
            logger.info(f"   • TP2: ${FADE_TP2_LOW:,.0f}")
            logger.info("   • Rationale: Acceptance below VWAP; weak bids on rebound signal continuation")
            logger.info("")
        logger.info("")
        logger.info(f"Current Price: ${current_close_1h:,.2f}")
        logger.info(f"Last 1H Close: ${current_close_1h:,.2f}, High: ${current_high_1h:,.2f}, Low: ${current_low_1h:,.2f}")
        logger.info(f"1H Volume: {current_volume_1h:,.0f}, 1H SMA: {avg_volume_1h:,.0f}, Rel_Vol: {current_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}")
        logger.info(f"5M Volume: {current_volume_5m:,.0f}, 5M SMA: {avg_volume_5m:,.0f}, Rel_Vol: {current_volume_5m/avg_volume_5m if avg_volume_5m > 0 else 0:.2f}")
        logger.info(f"Volume Confirmed: {'✅' if volume_confirmed else '❌'}")
        logger.info("")
        
        # Execution guardrails: Pick one path (breakout or breakdown)
        # Don't run both. If trigger fires without volume confirmation, pass or halve size.
        # If first entry stops, don't re-enter until a fresh 1h structure forms.
        price_position_in_range = (current_close_1h - current_lod) / current_range_width if current_range_width > 0 else 0.5
        logger.info(f"Price position in range: {price_position_in_range:.2%} (0% = LOD, 100% = HOD)")
        
        # Determine which path to prioritize based on direction filter and one-side management
        if direction == 'LONG':
            logger.info("🎯 Direction filter: LONG only - prioritizing breakout strategies")
            breakdown_priority = False
            breakout_priority = True
        elif direction == 'SHORT':
            logger.info("🎯 Direction filter: SHORT only - prioritizing breakdown strategies")
            breakdown_priority = True
            breakout_priority = False
        else:  # BOTH - use one-side management rule
            # Check if any long strategy is already triggered
            long_triggered = breakout_state.get("triggered", False) or retest_state.get("triggered", False)
            short_triggered = breakdown_state.get("triggered", False) or fade_state.get("triggered", False)
            
            if long_triggered:
                logger.info("🎯 One-side management: LONG strategy already triggered - prioritizing LONG")
                breakdown_priority = False
                breakout_priority = True
            elif short_triggered:
                logger.info("🎯 One-side management: SHORT strategy already triggered - prioritizing SHORT")
                breakdown_priority = True
                breakout_priority = False
            else:
                # No strategy triggered yet - use price position relative to day mid
                below_day_mid = current_close_1h < current_mid_range
                if below_day_mid:
                    logger.info(f"🎯 One-side management: Price (${current_close_1h:,.2f}) below day mid (${current_mid_range:,.2f}) = SHORT priority")
                    breakdown_priority = True
                    breakout_priority = False
                else:
                    logger.info(f"🎯 One-side management: Price (${current_close_1h:,.2f}) above day mid (${current_mid_range:,.2f}) = LONG priority")
                    breakdown_priority = False
                    breakout_priority = True
        
        # LONG (breakout) Strategy Conditions
        breakout_condition = (
            breakout_priority and
            current_close_1h >= BREAKOUT_ENTRY_LOW and 
            current_close_1h <= BREAKOUT_ENTRY_HIGH and 
            volume_confirmed and 
            not breakout_state.get("triggered", False) and
            not breakout_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # LONG (retest) Strategy Conditions
        retest_condition = (
            breakout_priority and
            current_close_1h >= RETEST_ENTRY_LOW and 
            current_close_1h <= RETEST_ENTRY_HIGH and 
            volume_confirmed and 
            not retest_state.get("triggered", False) and
            not retest_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # SHORT (breakdown) Strategy Conditions
        breakdown_condition = (
            breakdown_priority and
            current_close_1h <= BREAKDOWN_ENTRY_HIGH and 
            current_close_1h >= BREAKDOWN_ENTRY_LOW and 
            volume_confirmed and 
            not breakdown_state.get("triggered", False) and
            not breakdown_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # SHORT (retest) Strategy Conditions
        fade_condition = (
            breakdown_priority and
            current_close_1h <= FADE_ENTRY_HIGH and 
            current_close_1h >= FADE_ENTRY_LOW and 
            volume_confirmed and 
            not fade_state.get("triggered", False) and
            not fade_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy
        if long_strategies_enabled and not breakout_state.get("triggered", False) and not breakout_state.get("stopped_out", False):
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_close_1h <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and breakout_priority and bo_breakout_confirmed and wick_guard and volume_1h_confirmed
            
            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKOUT_ENTRY_LOW:,.0f}-${BREAKOUT_ENTRY_HIGH:,.0f}): {'✅' if in_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H: {current_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x, 5M: {current_volume_5m/avg_volume_5m if avg_volume_5m > 0 else 0:.2f}x): {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Strategy priority: {'✅' if breakout_priority else '❌'}")
            logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")
            
            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakout trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD LONG Breakout",
                    entry_price=current_close_1h,
                    stop_loss=BREAKOUT_STOP_LOSS,
                    take_profit=BREAKOUT_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 LONG - Breakout trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${BREAKOUT_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${BREAKOUT_TP1:,.2f}")
                    logger.info(f"TP2: ${BREAKOUT_TP2_LOW:,.2f}-${BREAKOUT_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Fresh expansion beyond HOD with momentum & confirmation")
                    
                    # Save trigger state
                    breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout trade failed: {trade_result}")
        
        # 2. LONG - Retest Strategy
        if not trade_executed and long_strategies_enabled and not retest_state.get("triggered", False) and not retest_state.get("stopped_out", False):
            in_retest_zone = RETEST_ENTRY_LOW <= current_close_1h <= RETEST_ENTRY_HIGH
            retest_ready = in_retest_zone and breakout_priority and hl_ok
            
            logger.info("🔍 LONG - Retest Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,.0f}-${RETEST_ENTRY_HIGH:,.0f}): {'✅' if in_retest_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Strategy priority: {'✅' if breakout_priority else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if retest_ready else '⏳ NO'}")
            
            if retest_ready:
                logger.info("")
                logger.info("🎯 LONG - Retest Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD LONG Retest",
                    entry_price=current_close_1h,
                    stop_loss=RETEST_STOP_LOSS,
                    take_profit=RETEST_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 LONG - Retest trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${RETEST_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${RETEST_TP1:,.2f}")
                    logger.info(f"TP2: ${RETEST_TP2_LOW:,.2f}")
                    logger.info("Strategy: Pullback to intraday support; hold > 20 EMA keeps bullish structure")
                    
                    # Save trigger state
                    retest_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Retest trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if not trade_executed and short_strategies_enabled and not breakdown_state.get("triggered", False) and not breakdown_state.get("stopped_out", False):
            in_breakdown_zone = BREAKDOWN_ENTRY_LOW <= current_close_1h <= BREAKDOWN_ENTRY_HIGH
            breakdown_ready = in_breakdown_zone and breakdown_priority and last_two_15m_below and volume_1h_confirmed
            
            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKDOWN_ENTRY_LOW:,.0f}-${BREAKDOWN_ENTRY_HIGH:,.0f}): {'✅' if in_breakdown_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")
            
            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Breakdown",
                    entry_price=current_close_1h,
                    stop_loss=BREAKDOWN_STOP_LOSS,
                    take_profit=BREAKDOWN_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Breakdown trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${BREAKDOWN_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${BREAKDOWN_TP1:,.2f}")
                    logger.info(f"TP2: ${BREAKDOWN_TP2_LOW:,.2f}-${BREAKDOWN_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Range failure + fresh low; room to prior 4h demand")
                    
                    # Save trigger state
                    breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(breakdown_state, BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown trade failed: {trade_result}")
        
        # 4. SHORT - Retest Strategy
        if not trade_executed and short_strategies_enabled and not fade_state.get("triggered", False) and not fade_state.get("stopped_out", False):
            in_fade_zone = FADE_ENTRY_LOW <= current_close_1h <= FADE_ENTRY_HIGH
            fade_ready = in_fade_zone and breakdown_priority and had_spike and reject_close_ok and sell_pressure
            
            logger.info("🔍 SHORT - Retest Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${FADE_ENTRY_LOW:,.0f}-${FADE_ENTRY_HIGH:,.0f}): {'✅' if in_fade_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")
            
            if fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Retest Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Retest",
                    entry_price=current_close_1h,
                    stop_loss=FADE_STOP_LOSS,
                    take_profit=FADE_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Retest trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${FADE_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${FADE_TP1:,.2f}")
                    logger.info(f"TP2: ${FADE_TP2_LOW:,.2f}")
                    logger.info("Strategy: Acceptance below VWAP; weak bids on rebound signal continuation")
                    
                    # Save trigger state
                    fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(fade_state, FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Retest trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for strategy conditions...")
            if direction != 'BOTH':
                logger.info(f"   Direction filter: {direction} only")
            if not volume_confirmed:
                logger.info("   Volume confirmation not met")
            
            if long_strategies_enabled:
                if breakout_state.get("triggered", False):
                    logger.info("   Breakout strategy already triggered")
                if retest_state.get("triggered", False):
                    logger.info("   Retest strategy already triggered")
            
            if short_strategies_enabled:
                if breakdown_state.get("triggered", False):
                    logger.info("   Breakdown strategy already triggered")
                if fade_state.get("triggered", False):
                    logger.info("   Retest strategy already triggered")
        
        # Reset triggers if price moves significantly away from entry zones
        # Execution guardrails: If first entry stops, stand down until new 1h structure forms
        if breakout_state.get("triggered", False):
            if current_close_1h < BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Breakout trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 1h structure forms")
                breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                logger.info("Breakout trigger state reset - standing down")
        
        if retest_state.get("triggered", False):
            if current_close_1h < RETEST_STOP_LOSS:
                logger.info("🔄 Resetting Retest trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 1h structure forms")
                retest_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                logger.info("Retest trigger state reset - standing down")
        
        if breakdown_state.get("triggered", False):
            if current_close_1h > BREAKDOWN_STOP_LOSS:
                logger.info("🔄 Resetting Breakdown trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 1h structure forms")
                breakdown_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(breakdown_state, BREAKDOWN_TRIGGER_FILE)
                logger.info("Breakdown trigger state reset - standing down")
        
        if fade_state.get("triggered", False):
            if current_close_1h > FADE_STOP_LOSS:
                logger.info("🔄 Resetting Retest trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 1h structure forms")
                fade_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(fade_state, FADE_TRIGGER_FILE)
                logger.info("Retest trigger state reset - standing down")
        
        logger.info("=== ETH-USD Trading Strategy Alert completed ===")
        return current_ts_1h
        
    except Exception as e:
        logger.error(f"Error in ETH-USD Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH-USD Trading Strategy Alert completed (with error) ===")
    return last_alert_ts



# --- ETH Intraday Plan (Aug 9, 2025) ---
def eth_intraday_plan_alert(cb_service, last_alert_ts=None, max_trades_per_day: int = 1):
    """
    ETH intraday plan per Aug 9, 2025:
    - Trigger/Acceptance on 15m with 1h VWAP filter
    - Volume: 15m volume >= 1.25x SMA20
    - Entry: limit on retest of level ±$3; else market on new 15m HH/LL within 30m
    - SL: trigger bar extreme ± 0.5× ATR(14, 15m) with floor/ceiling 0.8%
    - TP: 1.2R (primary bracket). Log 2.2R and 21-EMA trailing plan
    - Guards: max trades/day, chop filter, news-gap safety, 45m no-fill timeout
    - Position size: $5,000 notional (250 × 20)
    """

    def candles_to_df(candles_list):
        if not candles_list:
            return pd.DataFrame(columns=["start", "open", "high", "low", "close", "volume"])
        rows = []
        for c in candles_list:
            try:
                rows.append({
                    'start': int(c['start']),
                    'open': float(c['open']),
                    'high': float(c['high']),
                    'low': float(c['low']),
                    'close': float(c['close']),
                    'volume': float(c['volume'])
                })
            except Exception:
                continue
        return pd.DataFrame(rows).sort_values('start').reset_index(drop=True)

    def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
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
        return float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    def ema(series: pd.Series, length: int) -> float:
        if series is None or len(series) < length:
            return 0.0
        return float(series.ewm(span=length, adjust=False).mean().iloc[-1])

    def compute_vwap(df: pd.DataFrame) -> float:
        if df is None or df.empty:
            return 0.0
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        pv = tp * df['volume']
        total_vol = df['volume'].sum()
        return float(pv.sum() / total_vol) if total_vol > 0 else 0.0

    def get_min_base_size(product_id: str) -> float:
        return {'ETH-PERP-INTX': 0.001}.get(product_id, 0.001)

    def calc_base_size(product_id: str, notional_usd: float, px: float) -> float:
        min_sz = get_min_base_size(product_id)
        raw = notional_usd / max(px, 1e-9)
        decimals = len(str(min_sz).split('.')[-1])
        size = round(raw, decimals)
        return max(size, min_sz)

    # --- State ---
    state_file = "eth_intraday_trigger_state.json"
    state = load_trigger_state(state_file)
    now = datetime.now(UTC)

    # Reset daily counter
    today_str = now.strftime('%Y-%m-%d')
    if state.get('last_trade_date') != today_str:
        state['trades_today'] = 0
        state['last_trade_date'] = today_str
        state['active_side'] = None
        state['awaiting_acceptance'] = False
        state['trigger_info'] = None
        state['accepted_info'] = None
        state['entry_done'] = False
        save_trigger_state(state, state_file)

    if int(state.get('trades_today', 0)) >= max_trades_per_day:
        logger.info(f"⏹ Max trades for the day reached ({max_trades_per_day}).")
        return last_alert_ts

    try:
        # 15m for trigger/volume/ATR
        start_15m = now - timedelta(hours=48)
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, int(start_15m.timestamp()), int(now.timestamp()), GRANULARITY_15M)
        if not candles_15m or len(candles_15m) < 40:
            logger.warning("Not enough 15m data")
            return last_alert_ts
        df15 = candles_to_df(candles_15m)

        # 5m for 1h VWAP (12 bars)
        start_5m = now - timedelta(hours=1, minutes=5)
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, int(start_5m.timestamp()), int(now.timestamp()), GRANULARITY_5M)
        df5 = candles_to_df(candles_5m)

        current_price = float(df15['close'].iloc[-1])

        # Volume SMA20 on 15m (use last completed bar for SMA)
        df15['vol_sma20'] = df15['volume'].rolling(window=20).mean()
        last_vol_15m = float(df15['volume'].iloc[-1])
        vol_sma20_prev = float(df15['vol_sma20'].iloc[-2]) if len(df15) >= 22 else 0.0
        vol_ok = (last_vol_15m >= 1.25 * vol_sma20_prev) if vol_sma20_prev > 0 else False

        # ATR and ATR%
        atr_val = compute_atr(df15)
        atr_pct = (atr_val / current_price) * 100 if current_price > 0 else 0.0

        # 1h VWAP from last 12×5m
        vwap_1h = compute_vwap(df5.tail(12))

        # 21-EMA(15m) for trailing plan (logged only)
        ema21_15m = ema(df15['close'], 21)

        # Last completed and current 15m bars
        cprev = df15.iloc[-2]
        clast = df15.iloc[-1]

        LONG_LEVEL = 4095.0
        SHORT_LEVEL = 3881.0
        BAND = 3.0

        long_trigger = clast['close'] > LONG_LEVEL
        short_trigger = clast['close'] < SHORT_LEVEL

        # Chop filter at trigger
        chop = (atr_pct < 0.25) and (abs(clast['close'] - vwap_1h) / clast['close'] < 0.001)

        trg = state.get('trigger_info')
        acc = state.get('accepted_info')

        # Record trigger
        if state.get('active_side') is None and not state.get('awaiting_acceptance', False):
            if long_trigger and vol_ok and not chop:
                state['active_side'] = 'LONG'
                state['awaiting_acceptance'] = True
                state['trigger_info'] = {
                    'level': LONG_LEVEL,
                    'bar_close': float(clast['close']),
                    'bar_low': float(clast['low']),
                    'bar_high': float(clast['high']),
                    'bar_start': int(clast['start'])
                }
                save_trigger_state(state, state_file)
                logger.info("🔔 LONG trigger fired (15m close > 4095). Waiting for acceptance…")
                play_alert_sound()
            elif short_trigger and vol_ok and not chop:
                state['active_side'] = 'SHORT'
                state['awaiting_acceptance'] = True
                state['trigger_info'] = {
                    'level': SHORT_LEVEL,
                    'bar_close': float(clast['close']),
                    'bar_low': float(clast['low']),
                    'bar_high': float(clast['high']),
                    'bar_start': int(clast['start'])
                }
                save_trigger_state(state, state_file)
                logger.info("🔔 SHORT trigger fired (15m close < 3881). Waiting for acceptance…")
                play_alert_sound()

        # Acceptance check on next bar
        if state.get('awaiting_acceptance', False) and trg is None:
            trg = state.get('trigger_info')

        if state.get('awaiting_acceptance', False) and trg is not None and acc is None:
            side = state.get('active_side')
            accept_ok = False
            trigger_start = trg['bar_start']
            idx_list = df15.index[df15['start'] == trigger_start].tolist()
            if idx_list:
                i = idx_list[0]
                if i + 1 < len(df15):
                    acc_bar = df15.iloc[i + 1]
                    gap = abs(float(acc_bar['open']) - float(trg['bar_close'])) / float(trg['bar_close']) if trg['bar_close'] > 0 else 0.0
                    if gap > 0.007:
                        logger.info("⛔ News-gap safety tripped (>0.7%). Standing down.")
                        state['active_side'] = None
                        state['awaiting_acceptance'] = False
                        state['trigger_info'] = None
                        save_trigger_state(state, state_file)
                        return last_alert_ts
                    if side == 'LONG':
                        accept_ok = (float(acc_bar['close']) > LONG_LEVEL) and (current_price > vwap_1h)
                    else:
                        accept_ok = (float(acc_bar['close']) < SHORT_LEVEL) and (current_price < vwap_1h)
                    if accept_ok:
                        state['accepted_info'] = {
                            'bar_start': int(acc_bar['start']),
                            'bar_high': float(acc_bar['high']),
                            'bar_low': float(acc_bar['low']),
                            'accepted_at': int(now.timestamp())
                        }
                        state['awaiting_acceptance'] = False
                        save_trigger_state(state, state_file)
                        logger.info(f"✅ {side} acceptance confirmed; 1h VWAP filter passed. Watching for retest or HH/LL…")
                        play_alert_sound()

        # Entry logic after acceptance
        acc = state.get('accepted_info')
        if acc is not None and state.get('active_side') in ['LONG', 'SHORT'] and not state.get('entry_done', False):
            side = state['active_side']
            level = LONG_LEVEL if side == 'LONG' else SHORT_LEVEL
            lower = level - BAND
            upper = level + BAND
            in_band = (lower <= float(clast['low']) <= upper) or (lower <= float(clast['high']) <= upper)
            accept_time = datetime.fromtimestamp(int(acc['accepted_at']), UTC)
            timeout_30m = now >= (accept_time + timedelta(minutes=30))
            hh_ll = (side == 'LONG' and float(clast['high']) > float(acc['bar_high'])) or \
                    (side == 'SHORT' and float(clast['low']) < float(acc['bar_low']))

            # Compute SL/TP using trigger extremes and ATR
            trigger_low = float(state['trigger_info']['bar_low']) if state.get('trigger_info') else float(cprev['low'])
            trigger_high = float(state['trigger_info']['bar_high']) if state.get('trigger_info') else float(cprev['high'])
            entry_ref = level if in_band and not timeout_30m else current_price
            if side == 'LONG':
                sl_raw = trigger_low - 0.5 * atr_val
                sl_floor = entry_ref * (1 - 0.008)
                sl = min(sl_raw, sl_floor)
                r = entry_ref - sl
                tp1 = round(entry_ref + 1.2 * r, 2)
            else:
                sl_raw = trigger_high + 0.5 * atr_val
                sl_ceiling = entry_ref * (1 + 0.008)
                sl = max(sl_raw, sl_ceiling)
                r = sl - entry_ref
                tp1 = round(entry_ref - 1.2 * r, 2)
            sl = round(sl, 2)

            # Entry preference: limit on retest; else market on HH/LL within 30m
            if in_band and not timeout_30m:
                limit_price = round(level, 2)
                base_sz = calc_base_size(PRODUCT_ID, POSITION_SIZE_USD, limit_price)
                logger.info(f"🎯 {side} retest entry via limit: level=${limit_price:.2f} | SL=${sl:.2f} | TP1=${tp1:.2f} | ATR%={atr_pct:.2f}% | VWAP1h=${vwap_1h:.2f}")
                try:
                    play_alert_sound()
                except Exception:
                    pass
                try:
                    res = cb_service.place_limit_order_with_targets(
                        product_id=PRODUCT_ID,
                        side='BUY' if side == 'LONG' else 'SELL',
                        size=base_sz,
                        entry_price=limit_price,
                        take_profit_price=tp1,
                        stop_loss_price=sl,
                        leverage=str(LEVERAGE)
                    )
                    if 'error' in res:
                        logger.error(f"❌ Limit order error: {res['error']}")
                    else:
                        logger.info("⏳ Monitoring limit for fill up to 45m and placing brackets if needed…")
                        mon = cb_service.monitor_limit_order_and_place_bracket(
                            product_id=PRODUCT_ID,
                            order_id=res.get('order_id'),
                            size=base_sz,
                            take_profit_price=tp1,
                            stop_loss_price=sl,
                            leverage=str(LEVERAGE),
                            max_wait_time=2700
                        )
                        if mon.get('status') == 'success':
                            state['entry_done'] = True
                            state['trades_today'] = int(state.get('trades_today', 0)) + 1
                            save_trigger_state(state, state_file)
                            logger.info("✅ Limit filled and bracket placed")
                except Exception as e:
                    logger.error(f"Limit entry failed: {e}")
            elif timeout_30m and hh_ll:
                entry_price = current_price
                base_sz = calc_base_size(PRODUCT_ID, POSITION_SIZE_USD, entry_price)
                logger.info(f"🎯 {side} market entry on HH/LL: entry≈${entry_price:.2f} | SL=${sl:.2f} | TP1=${tp1:.2f}")
                try:
                    play_alert_sound()
                except Exception:
                    pass
                try:
                    result = cb_service.place_market_order_with_targets(
                        product_id=PRODUCT_ID,
                        side='BUY' if side == 'LONG' else 'SELL',
                        size=base_sz,
                        take_profit_price=tp1,
                        stop_loss_price=sl,
                        leverage=str(LEVERAGE)
                    )
                    if 'error' in result:
                        logger.error(f"❌ Order error: {result['error']}")
                    else:
                        logger.info("✅ Order placed with TP1/SL bracket")
                        state['entry_done'] = True
                        state['trades_today'] = int(state.get('trades_today', 0)) + 1
                        save_trigger_state(state, state_file)
                except Exception as e:
                    logger.error(f"Market entry failed: {e}")

            # No-fill timeout after acceptance
            if now >= (datetime.fromtimestamp(int(acc['accepted_at']), UTC) + timedelta(minutes=45)) and not state.get('entry_done'):
                logger.info("⏹ No-fill timeout (45m) after acceptance — standing down.")
                state['active_side'] = None
                state['trigger_info'] = None
                state['accepted_info'] = None
                state['awaiting_acceptance'] = False
                save_trigger_state(state, state_file)

        logger.info("=== ETH intraday plan check completed ===")
        return now
    except Exception as e:
        logger.error(f"Error in ETH intraday plan: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return last_alert_ts
# Replace main loop to use new alert
def main():
    parser = argparse.ArgumentParser(description='ETH Intraday Plan Monitor')
    parser.add_argument('--max-trades', type=int, default=1, help='Max trades per day (default: 1)')
    args = parser.parse_args()

    logger.info("Starting ETH Intraday Plan Monitor (15m triggers, 1h VWAP acceptance, retest-or-market entries)")
    logger.info(f"Max trades today: {args.max_trades}")
    
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    
    cb_service = setup_coinbase()
    last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5

    def poll_iteration():
        nonlocal last_alert_ts, consecutive_failures
        iteration_start_time = time.time()
        last_alert_ts = eth_intraday_plan_alert(cb_service, last_alert_ts, max_trades_per_day=args.max_trades)
        consecutive_failures = 0
        logger.info(f"✅ ETH alert cycle completed in {time.time() - iteration_start_time:.1f} seconds")
    
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
                logger.error("❌ Too many consecutive connection failures. Attempting to reconnect...")
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