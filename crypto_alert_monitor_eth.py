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

# ETH Trading Strategy Parameters (based on new clean two-sided plan)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_1H = "ONE_HOUR"  # 1-hour chart for context
GRANULARITY_5M = "FIVE_MINUTE"  # 5-minute chart for execution
GRANULARITY_15M = "FIFTEEN_MINUTE"  # 15-minute chart for execution
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (ETH ≈ $4,300; HOD $4,351; LOD $4,174)
CURRENT_ETH_PRICE = 4300.00
HOD_24H = 4351.00  # 24h High of day
LOD_24H = 4174.00  # 24h Low of day
RANGE_WIDTH_24H = HOD_24H - LOD_24H  # 177 points
MID_RANGE_PIVOT = (HOD_24H + LOD_24H) / 2  # 4262.50

# LONG (breakout) Strategy Parameters - Entry: 1h/5–15m close above 4,365–4,372 (HOD + ~0.3%)
BREAKOUT_ENTRY_LOW = 4365  # HOD + 14
BREAKOUT_ENTRY_HIGH = 4372  # HOD + 21
BREAKOUT_STOP_LOSS = 4322  # ≈1R below ~4,366
BREAKOUT_TP1 = 4410  # TP1: 4,410–4,431
BREAKOUT_TP2_LOW = 4431  # TP2: 4,462–4,475
BREAKOUT_TP2_HIGH = 4475

# LONG (retest) Strategy Parameters - After confirmed break/close >4,365, buy reclaim of 4,350–4,359
RECLAIM_ENTRY_LOW = 4350  # Reclaim zone after confirmed break
RECLAIM_ENTRY_HIGH = 4359
RECLAIM_STOP_LOSS = 4325  # Below reclaimed zone
RECLAIM_TP1 = 4410  # TP1
RECLAIM_TP2_LOW = 4460  # TP2: 4,460–4,480
RECLAIM_TP2_HIGH = 4480

# SHORT (breakdown) Strategy Parameters - Entry: 1h/5–15m close below 4,158–4,163 (LOD − ~0.3%)
BREAKDOWN_ENTRY_LOW = 4158  # LOD - 16
BREAKDOWN_ENTRY_HIGH = 4163  # LOD - 11
BREAKDOWN_STOP_LOSS = 4203  # ≈1R above ~4,161
BREAKDOWN_TP1 = 4119  # TP1: 4,119–4,099
BREAKDOWN_TP2_LOW = 4099  # TP2: 4,069–4,057
BREAKDOWN_TP2_HIGH = 4057

# SHORT (failed-retest) Strategy Parameters - After confirmed break <4,163, sell rejection at 4,168–4,175
RETEST_ENTRY_LOW = 4168  # Prior low rejection
RETEST_ENTRY_HIGH = 4175
RETEST_STOP_LOSS = 4205  # SL
RETEST_TP1 = 4120  # TP1
RETEST_TP2_LOW = 4070  # TP2: 4,070–4,050
RETEST_TP2_HIGH = 4050

# Volume confirmation requirements
VOLUME_SURGE_FACTOR_1H = 1.25  # ≥1.25× 20-period vol on 1h
VOLUME_SURGE_FACTOR_5M = 2.0   # ≥2× 20-SMA vol on 5m at trigger

# Risk management
RISK_PERCENTAGE_LOW = 1.0  # 1R ≈ 1.0% of entry
RISK_PERCENTAGE_HIGH = 1.0
PARTIAL_PROFIT_RANGE_LOW = 1.0  # Partial at +1.0R
PARTIAL_PROFIT_RANGE_HIGH = 1.5  # Partial at +1.5R

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# Execution guardrails
MAX_TRADES_PER_DAY = 2  # Max 2 trades/day
COOLDOWN_MINUTES = 30  # 30 min after a stop
MODE = "FAST"  # FAST = 5–15m close beyond trigger; CONSERVATIVE = 1h close
VCONF = True  # enforce volume rule

# Chop filter parameters
ATR_PERCENTAGE_THRESHOLD = 0.4  # Skip if 1h ATR% < 0.4
VOLUME_CHOP_FACTOR = 0.8  # and 5m vol < 0.8× average (chop filter)

# State files for each strategy
BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
RECLAIM_TRIGGER_FILE = "eth_reclaim_trigger_state.json"
BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"
RETEST_TRIGGER_FILE = "eth_retest_trigger_state.json"

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
        
        # Apply execution guardrails: halve size if volume confirmation not met
        if not volume_confirmed:
            position_size_usd = POSITION_SIZE_USD // 2  # Halve the position size
            logger.warning(f"⚠️ Volume confirmation not met - halving position size to ${position_size_usd:,} USD")
        else:
            position_size_usd = POSITION_SIZE_USD  # Use the full position size
        
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
    
    logger.info(f"Volume confirmation check:")
    logger.info(f"  1H: {current_volume_1h:,.0f} vs {VOLUME_SURGE_FACTOR_1H}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_1h_confirmed else '❌'}")
    logger.info(f"  5M: {current_volume_5m:,.0f} vs {VOLUME_SURGE_FACTOR_5M}x avg ({avg_volume_5m:,.0f}) -> {'✅' if volume_5m_confirmed else '❌'}")
    logger.info(f"  Overall: {'✅' if volume_confirmed else '❌'}")
    
    return volume_confirmed



def get_rolling_24h_hod_lod(cb_service, current_ts):
    """Get rolling 24h HOD and LOD"""
    try:
        # Get rolling 24h candles
        start_ts = int((current_ts - timedelta(hours=24)).timestamp())
        end_ts = int(current_ts.timestamp())
        
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_1H)
        
        if not candles_1h:
            return HOD_24H, LOD_24H  # Use default values if no data
        
        # Calculate rolling 24h HOD and LOD
        hod = max(float(candle['high']) for candle in candles_1h)
        lod = min(float(candle['low']) for candle in candles_1h)
        
        logger.info(f"Rolling 24h HOD: ${hod:,.2f}, LOD: ${lod:,.2f}")
        return hod, lod
        
    except Exception as e:
        logger.error(f"Error getting rolling 24h HOD/LOD: {e}")
        return HOD_24H, LOD_24H  # Use default values

def check_new_structure_formation(cb_service, current_ts, previous_hod, previous_lod):
    """Check if a new 24h structure has formed (new HOD or LOD)"""
    try:
        current_hod, current_lod = get_rolling_24h_hod_lod(cb_service, current_ts)
        
        # Check if we have new highs or lows
        new_hod = current_hod > previous_hod
        new_lod = current_lod < previous_lod
        
        if new_hod or new_lod:
            logger.info(f"🔄 New 24h structure detected: {'HOD' if new_hod else ''}{' and ' if new_hod and new_lod else ''}{'LOD' if new_lod else ''}")
            logger.info(f"Previous HOD: ${previous_hod:,.2f} -> Current HOD: ${current_hod:,.2f}")
            logger.info(f"Previous LOD: ${previous_lod:,.2f} -> Current LOD: ${current_lod:,.2f}")
            return True, current_hod, current_lod
        
        return False, current_hod, current_lod
        
    except Exception as e:
        logger.error(f"Error checking new structure formation: {e}")
        return False, previous_hod, previous_lod

def check_confirmed_break_and_reclaim(cb_service, current_ts):
    """
    Check for confirmed break/close >4,365 then reclaim of 4,350–4,359 on 5–15m with volume confirm
    Required for LONG retest strategy
    """
    try:
        # Get recent 1-hour, 5-minute and 15-minute candles
        end = current_ts
        start = current_ts - timedelta(hours=4)  # Check last 4 hours
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_1H)
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_15M)
        
        if not candles_1h or len(candles_1h) < 2:
            return False, "Not enough 1h data"
        
        candles_1h = sorted(candles_1h, key=lambda x: int(x['start']))
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
        
        # Check for confirmed break/close >4,365 on 1h
        break_level = 4365
        reclaim_level_low = 4350
        reclaim_level_high = 4359
        
        break_confirmed = False
        reclaim_detected = False
        
        # Check 1h timeframe for confirmed break
        for candle in candles_1h[-3:]:  # Check last 3 1h candles
            close = float(candle['close'])
            if close > break_level and not break_confirmed:
                break_confirmed = True
                logger.info(f"Break confirmed: 1h close ${close:,.2f} > ${break_level:,.2f}")
        
        # Check for reclaim of 4,350–4,359 after confirmed break
        if break_confirmed:
            for candle in candles_5m[-20:]:  # Check last 20 5m candles
                close = float(candle['close'])
                
                # Check for reclaim (close in 4350-4359 range) after break
                if close >= reclaim_level_low and close <= reclaim_level_high and not reclaim_detected:
                    reclaim_detected = True
                    logger.info(f"Reclaim detected: 5m close ${close:,.2f} in range ${reclaim_level_low:,.2f}-${reclaim_level_high:,.2f}")
            
            # Also check 15m timeframe
            for candle in candles_15m[-8:]:  # Check last 8 15m candles
                close = float(candle['close'])
                
                # Check for reclaim (close in 4350-4359 range) after break
                if close >= reclaim_level_low and close <= reclaim_level_high and not reclaim_detected:
                    reclaim_detected = True
                    logger.info(f"Reclaim detected: 15m close ${close:,.2f} in range ${reclaim_level_low:,.2f}-${reclaim_level_high:,.2f}")
        
        break_and_reclaim = break_confirmed and reclaim_detected
        
        logger.info(f"Break and reclaim check: Break={'✅' if break_confirmed else '❌'}, Reclaim={'✅' if reclaim_detected else '❌'}")
        logger.info(f"Overall: {'✅' if break_and_reclaim else '❌'}")
        
        return break_and_reclaim, f"Break: {break_confirmed}, Reclaim: {reclaim_detected}"
        
    except Exception as e:
        logger.error(f"Error checking break and reclaim: {e}")
        return False, str(e)

def check_confirmed_break_and_failed_retest(cb_service, current_ts):
    """
    Check for confirmed break <4,163 then rejection at 4,168–4,175 (prior low)
    Required for SHORT failed-retest strategy
    """
    try:
        # Get recent 1-hour, 5-minute and 15-minute candles
        end = current_ts
        start = current_ts - timedelta(hours=4)  # Check last 4 hours
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles_1h = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_1H)
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_15M)
        
        if not candles_1h or len(candles_1h) < 2:
            return False, "Not enough 1h data"
        
        candles_1h = sorted(candles_1h, key=lambda x: int(x['start']))
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
        
        # Check for confirmed break <4,163 on 1h
        break_level = 4163
        rejection_level_low = 4168
        rejection_level_high = 4175
        
        break_confirmed = False
        rejection_detected = False
        
        # Check 1h timeframe for confirmed break
        for candle in candles_1h[-3:]:  # Check last 3 1h candles
            close = float(candle['close'])
            if close < break_level and not break_confirmed:
                break_confirmed = True
                logger.info(f"Break confirmed: 1h close ${close:,.2f} < ${break_level:,.2f}")
        
        # Check for rejection at 4,168–4,175 after confirmed break
        if break_confirmed:
            for candle in candles_5m[-20:]:  # Check last 20 5m candles
                high = float(candle['high'])
                close = float(candle['close'])
                
                # Check for rejection (high reaches 4168-4175 but close below rejection zone)
                if high >= rejection_level_low and high <= rejection_level_high and close < rejection_level_low and not rejection_detected:
                    rejection_detected = True
                    logger.info(f"Rejection detected: 5m high ${high:,.2f} in range ${rejection_level_low:,.2f}-${rejection_level_high:,.2f}, close ${close:,.2f} < ${rejection_level_low:,.2f}")
            
            # Also check 15m timeframe
            for candle in candles_15m[-8:]:  # Check last 8 15m candles
                high = float(candle['high'])
                close = float(candle['close'])
                
                # Check for rejection (high reaches 4168-4175 but close below rejection zone)
                if high >= rejection_level_low and high <= rejection_level_high and close < rejection_level_low and not rejection_detected:
                    rejection_detected = True
                    logger.info(f"Rejection detected: 15m high ${high:,.2f} in range ${rejection_level_low:,.2f}-${rejection_level_high:,.2f}, close ${close:,.2f} < ${rejection_level_low:,.2f}")
        
        break_and_failed_retest = break_confirmed and rejection_detected
        
        logger.info(f"Break and failed retest check: Break={'✅' if break_confirmed else '❌'}, Rejection={'✅' if rejection_detected else '❌'}")
        logger.info(f"Overall: {'✅' if break_and_failed_retest else '❌'}")
        
        return break_and_failed_retest, f"Break: {break_confirmed}, Rejection: {rejection_detected}"
        
    except Exception as e:
        logger.error(f"Error checking break and failed retest: {e}")
        return False, str(e)



# --- ETH Trading Strategy Alert Logic ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH-USD Trading Strategy Alert - Implements clean two-sided ETH plan based on live levels
    Based on the trading plan: "Spiros — here's a clean, two-sided ETH plan for today based on live levels"
    
    Rules (both directions):
    - Timeframe: trigger on 1h; execute on 5–15m close
    - Volume confirm: ≥1.25× 20-period vol on 1h or ≥2× 20-SMA vol on 5m at trigger
    - Orders: market-only brackets; size so 1R ≈ 1.0% of entry; partial at +1.0–1.5R; trail if momentum persists
    - Position Size: Always margin x leverage = 250 x 20 = $5,000 USD
    - Skip if 1h ATR% < 0.4 and 5m vol < 0.8× average (chop filter)
    - If trigger whipsaws and closes back inside the level within 15m, stand down for 30–60m
    
    LONG SETUPS:
    - Breakout: 1h/5–15m close above 4,365–4,372 (HOD + ~0.3%), SL 4,322, TP1 4,410–4,431, TP2 4,462–4,475
    - Retest: After confirmed break/close >4,365, buy reclaim of 4,350–4,359, SL 4,325, TP1 4,410, TP2 4,460–4,480
    
    SHORT SETUPS:
    - Breakdown: 1h/5–15m close below 4,158–4,163 (LOD − ~0.3%), SL 4,203, TP1 4,119–4,099, TP2 4,069–4,057
    - Failed-retest: After confirmed break <4,163, sell rejection at 4,168–4,175, SL 4,205, TP1 4,120, TP2 4,070–4,050
    
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
    reclaim_state = load_trigger_state(RECLAIM_TRIGGER_FILE)
    breakdown_state = load_trigger_state(BREAKDOWN_TRIGGER_FILE)
    retest_state = load_trigger_state(RETEST_TRIGGER_FILE)
    
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
        
        # Get rolling 24h HOD and LOD
        current_hod, current_lod = get_rolling_24h_hod_lod(cb_service, current_ts_1h)
        
        # Calculate current range context
        current_range_width = current_hod - current_lod
        current_mid_range = (current_hod + current_lod) / 2
        
        logger.info(f"📊 Rolling 24h range context: ${current_lod:,.2f}-${current_hod:,.2f} (width ≈ {current_range_width:.0f})")
        logger.info(f"📊 Current mid-range pivot: ${current_mid_range:,.2f}")
        logger.info(f"📊 Rolling 24h HOD: ${current_hod:,.2f}, LOD: ${current_lod:,.2f}")
        
        # Check for new structure formation and reset stopped_out flags if needed
        # Get previous HOD/LOD from state files or use current values
        previous_hod = max(
            breakout_state.get("last_hod", current_hod),
            reclaim_state.get("last_hod", current_hod),
            breakdown_state.get("last_hod", current_hod),
            retest_state.get("last_hod", current_hod)
        )
        previous_lod = min(
            breakout_state.get("last_lod", current_lod),
            reclaim_state.get("last_lod", current_lod),
            breakdown_state.get("last_lod", current_lod),
            retest_state.get("last_lod", current_lod)
        )
        
        new_structure_formed, updated_hod, updated_lod = check_new_structure_formation(
            cb_service, current_ts_1h, previous_hod, previous_lod
        )
        
        if new_structure_formed:
            logger.info("🔄 New 24h structure formed - resetting stopped_out flags for all strategies")
            # Reset stopped_out flags for all strategies
            for strategy_name, state, state_file in [
                ("Breakout", breakout_state, BREAKOUT_TRIGGER_FILE),
                ("Reclaim", reclaim_state, RECLAIM_TRIGGER_FILE),
                ("Breakdown", breakdown_state, BREAKDOWN_TRIGGER_FILE),
                ("Retest", retest_state, RETEST_TRIGGER_FILE)
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
                                     (reclaim_state, RECLAIM_TRIGGER_FILE), 
                                     (breakdown_state, BREAKDOWN_TRIGGER_FILE), 
                                     (retest_state, RETEST_TRIGGER_FILE)]:
                state["last_hod"] = current_hod
                state["last_lod"] = current_lod
                save_trigger_state(state, state_file)
        
        # Calculate 20-period average volume for 1h (excluding current candle)
        volume_candles_1h = candles_1h[-(VOLUME_PERIOD+1):-1]
        avg_volume_1h = sum(float(c['volume']) for c in volume_candles_1h) / len(volume_candles_1h)
        
        # Get 5-minute and 15-minute candles for volume confirmation and retest analysis
        start_5m = now - timedelta(hours=2)
        start_ts_5m = int(start_5m.timestamp())
        end_ts_5m = int(now.timestamp())
        
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts_5m, GRANULARITY_5M)
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts_5m, GRANULARITY_15M)
        
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
        
        # Sort 15m candles if available
        if candles_15m:
            candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
        else:
            candles_15m = []
        
        # Check volume confirmation
        volume_confirmed = check_volume_confirmation(cb_service, current_volume_1h, current_volume_5m, avg_volume_1h, avg_volume_5m)
        
        # Check chop filter conditions
        def check_chop_filter(candles_1h, current_volume_5m, avg_volume_5m):
            """Check if market is choppy and should be skipped"""
            try:
                # Calculate 1h ATR percentage
                if len(candles_1h) >= 20:
                    # Calculate ATR for last 20 candles
                    atr_values = []
                    for i in range(1, len(candles_1h)):
                        high = float(candles_1h[i]['high'])
                        low = float(candles_1h[i]['low'])
                        prev_close = float(candles_1h[i-1]['close'])
                        
                        tr1 = high - low
                        tr2 = abs(high - prev_close)
                        tr3 = abs(low - prev_close)
                        tr = max(tr1, tr2, tr3)
                        atr_values.append(tr)
                    
                    avg_atr = sum(atr_values) / len(atr_values)
                    current_price = float(candles_1h[-1]['close'])
                    atr_percentage = (avg_atr / current_price) * 100
                    
                    # Check volume condition
                    volume_chop = current_volume_5m < (VOLUME_CHOP_FACTOR * avg_volume_5m) if avg_volume_5m > 0 else False
                    
                    # Chop filter: skip if 1h ATR% < 0.4 and 5m vol < 0.8× average
                    is_chop = atr_percentage < ATR_PERCENTAGE_THRESHOLD and volume_chop
                    
                    logger.info(f"Chop filter check: 1h ATR% = {atr_percentage:.2f}% (threshold: {ATR_PERCENTAGE_THRESHOLD}%)")
                    logger.info(f"Volume chop: 5m vol = {current_volume_5m:,.0f} vs {VOLUME_CHOP_FACTOR}x avg = {VOLUME_CHOP_FACTOR * avg_volume_5m:,.0f}")
                    logger.info(f"Chop filter: {'✅ SKIP' if is_chop else '❌ CONTINUE'}")
                    
                    return is_chop
                else:
                    logger.warning("Not enough data for chop filter calculation")
                    return False
            except Exception as e:
                logger.error(f"Error in chop filter calculation: {e}")
                return False
        
        # Apply chop filter
        chop_filter_active = check_chop_filter(candles_1h, current_volume_5m, avg_volume_5m)
        
        # Check for whipsaw conditions
        def check_whipsaw_condition(candles_5m, candles_15m, entry_level, is_long):
            """Check if trigger whipsaws and closes back inside the level within 15m"""
            try:
                # Check last 3 5m candles and 1 15m candle for whipsaw
                whipsaw_detected = False
                
                # Check 5m timeframe
                for candle in candles_5m[-3:]:
                    close = float(candle['close'])
                    if is_long:
                        # For long trades, check if price went above entry but closed back below
                        if close < entry_level:
                            whipsaw_detected = True
                            logger.info(f"Whipsaw detected: 5m close ${close:,.2f} < ${entry_level:,.2f} after trigger")
                    else:
                        # For short trades, check if price went below entry but closed back above
                        if close > entry_level:
                            whipsaw_detected = True
                            logger.info(f"Whipsaw detected: 5m close ${close:,.2f} > ${entry_level:,.2f} after trigger")
                
                # Check 15m timeframe
                if candles_15m:
                    candle = candles_15m[-1]
                    close = float(candle['close'])
                    if is_long:
                        if close < entry_level:
                            whipsaw_detected = True
                            logger.info(f"Whipsaw detected: 15m close ${close:,.2f} < ${entry_level:,.2f} after trigger")
                    else:
                        if close > entry_level:
                            whipsaw_detected = True
                            logger.info(f"Whipsaw detected: 15m close ${close:,.2f} > ${entry_level:,.2f} after trigger")
                
                logger.info(f"Whipsaw check: {'❌ WHIPSAW - STAND DOWN' if whipsaw_detected else '✅ NO WHIPSAW'}")
                return whipsaw_detected
                
            except Exception as e:
                logger.error(f"Error checking whipsaw condition: {e}")
                return False
        
        # Check whipsaw for each strategy
        breakout_whipsaw = check_whipsaw_condition(candles_5m, candles_15m, BREAKOUT_ENTRY_LOW, True)
        reclaim_whipsaw = check_whipsaw_condition(candles_5m, candles_15m, RECLAIM_ENTRY_LOW, True)
        breakdown_whipsaw = check_whipsaw_condition(candles_5m, candles_15m, BREAKDOWN_ENTRY_HIGH, False)
        retest_whipsaw = check_whipsaw_condition(candles_5m, candles_15m, RETEST_ENTRY_HIGH, False)
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — Clean Two-Sided ETH Plan for Today (Live Levels) Alert")
        logger.info("")
        logger.info("📊 Live Levels (ETH ≈ $4,300; HOD $4,351; LOD $4,174):")
        logger.info(f"   • ETH ≈ ${current_close_1h:,.0f}")
        logger.info(f"   • 24h HOD: ${current_hod:,.0f}")
        logger.info(f"   • 24h LOD: ${current_lod:,.0f}")
        logger.info(f"   • MID: ${current_mid_range:,.0f}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Timeframe: trigger on 1h; execute on 5–15m close")
        logger.info(f"   • Volume confirm: ≥{VOLUME_SURGE_FACTOR_1H}x 20-period vol on 1h OR ≥{VOLUME_SURGE_FACTOR_5M}x 20-SMA vol on 5m")
        logger.info(f"   • Orders: market-only brackets; size so 1R ≈ {RISK_PERCENTAGE_LOW}% of entry")
        logger.info(f"   • Partial at +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R; trail if momentum persists")
        logger.info(f"   • Skip if 1h ATR% < {ATR_PERCENTAGE_THRESHOLD} and 5m vol < {VOLUME_CHOP_FACTOR}× average (chop filter)")
        logger.info(f"   • If trigger whipsaws and closes back inside level within 15m, stand down for 30–60m")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} margin x {LEVERAGE}x leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("Type: Breakout")
            logger.info(f"   • Entry: 1h/5–15m close above ${BREAKOUT_ENTRY_LOW:,.0f}–${BREAKOUT_ENTRY_HIGH:,.0f} (HOD + ~0.3%)")
            logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,.0f} (≈1R below ~4,366)")
            logger.info(f"   • TP1: ${BREAKOUT_TP1:,.0f}–${BREAKOUT_TP2_LOW:,.0f}")
            logger.info(f"   • TP2: ${BREAKOUT_TP2_LOW:,.0f}–${BREAKOUT_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: range expansion above HOD with fresh participation; invalidation puts price back inside prior high")
            logger.info("")
            logger.info("Type: Retest")
            logger.info(f"   • Setup: After confirmed break/close >4,365, buy reclaim of ${RECLAIM_ENTRY_LOW:,.0f}–${RECLAIM_ENTRY_HIGH:,.0f}")
            logger.info(f"   • SL: ${RECLAIM_STOP_LOSS:,.0f} (below reclaimed zone)")
            logger.info(f"   • TP1: ${RECLAIM_TP1:,.0f}")
            logger.info(f"   • TP2: ${RECLAIM_TP2_LOW:,.0f}–${RECLAIM_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: classic break-and-retest; keeps risk tight to the reclaimed level")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("Type: Breakdown")
            logger.info(f"   • Entry: 1h/5–15m close below ${BREAKDOWN_ENTRY_LOW:,.0f}–${BREAKDOWN_ENTRY_HIGH:,.0f} (LOD − ~0.3%)")
            logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,.0f} (≈1R above ~4,161)")
            logger.info(f"   • TP1: ${BREAKDOWN_TP1:,.0f}–${BREAKDOWN_TP2_LOW:,.0f}")
            logger.info(f"   • TP2: ${BREAKDOWN_TP2_LOW:,.0f}–${BREAKDOWN_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: continuation lower after clearing day's liquidity; back-inside = exit")
            logger.info("")
            logger.info("Type: Failed-retest")
            logger.info(f"   • Setup: After confirmed break <4,163, sell rejection at ${RETEST_ENTRY_LOW:,.0f}–${RETEST_ENTRY_HIGH:,.0f}")
            logger.info(f"   • SL: ${RETEST_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${RETEST_TP1:,.0f}")
            logger.info(f"   • TP2: ${RETEST_TP2_LOW:,.0f}–${RETEST_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: underside test of broken support with clean structure")
            logger.info("")
        logger.info("")
        logger.info(f"Current Price: ${current_close_1h:,.2f}")
        logger.info(f"Last 1H Close: ${current_close_1h:,.2f}, High: ${current_high_1h:,.2f}, Low: ${current_low_1h:,.2f}")
        logger.info(f"1H Volume: {current_volume_1h:,.0f}, 1H SMA: {avg_volume_1h:,.0f}, Rel_Vol: {current_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}")
        logger.info(f"5M Volume: {current_volume_5m:,.0f}, 5M SMA: {avg_volume_5m:,.0f}, Rel_Vol: {current_volume_5m/avg_volume_5m if avg_volume_5m > 0 else 0:.2f}")
        logger.info(f"Volume Confirmed: {'✅' if volume_confirmed else '❌'}")
        logger.info("")
        
        # Execution guardrails
        # If trigger fires without volume confirmation → skip or halve size
        # After a stopped trade, stand down unless a fresh 1h structure forms
        # Prefer the breakout long path if >4,120 holds on 15m with rising 1h volume; otherwise neutral until 4k breaks (then take breakdown)
        price_position_in_range = (current_close_1h - current_lod) / current_range_width if current_range_width > 0 else 0.5
        logger.info(f"Price position in range: {price_position_in_range:.2%} (0% = LOD, 100% = HOD)")
        
        # Determine which path to prioritize based on direction filter and execution guardrails
        if direction == 'LONG':
            logger.info("🎯 Direction filter: LONG only - prioritizing breakout strategies")
            breakdown_priority = False
            breakout_priority = True
        elif direction == 'SHORT':
            logger.info("🎯 Direction filter: SHORT only - prioritizing breakdown strategies")
            breakdown_priority = True
            breakout_priority = False
        else:  # BOTH - use execution guardrails
            # Check if any strategy is already triggered (do not run long + short simultaneously)
            long_triggered = breakout_state.get("triggered", False) or reclaim_state.get("triggered", False)
            short_triggered = breakdown_state.get("triggered", False) or retest_state.get("triggered", False)
            
            if long_triggered:
                logger.info("🎯 Execution guardrail: LONG strategy already triggered - prioritizing LONG")
                breakdown_priority = False
                breakout_priority = True
            elif short_triggered:
                logger.info("🎯 Execution guardrail: SHORT strategy already triggered - prioritizing SHORT")
                breakdown_priority = True
                breakout_priority = False
            else:
                # No strategy triggered yet - use execution guardrail preference
                # Pick one mode (breakout or mean-revert) and stick to it today
                # If not sure, run only the breakout pair (cleaner invalidation)
                logger.info("🎯 Execution guardrail: Pick one mode (breakout or mean-revert) and stick to it today")
                logger.info("🎯 If not sure, run only the breakout pair (cleaner invalidation)")
                breakdown_priority = False
                breakout_priority = True
        
        # LONG (breakout) Strategy Conditions
        breakout_condition = (
            breakout_priority and
            current_close_1h >= BREAKOUT_ENTRY_LOW and 
            current_close_1h <= BREAKOUT_ENTRY_HIGH and 
            volume_confirmed and 
            not chop_filter_active and  # Skip if chop filter is active
            not breakout_whipsaw and  # Skip if whipsaw detected
            not breakout_state.get("triggered", False) and
            not breakout_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # Check for confirmed break and reclaim (required for retest strategy)
        break_and_reclaim_confirmed, break_reclaim_status = check_confirmed_break_and_reclaim(cb_service, current_ts_1h)
        
        # LONG (retest after confirmed break) Strategy Conditions
        reclaim_condition = (
            breakout_priority and
            current_close_1h >= RECLAIM_ENTRY_LOW and 
            current_close_1h <= RECLAIM_ENTRY_HIGH and 
            volume_confirmed and 
            break_and_reclaim_confirmed and  # Must have confirmed break and reclaim
            not chop_filter_active and  # Skip if chop filter is active
            not reclaim_whipsaw and  # Skip if whipsaw detected
            not reclaim_state.get("triggered", False) and
            not reclaim_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # SHORT (breakdown) Strategy Conditions
        breakdown_condition = (
            breakdown_priority and
            current_close_1h <= BREAKDOWN_ENTRY_HIGH and 
            current_close_1h >= BREAKDOWN_ENTRY_LOW and 
            volume_confirmed and 
            not chop_filter_active and  # Skip if chop filter is active
            not breakdown_whipsaw and  # Skip if whipsaw detected
            not breakdown_state.get("triggered", False) and
            not breakdown_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # Check for confirmed break and failed retest (required for failed-retest strategy)
        break_and_failed_retest_confirmed, break_failed_retest_status = check_confirmed_break_and_failed_retest(cb_service, current_ts_1h)
        
        # SHORT (failed-retest after confirmed break) Strategy Conditions
        retest_condition = (
            breakdown_priority and
            current_close_1h <= RETEST_ENTRY_HIGH and 
            current_close_1h >= RETEST_ENTRY_LOW and 
            volume_confirmed and 
            break_and_failed_retest_confirmed and  # Must have confirmed break and failed retest
            not chop_filter_active and  # Skip if chop filter is active
            not retest_whipsaw and  # Skip if whipsaw detected
            not retest_state.get("triggered", False) and
            not retest_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy
        if long_strategies_enabled:
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_close_1h <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and volume_confirmed and not chop_filter_active and not breakout_whipsaw and breakout_priority and not breakout_state.get("triggered", False) and not breakout_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKOUT_ENTRY_LOW:,.0f}-${BREAKOUT_ENTRY_HIGH:,.0f}): {'✅' if in_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H: {current_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x, 5M: {current_volume_5m/avg_volume_5m if avg_volume_5m > 0 else 0:.2f}x): {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if breakout_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakout_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if breakout_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if breakout_state.get('stopped_out', False) else '❌'}")
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
                    logger.info("Strategy: Expansion above session highs with strong tape tends to trend when 4.2k clears on volume")
                    
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
        if not trade_executed and long_strategies_enabled:
            in_reclaim_zone = RECLAIM_ENTRY_LOW <= current_close_1h <= RECLAIM_ENTRY_HIGH
            reclaim_ready = in_reclaim_zone and volume_confirmed and break_and_reclaim_confirmed and not chop_filter_active and not reclaim_whipsaw and breakout_priority and not reclaim_state.get("triggered", False) and not reclaim_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Retest Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${RECLAIM_ENTRY_LOW:,.0f}-${RECLAIM_ENTRY_HIGH:,.0f}): {'✅' if in_reclaim_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Break and reclaim confirmed: {'✅' if break_and_reclaim_confirmed else '❌'} ({break_reclaim_status})")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if reclaim_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakout_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if reclaim_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if reclaim_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if reclaim_ready else '⏳ NO'}")
            
            if reclaim_ready:
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
                    trade_type="ETH-USD LONG Retest (Break & Reclaim)",
                    entry_price=current_close_1h,
                    stop_loss=RECLAIM_STOP_LOSS,
                    take_profit=RECLAIM_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 LONG - Retest trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${RECLAIM_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${RECLAIM_TP1:,.2f}")
                    logger.info(f"TP2: ${RECLAIM_TP2_LOW:,.2f}-${RECLAIM_TP2_HIGH:,.2f}")
                    logger.info("Strategy: classic break-and-retest; keeps risk tight to the reclaimed level")
                    
                    # Save trigger state
                    reclaim_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(reclaim_state, RECLAIM_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Retest trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if not trade_executed and short_strategies_enabled:
            in_breakdown_zone = BREAKDOWN_ENTRY_LOW <= current_close_1h <= BREAKDOWN_ENTRY_HIGH
            breakdown_ready = in_breakdown_zone and volume_confirmed and not chop_filter_active and not breakdown_whipsaw and breakdown_priority and not breakdown_state.get("triggered", False) and not breakdown_state.get("stopped_out", False)
            
            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKDOWN_ENTRY_LOW:,.0f}-${BREAKDOWN_ENTRY_HIGH:,.0f}): {'✅' if in_breakdown_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if breakdown_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if breakdown_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if breakdown_state.get('stopped_out', False) else '❌'}")
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
                    logger.info("Strategy: Range failure through 4k with 1h confirmation often extends")
                    
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
        
        # 4. SHORT - Failed-retest Strategy
        if not trade_executed and short_strategies_enabled:
            in_retest_zone = RETEST_ENTRY_LOW <= current_close_1h <= RETEST_ENTRY_HIGH
            retest_ready = in_retest_zone and volume_confirmed and break_and_failed_retest_confirmed and not chop_filter_active and not retest_whipsaw and breakdown_priority and not retest_state.get("triggered", False) and not retest_state.get("stopped_out", False)
            
            logger.info("🔍 SHORT - Failed-retest Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,.0f}-${RETEST_ENTRY_HIGH:,.0f}): {'✅' if in_retest_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Break and failed retest confirmed: {'✅' if break_and_failed_retest_confirmed else '❌'} ({break_failed_retest_status})")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if retest_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if retest_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if retest_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Failed-retest Ready: {'🎯 YES' if retest_ready else '⏳ NO'}")
            
            if retest_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fade into resistance Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Failed-retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Failed-retest (Break & Rejection)",
                    entry_price=current_close_1h,
                    stop_loss=RETEST_STOP_LOSS,
                    take_profit=RETEST_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Failed-retest trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${RETEST_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${RETEST_TP1:,.2f}")
                    logger.info(f"TP2: ${RETEST_TP2_LOW:,.2f}-${RETEST_TP2_HIGH:,.2f}")
                    logger.info("Strategy: underside test of broken support with clean structure")
                    
                    # Save trigger state
                    retest_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Fade trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for strategy conditions...")
            if direction != 'BOTH':
                logger.info(f"   Direction filter: {direction} only")
            if not volume_confirmed:
                logger.info(f"   Volume confirmation not met")
            
            if long_strategies_enabled:
                if breakout_state.get("triggered", False):
                    logger.info("   Breakout strategy already triggered")
                if reclaim_state.get("triggered", False):
                    logger.info("   Retest strategy already triggered")
                elif not break_and_reclaim_confirmed:
                    logger.info("   Retest: Break and reclaim not confirmed")
            
            if short_strategies_enabled:
                if breakdown_state.get("triggered", False):
                    logger.info("   Breakdown strategy already triggered")
                if retest_state.get("triggered", False):
                    logger.info("   Failed-retest strategy already triggered")
                elif not break_and_failed_retest_confirmed:
                    logger.info("   Failed-retest: Break and failed retest not confirmed")
        
        # Reset triggers if price moves significantly away from entry zones
        # Execution guardrails: If first entry stops, stand down until new 24h structure forms
        if breakout_state.get("triggered", False):
            if current_close_1h < BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Breakout trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                logger.info("Breakout trigger state reset - standing down")
        
        if reclaim_state.get("triggered", False):
            if current_close_1h < RECLAIM_STOP_LOSS:
                logger.info("🔄 Resetting Retest trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                reclaim_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(reclaim_state, RECLAIM_TRIGGER_FILE)
                logger.info("Retest trigger state reset - standing down")
        
        if breakdown_state.get("triggered", False):
            if current_close_1h > BREAKDOWN_STOP_LOSS:
                logger.info("🔄 Resetting Breakdown trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                breakdown_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(breakdown_state, BREAKDOWN_TRIGGER_FILE)
                logger.info("Breakdown trigger state reset - standing down")
        
        if retest_state.get("triggered", False):
            if current_close_1h > RETEST_STOP_LOSS:
                logger.info("🔄 Resetting Fade into resistance trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                retest_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                logger.info("Fade into resistance trigger state reset - standing down")
        
        logger.info("=== ETH-USD Trading Strategy Alert completed ===")
        return current_ts_1h
        
    except Exception as e:
        logger.error(f"Error in ETH-USD Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH-USD Trading Strategy Alert completed (with error) ===")
    return last_alert_ts



# Replace main loop to use new alert
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH-USD Trading Strategy Monitor with optional direction filter')
    parser.add_argument('--direction', choices=['LONG', 'SHORT', 'BOTH'], default='BOTH',
                       help='Trading direction to monitor: LONG, SHORT, or BOTH (default: BOTH)')
    args = parser.parse_args()
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor_eth.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor_eth.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor_eth.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting ETH-USD Trading Strategy Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: Clean Two-Sided ETH Plan for Today based on Live Levels - LONG & SHORT with Execution Guardrails")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    
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
        last_alert_ts = eth_trading_strategy_alert(cb_service, last_alert_ts, direction)
        consecutive_failures = 0
        logger.info(f"✅ ETH alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
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