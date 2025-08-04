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

# ETH Trading Strategy Parameters (based on new ETH plan - Aug 4, 2025 levels)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_1H = "ONE_HOUR"  # 1-hour chart for trigger
GRANULARITY_5M = "FIVE_MINUTE"  # 5-minute chart for execution
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (ETH ≈ $3,545, Live range ≈ $3,445.8–$3,567.8)
CURRENT_ETH_PRICE = 3545
HOD = 3567.8  # High of day (live range)
LOD = 3445.8  # Low of day (live range)
TODAYS_RANGE_WIDTH = HOD - LOD  # 122 points
MID_RANGE_PIVOT = (HOD + LOD) / 2  # 3506.8

# LONG (breakout) Strategy Parameters
BREAKOUT_ENTRY_LOW = 3576  # $3,576–$3,586 (above HOD + buffer)
BREAKOUT_ENTRY_HIGH = 3586
BREAKOUT_STOP_LOSS = 3538  # $3,538 (back inside prior range)
BREAKOUT_TP1 = 3625  # TP1: $3,625
BREAKOUT_TP2_LOW = 3680  # TP2: $3,680–3,710
BREAKOUT_TP2_HIGH = 3710

# LONG (retest) Strategy Parameters
RETEST_ENTRY_LOW = 3505  # $3,505–$3,515 on a pullback that holds
RETEST_ENTRY_HIGH = 3515
RETEST_STOP_LOSS = 3479  # $3,479
RETEST_TP1 = 3550  # TP1: $3,550
RETEST_TP2_LOW = 3580  # TP2: $3,580–3,600
RETEST_TP2_HIGH = 3600

# SHORT (breakdown) Strategy Parameters
BREAKDOWN_ENTRY_LOW = 3438  # $3,438–$3,446 (through LOD)
BREAKDOWN_ENTRY_HIGH = 3446
BREAKDOWN_STOP_LOSS = 3468  # $3,468
BREAKDOWN_TP1 = 3388  # TP1: $3,388
BREAKDOWN_TP2_LOW = 3340  # TP2: $3,340–3,320
BREAKDOWN_TP2_HIGH = 3320

# SHORT (fade of highs) Strategy Parameters
FADE_ENTRY_LOW = 3565  # $3,565–$3,575 if HOD retest rejects with 5m lower-high + rising sell volume
FADE_ENTRY_HIGH = 3575
FADE_STOP_LOSS = 3588  # $3,588
FADE_TP1 = 3520  # TP1: $3,520
FADE_TP2_LOW = 3490  # TP2: $3,490–3,475
FADE_TP2_HIGH = 3475

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

def check_higher_low_retest(cb_service, current_price, current_ts):
    """Check for retest pattern: pullback to $3,505-3,515 on a pullback that holds; allow wick to $3,490 if 5m reclaims quickly"""
    try:
        # Get recent 5-minute candles to check for pullback and higher-low formation
        end = current_ts
        start = end - timedelta(hours=2)  # Check last 2 hours
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        
        if not candles_5m or len(candles_5m) < 6:
            return False
            
        # Sort by timestamp ascending
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        
        # Check if price pulled back to the entry zone
        pullback_occurred = False
        wick_to_3490_allowed = False
        
        for candle in candles_5m[-6:]:  # Check last 6 candles (30 minutes)
            low = float(candle['low'])
            high = float(candle['high'])
            close = float(candle['close'])
            
            # Check if price pulled back to the entry zone
            if RETEST_ENTRY_LOW <= low <= RETEST_ENTRY_HIGH:
                pullback_occurred = True
                logger.info(f"Pullback to retest zone detected at ${low:,.2f} in 5m candle")
                break
            
            # Check if wick went to $3,490 but 5m reclaimed quickly
            if low <= 3490 and close > 3490:
                wick_to_3490_allowed = True
                logger.info(f"Wick to $3,490 detected but 5m reclaimed quickly at ${close:,.2f}")
        
        if not pullback_occurred and not wick_to_3490_allowed:
            logger.info(f"No pullback to retest zone (${RETEST_ENTRY_LOW:,.2f}-${RETEST_ENTRY_HIGH:,.2f}) or allowed wick to $3,490 detected")
            return False
        
        # Check if the pullback held (bids defended mid-range)
        # Get current day's LOD
        current_hod, current_lod = get_current_day_hod_lod(cb_service, current_ts)
        
        # Check if current price is above the retest entry zone (bids defended)
        bids_defended = current_price >= RETEST_ENTRY_LOW
        
        logger.info(f"Retest check: current price ${current_price:,.2f} vs retest zone ${RETEST_ENTRY_LOW:,.2f}-${RETEST_ENTRY_HIGH:,.2f} -> {'✅' if bids_defended else '❌'}")
        logger.info(f"Strategy: Pullback to ${RETEST_ENTRY_LOW:,.2f}-${RETEST_ENTRY_HIGH:,.2f}, allow wick to $3,490 if 5m reclaims quickly")
        
        return bids_defended
        
    except Exception as e:
        logger.error(f"Error checking retest pattern: {e}")
        return False

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

def check_fade_of_highs(cb_service, current_price, current_ts):
    """Check for fade of highs pattern: $3,565-3,575 if HOD retest rejects with 5m lower-high + rising sell volume"""
    try:
        # Get recent 5-minute candles to check for HOD retest rejection pattern
        end = current_ts
        start = end - timedelta(hours=1)  # Check last hour
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        
        if not candles_5m or len(candles_5m) < 12:  # Need more candles for pattern analysis
            return False
            
        # Sort by timestamp ascending
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        
        # Get current day's HOD
        current_hod, current_lod = get_current_day_hod_lod(cb_service, current_ts)
        
        # Check for HOD retest rejection pattern in the entry zone (last 12 candles = 1 hour)
        hod_retest_rejection = False
        lower_high_formed = False
        rising_sell_volume = False
        
        # Check if price reached the fade entry zone (HOD retest area)
        for candle in candles_5m[-6:]:  # Check last 6 candles (30 minutes)
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            open_price = float(candle['open'])
            volume = float(candle['volume'])
            
            # Check if price reached the HOD retest zone
            if FADE_ENTRY_LOW <= high <= FADE_ENTRY_HIGH:
                # Calculate wick sizes
                body_size = abs(close - open_price)
                upper_wick = high - max(open_price, close)
                
                # Check for bearish rejection at HOD (bearish wick or lower close)
                is_bearish_wick = (upper_wick > body_size * 0.3 and close < open_price)
                is_lower_close = close < open_price
                
                if is_bearish_wick or is_lower_close:
                    hod_retest_rejection = True
                    logger.info(f"HOD retest rejection detected: high=${high:,.2f}, close=${close:,.2f}, bearish_wick=${is_bearish_wick}")
                    break
        
        if not hod_retest_rejection:
            logger.info(f"No HOD retest rejection at fade zone (${FADE_ENTRY_LOW:,.2f}-${FADE_ENTRY_HIGH:,.2f}) detected")
            return False
        
        # Check for 5m lower-high formation (last 12 candles)
        highs = [float(candle['high']) for candle in candles_5m[-12:]]
        if len(highs) >= 3:
            # Check if recent highs are forming lower highs
            recent_highs = highs[-3:]  # Last 3 candles
            if recent_highs[2] < recent_highs[1] < recent_highs[0]:  # Lower highs
                lower_high_formed = True
                logger.info(f"5m lower-high formation detected: {recent_highs}")
        
        # Check for rising sell volume (simplified - check if recent volume is higher than average)
        volumes = [float(candle['volume']) for candle in candles_5m[-6:]]
        if len(volumes) >= 3:
            recent_avg_volume = sum(volumes[-3:]) / 3
            if volumes[-1] > recent_avg_volume * 1.2:  # 20% higher than recent average
                rising_sell_volume = True
                logger.info(f"Rising sell volume detected: current=${volumes[-1]:,.0f} vs avg=${recent_avg_volume:,.0f}")
        
        # All conditions must be met for fade of highs
        fade_conditions_met = hod_retest_rejection and lower_high_formed and rising_sell_volume
        
        logger.info(f"Fade of highs check:")
        logger.info(f"  HOD retest rejection: {'✅' if hod_retest_rejection else '❌'}")
        logger.info(f"  5m lower-high formation: {'✅' if lower_high_formed else '❌'}")
        logger.info(f"  Rising sell volume: {'✅' if rising_sell_volume else '❌'}")
        logger.info(f"  Overall: {'✅' if fade_conditions_met else '❌'}")
        
        return fade_conditions_met
        
    except Exception as e:
        logger.error(f"Error checking fade of highs pattern: {e}")
        return False

# --- ETH Trading Strategy Alert Logic ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH-USD Trading Strategy Alert - Implements new ETH two-sided plan for Monday, Aug 4, 2025
    Based on the trading plan: "Spiros — ETH two-sided plan for Mon, Aug 4, 2025 (live ≈ $3,545). Facts: HOD ≈ $3,567.8, LOD ≈ $3,445.8 (intraday)."
    
    Rules (both directions):
    - Trigger: 1h level break; execute on 5–15m
    - Volume: ≥ 1.25× 20-SMA vol on 1h OR ≥ 2× 20-SMA vol on 5m at trigger
    - Risk: size so 1R = 0.8–1.2% of price; partial at +1.0–1.5R
    - Invalidation discipline: 5m close back inside broken level = exit
    - Position Size: Always margin x leverage = 250 x 20 = $5,000 USD
    
    LONG (breakout): Entry $3,576–3,586, SL $3,538, TP1 $3,625, TP2 $3,680–3,710
    LONG (retest): Entry $3,505–3,515 on pullback that holds, SL $3,479, TP1 $3,550, TP2 $3,580–3,600
    SHORT (breakdown): Entry $3,438–3,446, SL $3,468, TP1 $3,388, TP2 $3,340–3,320
    SHORT (fade of highs): Entry $3,565–3,575 if HOD retest rejects, SL $3,588, TP1 $3,520, TP2 $3,490–3,475
    
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
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 ETH Two-Sided Plan for Monday, Aug 4, 2025 (Live ≈ $3,545) Alert")
        logger.info("")
        logger.info("📊 Today's Levels:")
        logger.info(f"   • ETH ≈ ${current_close_1h:,.0f}")
        logger.info(f"   • HOD: ${current_hod:,.0f}")
        logger.info(f"   • LOD: ${current_lod:,.0f}")
        logger.info(f"   • Live range: ${current_lod:,.0f}–${current_hod:,.0f}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Timeframe: trigger on 1h; execute on 5-15m")
        logger.info(f"   • Volume confirm: ≥{VOLUME_SURGE_FACTOR_1H}x 20-period vol on 1h OR ≥{VOLUME_SURGE_FACTOR_5M}x 20-SMA vol on 5m")
        logger.info(f"   • Risk: Size so 1R is ~{RISK_PERCENTAGE_LOW}-{RISK_PERCENTAGE_HIGH}% of price")
        logger.info(f"   • Partial at +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} margin x {LEVERAGE}x leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG - Breakout Strategy:")
            logger.info(f"   • Entry: ${BREAKOUT_ENTRY_LOW:,.0f}-${BREAKOUT_ENTRY_HIGH:,.0f} (above HOD + buffer)")
            logger.info(f"   • SL: ${BREAKOUT_STOP_LOSS:,.0f} (back inside prior range)")
            logger.info(f"   • TP1: ${BREAKOUT_TP1:,.0f}")
            logger.info(f"   • TP2: ${BREAKOUT_TP2_LOW:,.0f}-${BREAKOUT_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: Range expansion above today's high with momentum; clean continuation if vol confirms")
            logger.info("")
            logger.info("📊 LONG - Retest Strategy:")
            logger.info(f"   • Entry: ${RETEST_ENTRY_LOW:,.0f}-${RETEST_ENTRY_HIGH:,.0f} on a pullback that holds; allow wick to $3,490 if 5m reclaims quickly")
            logger.info(f"   • SL: ${RETEST_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${RETEST_TP1:,.0f}")
            logger.info(f"   • TP2: ${RETEST_TP2_LOW:,.0f}-${RETEST_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: Bids defended mid-range in strong sessions; defined risk near round-number shelf")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT - Breakdown Strategy:")
            logger.info(f"   • Entry: ${BREAKDOWN_ENTRY_LOW:,.0f}-${BREAKDOWN_ENTRY_HIGH:,.0f} (through LOD)")
            logger.info(f"   • SL: ${BREAKDOWN_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${BREAKDOWN_TP1:,.0f}")
            logger.info(f"   • TP2: ${BREAKDOWN_TP2_LOW:,.0f}-${BREAKDOWN_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: Fresh day-low break after failed bounces tends to trend; ADR room below")
            logger.info("")
            logger.info("📊 SHORT - Fade of Highs Strategy:")
            logger.info(f"   • Entry: ${FADE_ENTRY_LOW:,.0f}-${FADE_ENTRY_HIGH:,.0f} if HOD retest rejects with 5m lower-high + rising sell volume")
            logger.info(f"   • SL: ${FADE_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${FADE_TP1:,.0f}")
            logger.info(f"   • TP2: ${FADE_TP2_LOW:,.0f}-${FADE_TP2_HIGH:,.0f}")
            logger.info(f"   • Why: First rejection at prior HOD often mean-reverts to mid-range when momentum stalls")
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
        
        # Determine which path to prioritize based on direction filter
        if direction == 'LONG':
            logger.info("🎯 Direction filter: LONG only - prioritizing breakout strategies")
            breakdown_priority = False
            breakout_priority = True
        elif direction == 'SHORT':
            logger.info("🎯 Direction filter: SHORT only - prioritizing breakdown strategies")
            breakdown_priority = True
            breakout_priority = False
        else:  # BOTH - allow both strategies to compete
            logger.info("🎯 Direction filter: BOTH - allowing all strategies to compete")
            breakdown_priority = True
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
            not retest_state.get("stopped_out", False) and  # Don't re-enter if stopped out
            check_higher_low_retest(cb_service, current_close_1h, current_ts_1h)
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
        
        # SHORT (fade of highs) Strategy Conditions
        fade_condition = (
            breakdown_priority and
            current_close_1h <= FADE_ENTRY_HIGH and 
            current_close_1h >= FADE_ENTRY_LOW and 
            volume_confirmed and 
            not fade_state.get("triggered", False) and
            not fade_state.get("stopped_out", False) and  # Don't re-enter if stopped out
            check_fade_of_highs(cb_service, current_close_1h, current_ts_1h)
        )
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy
        if long_strategies_enabled and not breakout_state.get("triggered", False) and not breakout_state.get("stopped_out", False):
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_close_1h <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and volume_confirmed and breakout_priority
            
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
                    logger.info("Strategy: Range expansion above today's high with momentum; clean continuation if vol confirms")
                    
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
            retest_pattern_detected = check_higher_low_retest(cb_service, current_close_1h, current_ts_1h)
            retest_ready = in_retest_zone and volume_confirmed and retest_pattern_detected and breakout_priority
            
            logger.info("🔍 LONG - Retest Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,.0f}-${RETEST_ENTRY_HIGH:,.0f}): {'✅' if in_retest_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Retest pattern detected: {'✅' if retest_pattern_detected else '❌'}")
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
                    logger.info(f"TP2: ${RETEST_TP2_LOW:,.2f}-${RETEST_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Bids defended mid-range in strong sessions; defined risk near round-number shelf")
                    
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
            breakdown_ready = in_breakdown_zone and volume_confirmed and breakdown_priority
            
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
                    logger.info("Strategy: Fresh day-low break after failed bounces tends to trend; ADR room below")
                    
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
        
        # 4. SHORT - Fade of Highs Strategy
        if not trade_executed and short_strategies_enabled and not fade_state.get("triggered", False) and not fade_state.get("stopped_out", False):
            in_fade_zone = FADE_ENTRY_LOW <= current_close_1h <= FADE_ENTRY_HIGH
            fade_of_highs_detected = check_fade_of_highs(cb_service, current_close_1h, current_ts_1h)
            fade_ready = in_fade_zone and volume_confirmed and fade_of_highs_detected and breakdown_priority
            
            logger.info("🔍 SHORT - Fade of Highs Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${FADE_ENTRY_LOW:,.0f}-${FADE_ENTRY_HIGH:,.0f}): {'✅' if in_fade_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Fade of highs pattern detected: {'✅' if fade_of_highs_detected else '❌'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Fade of Highs Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")
            
            if fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fade of Highs Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Fade of Highs trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Fade of Highs",
                    entry_price=current_close_1h,
                    stop_loss=FADE_STOP_LOSS,
                    take_profit=FADE_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Fade of Highs trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${FADE_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${FADE_TP1:,.2f}")
                    logger.info(f"TP2: ${FADE_TP2_LOW:,.2f}-${FADE_TP2_HIGH:,.2f}")
                    logger.info("Strategy: First rejection at prior HOD often mean-reverts to mid-range when momentum stalls")
                    
                    # Save trigger state
                    fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(fade_state, FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Fade of Highs trade failed: {trade_result}")
        
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
                if retest_state.get("triggered", False):
                    logger.info("   Retest strategy already triggered")
            
            if short_strategies_enabled:
                if breakdown_state.get("triggered", False):
                    logger.info("   Breakdown strategy already triggered")
                if fade_state.get("triggered", False):
                    logger.info("   Fade of Highs strategy already triggered")
        
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
                logger.info("🔄 Resetting Fade of Highs trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 1h structure forms")
                fade_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(fade_state, FADE_TRIGGER_FILE)
                logger.info("Fade of Highs trigger state reset - standing down")
        
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
        logger.info("Strategy: New ETH Two-Sided Plan (Aug 4, 2025) - LONG & SHORT")
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