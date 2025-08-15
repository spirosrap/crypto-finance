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

# ETH Trading Strategy Parameters (based on new ETH plan for Aug 15, 2025)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_1H = "ONE_HOUR"  # 1-hour chart for context
GRANULARITY_5M = "FIVE_MINUTE"  # 5-minute chart for execution
GRANULARITY_15M = "FIFTEEN_MINUTE"  # 15-minute chart for execution
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (ETH ≈ $4,567; intraday range ≈ $4,464 → $4,697)
CURRENT_ETH_PRICE = 4567.00
HOD_24H = 4697.00  # Today's high
LOD_24H = 4464.00  # Today's low
RANGE_WIDTH_24H = HOD_24H - LOD_24H  # 233 points
MID_RANGE_PIVOT = (HOD_24H + LOD_24H) / 2  # 4580.50

# LONG SETUPS

# 1) Breakout continuation
BREAKOUT_ENTRY_LOW = 4702  # Entry zone: $4,702–$4,715 on the confirmation candle or first shallow retest
BREAKOUT_ENTRY_HIGH = 4715
BREAKOUT_STOP_LOSS = 4660  # Stop: $4,660 (below base/failed breakout line)
BREAKOUT_TP1 = 4790  # First target: $4,790
BREAKOUT_TP2_LOW = 4860  # Optional runner toward $4,860–$4,890 if tape stays one-way
BREAKOUT_TP2_HIGH = 4890

# 2) Post-break retest (higher quality, fewer trades)
RETEST_ENTRY_LOW = 4685  # Entry zone: $4,685–$4,705 on hold/reclaim
RETEST_ENTRY_HIGH = 4705
RETEST_STOP_LOSS = 4650  # Stop: $4,650
RETEST_TP1 = 4780  # First target: $4,780
RETEST_TP2_LOW = 4860  # Optional runner toward $4,860–$4,890 if tape stays one-way
RETEST_TP2_HIGH = 4890

# SHORT SETUPS

# 3) Range break lower
RANGE_BREAK_ENTRY_LOW = 4450  # Entry zone: $4,450–$4,460 on breakdown/mini-pullback
RANGE_BREAK_ENTRY_HIGH = 4460
RANGE_BREAK_STOP_LOSS = 4500  # Stop: $4,500
RANGE_BREAK_TP1 = 4390  # First target: $4,390 (~1.4R from mid-entry)
RANGE_BREAK_TP2_LOW = 4350  # Additional target if momentum accelerates
RANGE_BREAK_TP2_HIGH = 4400

# Volume confirmation requirements
VOLUME_SURGE_FACTOR_15M = 1.3  # ≥1.3× 20-bar avg volume on 15m

# Risk management
RISK_PERCENTAGE_LOW = 0.8  # 1R ≈ 0.8% of entry
RISK_PERCENTAGE_HIGH = 1.2  # 1R ≈ 1.2% of entry
PARTIAL_PROFIT_RANGE_LOW = 1.0  # Partial at +1.0R
PARTIAL_PROFIT_RANGE_HIGH = 1.5  # Partial at +1.5R

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# Execution guardrails
MAX_TRADES_PER_DAY = 2  # Max 2 concurrent attempts per side to avoid chop
COOLDOWN_MINUTES = 30  # 30 min after a stop
MODE = "FAST"  # FAST = 15m close beyond trigger; CONSERVATIVE = 1h close
VCONF = True  # enforce volume rule

# Chop filter parameters - Avoid chop: If price sits between $4,490–$4,680, stand down unless you're scalping
CHOP_ZONE_LOW = 4490  # Chop zone: $4,490–$4,680
CHOP_ZONE_HIGH = 4680
ATR_PERCENTAGE_THRESHOLD = 0.4  # Skip if 1h ATR% < 0.4
VOLUME_CHOP_FACTOR = 0.8  # and 15m vol < 0.8× average (chop filter)

# State files for each strategy
BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
RETEST_TRIGGER_FILE = "eth_retest_trigger_state.json"
RANGE_BREAK_TRIGGER_FILE = "eth_range_break_trigger_state.json"

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

def check_volume_confirmation(cb_service, current_volume_15m, avg_volume_15m):
    """Check volume confirmation on 15m timeframe"""
    volume_15m_confirmed = current_volume_15m >= (VOLUME_SURGE_FACTOR_15M * avg_volume_15m)
    
    logger.info(f"Volume confirmation check:")
    logger.info(f"  15M: {current_volume_15m:,.0f} vs {VOLUME_SURGE_FACTOR_15M}x avg ({avg_volume_15m:,.0f}) -> {'✅' if volume_15m_confirmed else '❌'}")
    logger.info(f"  Overall: {'✅' if volume_15m_confirmed else '❌'}")
    
    return volume_15m_confirmed



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





# --- ETH Trading Strategy Alert Logic ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH-USD Trading Strategy Alert - Implements ETH plan for Aug 15, 2025
    Based on the trading plan: "Spiros — ETH intraday plan for Aug 15, 2025 (UTC+3)"
    
    Global rules (both directions):
    - Trigger: 15m close beyond trigger levels
    - Volume: fire only if 15m volume ≥ 1.3× its 20-bar avg
    - Risk: Size so a full stop = 1R (keep it constant). Scale out 50% at TP1
    - Position Size: Always margin x leverage = 250 x 20 = $5,000 USD
    - Avoid chop: If price sits between $4,490–$4,680, stand down unless you're scalping
    - Take the first clean signal only, no second chances
    
    LONG SETUPS:
    1) Breakout continuation
       - Trigger: 15m close > $4,698 (today's high) and 15m volume ≥ 1.3× its 20-bar avg
       - Entry zone: $4,702–$4,715 on the confirmation candle or first shallow retest
       - Stop: $4,660 (below base/failed breakout line)
       - First target: $4,790
       - Why: Expansion above intraday high with momentum; clear air toward the ATH pocket
    
    2) Post-break retest (higher quality, fewer trades)
       - Trigger: We first close above $4,698, then price retests $4,680–$4,700 and holds with a 15m higher-low or bullish engulfing
       - Entry zone: $4,685–$4,705 on hold/reclaim
       - Stop: $4,650
       - First target: $4,780 (optional runner toward $4,860–$4,890 if tape stays one-way)
       - Why: Classic break-and-retest toward the ATH cluster, avoids chasing the first impulse
    
    SHORT SETUPS:
    3) Range break lower
       - Trigger: 15m close < $4,460 (below today's low) and 15m volume ≥ 1.3× its 20-bar avg
       - Entry zone: $4,450–$4,460 on breakdown/mini-pullback
       - Stop: $4,500
       - First target: $4,390 (~1.4R from mid-entry)
       - Why: Lose day's floor → opens path to prior liquidity shelf; good asymmetry if momentum accelerates
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== ETH-USD Trading Strategy Alert (ETH Intraday Plan for Wed, Aug 13, 2025 - LONG & SHORT) ===")
    else:
        logger.info(f"=== ETH-USD Trading Strategy Alert ({direction} Strategy Only) ===")
    
    # Load trigger states for all strategies
    breakout_state = load_trigger_state(BREAKOUT_TRIGGER_FILE)
    retest_state = load_trigger_state(RETEST_TRIGGER_FILE)
    range_break_state = load_trigger_state(RANGE_BREAK_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 15-minute candles for analysis (primary timeframe)
        end = now
        start = now - timedelta(hours=VOLUME_PERIOD + 24)  # Enough data for volume analysis
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 15-minute candles for {VOLUME_PERIOD + 24} hours...")
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_15M)
        
        if not candles_15m or len(candles_15m) < VOLUME_PERIOD + 1:
            logger.warning("Not enough 15-minute candle data for trading strategy alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending
        candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
        
        # Get current 15-minute candle data
        current_candle_15m = candles_15m[-1]
        current_close_15m = float(current_candle_15m['close'])
        current_high_15m = float(current_candle_15m['high'])
        current_low_15m = float(current_candle_15m['low'])
        current_volume_15m = float(current_candle_15m['volume'])
        current_ts_15m = datetime.fromtimestamp(int(current_candle_15m['start']), UTC)
        
        # Get rolling 24h HOD and LOD
        current_hod, current_lod = get_rolling_24h_hod_lod(cb_service, current_ts_15m)
        
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
            retest_state.get("last_hod", current_hod),
            range_break_state.get("last_hod", current_hod)
        )
        previous_lod = min(
            breakout_state.get("last_lod", current_lod),
            retest_state.get("last_lod", current_lod),
            range_break_state.get("last_lod", current_lod)
        )
        
        new_structure_formed, updated_hod, updated_lod = check_new_structure_formation(
            cb_service, current_ts_15m, previous_hod, previous_lod
        )
        
        if new_structure_formed:
            logger.info("🔄 New 24h structure formed - resetting stopped_out flags for all strategies")
            # Reset stopped_out flags for all strategies
            for strategy_name, state, state_file in [
                ("Breakout", breakout_state, BREAKOUT_TRIGGER_FILE),
                ("Retest", retest_state, RETEST_TRIGGER_FILE),
                ("Range Break", range_break_state, RANGE_BREAK_TRIGGER_FILE)
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
                                     (range_break_state, RANGE_BREAK_TRIGGER_FILE)]:
                state["last_hod"] = current_hod
                state["last_lod"] = current_lod
                save_trigger_state(state, state_file)
        
        # Calculate 20-period average volume for 15m (excluding current candle)
        volume_candles_15m = candles_15m[-(VOLUME_PERIOD+1):-1]
        avg_volume_15m = sum(float(c['volume']) for c in volume_candles_15m) / len(volume_candles_15m)
        
        # Check volume confirmation
        volume_confirmed = check_volume_confirmation(cb_service, current_volume_15m, avg_volume_15m)
        
        # Check chop filter conditions
        def check_chop_filter(current_price, current_volume_15m, avg_volume_15m):
            """Check if market is choppy and should be skipped"""
            try:
                # Check if price is in the chop zone ($4,490–$4,680)
                in_chop_zone = CHOP_ZONE_LOW <= current_price <= CHOP_ZONE_HIGH
                
                # Check volume condition
                volume_chop = current_volume_15m < (VOLUME_CHOP_FACTOR * avg_volume_15m) if avg_volume_15m > 0 else False
                
                # Chop filter: skip if price is in chop zone and 15m vol < 0.8× average
                is_chop = in_chop_zone and volume_chop
                
                logger.info(f"Chop filter check: Price ${current_price:,.2f} in chop zone (${CHOP_ZONE_LOW:,.0f}-${CHOP_ZONE_HIGH:,.0f}): {'✅' if in_chop_zone else '❌'}")
                logger.info(f"Volume chop: 15m vol = {current_volume_15m:,.0f} vs {VOLUME_CHOP_FACTOR}x avg = {VOLUME_CHOP_FACTOR * avg_volume_15m:,.0f}")
                logger.info(f"Chop filter: {'✅ SKIP' if is_chop else '❌ CONTINUE'}")
                
                return is_chop
            except Exception as e:
                logger.error(f"Error in chop filter calculation: {e}")
                return False
        
        # Apply chop filter
        chop_filter_active = check_chop_filter(current_close_15m, current_volume_15m, avg_volume_15m)
        
        # Check for whipsaw conditions
        def check_whipsaw_condition(candles_15m, entry_level, is_long):
            """Check if trigger whipsaws and closes back inside the level within 15m"""
            try:
                # Check last 2 15m candles for whipsaw
                whipsaw_detected = False
                
                # Check 15m timeframe
                for candle in candles_15m[-2:]:
                    close = float(candle['close'])
                    if is_long:
                        # For long trades, check if price went above entry but closed back below
                        if close < entry_level:
                            whipsaw_detected = True
                            logger.info(f"Whipsaw detected: 15m close ${close:,.2f} < ${entry_level:,.2f} after trigger")
                    else:
                        # For short trades, check if price went below entry but closed back above
                        if close > entry_level:
                            whipsaw_detected = True
                            logger.info(f"Whipsaw detected: 15m close ${close:,.2f} > ${entry_level:,.2f} after trigger")
                
                logger.info(f"Whipsaw check: {'❌ WHIPSAW - STAND DOWN' if whipsaw_detected else '✅ NO WHIPSAW'}")
                return whipsaw_detected
                
            except Exception as e:
                logger.error(f"Error checking whipsaw condition: {e}")
                return False
        
        # Check whipsaw for each strategy
        breakout_whipsaw = check_whipsaw_condition(candles_15m, BREAKOUT_ENTRY_LOW, True)
        retest_whipsaw = check_whipsaw_condition(candles_15m, RETEST_ENTRY_LOW, True)
        range_break_whipsaw = check_whipsaw_condition(candles_15m, RANGE_BREAK_ENTRY_HIGH, False)
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — ETH Plan for Aug 15, 2025 Alert")
        logger.info("")
        logger.info("📊 Live Levels (ETH ≈ $4,567; intraday range ≈ $4,464 → $4,697):")
        logger.info(f"   • ETH ≈ ${current_close_15m:,.0f}")
        logger.info(f"   • Today's HOD: ${current_hod:,.0f}")
        logger.info(f"   • Today's LOD: ${current_lod:,.0f}")
        logger.info(f"   • MID: ${current_mid_range:,.0f}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Trigger: 15m close beyond trigger levels")
        logger.info(f"   • Volume: fire only if 15m volume ≥ {VOLUME_SURGE_FACTOR_15M}x its 20-bar avg")
        logger.info(f"   • Risk: Size so a full stop = 1R (keep it constant). Scale out 50% at TP1")
        logger.info(f"   • Avoid chop: If price sits between ${CHOP_ZONE_LOW:,.0f}–${CHOP_ZONE_HIGH:,.0f}, stand down unless you're scalping")
        logger.info(f"   • Take the first clean signal only, no second chances")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} margin x {LEVERAGE}x leverage)")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1) Breakout continuation")
            logger.info(f"   • Trigger: 15m close > ${HOD_24H:,.0f} (today's high) and 15m volume ≥ {VOLUME_SURGE_FACTOR_15M}x its 20-bar avg")
            logger.info(f"   • Entry zone: ${BREAKOUT_ENTRY_LOW:,.0f}–${BREAKOUT_ENTRY_HIGH:,.0f} on the confirmation candle or first shallow retest")
            logger.info(f"   • Stop: ${BREAKOUT_STOP_LOSS:,.0f} (below base/failed breakout line)")
            logger.info(f"   • First target: ${BREAKOUT_TP1:,.0f}")
            logger.info(f"   • Why: Expansion above intraday high with momentum; clear air toward the ATH pocket")
            logger.info("")
            logger.info("2) Post-break retest (higher quality, fewer trades)")
            logger.info(f"   • Trigger: We first close above ${HOD_24H:,.0f}, then price retests $4,680–$4,700 and holds with a 15m higher-low or bullish engulfing")
            logger.info(f"   • Entry zone: ${RETEST_ENTRY_LOW:,.0f}–${RETEST_ENTRY_HIGH:,.0f} on hold/reclaim")
            logger.info(f"   • Stop: ${RETEST_STOP_LOSS:,.0f}")
            logger.info(f"   • First target: ${RETEST_TP1:,.0f} (optional runner toward ${RETEST_TP2_LOW:,.0f}–${RETEST_TP2_HIGH:,.0f} if tape stays one-way)")
            logger.info(f"   • Why: Classic break-and-retest toward the ATH cluster, avoids chasing the first impulse")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("3) Range break lower")
            logger.info(f"   • Trigger: 15m close < ${LOD_24H:,.0f} (below today's low) and 15m volume ≥ {VOLUME_SURGE_FACTOR_15M}x its 20-bar avg")
            logger.info(f"   • Entry zone: ${RANGE_BREAK_ENTRY_LOW:,.0f}–${RANGE_BREAK_ENTRY_HIGH:,.0f} on breakdown/mini-pullback")
            logger.info(f"   • Stop: ${RANGE_BREAK_STOP_LOSS:,.0f}")
            logger.info(f"   • First target: ${RANGE_BREAK_TP1:,.0f} (~1.4R from mid-entry)")
            logger.info(f"   • Why: Lose day's floor → opens path to prior liquidity shelf; good asymmetry if momentum accelerates")
            logger.info("")
        logger.info("")
        logger.info(f"Current Price: ${current_close_15m:,.2f}")
        logger.info(f"Last 15M Close: ${current_close_15m:,.2f}, High: ${current_high_15m:,.2f}, Low: ${current_low_15m:,.2f}")
        logger.info(f"15M Volume: {current_volume_15m:,.0f}, 15M SMA: {avg_volume_15m:,.0f}, Rel_Vol: {current_volume_15m/avg_volume_15m if avg_volume_15m > 0 else 0:.2f}")
        logger.info(f"Volume Confirmed: {'✅' if volume_confirmed else '❌'}")
        logger.info("")
        
        # Execution guardrails
        # Take the first clean signal only, no second chances
        # Avoid chop: If price sits between $4,490–$4,680, stand down unless you're scalping
        price_position_in_range = (current_close_15m - current_lod) / current_range_width if current_range_width > 0 else 0.5
        logger.info(f"Price position in range: {price_position_in_range:.2%} (0% = LOD, 100% = HOD)")
        
        # Determine which path to prioritize based on direction filter
        if direction == 'LONG':
            logger.info("🎯 Direction filter: LONG only - prioritizing breakout strategies")
        elif direction == 'SHORT':
            logger.info("🎯 Direction filter: SHORT only - prioritizing breakdown strategies")
        else:  # BOTH
            logger.info("🎯 Direction filter: BOTH - monitoring all strategies")
        

        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check if any strategy is already triggered (do not run multiple strategies simultaneously)
        long_triggered = breakout_state.get("triggered", False) or retest_state.get("triggered", False)
        short_triggered = range_break_state.get("triggered", False)
        
        # 1. LONG - Breakout Strategy
        if long_strategies_enabled and not long_triggered:
            # Check if 15m close > today's high (4698) and volume confirmed
            breakout_trigger = current_close_15m > HOD_24H
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_close_15m <= BREAKOUT_ENTRY_HIGH
            breakout_ready = breakout_trigger and in_breakout_zone and volume_confirmed and not chop_filter_active and not breakout_whipsaw and not breakout_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • 15m close > today's high (${HOD_24H:,.0f}): {'✅' if breakout_trigger else '❌'} (current: ${current_close_15m:,.2f})")
            logger.info(f"   • Price in entry zone (${BREAKOUT_ENTRY_LOW:,.0f}-${BREAKOUT_ENTRY_HIGH:,.0f}): {'✅' if in_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (15M: {current_volume_15m/avg_volume_15m if avg_volume_15m > 0 else 0:.2f}x): {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if breakout_whipsaw else '✅ NO WHIPSAW'}")
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
                    entry_price=current_close_15m,
                    stop_loss=BREAKOUT_STOP_LOSS,
                    take_profit=BREAKOUT_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 LONG - Breakout trade executed successfully!")
                    logger.info(f"Entry: ${current_close_15m:,.2f}")
                    logger.info(f"Stop-loss: ${BREAKOUT_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${BREAKOUT_TP1:,.2f}")
                    logger.info(f"TP2: ${BREAKOUT_TP2_LOW:,.2f}-${BREAKOUT_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Expansion above intraday high with momentum; clear air toward the ATH pocket")
                    
                    # Save trigger state
                    breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_15m['start']),
                        "entry_price": current_close_15m
                    }
                    save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout trade failed: {trade_result}")
        
        # 2. LONG - Post-break retest Strategy (contingent on breakout first)
        if not trade_executed and long_strategies_enabled and not long_triggered:
            # Check if we first closed above today's high, then price retests 4680-4700
            breakout_occurred = breakout_state.get("triggered", False) or current_close_15m > HOD_24H
            in_retest_zone = 4680 <= current_close_15m <= 4700  # Retest zone as specified
            retest_ready = breakout_occurred and in_retest_zone and volume_confirmed and not chop_filter_active and not retest_whipsaw and not retest_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Post-break retest Strategy Analysis:")
            logger.info(f"   • Breakout occurred (15m close > ${HOD_24H:,.0f}): {'✅' if breakout_occurred else '❌'}")
            logger.info(f"   • Price in retest zone ($4,680-$4,700): {'✅' if in_retest_zone else '❌'} (current: ${current_close_15m:,.2f})")
            logger.info(f"   • Price in entry zone (${RETEST_ENTRY_LOW:,.0f}-${RETEST_ENTRY_HIGH:,.0f}): {'✅' if RETEST_ENTRY_LOW <= current_close_15m <= RETEST_ENTRY_HIGH else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if retest_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Already triggered: {'✅' if retest_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if retest_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Retest Ready: {'🎯 YES' if retest_ready else '⏳ NO'}")
            
            if retest_ready:
                logger.info("")
                logger.info("🎯 LONG - Post-break retest Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Retest trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD LONG Post-break retest",
                    entry_price=current_close_15m,
                    stop_loss=RETEST_STOP_LOSS,
                    take_profit=RETEST_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 LONG - Post-break retest trade executed successfully!")
                    logger.info(f"Entry: ${current_close_15m:,.2f}")
                    logger.info(f"Stop-loss: ${RETEST_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${RETEST_TP1:,.2f}")
                    logger.info(f"TP2: ${RETEST_TP2_LOW:,.2f}-${RETEST_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Classic break-and-retest toward the ATH cluster, avoids chasing the first impulse")
                    
                    # Save trigger state
                    retest_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_15m['start']),
                        "entry_price": current_close_15m
                    }
                    save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Retest trade failed: {trade_result}")
        
        # 3. SHORT - Range break lower Strategy
        if not trade_executed and short_strategies_enabled and not short_triggered:
            # Check if 15m close < today's low (4460) and volume confirmed
            range_break_trigger = current_close_15m < LOD_24H
            in_range_break_zone = RANGE_BREAK_ENTRY_LOW <= current_close_15m <= RANGE_BREAK_ENTRY_HIGH
            range_break_ready = range_break_trigger and in_range_break_zone and volume_confirmed and not chop_filter_active and not range_break_whipsaw and not range_break_state.get("stopped_out", False)
            
            logger.info("🔍 SHORT - Range break lower Strategy Analysis:")
            logger.info(f"   • 15m close < today's low (${LOD_24H:,.0f}): {'✅' if range_break_trigger else '❌'} (current: ${current_close_15m:,.2f})")
            logger.info(f"   • Price in entry zone (${RANGE_BREAK_ENTRY_LOW:,.0f}-${RANGE_BREAK_ENTRY_HIGH:,.0f}): {'✅' if in_range_break_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if range_break_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Already triggered: {'✅' if range_break_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if range_break_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Range break Ready: {'🎯 YES' if range_break_ready else '⏳ NO'}")
            
            if range_break_ready:
                logger.info("")
                logger.info("🎯 SHORT - Range break lower Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Range break trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Range break lower",
                    entry_price=current_close_15m,
                    stop_loss=RANGE_BREAK_STOP_LOSS,
                    take_profit=RANGE_BREAK_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Range break lower trade executed successfully!")
                    logger.info(f"Entry: ${current_close_15m:,.2f}")
                    logger.info(f"Stop-loss: ${RANGE_BREAK_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${RANGE_BREAK_TP1:,.2f}")
                    logger.info(f"TP2: ${RANGE_BREAK_TP2_LOW:,.2f}-${RANGE_BREAK_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Lose day's floor → opens path to prior liquidity shelf; good asymmetry if momentum accelerates")
                    
                    # Save trigger state
                    range_break_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_15m['start']),
                        "entry_price": current_close_15m
                    }
                    save_trigger_state(range_break_state, RANGE_BREAK_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range break trade failed: {trade_result}")
        
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
                    logger.info("   Post-break retest strategy already triggered")
            
            if short_strategies_enabled:
                if range_break_state.get("triggered", False):
                    logger.info("   Range break lower strategy already triggered")
        
        # Reset triggers if price moves significantly away from entry zones
        # Execution guardrails: If first entry stops, stand down until new 24h structure forms
        if breakout_state.get("triggered", False):
            if current_close_15m < BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Breakout trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                logger.info("Breakout trigger state reset - standing down")
        
        if retest_state.get("triggered", False):
            if current_close_15m < RETEST_STOP_LOSS:
                logger.info("🔄 Resetting Post-break retest trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                retest_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                logger.info("Post-break retest trigger state reset - standing down")
        
        if range_break_state.get("triggered", False):
            if current_close_15m > RANGE_BREAK_STOP_LOSS:
                logger.info("🔄 Resetting Range break lower trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                range_break_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(range_break_state, RANGE_BREAK_TRIGGER_FILE)
                logger.info("Range break lower trigger state reset - standing down")
        
        logger.info("=== ETH-USD Trading Strategy Alert completed ===")
        return current_ts_15m
        
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
        logger.info("Strategy: ETH Plan for Aug 15, 2025 - LONG & SHORT with Execution Guardrails")
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