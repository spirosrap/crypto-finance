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

# ETH Trading Strategy Parameters (based on new ETH plan for today)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_1H = "ONE_HOUR"  # 1-hour chart for context
GRANULARITY_15M = "FIFTEEN_MINUTE"  # 15-minute chart for execution
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (from plan)
CURRENT_ETH_PRICE = 4251.00
HOD_24H = 4569.00
LOD_24H = 4242.00
RANGE_WIDTH_24H = HOD_24H - LOD_24H
MID_RANGE_PIVOT = (HOD_24H + LOD_24H) / 2

# LONG SETUPS

# 1) Long — Breakout continuation
BREAKOUT_ENTRY_LOW = 4580  # Entry: 4,580–4,620 after a 1h close > 4,580 and hold on retest
BREAKOUT_ENTRY_HIGH = 4620
BREAKOUT_STOP_LOSS = 4510
BREAKOUT_TP1 = 4700
BREAKOUT_TP2_LOW = 4820
BREAKOUT_TP2_HIGH = 4900

# SHORT SETUPS

# 2) Short — Exhaustion fade into prior highs (repurposed FAILED_BREAKOUT constants)
FAILED_BREAKOUT_ENTRY_LOW = 4700  # Entry: 4,700–4,750 on 1h rejection (upper-wick + close back inside)
FAILED_BREAKOUT_ENTRY_HIGH = 4750
FAILED_BREAKOUT_STOP_LOSS = 4790
FAILED_BREAKOUT_TP1 = 4600
FAILED_BREAKOUT_TP2_LOW = 4450
FAILED_BREAKOUT_TP2_HIGH = 4450

# 3) Short — Breakdown trend short
RANGE_BREAK_ENTRY_LOW = 4190  # Entry: 4,210–4,190 after a 1h close < 4,210 and failed retest
RANGE_BREAK_ENTRY_HIGH = 4210
RANGE_BREAK_STOP_LOSS = 4260
RANGE_BREAK_TP1 = 4080
RANGE_BREAK_TP2_LOW = 4000
RANGE_BREAK_TP2_HIGH = 4000

# Volume confirmation requirements (global helpers)
VOLUME_SURGE_FACTOR_1H = 1.25
VOLUME_SURGE_FACTOR_15M = 0.9

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
MODE = "FAST"  # FAST = 5–15m close beyond trigger; CONSERVATIVE = 1h close
VCONF = True  # enforce volume rule

# Chop filter parameters
ATR_PERCENTAGE_THRESHOLD = 0.4  # Skip if 1h ATR% < 0.4
VOLUME_CHOP_FACTOR = 0.8  # and 5m vol < 0.8× average (chop filter)

# State files for each strategy
BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
PULLBACK_TRIGGER_FILE = "eth_pullback_trigger_state.json"
FAILED_BREAKOUT_TRIGGER_FILE = "eth_failed_breakout_trigger_state.json"
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
        
        # Fixed position size per requirement
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

def check_volume_confirmation(cb_service, current_volume_1h, current_volume_15m, avg_volume_1h, avg_volume_15m):
    """Check volume confirmation on both 1h and 15m timeframes"""
    volume_1h_confirmed = current_volume_1h >= (VOLUME_SURGE_FACTOR_1H * avg_volume_1h)
    volume_15m_confirmed = current_volume_15m <= (VOLUME_SURGE_FACTOR_15M * avg_volume_15m)  # For rejection trades, we want lower volume
    
    # Volume must be confirmed on either 1h OR 15m timeframe
    volume_confirmed = volume_1h_confirmed or volume_15m_confirmed
    
    logger.info(f"Volume confirmation check:")
    logger.info(f"  1H: {current_volume_1h:,.0f} vs {VOLUME_SURGE_FACTOR_1H}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_1h_confirmed else '❌'}")
    logger.info(f"  15M: {current_volume_15m:,.0f} vs ≤{VOLUME_SURGE_FACTOR_15M}x avg ({avg_volume_15m:,.0f}) -> {'✅' if volume_15m_confirmed else '❌'}")
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





# --- ETH Trading Strategy Alert Logic ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH-USD Trading Strategy Alert - Implements ETH plan for Thu, Aug 14, 2025
    Based on the trading plan: "Spiros — here's a clean, two-sided ETH plan for today (Thu, Aug 14, 2025)"
    
    Global rules (both directions):
    - Trigger TF: 1h close; execute: 5–15m
    - Volume: fire only if 1h vol ≥ 1.25× its 20-SMA or 5m vol ≥ 2× baseline at trigger
    - Risk: size so 1R ≈ 0.8–1.2% of price; partial at +1.2R, trail rest; invalidate on structure break
    - Position Size: Always margin x leverage = 250 x 20 = $5,000 USD
    - Skip signals that don't meet volume or that trigger within 5 minutes of each other (avoid chop)
    
    LONG SETUPS:
    1) Breakout continuation
       - Entry: buy-stop $4,805–$4,815 (clean through $4.8k & above today's high)
       - SL: $4,760 (back inside)
       - TP1/TP2: $4,880 / $4,940–$5,000
       - Why: round-number sweep + stop cluster above today's highs; needs expansion vol
    
    2) Higher-low retest
       - Entry zone: $4,700–$4,720 after pullback that holds > $4,680 with HL on 5–15m
       - SL: $4,660
       - TP1/TP2: $4,780 / $4,840–$4,860
       - Why: buy the base if demand absorbs under $4.72k; best R:R if trend remains intact
    
    SHORT SETUPS:
    3) Breakdown momentum
       - Entry: sell-stop $4,605–$4,595 (through today's low + buffer)
       - SL: $4,650
       - TP1/TP2: $4,540 / $4,480–$4,500
       - Why: range loss → liquidation run; confirm with impulse + rising 5m vol
    
    4) Lower-high rejection
       - Entry zone: $4,770–$4,780 only on rejection (bearish 1h candle or 5m failure >2× vol) below $4,800
       - SL: $4,805
       - TP1/TP2: $4,720 / $4,660
       - Why: fade the underside if $4.8k acts as a lid
    
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
    failed_breakout_state = {"triggered": False}
    range_break_state = load_trigger_state(RANGE_BREAK_TRIGGER_FILE)
    
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
        
        # Get current and last closed 1-hour candles
        current_candle_1h = candles_1h[-1]
        last_closed_1h = candles_1h[-2]
        current_close_1h = float(current_candle_1h['close'])
        current_high_1h = float(current_candle_1h['high'])
        current_low_1h = float(current_candle_1h['low'])
        current_volume_1h = float(current_candle_1h['volume'])
        current_ts_1h = datetime.fromtimestamp(int(current_candle_1h['start']), UTC)
        last_close_1h = float(last_closed_1h['close'])
        last_open_1h = float(last_closed_1h['open'])
        last_high_1h = float(last_closed_1h['high'])
        last_low_1h = float(last_closed_1h['low'])
        last_volume_1h = float(last_closed_1h['volume'])
        
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
            failed_breakout_state.get("last_hod", current_hod),
            range_break_state.get("last_hod", current_hod)
        )
        previous_lod = min(
            breakout_state.get("last_lod", current_lod),
            failed_breakout_state.get("last_lod", current_lod),
            range_break_state.get("last_lod", current_lod)
        )
        
        new_structure_formed, updated_hod, updated_lod = check_new_structure_formation(
            cb_service, current_ts_1h, previous_hod, previous_lod
        )
        
        if new_structure_formed:
            logger.info("🔄 New 24h structure formed - resetting stopped_out flags for all strategies")
            # Reset stopped_out flags for all strategies
            for strategy_name, state, state_file in [
                ("Breakout", breakout_state, BREAKOUT_TRIGGER_FILE),
                ("Failed Breakout", failed_breakout_state, FAILED_BREAKOUT_TRIGGER_FILE),
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
                                     (failed_breakout_state, FAILED_BREAKOUT_TRIGGER_FILE), 
                                     (range_break_state, RANGE_BREAK_TRIGGER_FILE)]:
                state["last_hod"] = current_hod
                state["last_lod"] = current_lod
                save_trigger_state(state, state_file)
        
        # Calculate 20-period average volume prior to last closed candle
        volume_candles_1h = candles_1h[-(VOLUME_PERIOD+2):-2] if len(candles_1h) >= VOLUME_PERIOD + 2 else candles_1h[:-2]
        avg_volume_1h = (sum(float(c['volume']) for c in volume_candles_1h) / len(volume_candles_1h)) if volume_candles_1h else 0
        
        # Get 15-minute candles for volume confirmation and retest analysis
        start_15m = now - timedelta(hours=2)
        start_ts_15m = int(start_15m.timestamp())
        end_ts_15m = int(now.timestamp())
        
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts_15m, end_ts_15m, GRANULARITY_15M)
        
        if candles_15m and len(candles_15m) >= VOLUME_PERIOD + 1:
            candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
            current_candle_15m = candles_15m[-1]
            current_volume_15m = float(current_candle_15m['volume'])
            
            # Calculate 20-period average volume for 15m (excluding current candle)
            volume_candles_15m = candles_15m[-(VOLUME_PERIOD+1):-1]
            avg_volume_15m = sum(float(c['volume']) for c in volume_candles_15m) / len(volume_candles_15m)
        else:
            current_volume_15m = 0
            avg_volume_15m = 0
        
        # Check volume confirmation
        volume_confirmed = check_volume_confirmation(cb_service, last_volume_1h, current_volume_15m, avg_volume_1h, avg_volume_15m)
        
        # Check chop filter conditions
        def check_chop_filter(candles_1h, current_volume_15m, avg_volume_15m):
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
                    volume_chop = current_volume_15m < (VOLUME_CHOP_FACTOR * avg_volume_15m) if avg_volume_15m > 0 else False
                    
                    # Chop filter: skip if 1h ATR% < 0.4 and 15m vol < 0.8× average
                    is_chop = atr_percentage < ATR_PERCENTAGE_THRESHOLD and volume_chop
                    
                    logger.info(f"Chop filter check: 1h ATR% = {atr_percentage:.2f}% (threshold: {ATR_PERCENTAGE_THRESHOLD}%)")
                    logger.info(f"Volume chop: 15m vol = {current_volume_15m:,.0f} vs {VOLUME_CHOP_FACTOR}x avg = {VOLUME_CHOP_FACTOR * avg_volume_15m:,.0f}")
                    logger.info(f"Chop filter: {'✅ SKIP' if is_chop else '❌ CONTINUE'}")
                    
                    return is_chop
                else:
                    logger.warning("Not enough data for chop filter calculation")
                    return False
            except Exception as e:
                logger.error(f"Error in chop filter calculation: {e}")
                return False
        
        # Apply chop filter
        chop_filter_active = check_chop_filter(candles_1h, current_volume_15m, avg_volume_15m)
        
        # Check for whipsaw conditions
        def check_whipsaw_condition(candles_15m, entry_level, is_long):
            """Check if trigger whipsaws and closes back inside the level within 15m"""
            try:
                # Check last 3 15m candles for whipsaw
                whipsaw_detected = False
                
                # Check 15m timeframe
                for candle in candles_15m[-3:]:
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
        
        # Check whipsaw for key triggers
        breakout_whipsaw = check_whipsaw_condition(candles_15m, BREAKOUT_ENTRY_LOW, True)
        range_break_whipsaw = check_whipsaw_condition(candles_15m, RANGE_BREAK_ENTRY_HIGH, False)
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros — Clean ETH setups for today (ETH)")
        logger.info("")
        logger.info(f"📊 Live Levels (price ≈ {CURRENT_ETH_PRICE:,.0f}. Intraday range: {LOD_24H:,.0f}–{HOD_24H:,.0f}):")
        logger.info(f"   • ETH ≈ ${current_close_1h:,.0f}")
        logger.info(f"   • 24h HOD: ${current_hod:,.0f}")
        logger.info(f"   • 24h LOD: ${current_lod:,.0f}")
        logger.info(f"   • MID: ${current_mid_range:,.0f}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Trigger TF: 1h; execute/manage on 1h/4h with 15m checks")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Execution: 0.5–1.0R partial at TP1, move SL to breakeven after TP1. Avoid overlap (one long and one short max).")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG SETUPS:")
            logger.info("")
            logger.info("1) Long — Range Breakout → Prior 7d supply")
            logger.info(f"   • Entry zone: ${BREAKOUT_ENTRY_LOW:,.0f}–${BREAKOUT_ENTRY_HIGH:,.0f} after a 1h close above $4,491 (24h high)")
            logger.info(f"   • Stop: ${BREAKOUT_STOP_LOSS:,.0f} (below breakout bar / range top)")
            logger.info(f"   • First target: ${BREAKOUT_TP1:,.0f}; stretch: ${BREAKOUT_TP2_LOW:,.0f}–${BREAKOUT_TP2_HIGH:,.0f} (7d supply area)")
            logger.info(f"   • Why: Continuation from multi-day uptrend; fresh highs above 24h range with room toward last week's supply band")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT SETUPS:")
            logger.info("")
            logger.info("2) Short — Fail/reject at range highs")
            logger.info(f"   • Entry zone: ${FAILED_BREAKOUT_ENTRY_LOW:,.0f}–${FAILED_BREAKOUT_ENTRY_HIGH:,.0f} on wick rejection + 15m lower-high confirmation")
            logger.info(f"   • Stop: ${FAILED_BREAKOUT_STOP_LOSS:,.0f}")
            logger.info(f"   • First target: ${FAILED_BREAKOUT_TP1:,.0f}; stretch: ${FAILED_BREAKOUT_TP2_LOW:,.0f}")
            logger.info(f"   • Why: Fade the 24h high if breakout fails; mean-reversion back into the intraday range")
            logger.info("")
            logger.info("3) Short — Breakdown → Range low loss")
            logger.info(f"   • Entry zone: ${RANGE_BREAK_ENTRY_LOW:,.0f}–${RANGE_BREAK_ENTRY_HIGH:,.0f} after a 1h close below $4,386 (24h low), preferably on a weak retest from beneath")
            logger.info(f"   • Stop: ${RANGE_BREAK_STOP_LOSS:,.0f}")
            logger.info(f"   • First target: ${RANGE_BREAK_TP1:,.0f}; stretch: ${RANGE_BREAK_TP2_LOW:,.0f}")
            logger.info(f"   • Why: Loss of 24h low opens downside toward prior intraday demand; momentum flip confirmed by retest failure")
            logger.info("")
        logger.info("")
        logger.info(f"Current Price: ${current_close_1h:,.2f}")
        logger.info(f"Last 1H (closed): ${last_close_1h:,.2f}, High: ${last_high_1h:,.2f}, Low: ${last_low_1h:,.2f}")
        logger.info(f"1H Volume: {last_volume_1h:,.0f}, 1H SMA(20 prior): {avg_volume_1h:,.0f}, Rel_Vol: {last_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}")
        logger.info(f"15M Volume: {current_volume_15m:,.0f}, 15M SMA: {avg_volume_15m:,.0f}, Rel_Vol: {current_volume_15m/avg_volume_15m if avg_volume_15m > 0 else 0:.2f}")
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
            long_triggered = breakout_state.get("triggered", False)
            short_triggered = failed_breakout_state.get("triggered", False) or range_break_state.get("triggered", False)
            
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
        
        # (Exhaustion fade replaces older failed_breakout_condition checks)
        
        # SHORT (breakdown momentum) Strategy Conditions
        range_break_condition = (
            breakdown_priority and
            current_close_1h <= RANGE_BREAK_ENTRY_HIGH and 
            current_close_1h >= RANGE_BREAK_ENTRY_LOW and 
            volume_confirmed and 
            not chop_filter_active and  # Skip if chop filter is active
            not range_break_whipsaw and  # Skip if whipsaw detected
            not range_break_state.get("triggered", False) and
            not range_break_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy
        if long_strategies_enabled:
            in_breakout_zone = BREAKOUT_ENTRY_LOW <= current_close_1h <= BREAKOUT_ENTRY_HIGH
            breakout_ready = in_breakout_zone and volume_confirmed and not chop_filter_active and not breakout_whipsaw and breakout_priority and not breakout_state.get("triggered", False) and not breakout_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${BREAKOUT_ENTRY_LOW:,.0f}-${BREAKOUT_ENTRY_HIGH:,.0f}): {'✅' if in_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H: {current_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x, 15M: {current_volume_15m/avg_volume_15m if avg_volume_15m > 0 else 0:.2f}x): {'✅' if volume_confirmed else '❌'}")
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
                    logger.info("Strategy: round-number sweep + stop cluster above today's highs; needs expansion vol")
                    
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
        

        
        # 2. SHORT - Exhaustion fade into prior highs Strategy
        if not trade_executed and short_strategies_enabled:
            # Define 1h rejection (upper wick + close back inside) with high inside entry zone
            total_range = max(last_high_1h - last_low_1h, 1e-6)
            upper_wick = max(last_high_1h - max(last_open_1h, last_close_1h), 0)
            upper_wick_ratio = upper_wick / total_range
            in_failed_breakout_zone = FAILED_BREAKOUT_ENTRY_LOW <= last_high_1h <= FAILED_BREAKOUT_ENTRY_HIGH
            failed_breakout_ready = in_failed_breakout_zone and (upper_wick_ratio >= 0.35) and (last_close_1h < last_high_1h)
            
            logger.info("🔍 SHORT - Exhaustion fade Strategy Analysis:")
            logger.info(f"   • 1h high in ${FAILED_BREAKOUT_ENTRY_LOW:,.0f}-${FAILED_BREAKOUT_ENTRY_HIGH:,.0f}: {'✅' if in_failed_breakout_zone else '❌'}")
            logger.info(f"   • Upper wick ratio ≥ 35%: {'✅' if upper_wick_ratio >= 0.35 else '❌'} ({upper_wick_ratio:.2f})")
            logger.info(f"   • Close back inside range: {'✅' if last_close_1h < last_high_1h else '❌'}")
            
            if failed_breakout_ready:
                logger.info("")
                logger.info("🎯 SHORT - Exhaustion fade conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Exhaustion fade trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Exhaustion fade",
                    entry_price=current_close_1h,
                    stop_loss=FAILED_BREAKOUT_STOP_LOSS,
                    take_profit=FAILED_BREAKOUT_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Exhaustion fade trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${FAILED_BREAKOUT_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${FAILED_BREAKOUT_TP1:,.2f}")
                    logger.info(f"TP2: ${FAILED_BREAKOUT_TP2_LOW:,.2f}-${FAILED_BREAKOUT_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Sell the first test of the Aug 12–13 swing high supply (~4,749)")
                    
                    # Save trigger state
                    failed_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(failed_breakout_state, FAILED_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Exhaustion fade trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown → Range low loss Strategy
        if not trade_executed and short_strategies_enabled:
            in_range_break_zone = RANGE_BREAK_ENTRY_LOW <= current_close_1h <= RANGE_BREAK_ENTRY_HIGH
            range_break_ready = in_range_break_zone and volume_confirmed and not chop_filter_active and not range_break_whipsaw and breakdown_priority and not range_break_state.get("triggered", False) and not range_break_state.get("stopped_out", False)
            
            logger.info("🔍 SHORT - Breakdown → Range low loss Strategy Analysis:")
            logger.info(f"   • Price in entry zone (${RANGE_BREAK_ENTRY_LOW:,.0f}-${RANGE_BREAK_ENTRY_HIGH:,.0f}): {'✅' if in_range_break_zone else '❌'}")
            logger.info(f"   • Volume confirmed: {'✅' if volume_confirmed else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if range_break_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if range_break_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if range_break_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Breakdown → Range low loss Ready: {'🎯 YES' if range_break_ready else '⏳ NO'}")
            
            if range_break_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown → Range low loss Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown → Range low loss trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD SHORT Breakdown → Range low loss",
                    entry_price=current_close_1h,
                    stop_loss=RANGE_BREAK_STOP_LOSS,
                    take_profit=RANGE_BREAK_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed
                )
                
                if trade_success:
                    logger.info("🎉 SHORT - Breakdown → Range low loss trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${RANGE_BREAK_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${RANGE_BREAK_TP1:,.2f}")
                    logger.info(f"TP2: ${RANGE_BREAK_TP2_LOW:,.2f}-${RANGE_BREAK_TP2_HIGH:,.2f}")
                    logger.info("Strategy: Loss of 24h low opens downside toward prior intraday demand; momentum flip confirmed by retest failure")
                    
                    # Save trigger state
                    range_break_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(range_break_state, RANGE_BREAK_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown → Range low loss trade failed: {trade_result}")
        

        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions…")
        
        # Reset triggers if price moves significantly away from entry zones
        # Execution guardrails: If first entry stops, stand down until new 24h structure forms
        if breakout_state.get("triggered", False):
            if current_close_1h < BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Breakout trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                logger.info("Breakout trigger state reset - standing down")
        

        
        if range_break_state.get("triggered", False):
            if current_close_1h > RANGE_BREAK_STOP_LOSS:
                logger.info("🔄 Resetting Breakdown → Range low loss trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                range_break_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(range_break_state, RANGE_BREAK_TRIGGER_FILE)
                logger.info("Breakdown → Range low loss trigger state reset - standing down")
        
        if failed_breakout_state.get("triggered", False):
            if current_close_1h > FAILED_BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Exhaustion fade state - SL hit")
                failed_breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
                save_trigger_state(failed_breakout_state, FAILED_BREAKOUT_TRIGGER_FILE)
        
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
        logger.info("Strategy: Clean ETH setups for today - LONG & SHORT with Execution Guardrails")
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