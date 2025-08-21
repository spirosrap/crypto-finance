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
GRANULARITY_30M = "THIRTY_MINUTE"  # 30-minute chart for execution
GRANULARITY_15M = "FIFTEEN_MINUTE"  # 15-minute chart for volume analysis
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (from new ETH plan - Jan 2025)
CURRENT_ETH_PRICE = 4310.00  # ETH now ~$4,310
HOD_24H = 4370.84  # Today's high: $4,370.84
LOD_24H = 4114.35  # Today's low: $4,114.35
RANGE_WIDTH_24H = HOD_24H - LOD_24H
MID_RANGE_PIVOT = (HOD_24H + LOD_24H) / 2
HALFBACK_LEVEL = 4242  # 50% intraday level
FIB_618_LEVEL = 4273  # .618 fib level

# LONG SETUPS

# 1) Breakout long — above H
LONG_BREAKOUT_TRIGGER = 4371  # Trigger: 1h close > $4,371
LONG_BREAKOUT_ENTRY_LOW = 4375  # Entry: $4,375–$4,391
LONG_BREAKOUT_ENTRY_HIGH = 4391
LONG_BREAKOUT_STOP_LOSS = 4345  # SL: $4,345
LONG_BREAKOUT_TP1 = 4452  # TP1: $4,452

# 2) Halfback pullback long
LONG_PULLBACK_TRIGGER = 4242  # Trigger: reclaim of 50% intraday at ~$4,242
LONG_PULLBACK_ENTRY_LOW = 4230  # Entry: $4,230–$4,255 after a bullish rejection wick
LONG_PULLBACK_ENTRY_HIGH = 4255
LONG_PULLBACK_STOP_LOSS = 4198  # SL: $4,198
LONG_PULLBACK_TP1 = 4310  # TP1: $4,310

# SHORT SETUPS

# 3) Breakdown short — below L
SHORT_BREAKDOWN_TRIGGER = 4114  # Trigger: 1h close < $4,114
SHORT_BREAKDOWN_ENTRY_LOW = 4096  # Entry: $4,096–$4,110
SHORT_BREAKDOWN_ENTRY_HIGH = 4110
SHORT_BREAKDOWN_STOP_LOSS = 4139  # SL: $4,139
SHORT_BREAKDOWN_TP1 = 4038  # TP1: $4,038

# 4) .618 fade short
SHORT_FADE_TRIGGER = 4273  # Trigger: rejection at ~$4,273 (.618 of today's move)
SHORT_FADE_ENTRY_LOW = 4273  # Entry: $4,273–$4,298
SHORT_FADE_ENTRY_HIGH = 4298
SHORT_FADE_STOP_LOSS = 4316  # SL: $4,316
SHORT_FADE_TP1 = 4215  # TP1: $4,215

# Volume confirmation requirements (from new plan)
VOLUME_SURGE_FACTOR_1H = 1.3  # 1h RVOL ≥1.3 vs 20-bar for breakouts
VOLUME_CONTROLLED_FACTOR_LOW = 0.9  # RVOL 0.9–1.1 (controlled rotation)
VOLUME_CONTROLLED_FACTOR_HIGH = 1.1
VOLUME_FADE_FACTOR = 0.9  # RVOL ≤0.9 for fade shorts

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
LONG_BREAKOUT_TRIGGER_FILE = "eth_long_breakout_trigger_state.json"
LONG_PULLBACK_TRIGGER_FILE = "eth_long_pullback_trigger_state.json"
SHORT_BREAKDOWN_TRIGGER_FILE = "eth_short_breakdown_trigger_state.json"
SHORT_FADE_TRIGGER_FILE = "eth_short_fade_trigger_state.json"

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

def check_volume_confirmation(cb_service, current_volume_1h, current_volume_30m, current_volume_15m, avg_volume_1h, avg_volume_30m, avg_volume_15m, strategy_type="default"):
    """Check volume confirmation based on strategy type (updated for new plan)"""
    if strategy_type == "long_breakout":
        # Breakout long: 1h RVOL ≥1.3 vs 20-bar
        volume_1h_confirmed = current_volume_1h >= (VOLUME_SURGE_FACTOR_1H * avg_volume_1h)
        volume_confirmed = volume_1h_confirmed
        logger.info(f"Volume confirmation check (Breakout Long):")
        logger.info(f"  1H: {current_volume_1h:,.0f} vs ≥{VOLUME_SURGE_FACTOR_1H}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_1h_confirmed else '❌'}")
    elif strategy_type == "short_breakdown":
        # Breakdown short: 1h RVOL ≥1.3 with sell imbalance
        volume_1h_confirmed = current_volume_1h >= (VOLUME_SURGE_FACTOR_1H * avg_volume_1h)
        volume_confirmed = volume_1h_confirmed
        logger.info(f"Volume confirmation check (Breakdown Short):")
        logger.info(f"  1H: {current_volume_1h:,.0f} vs ≥{VOLUME_SURGE_FACTOR_1H}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_1h_confirmed else '❌'}")
    elif strategy_type == "long_pullback":
        # Halfback pullback: RVOL 0.9–1.1 (controlled rotation)
        volume_controlled = (current_volume_1h >= (VOLUME_CONTROLLED_FACTOR_LOW * avg_volume_1h) and 
                           current_volume_1h <= (VOLUME_CONTROLLED_FACTOR_HIGH * avg_volume_1h))
        volume_confirmed = volume_controlled
        logger.info(f"Volume confirmation check (Halfback Pullback):")
        logger.info(f"  1H: {current_volume_1h:,.0f} vs {VOLUME_CONTROLLED_FACTOR_LOW}-{VOLUME_CONTROLLED_FACTOR_HIGH}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_controlled else '❌'}")
    elif strategy_type == "short_fade":
        # .618 fade: RVOL ≤0.9 and 1h RSI fails to hold 50
        volume_fading = current_volume_1h <= (VOLUME_FADE_FACTOR * avg_volume_1h)
        volume_confirmed = volume_fading
        logger.info(f"Volume confirmation check (.618 Fade Short):")
        logger.info(f"  1H: {current_volume_1h:,.0f} vs ≤{VOLUME_FADE_FACTOR}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_fading else '❌'}")
        # Note: RSI condition will be checked separately in strategy analysis
    else:
        # Default: 1h volume ≥ 1.3× 20-MA
        volume_1h_confirmed = current_volume_1h >= (VOLUME_SURGE_FACTOR_1H * avg_volume_1h)
        volume_confirmed = volume_1h_confirmed
        logger.info(f"Volume confirmation check ({strategy_type}):")
        logger.info(f"  1H: {current_volume_1h:,.0f} vs ≥{VOLUME_SURGE_FACTOR_1H}x avg ({avg_volume_1h:,.0f}) -> {'✅' if volume_1h_confirmed else '❌'}")
    
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
    ETH-USD Trading Strategy Alert - Updated January 2025
    
    Spiros, ETH now ~$4,310. Today's H/L: $4,370.84 / $4,114.35.
    
    Long setups:
    • Breakout long — above H
        • Trigger: 1h close > $4,371
        • Entry: $4,375–$4,391
        • SL: $4,345
        • TP1: $4,452
        • Timeframe: confirm on 1h, execute on 5–15m
        • Volume: 1h RVOL ≥1.3 vs 20-bar
        • Why: fresh price discovery after 24h range high. Targets ~1.8R from mid-entry.
    
    • Halfback pullback long
        • Trigger: reclaim of 50% intraday at ~$4,242
        • Entry: $4,230–$4,255 after a bullish rejection wick
        • SL: $4,198
        • TP1: $4,310
        • Timeframe: 15m entry, 1h management
        • Volume: RVOL 0.9–1.1 (controlled rotation)
        • Why: mean-reversion to VWAP/value near 50% of today's range.
    
    Short setups:
    • Breakdown short — below L
        • Trigger: 1h close < $4,114
        • Entry: $4,096–$4,110
        • SL: $4,139
        • TP1: $4,038
        • Timeframe: confirm on 1h, execute on 5–15m
        • Volume: 1h RVOL ≥1.3 with sell imbalance
        • Why: continuation after range loss. Targets ~1.8R.
    
    • .618 fade short
        • Trigger: rejection at ~$4,273 (.618 of today's move)
        • Entry: $4,273–$4,298
        • SL: $4,316
        • TP1: $4,215
        • Timeframe: 5–15m
        • Volume: RVOL ≤0.9 and 1h RSI fails to hold 50
        • Why: counter-trend fade into fib supply.
    
    Use fixed size. Risk ≤1R per idea. If funding/OI spike one-sided, deprioritize trades against that flow.
    Position size should always be margin×leverage = 250×20 = $5,000 USD
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== ETH-USD Trading Strategy Alert (Updated January 2025 - LONG & SHORT) ===")
    else:
        logger.info(f"=== ETH-USD Trading Strategy Alert (Updated January 2025 - {direction} Strategy Only) ===")
    
    # Load trigger states for all strategies
    long_breakout_state = load_trigger_state(LONG_BREAKOUT_TRIGGER_FILE)
    long_pullback_state = load_trigger_state(LONG_PULLBACK_TRIGGER_FILE)
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    short_fade_state = load_trigger_state(SHORT_FADE_TRIGGER_FILE)
    
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
            long_breakout_state.get("last_hod", current_hod),
            long_pullback_state.get("last_hod", current_hod),
            short_breakdown_state.get("last_hod", current_hod),
            short_fade_state.get("last_hod", current_hod)
        )
        previous_lod = min(
            long_breakout_state.get("last_lod", current_lod),
            long_pullback_state.get("last_lod", current_lod),
            short_breakdown_state.get("last_lod", current_lod),
            short_fade_state.get("last_lod", current_lod)
        )
        
        new_structure_formed, updated_hod, updated_lod = check_new_structure_formation(
            cb_service, current_ts_1h, previous_hod, previous_lod
        )
        
        if new_structure_formed:
            logger.info("🔄 New 24h structure formed - resetting stopped_out flags for all strategies")
            # Reset stopped_out flags for all strategies
            for strategy_name, state, state_file in [
                ("Long Breakout", long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE),
                ("Long Pullback", long_pullback_state, LONG_PULLBACK_TRIGGER_FILE),
                ("Short Breakdown", short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE),
                ("Short Fade", short_fade_state, SHORT_FADE_TRIGGER_FILE)
            ]:
                if state.get("stopped_out", False):
                    state["stopped_out"] = False
                    state["last_hod"] = updated_hod
                    state["last_lod"] = updated_lod
                    save_trigger_state(state, state_file)
                    logger.info(f"✅ Reset stopped_out flag for {strategy_name} strategy")
        else:
            # Update last HOD/LOD in all states
            for state, state_file in [(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE), 
                                     (long_pullback_state, LONG_PULLBACK_TRIGGER_FILE), 
                                     (short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE),
                                     (short_fade_state, SHORT_FADE_TRIGGER_FILE)]:
                state["last_hod"] = current_hod
                state["last_lod"] = current_lod
                save_trigger_state(state, state_file)
        
        # Calculate 20-period average volume prior to last closed candle
        volume_candles_1h = candles_1h[-(VOLUME_PERIOD+2):-2] if len(candles_1h) >= VOLUME_PERIOD + 2 else candles_1h[:-2]
        avg_volume_1h = (sum(float(c['volume']) for c in volume_candles_1h) / len(volume_candles_1h)) if volume_candles_1h else 0
        
        # Calculate RSI for .618 fade short strategy
        def calculate_rsi(candles, period=14):
            """Calculate RSI using pandas_ta"""
            try:
                if len(candles) < period + 1:
                    return None
                
                # Convert candles to DataFrame
                df_data = []
                for candle in candles[-(period + 10):]:  # Get extra data for RSI calculation
                    df_data.append({
                        'close': float(candle['close']),
                        'timestamp': int(candle['start'])
                    })
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('timestamp')
                
                # Calculate RSI
                rsi_series = ta.rsi(df['close'], length=period)
                return rsi_series.iloc[-1] if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else None
            except Exception as e:
                logger.error(f"Error calculating RSI: {e}")
                return None
        
        # Get current RSI
        current_rsi = calculate_rsi(candles_1h, 14)
        logger.info(f"Current 1h RSI(14): {current_rsi:.2f}" if current_rsi is not None else "Current 1h RSI(14): N/A")
        
        # Get 30-minute candles for breakdown strategy
        start_30m = now - timedelta(hours=2)
        start_ts_30m = int(start_30m.timestamp())
        end_ts_30m = int(now.timestamp())
        
        candles_30m = safe_get_candles(cb_service, PRODUCT_ID, start_ts_30m, end_ts_30m, GRANULARITY_30M)
        
        if candles_30m and len(candles_30m) >= VOLUME_PERIOD + 1:
            candles_30m = sorted(candles_30m, key=lambda x: int(x['start']))
            current_candle_30m = candles_30m[-1]
            current_volume_30m = float(current_candle_30m['volume'])
            
            # Calculate 20-period average volume for 30m (excluding current candle)
            volume_candles_30m = candles_30m[-(VOLUME_PERIOD+1):-1]
            avg_volume_30m = sum(float(c['volume']) for c in volume_candles_30m) / len(volume_candles_30m)
        else:
            current_volume_30m = 0
            avg_volume_30m = 0
        
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
        
        # Check volume confirmation for different strategies
        volume_confirmed_long_breakout = check_volume_confirmation(cb_service, last_volume_1h, current_volume_30m, current_volume_15m, avg_volume_1h, avg_volume_30m, avg_volume_15m, "long_breakout")
        volume_confirmed_long_pullback = check_volume_confirmation(cb_service, last_volume_1h, current_volume_30m, current_volume_15m, avg_volume_1h, avg_volume_30m, avg_volume_15m, "long_pullback")
        volume_confirmed_short_breakdown = check_volume_confirmation(cb_service, last_volume_1h, current_volume_30m, current_volume_15m, avg_volume_1h, avg_volume_30m, avg_volume_15m, "short_breakdown")
        volume_confirmed_short_fade = check_volume_confirmation(cb_service, last_volume_1h, current_volume_30m, current_volume_15m, avg_volume_1h, avg_volume_30m, avg_volume_15m, "short_fade")
        
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
        chop_filter_active = check_chop_filter(candles_1h, current_volume_30m, avg_volume_30m)
        
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
        long_breakout_whipsaw = check_whipsaw_condition(candles_15m, LONG_BREAKOUT_ENTRY_LOW, True)
        long_pullback_whipsaw = check_whipsaw_condition(candles_15m, LONG_PULLBACK_ENTRY_HIGH, True)
        short_breakdown_whipsaw = check_whipsaw_condition(candles_15m, SHORT_BREAKDOWN_ENTRY_HIGH, False)
        short_fade_whipsaw = check_whipsaw_condition(candles_15m, SHORT_FADE_ENTRY_HIGH, False)
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, ETH now ~$4,310. Today's H/L: $4,370.84 / $4,114.35.")
        logger.info("")
        logger.info(f"📊 Live Levels (current ETH intraday H/L: {current_hod:,.2f} / {current_lod:,.2f}):")
        logger.info(f"   • ETH ≈ ${current_close_1h:,.0f}")
        logger.info(f"   • Today's H: ${current_hod:,.0f}")
        logger.info(f"   • Today's L: ${current_lod:,.0f}")
        logger.info(f"   • 50% intraday: ${HALFBACK_LEVEL:,.0f}")
        logger.info(f"   • .618 fib: ${FIB_618_LEVEL:,.0f}")
        logger.info("")
        logger.info("📊 Global Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Volume: Strategy-specific volume requirements")
        logger.info(f"   • Cancel signals during low-liquidity chop; prefer confluence with BTC trend")
        logger.info("")
        
        # Show only relevant strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 Long setups")
            logger.info("")
            logger.info("• Breakout long — above H")
            logger.info(f"   • Trigger: 1h close > ${LONG_BREAKOUT_TRIGGER:,.0f}")
            logger.info(f"   • Entry: ${LONG_BREAKOUT_ENTRY_LOW:,.0f}–${LONG_BREAKOUT_ENTRY_HIGH:,.0f}")
            logger.info(f"   • SL: ${LONG_BREAKOUT_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${LONG_BREAKOUT_TP1:,.0f}")
            logger.info(f"   • Timeframe: confirm on 1h, execute on 5–15m")
            logger.info(f"   • Volume: 1h RVOL ≥1.3 vs 20-bar")
            logger.info(f"   • Why: fresh price discovery after 24h range high. Targets ~1.8R from mid-entry.")
            logger.info("")
            logger.info("• Halfback pullback long")
            logger.info(f"   • Trigger: reclaim of 50% intraday at ~${LONG_PULLBACK_TRIGGER:,.0f}")
            logger.info(f"   • Entry: ${LONG_PULLBACK_ENTRY_LOW:,.0f}–${LONG_PULLBACK_ENTRY_HIGH:,.0f} after a bullish rejection wick")
            logger.info(f"   • SL: ${LONG_PULLBACK_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${LONG_PULLBACK_TP1:,.0f}")
            logger.info(f"   • Timeframe: 15m entry, 1h management")
            logger.info(f"   • Volume: RVOL 0.9–1.1 (controlled rotation)")
            logger.info(f"   • Why: mean-reversion to VWAP/value near 50% of today's range.")
            logger.info("")
        if short_strategies_enabled:
            logger.info("📊 Short setups")
            logger.info("")
            logger.info("• Breakdown short — below L")
            logger.info(f"   • Trigger: 1h close < ${SHORT_BREAKDOWN_TRIGGER:,.0f}")
            logger.info(f"   • Entry: ${SHORT_BREAKDOWN_ENTRY_LOW:,.0f}–${SHORT_BREAKDOWN_ENTRY_HIGH:,.0f}")
            logger.info(f"   • SL: ${SHORT_BREAKDOWN_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${SHORT_BREAKDOWN_TP1:,.0f}")
            logger.info(f"   • Timeframe: confirm on 1h, execute on 5–15m")
            logger.info(f"   • Volume: 1h RVOL ≥1.3 with sell imbalance")
            logger.info(f"   • Why: continuation after range loss. Targets ~1.8R.")
            logger.info("")
            logger.info("• .618 fade short")
            logger.info(f"   • Trigger: rejection at ~${SHORT_FADE_TRIGGER:,.0f} (.618 of today's move)")
            logger.info(f"   • Entry: ${SHORT_FADE_ENTRY_LOW:,.0f}–${SHORT_FADE_ENTRY_HIGH:,.0f}")
            logger.info(f"   • SL: ${SHORT_FADE_STOP_LOSS:,.0f}")
            logger.info(f"   • TP1: ${SHORT_FADE_TP1:,.0f}")
            logger.info(f"   • Timeframe: 5–15m")
            logger.info(f"   • Volume: RVOL ≤0.9 and 1h RSI fails to hold 50")
            logger.info(f"   • Why: counter-trend fade into fib supply.")
            logger.info("")
        logger.info("")
        logger.info(f"Current Price: ${current_close_1h:,.2f}")
        logger.info(f"Last 1H (closed): ${last_close_1h:,.2f}, High: ${last_high_1h:,.2f}, Low: ${last_low_1h:,.2f}")
        logger.info(f"1H Volume: {last_volume_1h:,.0f}, 1H SMA(20 prior): {avg_volume_1h:,.0f}, Rel_Vol: {last_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}")
        logger.info(f"30M Volume: {current_volume_30m:,.0f}, 30M SMA: {avg_volume_30m:,.0f}, Rel_Vol: {current_volume_30m/avg_volume_30m if avg_volume_30m > 0 else 0:.2f}")
        logger.info(f"15M Volume: {current_volume_15m:,.0f}, 15M SMA: {avg_volume_15m:,.0f}, Rel_Vol: {current_volume_15m/avg_volume_15m if avg_volume_15m > 0 else 0:.2f}")
        logger.info("")
        logger.info("Use fixed size. Risk ≤1R per idea. If funding/OI spike one-sided, deprioritize trades against that flow.")
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
            long_triggered = long_breakout_state.get("triggered", False) or long_pullback_state.get("triggered", False)
            short_triggered = short_breakdown_state.get("triggered", False) or short_fade_state.get("triggered", False)
            
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
        
        # LONG (Breakout) Strategy Conditions - updated for new plan
        long_breakout_condition = (
            breakout_priority and
            last_close_1h > LONG_BREAKOUT_TRIGGER and  # 1h close > $4,371
            current_close_1h >= LONG_BREAKOUT_ENTRY_LOW and 
            current_close_1h <= LONG_BREAKOUT_ENTRY_HIGH and 
            volume_confirmed_long_breakout and 
            not chop_filter_active and  # Skip if chop filter is active
            not long_breakout_whipsaw and  # Skip if whipsaw detected
            not long_breakout_state.get("triggered", False) and
            not long_breakout_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # LONG (Pullback) Strategy Conditions - updated for halfback pullback
        long_pullback_condition = (
            breakout_priority and
            current_close_1h > LONG_PULLBACK_TRIGGER and  # reclaim of 50% intraday
            current_close_1h >= LONG_PULLBACK_ENTRY_LOW and 
            current_close_1h <= LONG_PULLBACK_ENTRY_HIGH and 
            volume_confirmed_long_pullback and 
            not chop_filter_active and  # Skip if chop filter is active
            not long_pullback_whipsaw and  # Skip if whipsaw detected
            not long_pullback_state.get("triggered", False) and
            not long_pullback_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # SHORT (Breakdown) Strategy Conditions - updated for new plan
        short_breakdown_condition = (
            breakdown_priority and
            last_close_1h < SHORT_BREAKDOWN_TRIGGER and  # 1h close < $4,114
            current_close_1h >= SHORT_BREAKDOWN_ENTRY_LOW and 
            current_close_1h <= SHORT_BREAKDOWN_ENTRY_HIGH and 
            volume_confirmed_short_breakdown and 
            not chop_filter_active and  # Skip if chop filter is active
            not short_breakdown_whipsaw and  # Skip if whipsaw detected
            not short_breakdown_state.get("triggered", False) and
            not short_breakdown_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # SHORT (.618 Fade) Strategy Conditions - updated for new plan
        rsi_fails_50 = current_rsi is not None and current_rsi < 50
        short_fade_condition = (
            breakdown_priority and
            current_close_1h >= SHORT_FADE_ENTRY_LOW and 
            current_close_1h <= SHORT_FADE_ENTRY_HIGH and 
            current_high_1h >= SHORT_FADE_TRIGGER and  # rejection at .618 level
            volume_confirmed_short_fade and 
            rsi_fails_50 and  # 1h RSI fails to hold 50
            not chop_filter_active and  # Skip if chop filter is active
            not short_fade_whipsaw and  # Skip if whipsaw detected
            not short_fade_state.get("triggered", False) and
            not short_fade_state.get("stopped_out", False)  # Don't re-enter if stopped out
        )
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy (updated for new plan)
        if long_strategies_enabled:
            trigger_condition = last_close_1h > LONG_BREAKOUT_TRIGGER
            in_long_breakout_zone = LONG_BREAKOUT_ENTRY_LOW <= current_close_1h <= LONG_BREAKOUT_ENTRY_HIGH
            long_breakout_ready = trigger_condition and in_long_breakout_zone and volume_confirmed_long_breakout and not chop_filter_active and not long_breakout_whipsaw and breakout_priority and not long_breakout_state.get("triggered", False) and not long_breakout_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • 1h close > ${LONG_BREAKOUT_TRIGGER:,.0f}: {'✅' if trigger_condition else '❌'} (last close: {last_close_1h:,.0f})")
            logger.info(f"   • Price in entry zone (${LONG_BREAKOUT_ENTRY_LOW:,.0f}-${LONG_BREAKOUT_ENTRY_HIGH:,.0f}): {'✅' if in_long_breakout_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H RVOL: {last_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x): {'✅' if volume_confirmed_long_breakout else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if long_breakout_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakout_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if long_breakout_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if long_breakout_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Breakout Long Ready: {'🎯 YES' if long_breakout_ready else '⏳ NO'}")
            
            if long_breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Long Breakout trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD Breakout Long",
                    entry_price=current_close_1h,
                    stop_loss=LONG_BREAKOUT_STOP_LOSS,
                    take_profit=LONG_BREAKOUT_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed_long_breakout
                )
                
                if trade_success:
                    logger.info("🎉 Breakout Long trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_BREAKOUT_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_BREAKOUT_TP1:,.2f}")
                    logger.info("Strategy: Fresh price discovery after 24h range high")
                    
                    # Save trigger state
                    long_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Breakout trade failed: {trade_result}")
        

        
        # 2. LONG - Halfback Pullback Strategy (updated for new plan)
        if not trade_executed and long_strategies_enabled:
            pullback_trigger_condition = current_close_1h > LONG_PULLBACK_TRIGGER
            in_long_pullback_zone = LONG_PULLBACK_ENTRY_LOW <= current_close_1h <= LONG_PULLBACK_ENTRY_HIGH
            long_pullback_ready = pullback_trigger_condition and in_long_pullback_zone and volume_confirmed_long_pullback and not chop_filter_active and not long_pullback_whipsaw and breakout_priority and not long_pullback_state.get("triggered", False) and not long_pullback_state.get("stopped_out", False)
            
            logger.info("🔍 LONG - Halfback Pullback Strategy Analysis:")
            logger.info(f"   • Reclaim 50% intraday (${LONG_PULLBACK_TRIGGER:,.0f}): {'✅' if pullback_trigger_condition else '❌'}")
            logger.info(f"   • Price in entry zone (${LONG_PULLBACK_ENTRY_LOW:,.0f}-${LONG_PULLBACK_ENTRY_HIGH:,.0f}): {'✅' if in_long_pullback_zone else '❌'}")
            logger.info(f"   • Volume confirmed (controlled RVOL: {last_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x): {'✅' if volume_confirmed_long_pullback else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if long_pullback_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakout_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if long_pullback_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if long_pullback_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Halfback Pullback Ready: {'🎯 YES' if long_pullback_ready else '⏳ NO'}")
            
            if long_pullback_ready:
                logger.info("")
                logger.info("🎯 LONG - Pullback Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Halfback Pullback trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD Halfback Pullback Long",
                    entry_price=current_close_1h,
                    stop_loss=LONG_PULLBACK_STOP_LOSS,
                    take_profit=LONG_PULLBACK_TP1,
                    side="BUY",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed_long_pullback
                )
                
                if trade_success:
                    logger.info("🎉 Halfback Pullback Long trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_PULLBACK_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_PULLBACK_TP1:,.2f}")
                    logger.info("Strategy: Mean-reversion to VWAP/value near 50% of today's range")
                    
                    # Save trigger state
                    long_pullback_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(long_pullback_state, LONG_PULLBACK_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Pullback trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy (updated for new plan)
        if not trade_executed and short_strategies_enabled:
            breakdown_trigger_condition = last_close_1h < SHORT_BREAKDOWN_TRIGGER
            in_short_breakdown_zone = SHORT_BREAKDOWN_ENTRY_LOW <= current_close_1h <= SHORT_BREAKDOWN_ENTRY_HIGH
            short_breakdown_ready = breakdown_trigger_condition and in_short_breakdown_zone and volume_confirmed_short_breakdown and not chop_filter_active and not short_breakdown_whipsaw and breakdown_priority and not short_breakdown_state.get("triggered", False) and not short_breakdown_state.get("stopped_out", False)
            
            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • 1h close < ${SHORT_BREAKDOWN_TRIGGER:,.0f}: {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_1h:,.0f})")
            logger.info(f"   • Price in entry zone (${SHORT_BREAKDOWN_ENTRY_LOW:,.0f}-${SHORT_BREAKDOWN_ENTRY_HIGH:,.0f}): {'✅' if in_short_breakdown_zone else '❌'}")
            logger.info(f"   • Volume confirmed (1H RVOL: {last_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x): {'✅' if volume_confirmed_short_breakdown else '❌'}")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if short_breakdown_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if short_breakdown_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if short_breakdown_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • Breakdown Short Ready: {'🎯 YES' if short_breakdown_ready else '⏳ NO'}")
            
            if short_breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute Breakdown Short trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD Breakdown Short",
                    entry_price=current_close_1h,
                    stop_loss=SHORT_BREAKDOWN_STOP_LOSS,
                    take_profit=SHORT_BREAKDOWN_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed_short_breakdown
                )
                
                if trade_success:
                    logger.info("🎉 Breakdown Short trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_BREAKDOWN_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_BREAKDOWN_TP1:,.2f}")
                    logger.info("Strategy: Continuation after range loss")
                    
                    # Save trigger state
                    short_breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Breakdown trade failed: {trade_result}")
        
        # 4. SHORT - .618 Fade Strategy (updated for new plan)
        if not trade_executed and short_strategies_enabled:
            fade_trigger_condition = current_high_1h >= SHORT_FADE_TRIGGER
            in_short_fade_zone = SHORT_FADE_ENTRY_LOW <= current_close_1h <= SHORT_FADE_ENTRY_HIGH
            rsi_condition = current_rsi is not None and current_rsi < 50
            short_fade_ready = fade_trigger_condition and in_short_fade_zone and volume_confirmed_short_fade and rsi_condition and not chop_filter_active and not short_fade_whipsaw and breakdown_priority and not short_fade_state.get("triggered", False) and not short_fade_state.get("stopped_out", False)
            
            logger.info("🔍 SHORT - .618 Fade Strategy Analysis:")
            logger.info(f"   • Rejection at .618 level (${SHORT_FADE_TRIGGER:,.0f}): {'✅' if fade_trigger_condition else '❌'} (current high: {current_high_1h:,.0f})")
            logger.info(f"   • Price in entry zone (${SHORT_FADE_ENTRY_LOW:,.0f}-${SHORT_FADE_ENTRY_HIGH:,.0f}): {'✅' if in_short_fade_zone else '❌'}")
            logger.info(f"   • Volume confirmed (fading RVOL: {last_volume_1h/avg_volume_1h if avg_volume_1h > 0 else 0:.2f}x): {'✅' if volume_confirmed_short_fade else '❌'}")
            rsi_display = f"{current_rsi:.1f}" if current_rsi is not None else "N/A"
            logger.info(f"   • RSI fails to hold 50: {'✅' if rsi_condition else '❌'} (current: {rsi_display})")
            logger.info(f"   • Chop filter: {'❌ SKIP' if chop_filter_active else '✅ CONTINUE'}")
            logger.info(f"   • Whipsaw check: {'❌ WHIPSAW' if short_fade_whipsaw else '✅ NO WHIPSAW'}")
            logger.info(f"   • Strategy priority: {'✅' if breakdown_priority else '❌'}")
            logger.info(f"   • Already triggered: {'✅' if short_fade_state.get('triggered', False) else '❌'}")
            logger.info(f"   • Stopped out: {'✅' if short_fade_state.get('stopped_out', False) else '❌'}")
            logger.info(f"   • .618 Fade Ready: {'🎯 YES' if short_fade_ready else '⏳ NO'}")
            
            if short_fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fade into resistance Strategy conditions met - executing trade...")
                
                # Play alert sound
                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")
                
                # Execute .618 Fade Short trade
                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH-USD .618 Fade Short",
                    entry_price=current_close_1h,
                    stop_loss=SHORT_FADE_STOP_LOSS,
                    take_profit=SHORT_FADE_TP1,
                    side="SELL",
                    product=PRODUCT_ID,
                    volume_confirmed=volume_confirmed_short_fade
                )
                
                if trade_success:
                    logger.info("🎉 .618 Fade Short trade executed successfully!")
                    logger.info(f"Entry: ${current_close_1h:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_FADE_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_FADE_TP1:,.2f}")
                    rsi_value = f"{current_rsi:.1f}" if current_rsi is not None else "N/A"
                    logger.info(f"RSI: {rsi_value}")
                    logger.info("Strategy: Counter-trend fade into fib supply")
                    
                    # Save trigger state
                    short_fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(current_candle_1h['start']),
                        "entry_price": current_close_1h
                    }
                    save_trigger_state(short_fade_state, SHORT_FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Fade trade failed: {trade_result}")
        

        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions…")
        
        # Reset triggers if price moves significantly away from entry zones
        # Execution guardrails: If first entry stops, stand down until new 24h structure forms
        if long_breakout_state.get("triggered", False):
            if current_close_1h < LONG_BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Long Breakout trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                long_breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                logger.info("Long Breakout trigger state reset - standing down")
        

        
        if long_pullback_state.get("triggered", False):
            if current_close_1h < LONG_PULLBACK_STOP_LOSS:
                logger.info("🔄 Resetting Long Pullback trigger state - price fell below stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                long_pullback_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(long_pullback_state, LONG_PULLBACK_TRIGGER_FILE)
                logger.info("Long Pullback trigger state reset - standing down")
        
        if short_breakdown_state.get("triggered", False):
            if current_close_1h > SHORT_BREAKDOWN_STOP_LOSS:
                logger.info("🔄 Resetting Short Breakdown trigger state - price rose above stop loss")
                logger.warning("⚠️ Execution guardrail: Standing down until new 24h structure forms")
                short_breakdown_state = {"triggered": False, "trigger_ts": None, "entry_price": None, "stopped_out": True}
                save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                logger.info("Short Breakdown trigger state reset - standing down")
        
        if short_fade_state.get("triggered", False):
            if current_close_1h > SHORT_FADE_STOP_LOSS:
                logger.info("🔄 Resetting Short Fade state - SL hit")
                short_fade_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
                save_trigger_state(short_fade_state, SHORT_FADE_TRIGGER_FILE)
        
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
        logger.info("Strategy: ETH Intraday Trading Strategy - LONG & SHORT with Execution Guardrails")
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