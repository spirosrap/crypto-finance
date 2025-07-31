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

# ETH Trading Strategy Parameters (based on image)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_1H = "ONE_HOUR"  # 1-hour chart for trigger
GRANULARITY_5M = "FIVE_MINUTE"  # 5-minute chart for execution
VOLUME_PERIOD = 20  # For volume confirmation

# LONG (breakout) Strategy Parameters
BREAKOUT_ENTRY_LOW = 3882  # Buy-stop order range
BREAKOUT_ENTRY_HIGH = 3892
BREAKOUT_STOP_LOSS = 3838  # Back inside prior range
BREAKOUT_TP1 = 3945  # First take profit
BREAKOUT_TP2_LOW = 4010  # Second take profit range
BREAKOUT_TP2_HIGH = 4030

# LONG (retest) Strategy Parameters
RETEST_ENTRY_LOW = 3765  # Entry range after sweep
RETEST_ENTRY_HIGH = 3795
RETEST_SWEEP_LOW = 3750  # Sweep zone
RETEST_SWEEP_HIGH = 3770
RETEST_STOP_LOSS = 3718  # Below LOD structure
RETEST_TP1 = 3850  # First take profit
RETEST_TP2_LOW = 3920  # Second take profit range
RETEST_TP2_HIGH = 3940

# SHORT (breakdown) Strategy Parameters
BREAKDOWN_ENTRY_LOW = 3678  # Sell-stop order range
BREAKDOWN_ENTRY_HIGH = 3685
BREAKDOWN_STOP_LOSS = 3720  # Above LOD
BREAKDOWN_TP1 = 3620  # First take profit
BREAKDOWN_TP2_LOW = 3550  # Second take profit range
BREAKDOWN_TP2_HIGH = 3580

# SHORT (fade into resistance) Strategy Parameters
FADE_ENTRY_LOW = 3925  # Entry range on spike + rejection
FADE_ENTRY_HIGH = 3950
FADE_STOP_LOSS = 3985  # Above resistance
FADE_TP1 = 3860  # First take profit
FADE_TP2_LOW = 3800  # Second take profit
FADE_TP2_HIGH = 3800

# Volume confirmation requirements
VOLUME_SURGE_FACTOR_1H = 1.25  # ≥1.25x 20-period volume on 1h chart
VOLUME_SURGE_FACTOR_5M = 2.0   # ≥2x 20-period SMA volume on 5m chart

# Risk management
RISK_PERCENTAGE = 0.8  # 0.8-1.2% of price for 1R
PARTIAL_PROFIT_RANGE_LOW = 1.0  # Partial profit at +1.0R
PARTIAL_PROFIT_RANGE_HIGH = 1.5  # Partial profit at +1.5R

# Trade parameters - Position size: margin x leverage = 20 x 250 = 5000 USD
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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = PRODUCT_ID):
    def _execute_trade():
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        position_size_usd = POSITION_SIZE_USD  # Use the fixed position size from parameters
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

def check_sweep_and_reclaim(cb_service, current_price, current_ts):
    """Check if there was a sweep of 3750-3770 and reclaim on 5-15m"""
    try:
        # Get recent 5-minute candles to check for sweep
        end = current_ts
        start = end - timedelta(hours=2)  # Check last 2 hours
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        
        if not candles_5m or len(candles_5m) < 3:
            return False
            
        # Sort by timestamp ascending
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        
        # Check if there was a sweep of the zone
        sweep_occurred = False
        for candle in candles_5m[:-1]:  # Exclude current candle
            low = float(candle['low'])
            if low <= RETEST_SWEEP_HIGH and low >= RETEST_SWEEP_LOW:
                sweep_occurred = True
                logger.info(f"Sweep detected at ${low:,.2f} in 5m candle")
                break
        
        if not sweep_occurred:
            logger.info("No sweep of 3750-3770 zone detected")
            return False
        
        # Check if price has reclaimed above entry zone
        reclaim_confirmed = current_price >= RETEST_ENTRY_LOW
        
        logger.info(f"Reclaim check: current price ${current_price:,.2f} vs entry zone ${RETEST_ENTRY_LOW}-${RETEST_ENTRY_HIGH} -> {'✅' if reclaim_confirmed else '❌'}")
        
        return reclaim_confirmed
        
    except Exception as e:
        logger.error(f"Error checking sweep and reclaim: {e}")
        return False

def check_spike_and_rejection(cb_service, current_price, current_ts):
    """Check for spike + rejection pattern on 5-15 minute timeframes"""
    try:
        # Get recent 5-minute candles to check for spike and rejection
        end = current_ts
        start = end - timedelta(hours=1)  # Check last hour
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        
        if not candles_5m or len(candles_5m) < 3:
            return False
            
        # Sort by timestamp ascending
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        
        # Check for spike (high wick) and rejection pattern
        for candle in candles_5m[-3:]:  # Check last 3 candles
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            open_price = float(candle['open'])
            
            # Calculate wick size (upper wick)
            body_size = abs(close - open_price)
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            
            # Check for upper wick rejection (spike + rejection)
            if (upper_wick > body_size * 0.5 and  # Upper wick is significant
                upper_wick > lower_wick * 2 and    # Upper wick is much larger than lower wick
                high >= FADE_ENTRY_LOW and         # Spike reached entry zone
                close < high * 0.95):              # Price rejected from high
                
                logger.info(f"Spike and rejection detected: high=${high:,.2f}, close=${close:,.2f}, upper_wick=${upper_wick:,.2f}")
                return True
        
        logger.info("No spike and rejection pattern detected")
        return False
        
    except Exception as e:
        logger.error(f"Error checking spike and rejection: {e}")
        return False

# --- ETH Trading Strategy Alert Logic ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None):
    logger.info("=== Starting ETH-USD Trading Strategy Alert ===")
    logger.info("🎯 Monitoring for all four strategies:")
    logger.info("📊 LONG (breakout): Buy-stop $3,882-$3,892 (above HOD + buffer)")
    logger.info("📊 LONG (retest): Entry $3,765-$3,795 (after sweep of $3,750-$3,770 and reclaim)")
    logger.info("📊 SHORT (breakdown): Sell-stop $3,678-$3,685 (through LOD)")
    logger.info("📊 SHORT (fade): Entry $3,925-$3,950 (spike + rejection at resistance)")
    
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
        
        logger.info(f"Current ETH price (1H): ${current_close_1h:,.2f}")
        logger.info(f"Current ETH price (5M): ${float(current_candle_5m['close']):,.2f}" if candles_5m else "No 5M data")
        
        # LONG (breakout) Strategy Conditions
        breakout_condition = (
            current_close_1h >= BREAKOUT_ENTRY_LOW and 
            current_close_1h <= BREAKOUT_ENTRY_HIGH and 
            volume_confirmed and 
            not breakout_state.get("triggered", False)
        )
        
        # LONG (retest) Strategy Conditions
        retest_condition = (
            current_close_1h >= RETEST_ENTRY_LOW and 
            current_close_1h <= RETEST_ENTRY_HIGH and 
            volume_confirmed and 
            not retest_state.get("triggered", False) and
            check_sweep_and_reclaim(cb_service, current_close_1h, current_ts_1h)
        )
        
        # SHORT (breakdown) Strategy Conditions
        breakdown_condition = (
            current_close_1h >= BREAKDOWN_ENTRY_LOW and 
            current_close_1h <= BREAKDOWN_ENTRY_HIGH and 
            volume_confirmed and 
            not breakdown_state.get("triggered", False)
        )
        
        # SHORT (fade into resistance) Strategy Conditions
        fade_condition = (
            current_close_1h >= FADE_ENTRY_LOW and 
            current_close_1h <= FADE_ENTRY_HIGH and 
            volume_confirmed and 
            not fade_state.get("triggered", False) and
            check_spike_and_rejection(cb_service, current_close_1h, current_ts_1h)
        )
        
        logger.info(f"Breakout condition: ${BREAKOUT_ENTRY_LOW}-${BREAKOUT_ENTRY_HIGH} -> {'✅' if breakout_condition else '❌'}")
        logger.info(f"Retest condition: ${RETEST_ENTRY_LOW}-${RETEST_ENTRY_HIGH} -> {'✅' if retest_condition else '❌'}")
        logger.info(f"Breakdown condition: ${BREAKDOWN_ENTRY_LOW}-${BREAKDOWN_ENTRY_HIGH} -> {'✅' if breakdown_condition else '❌'}")
        logger.info(f"Fade condition: ${FADE_ENTRY_LOW}-${FADE_ENTRY_HIGH} -> {'✅' if fade_condition else '❌'}")
        
        # Execute LONG (breakout) Strategy
        if breakout_condition:
            logger.info("🎯 BREAKOUT CONDITION MET - EXECUTING LONG BREAKOUT TRADE!")
            play_alert_sound()
            
            # Execute the trade with first target
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD LONG Breakout",
                entry_price=current_close_1h,
                stop_loss=BREAKOUT_STOP_LOSS,
                take_profit=BREAKOUT_TP1,  # First target
                side="BUY",
                product=PRODUCT_ID
            )
            
            logger.info(f"Breakout trade execution completed: success={trade_success}")
            
            if trade_success:
                logger.info(f"🎉 LONG Breakout trade executed successfully!")
                logger.info(f"Entry: ${current_close_1h:,.2f}")
                logger.info(f"Stop-loss: ${BREAKOUT_STOP_LOSS:,.2f}")
                logger.info(f"First profit target: ${BREAKOUT_TP1:,.2f}")
                logger.info(f"Second profit target: ${BREAKOUT_TP2_LOW}-${BREAKOUT_TP2_HIGH:,.2f}")
                logger.info(f"Risk: {RISK_PERCENTAGE}% of price for 1R")
                logger.info(f"Partial profit: +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R")
                logger.info(f"Trade output: {trade_result}")
                logger.info("📊 Strategy: Expansion above today's range high; momentum continuation if volume confirms")
            else:
                logger.error(f"❌ Breakout trade failed: {trade_result}")
            
            # Save trigger state to prevent duplicate trades
            breakout_state = {
                "triggered": True, 
                "trigger_ts": int(current_candle_1h['start']),
                "entry_price": current_close_1h
            }
            save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
            logger.info("Breakout trigger state saved")
        
        # Execute LONG (retest) Strategy
        elif retest_condition:
            logger.info("🎯 RETEST CONDITION MET - EXECUTING LONG RETEST TRADE!")
            play_alert_sound()
            
            # Execute the trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD LONG Retest",
                entry_price=current_close_1h,
                stop_loss=RETEST_STOP_LOSS,
                take_profit=RETEST_TP1,  # First target
                side="BUY",
                product=PRODUCT_ID
            )
            
            logger.info(f"Retest trade execution completed: success={trade_success}")
            
            if trade_success:
                logger.info(f"🎉 LONG Retest trade executed successfully!")
                logger.info(f"Entry: ${current_close_1h:,.2f}")
                logger.info(f"Stop-loss: ${RETEST_STOP_LOSS:,.2f}")
                logger.info(f"First profit target: ${RETEST_TP1:,.2f}")
                logger.info(f"Second profit target: ${RETEST_TP2_LOW}-${RETEST_TP2_HIGH:,.2f}")
                logger.info(f"Risk: {RISK_PERCENTAGE}% of price for 1R")
                logger.info(f"Partial profit: +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R")
                logger.info(f"Trade output: {trade_result}")
                logger.info("📊 Strategy: Higher low at mid-range; catch bid without chasing")
            else:
                logger.error(f"❌ Retest trade failed: {trade_result}")
            
            # Save trigger state to prevent duplicate trades
            retest_state = {
                "triggered": True, 
                "trigger_ts": int(current_candle_1h['start']),
                "entry_price": current_close_1h
            }
            save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
            logger.info("Retest trigger state saved")
        
        # Execute SHORT (breakdown) Strategy
        elif breakdown_condition:
            logger.info("🎯 BREAKDOWN CONDITION MET - EXECUTING SHORT BREAKDOWN TRADE!")
            play_alert_sound()
            
            # Execute the trade with first target
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD SHORT Breakdown",
                entry_price=current_close_1h,
                stop_loss=BREAKDOWN_STOP_LOSS,
                take_profit=BREAKDOWN_TP1,  # First target
                side="SELL",
                product=PRODUCT_ID
            )
            
            logger.info(f"Breakdown trade execution completed: success={trade_success}")
            
            if trade_success:
                logger.info(f"🎉 SHORT Breakdown trade executed successfully!")
                logger.info(f"Entry: ${current_close_1h:,.2f}")
                logger.info(f"Stop-loss: ${BREAKDOWN_STOP_LOSS:,.2f}")
                logger.info(f"First profit target: ${BREAKDOWN_TP1:,.2f}")
                logger.info(f"Second profit target: ${BREAKDOWN_TP2_LOW}-${BREAKDOWN_TP2_HIGH:,.2f}")
                logger.info(f"Risk: {RISK_PERCENTAGE}% of price for 1R")
                logger.info(f"Partial profit: +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R")
                logger.info(f"Trade output: {trade_result}")
                logger.info("📊 Strategy: Range failure + continuation if 1h candle closes below LOD on volume")
            else:
                logger.error(f"❌ Breakdown trade failed: {trade_result}")
            
            # Save trigger state to prevent duplicate trades
            breakdown_state = {
                "triggered": True, 
                "trigger_ts": int(current_candle_1h['start']),
                "entry_price": current_close_1h
            }
            save_trigger_state(breakdown_state, BREAKDOWN_TRIGGER_FILE)
            logger.info("Breakdown trigger state saved")
        
        # Execute SHORT (fade into resistance) Strategy
        elif fade_condition:
            logger.info("🎯 FADE CONDITION MET - EXECUTING SHORT FADE TRADE!")
            play_alert_sound()
            
            # Execute the trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="ETH-USD SHORT Fade",
                entry_price=current_close_1h,
                stop_loss=FADE_STOP_LOSS,
                take_profit=FADE_TP1,  # First target
                side="SELL",
                product=PRODUCT_ID
            )
            
            logger.info(f"Fade trade execution completed: success={trade_success}")
            
            if trade_success:
                logger.info(f"🎉 SHORT Fade trade executed successfully!")
                logger.info(f"Entry: ${current_close_1h:,.2f}")
                logger.info(f"Stop-loss: ${FADE_STOP_LOSS:,.2f}")
                logger.info(f"First profit target: ${FADE_TP1:,.2f}")
                logger.info(f"Second profit target: ${FADE_TP2_LOW}-${FADE_TP2_HIGH:,.2f}")
                logger.info(f"Risk: {RISK_PERCENTAGE}% of price for 1R")
                logger.info(f"Partial profit: +{PARTIAL_PROFIT_RANGE_LOW}-{PARTIAL_PROFIT_RANGE_HIGH}R")
                logger.info(f"Trade output: {trade_result}")
                logger.info("📊 Strategy: First test of overhead supply/round number often mean-reverts intraday")
            else:
                logger.error(f"❌ Fade trade failed: {trade_result}")
            
            # Save trigger state to prevent duplicate trades
            fade_state = {
                "triggered": True, 
                "trigger_ts": int(current_candle_1h['start']),
                "entry_price": current_close_1h
            }
            save_trigger_state(fade_state, FADE_TRIGGER_FILE)
            logger.info("Fade trigger state saved")
        
        else:
            logger.info("⏳ Waiting for any strategy conditions...")
            if not volume_confirmed:
                logger.info(f"   Volume confirmation not met")
            if breakout_state.get("triggered", False):
                logger.info("   Breakout strategy already triggered")
            if retest_state.get("triggered", False):
                logger.info("   Retest strategy already triggered")
            if breakdown_state.get("triggered", False):
                logger.info("   Breakdown strategy already triggered")
            if fade_state.get("triggered", False):
                logger.info("   Fade strategy already triggered")
        
        # Reset triggers if price moves significantly away from entry zones
        if breakout_state.get("triggered", False):
            if current_close_1h < BREAKOUT_STOP_LOSS:
                logger.info("🔄 Resetting Breakout trigger state - price fell below stop loss")
                breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
                save_trigger_state(breakout_state, BREAKOUT_TRIGGER_FILE)
                logger.info("Breakout trigger state reset")
        
        if retest_state.get("triggered", False):
            if current_close_1h < RETEST_STOP_LOSS:
                logger.info("🔄 Resetting Retest trigger state - price fell below stop loss")
                retest_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
                save_trigger_state(retest_state, RETEST_TRIGGER_FILE)
                logger.info("Retest trigger state reset")
        
        if breakdown_state.get("triggered", False):
            if current_close_1h > BREAKDOWN_STOP_LOSS:
                logger.info("🔄 Resetting Breakdown trigger state - price rose above stop loss")
                breakdown_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
                save_trigger_state(breakdown_state, BREAKDOWN_TRIGGER_FILE)
                logger.info("Breakdown trigger state reset")
        
        if fade_state.get("triggered", False):
            if current_close_1h > FADE_STOP_LOSS:
                logger.info("🔄 Resetting Fade trigger state - price rose above stop loss")
                fade_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
                save_trigger_state(fade_state, FADE_TRIGGER_FILE)
                logger.info("Fade trigger state reset")
        
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
    logger.info("Starting ETH-USD Trading Strategy Monitor")
    logger.info("🎯 Monitoring for all four strategies:")
    logger.info("📊 LONG (breakout): Buy-stop $3,882-$3,892 (above HOD + buffer)")
    logger.info("📊 LONG (retest): Entry $3,765-$3,795 (after sweep of $3,750-$3,770 and reclaim)")
    logger.info("📊 SHORT (breakdown): Sell-stop $3,678-$3,685 (through LOD)")
    logger.info("📊 SHORT (fade): Entry $3,925-$3,950 (spike + rejection at resistance)")
    logger.info("💡 Volume confirmation: ≥1.25x 20-period 1h volume OR ≥2x 20-period 5m volume")
    logger.info("🛑 LONG SL: Breakout $3,838, Retest $3,718 | SHORT SL: Breakdown $3,720, Fade $3,985")
    logger.info("🎯 LONG TP: Breakout $3,945 / $4,010-$4,030, Retest $3,850 / $3,920-$3,940")
    logger.info("🎯 SHORT TP: Breakdown $3,620 / $3,550-$3,580, Fade $3,860 / $3,800")
    logger.info("⏰ Timeframe: 1-hour trigger, 5-15 minute execution")
    logger.info("💰 Risk: 0.8-1.2% of price for 1R, Partial profit at +1.0-1.5R")
    logger.info(f"💰 Position size: ${MARGIN} x {LEVERAGE} = ${POSITION_SIZE_USD:,} USD")
    
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
        last_alert_ts = eth_trading_strategy_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        logger.info(f"✅ Alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)
                    wait_seconds = 300
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