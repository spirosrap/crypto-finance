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

# ETH Trading Strategy Parameters (New Setup)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_5M = "FIVE_MINUTE"  # Primary timeframe for triggers
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context
CURRENT_ETH_PRICE = 4739.00  # ETH Spot ~$4,739
TODAY_HIGH = 4879.00  # Today's high: $4,879
TODAY_LOW = 4211.00   # Today's low: $4,211

# LONG SETUPS
# Long - Breakout
LONG_BREAKOUT_TRIGGER_LOW = 4890    # Trigger: 5m close above 4,890-4,900
LONG_BREAKOUT_TRIGGER_HIGH = 4900
LONG_BREAKOUT_ENTRY = 4895          # Entry: 4,895
LONG_BREAKOUT_STOP_LOSS = 4835      # Stop: 4,835
LONG_BREAKOUT_TP1 = 5020            # TP1: 5,020
LONG_BREAKOUT_TP2 = 5120            # TP2: 5,120

# Long - Pullback
LONG_PULLBACK_TRIGGER_LOW = 4620    # Trigger: Hold 4,620-4,650 after dip
LONG_PULLBACK_TRIGGER_HIGH = 4650
LONG_PULLBACK_ENTRY = 4635          # Entry: 4,635
LONG_PULLBACK_STOP_LOSS = 4570      # Stop: 4,570
LONG_PULLBACK_TP1 = 4780            # TP1: 4,780
LONG_PULLBACK_TP2 = 4860            # TP2: 4,860

# SHORT SETUPS
# Short - Fail @ highs
SHORT_FAIL_TRIGGER_LOW = 4870       # Trigger: Wick into 4,870-4,890 then close <4,860
SHORT_FAIL_TRIGGER_HIGH = 4890
SHORT_FAIL_CLOSE_BELOW = 4860       # Close below 4,860
SHORT_FAIL_ENTRY = 4860             # Entry: 4,860
SHORT_FAIL_STOP_LOSS = 4905         # Stop: 4,905
SHORT_FAIL_TP1 = 4750               # TP1: 4,750
SHORT_FAIL_TP2 = 4700               # TP2: 4,700

# Short - Breakdown
SHORT_BREAKDOWN_TRIGGER = 4580      # Trigger: Acceptance <4,580
SHORT_BREAKDOWN_ENTRY = 4580        # Entry: 4,580
SHORT_BREAKDOWN_STOP_LOSS = 4630    # Stop: 4,630
SHORT_BREAKDOWN_TP1 = 4460          # TP1: 4,460
SHORT_BREAKDOWN_TP2 = 4380          # TP2: 4,380

# Volume confirmation requirement
VOLUME_CONFIRMATION_FACTOR = 1.5  # 5m RVOL ≥ 1.5 vs 20-bar average

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
LONG_PULLBACK_TRIGGER_FILE = "eth_pullback_trigger_state.json"
SHORT_FAIL_TRIGGER_FILE = "eth_fail_trigger_state.json"
SHORT_BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"

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
                start=int((datetime.now(UTC) - timedelta(hours=6)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity=GRANULARITY_5M
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

# --- ETH Trading Strategy Alert Logic (New Setup) ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH Trading Strategy Alert - New Setup
    
    Spiros, here are clean ETH setups for today. Spot ref now ~$4,739, intraday H/L $4,879 / $4,211. 
    OI up ~+15% 24h and funding mildly positive (~0.015%/8h) → longs somewhat crowded; respect failed-breakout shorts.

    Side	Trigger (5-min close + ≥1.5× 20-bar vol)	Entry zone	Invalidation (SL)	TP1	TP2	Est. R:R
    Long – Breakout	Above 4,890–4,900	4,895	4,835	5,020	5,120	≈2.1
    Long – Pullback	Hold 4,620–4,650 after dip	4,635	4,570	4,780	4,860	≈2.2
    Short – Fail @ highs	Wick into 4,870–4,890 then close <4,860	4,860	4,905	4,750	4,700	≈2.4
    Short – Breakdown	Acceptance <4,580	4,580	4,630	4,460	4,380	≈2.4

    Use: USD spot levels; map to your perp. Skip signals during low-vol churn; require the volume confirm. 
    If funding flips negative and OI drops >5% intraday, de-prioritize longs.
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== ETH Trading Strategy Alert (New Setup - LONG & SHORT) ===")
    else:
        logger.info(f"=== ETH Trading Strategy Alert (New Setup - {direction} Strategy Only) ===")
    
    # Load trigger states
    long_breakout_state = load_trigger_state(LONG_BREAKOUT_TRIGGER_FILE)
    long_pullback_state = load_trigger_state(LONG_PULLBACK_TRIGGER_FILE)
    short_fail_state = load_trigger_state(SHORT_FAIL_TRIGGER_FILE)
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 5-minute candles for analysis (primary timeframe)
        end = now
        start = now - timedelta(hours=2)  # Get enough data for 20-period volume average (max ~24 candles)
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 5-minute candles for {2} hours...")
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        
        if not candles_5m or len(candles_5m) < VOLUME_PERIOD + 3:
            logger.warning("Not enough 5-minute candle data for trading strategy alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending (oldest first)
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        
        # Get current and last closed 5-minute candles
        current_candle_5m = candles_5m[-1]  # Most recent (potentially incomplete)
        last_closed_5m = candles_5m[-2]     # Last completed candle
        prev_closed_5m = candles_5m[-3]     # Previous completed candle
        
        current_close_5m = float(current_candle_5m['close'])
        current_high_5m = float(current_candle_5m['high'])
        current_low_5m = float(current_candle_5m['low'])
        current_volume_5m = float(current_candle_5m['volume'])
        current_ts_5m = datetime.fromtimestamp(int(current_candle_5m['start']), UTC)
        
        last_close_5m = float(last_closed_5m['close'])
        last_high_5m = float(last_closed_5m['high'])
        last_low_5m = float(last_closed_5m['low'])
        last_volume_5m = float(last_closed_5m['volume'])
        last_ts_5m = datetime.fromtimestamp(int(last_closed_5m['start']), UTC)
        
        # Calculate 20-period average volume (excluding current incomplete candle)
        volume_candles_5m = candles_5m[-(VOLUME_PERIOD+1):-1]  # Last 20 completed candles
        avg_volume_5m = sum(float(c['volume']) for c in volume_candles_5m) / len(volume_candles_5m)
        relative_volume_5m = last_volume_5m / avg_volume_5m if avg_volume_5m > 0 else 0
        
        # Check volume confirmation
        volume_confirmed = relative_volume_5m >= VOLUME_CONFIRMATION_FACTOR
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, here are clean ETH setups for today.")
        logger.info(f"📊 Live Levels:")
        logger.info(f"   • ETH ≈ ${current_close_5m:,.2f}")
        logger.info(f"   • Today's H: ${TODAY_HIGH:.2f}")
        logger.info(f"   • Today's L: ${TODAY_LOW:.2f}")
        logger.info("")
        logger.info("📊 Strategy Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Primary timeframe: 5-minute candles")
        logger.info(f"   • Volume requirement: 5m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setups:")
            logger.info(f"   • Breakout: 5m close above ${LONG_BREAKOUT_TRIGGER_LOW:,}–${LONG_BREAKOUT_TRIGGER_HIGH:,} → Entry ${LONG_BREAKOUT_ENTRY:,}")
            logger.info(f"     Stop: ${LONG_BREAKOUT_STOP_LOSS:,}, TP1: ${LONG_BREAKOUT_TP1:,}, TP2: ${LONG_BREAKOUT_TP2:,} (R:R ≈2.1)")
            logger.info(f"   • Pullback: Hold ${LONG_PULLBACK_TRIGGER_LOW:,}–${LONG_PULLBACK_TRIGGER_HIGH:,} after dip → Entry ${LONG_PULLBACK_ENTRY:,}")
            logger.info(f"     Stop: ${LONG_PULLBACK_STOP_LOSS:,}, TP1: ${LONG_PULLBACK_TP1:,}, TP2: ${LONG_PULLBACK_TP2:,} (R:R ≈2.2)")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setups:")
            logger.info(f"   • Fail @ highs: Wick into ${SHORT_FAIL_TRIGGER_LOW:,}–${SHORT_FAIL_TRIGGER_HIGH:,} then close <${SHORT_FAIL_CLOSE_BELOW:,} → Entry ${SHORT_FAIL_ENTRY:,}")
            logger.info(f"     Stop: ${SHORT_FAIL_STOP_LOSS:,}, TP1: ${SHORT_FAIL_TP1:,}, TP2: ${SHORT_FAIL_TP2:,} (R:R ≈2.4)")
            logger.info(f"   • Breakdown: Acceptance <${SHORT_BREAKDOWN_TRIGGER:,} → Entry ${SHORT_BREAKDOWN_ENTRY:,}")
            logger.info(f"     Stop: ${SHORT_BREAKDOWN_STOP_LOSS:,}, TP1: ${SHORT_BREAKDOWN_TP1:,}, TP2: ${SHORT_BREAKDOWN_TP2:,} (R:R ≈2.4)")
            logger.info("")
        
        logger.info(f"Current Price: ${current_close_5m:,.2f}")
        logger.info(f"Last 5M Close: ${last_close_5m:,.2f}, High: ${last_high_5m:,.2f}, Low: ${last_low_5m:,.2f}")
        logger.info(f"5M Volume: {last_volume_5m:,.0f}, 5M SMA(20): {avg_volume_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}x")
        logger.info(f"Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'}")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Strategy
        if (long_strategies_enabled and 
            not long_breakout_state.get("triggered", False) and not trade_executed):
            
            # Check if 5m close above trigger level
            breakout_trigger_condition = (LONG_BREAKOUT_TRIGGER_LOW <= last_close_5m <= LONG_BREAKOUT_TRIGGER_HIGH)
            # Check if current price is at entry level
            breakout_entry_condition = abs(current_close_5m - LONG_BREAKOUT_ENTRY) <= 5  # Within $5 of entry
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and volume_confirmed

            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • 5m close in ${LONG_BREAKOUT_TRIGGER_LOW:,}–${LONG_BREAKOUT_TRIGGER_HIGH:,}: {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry at ${LONG_BREAKOUT_ENTRY:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Breakout Long Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakout Long (New Setup)",
                    entry_price=LONG_BREAKOUT_ENTRY,
                    stop_loss=LONG_BREAKOUT_STOP_LOSS,
                    take_profit=LONG_BREAKOUT_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakout Long trade executed successfully!")
                    logger.info(f"Entry: ${LONG_BREAKOUT_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_BREAKOUT_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_BREAKOUT_TP1:,.2f}, TP2: ${LONG_BREAKOUT_TP2:,.2f}")
                    logger.info("Strategy: Breakout above 4,890-4,900 with volume confirmation")
                    
                    # Save trigger state
                    long_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": LONG_BREAKOUT_ENTRY
                    }
                    save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Breakout trade failed: {trade_result}")
        
        # 2. LONG - Pullback Strategy
        if (long_strategies_enabled and 
            not long_pullback_state.get("triggered", False) and not trade_executed):
            
            # Check if price holds in pullback zone after a dip
            pullback_trigger_condition = (LONG_PULLBACK_TRIGGER_LOW <= current_close_5m <= LONG_PULLBACK_TRIGGER_HIGH)
            # Check if current price is at entry level
            pullback_entry_condition = abs(current_close_5m - LONG_PULLBACK_ENTRY) <= 5  # Within $5 of entry
            
            pullback_ready = pullback_trigger_condition and pullback_entry_condition and volume_confirmed

            logger.info("🔍 LONG - Pullback Strategy Analysis:")
            logger.info(f"   • Hold in ${LONG_PULLBACK_TRIGGER_LOW:,}–${LONG_PULLBACK_TRIGGER_HIGH:,}: {'✅' if pullback_trigger_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Entry at ${LONG_PULLBACK_ENTRY:,}: {'✅' if pullback_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Pullback Long Ready: {'🎯 YES' if pullback_ready else '⏳ NO'}")

            if pullback_ready:
                logger.info("")
                logger.info("🎯 LONG - Pullback Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Pullback Long (New Setup)",
                    entry_price=LONG_PULLBACK_ENTRY,
                    stop_loss=LONG_PULLBACK_STOP_LOSS,
                    take_profit=LONG_PULLBACK_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Pullback Long trade executed successfully!")
                    logger.info(f"Entry: ${LONG_PULLBACK_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_PULLBACK_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_PULLBACK_TP1:,.2f}, TP2: ${LONG_PULLBACK_TP2:,.2f}")
                    logger.info("Strategy: Hold 4,620-4,650 after dip with volume confirmation")
                    
                    # Save trigger state
                    long_pullback_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": LONG_PULLBACK_ENTRY
                    }
                    save_trigger_state(long_pullback_state, LONG_PULLBACK_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Pullback trade failed: {trade_result}")
        
        # 3. SHORT - Fail @ highs Strategy
        if (short_strategies_enabled and 
            not short_fail_state.get("triggered", False) and not trade_executed):
            
            # Check if there was a wick into the trigger zone
            wick_into_zone = (SHORT_FAIL_TRIGGER_LOW <= last_high_5m <= SHORT_FAIL_TRIGGER_HIGH)
            # Check if close is below the required level
            close_below_requirement = last_close_5m < SHORT_FAIL_CLOSE_BELOW
            # Check if current price is at entry level
            fail_entry_condition = abs(current_close_5m - SHORT_FAIL_ENTRY) <= 5  # Within $5 of entry
            
            fail_ready = wick_into_zone and close_below_requirement and fail_entry_condition and volume_confirmed

            logger.info("🔍 SHORT - Fail @ highs Strategy Analysis:")
            logger.info(f"   • Wick into ${SHORT_FAIL_TRIGGER_LOW:,}–${SHORT_FAIL_TRIGGER_HIGH:,}: {'✅' if wick_into_zone else '❌'} (last high: {last_high_5m:,.2f})")
            logger.info(f"   • Close <${SHORT_FAIL_CLOSE_BELOW:,}: {'✅' if close_below_requirement else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry at ${SHORT_FAIL_ENTRY:,}: {'✅' if fail_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Fail @ highs Short Ready: {'🎯 YES' if fail_ready else '⏳ NO'}")

            if fail_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fail @ highs Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Fail @ highs Short (New Setup)",
                    entry_price=SHORT_FAIL_ENTRY,
                    stop_loss=SHORT_FAIL_STOP_LOSS,
                    take_profit=SHORT_FAIL_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Fail @ highs Short trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_FAIL_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_FAIL_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_FAIL_TP1:,.2f}, TP2: ${SHORT_FAIL_TP2:,.2f}")
                    logger.info("Strategy: Wick into 4,870-4,890 then close <4,860 with volume confirmation")
                    
                    # Save trigger state
                    short_fail_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_FAIL_ENTRY
                    }
                    save_trigger_state(short_fail_state, SHORT_FAIL_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Fail @ highs trade failed: {trade_result}")
        
        # 4. SHORT - Breakdown Strategy
        if (short_strategies_enabled and 
            not short_breakdown_state.get("triggered", False) and not trade_executed):
            
            # Check if price accepts below breakdown level
            breakdown_trigger_condition = current_close_5m < SHORT_BREAKDOWN_TRIGGER
            # Check if current price is at entry level
            breakdown_entry_condition = abs(current_close_5m - SHORT_BREAKDOWN_ENTRY) <= 5  # Within $5 of entry
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and volume_confirmed

            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • Acceptance <${SHORT_BREAKDOWN_TRIGGER:,}: {'✅' if breakdown_trigger_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Entry at ${SHORT_BREAKDOWN_ENTRY:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Breakdown Short Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakdown Short (New Setup)",
                    entry_price=SHORT_BREAKDOWN_ENTRY,
                    stop_loss=SHORT_BREAKDOWN_STOP_LOSS,
                    take_profit=SHORT_BREAKDOWN_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown Short trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_BREAKDOWN_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_BREAKDOWN_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_BREAKDOWN_TP1:,.2f}, TP2: ${SHORT_BREAKDOWN_TP2:,.2f}")
                    logger.info("Strategy: Acceptance <4,580 with volume confirmation")
                    
                    # Save trigger state
                    short_breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_BREAKDOWN_ENTRY
                    }
                    save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Breakdown trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Long breakout triggered: {long_breakout_state.get('triggered', False)}")
            logger.info(f"Long pullback triggered: {long_pullback_state.get('triggered', False)}")
            logger.info(f"Short fail @ highs triggered: {short_fail_state.get('triggered', False)}")
            logger.info(f"Short breakdown triggered: {short_breakdown_state.get('triggered', False)}")
        
        logger.info("=== ETH Trading Strategy Alert completed ===")
        return current_ts_5m
        
    except Exception as e:
        logger.error(f"Error in ETH Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Trading Strategy Alert completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Trading Strategy Monitor (New Setup) with optional direction filter')
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
    
    logger.info("Starting ETH Trading Strategy Monitor (New Setup)")
    if direction == 'BOTH':
        logger.info("Strategy: ETH New Setup - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("LONG - Breakout: 5m close above 4,890-4,900 → Entry 4,895")
    logger.info("LONG - Pullback: Hold 4,620-4,650 after dip → Entry 4,635")
    logger.info("SHORT - Fail @ highs: Wick into 4,870-4,890 then close <4,860 → Entry 4,860")
    logger.info("SHORT - Breakdown: Acceptance <4,580 → Entry 4,580")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info(f"Volume: 5m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
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
