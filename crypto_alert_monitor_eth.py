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

# ETH Trading Strategy Parameters (Aug 22, 2025 Setup)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_15M = "FIFTEEN_MINUTE"  # Primary timeframe for triggers
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context (Aug 22, 2025)
CURRENT_ETH_PRICE = 4306.00  # ETH Spot ~$4,306
TODAY_HIGH = 4318.09  # Today's high: $4,318.09
TODAY_LOW = 4209.91   # Today's low: $4,209.91

# LONG SETUP - Breakout of today's high
LONG_BREAKOUT_TRIGGER = 4318.1    # Trigger: 15m close above 4,318.1
LONG_BREAKOUT_ENTRY_LOW = 4320     # Entry: 4,320–4,330
LONG_BREAKOUT_ENTRY_HIGH = 4330
LONG_BREAKOUT_STOP_LOSS_PERCENTAGE = 0.007  # 0.7% below entry
LONG_BREAKOUT_TP1_PERCENTAGE = 0.015  # +1.5% take profit

# SHORT SETUP - Breakdown of today's low  
SHORT_BREAKDOWN_TRIGGER = 4209.9   # Trigger: 15m close below 4,209.9
SHORT_BREAKDOWN_ENTRY_LOW = 4190    # Entry: 4,205–4,190
SHORT_BREAKDOWN_ENTRY_HIGH = 4205
SHORT_BREAKDOWN_STOP_LOSS_PERCENTAGE = 0.007  # 0.7% above entry
SHORT_BREAKDOWN_TP1_PERCENTAGE = 0.015  # -1.5% take profit

# Volume confirmation requirement
VOLUME_CONFIRMATION_FACTOR = 1.5  # 15m RVOL ≥ 1.5 vs 20-bar average

# Alert levels
LONG_ALERT_LEVEL = 4319.5   # Set alert at 4,319.5
SHORT_ALERT_LEVEL = 4209.5  # Set alert at 4,209.5

# Invalidation conditions
INVALIDATION_CANDLES = 2  # Two 15m closes back inside invalidate the setup

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
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
                start=int((datetime.now(UTC) - timedelta(days=2)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity=GRANULARITY_15M
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

# --- ETH Trading Strategy Alert Logic (Aug 22, 2025) ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH Trading Strategy Alert - August 22, 2025
    
    Spiros, ETH today 22 Aug 2025. Spot ~$4,306. H: 4,318.09. L: 4,209.91.

    Long (breakout of today's high)
        • Trigger: 15m close above 4,318.1, then enter 4,320–4,330.
        • Stop: 0.7% below entry (formula: SL = entry × 0.993).
        • TP1: +1.5% (TP1 = entry × 1.015). Optional trail after +1%.
        • Volume: 15m RVOL ≥ 1.5 vs 20-bar average.
        • Invalidation: two 15m closes back inside 4,318.

    Short (breakdown of today's low)
        • Trigger: 15m close below 4,209.9, then enter 4,205–4,190.
        • Stop: 0.7% above entry (SL = entry × 1.007).
        • TP1: −1.5% (TP1 = entry × 0.985). Optional trail after −1%.
        • Volume: 15m RVOL ≥ 1.5 vs 20-bar average.
        • Invalidation: two 15m closes back above 4,210.

    Set alerts: 4,319.5 and 4,209.5. Trade only on confirmed 15m closes.
    Position size: margin×leverage = 250×20 = $5,000 USD
    
    Args:
        cb_service: Coinbase service instance
        last_alert_ts: Last alert timestamp
        direction: Trading direction to monitor ('LONG', 'SHORT', or 'BOTH')
    """
    if direction == 'BOTH':
        logger.info("=== ETH Trading Strategy Alert (Aug 22, 2025 - LONG & SHORT) ===")
    else:
        logger.info(f"=== ETH Trading Strategy Alert (Aug 22, 2025 - {direction} Strategy Only) ===")
    
    # Load trigger states
    long_breakout_state = load_trigger_state(LONG_BREAKOUT_TRIGGER_FILE)
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 15-minute candles for analysis (primary timeframe)
        end = now
        start = now - timedelta(hours=8)  # Get enough data for 20-period volume average
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 15-minute candles for {8} hours...")
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_15M)
        
        if not candles_15m or len(candles_15m) < VOLUME_PERIOD + 3:
            logger.warning("Not enough 15-minute candle data for trading strategy alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending (oldest first)
        candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
        
        # Get current and last closed 15-minute candles
        current_candle_15m = candles_15m[-1]  # Most recent (potentially incomplete)
        last_closed_15m = candles_15m[-2]     # Last completed candle
        prev_closed_15m = candles_15m[-3]     # Previous completed candle
        
        current_close_15m = float(current_candle_15m['close'])
        current_high_15m = float(current_candle_15m['high'])
        current_low_15m = float(current_candle_15m['low'])
        current_volume_15m = float(current_candle_15m['volume'])
        current_ts_15m = datetime.fromtimestamp(int(current_candle_15m['start']), UTC)
        
        last_close_15m = float(last_closed_15m['close'])
        last_high_15m = float(last_closed_15m['high'])
        last_low_15m = float(last_closed_15m['low'])
        last_volume_15m = float(last_closed_15m['volume'])
        last_ts_15m = datetime.fromtimestamp(int(last_closed_15m['start']), UTC)
        
        # Calculate 20-period average volume (excluding current incomplete candle)
        volume_candles_15m = candles_15m[-(VOLUME_PERIOD+1):-1]  # Last 20 completed candles
        avg_volume_15m = sum(float(c['volume']) for c in volume_candles_15m) / len(volume_candles_15m)
        relative_volume_15m = last_volume_15m / avg_volume_15m if avg_volume_15m > 0 else 0
        
        # Check volume confirmation
        volume_confirmed = relative_volume_15m >= VOLUME_CONFIRMATION_FACTOR
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, ETH today 22 Aug 2025. Spot ~$4,306. H: 4,318.09. L: 4,209.91.")
        logger.info("")
        logger.info(f"📊 Live Levels:")
        logger.info(f"   • ETH ≈ ${current_close_15m:,.2f}")
        logger.info(f"   • Today's H: ${TODAY_HIGH:.2f}")
        logger.info(f"   • Today's L: ${TODAY_LOW:.2f}")
        logger.info("")
        logger.info("📊 Strategy Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Primary timeframe: 15-minute candles")
        logger.info(f"   • Volume requirement: 15m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
        logger.info(f"   • Alert levels: Long ${LONG_ALERT_LEVEL:.1f}, Short ${SHORT_ALERT_LEVEL:.1f}")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setup (breakout of today's high):")
            logger.info(f"   • Trigger: 15m close above ${LONG_BREAKOUT_TRIGGER:.1f}")
            logger.info(f"   • Entry: ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}")
            logger.info(f"   • Stop: 0.7% below entry (SL = entry × 0.993)")
            logger.info(f"   • TP1: +1.5% (TP1 = entry × 1.015)")
            logger.info(f"   • Volume: 15m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
            logger.info(f"   • Invalidation: two 15m closes back inside ${LONG_BREAKOUT_TRIGGER:.1f}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setup (breakdown of today's low):")
            logger.info(f"   • Trigger: 15m close below ${SHORT_BREAKDOWN_TRIGGER:.1f}")
            logger.info(f"   • Entry: ${SHORT_BREAKDOWN_ENTRY_HIGH:,}–${SHORT_BREAKDOWN_ENTRY_LOW:,}")
            logger.info(f"   • Stop: 0.7% above entry (SL = entry × 1.007)")
            logger.info(f"   • TP1: −1.5% (TP1 = entry × 0.985)")
            logger.info(f"   • Volume: 15m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
            logger.info(f"   • Invalidation: two 15m closes back above {SHORT_BREAKDOWN_TRIGGER + 0.1:.1f}")
            logger.info("")
        
        logger.info(f"Current Price: ${current_close_15m:,.2f}")
        logger.info(f"Last 15M Close: ${last_close_15m:,.2f}, High: ${last_high_15m:,.2f}, Low: ${last_low_15m:,.2f}")
        logger.info(f"15M Volume: {last_volume_15m:,.0f}, 15M SMA(20): {avg_volume_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}x")
        logger.info(f"Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'}")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Check for invalidation (two 15m closes back inside trigger levels)
        def check_invalidation(candles, trigger_level, is_long, state):
            """Check if invalidation condition is met"""
            if not state.get("triggered", False):
                return False
                
            # Check last 2 completed candles for closes back inside trigger level
            invalidation_count = 0
            for candle in candles[-3:-1]:  # Last 2 completed candles
                close = float(candle['close'])
                if is_long:
                    # For long, invalidation is closes back below trigger
                    if close <= trigger_level:
                        invalidation_count += 1
                else:
                    # For short, invalidation is closes back above trigger
                    if close >= trigger_level:
                        invalidation_count += 1
            
            return invalidation_count >= INVALIDATION_CANDLES
        
        # Check invalidations
        long_invalidated = check_invalidation(candles_15m, LONG_BREAKOUT_TRIGGER, True, long_breakout_state)
        short_invalidated = check_invalidation(candles_15m, SHORT_BREAKDOWN_TRIGGER, False, short_breakdown_state)
        
        if long_invalidated:
            logger.info("❌ LONG setup invalidated - two 15m closes back inside trigger level")
            long_breakout_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
            save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
            
        if short_invalidated:
            logger.info("❌ SHORT setup invalidated - two 15m closes back above trigger level")
            short_breakdown_state = {"triggered": False, "trigger_ts": None, "entry_price": None}
            save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
        
        # 1. LONG - Breakout Strategy
        if (long_strategies_enabled and 
            not long_breakout_state.get("triggered", False) and not long_invalidated):
            
            # Check if 15m close above trigger level
            breakout_trigger_condition = last_close_15m > LONG_BREAKOUT_TRIGGER
            # Check if current price is in entry zone
            breakout_entry_condition = (LONG_BREAKOUT_ENTRY_LOW <= current_close_15m <= LONG_BREAKOUT_ENTRY_HIGH)
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and volume_confirmed

            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • 15m close > ${LONG_BREAKOUT_TRIGGER:.1f}: {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry zone ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Breakout Long Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Strategy conditions met - executing trade...")

                # Calculate dynamic stop-loss and take-profit
                entry_price = current_close_15m
                stop_loss = entry_price * (1 - LONG_BREAKOUT_STOP_LOSS_PERCENTAGE)
                take_profit = entry_price * (1 + LONG_BREAKOUT_TP1_PERCENTAGE)

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakout Long (Aug 22, 2025)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakout Long trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f} (0.7% below entry)")
                    logger.info(f"TP1: ${take_profit:,.2f} (+1.5% above entry)")
                    logger.info("Strategy: Breakout of today's high with volume confirmation")
                    
                    # Save trigger state
                    long_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_15m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Breakout trade failed: {trade_result}")
        
        # 2. SHORT - Breakdown Strategy
        if (short_strategies_enabled and not trade_executed and
            not short_breakdown_state.get("triggered", False) and not short_invalidated):
            
            # Check if 15m close below trigger level
            breakdown_trigger_condition = last_close_15m < SHORT_BREAKDOWN_TRIGGER
            # Check if current price is in entry zone
            breakdown_entry_condition = (SHORT_BREAKDOWN_ENTRY_LOW <= current_close_15m <= SHORT_BREAKDOWN_ENTRY_HIGH)
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and volume_confirmed

            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • 15m close < ${SHORT_BREAKDOWN_TRIGGER:.1f}: {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry zone ${SHORT_BREAKDOWN_ENTRY_HIGH:,}–${SHORT_BREAKDOWN_ENTRY_LOW:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Breakdown Short Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Strategy conditions met - executing trade...")

                # Calculate dynamic stop-loss and take-profit
                entry_price = current_close_15m
                stop_loss = entry_price * (1 + SHORT_BREAKDOWN_STOP_LOSS_PERCENTAGE)
                take_profit = entry_price * (1 - SHORT_BREAKDOWN_TP1_PERCENTAGE)

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakdown Short (Aug 22, 2025)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown Short trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f} (0.7% above entry)")
                    logger.info(f"TP1: ${take_profit:,.2f} (-1.5% below entry)")
                    logger.info("Strategy: Breakdown of today's low with volume confirmation")
                    
                    # Save trigger state
                    short_breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_15m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Breakdown trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Long breakout triggered: {long_breakout_state.get('triggered', False)}")
            logger.info(f"Short breakdown triggered: {short_breakdown_state.get('triggered', False)}")
        
        logger.info("=== ETH Trading Strategy Alert completed ===")
        return current_ts_15m
        
    except Exception as e:
        logger.error(f"Error in ETH Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Trading Strategy Alert completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Trading Strategy Monitor (Aug 22, 2025) with optional direction filter')
    parser.add_argument('--direction', choices=['LONG', 'SHORT', 'BOTH'], default='BOTH',
                       help='Trading direction to monitor: LONG, SHORT, or BOTH (default: BOTH)')
    args = parser.parse_args()
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor_eth_new.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor_eth_new.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor_eth_new.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting ETH Trading Strategy Monitor (Aug 22, 2025)")
    if direction == 'BOTH':
        logger.info("Strategy: ETH Breakout/Breakdown - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info(f"LONG: 15m close > {LONG_BREAKOUT_TRIGGER:.1f} → Entry {LONG_BREAKOUT_ENTRY_LOW:,}–{LONG_BREAKOUT_ENTRY_HIGH:,}")
    logger.info(f"SHORT: 15m close < {SHORT_BREAKDOWN_TRIGGER:.1f} → Entry {SHORT_BREAKDOWN_ENTRY_HIGH:,}–{SHORT_BREAKDOWN_ENTRY_LOW:,}")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info(f"Stop: 0.7% from entry, TP: 1.5% from entry")
    logger.info(f"Volume: 15m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
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
