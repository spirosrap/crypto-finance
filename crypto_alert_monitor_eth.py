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

# ETH Trading Strategy Parameters (New Intraday Setup)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_5M = "FIVE_MINUTE"  # Primary timeframe for triggers
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context - Updated with new 24h levels
CURRENT_ETH_PRICE = 4739.00  # ETH Spot ~$4,739
TODAY_HIGH = 4887.59  # 24h high = 4,887.59
TODAY_LOW = 4659.70   # 24h low = 4,659.70

# LONG SETUPS
# Long - Breakout
LONG_BREAKOUT_TRIGGER = 4888    # Trigger: 5-min close above 4,888
LONG_BREAKOUT_ENTRY_LOW = 4880  # Entry zone: 4,880–4,900
LONG_BREAKOUT_ENTRY_HIGH = 4900
LONG_BREAKOUT_ENTRY = 4890      # Entry: 4,890 (mid of zone)
LONG_BREAKOUT_STOP_LOSS = 4853.38  # Stop: 4,853.38 (~0.7% below 4,888)
LONG_BREAKOUT_TP1 = 5001.54     # TP1: 5,001.54 (~½ 24h range)
LONG_BREAKOUT_TP2 = 5115.48     # TP2: 5,115.48 (~full 24h range)

# Long - Reclaim at range low
LONG_RECLAIM_TRIGGER = 4660     # Trigger: Flush < 4,660 then fast reclaim
LONG_RECLAIM_ENTRY_LOW = 4665   # Entry zone: 4,665–4,685
LONG_RECLAIM_ENTRY_HIGH = 4685
LONG_RECLAIM_ENTRY = 4675       # Entry: 4,675 (mid of zone)
LONG_RECLAIM_STOP_LOSS = 4636.40  # Stop: 4,636.40 (~0.5% below 4,660)
LONG_RECLAIM_TP1 = 4773.65      # TP1: 4,773.65 (mid-range)
LONG_RECLAIM_TP2 = 4887.59      # TP2: 4,887.59

# SHORT SETUPS
# Short - Breakdown
SHORT_BREAKDOWN_TRIGGER = 4660  # Trigger: 5-min close below 4,660
SHORT_BREAKDOWN_ENTRY_LOW = 4648  # Entry zone: 4,648–4,660
SHORT_BREAKDOWN_ENTRY_HIGH = 4660
SHORT_BREAKDOWN_ENTRY = 4654    # Entry: 4,654 (mid of zone)
SHORT_BREAKDOWN_STOP_LOSS = 4692.32  # Stop: 4,692.32 (~0.7% above 4,660)
SHORT_BREAKDOWN_TP1 = 4545.75   # TP1: 4,545.75
SHORT_BREAKDOWN_TP2 = 4431.81   # TP2: 4,431.81

# Short - Fade at range high
SHORT_FADE_TRIGGER_LOW = 4880   # Trigger: Spike to 4,880–4,900 with rejection
SHORT_FADE_TRIGGER_HIGH = 4900
SHORT_FADE_ENTRY = 4890         # Entry: 4,890 (mid of trigger zone)
SHORT_FADE_STOP_LOSS = 4912.03  # Stop: 4,912.03 (~0.5% above 4,888)
SHORT_FADE_TP1 = 4773.65        # TP1: 4,773.65 (mid-range)
SHORT_FADE_TP2 = 4659.70        # TP2: 4,659.70

# Volume confirmation requirement
VOLUME_CONFIRMATION_FACTOR = 2.0  # 5m RVOL ≥ 2× vs 20-bar average

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
LONG_RECLAIM_TRIGGER_FILE = "eth_reclaim_trigger_state.json"
SHORT_BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"
SHORT_FADE_TRIGGER_FILE = "eth_fade_trigger_state.json"

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
    ETH Intraday Trading Strategy Alert - New Setup
    
    Spiros, here are intraday ETH setups for today. Anchors: 24h high = 4,887.59, 24h low = 4,659.70 on Binance spot right now.

    Setup	Trigger & Entry	Invalidation (SL)	TP1	TP2	Confirmations	TF
    Long breakout	5-min close above 4,888, buy 4,880–4,900 on retest hold	4,853.38 (~0.7% below 4,888)	5,001.54 (~½ 24h range)	5,115.48 (~full 24h range)	5-min volume ≥2× 20-bar median; no bearish RSI/MACD divergence	5–15m
    Short breakdown	5-min close below 4,660, sell 4,648–4,660 on failed retest	4,692.32 (~0.7% above 4,660)	4,545.75	4,431.81	5-min volume ≥2× 20-bar median; weak bounce failing at 4,660	5–15m
    Short fade at range high	Spike to 4,880–4,900 with rejection (upper wick/divergence)	4,912.03 (~0.5% above 4,888)	4,773.65 (mid-range)	4,659.70	Rejection candle + declining buy volume	5–15m
    Long reclaim at range low	Flush < 4,660 then fast reclaim; buy 4,665–4,685	4,636.40 (~0.5% below 4,660)	4,773.65	4,887.59	Strong buyback; delta/volume spike on reclaim	5–15m

    Notes:
    •	Targets use today's 24h range height (≈227.89). Levels will shift if a new 24h high/low prints.
    •	Manage size so a full-stop loss ≤1R of your plan. Avoid chasing if confirmations are absent.
    
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
    long_reclaim_state = load_trigger_state(LONG_RECLAIM_TRIGGER_FILE)
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    short_fade_state = load_trigger_state(SHORT_FADE_TRIGGER_FILE)
    
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
        logger.info("🚀 Spiros, here are intraday ETH setups for today.")
        logger.info(f"📊 Live Levels:")
        logger.info(f"   • ETH ≈ ${current_close_5m:,.2f}")
        logger.info(f"   • 24h High: ${TODAY_HIGH:.2f}")
        logger.info(f"   • 24h Low: ${TODAY_LOW:.2f}")
        logger.info("")
        logger.info("📊 Strategy Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Primary timeframe: 5-minute candles")
        logger.info(f"   • Volume requirement: 5m RVOL ≥ {VOLUME_CONFIRMATION_FACTOR}x vs 20-bar average")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setups:")
            logger.info(f"   • Breakout: 5-min close above ${LONG_BREAKOUT_TRIGGER:,}, buy ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,} on retest hold")
            logger.info(f"     Stop: ${LONG_BREAKOUT_STOP_LOSS:,} (~0.7% below ${LONG_BREAKOUT_TRIGGER:,})")
            logger.info(f"     TP1: ${LONG_BREAKOUT_TP1:,} (~½ 24h range), TP2: ${LONG_BREAKOUT_TP2:,} (~full 24h range)")
            logger.info(f"   • Reclaim: Flush < ${LONG_RECLAIM_TRIGGER:,} then fast reclaim; buy ${LONG_RECLAIM_ENTRY_LOW:,}–${LONG_RECLAIM_ENTRY_HIGH:,}")
            logger.info(f"     Stop: ${LONG_RECLAIM_STOP_LOSS:,} (~0.5% below ${LONG_RECLAIM_TRIGGER:,})")
            logger.info(f"     TP1: ${LONG_RECLAIM_TP1:,} (mid-range), TP2: ${LONG_RECLAIM_TP2:,}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setups:")
            logger.info(f"   • Breakdown: 5-min close below ${SHORT_BREAKDOWN_TRIGGER:,}, sell ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,} on failed retest")
            logger.info(f"     Stop: ${SHORT_BREAKDOWN_STOP_LOSS:,} (~0.7% above ${SHORT_BREAKDOWN_TRIGGER:,})")
            logger.info(f"     TP1: ${SHORT_BREAKDOWN_TP1:,}, TP2: ${SHORT_BREAKDOWN_TP2:,}")
            logger.info(f"   • Fade: Spike to ${SHORT_FADE_TRIGGER_LOW:,}–${SHORT_FADE_TRIGGER_HIGH:,} with rejection (upper wick/divergence)")
            logger.info(f"     Stop: ${SHORT_FADE_STOP_LOSS:,} (~0.5% above ${LONG_BREAKOUT_TRIGGER:,})")
            logger.info(f"     TP1: ${SHORT_FADE_TP1:,} (mid-range), TP2: ${SHORT_FADE_TP2:,}")
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
            breakout_trigger_condition = last_close_5m > LONG_BREAKOUT_TRIGGER
            # Check if current price is in entry zone
            breakout_entry_condition = (LONG_BREAKOUT_ENTRY_LOW <= current_close_5m <= LONG_BREAKOUT_ENTRY_HIGH)
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and volume_confirmed

            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • 5m close above ${LONG_BREAKOUT_TRIGGER:,}: {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry in ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
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
        
        # 2. LONG - Reclaim Strategy
        if (long_strategies_enabled and 
            not long_reclaim_state.get("triggered", False) and not trade_executed):
            
            # Check if price flushes below reclaim trigger
            reclaim_trigger_condition = last_close_5m < LONG_RECLAIM_TRIGGER
            # Check if current price is in entry zone (reclaim after flush)
            reclaim_entry_condition = (LONG_RECLAIM_ENTRY_LOW <= current_close_5m <= LONG_RECLAIM_ENTRY_HIGH)
            
            reclaim_ready = reclaim_trigger_condition and reclaim_entry_condition and volume_confirmed

            logger.info("🔍 LONG - Reclaim Strategy Analysis:")
            logger.info(f"   • Flush < ${LONG_RECLAIM_TRIGGER:,}: {'✅' if reclaim_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry in ${LONG_RECLAIM_ENTRY_LOW:,}–${LONG_RECLAIM_ENTRY_HIGH:,}: {'✅' if reclaim_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Reclaim Long Ready: {'🎯 YES' if reclaim_ready else '⏳ NO'}")

            if reclaim_ready:
                logger.info("")
                logger.info("🎯 LONG - Reclaim Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Reclaim Long (New Setup)",
                    entry_price=LONG_RECLAIM_ENTRY,
                    stop_loss=LONG_RECLAIM_STOP_LOSS,
                    take_profit=LONG_RECLAIM_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Reclaim Long trade executed successfully!")
                    logger.info(f"Entry: ${LONG_RECLAIM_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_RECLAIM_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_RECLAIM_TP1:,.2f}, TP2: ${LONG_RECLAIM_TP2:,.2f}")
                    logger.info("Strategy: Flush < 4,660 then fast reclaim with volume confirmation")
                    
                    # Save trigger state
                    long_reclaim_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": LONG_RECLAIM_ENTRY
                    }
                    save_trigger_state(long_reclaim_state, LONG_RECLAIM_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Long Reclaim trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if (short_strategies_enabled and 
            not short_breakdown_state.get("triggered", False) and not trade_executed):
            
            # Check if 5-min close below breakdown level
            breakdown_trigger_condition = last_close_5m < SHORT_BREAKDOWN_TRIGGER
            # Check if current price is in entry zone
            breakdown_entry_condition = (SHORT_BREAKDOWN_ENTRY_LOW <= current_close_5m <= SHORT_BREAKDOWN_ENTRY_HIGH)
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and volume_confirmed

            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • 5m close below ${SHORT_BREAKDOWN_TRIGGER:,}: {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry in ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
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
        
        # 4. SHORT - Fade Strategy
        if (short_strategies_enabled and 
            not short_fade_state.get("triggered", False) and not trade_executed):
            
            # Check if price spikes above fade trigger
            fade_trigger_condition = (SHORT_FADE_TRIGGER_LOW <= last_high_5m <= SHORT_FADE_TRIGGER_HIGH)
            # Check if current price is at entry level (rejection after spike)
            fade_entry_condition = abs(current_close_5m - SHORT_FADE_ENTRY) <= 5  # Within $5 of entry
            
            fade_ready = fade_trigger_condition and fade_entry_condition and volume_confirmed

            logger.info("🔍 SHORT - Fade Strategy Analysis:")
            logger.info(f"   • Spike to ${SHORT_FADE_TRIGGER_LOW:,}–${SHORT_FADE_TRIGGER_HIGH:,}: {'✅' if fade_trigger_condition else '❌'} (last high: {last_high_5m:,.2f})")
            logger.info(f"   • Entry at ${SHORT_FADE_ENTRY:,}: {'✅' if fade_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{VOLUME_CONFIRMATION_FACTOR}x): {'✅' if volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Fade Short Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")

            if fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - Fade Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Fade Short (New Setup)",
                    entry_price=SHORT_FADE_ENTRY,
                    stop_loss=SHORT_FADE_STOP_LOSS,
                    take_profit=SHORT_FADE_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Fade Short trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_FADE_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_FADE_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_FADE_TP1:,.2f}, TP2: ${SHORT_FADE_TP2:,.2f}")
                    logger.info("Strategy: Spike to 4,880-4,900 with rejection with volume confirmation")
                    
                    # Save trigger state
                    short_fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_FADE_ENTRY
                    }
                    save_trigger_state(short_fade_state, SHORT_FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Short Fade trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Long breakout triggered: {long_breakout_state.get('triggered', False)}")
            logger.info(f"Long reclaim triggered: {long_reclaim_state.get('triggered', False)}")
            logger.info(f"Short breakdown triggered: {short_breakdown_state.get('triggered', False)}")
            logger.info(f"Short fade triggered: {short_fade_state.get('triggered', False)}")
        
        logger.info("=== ETH Intraday Trading Strategy Alert completed ===")
        return current_ts_5m
        
    except Exception as e:
        logger.error(f"Error in ETH Intraday Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Intraday Trading Strategy Alert completed (with error) ===")
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
    
    logger.info("Starting ETH Intraday Trading Strategy Monitor (New Setup)")
    if direction == 'BOTH':
        logger.info("Strategy: ETH Intraday New Setup - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("LONG - Breakout: 5-min close above 4,888, buy 4,880–4,900 on retest hold")
    logger.info("LONG - Reclaim: Flush < 4,660 then fast reclaim; buy 4,665–4,685")
    logger.info("SHORT - Breakdown: 5-min close below 4,660, sell 4,648–4,660 on failed retest")
    logger.info("SHORT - Fade: Spike to 4,880–4,900 with rejection (upper wick/divergence)")
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
