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
import csv

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

# ETH Trading Strategy Parameters (New Two-Sided Intraday Plan)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_15M = "FIFTEEN_MINUTE"  # Primary timeframe for triggers
GRANULARITY_5M = "FIVE_MINUTE"      # Secondary timeframe
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context - Updated with new levels
CURRENT_ETH_PRICE = 4644.00  # ETH Spot ~$4,644
TODAY_HIGH = 4644.00         # 24h high = $4,644
TODAY_LOW = 4396.00          # 24h low = $4,396

# LONG SETUPS
# Long - Breakout long
LONG_BREAKOUT_TRIGGER = 4644.0     # Trigger: 15m close above the 24h high (~$4,644)
LONG_BREAKOUT_ENTRY_LOW = 4640     # Entry zone: 4,640–4,650 on retest/hold
LONG_BREAKOUT_ENTRY_HIGH = 4650
LONG_BREAKOUT_ENTRY = 4645         # Entry: 4,645 (mid of zone)
LONG_BREAKOUT_STOP_LOSS = 4608     # Invalidation: 0.8% below trigger bar (~$4,608)
LONG_BREAKOUT_TP1 = 4725           # TP1: 2R target
LONG_BREAKOUT_TP2 = 4805           # TP2: Trail target

# Long - Sweep-and-reclaim
LONG_SWEEP_TRIGGER_LOW = 4396      # Trigger: Wick below the 24h low (~$4,396)
LONG_SWEEP_TRIGGER_HIGH = 4400     # Then 15m close back inside range and above VWAP
LONG_SWEEP_ENTRY_LOW = 4405        # Entry zone: 4,405–4,415
LONG_SWEEP_ENTRY_HIGH = 4415
LONG_SWEEP_ENTRY = 4410            # Entry: 4,410 (mid of zone)
LONG_SWEEP_STOP_LOSS = 4396        # Invalidation: low of sweep
LONG_SWEEP_TP1 = 4485              # TP1: 1.5R target
LONG_SWEEP_TP2 = 4560              # TP2: 2.5R target

# SHORT SETUPS
# Short - Breakdown
SHORT_BREAKDOWN_TRIGGER = 4396     # Trigger: 15m close below the 24h low (~$4,396) with rising OI
SHORT_BREAKDOWN_ENTRY_LOW = 4390   # Entry zone: 4,390–4,400 on failed reclaim
SHORT_BREAKDOWN_ENTRY_HIGH = 4400
SHORT_BREAKDOWN_ENTRY = 4395       # Entry: 4,395 (mid of zone)
SHORT_BREAKDOWN_STOP_LOSS = 4432   # Invalidation: 0.8% above trigger bar (~$4,432)
SHORT_BREAKDOWN_TP1 = 4310         # TP1: 2R target
SHORT_BREAKDOWN_TP2 = 4225         # TP2: Prior liquidity pockets

# Short - Exhaustion fade
SHORT_EXHAUSTION_TRIGGER_LOW = 4644  # Trigger: 1–2% spike above the 24h high that immediately fails
SHORT_EXHAUSTION_TRIGGER_HIGH = 4737  # ~2% above 4,644 = 4,737
SHORT_EXHAUSTION_ENTRY_LOW = 4630     # Entry zone: 4,630–4,640 on 5m lower high
SHORT_EXHAUSTION_ENTRY_HIGH = 4640
SHORT_EXHAUSTION_ENTRY = 4635         # Entry: 4,635 (mid of zone)
SHORT_EXHAUSTION_STOP_LOSS = 4737     # Invalidation: wick high
SHORT_EXHAUSTION_TP1 = 4550           # TP1: VWAP
SHORT_EXHAUSTION_TP2 = 4450           # TP2: 1.8–2.5R target

# Volume confirmation requirements
LONG_VOLUME_FACTOR = 1.5   # Volume confirmation ≥1.5× 20-SMA on the trigger candle
SHORT_VOLUME_FACTOR = 1.5  # Volume confirmation ≥1.5× 20-SMA on the trigger candle

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
LONG_SWEEP_TRIGGER_FILE = "eth_sweep_trigger_state.json"
SHORT_BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"
SHORT_EXHAUSTION_TRIGGER_FILE = "eth_exhaustion_trigger_state.json"

def log_trade_to_csv(trade_data):
    """
    Log trade details to CSV file
    
    Args:
        trade_data: Dictionary containing trade information
    """
    csv_file = "chatgpt_trades.csv"
    
    # Define CSV headers
    headers = [
        'timestamp', 'strategy', 'symbol', 'side', 'entry_price', 'stop_loss', 
        'take_profit', 'position_size_usd', 'margin', 'leverage', 'volume_sma', 
        'volume_ratio', 'current_price', 'market_conditions', 'trade_status', 
        'execution_time', 'notes'
    ]
    
    try:
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            
            # Write headers if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write trade data
            writer.writerow(trade_data)
            
        logger.info(f"✅ Trade logged to {csv_file}")
        
    except Exception as e:
        logger.error(f"❌ Failed to log trade to CSV: {e}")

def test_csv_logging():
    """
    Test function to verify CSV logging is working correctly
    """
    logger.info("🧪 Testing CSV logging functionality...")
    
    # Test ETH Breakout Long trade data
    eth_breakout_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-Breakout',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4645.0,
        'stop_loss': 4608.0,
        'take_profit': 4725.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 800.0,
        'volume_ratio': 1.8,
        'current_price': 4645.0,
        'market_conditions': '24h Range: $4,396-$4,644',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Breakout Long'
    }
    
    # Test ETH Sweep-and-reclaim trade data
    eth_sweep_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-Sweep',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4410.0,
        'stop_loss': 4396.0,
        'take_profit': 4485.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 750.0,
        'volume_ratio': 1.6,
        'current_price': 4410.0,
        'market_conditions': '24h Range: $4,396-$4,644',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Sweep-and-reclaim'
    }
    
    # Log test trades
    log_trade_to_csv(eth_breakout_data)
    log_trade_to_csv(eth_sweep_data)
    
    logger.info("✅ CSV logging test completed!")
    logger.info("📊 Check chatgpt_trades.csv to verify test trades were added correctly")

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
    ETH Two-Sided Intraday Trading Strategy Alert
    
    Spiros, here's a two-sided ETH intraday plan for today.

    Longs
    •	Breakout: 15m close above the 24h high (~$4,644). Enter on a quick retest/hold above. Invalidation: 0.8% below trigger bar or last swing. Targets: 2R then trail.
    •	Sweep-and-reclaim: Wick below the 24h low (~$4,396), then 15m close back inside the range and above VWAP. Invalidation: low of sweep. Targets: 1.5–2.5R. Note: funding is slightly negative across majors, so failed breaks down can squeeze up.

    Shorts
    •	Breakdown: 15m close below the 24h low (~$4,396) with rising OI. Enter on failed reclaim. Invalidation: 0.8% above trigger bar. Targets: 2R then prior liquidity pockets. Monitor OI/funding to confirm.
    •	Exhaustion fade: 1–2% spike above the 24h high that immediately fails, with funding flipping positive across majors and OI jumping. Entry on 5m lower high. Invalidation: wick high. Targets: VWAP, then 1.8–2.5R.

    Context checks before entry
    •	Funding tilt: slight negative now → don't chase late shorts into lows. Recheck during NY hours.
    •	Use volume confirmation ≥1.5× 20-SMA on the trigger candle.
    •	Size by 15m ATR; keep R≥2.
    
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
    long_sweep_state = load_trigger_state(LONG_SWEEP_TRIGGER_FILE)
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    short_exhaustion_state = load_trigger_state(SHORT_EXHAUSTION_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 15-minute candles for analysis (primary timeframe)
        end = now
        start = now - timedelta(hours=6)  # Get enough data for 20-period volume average
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 15-minute candles for {6} hours...")
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_15M)
        
        # Also get 5-minute candles for secondary analysis
        logger.info(f"Fetching 5-minute candles for {2} hours...")
        start_5m = now - timedelta(hours=2)
        start_ts_5m = int(start_5m.timestamp())
        candles_5m = safe_get_candles(cb_service, PRODUCT_ID, start_ts_5m, end_ts, GRANULARITY_5M)
        
        if not candles_15m or len(candles_15m) < VOLUME_PERIOD + 3:
            logger.warning("Not enough 15-minute candle data for trading strategy alert.")
            return last_alert_ts
            
        if not candles_5m or len(candles_5m) < VOLUME_PERIOD + 3:
            logger.warning("Not enough 5-minute candle data for trading strategy alert.")
            return last_alert_ts
            
        # Sort by timestamp ascending (oldest first)
        candles_15m = sorted(candles_15m, key=lambda x: int(x['start']))
        candles_5m = sorted(candles_5m, key=lambda x: int(x['start']))
        
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
        
        # Get current and last closed 5-minute candles
        current_candle_5m = candles_5m[-1]  # Most recent (potentially incomplete)
        last_closed_5m = candles_5m[-2]     # Last completed candle
        
        current_close_5m = float(current_candle_5m['close'])
        current_high_5m = float(current_candle_5m['high'])
        current_low_5m = float(current_candle_5m['low'])
        current_volume_5m = float(current_candle_5m['volume'])
        
        last_close_5m = float(last_closed_5m['close'])
        last_high_5m = float(last_closed_5m['high'])
        last_low_5m = float(last_closed_5m['low'])
        last_volume_5m = float(last_closed_5m['volume'])
        last_ts_5m = datetime.fromtimestamp(int(last_closed_5m['start']), UTC)
        
        # Calculate 20-period average volume for both timeframes
        volume_candles_15m = candles_15m[-(VOLUME_PERIOD+1):-1]  # Last 20 completed candles
        avg_volume_15m = sum(float(c['volume']) for c in volume_candles_15m) / len(volume_candles_15m)
        relative_volume_15m = last_volume_15m / avg_volume_15m if avg_volume_15m > 0 else 0
        
        volume_candles_5m = candles_5m[-(VOLUME_PERIOD+1):-1]  # Last 20 completed candles
        avg_volume_5m = sum(float(c['volume']) for c in volume_candles_5m) / len(volume_candles_5m)
        relative_volume_5m = last_volume_5m / avg_volume_5m if avg_volume_5m > 0 else 0
        
        # Check volume confirmation for different strategies
        long_volume_confirmed = relative_volume_15m >= LONG_VOLUME_FACTOR
        short_volume_confirmed = relative_volume_15m >= SHORT_VOLUME_FACTOR
        
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
        logger.info(f"   • Primary timeframe: 15-minute candles")
        logger.info(f"   • Volume requirements: 15m RVOL ≥ {LONG_VOLUME_FACTOR}x for long, {SHORT_VOLUME_FACTOR}x for short")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setups:")
            logger.info(f"   • Breakout: 15m close above ${LONG_BREAKOUT_TRIGGER:,}, buy ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,} on retest hold")
            logger.info(f"     Invalidation: 0.8% below trigger bar (${LONG_BREAKOUT_STOP_LOSS:,})")
            logger.info(f"     TP1: ${LONG_BREAKOUT_TP1:,} (2R), TP2: ${LONG_BREAKOUT_TP2:,} (trail)")
            logger.info(f"   • Sweep-and-reclaim: Wick below ${LONG_SWEEP_TRIGGER_LOW:,}, then 15m close back inside range")
            logger.info(f"     Entry: ${LONG_SWEEP_ENTRY_LOW:,}–${LONG_SWEEP_ENTRY_HIGH:,}, Invalidation: low of sweep (${LONG_SWEEP_STOP_LOSS:,})")
            logger.info(f"     TP1: ${LONG_SWEEP_TP1:,} (1.5R), TP2: ${LONG_SWEEP_TP2:,} (2.5R)")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setups:")
            logger.info(f"   • Breakdown: 15m close below ${SHORT_BREAKDOWN_TRIGGER:,}, sell ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,} on failed reclaim")
            logger.info(f"     Invalidation: 0.8% above trigger bar (${SHORT_BREAKDOWN_STOP_LOSS:,})")
            logger.info(f"     TP1: ${SHORT_BREAKDOWN_TP1:,} (2R), TP2: ${SHORT_BREAKDOWN_TP2:,} (prior liquidity pockets)")
            logger.info(f"   • Exhaustion fade: 1–2% spike above ${SHORT_EXHAUSTION_TRIGGER_HIGH:,}, sell ${SHORT_EXHAUSTION_ENTRY_LOW:,}–${SHORT_EXHAUSTION_ENTRY_HIGH:,} on 5m lower high")
            logger.info(f"     Invalidation: wick high (${SHORT_EXHAUSTION_STOP_LOSS:,})")
            logger.info(f"     TP1: ${SHORT_EXHAUSTION_TP1:,} (VWAP), TP2: ${SHORT_EXHAUSTION_TP2:,} (1.8–2.5R)")
            logger.info("")
        
        logger.info(f"Current Price: ${current_close_15m:,.2f}")
        logger.info(f"Last 15M Close: ${last_close_15m:,.2f}, High: ${last_high_15m:,.2f}, Low: ${last_low_15m:,.2f}")
        logger.info(f"15M Volume: {last_volume_15m:,.0f}, 15M SMA(20): {avg_volume_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}x")
        logger.info(f"Last 5M Close: ${last_close_5m:,.2f}, 5M Volume: {last_volume_5m:,.0f}, 5M SMA(20): {avg_volume_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}x")
        logger.info(f"Long Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if long_volume_confirmed else '❌'}")
        logger.info(f"Short Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if short_volume_confirmed else '❌'}")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - Breakout Long Strategy
        if (long_strategies_enabled and 
            not long_breakout_state.get("triggered", False) and not trade_executed):
            
            # Check if 15m close above breakout trigger level
            breakout_trigger_condition = last_close_15m > LONG_BREAKOUT_TRIGGER
            # Check if current price is in entry zone
            breakout_entry_condition = (LONG_BREAKOUT_ENTRY_LOW <= current_close_15m <= LONG_BREAKOUT_ENTRY_HIGH)
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and (relative_volume_15m >= LONG_VOLUME_FACTOR)

            logger.info("🔍 LONG - Breakout Long Strategy Analysis:")
            logger.info(f"   • 15m close above ${LONG_BREAKOUT_TRIGGER:,}: {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry in ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if long_volume_confirmed else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Breakout Long Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Long Strategy conditions met - executing trade...")

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
                    logger.info("Strategy: 15m close above 4,644 with volume confirmation")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Breakout',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': LONG_BREAKOUT_ENTRY,
                        'stop_loss': LONG_BREAKOUT_STOP_LOSS,
                        'take_profit': LONG_BREAKOUT_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_15m,
                        'volume_ratio': relative_volume_15m,
                        'current_price': current_close_15m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 15m close > ${LONG_BREAKOUT_TRIGGER:,}, Volume: {relative_volume_15m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_15m['start']),
                        "entry_price": LONG_BREAKOUT_ENTRY
                    }
                    save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout Long trade failed: {trade_result}")
        
        # 2. LONG - Sweep-and-reclaim Strategy
        if (long_strategies_enabled and 
            not long_sweep_state.get("triggered", False) and not trade_executed):
            
            # Check if Wick below 24h low and then 15m close back inside range
            sweep_trigger_condition = (last_low_15m < LONG_SWEEP_TRIGGER_LOW and 
                                     last_close_15m > LONG_SWEEP_TRIGGER_HIGH)
            # Check if current price is in entry zone (after reclaim)
            sweep_entry_condition = (LONG_SWEEP_ENTRY_LOW <= current_close_15m <= LONG_SWEEP_ENTRY_HIGH)
            
            sweep_ready = sweep_trigger_condition and sweep_entry_condition and (relative_volume_15m >= LONG_VOLUME_FACTOR)

            logger.info("🔍 LONG - Sweep-and-reclaim Strategy Analysis:")
            logger.info(f"   • Wick below ${LONG_SWEEP_TRIGGER_LOW:,} and close back above ${LONG_SWEEP_TRIGGER_HIGH:,}: {'✅' if sweep_trigger_condition else '❌'} (last low: {last_low_15m:,.2f}, last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry in ${LONG_SWEEP_ENTRY_LOW:,}–${LONG_SWEEP_ENTRY_HIGH:,}: {'✅' if sweep_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if (relative_volume_15m >= LONG_VOLUME_FACTOR) else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Sweep-and-reclaim Ready: {'🎯 YES' if sweep_ready else '⏳ NO'}")

            if sweep_ready:
                logger.info("")
                logger.info("🎯 LONG - Sweep-and-reclaim Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Sweep-and-reclaim (New Setup)",
                    entry_price=LONG_SWEEP_ENTRY,
                    stop_loss=LONG_SWEEP_STOP_LOSS,
                    take_profit=LONG_SWEEP_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Sweep-and-reclaim trade executed successfully!")
                    logger.info(f"Entry: ${LONG_SWEEP_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_SWEEP_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_SWEEP_TP1:,.2f}, TP2: ${LONG_SWEEP_TP2:,.2f}")
                    logger.info("Strategy: Wick below 4,396, then 15m close back inside range")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Sweep',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': LONG_SWEEP_ENTRY,
                        'stop_loss': LONG_SWEEP_STOP_LOSS,
                        'take_profit': LONG_SWEEP_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_15m,
                        'volume_ratio': relative_volume_15m,
                        'current_price': current_close_15m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Wick below ${LONG_SWEEP_TRIGGER_LOW:,}, Volume: {relative_volume_15m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_sweep_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_15m['start']),
                        "entry_price": LONG_SWEEP_ENTRY
                    }
                    save_trigger_state(long_sweep_state, LONG_SWEEP_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Sweep-and-reclaim trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if (short_strategies_enabled and 
            not short_breakdown_state.get("triggered", False) and not trade_executed):
            
            # Check if 15m close below breakdown trigger level
            breakdown_trigger_condition = last_close_15m < SHORT_BREAKDOWN_TRIGGER
            # Check if current price is in entry zone
            breakdown_entry_condition = (SHORT_BREAKDOWN_ENTRY_LOW <= current_close_15m <= SHORT_BREAKDOWN_ENTRY_HIGH)
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and (relative_volume_15m >= SHORT_VOLUME_FACTOR)

            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • 15m close below ${SHORT_BREAKDOWN_TRIGGER:,}: {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry in ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if (relative_volume_15m >= SHORT_VOLUME_FACTOR) else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

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
                    trade_type="ETH Breakdown (New Setup)",
                    entry_price=SHORT_BREAKDOWN_ENTRY,
                    stop_loss=SHORT_BREAKDOWN_STOP_LOSS,
                    take_profit=SHORT_BREAKDOWN_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_BREAKDOWN_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_BREAKDOWN_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_BREAKDOWN_TP1:,.2f}, TP2: ${SHORT_BREAKDOWN_TP2:,.2f}")
                    logger.info("Strategy: 15m close below 4,396 with volume confirmation")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Breakdown',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': SHORT_BREAKDOWN_ENTRY,
                        'stop_loss': SHORT_BREAKDOWN_STOP_LOSS,
                        'take_profit': SHORT_BREAKDOWN_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_15m,
                        'volume_ratio': relative_volume_15m,
                        'current_price': current_close_15m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 15m close < ${SHORT_BREAKDOWN_TRIGGER:,}, Volume: {relative_volume_15m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_15m['start']),
                        "entry_price": SHORT_BREAKDOWN_ENTRY
                    }
                    save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown trade failed: {trade_result}")
        
        # 4. SHORT - Exhaustion fade Strategy
        if (short_strategies_enabled and 
            not short_exhaustion_state.get("triggered", False) and not trade_executed):
            
            # Check if 1–2% spike above 24h high (using 5m timeframe as specified)
            exhaustion_trigger_condition = last_high_5m > SHORT_EXHAUSTION_TRIGGER_HIGH
            # Check if current price is in entry zone (after spike)
            exhaustion_entry_condition = (SHORT_EXHAUSTION_ENTRY_LOW <= current_close_5m <= SHORT_EXHAUSTION_ENTRY_HIGH)
            
            exhaustion_ready = exhaustion_trigger_condition and exhaustion_entry_condition and (relative_volume_5m >= SHORT_VOLUME_FACTOR)

            logger.info("🔍 SHORT - Exhaustion fade Strategy Analysis:")
            logger.info(f"   • 1–2% spike above ${SHORT_EXHAUSTION_TRIGGER_HIGH:,}: {'✅' if exhaustion_trigger_condition else '❌'} (last high: {last_high_5m:,.2f})")
            logger.info(f"   • Entry in ${SHORT_EXHAUSTION_ENTRY_LOW:,}–${SHORT_EXHAUSTION_ENTRY_HIGH:,}: {'✅' if exhaustion_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if (relative_volume_5m >= SHORT_VOLUME_FACTOR) else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Exhaustion fade Ready: {'🎯 YES' if exhaustion_ready else '⏳ NO'}")

            if exhaustion_ready:
                logger.info("")
                logger.info("🎯 SHORT - Exhaustion fade Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Exhaustion fade (New Setup)",
                    entry_price=SHORT_EXHAUSTION_ENTRY,
                    stop_loss=SHORT_EXHAUSTION_STOP_LOSS,
                    take_profit=SHORT_EXHAUSTION_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Exhaustion fade trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_EXHAUSTION_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_EXHAUSTION_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_EXHAUSTION_TP1:,.2f}, TP2: ${SHORT_EXHAUSTION_TP2:,.2f}")
                    logger.info("Strategy: 1–2% spike above 4,644, sell 4,630-4,640 on 5m lower high")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Exhaustion',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': SHORT_EXHAUSTION_ENTRY,
                        'stop_loss': SHORT_EXHAUSTION_STOP_LOSS,
                        'take_profit': SHORT_EXHAUSTION_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 1–2% spike above ${SHORT_EXHAUSTION_TRIGGER_HIGH:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_exhaustion_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_EXHAUSTION_ENTRY
                    }
                    save_trigger_state(short_exhaustion_state, SHORT_EXHAUSTION_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Exhaustion fade trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Breakout Long triggered: {long_breakout_state.get('triggered', False)}")
            logger.info(f"Sweep-and-reclaim triggered: {long_sweep_state.get('triggered', False)}")
            logger.info(f"Breakdown triggered: {short_breakdown_state.get('triggered', False)}")
            logger.info(f"Exhaustion fade triggered: {short_exhaustion_state.get('triggered', False)}")
        
        logger.info("=== ETH ATH Trading Strategy Alert completed ===")
        return current_ts_15m
        
    except Exception as e:
        logger.error(f"Error in ETH Intraday Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH ATH Trading Strategy Alert completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Two-Sided Intraday Trading Strategy Monitor with optional direction filter')
    parser.add_argument('--direction', choices=['LONG', 'SHORT', 'BOTH'], default='BOTH',
                       help='Trading direction to monitor: LONG, SHORT, or BOTH (default: BOTH)')
    parser.add_argument('--test-csv', action='store_true', help='Test CSV logging functionality')
    args = parser.parse_args()
    
    # Test CSV logging if requested
    if args.test_csv:
        test_csv_logging()
        return
    
    # Print usage examples
    logger.info("Usage examples:")
    logger.info("  python crypto_alert_monitor_eth.py                    # Monitor both LONG and SHORT strategies")
    logger.info("  python crypto_alert_monitor_eth.py --direction LONG   # Monitor only LONG strategies")
    logger.info("  python crypto_alert_monitor_eth.py --direction SHORT  # Monitor only SHORT strategies")
    logger.info("")
    
    direction = args.direction.upper()
    
    logger.info("Starting ETH Two-Sided Intraday Trading Strategy Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: ETH Two-Sided Intraday - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("LONG - Breakout: 15m close above 4,644, buy 4,640–4,650 on retest hold")
    logger.info("LONG - Sweep-and-reclaim: Wick below 4,396, then 15m close back inside range")
    logger.info("SHORT - Breakdown: 15m close below 4,396, sell 4,390–4,400 on failed reclaim")
    logger.info("SHORT - Exhaustion fade: 1–2% spike above 4,644, sell 4,630–4,640 on 5m lower high")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info(f"Volume: 15m RVOL ≥ {LONG_VOLUME_FACTOR}x for all strategies")
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
