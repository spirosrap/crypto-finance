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

# Current market context - Updated with new levels from Aug 28 setup
CURRENT_ETH_PRICE = 4650.00  # ETH Spot ~$4,650
TODAY_HIGH = 4650.00         # 24h high = $4,650
TODAY_LOW = 4530.00          # 24h low = $4,530

# Key S/R levels from Barchart pivots
R1 = 4656  # Resistance 1
R2 = 4721  # Resistance 2  
R3 = 4782  # Resistance 3
S1 = 4531  # Support 1
S2 = 4470  # Support 2
S3 = 4405  # Support 3

# LONG SETUPS
# Long - Breakout long
LONG_BREAKOUT_TRIGGER = 4660.0     # Trigger: 5-min close > 4,660
LONG_BREAKOUT_ENTRY_LOW = 4660     # Entry zone: 4,660–4,675 on retest
LONG_BREAKOUT_ENTRY_HIGH = 4675
LONG_BREAKOUT_ENTRY = 4668         # Entry: 4,668 (mid of zone)
LONG_BREAKOUT_STOP_LOSS = 4628     # Invalidation: < 4,628
LONG_BREAKOUT_TP1 = 4721           # TP1: 4,721
LONG_BREAKOUT_TP2 = 4782           # TP2: 4,782

# Long - Pullback
LONG_PULLBACK_TRIGGER_LOW = 4531   # Trigger: Bid 4,535–4,545 (S1 zone) with wick rejection
LONG_PULLBACK_TRIGGER_HIGH = 4545
LONG_PULLBACK_ENTRY_LOW = 4535     # Entry zone: 4,535–4,545
LONG_PULLBACK_ENTRY_HIGH = 4545
LONG_PULLBACK_ENTRY = 4540         # Entry: 4,540 (mid of zone)
LONG_PULLBACK_STOP_LOSS = 4505     # Invalidation: < 4,505
LONG_PULLBACK_TP1 = 4656           # TP1: 4,656
LONG_PULLBACK_TP2 = 4721           # TP2: 4,721

# SHORT SETUPS
# Short - Breakdown
SHORT_BREAKDOWN_TRIGGER = 4530     # Trigger: 5-min close < 4,530
SHORT_BREAKDOWN_ENTRY_LOW = 4520   # Entry zone: 4,520–4,528 on retest
SHORT_BREAKDOWN_ENTRY_HIGH = 4528
SHORT_BREAKDOWN_ENTRY = 4524       # Entry: 4,524 (mid of zone)
SHORT_BREAKDOWN_STOP_LOSS = 4548   # Invalidation: > 4,548
SHORT_BREAKDOWN_TP1 = 4470         # TP1: 4,470
SHORT_BREAKDOWN_TP2 = 4405         # TP2: 4,405

# Short - Fade
SHORT_FADE_TRIGGER_LOW = 4650      # Trigger: Rejection at 4,650–4,660 (failed break of R1)
SHORT_FADE_TRIGGER_HIGH = 4660
SHORT_FADE_ENTRY_LOW = 4650        # Entry zone: 4,650–4,660
SHORT_FADE_ENTRY_HIGH = 4660
SHORT_FADE_ENTRY = 4655            # Entry: 4,655 (mid of zone)
SHORT_FADE_STOP_LOSS = 4678        # Invalidation: > 4,678
SHORT_FADE_TP1 = 4600              # TP1: 4,600
SHORT_FADE_TP2 = 4531              # TP2: 4,531

# Volume confirmation requirements
LONG_VOLUME_FACTOR = 1.5   # Volume confirmation ≥1.5× 20-SMA on the trigger candle
SHORT_VOLUME_FACTOR = 1.5  # Volume confirmation ≥1.5× 20-SMA on the trigger candle

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
LONG_PULLBACK_TRIGGER_FILE = "eth_pullback_trigger_state.json"
SHORT_BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"
SHORT_FADE_TRIGGER_FILE = "eth_fade_trigger_state.json"

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
        'entry_price': 4668.0,
        'stop_loss': 4628.0,
        'take_profit': 4721.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 800.0,
        'volume_ratio': 1.8,
        'current_price': 4668.0,
        'market_conditions': 'Key S/R: R1=$4,656, S1=$4,531',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Breakout Long (Aug 28 Setup)'
    }
    
    # Test ETH Pullback trade data
    eth_pullback_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-Pullback',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4540.0,
        'stop_loss': 4505.0,
        'take_profit': 4656.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 750.0,
        'volume_ratio': 1.6,
        'current_price': 4540.0,
        'market_conditions': 'Key S/R: R1=$4,656, S1=$4,531',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Pullback (Aug 28 Setup)'
    }
    
    # Log test trades
    log_trade_to_csv(eth_breakout_data)
    log_trade_to_csv(eth_pullback_data)
    
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
    
    Spiros, here are ETH intraday setups for Thu, Aug 28 (America/Chicago).

    Context:
    •	Funding is near flat to mildly positive across majors. Avoid chasing longs if funding spikes.
    •	Key S/R today from Barchart pivots: R1 ≈ 4,656, R2 ≈ 4,721, R3 ≈ 4,782. S1 ≈ 4,531, S2 ≈ 4,470, S3 ≈ 4,405.

    Bias	Trigger & Entry	Invalidation (SL)	Targets	Confirmations
    Long BO	5-min close > 4,660, buy 4,660–4,675 on retest	< 4,628	4,721 → 4,782	5-min vol > 1.5× 20-SMA; OI rising ≥1% vs last hour; funding ≤ +0.01% 8h.
    Long PB	Bid 4,535–4,545 (S1 zone) with wick rejection	< 4,505	4,656 → 4,721	Delta flips positive; heatmap shows clustered shorts above entry.
    Short BD	5-min close < 4,530, sell 4,520–4,528 on retest	> 4,548	4,470 → 4,405	OI expanding; long liquidations printing.
    Short Fade	Rejection at 4,650–4,660 (failed break of R1)	> 4,678	4,600 → 4,531	Negative delta at highs; funding tilts positive into the push.

    Notes:
    •	Live price ref: Coingecko/CMC if you need exact prints.
    •	Derivatives dashboard for OI/liqs/funding if you want a single pane.

    Risk rules:
    •	0.5R per attempt. Max two attempts per side. Hard stop = market.
    •	Skip entries 3 minutes before and after scheduled data prints.
    •	If OI contracts while price moves in your favor, take TP1 faster.
    
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
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    short_fade_state = load_trigger_state(SHORT_FADE_TRIGGER_FILE)
    
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
        current_ts_5m = datetime.fromtimestamp(int(current_candle_5m['start']), UTC)
        
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
        logger.info("🚀 Spiros, here are ETH intraday setups for Thu, Aug 28 (America/Chicago).")
        logger.info(f"📊 Live Levels:")
        logger.info(f"   • ETH ≈ ${current_close_5m:,.2f}")
        logger.info(f"   • Key S/R: R1=${R1:,}, R2=${R2:,}, R3=${R3:,} | S1=${S1:,}, S2=${S2:,}, S3=${S3:,}")
        logger.info("")
        logger.info("📊 Strategy Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Primary timeframe: 5-minute candles")
        logger.info(f"   • Volume requirements: 5m RVOL ≥ {LONG_VOLUME_FACTOR}x for all strategies")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setups:")
            logger.info(f"   • Breakout: 5-min close > ${LONG_BREAKOUT_TRIGGER:,}, buy ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,} on retest")
            logger.info(f"     Invalidation: < ${LONG_BREAKOUT_STOP_LOSS:,}")
            logger.info(f"     TP1: ${LONG_BREAKOUT_TP1:,}, TP2: ${LONG_BREAKOUT_TP2:,}")
            logger.info(f"   • Pullback: Bid ${LONG_PULLBACK_ENTRY_LOW:,}–${LONG_PULLBACK_ENTRY_HIGH:,} (S1 zone) with wick rejection")
            logger.info(f"     Entry: ${LONG_PULLBACK_ENTRY_LOW:,}–${LONG_PULLBACK_ENTRY_HIGH:,}, Invalidation: < ${LONG_PULLBACK_STOP_LOSS:,}")
            logger.info(f"     TP1: ${LONG_PULLBACK_TP1:,}, TP2: ${LONG_PULLBACK_TP2:,}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setups:")
            logger.info(f"   • Breakdown: 5-min close < ${SHORT_BREAKDOWN_TRIGGER:,}, sell ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,} on retest")
            logger.info(f"     Invalidation: > ${SHORT_BREAKDOWN_STOP_LOSS:,}")
            logger.info(f"     TP1: ${SHORT_BREAKDOWN_TP1:,}, TP2: ${SHORT_BREAKDOWN_TP2:,}")
            logger.info(f"   • Fade: Rejection at ${SHORT_FADE_TRIGGER_LOW:,}–${SHORT_FADE_TRIGGER_HIGH:,} (failed break of R1)")
            logger.info(f"     Entry: ${SHORT_FADE_ENTRY_LOW:,}–${SHORT_FADE_ENTRY_HIGH:,}, Invalidation: > ${SHORT_FADE_STOP_LOSS:,}")
            logger.info(f"     TP1: ${SHORT_FADE_TP1:,}, TP2: ${SHORT_FADE_TP2:,}")
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
            
            # Check if 5-min close above breakout trigger level
            breakout_trigger_condition = last_close_5m > LONG_BREAKOUT_TRIGGER
            # Check if current price is in entry zone
            breakout_entry_condition = (LONG_BREAKOUT_ENTRY_LOW <= current_close_5m <= LONG_BREAKOUT_ENTRY_HIGH)
            
            breakout_ready = breakout_trigger_condition and breakout_entry_condition and (relative_volume_5m >= LONG_VOLUME_FACTOR)

            logger.info("🔍 LONG - Breakout Long Strategy Analysis:")
            logger.info(f"   • 5-min close > ${LONG_BREAKOUT_TRIGGER:,}: {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry in ${LONG_BREAKOUT_ENTRY_LOW:,}–${LONG_BREAKOUT_ENTRY_HIGH:,}: {'✅' if breakout_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if (relative_volume_5m >= LONG_VOLUME_FACTOR) else '❌'} (current: {relative_volume_5m:.2f}x)")
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
                    trade_type="ETH Breakout Long (Aug 28 Setup)",
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
                    logger.info("Strategy: 5-min close > 4,660, buy 4,660–4,675 on retest")
                    
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
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"Key S/R: R1=${R1:,}, S1=${S1:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 5-min close > ${LONG_BREAKOUT_TRIGGER:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": LONG_BREAKOUT_ENTRY
                    }
                    save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout Long trade failed: {trade_result}")
        
        # 2. LONG - Pullback Strategy
        if (long_strategies_enabled and 
            not long_pullback_state.get("triggered", False) and not trade_executed):
            
            # Check if price is in S1 zone with wick rejection (pullback to support)
            pullback_trigger_condition = (LONG_PULLBACK_TRIGGER_LOW <= current_low_5m <= LONG_PULLBACK_TRIGGER_HIGH)
            # Check if current price is in entry zone
            pullback_entry_condition = (LONG_PULLBACK_ENTRY_LOW <= current_close_5m <= LONG_PULLBACK_ENTRY_HIGH)
            
            pullback_ready = pullback_trigger_condition and pullback_entry_condition and (relative_volume_5m >= LONG_VOLUME_FACTOR)

            logger.info("🔍 LONG - Pullback Strategy Analysis:")
            logger.info(f"   • Bid ${LONG_PULLBACK_TRIGGER_LOW:,}–${LONG_PULLBACK_TRIGGER_HIGH:,} (S1 zone) with wick rejection: {'✅' if pullback_trigger_condition else '❌'} (current low: {current_low_5m:,.2f})")
            logger.info(f"   • Entry in ${LONG_PULLBACK_ENTRY_LOW:,}–${LONG_PULLBACK_ENTRY_HIGH:,}: {'✅' if pullback_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if (relative_volume_5m >= LONG_VOLUME_FACTOR) else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Pullback Ready: {'🎯 YES' if pullback_ready else '⏳ NO'}")

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
                    trade_type="ETH Pullback (Aug 28 Setup)",
                    entry_price=LONG_PULLBACK_ENTRY,
                    stop_loss=LONG_PULLBACK_STOP_LOSS,
                    take_profit=LONG_PULLBACK_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Pullback trade executed successfully!")
                    logger.info(f"Entry: ${LONG_PULLBACK_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_PULLBACK_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_PULLBACK_TP1:,.2f}, TP2: ${LONG_PULLBACK_TP2:,.2f}")
                    logger.info("Strategy: Bid 4,535–4,545 (S1 zone) with wick rejection")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Pullback',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': LONG_PULLBACK_ENTRY,
                        'stop_loss': LONG_PULLBACK_STOP_LOSS,
                        'take_profit': LONG_PULLBACK_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"Key S/R: R1=${R1:,}, S1=${S1:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Bid ${LONG_PULLBACK_TRIGGER_LOW:,}–${LONG_PULLBACK_TRIGGER_HIGH:,} (S1 zone), Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_pullback_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": LONG_PULLBACK_ENTRY
                    }
                    save_trigger_state(long_pullback_state, LONG_PULLBACK_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Pullback trade failed: {trade_result}")
        
        # 3. SHORT - Breakdown Strategy
        if (short_strategies_enabled and 
            not short_breakdown_state.get("triggered", False) and not trade_executed):
            
            # Check if 5-min close below breakdown trigger level
            breakdown_trigger_condition = last_close_5m < SHORT_BREAKDOWN_TRIGGER
            # Check if current price is in entry zone
            breakdown_entry_condition = (SHORT_BREAKDOWN_ENTRY_LOW <= current_close_5m <= SHORT_BREAKDOWN_ENTRY_HIGH)
            
            breakdown_ready = breakdown_trigger_condition and breakdown_entry_condition and (relative_volume_5m >= SHORT_VOLUME_FACTOR)

            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • 5-min close < ${SHORT_BREAKDOWN_TRIGGER:,}: {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry in ${SHORT_BREAKDOWN_ENTRY_LOW:,}–${SHORT_BREAKDOWN_ENTRY_HIGH:,}: {'✅' if breakdown_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if (relative_volume_5m >= SHORT_VOLUME_FACTOR) else '❌'} (current: {relative_volume_5m:.2f}x)")
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
                    trade_type="ETH Breakdown (Aug 28 Setup)",
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
                    logger.info("Strategy: 5-min close < 4,530, sell 4,520–4,528 on retest")
                    
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
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"Key S/R: R1=${R1:,}, S1=${S1:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 5-min close < ${SHORT_BREAKDOWN_TRIGGER:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_BREAKDOWN_ENTRY
                    }
                    save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown trade failed: {trade_result}")
        
        # 4. SHORT - Fade Strategy
        if (short_strategies_enabled and 
            not short_fade_state.get("triggered", False) and not trade_executed):
            
            # Check if rejection at 4,650–4,660 (failed break of R1)
            fade_trigger_condition = (SHORT_FADE_TRIGGER_LOW <= last_high_5m <= SHORT_FADE_TRIGGER_HIGH)
            # Check if current price is in entry zone (after rejection)
            fade_entry_condition = (SHORT_FADE_ENTRY_LOW <= current_close_5m <= SHORT_FADE_ENTRY_HIGH)
            
            fade_ready = fade_trigger_condition and fade_entry_condition and (relative_volume_5m >= SHORT_VOLUME_FACTOR)

            logger.info("🔍 SHORT - Fade Strategy Analysis:")
            logger.info(f"   • Rejection at ${SHORT_FADE_TRIGGER_LOW:,}–${SHORT_FADE_TRIGGER_HIGH:,} (failed break of R1): {'✅' if fade_trigger_condition else '❌'} (last high: {last_high_5m:,.2f})")
            logger.info(f"   • Entry in ${SHORT_FADE_ENTRY_LOW:,}–${SHORT_FADE_ENTRY_HIGH:,}: {'✅' if fade_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if (relative_volume_5m >= SHORT_VOLUME_FACTOR) else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Fade Ready: {'🎯 YES' if fade_ready else '⏳ NO'}")

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
                    trade_type="ETH Fade (Aug 28 Setup)",
                    entry_price=SHORT_FADE_ENTRY,
                    stop_loss=SHORT_FADE_STOP_LOSS,
                    take_profit=SHORT_FADE_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Fade trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_FADE_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_FADE_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_FADE_TP1:,.2f}, TP2: ${SHORT_FADE_TP2:,.2f}")
                    logger.info("Strategy: Rejection at 4,650–4,660 (failed break of R1)")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Fade',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': SHORT_FADE_ENTRY,
                        'stop_loss': SHORT_FADE_STOP_LOSS,
                        'take_profit': SHORT_FADE_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"Key S/R: R1=${R1:,}, S1=${S1:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Rejection at ${SHORT_FADE_TRIGGER_LOW:,}–${SHORT_FADE_TRIGGER_HIGH:,} (failed break of R1), Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_FADE_ENTRY
                    }
                    save_trigger_state(short_fade_state, SHORT_FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Fade trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Breakout Long triggered: {long_breakout_state.get('triggered', False)}")
            logger.info(f"Pullback triggered: {long_pullback_state.get('triggered', False)}")
            logger.info(f"Breakdown triggered: {short_breakdown_state.get('triggered', False)}")
            logger.info(f"Fade triggered: {short_fade_state.get('triggered', False)}")
        
        logger.info("=== ETH Aug 28 Trading Strategy Alert completed ===")
        return current_ts_5m
        
    except Exception as e:
        logger.error(f"Error in ETH Aug 28 Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Aug 28 Trading Strategy Alert completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Aug 28 Intraday Trading Strategy Monitor with optional direction filter')
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
    
    logger.info("Starting ETH Aug 28 Intraday Trading Strategy Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: ETH Aug 28 Intraday - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("LONG - Breakout: 5-min close > 4,660, buy 4,660–4,675 on retest")
    logger.info("LONG - Pullback: Bid 4,535–4,545 (S1 zone) with wick rejection")
    logger.info("SHORT - Breakdown: 5-min close < 4,530, sell 4,520–4,528 on retest")
    logger.info("SHORT - Fade: Rejection at 4,650–4,660 (failed break of R1)")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info(f"Volume: 5m RVOL ≥ {LONG_VOLUME_FACTOR}x for all strategies")
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
