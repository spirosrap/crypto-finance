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

# ETH Trading Strategy Parameters (New August 30th Setup)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_5M = "FIVE_MINUTE"      # Primary timeframe for triggers
GRANULARITY_1M = "ONE_MINUTE"       # For 1-minute volume spikes
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context - Updated with new levels from Aug 30 setup
CURRENT_ETH_PRICE = 4382.51  # ETH Spot ~$4,382.51
TODAY_HIGH = 4482.49         # HOD = $4,482.49
TODAY_LOW = 4267.43          # LOD = $4,267.43
CURRENT_PRICE = 4382.51      # Last ≈ $4,382.51
MID_RANGE = 4374.96          # Mid-range = (H+L)/2

# Key levels from Aug 30 setup
HOD_LEVEL = 4482.49          # High of Day
LOD_LEVEL = 4267.43          # Low of Day
MID_LEVEL = 4374.96          # Mid-range level
BUFFER_BPS = 10              # 10 basis points buffer

# Setup 1: Breakout-Retest (LONG)
BREAKOUT_RETEST_TRIGGER = 4486.97  # H + 10 bps = 4482.49 + 4.48
BREAKOUT_RETEST_ENTRY_LOW = 4486.97
BREAKOUT_RETEST_ENTRY_HIGH = 4492.00
BREAKOUT_RETEST_SL_BUFFER_BPS = 35
BREAKOUT_RETEST_VOLUME_FACTOR = 1.25
BREAKOUT_RETEST_CONSECUTIVE_CLOSES = 2

# Setup 2: Liquidity-Sweep Reclaim (LONG)
LIQUIDITY_SWEEP_LEVEL = 4267.43
LIQUIDITY_SWEEP_ENTRY_OFFSET_BPS = 5
LIQUIDITY_SWEEP_SL_BUFFER_BPS = 15
LIQUIDITY_SWEEP_DELTA_SIGMA_MIN = 1

# Setup 3: Range-Fade (SHORT)
RANGE_FADE_REJECTION_LOW = 4482.00
RANGE_FADE_REJECTION_HIGH = 4500.00
RANGE_FADE_ENTRY_PRICE = 4492.00
RANGE_FADE_ENTRY_TOLERANCE_BPS = 5
RANGE_FADE_SL_BUFFER_BPS = 25
RANGE_FADE_VOLUME_FACTOR_MAX = 0.80
RANGE_FADE_VOLUME_FACTOR_MIN = 1.20  # Do not take if RVOL_5m >= 1.2x

# Setup 4: Breakdown-Retest (SHORT)
BREAKDOWN_RETEST_TRIGGER = 4263.16  # L - 10 bps = 4267.43 - 4.27
BREAKDOWN_RETEST_ENTRY_LOW = 4263.00
BREAKDOWN_RETEST_ENTRY_HIGH = 4268.00
BREAKDOWN_RETEST_SL_BUFFER_BPS = 35
BREAKDOWN_RETEST_VOLUME_FACTOR = 1.50
BREAKDOWN_RETEST_CONSECUTIVE_CLOSES = 2

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# Risk management
MAX_CONCURRENT_TRADES = 1
PER_TRADE_RISK_PCT = 0.5
SKIP_IF_SPREAD_BPS_GT = 3

# State files for strategy tracking
BREAKOUT_RETEST_TRIGGER_FILE = "eth_breakout_retest_trigger_state.json"
LIQUIDITY_SWEEP_TRIGGER_FILE = "eth_liquidity_sweep_trigger_state.json"
RANGE_FADE_TRIGGER_FILE = "eth_range_fade_trigger_state.json"
BREAKDOWN_RETEST_TRIGGER_FILE = "eth_breakdown_retest_trigger_state.json"

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
    
    # Test ETH Breakout-Retest trade data
    eth_breakout_retest_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-Breakout-Retest',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4490.0,
        'stop_loss': 4455.0,
        'take_profit': 4535.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 800.0,
        'volume_ratio': 1.3,
        'current_price': 4490.0,
        'market_conditions': 'HOD=$4,482.49, LOD=$4,267.43',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Breakout-Retest (Aug 30 Setup)'
    }
    
    # Test ETH Liquidity-Sweep Reclaim trade data
    eth_liquidity_sweep_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-Liquidity-Sweep-Reclaim',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4270.0,
        'stop_loss': 4255.0,
        'take_profit': 4482.5,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 750.0,
        'volume_ratio': 1.1,
        'current_price': 4270.0,
        'market_conditions': 'HOD=$4,482.49, LOD=$4,267.43',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Liquidity-Sweep Reclaim (Aug 30 Setup)'
    }
    
    # Log test trades
    log_trade_to_csv(eth_breakout_retest_data)
    log_trade_to_csv(eth_liquidity_sweep_data)
    
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

# --- ETH Trading Strategy Alert Logic (New August 30th Setup) ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH Intraday Trading Strategy Alert - August 30th
    
    Spiros, here are 4 executable ETH setups for today (Sat, Aug 30, 2025). 
    Spot ref: $4,382.51. Intraday H/L: $4,482.49 / $4,267.43.

    Setup	Bias	Trigger (numbers today)	Entry	Invalidation / SL	Targets	Confirmation	Filters
    Breakout-Retest	Long	5-min close > 4,486.97 (H +10 bps)	Bid on retest of 4,486.97–4,492	SL = retest low − 35 bps	TP1 = 1R, TP2 = 2R, runner = 3R	≥2 closes above trigger	RVOL_5m ≥ 1.25× 20-SMA vol
    Liquidity-Sweep Reclaim	Long	Wick below 4,267.43 then 5-min close back above low	Market on close reclaim or limit at low+5–10 bps	SL = sweep low − 15 bps	TP1 = VWAP, TP2 = Mid-range 4,374.96, TP3 = 2R	1 reclaim close	CVD absorption or delta ≥ +1σ; RVOL_1m spike
    Range-Fade	Short	Rejection at 4,482–4,500 with bearish wick	Limit at 4,492 ± 5	SL = wick high + 25 bps	TP1 = Mid 4,374.96, TP2 = 2R	1 rejection candle	RVOL_5m ≤ 0.80×; do not take if RVOL_5m ≥ 1.2×
    Breakdown-Retest	Short	5-min close < 4,263.16 (L −10 bps)	Offer on retest of 4,263–4,268	SL = retest high + 35 bps	TP1 = 1R, TP2 = 2R, runner = 3R	≥2 closes below trigger	RVOL_5m ≥ 1.50× 20-SMA vol

    Notes for your bot:
    •	RVOL_5m uses volume / 20-bar SMA volume on 5-min.
    •	"R" = entry→SL distance.
    •	Skip signals during news spikes or abnormal spread.
    •	Update H/L if new extremes print; recompute trigger = H±10 bps, mid = (H+L)/2.
    •	Position size: margin × leverage = 250 × 20 = 5000 USD (fixed).
    
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
    breakout_retest_state = load_trigger_state(BREAKOUT_RETEST_TRIGGER_FILE)
    liquidity_sweep_state = load_trigger_state(LIQUIDITY_SWEEP_TRIGGER_FILE)
    range_fade_state = load_trigger_state(RANGE_FADE_TRIGGER_FILE)
    breakdown_retest_state = load_trigger_state(BREAKDOWN_RETEST_TRIGGER_FILE)
    
    try:
        now = datetime.now(UTC)
        
        # Get 15-minute candles for analysis (primary timeframe)
        end = now
        start = now - timedelta(hours=6)  # Get enough data for 20-period volume average
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        logger.info(f"Fetching 15-minute candles for {6} hours...")
        candles_15m = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY_5M)
        
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
        breakout_retest_volume_confirmed = relative_volume_5m >= BREAKOUT_RETEST_VOLUME_FACTOR
        liquidity_sweep_volume_confirmed = relative_volume_5m >= LIQUIDITY_SWEEP_DELTA_SIGMA_MIN
        range_fade_volume_confirmed = relative_volume_5m <= RANGE_FADE_VOLUME_FACTOR_MAX
        breakdown_retest_volume_confirmed = relative_volume_5m >= BREAKDOWN_RETEST_VOLUME_FACTOR
        
        # Filter strategies based on direction parameter
        breakout_retest_enabled = direction in ['LONG', 'BOTH']
        liquidity_sweep_enabled = direction in ['LONG', 'BOTH']
        range_fade_enabled = direction in ['SHORT', 'BOTH']
        breakdown_retest_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, here are 4 executable ETH setups for today (Sat, Aug 30, 2025).")
        logger.info(f"📊 Live Levels:")
        logger.info(f"   • ETH ≈ ${current_close_5m:,.2f}")
        logger.info(f"   • HOD ≈ ${HOD_LEVEL:,.2f}, LOD ≈ ${LOD_LEVEL:,.2f}")
        logger.info(f"   • Mid-range: ${MID_LEVEL:,.2f}")
        logger.info("")
        logger.info("📊 Strategy Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Primary timeframe: 5-minute candles")
        logger.info(f"   • Volume requirements: RVOL = vol(5m)/SMA20(5m)")
        logger.info(f"   • Risk: {PER_TRADE_RISK_PCT}% per trade, max {MAX_CONCURRENT_TRADES} concurrent trade")
        logger.info("")
        
        # Show strategies based on direction
        if breakout_retest_enabled:
            logger.info("📊 Setup 1: Breakout-Retest (LONG):")
            logger.info(f"   • Trigger: 5m close > ${BREAKOUT_RETEST_TRIGGER:,.2f} (H + {BUFFER_BPS}bps)")
            logger.info(f"   • Entry: Retest of ${BREAKOUT_RETEST_ENTRY_LOW:,.2f}–${BREAKOUT_RETEST_ENTRY_HIGH:,.2f}")
            logger.info(f"   • Stop: Retest low − {BREAKOUT_RETEST_SL_BUFFER_BPS}bps")
            logger.info(f"   • Targets: TP1 = 1R, TP2 = 2R, runner = 3R")
            logger.info(f"   • Volume: RVOL ≥ {BREAKOUT_RETEST_VOLUME_FACTOR}x 20-SMA vol")
            logger.info("")
        
        if liquidity_sweep_enabled:
            logger.info("📊 Setup 2: Liquidity-Sweep Reclaim (LONG):")
            logger.info(f"   • Trigger: Wick below ${LIQUIDITY_SWEEP_LEVEL:,.2f} then 5m close back above low")
            logger.info(f"   • Entry: Market on close reclaim or limit at low+{LIQUIDITY_SWEEP_ENTRY_OFFSET_BPS}bps")
            logger.info(f"   • Stop: Sweep low − {LIQUIDITY_SWEEP_SL_BUFFER_BPS}bps")
            logger.info(f"   • Targets: TP1 = VWAP, TP2 = Mid-range ${MID_LEVEL:,.2f}, TP3 = 2R")
            logger.info(f"   • Volume: RVOL spike with delta ≥ +{LIQUIDITY_SWEEP_DELTA_SIGMA_MIN}σ")
            logger.info("")
        
        if range_fade_enabled:
            logger.info("📊 Setup 3: Range-Fade (SHORT):")
            logger.info(f"   • Trigger: Rejection at ${RANGE_FADE_REJECTION_LOW:,.2f}–${RANGE_FADE_REJECTION_HIGH:,.2f} with bearish wick")
            logger.info(f"   • Entry: Limit at ${RANGE_FADE_ENTRY_PRICE:,.2f} ± {RANGE_FADE_ENTRY_TOLERANCE_BPS}bps")
            logger.info(f"   • Stop: Wick high + {RANGE_FADE_SL_BUFFER_BPS}bps")
            logger.info(f"   • Targets: TP1 = Mid ${MID_LEVEL:,.2f}, TP2 = 2R")
            logger.info(f"   • Volume: RVOL ≤ {RANGE_FADE_VOLUME_FACTOR_MAX}x (do not take if ≥ {RANGE_FADE_VOLUME_FACTOR_MIN}x)")
            logger.info("")
        
        if breakdown_retest_enabled:
            logger.info("📊 Setup 4: Breakdown-Retest (SHORT):")
            logger.info(f"   • Trigger: 5m close < ${BREAKDOWN_RETEST_TRIGGER:,.2f} (L − {BUFFER_BPS}bps)")
            logger.info(f"   • Entry: Retest of ${BREAKDOWN_RETEST_ENTRY_LOW:,.2f}–${BREAKDOWN_RETEST_ENTRY_HIGH:,.2f}")
            logger.info(f"   • Stop: Retest high + {BREAKDOWN_RETEST_SL_BUFFER_BPS}bps")
            logger.info(f"   • Targets: TP1 = 1R, TP2 = 2R, runner = 3R")
            logger.info(f"   • Volume: RVOL ≥ {BREAKDOWN_RETEST_VOLUME_FACTOR}x 20-SMA vol")
            logger.info("")
        
        logger.info(f"Current Price: ${current_close_15m:,.2f}")
        logger.info(f"Last 15M Close: ${last_close_15m:,.2f}, High: ${last_high_15m:,.2f}, Low: ${last_low_15m:,.2f}")
        logger.info(f"15M Volume: {last_volume_15m:,.0f}, 15M SMA(20): {avg_volume_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}x")
        logger.info(f"Last 5M Close: ${last_close_5m:,.2f}, 5M Volume: {last_volume_5m:,.0f}, 5M SMA(20): {avg_volume_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}x")
        logger.info(f"Setup 1 Volume confirmed (≥{BREAKOUT_RETEST_VOLUME_FACTOR}x): {'✅' if breakout_retest_volume_confirmed else '❌'}")
        logger.info(f"Setup 2 Volume confirmed (≥{LIQUIDITY_SWEEP_DELTA_SIGMA_MIN}x): {'✅' if liquidity_sweep_volume_confirmed else '❌'}")
        logger.info(f"Setup 3 Volume confirmed (≤{RANGE_FADE_VOLUME_FACTOR_MAX}x): {'✅' if range_fade_volume_confirmed else '❌'}")
        logger.info(f"Setup 4 Volume confirmed (≥{BREAKDOWN_RETEST_VOLUME_FACTOR}x): {'✅' if breakdown_retest_volume_confirmed else '❌'}")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # Calculate VWAP (simplified - using session average)
        session_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        session_candles = [c for c in candles_5m if datetime.fromtimestamp(int(c['start']), UTC) >= session_start]
        if session_candles:
            vwap = sum(float(c['close']) * float(c['volume']) for c in session_candles) / sum(float(c['volume']) for c in session_candles)
        else:
            vwap = current_close_5m  # Fallback to current price
        
        # 1. Setup 1: Breakout-Retest Strategy
        if (breakout_retest_enabled and 
            not breakout_retest_state.get("triggered", False) and not trade_executed):
            
            # Check if 5m close above breakout trigger level (clean break of HOD)
            breakout_trigger_condition = last_close_5m > BREAKOUT_RETEST_ENTRY_HIGH
            # Volume confirmation
            breakout_volume_condition = relative_volume_5m >= BREAKOUT_RETEST_VOLUME_FACTOR
            
            breakout_ready = breakout_trigger_condition and breakout_volume_condition

            logger.info("🔍 Setup 1 - Breakout-Retest Analysis:")
            logger.info(f"   • 5m close > ${BREAKOUT_RETEST_ENTRY_HIGH:,} (H + {BUFFER_BPS}bps): {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{BREAKOUT_RETEST_VOLUME_FACTOR}x): {'✅' if breakout_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 Setup 1 - Breakout-Retest conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = min(last_low_5m, entry_price * (1 - BREAKOUT_RETEST_SL_BUFFER_BPS / 10000))
                tp1 = entry_price * (1 + PER_TRADE_RISK_PCT / 100)
                tp2 = entry_price * (1 + PER_TRADE_RISK_PCT * 2 / 100)

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakout-Retest (Aug 30 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakout-Retest trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: 5m close > 4,635 (clean break of HOD) and RVOL ≥ 1.5")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Breakout-Retest',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"HOD=${HOD_LEVEL:,.1f}, LOD=${LOD_LEVEL:,.2f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 5m close > ${BREAKOUT_RETEST_ENTRY_HIGH:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    breakout_retest_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price,
                        "consecutive_closes": breakout_retest_state.get('consecutive_closes', 0) + 1
                    }
                    save_trigger_state(breakout_retest_state, BREAKOUT_RETEST_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout-Retest trade failed: {trade_result}")
        
        # 2. Setup 2: Liquidity-Sweep Reclaim Strategy
        if (liquidity_sweep_enabled and 
            not liquidity_sweep_state.get("triggered", False) and not trade_executed):
            
            # Check if wick below 4,267.43 then 5-min close back above low
            sweep_condition = last_low_5m < LIQUIDITY_SWEEP_LEVEL  # Wick below the level
            reclaim_condition = last_close_5m > LIQUIDITY_SWEEP_LEVEL  # Close back above the level
            liquidity_sweep_volume_condition = relative_volume_5m >= LIQUIDITY_SWEEP_DELTA_SIGMA_MIN
            
            liquidity_sweep_ready = sweep_condition and reclaim_condition and liquidity_sweep_volume_condition

            logger.info("🔍 Setup 2 - Liquidity-Sweep Reclaim Analysis:")
            logger.info(f"   • Wick below ${LIQUIDITY_SWEEP_LEVEL:,.2f}: {'✅' if sweep_condition else '❌'} (last low: {last_low_5m:,.2f})")
            logger.info(f"   • Close back above ${LIQUIDITY_SWEEP_LEVEL:,.2f}: {'✅' if reclaim_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LIQUIDITY_SWEEP_DELTA_SIGMA_MIN}x): {'✅' if liquidity_sweep_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Liquidity Sweep Ready: {'🎯 YES' if liquidity_sweep_ready else '⏳ NO'}")

            if liquidity_sweep_ready:
                logger.info("")
                logger.info("🎯 Setup 2 - Liquidity-Sweep Reclaim conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = LIQUIDITY_SWEEP_LEVEL * (1 - LIQUIDITY_SWEEP_SL_BUFFER_BPS / 10000)
                tp1 = vwap  # VWAP target
                tp2 = MID_LEVEL  # Mid-range target

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Liquidity-Sweep Reclaim (Aug 30 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("�� Liquidity-Sweep Reclaim trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: Wick below 4,267.43 then 5-min close back above low")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Liquidity-Sweep-Reclaim',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"HOD=${HOD_LEVEL:,.2f}, LOD=${LOD_LEVEL:,.2f}, VWAP=${vwap:,.2f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Liquidity sweep reclaim, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    liquidity_sweep_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(liquidity_sweep_state, LIQUIDITY_SWEEP_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Liquidity-Sweep Reclaim trade failed: {trade_result}")
        
        # 3. Setup 3: Range-Fade Strategy
        if (range_fade_enabled and 
            not range_fade_state.get("triggered", False) and not trade_executed):
            
            # Check if rejection at 4,482–4,500 with bearish wick
            rejection_zone_condition = RANGE_FADE_REJECTION_LOW <= last_high_5m <= RANGE_FADE_REJECTION_HIGH
            bearish_wick_condition = (last_high_5m - last_close_5m) > (last_close_5m - last_low_5m)  # Upper wick longer than lower wick
            range_fade_volume_condition = relative_volume_5m <= RANGE_FADE_VOLUME_FACTOR_MAX
            range_fade_volume_not_high = relative_volume_5m < RANGE_FADE_VOLUME_FACTOR_MIN  # Do not take if RVOL_5m >= 1.2x
            
            range_fade_ready = rejection_zone_condition and bearish_wick_condition and range_fade_volume_condition and range_fade_volume_not_high

            logger.info("🔍 Setup 3 - Range-Fade Analysis:")
            logger.info(f"   • Rejection at ${RANGE_FADE_REJECTION_LOW:,}–${RANGE_FADE_REJECTION_HIGH:,}: {'✅' if rejection_zone_condition else '❌'} (last high: {last_high_5m:,.2f})")
            logger.info(f"   • Bearish wick: {'✅' if bearish_wick_condition else '❌'} (upper wick longer than lower)")
            logger.info(f"   • Volume exhausted (≤{RANGE_FADE_VOLUME_FACTOR_MAX}x): {'✅' if range_fade_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Volume not high (<{RANGE_FADE_VOLUME_FACTOR_MIN}x): {'✅' if range_fade_volume_not_high else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Range-fade Ready: {'🎯 YES' if range_fade_ready else '⏳ NO'}")

            if range_fade_ready:
                logger.info("")
                logger.info("🎯 Setup 3 - Range-Fade conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = last_high_5m * (1 + RANGE_FADE_SL_BUFFER_BPS / 10000)  # Above wick high + 25 bps
                tp1 = MID_LEVEL  # Mid-range target
                tp2 = entry_price * (1 - PER_TRADE_RISK_PCT * 2 / 100)  # 2R target

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Range-Fade (Aug 30 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Range-Fade trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: Rejection at 4,482–4,500 with bearish wick")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Range-Fade',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"HOD=${HOD_LEVEL:,.1f}, LOD=${LOD_LEVEL:,.2f}, VWAP=${vwap:,.2f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Range fade rejection, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    range_fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(range_fade_state, RANGE_FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range-Fade trade failed: {trade_result}")
        
        # 4. Setup 4: Breakdown-Retest Strategy
        if (breakdown_retest_enabled and 
            not breakdown_retest_state.get("triggered", False) and not trade_executed):
            
            # Check if 5m close below breakdown trigger level (LOD breach)
            breakdown_trigger_condition = last_close_5m < BREAKDOWN_RETEST_ENTRY_HIGH
            breakdown_volume_condition = relative_volume_5m >= BREAKDOWN_RETEST_VOLUME_FACTOR
            
            breakdown_ready = breakdown_trigger_condition and breakdown_volume_condition

            logger.info("🔍 Setup 4 - Breakdown-Retest Analysis:")
            logger.info(f"   • 5m close < ${BREAKDOWN_RETEST_ENTRY_HIGH:,} (L - {BUFFER_BPS}bps): {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{BREAKDOWN_RETEST_VOLUME_FACTOR}x): {'✅' if breakdown_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 Setup 4 - Breakdown-Retest conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = entry_price * (1 + BREAKDOWN_RETEST_SL_BUFFER_BPS / 10000)
                tp1 = entry_price * (1 - PER_TRADE_RISK_PCT / 100)
                tp2 = entry_price * (1 - PER_TRADE_RISK_PCT * 2 / 100)

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakdown-Retest (Aug 30 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown-Retest trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: 5m close < 4,440 (LOD breach) and RVOL ≥ 1.5")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Breakdown-Retest',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': tp1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"HOD=${HOD_LEVEL:,.1f}, LOD=${LOD_LEVEL:,.2f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 5m close < ${BREAKDOWN_RETEST_ENTRY_HIGH:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    breakdown_retest_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price,
                        "consecutive_closes": breakdown_retest_state.get('consecutive_closes', 0) + 1
                    }
                    save_trigger_state(breakdown_retest_state, BREAKDOWN_RETEST_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown-Retest trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Setup 1 triggered: {breakout_retest_state.get('triggered', False)}")
            logger.info(f"Setup 2 triggered: {liquidity_sweep_state.get('triggered', False)}")
            logger.info(f"Setup 3 triggered: {range_fade_state.get('triggered', False)}")
            logger.info(f"Setup 4 triggered: {breakdown_retest_state.get('triggered', False)}")
        
        logger.info("=== ETH Aug 30 Trading Strategy Alert completed ===")
        return current_ts_5m
        
    except Exception as e:
        logger.error(f"Error in ETH Aug 30 Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Aug 30 Trading Strategy Alert completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Aug 30 Intraday Trading Strategy Monitor with optional direction filter')
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
    
    logger.info("Starting ETH Aug 30 Intraday Trading Strategy Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: ETH Aug 30 Intraday - 4 Executable Setups")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("Setup 1 - Breakout-Retest (LONG): 5m close > 4,486.97 (H +10 bps) with RVOL ≥ 1.25x")
    logger.info("Setup 2 - Liquidity-Sweep Reclaim (LONG): Wick below 4,267.43 then 5m close back above low")
    logger.info("Setup 3 - Range-Fade (SHORT): Rejection at 4,482–4,500 with RVOL ≤ 0.80x")
    logger.info("Setup 4 - Breakdown-Retest (SHORT): 5m close < 4,263.16 (L -10 bps) with RVOL ≥ 1.50x")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("Volume: RVOL = vol(5m)/SMA20(5m)")
    logger.info("Risk: 0.5% per trade, max 1 concurrent trade")
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
