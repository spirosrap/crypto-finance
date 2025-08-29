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

# ETH Trading Strategy Parameters (New August 29th Setup)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_15M = "FIFTEEN_MINUTE"  # Primary timeframe for triggers
GRANULARITY_5M = "FIVE_MINUTE"      # Secondary timeframe
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context - Updated with new levels from Aug 29 setup
CURRENT_ETH_PRICE = 4475.00  # ETH Spot ~$4,475
TODAY_HIGH = 4626.50         # HOD = $4,626.5
TODAY_LOW = 4438.98          # LOD = $4,438.98
CURRENT_PRICE = 4475.00      # Last ≈ $4,475
RANGE_HIGH = 4629            # 24h range high ~4,629
RANGE_LOW = 4430             # 24h range low ~4,430

# Key levels from Aug 29 setup
HOD_LEVEL = 4626.50          # High of Day
LOD_LEVEL = 4438.98          # Low of Day
BREAKOUT_LEVEL = 4635        # Clean break of HOD
FAILED_BREAKOUT_HIGH = 4640  # Wick through 4,635–4,640
FAILED_BREAKOUT_CLOSE = 4620 # 5m close < 4,620
BREAKDOWN_LEVEL = 4440       # LOD breach
RANGE_FADE_LOW = 4445        # Sweep 4,445–4,455
RANGE_FADE_HIGH = 4455
RANGE_FADE_RECLAIM = 4460    # Quick reclaim > 4,460

# LONG SETUPS
# Long - Breakout
LONG_BREAKOUT_TRIGGER = 4635.0      # Trigger: 5m close > 4,635 (clean break of HOD)
LONG_BREAKOUT_VOLUME_FACTOR = 1.5   # RVOL ≥ 1.5
LONG_BREAKOUT_STOP_PERCENT = 0.6    # Low of trigger bar or 0.6%
LONG_BREAKOUT_TP1_PERCENT = 1.2     # TP1 +1.2%
LONG_BREAKOUT_TP2_PERCENT = 2.4     # TP2 +2.4%

# Long - VWAP Reclaim
LONG_VWAP_VOLUME_FACTOR = 1.25      # RVOL ≥ 1.25
LONG_VWAP_RETEST_PERCENT = 0.05     # Retest of VWAP ±0.05%
LONG_VWAP_STOP_PERCENT = 0.5        # 0.5% below VWAP
LONG_VWAP_TP1_PERCENT = 0.0         # TP1 HOD
LONG_VWAP_TP2_PERCENT = 0.8         # TP2 HOD +0.8%

# SHORT SETUPS
# Short - Failed Breakout
SHORT_FAILED_BREAKOUT_VOLUME_FACTOR = 2.0  # RVOL spike ≥ 2.0
SHORT_FAILED_BREAKOUT_STOP = 4645          # Stop at 4,645
SHORT_FAILED_BREAKOUT_TP1_PERCENT = -1.0   # TP1 VWAP (will be calculated)
SHORT_FAILED_BREAKOUT_TP2 = 4540           # TP2 4,540 mid-range

# Short - Breakdown
SHORT_BREAKDOWN_TRIGGER = 4440.0           # Trigger: 5m close < 4,440 (LOD breach)
SHORT_BREAKDOWN_VOLUME_FACTOR = 1.5        # RVOL ≥ 1.5
SHORT_BREAKDOWN_STOP_PERCENT = 0.6         # 0.6% above entry
SHORT_BREAKDOWN_TP1_PERCENT = -1.0         # TP1 -1.0%
SHORT_BREAKDOWN_TP2_PERCENT = -2.0         # TP2 -2.0%

# Long - Range-fade
LONG_RANGE_FADE_VOLUME_FACTOR = 0.9        # RVOL ≤ 0.9 (exhausted sellers)
LONG_RANGE_FADE_STOP = 4440                # Stop at 4,440
LONG_RANGE_FADE_TP1 = 4500                 # TP1 4,500
LONG_RANGE_FADE_TP2_PERCENT = 0.0          # TP2 VWAP / 4,530 (will be calculated)

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_BREAKOUT_TRIGGER_FILE = "eth_breakout_trigger_state.json"
LONG_VWAP_TRIGGER_FILE = "eth_vwap_trigger_state.json"
SHORT_FAILED_BREAKOUT_TRIGGER_FILE = "eth_failed_breakout_trigger_state.json"
SHORT_BREAKDOWN_TRIGGER_FILE = "eth_breakdown_trigger_state.json"
LONG_RANGE_FADE_TRIGGER_FILE = "eth_range_fade_trigger_state.json"

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
        'entry_price': 4635.0,
        'stop_loss': 4607.0,
        'take_profit': 4691.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 800.0,
        'volume_ratio': 1.8,
        'current_price': 4635.0,
        'market_conditions': 'HOD=$4,626.5, LOD=$4,438.98',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Breakout Long (Aug 29 Setup)'
    }
    
    # Test ETH VWAP Reclaim trade data
    eth_vwap_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-VWAP-Reclaim',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4500.0,
        'stop_loss': 4477.5,
        'take_profit': 4626.5,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 750.0,
        'volume_ratio': 1.4,
        'current_price': 4500.0,
        'market_conditions': 'HOD=$4,626.5, LOD=$4,438.98',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH VWAP Reclaim (Aug 29 Setup)'
    }
    
    # Log test trades
    log_trade_to_csv(eth_breakout_data)
    log_trade_to_csv(eth_vwap_data)
    
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

# --- ETH Trading Strategy Alert Logic (New August 29th Setup) ---
def eth_trading_strategy_alert(cb_service, last_alert_ts=None, direction='BOTH'):
    """
    ETH Intraday Trading Strategy Alert - August 29th
    
    Spiros, here are ETH intraday plays for Fri, Aug 29 (America/Chicago). 
    Levels reference live HOD/LOD; code straight from triggers.

    Key levels now: HOD ≈ 4,626.5, LOD ≈ 4,438.98, last ≈ 4,475. 24h range ~4,430–4,629.

    Funding backdrop: mixed, near flat to slightly positive across majors. Treat as neutral for bias.

    Setups (5-min chart, RVOL vs 20-SMA volume):
    
    Setup	Entry trigger	Invalid/Stop	Targets	Confirmers
    Breakout LONG	5m close > 4,635 (clean break of HOD) and RVOL ≥ 1.5	Low of trigger bar or 0.6%	TP1 +1.2%, TP2 +2.4% then trail to 5m EMA20	Tick-to-trade improves, spread ≤ 0.5$, no immediate fade
    VWAP Reclaim LONG	Lose VWAP, then 5m close above VWAP with HL structure and RVOL ≥ 1.25; enter on retest of VWAP ±0.05%	0.5% below VWAP	TP1 HOD, TP2 HOD +0.8%	Delta turns positive, tape not stalling at VWAP
    Failed Breakout SHORT	Wick through 4,635–4,640 then 5m close < 4,620 with RVOL spike ≥ 2.0	4,645	TP1 VWAP, TP2 4,540 mid-range	Rejection on stacked offers, no fresh high prints
    Breakdown SHORT	5m close < 4,440 (LOD breach) and RVOL ≥ 1.5	0.6% above entry	TP1 -1.0%, TP2 -2.0%	Expansion in range, no instant reclaim above 4,440
    Range-fade LONG	Sweep 4,445–4,455, quick reclaim > 4,460 with RVOL ≤ 0.9 (exhausted sellers)	4,440	TP1 4,500, TP2 VWAP / 4,530	Long-wick lows, decreasing sell MBO, funding not spiking

    Notes for your bot:
    •	Compute RVOL = vol(5m)/SMA20(5m).
    •	VWAP = session vwap from 00:00 UTC-5 open or your session start.
    •	Use exchange's ETH-PERP; map USD levels 1:1.
    •	Position sizing for ~2R profile; skip if spread widens or RVOL criteria not met.
    •	Avoid longs if funding jumps broadly positive at trigger; avoid shorts if it flips negative.
    
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
    long_vwap_state = load_trigger_state(LONG_VWAP_TRIGGER_FILE)
    short_failed_breakout_state = load_trigger_state(SHORT_FAILED_BREAKOUT_TRIGGER_FILE)
    short_breakdown_state = load_trigger_state(SHORT_BREAKDOWN_TRIGGER_FILE)
    long_range_fade_state = load_trigger_state(LONG_RANGE_FADE_TRIGGER_FILE)
    
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
        long_volume_confirmed = relative_volume_5m >= LONG_BREAKOUT_VOLUME_FACTOR
        short_volume_confirmed = relative_volume_5m >= SHORT_BREAKDOWN_VOLUME_FACTOR
        
        # Filter strategies based on direction parameter
        long_strategies_enabled = direction in ['LONG', 'BOTH']
        short_strategies_enabled = direction in ['SHORT', 'BOTH']
        
        # --- Reporting ---
        logger.info("")
        logger.info("🚀 Spiros, here are ETH intraday plays for Fri, Aug 29 (America/Chicago).")
        logger.info(f"📊 Live Levels:")
        logger.info(f"   • ETH ≈ ${current_close_5m:,.2f}")
        logger.info(f"   • HOD ≈ ${HOD_LEVEL:,.1f}, LOD ≈ ${LOD_LEVEL:,.2f}")
        logger.info(f"   • 24h range: ${RANGE_LOW:,}–${RANGE_HIGH:,}")
        logger.info("")
        logger.info("📊 Strategy Rules:")
        logger.info(f"   • Position Size: ${POSITION_SIZE_USD:,.0f} USD (${MARGIN} × {LEVERAGE}x) — fixed")
        logger.info(f"   • Primary timeframe: 5-minute candles")
        logger.info(f"   • Volume requirements: RVOL = vol(5m)/SMA20(5m)")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setups:")
            logger.info(f"   • Breakout: 5m close > ${BREAKOUT_LEVEL:,} (clean break of HOD) and RVOL ≥ {LONG_BREAKOUT_VOLUME_FACTOR}")
            logger.info(f"     Stop: Low of trigger bar or {LONG_BREAKOUT_STOP_PERCENT}%")
            logger.info(f"     TP1: +{LONG_BREAKOUT_TP1_PERCENT}%, TP2: +{LONG_BREAKOUT_TP2_PERCENT}% then trail to 5m EMA20")
            logger.info(f"   • VWAP Reclaim: Lose VWAP, then 5m close above VWAP with HL structure and RVOL ≥ {LONG_VWAP_VOLUME_FACTOR}")
            logger.info(f"     Entry: Retest of VWAP ±{LONG_VWAP_RETEST_PERCENT}%, Stop: {LONG_VWAP_STOP_PERCENT}% below VWAP")
            logger.info(f"     TP1: HOD, TP2: HOD +{LONG_VWAP_TP2_PERCENT}%")
            logger.info(f"   • Range-fade: Sweep ${RANGE_FADE_LOW:,}–${RANGE_FADE_HIGH:,}, quick reclaim > ${RANGE_FADE_RECLAIM:,} with RVOL ≤ {LONG_RANGE_FADE_VOLUME_FACTOR}")
            logger.info(f"     Stop: ${LONG_RANGE_FADE_STOP:,}, TP1: ${LONG_RANGE_FADE_TP1:,}, TP2: VWAP / 4,530")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setups:")
            logger.info(f"   • Failed Breakout: Wick through ${BREAKOUT_LEVEL:,}–${FAILED_BREAKOUT_HIGH:,} then 5m close < ${FAILED_BREAKOUT_CLOSE:,} with RVOL spike ≥ {SHORT_FAILED_BREAKOUT_VOLUME_FACTOR}")
            logger.info(f"     Stop: ${SHORT_FAILED_BREAKOUT_STOP:,}, TP1: VWAP, TP2: ${SHORT_FAILED_BREAKOUT_TP2:,} mid-range")
            logger.info(f"   • Breakdown: 5m close < ${BREAKDOWN_LEVEL:,} (LOD breach) and RVOL ≥ {SHORT_BREAKDOWN_VOLUME_FACTOR}")
            logger.info(f"     Stop: {SHORT_BREAKDOWN_STOP_PERCENT}% above entry, TP1: {SHORT_BREAKDOWN_TP1_PERCENT}%, TP2: {SHORT_BREAKDOWN_TP2_PERCENT}%")
            logger.info("")
        
        logger.info(f"Current Price: ${current_close_15m:,.2f}")
        logger.info(f"Last 15M Close: ${last_close_15m:,.2f}, High: ${last_high_15m:,.2f}, Low: ${last_low_15m:,.2f}")
        logger.info(f"15M Volume: {last_volume_15m:,.0f}, 15M SMA(20): {avg_volume_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}x")
        logger.info(f"Last 5M Close: ${last_close_5m:,.2f}, 5M Volume: {last_volume_5m:,.0f}, 5M SMA(20): {avg_volume_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}x")
        logger.info(f"Long Volume confirmed (≥{LONG_BREAKOUT_VOLUME_FACTOR}x): {'✅' if long_volume_confirmed else '❌'}")
        logger.info(f"Short Volume confirmed (≥{SHORT_BREAKDOWN_VOLUME_FACTOR}x): {'✅' if short_volume_confirmed else '❌'}")
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
        
        # 1. LONG - Breakout Strategy
        if (long_strategies_enabled and 
            not long_breakout_state.get("triggered", False) and not trade_executed):
            
            # Check if 5m close above breakout trigger level (clean break of HOD)
            breakout_trigger_condition = last_close_5m > BREAKOUT_LEVEL
            # Volume confirmation
            breakout_volume_condition = relative_volume_5m >= LONG_BREAKOUT_VOLUME_FACTOR
            
            breakout_ready = breakout_trigger_condition and breakout_volume_condition

            logger.info("🔍 LONG - Breakout Strategy Analysis:")
            logger.info(f"   • 5m close > ${BREAKOUT_LEVEL:,} (clean break of HOD): {'✅' if breakout_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LONG_BREAKOUT_VOLUME_FACTOR}x): {'✅' if breakout_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Breakout Ready: {'🎯 YES' if breakout_ready else '⏳ NO'}")

            if breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - Breakout Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = min(last_low_5m, entry_price * (1 - LONG_BREAKOUT_STOP_PERCENT / 100))
                tp1 = entry_price * (1 + LONG_BREAKOUT_TP1_PERCENT / 100)
                tp2 = entry_price * (1 + LONG_BREAKOUT_TP2_PERCENT / 100)

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakout Long (Aug 29 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakout Long trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: 5m close > 4,635 (clean break of HOD) and RVOL ≥ 1.5")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Breakout',
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
                        'notes': f"Trigger: 5m close > ${BREAKOUT_LEVEL:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(long_breakout_state, LONG_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakout Long trade failed: {trade_result}")
        
        # 2. LONG - VWAP Reclaim Strategy
        if (long_strategies_enabled and 
            not long_vwap_state.get("triggered", False) and not trade_executed):
            
            # Check if price lost VWAP and then reclaimed it
            vwap_lost = any(float(c['close']) < vwap for c in candles_5m[-5:])  # Check last 5 candles
            vwap_reclaimed = last_close_5m > vwap
            vwap_volume_condition = relative_volume_5m >= LONG_VWAP_VOLUME_FACTOR
            vwap_retest_condition = abs(current_close_5m - vwap) / vwap <= LONG_VWAP_RETEST_PERCENT / 100
            
            vwap_ready = vwap_lost and vwap_reclaimed and vwap_volume_condition and vwap_retest_condition

            logger.info("🔍 LONG - VWAP Reclaim Strategy Analysis:")
            logger.info(f"   • VWAP lost and reclaimed: {'✅' if (vwap_lost and vwap_reclaimed) else '❌'}")
            logger.info(f"   • Volume confirmed (≥{LONG_VWAP_VOLUME_FACTOR}x): {'✅' if vwap_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • VWAP retest (±{LONG_VWAP_RETEST_PERCENT}%): {'✅' if vwap_retest_condition else '❌'} (current: {current_close_5m:,.2f}, VWAP: {vwap:,.2f})")
            logger.info(f"   • VWAP Reclaim Ready: {'🎯 YES' if vwap_ready else '⏳ NO'}")

            if vwap_ready:
                logger.info("")
                logger.info("🎯 LONG - VWAP Reclaim Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = vwap * (1 - LONG_VWAP_STOP_PERCENT / 100)
                tp1 = HOD_LEVEL
                tp2 = HOD_LEVEL * (1 + LONG_VWAP_TP2_PERCENT / 100)

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH VWAP Reclaim (Aug 29 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 VWAP Reclaim trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: VWAP reclaim with HL structure and RVOL ≥ 1.25")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-VWAP-Reclaim',
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
                        'market_conditions': f"HOD=${HOD_LEVEL:,.1f}, LOD=${LOD_LEVEL:,.2f}, VWAP=${vwap:,.2f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: VWAP reclaim, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_vwap_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(long_vwap_state, LONG_VWAP_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ VWAP Reclaim trade failed: {trade_result}")
        
        # 3. SHORT - Failed Breakout Strategy
        if (short_strategies_enabled and 
            not short_failed_breakout_state.get("triggered", False) and not trade_executed):
            
            # Check if wick through 4,635–4,640 then 5m close < 4,620
            wick_through_condition = (BREAKOUT_LEVEL <= last_high_5m <= FAILED_BREAKOUT_HIGH)
            failed_close_condition = last_close_5m < FAILED_BREAKOUT_CLOSE
            failed_volume_condition = relative_volume_5m >= SHORT_FAILED_BREAKOUT_VOLUME_FACTOR
            
            failed_breakout_ready = wick_through_condition and failed_close_condition and failed_volume_condition

            logger.info("🔍 SHORT - Failed Breakout Strategy Analysis:")
            logger.info(f"   • Wick through ${BREAKOUT_LEVEL:,}–${FAILED_BREAKOUT_HIGH:,}: {'✅' if wick_through_condition else '❌'} (last high: {last_high_5m:,.2f})")
            logger.info(f"   • 5m close < ${FAILED_BREAKOUT_CLOSE:,}: {'✅' if failed_close_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Volume spike (≥{SHORT_FAILED_BREAKOUT_VOLUME_FACTOR}x): {'✅' if failed_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Failed Breakout Ready: {'🎯 YES' if failed_breakout_ready else '⏳ NO'}")

            if failed_breakout_ready:
                logger.info("")
                logger.info("🎯 SHORT - Failed Breakout Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = SHORT_FAILED_BREAKOUT_STOP
                tp1 = vwap  # VWAP target
                tp2 = SHORT_FAILED_BREAKOUT_TP2

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Failed Breakout (Aug 29 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Failed Breakout trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: Wick through 4,635–4,640 then 5m close < 4,620 with RVOL spike ≥ 2.0")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Failed-Breakout',
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
                        'notes': f"Trigger: Failed breakout, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_failed_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(short_failed_breakout_state, SHORT_FAILED_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Failed Breakout trade failed: {trade_result}")
        
        # 4. SHORT - Breakdown Strategy
        if (short_strategies_enabled and 
            not short_breakdown_state.get("triggered", False) and not trade_executed):
            
            # Check if 5m close below breakdown trigger level (LOD breach)
            breakdown_trigger_condition = last_close_5m < BREAKDOWN_LEVEL
            breakdown_volume_condition = relative_volume_5m >= SHORT_BREAKDOWN_VOLUME_FACTOR
            
            breakdown_ready = breakdown_trigger_condition and breakdown_volume_condition

            logger.info("🔍 SHORT - Breakdown Strategy Analysis:")
            logger.info(f"   • 5m close < ${BREAKDOWN_LEVEL:,} (LOD breach): {'✅' if breakdown_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_BREAKDOWN_VOLUME_FACTOR}x): {'✅' if breakdown_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Breakdown Ready: {'🎯 YES' if breakdown_ready else '⏳ NO'}")

            if breakdown_ready:
                logger.info("")
                logger.info("🎯 SHORT - Breakdown Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = entry_price * (1 + SHORT_BREAKDOWN_STOP_PERCENT / 100)
                tp1 = entry_price * (1 + SHORT_BREAKDOWN_TP1_PERCENT / 100)
                tp2 = entry_price * (1 + SHORT_BREAKDOWN_TP2_PERCENT / 100)

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Breakdown (Aug 29 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Breakdown trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: 5m close < 4,440 (LOD breach) and RVOL ≥ 1.5")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Breakdown',
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
                        'notes': f"Trigger: 5m close < ${BREAKDOWN_LEVEL:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_breakdown_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(short_breakdown_state, SHORT_BREAKDOWN_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Breakdown trade failed: {trade_result}")
        
        # 5. LONG - Range-fade Strategy
        if (long_strategies_enabled and 
            not long_range_fade_state.get("triggered", False) and not trade_executed):
            
            # Check if sweep 4,445–4,455, quick reclaim > 4,460
            sweep_condition = (RANGE_FADE_LOW <= last_low_5m <= RANGE_FADE_HIGH)
            reclaim_condition = current_close_5m > RANGE_FADE_RECLAIM
            range_fade_volume_condition = relative_volume_5m <= LONG_RANGE_FADE_VOLUME_FACTOR  # Note: ≤ for exhausted sellers
            
            range_fade_ready = sweep_condition and reclaim_condition and range_fade_volume_condition

            logger.info("🔍 LONG - Range-fade Strategy Analysis:")
            logger.info(f"   • Sweep ${RANGE_FADE_LOW:,}–${RANGE_FADE_HIGH:,}: {'✅' if sweep_condition else '❌'} (last low: {last_low_5m:,.2f})")
            logger.info(f"   • Quick reclaim > ${RANGE_FADE_RECLAIM:,}: {'✅' if reclaim_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume exhausted (≤{LONG_RANGE_FADE_VOLUME_FACTOR}x): {'✅' if range_fade_volume_condition else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Range-fade Ready: {'🎯 YES' if range_fade_ready else '⏳ NO'}")

            if range_fade_ready:
                logger.info("")
                logger.info("🎯 LONG - Range-fade Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                # Calculate entry, stop, and targets
                entry_price = current_close_5m
                stop_loss = LONG_RANGE_FADE_STOP
                tp1 = LONG_RANGE_FADE_TP1
                tp2 = vwap  # VWAP target

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Range-fade (Aug 29 Setup)",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=tp1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Range-fade trade executed successfully!")
                    logger.info(f"Entry: ${entry_price:,.2f}")
                    logger.info(f"Stop-loss: ${stop_loss:,.2f}")
                    logger.info(f"TP1: ${tp1:,.2f}, TP2: ${tp2:,.2f}")
                    logger.info("Strategy: Sweep 4,445–4,455, quick reclaim > 4,460 with RVOL ≤ 0.9")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Range-Fade',
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
                        'market_conditions': f"HOD=${HOD_LEVEL:,.1f}, LOD=${LOD_LEVEL:,.2f}, VWAP=${vwap:,.2f}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Range fade, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_range_fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": entry_price
                    }
                    save_trigger_state(long_range_fade_state, LONG_RANGE_FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range-fade trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"Breakout triggered: {long_breakout_state.get('triggered', False)}")
            logger.info(f"VWAP Reclaim triggered: {long_vwap_state.get('triggered', False)}")
            logger.info(f"Failed Breakout triggered: {short_failed_breakout_state.get('triggered', False)}")
            logger.info(f"Breakdown triggered: {short_breakdown_state.get('triggered', False)}")
            logger.info(f"Range-fade triggered: {long_range_fade_state.get('triggered', False)}")
        
        logger.info("=== ETH Aug 29 Trading Strategy Alert completed ===")
        return current_ts_5m
        
    except Exception as e:
        logger.error(f"Error in ETH Aug 29 Trading Strategy Alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== ETH Aug 29 Trading Strategy Alert completed (with error) ===")
    return last_alert_ts

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ETH Aug 29 Intraday Trading Strategy Monitor with optional direction filter')
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
    
    logger.info("Starting ETH Aug 29 Intraday Trading Strategy Monitor")
    if direction == 'BOTH':
        logger.info("Strategy: ETH Aug 29 Intraday - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("LONG - Breakout: 5m close > 4,635 (clean break of HOD) and RVOL ≥ 1.5")
    logger.info("LONG - VWAP Reclaim: Lose VWAP, then 5m close above VWAP with HL structure and RVOL ≥ 1.25")
    logger.info("LONG - Range-fade: Sweep 4,445–4,455, quick reclaim > 4,460 with RVOL ≤ 0.9")
    logger.info("SHORT - Failed Breakout: Wick through 4,635–4,640 then 5m close < 4,620 with RVOL spike ≥ 2.0")
    logger.info("SHORT - Breakdown: 5m close < 4,440 (LOD breach) and RVOL ≥ 1.5")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info("Volume: RVOL = vol(5m)/SMA20(5m)")
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
