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

# ETH Trading Strategy Parameters (New ATH Setup)
PRODUCT_ID = "ETH-PERP-INTX"
GRANULARITY_15M = "FIFTEEN_MINUTE"  # Primary timeframe for triggers
GRANULARITY_5M = "FIVE_MINUTE"      # Secondary timeframe
VOLUME_PERIOD = 20  # For volume confirmation

# Current market context - Updated with new ATH levels
CURRENT_ETH_PRICE = 4900.00  # ETH Spot ~$4,900
ATH_LEVEL = 4950.00          # ATH around $4,950
TODAY_HIGH = 4953.00         # 24h high = 4,953
TODAY_LOW = 4676.00          # 24h low = 4,676

# LONG SETUPS
# Long - ATH breakout-retest
LONG_ATH_BREAKOUT_TRIGGER = 4900    # Trigger: Clean 15m close > 4,900
LONG_ATH_BREAKOUT_ENTRY_LOW = 4905  # Entry zone: 4,905–4,935
LONG_ATH_BREAKOUT_ENTRY_HIGH = 4935
LONG_ATH_BREAKOUT_ENTRY = 4920      # Entry: 4,920 (mid of zone)
LONG_ATH_BREAKOUT_STOP_LOSS = 4880  # Invalidation: < 4,880
LONG_ATH_BREAKOUT_TP1 = 5050        # TP1: 5,050
LONG_ATH_BREAKOUT_TP2 = 5300        # TP2: 5,300

# Long - Prior-breakout dip buy
LONG_DIP_BUY_TRIGGER_LOW = 4700     # Trigger: Pullback to 4,700–4,750
LONG_DIP_BUY_TRIGGER_HIGH = 4750
LONG_DIP_BUY_ENTRY_LOW = 4710       # Entry zone: 4,710–4,745
LONG_DIP_BUY_ENTRY_HIGH = 4745
LONG_DIP_BUY_ENTRY = 4727.5         # Entry: 4,727.5 (mid of zone)
LONG_DIP_BUY_STOP_LOSS = 4660       # Invalidation: < 4,660
LONG_DIP_BUY_TP1 = 4850             # TP1: 4,850
LONG_DIP_BUY_TP2 = 4920             # TP2: 4,920

# SHORT SETUPS
# Short - ATH deviation fade
SHORT_ATH_FADE_TRIGGER_LOW = 4930   # Trigger: Wick above 4,930–5,000
SHORT_ATH_FADE_TRIGGER_HIGH = 5000
SHORT_ATH_FADE_ENTRY_LOW = 4890     # Entry zone: 4,890–4,910
SHORT_ATH_FADE_ENTRY_HIGH = 4910
SHORT_ATH_FADE_ENTRY = 4900         # Entry: 4,900 (mid of zone)
SHORT_ATH_FADE_STOP_LOSS = 5030     # Invalidation: > 5,030
SHORT_ATH_FADE_TP1 = 4820           # TP1: 4,820
SHORT_ATH_FADE_TP2 = 4750           # TP2: 4,750

# Short - Range break down
SHORT_RANGE_BREAK_TRIGGER_LOW = 4680  # Trigger: Loss of 4,680–4,700
SHORT_RANGE_BREAK_TRIGGER_HIGH = 4700
SHORT_RANGE_BREAK_ENTRY_LOW = 4690    # Entry zone: 4,690–4,705
SHORT_RANGE_BREAK_ENTRY_HIGH = 4705
SHORT_RANGE_BREAK_ENTRY = 4697.5      # Entry: 4,697.5 (mid of zone)
SHORT_RANGE_BREAK_STOP_LOSS = 4730    # Invalidation: > 4,730
SHORT_RANGE_BREAK_TP1 = 4620          # TP1: 4,620
SHORT_RANGE_BREAK_TP2 = 4540          # TP2: 4,540

# Volume confirmation requirements
LONG_VOLUME_FACTOR = 1.5   # RVOL ≥ 1.5× for long setups
SHORT_VOLUME_FACTOR = 1.5  # RVOL ≥ 1.5× for short setups
DIP_BUY_VOLUME_FACTOR = 1.25  # RVOL ≥ 1.25× for dip buy

# Trade parameters - Position size: margin x leverage = 250 x 20 = 5000 USD
MARGIN = 250  # USD
LEVERAGE = 20  # 20x leverage
POSITION_SIZE_USD = MARGIN * LEVERAGE  # 5000 USD

# State files for strategy tracking
LONG_ATH_BREAKOUT_TRIGGER_FILE = "eth_ath_breakout_trigger_state.json"
LONG_DIP_BUY_TRIGGER_FILE = "eth_dip_buy_trigger_state.json"
SHORT_ATH_FADE_TRIGGER_FILE = "eth_ath_fade_trigger_state.json"
SHORT_RANGE_BREAK_TRIGGER_FILE = "eth_range_break_trigger_state.json"

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
    
    # Test ETH ATH Breakout trade data
    eth_ath_breakout_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-ATH-Breakout',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4920.0,
        'stop_loss': 4880.0,
        'take_profit': 5050.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 800.0,
        'volume_ratio': 1.8,
        'current_price': 4920.0,
        'market_conditions': '24h Range: $4,676-$4,953',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH ATH Breakout'
    }
    
    # Test ETH Dip Buy trade data
    eth_dip_buy_data = {
        'timestamp': datetime.now(UTC).isoformat(),
        'strategy': 'TEST-ETH-Dip-Buy',
        'symbol': 'ETH-PERP-INTX',
        'side': 'BUY',
        'entry_price': 4727.5,
        'stop_loss': 4660.0,
        'take_profit': 4850.0,
        'position_size_usd': 5000.0,
        'margin': 250.0,
        'leverage': 20.0,
        'volume_sma': 750.0,
        'volume_ratio': 1.4,
        'current_price': 4727.5,
        'market_conditions': '24h Range: $4,676-$4,953',
        'trade_status': 'TEST',
        'execution_time': datetime.now(UTC).isoformat(),
        'notes': 'TEST TRADE - ETH Prior-Breakout Dip Buy'
    }
    
    # Log test trades
    log_trade_to_csv(eth_ath_breakout_data)
    log_trade_to_csv(eth_dip_buy_data)
    
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
    ETH ATH Trading Strategy Alert - New Setup
    
    Spiros: ETH is trading near fresh ATH ($4.88k–$4.95k). 24h range prints ≈$4,676–$4,953. Funding positive across majors (~0.01%/8h). Expect whips at highs.

    Setup	Bias	Trigger (confirm on 15m close)	Entry zone	Invalidation	TP1 / TP2	Timeframe	RVOL
    ATH breakout-retest	Long	Clean 15m close > 4,900, then shallow retest holds	4,905–4,935	< 4,880	5,050 / 5,300	15m–1h	≥1.5×
    Prior-breakout dip buy	Long	Pullback to 4,700–4,750 with higher-low + absorption	4,710–4,745	< 4,660	4,850 / 4,920	15m–1h	≥1.25×
    ATH deviation fade	Short	Wick above 4,930–5,000 then 15m close back < 4,900	4,890–4,910	> 5,030	4,820 / 4,750	5m–15m	≥1.5×
    Range break down	Short	Loss of 4,680–4,700, then LH under 4,700	4,690–4,705	> 4,730	4,620 / 4,540	15m–1h	≥1.5×

    Notes:
    •	Use perps context: funding clustered around +0.01% favors fading spikes if RVOL is high and structure confirms.
    •	ATH context is new; confirm with reputable feeds before executing. Coindesk/WSJ reported records near $4.87–$4.90k.
    
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
    long_ath_breakout_state = load_trigger_state(LONG_ATH_BREAKOUT_TRIGGER_FILE)
    long_dip_buy_state = load_trigger_state(LONG_DIP_BUY_TRIGGER_FILE)
    short_ath_fade_state = load_trigger_state(SHORT_ATH_FADE_TRIGGER_FILE)
    short_range_break_state = load_trigger_state(SHORT_RANGE_BREAK_TRIGGER_FILE)
    
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
        dip_buy_volume_confirmed = relative_volume_5m >= DIP_BUY_VOLUME_FACTOR
        
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
        logger.info(f"   • Volume requirements: 15m RVOL ≥ {LONG_VOLUME_FACTOR}x for long, {SHORT_VOLUME_FACTOR}x for short, 5m RVOL ≥ {DIP_BUY_VOLUME_FACTOR}x for dip buy")
        logger.info("")
        
        # Show strategies based on direction
        if long_strategies_enabled:
            logger.info("📊 LONG Setups:")
            logger.info(f"   • ATH Breakout: 15-min close above ${LONG_ATH_BREAKOUT_TRIGGER:,}, buy ${LONG_ATH_BREAKOUT_ENTRY_LOW:,}–${LONG_ATH_BREAKOUT_ENTRY_HIGH:,} on retest hold")
            logger.info(f"     Invalidation: < ${LONG_ATH_BREAKOUT_STOP_LOSS:,}")
            logger.info(f"     TP1: ${LONG_ATH_BREAKOUT_TP1:,} (~½ 24h range), TP2: ${LONG_ATH_BREAKOUT_TP2:,} (~full 24h range)")
            logger.info(f"   • Prior-Breakout Dip: Pullback to ${LONG_DIP_BUY_TRIGGER_LOW:,}–${LONG_DIP_BUY_TRIGGER_HIGH:,}, buy ${LONG_DIP_BUY_ENTRY_LOW:,}–${LONG_DIP_BUY_ENTRY_HIGH:,}")
            logger.info(f"     Invalidation: < ${LONG_DIP_BUY_STOP_LOSS:,}")
            logger.info(f"     TP1: ${LONG_DIP_BUY_TP1:,} (mid-range), TP2: ${LONG_DIP_BUY_TP2:,}")
            logger.info("")
        
        if short_strategies_enabled:
            logger.info("📊 SHORT Setups:")
            logger.info(f"   • ATH Deviation Fade: Wick above ${SHORT_ATH_FADE_TRIGGER_LOW:,}–${SHORT_ATH_FADE_TRIGGER_HIGH:,}, sell ${SHORT_ATH_FADE_ENTRY_LOW:,}–${SHORT_ATH_FADE_ENTRY_HIGH:,}")
            logger.info(f"     Invalidation: > ${SHORT_ATH_FADE_STOP_LOSS:,}")
            logger.info(f"     TP1: ${SHORT_ATH_FADE_TP1:,} (mid-range), TP2: ${SHORT_ATH_FADE_TP2:,}")
            logger.info(f"   • Range Break Down: Loss of ${SHORT_RANGE_BREAK_TRIGGER_LOW:,}–${SHORT_RANGE_BREAK_TRIGGER_HIGH:,}, sell ${SHORT_RANGE_BREAK_ENTRY_LOW:,}–${SHORT_RANGE_BREAK_ENTRY_HIGH:,}")
            logger.info(f"     Invalidation: > ${SHORT_RANGE_BREAK_STOP_LOSS:,}")
            logger.info(f"     TP1: ${SHORT_RANGE_BREAK_TP1:,} (mid-range), TP2: ${SHORT_RANGE_BREAK_TP2:,}")
            logger.info("")
        
        logger.info(f"Current Price: ${current_close_15m:,.2f}")
        logger.info(f"Last 15M Close: ${last_close_15m:,.2f}, High: ${last_high_15m:,.2f}, Low: ${last_low_15m:,.2f}")
        logger.info(f"15M Volume: {last_volume_15m:,.0f}, 15M SMA(20): {avg_volume_15m:,.0f}, Rel_Vol: {relative_volume_15m:.2f}x")
        logger.info(f"Last 5M Close: ${last_close_5m:,.2f}, 5M Volume: {last_volume_5m:,.0f}, 5M SMA(20): {avg_volume_5m:,.0f}, Rel_Vol: {relative_volume_5m:.2f}x")
        logger.info(f"Long Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if long_volume_confirmed else '❌'}")
        logger.info(f"Short Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if short_volume_confirmed else '❌'}")
        logger.info(f"Dip Buy Volume confirmed (≥{DIP_BUY_VOLUME_FACTOR}x): {'✅' if dip_buy_volume_confirmed else '❌'}")
        logger.info("")
        
        # --- Strategy Analysis ---
        trade_executed = False
        
        # 1. LONG - ATH Breakout Strategy
        if (long_strategies_enabled and 
            not long_ath_breakout_state.get("triggered", False) and not trade_executed):
            
            # Check if 15m close above trigger level
            ath_breakout_trigger_condition = last_close_15m > LONG_ATH_BREAKOUT_TRIGGER
            # Check if current price is in entry zone
            ath_breakout_entry_condition = (LONG_ATH_BREAKOUT_ENTRY_LOW <= current_close_15m <= LONG_ATH_BREAKOUT_ENTRY_HIGH)
            
            ath_breakout_ready = ath_breakout_trigger_condition and ath_breakout_entry_condition and long_volume_confirmed

            logger.info("🔍 LONG - ATH Breakout Strategy Analysis:")
            logger.info(f"   • 15m close above ${LONG_ATH_BREAKOUT_TRIGGER:,}: {'✅' if ath_breakout_trigger_condition else '❌'} (last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry in ${LONG_ATH_BREAKOUT_ENTRY_LOW:,}–${LONG_ATH_BREAKOUT_ENTRY_HIGH:,}: {'✅' if ath_breakout_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{LONG_VOLUME_FACTOR}x): {'✅' if long_volume_confirmed else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • ATH Breakout Long Ready: {'🎯 YES' if ath_breakout_ready else '⏳ NO'}")

            if ath_breakout_ready:
                logger.info("")
                logger.info("🎯 LONG - ATH Breakout Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH ATH Breakout Long (New Setup)",
                    entry_price=LONG_ATH_BREAKOUT_ENTRY,
                    stop_loss=LONG_ATH_BREAKOUT_STOP_LOSS,
                    take_profit=LONG_ATH_BREAKOUT_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 ATH Breakout Long trade executed successfully!")
                    logger.info(f"Entry: ${LONG_ATH_BREAKOUT_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_ATH_BREAKOUT_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_ATH_BREAKOUT_TP1:,.2f}, TP2: ${LONG_ATH_BREAKOUT_TP2:,.2f}")
                    logger.info("Strategy: 15-min close > 4,920 with volume confirmation")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-ATH-Breakout',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': LONG_ATH_BREAKOUT_ENTRY,
                        'stop_loss': LONG_ATH_BREAKOUT_STOP_LOSS,
                        'take_profit': LONG_ATH_BREAKOUT_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_15m,
                        'volume_ratio': relative_volume_15m,
                        'current_price': current_close_15m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: 15m close > ${LONG_ATH_BREAKOUT_TRIGGER:,}, Volume: {relative_volume_15m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_ath_breakout_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_15m['start']),
                        "entry_price": LONG_ATH_BREAKOUT_ENTRY
                    }
                    save_trigger_state(long_ath_breakout_state, LONG_ATH_BREAKOUT_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ ATH Breakout Long trade failed: {trade_result}")
        
        # 2. LONG - Prior-Breakout Dip Buy Strategy
        if (long_strategies_enabled and 
            not long_dip_buy_state.get("triggered", False) and not trade_executed):
            
            # Check if price pulls back below dip buy trigger
            dip_buy_trigger_condition = last_close_5m < LONG_DIP_BUY_TRIGGER_LOW
            # Check if current price is in entry zone (dip buy after pullback)
            dip_buy_entry_condition = (LONG_DIP_BUY_ENTRY_LOW <= current_close_5m <= LONG_DIP_BUY_ENTRY_HIGH)
            
            dip_buy_ready = dip_buy_trigger_condition and dip_buy_entry_condition and dip_buy_volume_confirmed

            logger.info("🔍 LONG - Prior-Breakout Dip Buy Strategy Analysis:")
            logger.info(f"   • Pullback to ${LONG_DIP_BUY_TRIGGER_LOW:,}–${LONG_DIP_BUY_TRIGGER_HIGH:,}: {'✅' if dip_buy_trigger_condition else '❌'} (last close: {last_close_5m:,.2f})")
            logger.info(f"   • Entry in ${LONG_DIP_BUY_ENTRY_LOW:,}–${LONG_DIP_BUY_ENTRY_HIGH:,}: {'✅' if dip_buy_entry_condition else '❌'} (current: {current_close_5m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{DIP_BUY_VOLUME_FACTOR}x): {'✅' if dip_buy_volume_confirmed else '❌'} (current: {relative_volume_5m:.2f}x)")
            logger.info(f"   • Dip Buy Long Ready: {'🎯 YES' if dip_buy_ready else '⏳ NO'}")

            if dip_buy_ready:
                logger.info("")
                logger.info("🎯 LONG - Prior-Breakout Dip Buy Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Prior-Breakout Dip Buy Long (New Setup)",
                    entry_price=LONG_DIP_BUY_ENTRY,
                    stop_loss=LONG_DIP_BUY_STOP_LOSS,
                    take_profit=LONG_DIP_BUY_TP1,
                    side="BUY",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Prior-Breakout Dip Buy Long trade executed successfully!")
                    logger.info(f"Entry: ${LONG_DIP_BUY_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${LONG_DIP_BUY_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${LONG_DIP_BUY_TP1:,.2f}, TP2: ${LONG_DIP_BUY_TP2:,.2f}")
                    logger.info("Strategy: Pullback to 4,727.5 with volume confirmation")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Dip-Buy',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'BUY',
                        'entry_price': LONG_DIP_BUY_ENTRY,
                        'stop_loss': LONG_DIP_BUY_STOP_LOSS,
                        'take_profit': LONG_DIP_BUY_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Pullback to ${LONG_DIP_BUY_TRIGGER_LOW:,}-${LONG_DIP_BUY_TRIGGER_HIGH:,} with volume confirmation"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    long_dip_buy_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": LONG_DIP_BUY_ENTRY
                    }
                    save_trigger_state(long_dip_buy_state, LONG_DIP_BUY_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Prior-Breakout Dip Buy trade failed: {trade_result}")
        
        # 3. SHORT - ATH Deviation Fade Strategy
        if (short_strategies_enabled and 
            not short_ath_fade_state.get("triggered", False) and not trade_executed):
            
            # Check if price spikes above fade trigger
            ath_fade_trigger_condition = (SHORT_ATH_FADE_TRIGGER_LOW <= last_high_15m <= SHORT_ATH_FADE_TRIGGER_HIGH)
            # Check if current price is at entry level (rejection after spike)
            ath_fade_entry_condition = abs(current_close_15m - SHORT_ATH_FADE_ENTRY) <= 5  # Within $5 of entry
            
            ath_fade_ready = ath_fade_trigger_condition and ath_fade_entry_condition and short_volume_confirmed

            logger.info("🔍 SHORT - ATH Deviation Fade Strategy Analysis:")
            logger.info(f"   • Wick above ${SHORT_ATH_FADE_TRIGGER_LOW:,}–${SHORT_ATH_FADE_TRIGGER_HIGH:,}: {'✅' if ath_fade_trigger_condition else '❌'} (last high: {last_high_15m:,.2f})")
            logger.info(f"   • Entry at ${SHORT_ATH_FADE_ENTRY:,}: {'✅' if ath_fade_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if short_volume_confirmed else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • ATH Deviation Fade Short Ready: {'🎯 YES' if ath_fade_ready else '⏳ NO'}")

            if ath_fade_ready:
                logger.info("")
                logger.info("🎯 SHORT - ATH Deviation Fade Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH ATH Deviation Fade Short (New Setup)",
                    entry_price=SHORT_ATH_FADE_ENTRY,
                    stop_loss=SHORT_ATH_FADE_STOP_LOSS,
                    take_profit=SHORT_ATH_FADE_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 ATH Deviation Fade Short trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_ATH_FADE_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_ATH_FADE_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_ATH_FADE_TP1:,.2f}, TP2: ${SHORT_ATH_FADE_TP2:,.2f}")
                    logger.info("Strategy: Wick above 4,900-4,910 with rejection with volume confirmation")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-ATH-Fade',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': SHORT_ATH_FADE_ENTRY,
                        'stop_loss': SHORT_ATH_FADE_STOP_LOSS,
                        'take_profit': SHORT_ATH_FADE_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Wick above ${SHORT_ATH_FADE_TRIGGER_LOW:,}-${SHORT_ATH_FADE_TRIGGER_HIGH:,} with rejection, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_ath_fade_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_ATH_FADE_ENTRY
                    }
                    save_trigger_state(short_ath_fade_state, SHORT_ATH_FADE_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ ATH Deviation Fade trade failed: {trade_result}")
        
        # 4. SHORT - Range Break Down Strategy
        if (short_strategies_enabled and 
            not short_range_break_state.get("triggered", False) and not trade_executed):
            
            # Check if price loss of breakdown level
            range_break_trigger_condition = last_close_15m < SHORT_RANGE_BREAK_TRIGGER_LOW
            # Check if current price is in entry zone
            range_break_entry_condition = (SHORT_RANGE_BREAK_ENTRY_LOW <= current_close_15m <= SHORT_RANGE_BREAK_ENTRY_HIGH)
            
            range_break_ready = range_break_trigger_condition and range_break_entry_condition and short_volume_confirmed

            logger.info("🔍 SHORT - Range Break Down Strategy Analysis:")
            logger.info(f"   • Loss of ${SHORT_RANGE_BREAK_TRIGGER_LOW:,}–${SHORT_RANGE_BREAK_TRIGGER_HIGH:,}: {'✅' if range_break_trigger_condition else '❌'} (last close: {last_close_15m:,.2f})")
            logger.info(f"   • Entry in ${SHORT_RANGE_BREAK_ENTRY_LOW:,}–${SHORT_RANGE_BREAK_ENTRY_HIGH:,}: {'✅' if range_break_entry_condition else '❌'} (current: {current_close_15m:,.2f})")
            logger.info(f"   • Volume confirmed (≥{SHORT_VOLUME_FACTOR}x): {'✅' if short_volume_confirmed else '❌'} (current: {relative_volume_15m:.2f}x)")
            logger.info(f"   • Range Break Down Short Ready: {'🎯 YES' if range_break_ready else '⏳ NO'}")

            if range_break_ready:
                logger.info("")
                logger.info("🎯 SHORT - Range Break Down Strategy conditions met - executing trade...")

                try:
                    play_alert_sound()
                    logger.info("Alert sound played successfully")
                except Exception as e:
                    logger.error(f"Failed to play alert sound: {e}")

                trade_success, trade_result = execute_crypto_trade(
                    cb_service=cb_service,
                    trade_type="ETH Range Break Down Short (New Setup)",
                    entry_price=SHORT_RANGE_BREAK_ENTRY,
                    stop_loss=SHORT_RANGE_BREAK_STOP_LOSS,
                    take_profit=SHORT_RANGE_BREAK_TP1,
                    side="SELL",
                    product=PRODUCT_ID
                )

                if trade_success:
                    logger.info("🎉 Range Break Down Short trade executed successfully!")
                    logger.info(f"Entry: ${SHORT_RANGE_BREAK_ENTRY:,.2f}")
                    logger.info(f"Stop-loss: ${SHORT_RANGE_BREAK_STOP_LOSS:,.2f}")
                    logger.info(f"TP1: ${SHORT_RANGE_BREAK_TP1:,.2f}, TP2: ${SHORT_RANGE_BREAK_TP2:,.2f}")
                    logger.info("Strategy: Loss of 4,697.5 with volume confirmation")
                    
                    # Log trade to CSV
                    trade_data = {
                        'timestamp': datetime.now(UTC).isoformat(),
                        'strategy': 'ETH-Range-Break',
                        'symbol': 'ETH-PERP-INTX',
                        'side': 'SELL',
                        'entry_price': SHORT_RANGE_BREAK_ENTRY,
                        'stop_loss': SHORT_RANGE_BREAK_STOP_LOSS,
                        'take_profit': SHORT_RANGE_BREAK_TP1,
                        'position_size_usd': POSITION_SIZE_USD,
                        'margin': MARGIN,
                        'leverage': LEVERAGE,
                        'volume_sma': avg_volume_5m,
                        'volume_ratio': relative_volume_5m,
                        'current_price': current_close_5m,
                        'market_conditions': f"24h Range: ${TODAY_LOW:,}-${TODAY_HIGH:,}",
                        'trade_status': 'EXECUTED',
                        'execution_time': datetime.now(UTC).isoformat(),
                        'notes': f"Trigger: Loss of ${SHORT_RANGE_BREAK_TRIGGER_LOW:,}-${SHORT_RANGE_BREAK_TRIGGER_HIGH:,}, Volume: {relative_volume_5m:.2f}x SMA"
                    }
                    log_trade_to_csv(trade_data)
                    
                    # Save trigger state
                    short_range_break_state = {
                        "triggered": True, 
                        "trigger_ts": int(last_closed_5m['start']),
                        "entry_price": SHORT_RANGE_BREAK_ENTRY
                    }
                    save_trigger_state(short_range_break_state, SHORT_RANGE_BREAK_TRIGGER_FILE)
                    trade_executed = True
                else:
                    logger.error(f"❌ Range Break Down trade failed: {trade_result}")
        
        # Check if any strategy was triggered
        if not trade_executed:
            logger.info("⏳ Waiting for setup conditions or monitoring active trade...")
            logger.info(f"ATH Breakout triggered: {long_ath_breakout_state.get('triggered', False)}")
            logger.info(f"Prior-Breakout Dip triggered: {long_dip_buy_state.get('triggered', False)}")
            logger.info(f"ATH Deviation Fade triggered: {short_ath_fade_state.get('triggered', False)}")
            logger.info(f"Range Break Down triggered: {short_range_break_state.get('triggered', False)}")
        
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
    parser = argparse.ArgumentParser(description='ETH ATH Trading Strategy Monitor (New Setup) with optional direction filter')
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
    
    logger.info("Starting ETH ATH Trading Strategy Monitor (New Setup)")
    if direction == 'BOTH':
        logger.info("Strategy: ETH ATH New Setup - LONG & SHORT")
    else:
        logger.info(f"Strategy: {direction} only")
    logger.info("")
    logger.info("Strategy Summary:")
    logger.info("LONG - ATH Breakout: 15-min close above 4,900, buy 4,905–4,935 on retest hold")
    logger.info("LONG - Prior-Breakout Dip: Pullback to 4,700–4,750, buy 4,710–4,745")
    logger.info("SHORT - ATH Deviation Fade: Wick above 4,930–5,000, sell 4,890–4,910")
    logger.info("SHORT - Range Break Down: Loss of 4,680–4,700, sell 4,690–4,705")
    logger.info(f"Position Size: ${POSITION_SIZE_USD:,} ({MARGIN} × {LEVERAGE}x)")
    logger.info(f"Volume: 15m RVOL ≥ {LONG_VOLUME_FACTOR}x for long, {SHORT_VOLUME_FACTOR}x for short, 5m RVOL ≥ {DIP_BUY_VOLUME_FACTOR}x for dip buy")
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
