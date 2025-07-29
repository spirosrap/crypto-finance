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
        logging.FileHandler('btc_dual_strategy_alert_debug.log'),
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
    """Calculate exponential backoff delay with jitter"""
    import random
    delay = min(INITIAL_RETRY_DELAY * (BACKOFF_MULTIPLIER ** attempt), MAX_RETRY_DELAY)
    # Add jitter to prevent thundering herd
    jitter = delay * 0.1 * random.random()
    return delay + jitter

def retry_with_backoff(func, *args, **kwargs):
    """
    Retry a function with exponential backoff on connection errors
    
    Args:
        func: Function to retry
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Function result on success, None on permanent failure
    """
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
            # For non-connection errors, don't retry
            logger.error(f"❌ Non-recoverable error: {e}")
            return None
    
    return None

def safe_get_candles(cb_service, product_id, start_ts, end_ts, granularity):
    """
    Safely get candles with retry logic
    
    Args:
        cb_service: Coinbase service instance
        product_id: Trading product ID
        start_ts: Start timestamp
        end_ts: End timestamp
        granularity: Candle granularity
    
    Returns:
        Candles list on success, None on failure
    """
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

# Constants for dual strategy monitoring
GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"

# Trade parameters for BTC dual strategy
BTC_DUAL_STRATEGY_MARGIN = 250  # USD
BTC_DUAL_STRATEGY_LEVERAGE = 20  # 20x leverage

# Trade tracking
btc_dual_strategy_trade_taken = False

TRIGGER_STATE_FILE = "btc_dual_strategy_trigger_state.json"

def load_trigger_state():
    if os.path.exists(TRIGGER_STATE_FILE):
        try:
            with open(TRIGGER_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"triggered": False, "trigger_ts": None}
    return {"triggered": False, "trigger_ts": None}

def save_trigger_state(state):
    try:
        with open(TRIGGER_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trigger state: {e}")

def play_alert_sound(filename="alert_sound.wav"):
    """
    Play the alert sound using system commands
    """
    try:
        system = platform.system()
        
        if system == "Darwin":  # macOS
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
    """Setup Coinbase service with connection validation"""
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    
    def _create_service():
        service = CoinbaseService(api_key, api_secret)
        # Test the connection with a simple API call
        try:
            # Try to get a small amount of candle data to validate connection
            test_response = service.client.get_public_candles(
                product_id="BTC-PERP-INTX",
                start=int((datetime.now(UTC) - timedelta(hours=2)).timestamp()),
                end=int(datetime.now(UTC).timestamp()),
                granularity="ONE_HOUR"
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
                     margin: float = 250, leverage: int = 20, side: str = "BUY", product: str = "BTC-PERP-INTX"):
    """
    General crypto trade execution function using trade_btc_perp.py with retry logic
    
    Args:
        cb_service: Coinbase service instance
        trade_type: Description of the trade type for logging
        entry_price: Entry price for logging
        stop_loss: Stop-loss price
        take_profit: Take-profit price
        margin: USD amount to risk (default: 250)
        leverage: Leverage multiplier (default: 20)
        side: Trade side - "BUY" or "SELL" (default: "BUY")
        product: Trading product (default: "BTC-PERP-INTX")
    """
    def _execute_trade():
        logger.info(f"Executing crypto trade: {trade_type} at ${entry_price:,.2f}")
        logger.info(f"Trade params: Margin=${margin}, Leverage={leverage}x, Side={side}, Product={product}")
        
        # Calculate position size based on margin and leverage
        position_size_usd = margin * leverage
        
        # Use subprocess to call trade_btc_perp.py
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', product,
            '--side', side,
            '--size', str(position_size_usd),
            '--leverage', str(leverage),
            '--tp', str(take_profit),
            '--sl', str(stop_loss),
            '--no-confirm'  # Skip confirmation for automated trading
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute the trade command
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
        # Use retry logic for trade execution
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


class BreakoutState:
    def __init__(self):
        self.state = "WATCH"  # WATCH, TRIGGERED, ENTER
        self.entry_price = None

    def to_dict(self):
        return {"state": self.state, "entry_price": self.entry_price}

    def from_dict(self, d):
        self.state = d.get("state", "WATCH")
        self.entry_price = d.get("entry_price")


def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index) for a list of prices
    
    Args:
        prices: List of price values
        period: RSI period (default: 14)
    
    Returns:
        RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50  # Default to neutral if not enough data
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_btc_perp_position_size(cb_service):
    """
    Returns the open position size for BTC-PERP-INTX (absolute value, in base currency units).
    Returns 0.0 if no open position.
    Handles both dict and SDK object responses.
    """
    try:
        # Get the INTX portfolio UUID and breakdown
        ports = cb_service.client.get_portfolios()
        portfolio_uuid = None
        for p in ports['portfolios']:
            if p['type'] == "INTX":
                portfolio_uuid = p['uuid']
                break
        if not portfolio_uuid:
            logger.error("Could not find INTX portfolio")
            return 0.0
        portfolio = cb_service.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
        # Convert to dict if needed
        if not isinstance(portfolio, dict):
            if hasattr(portfolio, '__dict__'):
                portfolio = vars(portfolio)
            else:
                logger.error("Portfolio breakdown is not a dict and has no __dict__")
                return 0.0
        breakdown = portfolio.get('breakdown', {})
        # Convert breakdown to dict if needed
        if not isinstance(breakdown, dict):
            if hasattr(breakdown, '__dict__'):
                breakdown = vars(breakdown)
            else:
                logger.error("Breakdown is not a dict and has no __dict__")
                return 0.0
        positions = breakdown.get('perp_positions', [])
        for pos in positions:
            # Convert pos to dict if needed
            if not isinstance(pos, dict):
                if hasattr(pos, '__dict__'):
                    pos = vars(pos)
                else:
                    continue
            if pos.get('symbol') == "BTC-PERP-INTX":
                return abs(float(pos.get('net_size', 0)))
        return 0.0
    except Exception as e:
        logger.error(f"Error getting BTC-PERP-INTX position size: {e}")
        return 0.0


def btc_dual_strategy_alert(cb_service, last_alert_ts=None):
    """
    BTC Dual Strategy Alert - Implements both Plan A (breakout) and Plan B (pullback)
    Based on the trading plans from the image
    """
    logger.info("=== BTC-USD Dual Strategy Alert (Plan A: Breakout + Plan B: Pullback) ===")
    PRODUCT_ID = "BTC-PERP-INTX"
    GRANULARITY = "ONE_HOUR"  # 1-hour candles for both strategies

    # Plan A: Breakout Long (Momentum) Parameters
    PLAN_A_TRIGGER_CONSERVATIVE = 119650  # $119,650 (conservative trigger)
    PLAN_A_TRIGGER_AGGRESSIVE = 120150    # $120,150 (alternative trigger)
    PLAN_A_STOP_LOSS = 118800            # $118,800 (below breakout bar)
    PLAN_A_TP1 = 120900                  # $120,900 (first target)
    PLAN_A_TP2_LOW = 122500              # $122,500 (second target range)
    PLAN_A_TP2_HIGH = 123500             # $123,500 (second target range)
    PLAN_A_VOLUME_THRESHOLD = 1.25       # 1.25x 20-period SMA volume

    # Plan B: Pullback Long (Fade to Support) Parameters
    PLAN_B_BID_ZONE_LOW = 117350         # $117,350 (bid zone)
    PLAN_B_BID_ZONE_HIGH = 117600        # $117,600 (bid zone)
    PLAN_B_STOP_LOSS = 116600            # $116,600
    PLAN_B_TP_LOW = 118900               # $118,900 (take profit range)
    PLAN_B_TP_HIGH = 119500              # $119,500 (take profit range)

    # Common parameters
    MARGIN = 250                         # USD margin
    LEVERAGE = 20                        # 20x leverage
    VOLUME_PERIOD = 20                   # For volume SMA calculation
    RSI_PERIOD = 14                      # RSI period
    periods_needed = max(VOLUME_PERIOD, RSI_PERIOD) + 5

    # Load trigger states for both plans
    trigger_state_file = "btc_dual_strategy_trigger_state.json"
    
    def load_dual_trigger_state():
        if os.path.exists(trigger_state_file):
            try:
                with open(trigger_state_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {"plan_a_triggered": False, "plan_b_triggered": False, "last_trigger_ts": None}
        return {"plan_a_triggered": False, "plan_b_triggered": False, "last_trigger_ts": None}

    def save_dual_trigger_state(state):
        try:
            with open(trigger_state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.error(f"Failed to save dual trigger state: {e}")

    trigger_state = load_dual_trigger_state()

    try:
        logger.info("Setting up time parameters for 1-hour candles...")
        now = datetime.now(UTC)
        # Get the start of the current hour
        now = now.replace(minute=0, second=0, microsecond=0)
        start = now - timedelta(hours=periods_needed)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        logger.info(f"Time range: {start} to {end}")

        logger.info("Fetching 1-hour candles from API...")
        candles = safe_get_candles(cb_service, PRODUCT_ID, start_ts, end_ts, GRANULARITY)
        logger.info(f"Candles fetched: {len(candles) if candles else 0} candles")

        if not candles or len(candles) < periods_needed:
            logger.warning(f"Not enough BTC {GRANULARITY} candle data for dual strategy alert.")
            logger.info("=== BTC-USD Dual Strategy Alert completed (insufficient data) ===")
            return last_alert_ts

        def get_candle_value(candle, key):
            if isinstance(candle, dict):
                value = candle.get(key)
            else:
                value = getattr(candle, key, None)
            return value

        # Use candles[1] as the last fully closed candle (skip in-progress)
        last_candle = candles[1]
        ts = datetime.fromtimestamp(int(get_candle_value(last_candle, 'start')), UTC)
        close = float(get_candle_value(last_candle, 'close'))
        high = float(get_candle_value(last_candle, 'high'))
        low = float(get_candle_value(last_candle, 'low'))
        volume = float(get_candle_value(last_candle, 'volume'))

        # Calculate volume confirmation for Plan A
        historical_candles = candles[2:VOLUME_PERIOD+2]
        avg_volume = sum(float(get_candle_value(c, 'volume')) for c in historical_candles) / len(historical_candles)
        relative_volume = volume / avg_volume if avg_volume > 0 else 0

        # Calculate RSI
        closes = [float(get_candle_value(c, 'close')) for c in candles[1:RSI_PERIOD+2]]
        rsi = calculate_rsi(closes, RSI_PERIOD) if len(closes) >= RSI_PERIOD else 50

        # Get previous candle for Plan B analysis
        prev_candle = candles[2] if len(candles) > 2 else None
        prev_close = float(get_candle_value(prev_candle, 'close')) if prev_candle else close
        prev_volume = float(get_candle_value(prev_candle, 'volume')) if prev_candle else volume

        # --- Reporting ---
        logger.info("")
        logger.info("🚀 BTC/USD Dual Strategy Alert")
        logger.info("")
        logger.info("📊 Plan A: Breakout Long (Momentum)")
        logger.info(f"   • Trigger: 1h close ≥ ${PLAN_A_TRIGGER_CONSERVATIVE:,} (conservative) or ≥ ${PLAN_A_TRIGGER_AGGRESSIVE:,} (aggressive)")
        logger.info(f"   • Entry: Buy-stop at ${PLAN_A_TRIGGER_CONSERVATIVE:,} or ${PLAN_A_TRIGGER_AGGRESSIVE:,}")
        logger.info(f"   • Stop Loss: ${PLAN_A_STOP_LOSS:,} (below breakout bar)")
        logger.info(f"   • TP1: ${PLAN_A_TP1:,}")
        logger.info(f"   • TP2: ${PLAN_A_TP2_LOW:,}-${PLAN_A_TP2_HIGH:,}")
        logger.info(f"   • Volume Condition: ≥ {PLAN_A_VOLUME_THRESHOLD}x 20-period SMA")
        logger.info("")
        logger.info("📊 Plan B: Pullback Long (Fade to Support)")
        logger.info(f"   • Bid Zone: ${PLAN_B_BID_ZONE_LOW:,}-${PLAN_B_BID_ZONE_HIGH:,} (wick + quick reclaim only)")
        logger.info(f"   • Stop Loss: ${PLAN_B_STOP_LOSS:,}")
        logger.info(f"   • Take Profit: ${PLAN_B_TP_LOW:,}-${PLAN_B_TP_HIGH:,} (trail if strength persists)")
        logger.info(f"   • Conditions: Zone reclaimed within 1-2 candles, no fill on heavy sell volume")
        logger.info("")
        logger.info(f"Current 1-Hour Candle: close=${close:,.2f}, high=${high:,.2f}, low=${low:,.2f}")
        logger.info(f"Volume: {volume:,.0f}, Avg20: {avg_volume:,.0f}, Rel_Vol: {relative_volume:.2f}")
        logger.info(f"RSI: {rsi:.1f}")
        logger.info("")

        # --- Plan A: Breakout Long Logic ---
        plan_a_conditions = []
        
        # Check if price closed above either trigger level
        conservative_trigger = close >= PLAN_A_TRIGGER_CONSERVATIVE
        aggressive_trigger = close >= PLAN_A_TRIGGER_AGGRESSIVE
        price_trigger = conservative_trigger or aggressive_trigger
        
        # Volume confirmation
        volume_confirmed = relative_volume >= PLAN_A_VOLUME_THRESHOLD
        
        # RSI not overbought
        rsi_ok = rsi <= 70
        
        plan_a_conditions = [price_trigger, volume_confirmed, rsi_ok]
        plan_a_ready = all(plan_a_conditions) and not trigger_state.get("plan_a_triggered", False)

        logger.info("🔍 Plan A (Breakout) Analysis:")
        logger.info(f"   • Price trigger (≥${PLAN_A_TRIGGER_CONSERVATIVE:,} or ≥${PLAN_A_TRIGGER_AGGRESSIVE:,}): {'✅' if price_trigger else '❌'}")
        logger.info(f"   • Volume ≥ {PLAN_A_VOLUME_THRESHOLD}x avg: {'✅' if volume_confirmed else '❌'}")
        logger.info(f"   • RSI ≤ 70: {'✅' if rsi_ok else '❌'}")
        logger.info(f"   • Not already triggered: {'✅' if not trigger_state.get("plan_a_triggered", False) else '❌'}")
        logger.info(f"   • Plan A Ready: {'🎯 YES' if plan_a_ready else '⏳ NO'}")

        # --- Plan B: Pullback Long Logic ---
        plan_b_conditions = []
        
        # Check if price is in bid zone
        in_bid_zone = PLAN_B_BID_ZONE_LOW <= low <= PLAN_B_BID_ZONE_HIGH
        
        # Check for wick (price touched the zone)
        wick_in_zone = low <= PLAN_B_BID_ZONE_HIGH and low >= PLAN_B_BID_ZONE_LOW
        
        # Check for quick reclaim (price closed above the zone)
        quick_reclaim = close > PLAN_B_BID_ZONE_HIGH
        
        # Check for no heavy sell volume (volume not significantly higher than average)
        no_heavy_sell = relative_volume <= 2.0  # Not more than 2x average volume
        
        # Check if we haven't already triggered Plan B
        not_already_triggered = not trigger_state.get("plan_b_triggered", False)
        
        plan_b_conditions = [wick_in_zone, quick_reclaim, no_heavy_sell, not_already_triggered]
        plan_b_ready = all(plan_b_conditions)

        logger.info("")
        logger.info("🔍 Plan B (Pullback) Analysis:")
        logger.info(f"   • Wick in bid zone (${PLAN_B_BID_ZONE_LOW:,}-${PLAN_B_BID_ZONE_HIGH:,}): {'✅' if wick_in_zone else '❌'}")
        logger.info(f"   • Quick reclaim (close > ${PLAN_B_BID_ZONE_HIGH:,}): {'✅' if quick_reclaim else '❌'}")
        logger.info(f"   • No heavy sell volume (≤2x avg): {'✅' if no_heavy_sell else '❌'}")
        logger.info(f"   • Not already triggered: {'✅' if not_already_triggered else '❌'}")
        logger.info(f"   • Plan B Ready: {'🎯 YES' if plan_b_ready else '⏳ NO'}")

        # --- Execute Trades ---
        trade_executed = False

        if plan_a_ready:
            logger.info("")
            logger.info("🎯 Plan A (Breakout) conditions met - executing trade...")
            
            # Determine entry price based on which trigger was hit
            if conservative_trigger:
                entry_price = PLAN_A_TRIGGER_CONSERVATIVE
                trigger_type = "conservative"
            else:
                entry_price = PLAN_A_TRIGGER_AGGRESSIVE
                trigger_type = "aggressive"
            
            logger.info(f"Trade Setup: Entry=${entry_price:,} ({trigger_type}), SL=${PLAN_A_STOP_LOSS:,}, TP1=${PLAN_A_TP1:,}, TP2=${PLAN_A_TP2_LOW:,}-${PLAN_A_TP2_HIGH:,}")
            logger.info(f"Risk: ${MARGIN}, Leverage: {LEVERAGE}x")

            # Play alert sound
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")

            # Execute Plan A trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="BTC/USD Plan A Breakout Long",
                entry_price=entry_price,
                stop_loss=PLAN_A_STOP_LOSS,
                take_profit=PLAN_A_TP1,  # Use TP1 as primary target
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )

            if trade_success:
                logger.info(f"🎉 Plan A (Breakout) trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
                trigger_state["plan_a_triggered"] = True
                trigger_state["last_trigger_ts"] = int(get_candle_value(last_candle, 'start'))
                save_dual_trigger_state(trigger_state)
                trade_executed = True
            else:
                logger.error(f"❌ Plan A (Breakout) trade failed: {trade_result}")

        elif plan_b_ready:
            logger.info("")
            logger.info("🎯 Plan B (Pullback) conditions met - executing trade...")
            
            # Use the current close price as entry (since we're buying after the reclaim)
            entry_price = close
            
            logger.info(f"Trade Setup: Entry=${entry_price:,}, SL=${PLAN_B_STOP_LOSS:,}, TP=${PLAN_B_TP_LOW:,}-${PLAN_B_TP_HIGH:,}")
            logger.info(f"Risk: ${MARGIN}, Leverage: {LEVERAGE}x")

            # Play alert sound
            try:
                play_alert_sound()
                logger.info("Alert sound played successfully")
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")

            # Execute Plan B trade
            trade_success, trade_result = execute_crypto_trade(
                cb_service=cb_service,
                trade_type="BTC/USD Plan B Pullback Long",
                entry_price=entry_price,
                stop_loss=PLAN_B_STOP_LOSS,
                take_profit=PLAN_B_TP_LOW,  # Use lower TP as primary target
                margin=MARGIN,
                leverage=LEVERAGE,
                side="BUY",
                product=PRODUCT_ID
            )

            if trade_success:
                logger.info(f"🎉 Plan B (Pullback) trade executed successfully!")
                logger.info(f"Trade output: {trade_result}")
                trigger_state["plan_b_triggered"] = True
                trigger_state["last_trigger_ts"] = int(get_candle_value(last_candle, 'start'))
                save_dual_trigger_state(trigger_state)
                trade_executed = True
            else:
                logger.error(f"❌ Plan B (Pullback) trade failed: {trade_result}")

        if not trade_executed:
            logger.info("")
            logger.info("⏳ No trade conditions met for either plan")
            logger.info(f"Plan A triggered: {trigger_state.get('plan_a_triggered', False)}")
            logger.info(f"Plan B triggered: {trigger_state.get('plan_b_triggered', False)}")

        logger.info("=== BTC-USD Dual Strategy Alert completed ===")
        return ts if trade_executed else last_alert_ts

    except Exception as e:
        logger.error(f"Error in BTC-USD Dual Strategy alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=== BTC-USD Dual Strategy Alert completed (with error) ===")
    return last_alert_ts

# Remove old alert functions
def main():
    logger.info("Starting BTC/USD Dual Strategy Alert Monitor (Plan A: Breakout + Plan B: Pullback)")
    logger.info("")
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    logger.info("")
    cb_service = setup_coinbase()
    last_alert_ts = None
    consecutive_failures = 0
    max_consecutive_failures = 5
    def poll_iteration():
        nonlocal last_alert_ts, consecutive_failures
        iteration_start_time = time.time()
        last_alert_ts = btc_dual_strategy_alert(cb_service, last_alert_ts)
        consecutive_failures = 0
        logger.info(f"✅ Dual Strategy alert cycle completed successfully in {time.time() - iteration_start_time:.1f} seconds")
    while True:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(poll_iteration)
                try:
                    future.result(timeout=120)  # 2 minute max per poll
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