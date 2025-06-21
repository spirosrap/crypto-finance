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

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for get_recent_hourly_candles
GRANULARITY = "ONE_HOUR"
PRODUCT_ID = "BTC-PERP-INTX"

# Trade parameters for BTC breakout
BTC_BREAKOUT_MARGIN = 300  # USD
BTC_BREAKOUT_LEVERAGE = 20  # 20x leverage
BTC_BREAKOUT_STOP_LOSS = 102000  # Stop-loss at $102,000
BTC_BREAKOUT_TAKE_PROFIT = 108000  # First profit target at $108,000

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
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    return CoinbaseService(api_key, api_secret)


def get_recent_hourly_candles(cb_service, num_candles=24):
    now = datetime.now(UTC)
    now = now.replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=num_candles)
    end = now
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    logger.info(f"Fetching last {num_candles} hourly candles from {start} to {end}")
    response = cb_service.client.get_public_candles(
        product_id=PRODUCT_ID,
        start=start_ts,
        end=end_ts,
        granularity=GRANULARITY
    )
    if hasattr(response, 'candles'):
        candles = response.candles
    else:
        candles = response.get('candles', [])
    return candles


def btc_triangle_breakout_alert(cb_service, last_alert_ts=None):
    """
    Alerts on an intraday triangle breakout for BTC.
    Entry trigger: 1-hour close > 104,200 with >=2x 20-period volume.
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    ENTRY_PRICE_THRESHOLD = 104200
    VOLUME_PERIOD = 20
    VOLUME_MULTIPLIER = 2.0
    ENTRY_ZONE_LOW = 104200
    ENTRY_ZONE_HIGH = 104800

    try:
        # 1. Get candles for analysis (volume period + 2 for current and last closed)
        candles_raw = get_recent_hourly_candles(cb_service, num_candles=VOLUME_PERIOD + 2)
        if not candles_raw or len(candles_raw) < VOLUME_PERIOD + 2:
            logger.warning(f"Not enough BTC hourly candle data for triangle breakout alert. Need {VOLUME_PERIOD + 2}, got {len(candles_raw)}.")
            return last_alert_ts

        # 2. Prepare data for analysis
        last_closed_candle_raw = candles_raw[1]
        ts = datetime.fromtimestamp(int(last_closed_candle_raw['start']), UTC)

        # Avoid re-alerting for the same candle
        if ts == last_alert_ts:
            return last_alert_ts
        
        # Historical candles for volume average (20 periods before the last closed candle)
        historical_candles = candles_raw[2:VOLUME_PERIOD + 2]
        if len(historical_candles) < VOLUME_PERIOD:
            logger.warning(f"Not enough historical BTC hourly candle data for volume average. Need {VOLUME_PERIOD}, got {len(historical_candles)}.")
            return last_alert_ts

        volumes = [float(c['volume']) for c in historical_candles]
        avg_volume = sum(volumes) / len(volumes)

        last_close = float(last_closed_candle_raw['close'])
        last_volume = float(last_closed_candle_raw['volume'])

        logger.info(f"BTC Triangle Breakout Check: Last Close=${last_close:,.2f}, Last Volume={last_volume:,.0f}, Avg Volume({VOLUME_PERIOD})={avg_volume:,.0f}")

        # 4. Check alert conditions
        is_breakout_price = last_close > ENTRY_PRICE_THRESHOLD
        is_high_volume = last_volume >= (avg_volume * VOLUME_MULTIPLIER)
        
        if is_breakout_price and is_high_volume:
            logger.info(f"--- BTC TRIANGLE BREAKOUT ALERT ---")
            logger.info(f"Entry condition met: 1-hour close > ${ENTRY_PRICE_THRESHOLD:,.0f} with volume >= {VOLUME_MULTIPLIER}x 20-period average.")
            
            if ENTRY_ZONE_LOW <= last_close <= ENTRY_ZONE_HIGH:
                logger.info(f"Price ${last_close:,.2f} is within designated entry zone (${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}).")
            else:
                logger.warning(f"Price ${last_close:,.2f} is outside designated entry zone (${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}).")

            logger.info(f"Details: Timestamp={ts}, Close=${last_close:,.2f}, Volume={last_volume:,.0f}, Avg Volume={avg_volume:,.0f}")

            # Execute the trade
            logger.info("Executing BTC breakout trade...")
            breakout_type = f"triangle_breakout_{ENTRY_PRICE_THRESHOLD}"
            trade_success, trade_result = execute_btc_breakout_trade(cb_service, breakout_type, last_close)

            if trade_success:
                logger.info("BTC breakout trade executed successfully!")
                logger.info(f"Trade parameters: Margin=${BTC_BREAKOUT_MARGIN}, Leverage={BTC_BREAKOUT_LEVERAGE}x")
                logger.info(f"Stop Loss: ${BTC_BREAKOUT_STOP_LOSS:,.0f}, Take Profit: ${BTC_BREAKOUT_TAKE_PROFIT:,.0f}")
            else:
                logger.error(f"BTC breakout trade failed: {trade_result}")

            logger.info("")
            return ts

    except Exception as e:
        logger.error(f"Error in BTC triangle breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return last_alert_ts


def fartcoin_daily_alert(cb_service, last_alert_ts=None):
    PRODUCT_ID = "FARTCOIN-PERP-INTX"
    GRANULARITY = "ONE_HOUR"  # Get 1-hour candles to aggregate
    
    # Alert thresholds
    ALERT_1_00 = 1.00
    ALERT_0_95 = 0.95
    ALERT_0_90 = 0.90
    RSI_THRESHOLD = 30
    RSI_PERIOD = 14
    VOLUME_PERIOD = 15  # Reduced from 20 to 15 for 6-hour candles
    CONSECUTIVE_RSI_PERIODS = 2  # Need 2 consecutive periods with RSI < 30

    try:
        now = datetime.now(UTC)
        now = now.replace(minute=0, second=0, microsecond=0)  # Round to hour
        # Get more hourly data to aggregate into 6-hour candles
        start = now - timedelta(hours=150)  # Increased from 120 to 150 hours (6.25 days)
        end = now
        start_ts = int(start.timestamp())
        end_ts = int(end.timestamp())
        
        response = cb_service.client.get_public_candles(
            product_id=PRODUCT_ID,
            start=start_ts,
            end=end_ts,
            granularity=GRANULARITY
        )
        if hasattr(response, 'candles'):
            candles = response.candles
        else:
            candles = response.get('candles', [])
        if not candles or len(candles) < max(RSI_PERIOD * 6 + 1, VOLUME_PERIOD * 6 + 1):  # Need enough 1-hour candles
            logger.warning("Not enough FARTCOIN hourly candle data for 6-hour aggregation and analysis.")
            return last_alert_ts
        
        # Aggregate 1-hour candles into 6-hour candles
        six_hour_candles = []
        for i in range(0, len(candles) - 5, 6):  # Step by 6 hours
            if i + 5 < len(candles):
                hour_candles = candles[i:i+6]  # Get 6 consecutive 1-hour candles
                
                # Aggregate into 6-hour candle
                opens = [float(c['open']) for c in hour_candles]
                highs = [float(c['high']) for c in hour_candles]
                lows = [float(c['low']) for c in hour_candles]
                closes = [float(c['close']) for c in hour_candles]
                volumes = [float(c['volume']) for c in hour_candles]
                
                six_hour_candle = {
                    'open': opens[0],  # First hour's open
                    'high': max(highs),  # Highest high
                    'low': min(lows),    # Lowest low
                    'close': closes[-1],  # Last hour's close
                    'volume': sum(volumes),  # Sum of volumes
                    'start': hour_candles[0]['start']  # Start time of first hour
                }
                six_hour_candles.append(six_hour_candle)
        
        if len(six_hour_candles) < max(RSI_PERIOD + 1, VOLUME_PERIOD + 1):
            logger.warning(f"Not enough 6-hour candles for analysis. Need {max(RSI_PERIOD + 1, VOLUME_PERIOD + 1)}, got {len(six_hour_candles)}.")
            return last_alert_ts
        
        # Use the most recent 6-hour candle
        last_candle = six_hour_candles[1]  # Second most recent (most recent completed)
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)

        if ts == last_alert_ts:
            return last_alert_ts

        close = float(last_candle['close'])
        current_volume = float(last_candle['volume'])
        
        # Calculate RSI using 6-hour candles
        candle_data = []
        for candle in six_hour_candles:
            candle_data.append({
                'close': float(candle['close']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'volume': float(candle['volume']),
                'start': candle['start']
            })
        
        df = pd.DataFrame(candle_data)
        
        # Calculate RSI using pandas_ta
        rsi = ta.rsi(df['close'], length=RSI_PERIOD)
        current_rsi = rsi.iloc[-1]
        
        # Check momentum signal: RSI < 30 for at least 2 consecutive periods
        momentum_signal = False
        if len(rsi) >= CONSECUTIVE_RSI_PERIODS:
            recent_rsi_values = rsi.iloc[-CONSECUTIVE_RSI_PERIODS:].dropna()
            if len(recent_rsi_values) >= CONSECUTIVE_RSI_PERIODS:
                momentum_signal = all(rsi_val < RSI_THRESHOLD for rsi_val in recent_rsi_values)
        
        # Calculate volume confirmation: Current volume >= 20-period average
        volume_confirmation = False
        if len(six_hour_candles) >= VOLUME_PERIOD + 1:
            volume_period_candles = six_hour_candles[1:VOLUME_PERIOD + 1]  # Exclude current candle
            avg_volume = sum(float(c['volume']) for c in volume_period_candles) / len(volume_period_candles)
            volume_confirmation = current_volume >= avg_volume
        
        logger.info(f"FARTCOIN Check (6H): Close=${close:.5f}, RSI={current_rsi:.2f}, Volume={current_volume:,.0f}")
        logger.info(f"Momentum Signal (RSI<30 for 2 periods): {momentum_signal}")
        logger.info(f"Volume Confirmation (≥20-period avg): {volume_confirmation}")
        
        # Alert conditions with momentum and volume confirmation
        alert_triggered = False
        
        # Alert 1: $1.00 with momentum and volume confirmation
        if close >= ALERT_1_00 and momentum_signal and volume_confirmation:
            logger.info(f"--- FARTCOIN ALERT 1 (6H) ---")
            logger.info(f"Price >= ${ALERT_1_00} with momentum signal and volume confirmation!")
            logger.info(f"Timestamp: {ts}, Close: ${close:.5f}, RSI: {current_rsi:.2f}")
            alert_triggered = True
        
        # Alert 2: Below $0.95 with momentum and volume confirmation
        elif close < ALERT_0_95 and momentum_signal and volume_confirmation:
            logger.info(f"--- FARTCOIN ALERT 2 (6H) ---")
            logger.info(f"Price < ${ALERT_0_95} with momentum signal and volume confirmation!")
            logger.info(f"Timestamp: {ts}, Close: ${close:.5f}, RSI: {current_rsi:.2f}")
            alert_triggered = True
        
        # Alert 3: $0.90 (regardless of RSI/volume - emergency alert)
        elif close <= ALERT_0_90:
            logger.info(f"--- FARTCOIN ALERT 3 (6H) - EMERGENCY ---")
            logger.info(f"Price <= ${ALERT_0_90} - Emergency alert triggered!")
            logger.info(f"Timestamp: {ts}, Close: ${close:.5f}, RSI: {current_rsi:.2f}")
            alert_triggered = True
        
        if alert_triggered:
            logger.info("")  # Empty line for visual separation
            
            # Play alert sound
            try:
                play_alert_sound()
            except Exception as e:
                logger.error(f"Failed to play alert sound: {e}")
            
            return ts

    except Exception as e:
        logger.error(f"Error in FARTCOIN alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return last_alert_ts


def execute_btc_breakout_trade(cb_service, breakout_type: str, entry_price: float):
    """
    Execute BTC breakout trade using trade_btc_perp.py functionality
    """
    try:
        logger.info(f"Executing BTC breakout trade: {breakout_type} at ${entry_price:,.2f}")
        
        # Calculate position size based on margin and leverage
        position_size_usd = BTC_BREAKOUT_MARGIN * BTC_BREAKOUT_LEVERAGE
        
        # Use subprocess to call trade_btc_perp.py
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', 'BTC-PERP-INTX',
            '--side', 'BUY',
            '--size', str(position_size_usd),
            '--leverage', str(BTC_BREAKOUT_LEVERAGE),
            '--tp', str(BTC_BREAKOUT_TAKE_PROFIT),
            '--sl', str(BTC_BREAKOUT_STOP_LOSS),
            '--no-confirm'  # Skip confirmation for automated trading
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Execute the trade command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("BTC breakout trade executed successfully!")
            logger.info(f"Trade output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"BTC breakout trade failed!")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error executing BTC breakout trade: {e}")
        return False, str(e)


def main():
    logger.info("Starting multi-asset alert script")
    logger.info("")  # Empty line for visual separation
    
    # Check if alert sound file exists
    alert_sound_file = "alert_sound.wav"
    if not os.path.exists(alert_sound_file):
        logger.error(f"❌ Alert sound file '{alert_sound_file}' not found!")
        logger.error("Please run 'python synthesize_alert_sound.py' first to create the sound file.")
        logger.error("Then run this script again.")
        return
    else:
        logger.info(f"✅ Alert sound file '{alert_sound_file}' found and ready")
    
    logger.info("")  # Empty line for visual separation
    
    cb_service = setup_coinbase()
    btc_triangle_breakout_last_alert_ts = None
    fartcoin_last_alert_ts = None
    
    while True:
        try:
            # BTC triangle breakout alert
            # btc_triangle_breakout_last_alert_ts = btc_triangle_breakout_alert(cb_service, btc_triangle_breakout_last_alert_ts)

            # FARTCOIN daily alert (runs hourly but condition only changes daily)
            fartcoin_last_alert_ts = fartcoin_daily_alert(cb_service, fartcoin_last_alert_ts)

            # Wait until next hour + 1 minute
            now = datetime.now(UTC)
            next_poll = (now + timedelta(hours=1)).replace(minute=1, second=0, microsecond=0)
            wait_seconds = (next_poll - now).total_seconds()
            logger.info(f"Waiting {wait_seconds:.0f} seconds until next poll")
            logger.info("")  # Empty line for visual separation
            time.sleep(wait_seconds)
            
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in alert loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(60)

if __name__ == "__main__":
    main() 