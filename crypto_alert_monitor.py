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
BTC_BREAKOUT_STOP_LOSS = 104300  # Stop-loss at $104,300
BTC_BREAKOUT_TAKE_PROFIT = 108500  # First profit target at $108,500

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


def get_current_btc_data(cb_service):
    """
    Get current live BTC price, volume, and RSI data
    """
    try:
        current_price = None
        current_volume = None
        current_rsi = None
        
        # Get recent candles for price, volume and RSI calculation (last 20 minutes for RSI)
        now = datetime.now(UTC)
        start = now - timedelta(minutes=20)
        start_ts = int(start.timestamp())
        end_ts = int(now.timestamp())
        
        response = cb_service.client.get_public_candles(
            product_id="BTC-PERP-INTX",
            start=start_ts,
            end=end_ts,
            granularity="ONE_MINUTE"
        )
        
        if hasattr(response, 'candles'):
            recent_candles = response.candles
        else:
            recent_candles = response.get('candles', [])
        
        if recent_candles:
            # Get current price from the most recent candle
            current_price = float(recent_candles[-1]['close'])
            
            # Sum volume from last 5 minutes
            current_volume = sum(float(c['volume']) for c in recent_candles[-5:])
            
            # Calculate current RSI using recent closes
            closes = [float(c['close']) for c in recent_candles]
            
            if len(closes) >= 14:  # Need at least 14 periods for RSI
                closes_series = pd.Series(closes)
                rsi_series = ta.rsi(closes_series, length=14)
                current_rsi = rsi_series.iloc[-1] if not rsi_series.isna().all() else None
        
        return current_price, current_volume, current_rsi
        
    except Exception as e:
        logger.error(f"Error getting current BTC data: {e}")
        return None, None, None


def btc_continuation_alert(cb_service, last_alert_ts=None):
    """
    Alerts on BTC continuation above ~$105k range.
    Entry trigger: 1-hour close > 105,700 on spike volume (>20% above avg).
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    ENTRY_PRICE_THRESHOLD = 105700
    VOLUME_PERIOD = 20
    VOLUME_MULTIPLIER = 1.2  # >20% above average
    ENTRY_ZONE_LOW = 105700
    ENTRY_ZONE_HIGH = 106200

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

        # Calculate 14-period RSI using pandas_ta
        closes = [float(c['close']) for c in candles_raw[1:VOLUME_PERIOD+2]]  # last_closed + previous 20
        closes_series = pd.Series(closes)
        rsi_series = ta.rsi(closes_series, length=14)
        last_rsi = rsi_series.iloc[-1] if not rsi_series.isna().all() else None
        rsi_str = f"{last_rsi:.2f}" if last_rsi is not None and not pd.isna(last_rsi) else "N/A"

        # Get current live data
        current_price, current_volume, current_rsi = get_current_btc_data(cb_service)
        current_price_str = f"${current_price:,.2f}" if current_price is not None else "N/A"
        current_volume_str = f"{current_volume:,.0f}" if current_volume is not None else "N/A"
        current_rsi_str = f"{current_rsi:.2f}" if current_rsi is not None and not pd.isna(current_rsi) else "N/A"

        # Report both current (live) and completed candle data
        logger.info(f"=== BTC MARKET DATA REPORT ===")
        logger.info(f"📊 COMPLETED CANDLE (1H): Close=${last_close:,.2f}, Volume={last_volume:,.0f}, RSI(14)={rsi_str}")
        logger.info(f"📈 CURRENT LIVE: Price={current_price_str}, Volume(5min)={current_volume_str}, RSI(14)={current_rsi_str}")
        logger.info(f"📊 HISTORICAL: Avg Volume({VOLUME_PERIOD})={avg_volume:,.0f}")
        logger.info(f"=================================")

        # 4. Check alert conditions
        is_breakout_price = last_close > ENTRY_PRICE_THRESHOLD
        is_high_volume = last_volume >= (avg_volume * VOLUME_MULTIPLIER)
        
        # Log condition status
        logger.info(f"  - Price > ${ENTRY_PRICE_THRESHOLD:,.0f}: {'✅ Met' if is_breakout_price else '❌ Not Met'}")
        logger.info(f"  - Volume >= {VOLUME_MULTIPLIER}x Avg ({avg_volume * VOLUME_MULTIPLIER:,.0f}): {'✅ Met' if is_high_volume else '❌ Not Met'}")
        
        if is_breakout_price and is_high_volume:
            logger.info(f"--- BTC CONTINUATION ALERT ---")
            logger.info(f"Entry condition met: 1-hour close > ${ENTRY_PRICE_THRESHOLD:,.0f} with volume >= {VOLUME_MULTIPLIER}x 20-period average.")
            
            if ENTRY_ZONE_LOW <= last_close <= ENTRY_ZONE_HIGH:
                logger.info(f"Price ${last_close:,.2f} is within designated entry zone (${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}).")
            else:
                logger.warning(f"Price ${last_close:,.2f} is outside designated entry zone (${ENTRY_ZONE_LOW:,.0f}-${ENTRY_ZONE_HIGH:,.0f}).")

            logger.info(f"Details: Timestamp={ts}, Close=${last_close:,.2f}, Volume={last_volume:,.0f}, Avg Volume={avg_volume:,.0f}")

            # Execute the trade
            logger.info("Executing BTC continuation trade...")
            breakout_type = f"continuation_{ENTRY_PRICE_THRESHOLD}"
            trade_success, trade_result = execute_btc_continuation_trade(cb_service, breakout_type, last_close)

            if trade_success:
                logger.info("BTC continuation trade executed successfully!")
                logger.info(f"Trade parameters: Margin=${BTC_BREAKOUT_MARGIN}, Leverage={BTC_BREAKOUT_LEVERAGE}x")
                logger.info(f"Stop Loss: ${BTC_BREAKOUT_STOP_LOSS:,.0f}, Take Profit: ${BTC_BREAKOUT_TAKE_PROFIT:,.0f}")
            else:
                logger.error(f"BTC continuation trade failed: {trade_result}")

            logger.info("")
            return ts

    except Exception as e:
        logger.error(f"Error in BTC continuation alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return last_alert_ts


def fartcoin_daily_alert(cb_service, last_alert_ts=None):
    PRODUCT_ID = "FARTCOIN-PERP-INTX"
    GRANULARITY = "ONE_HOUR"  # Get 1-hour candles to aggregate
    
    # Alert thresholds
    RECOVERY_THRESHOLD = 0.90
    SUPPORT_LOW = 0.78
    SUPPORT_HIGH = 0.80
    RSI_THRESHOLD = 30
    RSI_PERIOD = 14
    VOLUME_PERIOD = 15  # For volume confirmation
    CONSECUTIVE_CLOSES_NEEDED = 2  # Need 2 consecutive 6-hour closes for support confirmation

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
        
        logger.info(f"FARTCOIN Check (6H): Close=${close:.5f}, RSI={current_rsi:.2f}, Volume={current_volume:,.0f}")
        
        # Alert conditions
        alert_triggered = False
        
        # Alert 1: Recovery above $0.90 with RSI < 30 still intact
        if close >= RECOVERY_THRESHOLD and current_rsi < RSI_THRESHOLD:
            logger.info(f"--- FARTCOIN RECOVERY ALERT (6H) ---")
            logger.info(f"Recovery above ${RECOVERY_THRESHOLD} with RSI < {RSI_THRESHOLD} still intact!")
            logger.info(f"Timestamp: {ts}, Close: ${close:.5f}, RSI: {current_rsi:.2f}")
            alert_triggered = True
        
        # Alert 2: Hold at new support $0.78-$0.80 confirmed by two consecutive 6 hr closes and RSI divergence
        elif SUPPORT_LOW <= close <= SUPPORT_HIGH:
            # Check for two consecutive closes in support zone
            if len(six_hour_candles) >= CONSECUTIVE_CLOSES_NEEDED + 1:
                recent_closes = []
                recent_rsi_values = []
                
                # Get the last 3 candles for analysis (current + 2 previous)
                for i in range(1, min(4, len(six_hour_candles))):
                    candle_close = float(six_hour_candles[i]['close'])
                    recent_closes.append(candle_close)
                    if i < len(rsi):
                        recent_rsi_values.append(rsi.iloc[-i])
                
                # Check if we have enough data for RSI divergence analysis
                if len(recent_closes) >= 3 and len(recent_rsi_values) >= 3:
                    # Check for two consecutive closes in support zone
                    consecutive_support_closes = 0
                    for i in range(min(2, len(recent_closes))):
                        if SUPPORT_LOW <= recent_closes[i] <= SUPPORT_HIGH:
                            consecutive_support_closes += 1
                        else:
                            break
                    
                    # Check for RSI divergence (price makes lower low, RSI makes higher low)
                    price_lower_low = recent_closes[0] < recent_closes[2]  # Current close < 2 periods ago
                    rsi_higher_low = recent_rsi_values[0] > recent_rsi_values[2]  # Current RSI > 2 periods ago
                    
                    if consecutive_support_closes >= CONSECUTIVE_CLOSES_NEEDED and price_lower_low and rsi_higher_low:
                        logger.info(f"--- FARTCOIN SUPPORT ALERT (6H) ---")
                        logger.info(f"Hold at new support ${SUPPORT_LOW}-${SUPPORT_HIGH} confirmed!")
                        logger.info(f"Two consecutive closes in support zone: {consecutive_support_closes}")
                        logger.info(f"RSI divergence detected: Price lower low, RSI higher low")
                        logger.info(f"Timestamp: {ts}, Close: ${close:.5f}, RSI: {current_rsi:.2f}")
                        logger.info(f"Recent closes: {recent_closes[:3]}")
                        logger.info(f"Recent RSI values: {[f'{r:.2f}' for r in recent_rsi_values[:3]]}")
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


def execute_btc_continuation_trade(cb_service, continuation_type: str, entry_price: float):
    """
    Execute BTC continuation trade using trade_btc_perp.py functionality
    """
    try:
        logger.info(f"Executing BTC continuation trade: {continuation_type} at ${entry_price:,.2f}")
        
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
            logger.info("BTC continuation trade executed successfully!")
            logger.info(f"Trade output: {result.stdout}")
            return True, result.stdout
        else:
            logger.error(f"BTC continuation trade failed!")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error executing BTC continuation trade: {e}")
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
    btc_continuation_last_alert_ts = None
    fartcoin_last_alert_ts = None
    
    while True:
        try:
            # BTC continuation alert
            btc_continuation_last_alert_ts = btc_continuation_alert(cb_service, btc_continuation_last_alert_ts)

            # FARTCOIN daily alert (runs hourly but condition only changes daily)
            # fartcoin_last_alert_ts = fartcoin_daily_alert(cb_service, fartcoin_last_alert_ts)

            # Wait 5 minutes until next poll
            wait_seconds = 300  # 5 minutes
            logger.info(f"Waiting {wait_seconds} seconds until next poll")
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