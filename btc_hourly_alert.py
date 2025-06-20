import time
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas as pd
import pandas_ta as ta
import subprocess
import sys

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
BTC_BREAKOUT_TAKE_PROFIT = 112000  # First profit target at $112,000

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


def get_recent_4h_candles(cb_service, num_candles=48):
    """Get 4-hour candles by aggregating hourly data"""
    now = datetime.now(UTC)
    now = now.replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=num_candles)
    end = now
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())
    logger.info(f"Fetching last {num_candles} hourly candles for 4h aggregation from {start} to {end}")
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


def process_candle(candle):
    return {
        'timestamp': datetime.fromtimestamp(int(candle['start']), UTC),
        'open': float(candle['open']),
        'high': float(candle['high']),
        'low': float(candle['low']),
        'close': float(candle['close']),
        'volume': float(candle['volume'])
    }

def aggregate_to_4h_candles(hourly_candles):
    """Aggregate hourly candles to 4-hour candles"""
    if len(hourly_candles) < 4:
        return []
    
    # Reverse to get chronological order
    hourly_candles = hourly_candles[::-1]
    
    four_hour_candles = []
    for i in range(0, len(hourly_candles) - 3, 4):
        if i + 3 < len(hourly_candles):
            group = hourly_candles[i:i+4]
            
            # Aggregate OHLCV
            open_price = float(group[0]['open'])
            high_price = max(float(c['high']) for c in group)
            low_price = min(float(c['low']) for c in group)
            close_price = float(group[-1]['close'])
            volume = sum(float(c['volume']) for c in group)
            
            four_hour_candle = {
                'start': group[0]['start'],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            four_hour_candles.append(four_hour_candle)
    
    return four_hour_candles[::-1]  # Reverse back to newest first

def btc_intraday_rsi_alert(cb_service, last_alert_ts=None):
    """
    Alerts when BTC price is near the intraday low with an oversold RSI.
    Entry: ~$104,000 (near current intraday low with RSI oversold conditions)
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    TARGET_PRICE = 104000
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    INTRADAY_LOW_PROXIMITY_PERCENT = 0.5  # Within 0.5% of intraday low

    try:
        # 1. Get candles for analysis (RSI period + 24h for intraday low)
        candles_raw = get_recent_hourly_candles(cb_service, num_candles=RSI_PERIOD + 24)
        if not candles_raw or len(candles_raw) < RSI_PERIOD + 1:
            logger.warning("Not enough BTC hourly candle data for RSI alert.")
            return last_alert_ts

        # 2. Prepare data for analysis
        last_closed_candle_raw = candles_raw[1]
        ts = datetime.fromtimestamp(int(last_closed_candle_raw['start']), UTC)

        # Avoid re-alerting for the same candle
        if ts == last_alert_ts:
            return last_alert_ts
        
        # Reverse historical candles for correct indicator calculation order
        historical_candles = candles_raw[1:]
        historical_candles.reverse()
        
        # Convert list of Candle objects to list of dicts for DataFrame compatibility
        historical_candles_dicts = [
            {'start': c['start'], 'open': c['open'], 'high': c['high'], 'low': c['low'], 'close': c['close'], 'volume': c['volume']} 
            for c in historical_candles
        ]

        df = pd.DataFrame(historical_candles_dicts)
        df['close'] = pd.to_numeric(df['close'])
        df['low'] = pd.to_numeric(df['low'])

        # 3. Calculate indicators and intraday low
        df.ta.rsi(length=RSI_PERIOD, append=True)
        
        latest_stats = df.iloc[-1]
        last_close = latest_stats['close']
        last_rsi = latest_stats[f'RSI_{RSI_PERIOD}']
        
        # Intraday low is the min of the last 24 closed candles
        intraday_low = df['low'][-24:].min()
        
        logger.info(f"BTC Check: Last Close=${last_close:,.2f}, Intraday Low=${intraday_low:,.2f}, RSI({RSI_PERIOD})={last_rsi:.2f}")

        # 4. Check alert conditions
        is_near_target = last_close <= TARGET_PRICE
        is_near_low = last_close <= (intraday_low * (1 + INTRADAY_LOW_PROXIMITY_PERCENT / 100))
        is_rsi_oversold = last_rsi < RSI_OVERSOLD
        
        if is_near_target and is_near_low and is_rsi_oversold:
            logger.info(f"--- BTC ALERT ---")
            logger.info(f"Entry condition met: Price near ${TARGET_PRICE:,.0f}, near intraday low, and RSI oversold.")
            logger.info(f"Details: Timestamp={ts}, Close=${last_close:,.2f}, Intraday Low=${intraday_low:,.2f}, RSI={last_rsi:.2f}")
            logger.info("")  # Empty line for visual separation
            return ts  # Update last alert timestamp

    except Exception as e:
        logger.error(f"Error in BTC alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return last_alert_ts


def fartcoin_daily_alert(cb_service, last_alert_ts=None):
    PRODUCT_ID = "FARTCOIN-PERP-INTX"
    GRANULARITY = "ONE_DAY"
    BREAKOUT_THRESHOLD = 1.10
    BREAKOUT_VOLUME_MULTIPLIER = 1.2  # 20% above average

    try:
        now = datetime.now(UTC)
        now = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = now - timedelta(days=21)
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
        if not candles or len(candles) < 2:
            logger.warning("Not enough FARTCOIN daily candle data.")
            return last_alert_ts
        
        last_candle = candles[1]
        ts = datetime.fromtimestamp(int(last_candle['start']), UTC)

        if ts == last_alert_ts:
            return last_alert_ts

        session_candles = candles[1:21]
        session_volumes = [float(c['volume']) for c in session_candles]
        avg_volume = sum(session_volumes) / 20 if len(session_volumes) == 20 else sum(session_volumes) / len(session_volumes)
        close = float(last_candle['close'])
        volume = float(last_candle['volume'])
        
        logger.info(f"FARTCOIN Check: Last Daily Close=${close:.5f}, Volume={volume:,.0f}, Avg Volume={avg_volume:,.0f}")
        
        if close > BREAKOUT_THRESHOLD and volume >= BREAKOUT_VOLUME_MULTIPLIER * avg_volume:
            logger.info(f"--- FARTCOIN ALERT ---")
            logger.info(f"Daily close > {BREAKOUT_THRESHOLD} and volume ≥ 20% above 20-day average!")
            logger.info(f"Timestamp: {ts}, Close: {close}, Volume: {volume:,.0f}, Avg Volume: {avg_volume:,.0f}")
            logger.info("")  # Empty line for visual separation
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


def btc_breakout_alert(cb_service, last_alert_ts=None):
    """
    Alerts when BTC breaks out above $106,100 (4-hour close) or above $108,000-$109,000 resistance zone with high volume.
    Entry: Buy at breakout above $106,100 (4‑h close) or above $108,000–109,000 resistance zone with high volume.
    """
    PRODUCT_ID = "BTC-PERP-INTX"
    BREAKOUT_LEVEL_1 = 106100  # 4-hour close breakout
    RESISTANCE_LOW = 108000    # Lower resistance zone
    RESISTANCE_HIGH = 109000   # Upper resistance zone
    VOLUME_MULTIPLIER = 1.5    # 50% above average volume for breakout confirmation

    try:
        # 1. Get hourly candles for 4-hour aggregation
        hourly_candles_raw = get_recent_4h_candles(cb_service, num_candles=48)
        if not hourly_candles_raw or len(hourly_candles_raw) < 8:
            logger.warning("Not enough BTC hourly candle data for 4h breakout alert.")
            return last_alert_ts

        # 2. Aggregate to 4-hour candles
        four_hour_candles = aggregate_to_4h_candles(hourly_candles_raw)
        if len(four_hour_candles) < 2:
            logger.warning("Not enough 4-hour candles for breakout analysis.")
            return last_alert_ts

        # 3. Get the last completed 4-hour candle
        last_4h_candle = four_hour_candles[1]  # Second newest (last completed)
        ts = datetime.fromtimestamp(int(last_4h_candle['start']), UTC)

        # Avoid re-alerting for the same candle
        if ts == last_alert_ts:
            return last_alert_ts

        close_4h = float(last_4h_candle['close'])
        volume_4h = float(last_4h_candle['volume'])
        
        # 4. Calculate average volume for comparison
        recent_volumes = [float(c['volume']) for c in four_hour_candles[1:7]]  # Last 6 completed 4h candles
        avg_volume = sum(recent_volumes) / len(recent_volumes) if recent_volumes else volume_4h
        
        # 5. Check breakout conditions
        breakout_level_1 = close_4h > BREAKOUT_LEVEL_1
        breakout_resistance_zone = RESISTANCE_LOW <= close_4h <= RESISTANCE_HIGH
        high_volume = volume_4h >= (avg_volume * VOLUME_MULTIPLIER)
        
        logger.info(f"BTC Breakout Check: 4h Close=${close_4h:,.2f}, Volume={volume_4h:,.0f}, Avg Volume={avg_volume:,.0f}")
        logger.info(f"Conditions: Level1={breakout_level_1}, Resistance={breakout_resistance_zone}, HighVolume={high_volume}")

        # 6. Alert conditions and trade execution
        if (breakout_level_1 or breakout_resistance_zone) and high_volume:
            logger.info(f"--- BTC BREAKOUT ALERT ---")
            
            # Determine breakout type for logging
            if breakout_level_1:
                breakout_type = f"4h_close_above_{BREAKOUT_LEVEL_1}"
                logger.info(f"4-hour close breakout above ${BREAKOUT_LEVEL_1:,.0f} with high volume!")
            if breakout_resistance_zone:
                breakout_type = f"resistance_zone_{RESISTANCE_LOW}_{RESISTANCE_HIGH}"
                logger.info(f"Breakout in resistance zone ${RESISTANCE_LOW:,.0f}-${RESISTANCE_HIGH:,.0f} with high volume!")
            
            logger.info(f"Details: Timestamp={ts}, 4h Close=${close_4h:,.2f}, Volume={volume_4h:,.0f}, Avg Volume={avg_volume:,.0f}")
            
            # Execute the trade
            logger.info("Executing BTC breakout trade...")
            trade_success, trade_result = execute_btc_breakout_trade(cb_service, breakout_type, close_4h)
            
            if trade_success:
                logger.info("BTC breakout trade executed successfully!")
                logger.info(f"Trade parameters: Margin=${BTC_BREAKOUT_MARGIN}, Leverage={BTC_BREAKOUT_LEVERAGE}x")
                logger.info(f"Stop Loss: ${BTC_BREAKOUT_STOP_LOSS:,.0f}, Take Profit: ${BTC_BREAKOUT_TAKE_PROFIT:,.0f}")
            else:
                logger.error(f"BTC breakout trade failed: {trade_result}")
            
            logger.info("")  # Empty line for visual separation
            return ts  # Update last alert timestamp

    except Exception as e:
        logger.error(f"Error in BTC breakout alert logic: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return last_alert_ts


def main():
    logger.info("Starting multi-asset alert script")
    logger.info("")  # Empty line for visual separation
    cb_service = setup_coinbase()
    btc_last_alert_ts = None
    btc_breakout_last_alert_ts = None
    fartcoin_last_alert_ts = None
    
    while True:
        try:
            # BTC hourly alert logic
            btc_last_alert_ts = btc_intraday_rsi_alert(cb_service, btc_last_alert_ts)

            # BTC breakout alert logic
            btc_breakout_last_alert_ts = btc_breakout_alert(cb_service, btc_breakout_last_alert_ts)

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