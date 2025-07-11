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

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Connection retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5  # seconds
MAX_RETRY_DELAY = 60  # seconds
BACKOFF_MULTIPLIER = 2

# Trade parameters from the table
ETH_ENTRY_ZONE_LOW = 2780     # Lower entry zone
ETH_ENTRY_ZONE_HIGH = 2810    # Upper entry zone (retest of triangle apex)
ETH_STOP_LOSS = 2680          # Below apex & 20-EMA
ETH_FIRST_TARGET = 3100       # Triangle measured move
ETH_MARGIN = 300              # USD margin
ETH_LEVERAGE = 20             # 20x leverage
PRODUCT_ID = "ETH-PERP-INTX"

# Monitoring configuration
CHECK_INTERVAL = 30  # seconds between checks
TRADE_STATE_FILE = "eth_triangle_retest_trade_state.json"

# Trade tracking variables
trade_taken = False
last_alert_time = None

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
    """
    CONNECTION_ERRORS = (
        ConnectionError,
        TimeoutError,
        OSError,
        Exception
    )
    
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

def play_alert_sound(filename="alert_sound.wav"):
    """Play the alert sound using system commands"""
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
            test_response = service.client.get_public_candles(
                product_id=PRODUCT_ID,
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

def load_trade_state():
    """Load trade state from file"""
    if os.path.exists(TRADE_STATE_FILE):
        try:
            with open(TRADE_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {"trade_taken": False, "trade_timestamp": None}
    return {"trade_taken": False, "trade_timestamp": None}

def save_trade_state(state):
    """Save trade state to file"""
    try:
        with open(TRADE_STATE_FILE, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        logger.error(f"Failed to save trade state: {e}")

def get_current_eth_price(cb_service):
    """Get current ETH price with retry logic"""
    def _get_price():
        trades = cb_service.client.get_market_trades(product_id=PRODUCT_ID, limit=1)
        return float(trades['trades'][0]['price'])
    
    return retry_with_backoff(_get_price)

def execute_eth_trade(cb_service, current_price: float):
    """Execute ETH trade using trade_btc_perp.py"""
    def _execute_trade():
        print(f"🚀 Executing ETH Triangle Retest trade at ${current_price:,.2f}")
        print(f"💼 Trade params: Margin=${ETH_MARGIN}, Leverage={ETH_LEVERAGE}x")
        
        # Calculate position size based on margin and leverage
        position_size_usd = ETH_MARGIN * ETH_LEVERAGE
        
        # Use subprocess to call trade_btc_perp.py
        cmd = [
            sys.executable, 'trade_btc_perp.py',
            '--product', PRODUCT_ID,
            '--side', 'BUY',
            '--size', str(position_size_usd),
            '--leverage', str(ETH_LEVERAGE),
            '--tp', str(ETH_FIRST_TARGET),
            '--sl', str(ETH_STOP_LOSS),
            '--no-confirm'  # Skip confirmation for automated trading
        ]
        
        print(f"⚡ Executing trade command...")
        
        # Execute the trade command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ ETH Triangle Retest trade executed successfully!")
            print(f"📋 Trade output:\n{result.stdout}")
            return True, result.stdout
        else:
            print("❌ ETH Triangle Retest trade failed!")
            print(f"📋 Error output:\n{result.stderr}")
            return False, result.stderr
    
    try:
        result = retry_with_backoff(_execute_trade)
        if result is None:
            return False, "Failed after multiple retry attempts"
        return result
    except subprocess.TimeoutExpired:
        print("⏰ Trade execution timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ Error executing ETH trade: {e}")
        return False, str(e)

def check_eth_triangle_retest_conditions(cb_service):
    """
    Check if ETH price is in the entry zone for triangle retest
    Entry: $2,780 - $2,810 (retest of triangle apex)
    Stop-loss: $2,680 (below apex & 20-EMA)
    First target: $3,100 (triangle measured move)
    """
    global trade_taken, last_alert_time
    
    # Load trade state
    trade_state = load_trade_state()
    if trade_state["trade_taken"]:
        print("\n⚠️  Trade already taken, skipping further checks\n")
        return False
    
    # Get current price
    current_price = get_current_eth_price(cb_service)
    if current_price is None:
        print("\n❌ Failed to get current ETH price\n")
        return False
    
    # Create a clean status report
    current_time = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    print(f"\n{'='*80}")
    print(f"📊 ETH TRIANGLE RETEST MONITOR - {current_time}")
    print(f"{'='*80}")
    print(f"💰 Current ETH Price:     ${current_price:>10,.2f}")
    print(f"📈 Entry Zone:            ${ETH_ENTRY_ZONE_LOW:>10,.0f} - ${ETH_ENTRY_ZONE_HIGH:,.0f}")
    print(f"🛑 Stop Loss:             ${ETH_STOP_LOSS:>10,.0f}")
    print(f"🎯 First Target:          ${ETH_FIRST_TARGET:>10,.0f}")
    print(f"💼 Position Size:         ${ETH_MARGIN * ETH_LEVERAGE:>10,.0f} ({ETH_LEVERAGE}x leverage)")
    print(f"{'='*80}")
    
    # Check if price is in entry zone
    if ETH_ENTRY_ZONE_LOW <= current_price <= ETH_ENTRY_ZONE_HIGH:
        print(f"🔥 ALERT: ETH PRICE IN ENTRY ZONE!")
        print(f"🚀 EXECUTING TRADE...")
        print(f"{'='*80}\n")
        
        # Play alert sound
        play_alert_sound()
        
        # Rate limit alerts to avoid spam
        current_time_obj = datetime.now(UTC)
        if last_alert_time is None or (current_time_obj - last_alert_time).total_seconds() > 300:  # 5 minutes
            last_alert_time = current_time_obj
            
            # Execute the trade
            success, output = execute_eth_trade(cb_service, current_price)
            
            if success:
                # Mark trade as taken
                trade_state = {
                    "trade_taken": True,
                    "trade_timestamp": current_time_obj.isoformat(),
                    "entry_price": current_price,
                    "trade_output": output
                }
                save_trade_state(trade_state)
                trade_taken = True
                
                print(f"\n🎉 ETH TRIANGLE RETEST TRADE COMPLETED!")
                print(f"{'='*80}")
                print(f"✅ Entry Price:           ${current_price:>10,.2f}")
                print(f"📊 Position Size:         ${ETH_MARGIN * ETH_LEVERAGE:>10,.0f}")
                print(f"🎯 Target:                ${ETH_FIRST_TARGET:>10,.0f}")
                print(f"🛑 Stop Loss:             ${ETH_STOP_LOSS:>10,.0f}")
                print(f"{'='*80}\n")
                
                return True
            else:
                print(f"\n❌ TRADE EXECUTION FAILED: {output}\n")
                return False
    else:
        # Calculate and display distance to entry zone
        if current_price < ETH_ENTRY_ZONE_LOW:
            distance = ETH_ENTRY_ZONE_LOW - current_price
            distance_pct = (distance / current_price) * 100
            print(f"📉 Status: ETH ${distance:,.0f} ({distance_pct:.2f}%) BELOW entry zone")
        else:
            distance = current_price - ETH_ENTRY_ZONE_HIGH
            distance_pct = (distance / current_price) * 100
            print(f"📈 Status: ETH ${distance:,.0f} ({distance_pct:.2f}%) ABOVE entry zone")
        
        print(f"⏳ Waiting for triangle retest entry zone...")
        print(f"{'='*80}\n")
    
    return False

def main():
    """Main monitoring loop"""
    print("\n" + "="*80)
    print("🔺 ETH TRIANGLE RETEST MONITOR")
    print("="*80)
    print(f"📈 Entry Zone:            ${ETH_ENTRY_ZONE_LOW:>10,.0f} - ${ETH_ENTRY_ZONE_HIGH:,.0f}")
    print(f"🛑 Stop Loss:             ${ETH_STOP_LOSS:>10,.0f}")
    print(f"🎯 First Target:          ${ETH_FIRST_TARGET:>10,.0f}")
    print(f"💰 Margin:                ${ETH_MARGIN:>10,.0f}")
    print(f"⚡ Leverage:              {ETH_LEVERAGE:>10}x")
    print(f"💼 Position Size:         ${ETH_MARGIN * ETH_LEVERAGE:>10,.0f}")
    print(f"⏰ Check Interval:        {CHECK_INTERVAL:>10} seconds")
    print("="*80)
    print("🔄 Initializing connection...")
    
    try:
        # Setup Coinbase connection
        cb_service = setup_coinbase()
        print("✅ Coinbase service initialized successfully")
        print("🔍 Starting ETH price monitoring...\n")
        
        check_count = 0
        while True:
            try:
                check_count += 1
                
                # Check trade conditions
                trade_executed = check_eth_triangle_retest_conditions(cb_service)
                
                if trade_executed:
                    print("🎯 Trade executed successfully! Exiting monitor...")
                    break
                
                # Wait before next check with countdown
                print(f"⏰ Check #{check_count} completed. Next check in {CHECK_INTERVAL} seconds...")
                
                # Show countdown for better user experience
                for remaining in range(CHECK_INTERVAL, 0, -5):
                    if remaining <= 10:
                        print(f"⏳ Next check in {remaining} seconds...", end='\r')
                        time.sleep(1)
                    else:
                        time.sleep(5)
                
                print(" " * 50, end='\r')  # Clear countdown line
                
            except KeyboardInterrupt:
                print("\n🛑 Monitor stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                print(f"🔄 Retrying in {CHECK_INTERVAL} seconds...\n")
                time.sleep(CHECK_INTERVAL)
                
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        return False
    
    print("\n👋 ETH Triangle Retest Monitor finished")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    main() 