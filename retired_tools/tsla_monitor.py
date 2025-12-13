#!/usr/bin/env python3
"""
TSLA Monitor - Real-time trade management rule evaluation
Monitors TSLA during US market hours and triggers alerts based on price/volume conditions
"""

import requests
import pandas as pd
import schedule
import time
import logging
from datetime import datetime, time as dt_time
import pytz
from config import ALPHA_VANTAGE_API_KEY

# Configuration constants
STOP_LEVEL = 305
TARGET_LEVEL = 330
VOL_MULT = 1.5
MARKET_OPEN = dt_time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = dt_time(16, 0)  # 4:00 PM ET
MONITOR_END = dt_time(20, 5)   # 8:05 PM ET

# Alpha Vantage API
BASE_URL = "https://www.alphavantage.co/query"
SYMBOL = "TSLA"

# Global state tracking
last_state = {
    'STOP_HIT': False,
    'CLOSE_ABOVE_320': False,
    'FAILED_BREAKOUT': False,
    'TARGET_HIT': False
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler('tsla_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_current_time_et():
    """Get current time in Eastern Time"""
    et_tz = pytz.timezone('US/Eastern')
    return datetime.now(et_tz)

def is_market_hours():
    """Check if current time is during market hours"""
    current_time = get_current_time_et()
    current_time_only = current_time.time()
    return MARKET_OPEN <= current_time_only <= MONITOR_END

def fetch_tsla_data():
    """Fetch TSLA intraday data from Alpha Vantage"""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": SYMBOL,
        "interval": "1min",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Time Series (1min)" not in data:
            logger.error(f"API Error: {data.get('Note', 'Unknown error')}")
            return None
            
        df = pd.DataFrame(data["Time Series (1min)"]).T
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def fetch_daily_data():
    """Fetch TSLA daily data for volume comparison"""
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": SYMBOL,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            return None
            
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }, inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching daily data: {e}")
        return None

def evaluate_rules():
    """Evaluate all trade management rules"""
    if not is_market_hours():
        return
    
    current_time = get_current_time_et()
    current_time_only = current_time.time()
    
    # Fetch data
    intraday_data = fetch_tsla_data()
    if intraday_data is None or len(intraday_data) == 0:
        return
    
    daily_data = fetch_daily_data()
    if daily_data is None or len(daily_data) < 20:
        return
    
    latest_price = intraday_data['Close'].iloc[-1]
    today_high = intraday_data['High'].iloc[-1]
    
    # Get today's daily data
    today_date = current_time.date()
    today_daily = daily_data[daily_data.index.date == today_date]
    
    if len(today_daily) == 0:
        return
    
    today_close = today_daily['Close'].iloc[-1]
    today_volume = today_daily['Volume'].iloc[-1]
    
    # Calculate 20-day average volume
    avg_volume = daily_data['Volume'].tail(20).mean()
    
    # Rule evaluations
    current_state = {}
    
    # STOP_HIT: last price ≤ STOP_LEVEL
    current_state['STOP_HIT'] = latest_price <= STOP_LEVEL
    
    # CLOSE_ABOVE_320: today's daily close ≥ 320 AND today's volume ≥ 1.5 × 20-day avg vol
    # Only check after 16:01 ET
    if current_time_only >= dt_time(16, 1):
        current_state['CLOSE_ABOVE_320'] = (today_close >= 320 and 
                                           today_volume >= VOL_MULT * avg_volume)
    else:
        current_state['CLOSE_ABOVE_320'] = False
    
    # FAILED_BREAKOUT: today's high ≥ 320 AND (latest price or daily close) < 312
    current_state['FAILED_BREAKOUT'] = (today_high >= 320 and 
                                       (latest_price < 312 or today_close < 312))
    
    # TARGET_HIT: last price ≥ TARGET_LEVEL
    current_state['TARGET_HIT'] = latest_price >= TARGET_LEVEL
    
    # Check for state changes and log triggers
    for rule, current_value in current_state.items():
        if current_value and not last_state[rule]:
            timestamp = current_time.strftime('%Y-%m-%d %H:%M ET')
            logger.info(f"[{timestamp}] {rule} triggered (price {latest_price:.2f})")
            last_state[rule] = True

def main():
    """Main function - schedule and run the monitor"""
    logger.info("TSLA Monitor starting...")
    
    # Schedule the evaluation every minute
    schedule.every().minute.do(evaluate_rules)
    
    # Run until monitor end time
    while True:
        current_time = get_current_time_et()
        
        if current_time.time() >= MONITOR_END:
            logger.info("TSLA Monitor stopping (end time reached)")
            break
        
        schedule.run_pending()
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    main() 