#!/usr/bin/env python3
# ATR Analysis for BTC-INTX-PERP
# Calculates ATR% percentiles for volatility threshold determination

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import logging
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY_PERPS, API_SECRET_PERPS
import pandas_ta as ta
import yfinance as yf
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_perp_product(product_id):
    """Convert spot product ID to perpetual futures product ID"""
    perp_map = {
        'BTC-USDC': 'BTC-PERP-INTX',
        'ETH-USDC': 'ETH-PERP-INTX',
        'DOGE-USDC': 'DOGE-PERP-INTX',
        'SOL-USDC': 'SOL-PERP-INTX',
        'SHIB-USDC': '1000SHIB-PERP-INTX'
    }
    return perp_map.get(product_id, 'BTC-PERP-INTX')

def fetch_historical_data(cb, product_id, days=30):
    """Fetch historical data for the specified product over the given number of days"""
    now = datetime.now(UTC)
    start = now - timedelta(days=days)
    end = now
    
    # Use 5-minute candles for more granular data
    granularity = "FIVE_MINUTE"
    
    # Convert to perpetual futures product ID if needed
    if not product_id.endswith('-PERP-INTX'):
        perp_product = get_perp_product(product_id)
        logger.info(f"Converting {product_id} to perpetual futures product ID: {perp_product}")
        product_id = perp_product
    
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, granularity)
    
    # Check if we got any data
    if not raw_data:
        logger.error(f"No data returned for {product_id}")
        return pd.DataFrame()
    
    df = pd.DataFrame(raw_data)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle timestamp - convert Unix timestamp to datetime
    if 'start' in df.columns:
        df['start'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s', utc=True)        
        df.set_index('start', inplace=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
    
    return df

def calculate_atr_percent(df, ta, product_id, period=14):
    """Calculate ATR% for the dataset"""
    if df.empty:
        logger.error("Cannot calculate ATR% for empty DataFrame")
        return df
    
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    # Calculate ATR
    atr = ta.compute_atr(candles, period=period)
    
    # Calculate ATR% (ATR as percentage of price)
    df['atr'] = atr
    df['atr_percent'] = (df['atr'] / df['close']) * 100
    
    return df

def calculate_percentiles(df, percentiles=[70, 90]):
    """Calculate specified percentiles of ATR%"""
    if df.empty or 'atr_percent' not in df.columns:
        logger.error("Cannot calculate percentiles: DataFrame is empty or missing ATR% column")
        return {}
    
    results = {}
    for p in percentiles:
        results[f'percentile_{p}'] = np.percentile(df['atr_percent'], p)
    
    return results

def check_atr_expansion(symbol: str = "BTC-USD", lookback: int = 5) -> Tuple[bool, float, float]:
    """
    Check if the ATR (Average True Range) is expanding by comparing current value with historical value.
    
    Args:
        symbol (str): Trading pair symbol (default: "BTC-USD")
        lookback (int): Number of periods to look back for comparison (default: 5)
    
    Returns:
        Tuple[bool, float, float]: A tuple containing:
            - Boolean indicating if ATR is expanding
            - Current ATR value
            - Historical ATR value (lookback periods ago)
    """
    # Download hourly data
    df = yf.download(symbol, interval="1h", period="1mo")
    
    # Calculate 14-period ATR
    df['ATR'] = df.ta.atr(length=14)
    
    # Get current and historical ATR values
    current_atr = df['ATR'].iloc[-1]
    historical_atr = df['ATR'].iloc[-(lookback + 1)]
    
    # Check if ATR is expanding
    is_expanding = current_atr > historical_atr
    
    return is_expanding, current_atr, historical_atr

def main():
    # Initialize services
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    ta = TechnicalAnalysis(cb)
    
    # Product to analyze - use BTC-USDC which will be converted to BTC-PERP-INTX
    product_id = 'BTC-USDC'
    
    # Fetch historical data (30 days)
    logger.info(f"Fetching historical data for {product_id} over the last 30 days...")
    df = fetch_historical_data(cb, product_id, days=30)
    
    if df.empty:
        logger.error("Failed to fetch historical data. Exiting.")
        return
    
    # Calculate ATR%
    logger.info("Calculating ATR%...")
    df = calculate_atr_percent(df, ta, product_id)
    
    if df.empty or 'atr_percent' not in df.columns:
        logger.error("Failed to calculate ATR%. Exiting.")
        return
    
    # Calculate percentiles
    logger.info("Calculating percentiles...")
    percentiles = calculate_percentiles(df)
    
    if not percentiles:
        logger.error("Failed to calculate percentiles. Exiting.")
        return
    
    # Print results
    logger.info("\n=== ATR% Analysis Results ===")
    logger.info(f"Product: {product_id}")
    logger.info(f"Time period: Last 30 days")
    logger.info(f"Number of data points: {len(df)}")
    logger.info(f"70th percentile: {percentiles['percentile_70']:.4f}%")
    logger.info(f"90th percentile: {percentiles['percentile_90']:.4f}%")
    
    logger.info("\n=== Suggested Volatility Thresholds ===")
    logger.info(f"strong_threshold = {percentiles['percentile_90']:.4f}  # 90th percentile")
    logger.info(f"moderate_threshold = {percentiles['percentile_70']:.4f}  # 70th percentile")
    
    # Save results to CSV for further analysis
    df.to_csv(f"{product_id}_atr_analysis.csv")
    logger.info(f"\nDetailed data saved to {product_id}_atr_analysis.csv")

if __name__ == "__main__":
    # Example usage
    is_expanding, current_atr, historical_atr = check_atr_expansion()
    
    print(f"BTC/USD ATR Analysis:")
    print(f"Current ATR: {current_atr:.2f}")
    print(f"ATR 5 periods ago: {historical_atr:.2f}")
    print(f"ATR is expanding: {is_expanding}")
    
    main() 