# Simplified Trading Bot
# Single coin (BTC-USDC), single timeframe (5-min), single logic (RSI + EMA + volume)
# No AI prompts, no ML classifiers, no market regimes

from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from datetime import datetime, timedelta, UTC
import pandas as pd
from config import API_KEY_PERPS, API_SECRET_PERPS

# Parameters
PAIR = "BTC-USDC"
GRANULARITY = "FIVE_MINUTE"
RSI_THRESHOLD = 30
VOLUME_LOOKBACK = 20
TP_PERCENT = 0.015
SL_PERCENT = 0.007

# Services
cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
ta = TechnicalAnalysis(cb)

# Fetch candles
def fetch_candles():
    now = datetime.now(UTC)
    start = now - timedelta(minutes=5 * 200)
    df = cb.historical_data.get_historical_data(PAIR, start, now, GRANULARITY)
    df = pd.DataFrame(df)
    
    # Convert string columns to numeric
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Analyze market
def analyze(df: pd.DataFrame):
    # Convert DataFrame to list of dictionaries for the technical analysis methods
    candles = df.to_dict('records')
    
    # Calculate RSI
    rsi = ta.compute_rsi(PAIR, candles, period=14)
    
    # Calculate EMA
    ema_50 = ta.get_moving_average(candles, period=50, ma_type='ema')
    
    # Get current values
    current = df.iloc[-1]
    avg_volume = df["volume"].tail(VOLUME_LOOKBACK).mean()

    if (
        rsi < RSI_THRESHOLD
        and current["close"] > ema_50
        and current["volume"] > avg_volume
    ):
        return True, current["close"]
    return False, None

# Execute logic
if __name__ == "__main__":
    candles = fetch_candles()
    signal, entry = analyze(candles)
    if signal:
        tp = entry * (1 + TP_PERCENT)
        sl = entry * (1 - SL_PERCENT)
        print(f"[SIGNAL] BUY {PAIR} at {entry:.2f} | TP: {tp:.2f}, SL: {sl:.2f}")
    else:
        print("[NO SIGNAL] Conditions not met.")
