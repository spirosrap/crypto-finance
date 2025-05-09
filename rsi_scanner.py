import requests
import pandas as pd
import numpy as np
import time

from config import ALPHA_VANTAGE_API_KEY
BASE_URL = "https://www.alphavantage.co/query"

stock_list = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC',
    'JPM', 'BAC', 'WFC', 'V', 'MA', 'PYPL', 'BRK.B', 'UNH', 'JNJ', 'PFE',
    'XOM', 'CVX', 'COP', 'T', 'VZ', 'KO', 'PEP', 'MCD', 'WMT', 'COST',
    'DIS', 'NKE', 'HD', 'LOW', 'BA', 'CRM', 'ABBV', 'TMO', 'MRK', 'LLY',
    'ORCL', 'IBM', 'QCOM', 'TXN', 'AVGO', 'ADBE', 'GE', 'GS', 'BKNG', 'SBUX'
]
rsi_threshold = 30
volume_filter = 1.5
atr_pct_threshold = 0.2

def fetch_daily_data(symbol):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    try:
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'  # <- This is correct for TIME_SERIES_DAILY
        }, inplace=True)        
        return df
    except KeyError:
        print(f"Data fetch error for {symbol}: {data.get('Note', 'Unknown error')}")
        return None

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

results = []

for symbol in stock_list:
    print(f"Scanning {symbol}...")
    df = fetch_daily_data(symbol)
    if df is None or len(df) < 20:
        continue

    df['RSI'] = compute_rsi(df['Close'])
    df['ATR'] = compute_atr(df['High'], df['Low'], df['Close'])
    df['ATR%'] = (df['ATR'] / df['Close']) * 100
    df['AvgVolume'] = df['Volume'].rolling(window=20).mean()
    df['RelVolume'] = df['Volume'] / df['AvgVolume']

    latest = df.iloc[-1]
    if (
        latest['RSI'] < rsi_threshold and
        latest['RelVolume'] > volume_filter and
        latest['ATR%'] > atr_pct_threshold
    ):
        results.append({
            'Symbol': symbol,
            'RSI': round(latest['RSI'], 2),
            'ATR%': round(latest['ATR%'], 2),
            'RelVolume': round(latest['RelVolume'], 2),
            'Close': round(latest['Close'], 2),
        })

    time.sleep(15)  # Alpha Vantage: 5 API calls per minute (free tier)

results_df = pd.DataFrame(results)
print("\n=== Alpha Vantage RSI Scanner Results ===")
print(results_df if not results_df.empty else "No setups found.")