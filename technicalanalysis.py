import numpy as np
import pandas as pd
from typing import List, Tuple
from coinbaseservice import CoinbaseService

class TechnicalAnalysis:
    def __init__(self, coinbase_service: CoinbaseService):
        self.coinbase_service = coinbase_service

    def compute_rsi(self, product_id, candles, period=14):
        prices = [float(candle['close']) for candle in candles]
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi[-1]

    def identify_trend(self, product_id, candles, window=20):
        prices = [float(candle['close']) for candle in candles]
        if len(prices) < window:
            return "Not enough data"
        
        # Calculate Simple Moving Average
        sma = np.convolve(prices, np.ones(window), 'valid') / window
        
        # Calculate the slope of the SMA
        slope = np.gradient(sma)
        
        # Determine the trend based on the recent slope
        recent_slope = slope[-5:].mean()  # Use the average of the last 5 slope values
        
        if recent_slope > 0.01:  # Threshold can be adjusted
            return "Uptrend"
        elif recent_slope < -0.01:  # Threshold can be adjusted
            return "Downtrend"
        else:
            return "Sideways"

    def compute_macd(self, product_id, candles):
        prices = [float(candle['close']) for candle in candles]
        print(f"length of prices: {len(prices)}")
        ema12 = self.exponential_moving_average(prices, 12)
        ema26 = self.exponential_moving_average(prices, 26)
        macd = ema12 - ema26
        signal = self.exponential_moving_average(macd, 9)
        histogram = macd - signal
        return macd[-1], signal[-1], histogram[-1]



    def exponential_moving_average(self, data, span):
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    def compute_bollinger_bands(self, candles: List[dict], window: int = 20, num_std: float = 2) -> Tuple[float, float, float]:
        prices = [float(candle['close']) for candle in candles]
        prices_series = pd.Series(prices)
        
        rolling_mean = prices_series.rolling(window=window).mean()
        rolling_std = prices_series.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band.iloc[-1], rolling_mean.iloc[-1], lower_band.iloc[-1]

    def generate_bollinger_bands_signal(self, candles: List[dict]) -> str:
        upper_band, middle_band, lower_band = self.compute_bollinger_bands(candles)
        current_price = float(candles[-1]['close'])
        
        if current_price > upper_band:
            return "SELL"
        elif current_price < lower_band:
            return "BUY"
        else:
            return "HOLD"

    def compute_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        
        # Avoid division by zero
        if down == 0:
            rs = float('inf')
        else:
            rs = up/down
        
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            
            # Avoid division by zero
            if down == 0:
                rs = float('inf')
            else:
                rs = up/down
            
            rsi[i] = 100. - 100./(1. + rs)

        return rsi[-1]

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        ema12 = self.exponential_moving_average(prices, 12)
        ema26 = self.exponential_moving_average(prices, 26)
        macd = ema12 - ema26
        signal = self.exponential_moving_average(macd, 9)
        histogram = macd - signal
        return macd[-1], signal[-1], histogram[-1]

    def generate_combined_signal(self, rsi, macd, signal, histogram, candles):
        rsi_signal = self.generate_signal(rsi)

        if macd > signal or histogram >= 0:
            macd_signal = "BUY"
        elif macd < signal or histogram <= 0:
            macd_signal = "SELL"
        else:
            macd_signal = "HOLD"
        
        bollinger_signal = self.generate_bollinger_bands_signal(candles)
        
        signals = [rsi_signal, macd_signal, bollinger_signal]
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        hold_count = signals.count("HOLD")
        
        if buy_count > sell_count and buy_count > hold_count:
            return "BUY"
        elif sell_count > buy_count and sell_count > hold_count:
            return "SELL"
        elif hold_count >= buy_count and hold_count >= sell_count:
            return "HOLD"
        else:
            return "CONFLICTING"

 

    def generate_signal(self, rsi):
        if rsi > 70:
            return "SELL"
        elif rsi < 30:
            return "BUY"
        else:
            return "HOLD"




    # ... (other technical analysis methods)