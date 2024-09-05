import numpy as np
import pandas as pd
from typing import List, Tuple
from coinbaseservice import CoinbaseService
import time

class TechnicalAnalysis:
    def __init__(self, coinbase_service: CoinbaseService):
        self.coinbase_service = coinbase_service
        self.signal_history = []  # To store recent signals
        self.volatility_history = []  # To store recent volatility readings

    def calculate_rsi(self, prices: List[float], period: int) -> float:
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        # Avoid division by zero
        if down == 0:
            rs = float('inf')
        else:
            rs = up / down

        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            upval = delta if delta > 0 else 0.
            downval = -delta if delta < 0 else 0.
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            # Avoid division by zero
            rs = up / down if down != 0 else float('inf')
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi[-1]

    def compute_rsi(self, product_id, candles, period=14):
        prices = [float(candle['close']) for candle in candles]
        return self.calculate_rsi(prices, period)

    def compute_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        return self.calculate_rsi(prices, period)

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
        return self.compute_macd_from_prices(prices)

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
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

    def generate_combined_signal(self, rsi, macd, signal, histogram, candles):
        # Generate individual signals
        rsi_signal = self.generate_signal(rsi)
        macd_signal = self.generate_macd_signal(macd, signal, histogram)
        bollinger_signal = self.generate_bollinger_bands_signal(candles)
        ma_crossover_signal = self.compute_moving_average_crossover(candles)
        stochastic_k, stochastic_d = self.compute_stochastic_oscillator(candles)
        
        trend = self.identify_trend(None, candles)
        volume_profile = self.compute_volume_profile(candles)
        
        # Initialize signal strength
        signal_strength = 0

        # RSI
        if rsi_signal == "BUY":
            signal_strength += 1
        elif rsi_signal == "SELL":
            signal_strength -= 1

        # MACD
        if macd_signal == "BUY":
            signal_strength += 1
        elif macd_signal == "SELL":
            signal_strength -= 1

        # Bollinger Bands
        if bollinger_signal == "BUY":
            signal_strength += 1
        elif bollinger_signal == "SELL":
            signal_strength -= 1

        # Moving Average Crossover
        if ma_crossover_signal == "BUY":
            signal_strength += 1
        elif ma_crossover_signal == "SELL":
            signal_strength -= 1

        # Stochastic Oscillator
        if stochastic_k > 80 and stochastic_d > 80:
            signal_strength -= 1  # Overbought condition
        elif stochastic_k < 20 and stochastic_d < 20:
            signal_strength += 1  # Oversold condition

        # Trend
        if trend == "Uptrend":
            signal_strength += 1
        elif trend == "Downtrend":
            signal_strength -= 1

        # Volume Profile
        current_price = float(candles[-1]['close'])
        volume_resistance = max(volume_profile, key=lambda x: x[1])[0]
        if current_price > volume_resistance:
            signal_strength += 1
        elif current_price < volume_resistance:
            signal_strength -= 1

        # ATR for volatility
        atr = self.compute_atr(candles)
        avg_price = np.mean([float(candle['close']) for candle in candles[-14:]])
        volatility = atr / avg_price
        self.update_volatility_history(volatility)

        # Adjust signal strength based on volatility
        avg_volatility = np.mean([v['volatility'] for v in self.volatility_history[-10:]])
        if volatility > avg_volatility * 1.2:  # If current volatility is 20% higher than average
            signal_strength *= 0.8  # Reduce signal strength in high volatility

        # Add trend-following component
        short_ma = self.calculate_sma(candles, 10)
        long_ma = self.calculate_sma(candles, 50)
        trend = "Uptrend" if short_ma > long_ma else "Downtrend"

        # Add dynamic support/resistance
        current_price = float(candles[-1]['close'])
        ma_200 = self.calculate_sma(candles, 200)

        # Incorporate volume analysis
        volume_signal = self.analyze_volume(candles)

        # Implement pullback strategy
        pullback_signal = self.detect_pullback(candles)

        # Adjust signal strength based on new components
        if trend == "Uptrend":
            signal_strength += 1
        else:
            signal_strength -= 1

        if current_price < ma_200:
            signal_strength += 1  # Price below 200 MA might be a good buying opportunity
        else:
            signal_strength -= 1  # Price above 200 MA might be risky for buying

        if volume_signal == "High":
            signal_strength += 1
        elif volume_signal == "Low":
            signal_strength -= 1

        if pullback_signal == "Buy":
            signal_strength += 2
        elif pullback_signal == "Sell":
            signal_strength -= 2

        # Adjust thresholds for more conservative buying
        if signal_strength >= 4:
            final_signal = "STRONG BUY"
        elif 2 <= signal_strength < 4:
            final_signal = "BUY"
        elif -2 < signal_strength < 2:
            final_signal = "HOLD"
        elif -4 < signal_strength <= -2:
            final_signal = "SELL"
        else:
            final_signal = "STRONG SELL"

        return final_signal

    def generate_macd_signal(self, macd, signal, histogram):
        if macd > signal or histogram >= 0:
            return "BUY"
        elif macd < signal or histogram <= 0:
            return "SELL"
        else:
            return "HOLD"

    def compute_volume_profile(self, candles: List[dict], num_bins: int = 10) -> List[Tuple[float, float]]:
        prices = [float(candle['close']) for candle in candles]
        volumes = [float(candle['volume']) for candle in candles]
        
        min_price, max_price = min(prices), max(prices)
        bins = np.linspace(min_price, max_price, num_bins + 1)
        
        digitized = np.digitize(prices, bins)
        volume_profile = [sum(volumes[i] for i in range(len(volumes)) if digitized[i] == j) for j in range(1, len(bins))]
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return list(zip(bin_centers, volume_profile))

    def generate_signal(self, rsi):
        if rsi > 70:
            return "SELL"
        elif rsi < 30:
            return "BUY"
        else:
            return "HOLD"

    def calculate_position_size(self, account_balance: float, risk_per_trade: float, stop_loss_percent: float) -> float:
        max_loss_amount = account_balance * risk_per_trade
        position_size = max_loss_amount / stop_loss_percent
        return position_size

    def backtest_strategy(self, product_id: str, start_date: str, end_date: str, initial_balance: float, risk_per_trade: float, stop_loss_percent: float):
        candles = self.coinbase_service.get_product_candles(product_id, start_date, end_date)
        balance = initial_balance
        position = 0
        trades = []

        for i in range(len(candles) - 1):
            prices = [float(candle['close']) for candle in candles[:i+1]]
            rsi = self.compute_rsi_from_prices(prices)
            macd, signal, histogram = self.compute_macd_from_prices(prices)
            combined_signal = self.generate_combined_signal(rsi, macd, signal, histogram, candles[:i+1])

            current_price = float(candles[i]['close'])
            next_price = float(candles[i+1]['close'])

            if combined_signal == "BUY" and position == 0:
                position_size = self.calculate_position_size(balance, risk_per_trade, stop_loss_percent)
                position = position_size / current_price
                balance -= position_size
                trades.append(("BUY", current_price, position))
            elif combined_signal == "SELL" and position > 0:
                balance += position * current_price
                trades.append(("SELL", current_price, position))
                position = 0

        if position > 0:
            balance += position * next_price
            trades.append(("SELL", next_price, position))

        return balance, trades

    def compute_stochastic_oscillator(self, candles: List[dict], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        prices = pd.DataFrame({
            'high': [float(candle['high']) for candle in candles],
            'low': [float(candle['low']) for candle in candles],
            'close': [float(candle['close']) for candle in candles]
        })
        
        low_min = prices['low'].rolling(window=k_period).min()
        high_max = prices['high'].rolling(window=k_period).max()
        
        k = 100 * (prices['close'] - low_min) / (high_max - low_min)
        d = k.rolling(window=d_period).mean()
        
        return k.iloc[-1], d.iloc[-1]

    def compute_moving_average_crossover(self, candles: List[dict], short_period: int = 50, long_period: int = 200) -> str:
        prices = [float(candle['close']) for candle in candles]
        short_ma = pd.Series(prices).rolling(window=short_period).mean().iloc[-1]
        long_ma = pd.Series(prices).rolling(window=long_period).mean().iloc[-1]
        
        if short_ma > long_ma:
            return "BUY"
        elif short_ma < long_ma:
            return "SELL"
        else:
            return "HOLD"

    def compute_atr(self, candles: List[dict], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 0  # or some other default value

        high = [float(candle['high']) for candle in candles]
        low = [float(candle['low']) for candle in candles]
        close = [float(candle['close']) for candle in candles]
        
        tr1 = [high[i] - low[i] for i in range(len(high))]
        tr2 = [abs(high[i] - close[i-1]) for i in range(1, len(high))]
        tr3 = [abs(low[i] - close[i-1]) for i in range(1, len(low))]
        
        tr = [max(tr1[i], tr2[i-1], tr3[i-1]) for i in range(1, len(tr1))]
        atr = sum(tr[-period:]) / period
        
        return atr

    def update_volatility_history(self, volatility: float):
        self.volatility_history.append({'timestamp': time.time(), 'volatility': volatility})
        # Keep only the last 100 volatility readings
        self.volatility_history = self.volatility_history[-100:]

    def calculate_sma(self, candles, period):
        prices = [float(candle['close']) for candle in candles[-period:]]
        return sum(prices) / period

    def analyze_volume(self, candles):
        recent_volumes = [float(candle['volume']) for candle in candles[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(candles[-1]['volume'])
        
        if current_volume > avg_volume * 1.5:
            return "High"
        elif current_volume < avg_volume * 0.5:
            return "Low"
        else:
            return "Normal"

    def detect_pullback(self, candles):
        recent_prices = [float(candle['close']) for candle in candles[-5:]]
        if recent_prices[-1] < min(recent_prices[:-1]) and self.calculate_sma(candles, 20) > self.calculate_sma(candles, 50):
            return "Buy"
        elif recent_prices[-1] > max(recent_prices[:-1]) and self.calculate_sma(candles, 20) < self.calculate_sma(candles, 50):
            return "Sell"
        else:
            return "Hold"