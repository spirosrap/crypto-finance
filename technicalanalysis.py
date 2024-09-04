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
        # Generate individual signals
        rsi_signal = self.generate_signal(rsi)
        macd_signal = self.generate_macd_signal(macd, signal, histogram)
        bollinger_signal = self.generate_bollinger_bands_signal(candles)
        ma_crossover_signal = self.compute_moving_average_crossover(candles)
        stochastic_k, stochastic_d = self.compute_stochastic_oscillator(candles)
        
        # Simplify trend identification
        trend = self.identify_trend(None, candles)

        # Count buy and sell signals
        signals = [rsi_signal, macd_signal, bollinger_signal, ma_crossover_signal]
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")

        # Initialize signal strength
        signal_strength = 0

        # Simplified decision making
        if buy_count > sell_count:
            signal_strength += 1
        elif sell_count > buy_count:
            signal_strength -= 1

        # Consider trend
        if trend == "Uptrend":
            signal_strength += 1
        elif trend == "Downtrend":
            signal_strength -= 1

        # Consider Stochastic Oscillator
        if stochastic_k > 70 and stochastic_d >70:
            signal_strength -= 1  # Overbought condition
        elif stochastic_k < 30 and stochastic_d < 30:
            signal_strength += 1  # Oversold condition

        # Make final decision
        if signal_strength >= 1:
            final_signal = "BUY"
        elif signal_strength <= -1:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        # Update volatility history (keep this for potential future use)
        atr = self.compute_atr(candles, period=14)
        avg_price = np.mean([float(candle['close']) for candle in candles[-14:]])
        volatility = atr / avg_price
        self.update_volatility_history(volatility)

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