import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from coinbaseservice import CoinbaseService
import time
import yfinance as yf

class TechnicalAnalysis:
    def __init__(self, coinbase_service: CoinbaseService):
        self.coinbase_service = coinbase_service
        self.signal_history = []  # To store recent signals
        self.volatility_history = []  # To store recent volatility readings
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volatility_threshold = 0.03  # 3% daily volatility threshold

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

    def compute_rsi(self, product_id: str, candles: List[Dict], period: int = 14) -> float:
        prices = [float(candle['close']) for candle in candles]
        return self.calculate_rsi(prices, period)

    def compute_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        return self.calculate_rsi(prices, period)

    def identify_trend(self, product_id: str, candles: List[Dict], window: int = 20) -> str:
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

    def compute_macd(self, product_id: str, candles: List[Dict]) -> Tuple[float, float, float]:
        prices = [float(candle['close']) for candle in candles]
        return self.compute_macd_from_prices(prices)

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        ema12 = self.exponential_moving_average(prices, 12)
        ema26 = self.exponential_moving_average(prices, 26)
        macd = ema12 - ema26
        signal = self.exponential_moving_average(macd, 9)
        histogram = macd - signal
        return macd[-1], signal[-1], histogram[-1]

    def exponential_moving_average(self, data: List[float], span: int) -> np.ndarray:
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    def compute_bollinger_bands(self, candles: List[Dict], window: int = 20, num_std: float = 2) -> Tuple[float, float, float]:
        prices = [float(candle['close']) for candle in candles]
        prices_series = pd.Series(prices)
        
        rolling_mean = prices_series.rolling(window=window).mean()
        rolling_std = prices_series.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band.iloc[-1], rolling_mean.iloc[-1], lower_band.iloc[-1]

    def generate_bollinger_bands_signal(self, candles: List[Dict]) -> str:
        upper_band, middle_band, lower_band = self.compute_bollinger_bands(candles)
        current_price = float(candles[-1]['close'])
        
        if current_price > upper_band:
            return "SELL"
        elif current_price < lower_band:
            return "BUY"
        else:
            return "HOLD"

    def generate_combined_signal(self, rsi: float, macd: float, signal: float, histogram: float, 
                                 candles: List[Dict], market_conditions: Optional[str] = None) -> str:
        # Calculate current price once
        current_price = float(candles[-1]['close'])

        if market_conditions is None:
            market_conditions = self.analyze_market_conditions(candles)
        
        # Calculate volatility using standard deviation of returns
        prices = [float(candle['close']) for candle in candles[-20:]]  # Use last 20 candles
        returns = np.diff(np.log(prices))
        volatility_std = np.std(returns) * np.sqrt(365)  # Annualized volatility

        # Generate individual signals
        rsi_signal = self.generate_signal(rsi, volatility_std)
        macd_signal = self.generate_macd_signal(macd, signal, histogram)
        bollinger_signal = self.generate_bollinger_bands_signal(candles)
        ma_crossover_signal = self.compute_moving_average_crossover(candles)
        stochastic_k, stochastic_d = self.compute_stochastic_oscillator(candles)
        
        price_trend = self.identify_trend(None, candles)
        volume_profile = self.compute_volume_profile(candles)
        
        # Initialize signal strength
        signal_strength = 0

        # RSI
        signal_strength += 1 if rsi_signal == "BUY" else -1 if rsi_signal == "SELL" else 0

        # MACD
        signal_strength += 1 if macd_signal == "BUY" else -1 if macd_signal == "SELL" else 0

        # Bollinger Bands
        signal_strength += 1 if bollinger_signal == "BUY" else -1 if bollinger_signal == "SELL" else 0

        # Moving Average Crossover
        signal_strength += 1 if ma_crossover_signal == "BUY" else -1 if ma_crossover_signal == "SELL" else 0

        # Stochastic Oscillator
        if stochastic_k > 80 and stochastic_d > 80:
            signal_strength -= 1  # Overbought condition
        elif stochastic_k < 20 and stochastic_d < 20:
            signal_strength += 1  # Oversold condition

        # Trend
        signal_strength += 1 if price_trend == "Uptrend" else -1 if price_trend == "Downtrend" else 0

        # Volume Profile
        volume_resistance = max(volume_profile, key=lambda x: x[1])[0]
        if current_price > volume_resistance:
            signal_strength += 1
        elif current_price < volume_resistance:
            signal_strength -= 1

        # ATR for volatility
        atr = self.compute_atr(candles)
        avg_price = np.mean([float(candle['close']) for candle in candles[-14:]])
        volatility_atr = atr / avg_price
        self.update_volatility_history(volatility_atr)

        # Get dynamic volatility thresholds
        high_volatility, very_high_volatility = self.get_dynamic_volatility_threshold()

        # Adjust signal strength based on volatility
        if volatility_atr > very_high_volatility:
            signal_strength *= 0.5  # Significantly reduce signal strength in very high volatility
        elif volatility_atr > high_volatility:
            signal_strength *= 0.75  # Moderately reduce signal strength in high volatility

        # Add trend-following component
        short_ma = self.calculate_sma(candles, 10)
        long_ma = self.calculate_sma(candles, 50)
        ma_trend = "Uptrend" if short_ma > long_ma else "Downtrend"

        # Add dynamic support/resistance
        ma_200 = self.calculate_sma(candles, 200)

        # Incorporate volume analysis
        volume_signal = self.analyze_volume(candles)

        # Implement pullback strategy
        pullback_signal = self.detect_pullback(candles)

        # Incorporate market conditions
        if market_conditions == "Bear Market":
            signal_strength -= 2  # Strongly discourage buying in a bear market
        elif market_conditions == "Bull Market":
            signal_strength += 2  # Strongly encourage buying in a bull market
        elif market_conditions == "Bullish":
            signal_strength += 1
        elif market_conditions == "Bearish":
            signal_strength -= 1
        
        # Adjust signal strength based on new components
        signal_strength += 1 if ma_trend == "Uptrend" else -1

        if current_price < ma_200:
            signal_strength += 1  # Price below 200 MA might be a good buying opportunity
        else:
            signal_strength -= 1  # Price above 200 MA might be risky for buying

        signal_strength += 1 if volume_signal == "High" else -1 if volume_signal == "Low" else 0
        signal_strength += 2 if pullback_signal == "Buy" else -2 if pullback_signal == "Sell" else 0

        # Adjust thresholds for more aggressive buying and selling in a bull market
        if market_conditions == "Bull Market":
            if signal_strength >= 3:
                final_signal = "STRONG BUY"  # Lower threshold for strong buying in a bull market
            elif 1 <= signal_strength < 3:
                final_signal = "BUY"
            elif -1 < signal_strength < 1:
                final_signal = "HOLD"
            elif -3 < signal_strength <= -1:
                final_signal = "SELL"
            else:
                final_signal = "STRONG SELL"
        elif market_conditions == "Bear Market":
            if signal_strength >= 5:
                final_signal = "BUY"  # Require a higher threshold for buying in a bear market
            elif 0 <= signal_strength < 5:
                final_signal = "HOLD"
            elif -5 < signal_strength < 0:
                final_signal = "SELL"
            else:
                final_signal = "STRONG SELL"  # Stronger selling signal in a bear market
        else:
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

    def generate_macd_signal(self, macd: float, signal: float, histogram: float) -> str:
        if macd > signal or histogram >= 0:
            return "BUY"
        elif macd < signal or histogram <= 0:
            return "SELL"
        else:
            return "HOLD"

    def compute_volume_profile(self, candles: List[Dict], num_bins: int = 10) -> List[Tuple[float, float]]:
        prices = [float(candle['close']) for candle in candles]
        volumes = [float(candle['volume']) for candle in candles]
        
        min_price, max_price = min(prices), max(prices)
        bins = np.linspace(min_price, max_price, num_bins + 1)
        
        digitized = np.digitize(prices, bins)
        volume_profile = [sum(volumes[i] for i in range(len(volumes)) if digitized[i] == j) for j in range(1, len(bins))]
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return list(zip(bin_centers, volume_profile))

    def generate_signal(self, rsi: float, volatility: float) -> str:
        # Adjust RSI levels based on volatility
        if volatility > self.volatility_threshold:
            adjusted_overbought = self.rsi_overbought + 10  # More room for upside in volatile markets
            adjusted_oversold = self.rsi_oversold - 10  # More room for downside in volatile markets
        else:
            adjusted_overbought = self.rsi_overbought
            adjusted_oversold = self.rsi_oversold

        return "SELL" if rsi > adjusted_overbought else "BUY" if rsi < adjusted_oversold else "HOLD"

    def calculate_position_size(self, account_balance: float, risk_per_trade: float, stop_loss_percent: float) -> float:
        max_loss_amount = account_balance * risk_per_trade
        position_size = max_loss_amount / stop_loss_percent
        return position_size


    def compute_stochastic_oscillator(self, candles: List[Dict], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
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

    def compute_moving_average_crossover(self, candles: List[Dict], short_period: int = 50, long_period: int = 200) -> str:
        prices = [float(candle['close']) for candle in candles]
        short_ma = pd.Series(prices).rolling(window=short_period).mean().iloc[-1]
        long_ma = pd.Series(prices).rolling(window=long_period).mean().iloc[-1]
        
        return "BUY" if short_ma > long_ma else "SELL" if short_ma < long_ma else "HOLD"

    def compute_atr(self, candles: List[Dict], period: int = 14) -> float:
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

    def get_dynamic_volatility_threshold(self) -> Tuple[float, float]:
        volatilities = [v['volatility'] for v in self.volatility_history]
        mean_volatility = np.mean(volatilities)
        std_volatility = np.std(volatilities)
        
        # Set thresholds at 1 and 2 standard deviations above the mean
        high_threshold = mean_volatility + std_volatility
        very_high_threshold = mean_volatility + 2 * std_volatility
        
        return high_threshold, very_high_threshold

    def calculate_sma(self, candles: List[Dict], period: int) -> float:
        prices = [float(candle['close']) for candle in candles[-period:]]
        return sum(prices) / period

    def analyze_volume(self, candles: List[Dict]) -> str:
        recent_volumes = [float(candle['volume']) for candle in candles[-10:]]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = float(candles[-1]['volume'])
        
        if current_volume > avg_volume * 1.5:
            return "High"
        elif current_volume < avg_volume * 0.5:
            return "Low"
        else:
            return "Normal"

    def detect_pullback(self, candles: List[Dict]) -> str:
        recent_prices = [float(candle['close']) for candle in candles[-5:]]
        if recent_prices[-1] < min(recent_prices[:-1]) and self.calculate_sma(candles, 20) > self.calculate_sma(candles, 50):
            return "Buy"
        elif recent_prices[-1] > max(recent_prices[:-1]) and self.calculate_sma(candles, 20) < self.calculate_sma(candles, 50):
            return "Sell"
        else:
            return "Hold"

    def analyze_market_conditions(self, candles: List[Dict]) -> str:
        # Convert 50 days to hours
        long_term_period = 200 * 24

        # Calculate the long-term moving average
        long_term_ma = self.calculate_sma(candles, long_term_period)

        # Analyze market conditions based on price action, volume, and volatility
        prices = [float(candle['close']) for candle in candles]
        volumes = [float(candle['volume']) for candle in candles]
        
        # Calculate price change and volume change
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_change = (volumes[-1] - volumes[0]) / volumes[0]
        
        # Calculate volatility (e.g., using Average True Range)
        volatility = self.compute_atr(candles)
        
        # Calculate percentage drawdown from peak
        peak_price = max(prices)
        current_price = prices[-1]
        drawdown = (peak_price - current_price) / peak_price

        # Calculate the shorter-term moving average (e.g., 50-day MA)
        short_term_period = 50 * 24
        short_term_ma = self.calculate_sma(candles, short_term_period)

        # Calculate the average volume change during bull markets
        bull_market_volume_change = self.calculate_average_bull_market_volume_change(candles)

        # Determine the current market regime based on volatility
        market_regime = self.determine_market_regime(candles)

        # Define dynamic bull market thresholds
        bull_market_price_threshold = short_term_ma * 1.2  # Price above 20% of short-term MA
        bull_market_volume_threshold = bull_market_volume_change * 0.8  # 80% of average bull market volume change
        bull_market_volatility_threshold = self.get_volatility_threshold(market_regime)

        # Define bear market thresholds
        bear_market_drawdown_threshold = 0.2  # 20% drawdown
        bear_market_price_threshold = 0.8  # Price below 80% of long-term MA

        # Check for bull market conditions
        if current_price > bull_market_price_threshold and volume_change > bull_market_volume_threshold and volatility < bull_market_volatility_threshold:
            return "Bull Market"
        # Check for bear market conditions
        elif drawdown > bear_market_drawdown_threshold and current_price < bear_market_price_threshold * long_term_ma:
            return "Bear Market"
        elif price_change > 0.05 and volume_change > 0.1 and volatility > 0.03:
            return "Bullish"
        elif price_change < -0.05 and volume_change < -0.1 and volatility > 0.03:
            return "Bearish"
        else:
            return "Neutral"

    def calculate_average_bull_market_volume_change(self, candles: List[Dict]) -> float:
        prices = [float(candle['close']) for candle in candles]
        volumes = [float(candle['volume']) for candle in candles]
        
        # Define bull market criteria (e.g., price increasing for 5 consecutive days)
        bull_market_periods = []
        for i in range(5, len(prices)):
            if all(prices[j] < prices[j+1] for j in range(i-5, i)):
                bull_market_periods.append(i)
        
        # Calculate volume changes during bull market periods
        volume_changes = []
        for period in bull_market_periods:
            start_volume = volumes[period-5]
            end_volume = volumes[period]
            volume_change = (end_volume - start_volume) / start_volume
            volume_changes.append(volume_change)
        
        # Return the average volume change during bull markets
        return np.mean(volume_changes) if volume_changes else 0.1  # Default to 10% if no bull markets found

    def determine_market_regime(self, candles: List[Dict]) -> str:
        # Calculate historical volatility using a 20-day rolling window
        prices = [float(candle['close']) for candle in candles]
        returns = np.log(np.array(prices[1:]) / np.array(prices[:-1]))
        volatility = np.std(returns[-20:]) * np.sqrt(252)  # Annualized volatility
        
        # Define volatility thresholds
        if volatility < 0.15:
            return "Low Volatility"
        elif 0.15 <= volatility < 0.30:
            return "Medium Volatility"
        else:
            return "High Volatility"

    def get_volatility_threshold(self, market_regime: str) -> float:
        if market_regime == "Low Volatility":
            return 0.01  # 1% daily volatility
        elif market_regime == "Medium Volatility":
            return 0.02  # 2% daily volatility
        else:  # High Volatility
            return 0.03  # 3% daily volatility
        
    def calculate_atr_position_size(self, balance: float, price: float, atr: float) -> float:
        risk_amount = balance * self.risk_per_trade
        position_size = risk_amount / (self.atr_multiplier * atr)
        return min(position_size * price, balance)  # Ensure we don't exceed available balance        