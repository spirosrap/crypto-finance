import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from coinbaseservice import CoinbaseService
import time
import yfinance as yf
import talib

class TechnicalAnalysis:
    def __init__(self, coinbase_service: CoinbaseService):
        self.coinbase_service = coinbase_service
        self.signal_history = []
        self.volatility_history = []
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volatility_threshold = 0.03
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bollinger_window = 20
        self.bollinger_std = 2

    def calculate_rsi(self, prices: List[float], period: int) -> float:
        """
        Calculate the Relative Strength Index (RSI) using TA-Lib.
        """
        return talib.RSI(np.array(prices), timeperiod=period)[-1]

    def compute_rsi(self, product_id: str, candles: List[Dict], period: int = None) -> float:
        """
        Compute the RSI for a given product using its historical candle data.

        :param product_id: Product identifier.
        :param candles: List of historical candle data.
        :param period: Period for RSI calculation.
        :return: RSI value.
        """
        period = period or self.rsi_period
        try:
            prices = self.extract_prices(candles)
            return self.calculate_rsi(prices, period)
        except IndexError as e:
            print(f"Error computing RSI: Not enough data for product {product_id}. {e}")
        except Exception as e:
            print(f"Error computing RSI for product {product_id}: {e}")
        return 0.0

    def compute_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        return self.calculate_rsi(prices, period)

    def identify_trend(self, product_id: str, candles: List[Dict], window: int = 20) -> str:
        prices = self.extract_prices(candles)
        if len(prices) < window:
            return "Not enough data"
        
        # Calculate Simple Moving Average
        sma = np.convolve(prices, np.ones(window), 'valid') / window
        
        # Check if we have enough data points to calculate the gradient
        if len(sma) < 2:
            return "Not enough data"
        
        # Calculate the slope of the SMA
        slope = np.gradient(sma)
        
        # Determine the trend based on the recent slope
        recent_slope = slope[-5:].mean() if len(slope) >= 5 else slope.mean()
        
        if recent_slope > 0.01:  # Threshold can be adjusted
            return "Uptrend"
        elif recent_slope < -0.01:  # Threshold can be adjusted
            return "Downtrend"
        else:
            return "Sideways"

    def compute_macd(self, product_id: str, candles: List[Dict]) -> Tuple[float, float, float]:
        prices = self.extract_prices(candles)
        return self.compute_macd_from_prices(prices)

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        macd, signal, histogram = talib.MACD(np.array(prices), 
                                             fastperiod=self.macd_fast, 
                                             slowperiod=self.macd_slow, 
                                             signalperiod=self.macd_signal)
        return macd[-1], signal[-1], histogram[-1]

    def exponential_moving_average(self, data: List[float], span: int) -> np.ndarray:
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    def compute_bollinger_bands(self, candles: List[Dict], window: int = None, num_std: float = None) -> Tuple[float, float, float]:
        window = window or self.bollinger_window
        num_std = num_std or self.bollinger_std
        prices = np.array(self.extract_prices(candles))
        
        upper, middle, lower = talib.BBANDS(prices, timeperiod=window, nbdevup=num_std, nbdevdn=num_std)
        
        return upper[-1], middle[-1], lower[-1]

    def generate_bollinger_bands_signal(self, candles: List[Dict]) -> str:
        upper_band, middle_band, lower_band = self.compute_bollinger_bands(candles)
        current_price = self.extract_prices(candles)[-1]
        
        if current_price > upper_band:
            return "SELL"
        elif current_price < lower_band:
            return "BUY"
        else:
            return "HOLD"

    def generate_combined_signal(self, rsi: float, macd: float, signal: float, histogram: float, 
                                 candles: List[Dict], market_conditions: Optional[str] = None) -> str:
        current_price = float(candles[-1]['close'])
        market_conditions = market_conditions or self.analyze_market_conditions(candles)
        volatility_std = self.calculate_volatility(candles)
        signal_strength = self.calculate_signal_strength(rsi, macd, signal, histogram, market_conditions, candles, volatility_std)
        final_signal = self.determine_final_signal(signal_strength, market_conditions, current_price, candles)
        return final_signal

    def calculate_volatility(self, candles: List[Dict]) -> float:
        prices = self.extract_prices(candles[-20:])
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(365)

    def calculate_signal_strength(self, rsi: float, macd: float, signal: float, histogram: float, market_conditions: str,
                                  candles: List[Dict], volatility_std: float) -> int:
        current_price = float(candles[-1]['close'])
        signal_strength = 0

        # Define weights for each signal
        weights = {
            'rsi': 2,
            'macd': 2,
            'bollinger': 1,
            'ma_crossover': 1,
            'stochastic': 1,
            'trend': 2,
            'volume_profile': 1,
            'short_term_trend': 2,
            'long_term_trend': 1,
            'volume': 1
        }

        # Adjust weights for bear markets
        if market_conditions in [ "Bear Market", "Bearish"]:
            weights['rsi'] = 1
            weights['trend'] = 1
            weights['bollinger'] = 1
            weights['volume_profile'] = 1
            weights['short_term_trend'] = 1
            weights['long_term_trend'] = 1
            weights['volume'] = 1
            weights['macd'] = 1
            weights['stochastic'] = 1
            weights['ma_crossover'] = 1

        signal_strength += weights['rsi'] * self.evaluate_rsi_signal(rsi, volatility_std)
        signal_strength += weights['macd'] * self.evaluate_macd_signal(macd, signal, histogram)
        signal_strength += weights['bollinger'] * self.evaluate_bollinger_signal(candles)
        signal_strength += weights['ma_crossover'] * self.evaluate_ma_crossover_signal(candles)
        signal_strength += weights['stochastic'] * self.evaluate_stochastic_signal(candles)
        signal_strength += weights['trend'] * self.evaluate_trend_signal(candles)
        signal_strength += weights['volume_profile'] * self.evaluate_volume_profile_signal(candles, current_price)
        signal_strength = self.adjust_signal_for_volatility(signal_strength, candles)
        signal_strength = self.adjust_signal_for_market_conditions(signal_strength, market_conditions, current_price, candles)

        # Add more weight to recent performance
        short_term_trend = self.identify_trend(None, candles[-20:])
        long_term_trend = self.identify_trend(None, candles)
        
        if short_term_trend != "Not enough data" and long_term_trend != "Not enough data":
            if short_term_trend == "Uptrend" and long_term_trend == "Uptrend":
                signal_strength += weights['short_term_trend'] * 2
            elif short_term_trend == "Downtrend" and long_term_trend == "Downtrend":
                signal_strength -= weights['short_term_trend'] * 2
        
        # Consider volume
        volume_signal = self.analyze_volume(candles)
        if volume_signal == "High":
            signal_strength += weights['volume']
        elif volume_signal == "Low":
            signal_strength -= weights['volume']
        
        return signal_strength  # Return an integer

    def evaluate_rsi_signal(self, rsi: float, volatility_std: float) -> int:
        rsi_signal = self.generate_signal(rsi, volatility_std)
        return 1 if rsi_signal == "BUY" else -1 if rsi_signal == "SELL" else 0

    def evaluate_macd_signal(self, macd: float, signal: float, histogram: float) -> int:
        macd_signal = self.generate_macd_signal(macd, signal, histogram)
        return 1 if macd_signal == "BUY" else -1 if macd_signal == "SELL" else 0

    def evaluate_bollinger_signal(self, candles: List[Dict]) -> int:
        bollinger_signal = self.generate_bollinger_bands_signal(candles)
        return 1 if bollinger_signal == "BUY" else -1 if bollinger_signal == "SELL" else 0

    def evaluate_ma_crossover_signal(self, candles: List[Dict]) -> int:
        ma_crossover_signal = self.compute_moving_average_crossover(candles)
        return 1 if ma_crossover_signal == "BUY" else -1 if ma_crossover_signal == "SELL" else 0

    def evaluate_stochastic_signal(self, candles: List[Dict]) -> int:
        stochastic_k, stochastic_d = self.compute_stochastic_oscillator(candles)
        if stochastic_k > 80 and stochastic_d > 80:
            return -1  # Overbought condition
        elif stochastic_k < 20 and stochastic_d < 20:
            return 1  # Oversold condition
        return 0

    def evaluate_trend_signal(self, candles: List[Dict]) -> int:
        price_trend = self.identify_trend(None, candles)
        return 1 if price_trend == "Uptrend" else -1 if price_trend == "Downtrend" else 0

    def evaluate_volume_profile_signal(self, candles: List[Dict], current_price: float) -> int:
        volume_profile = self.compute_volume_profile(candles)
        volume_resistance = max(volume_profile, key=lambda x: x[1])[0]
        if current_price > volume_resistance:
            return 1
        elif current_price < volume_resistance:
            return -1
        return 0

    def adjust_signal_for_volatility(self, signal_strength: int, candles: List[Dict]) -> int:
        atr = self.compute_atr(candles)
        avg_price = np.mean(self.extract_prices(candles)[-14:])
        volatility_atr = atr / avg_price
        self.update_volatility_history(volatility_atr)

        high_volatility, very_high_volatility = self.get_dynamic_volatility_threshold()

        if volatility_atr > very_high_volatility:
            return int(signal_strength * 0.5)  # Significantly reduce signal strength in very high volatility
        elif volatility_atr > high_volatility:
            return int(signal_strength * 0.75)  # Moderately reduce signal strength in high volatility
        return signal_strength

    def adjust_signal_for_market_conditions(self, signal_strength: int, market_conditions: str, 
                                            current_price: float, candles: List[Dict]) -> int:
        short_ma = self.calculate_sma(candles, 10)
        long_ma = self.calculate_sma(candles, 50)
        ma_trend = "Uptrend" if short_ma > long_ma else "Downtrend"

        ma_200 = self.calculate_sma(candles, 200)
        volume_signal = self.analyze_volume(candles)
        pullback_signal = self.detect_pullback(candles)

        market_condition_weights = {
            "Bear Market": -2,
            "Bull Market": 2,
            "Bullish": 1,
            "Bearish": -1
        }
        signal_strength += market_condition_weights.get(market_conditions, 0)
        signal_strength += 1 if ma_trend == "Uptrend" else -1
        signal_strength += 1 if current_price < ma_200 else -1
        signal_strength += 1 if volume_signal == "High" else -1 if volume_signal == "Low" else 0
        signal_strength += 2 if pullback_signal == "Buy" else -2 if pullback_signal == "Sell" else 0

        return signal_strength

    def determine_final_signal(self, signal_strength: int, market_conditions: str, 
                               current_price: float, candles: List[Dict]) -> str:
        # Adjust thresholds based on market conditions
        if market_conditions == "Bull Market":
            buy_threshold = 2
            sell_threshold = -3
        elif market_conditions == "Bear Market":
            buy_threshold = 3
            sell_threshold = -2
        else:
            buy_threshold = 3
            sell_threshold = -3
        
        if signal_strength >= buy_threshold:
            return "STRONG BUY" if signal_strength >= buy_threshold + 2 else "BUY"
        elif signal_strength <= sell_threshold:
            return "STRONG SELL" if signal_strength <= sell_threshold - 2 else "SELL"
        else:
            return "HOLD"

    def generate_macd_signal(self, macd: float, signal: float, histogram: float) -> str:
        if macd > signal or histogram >= 0:
            return "BUY"
        elif macd < signal or histogram <= 0:
            return "SELL"
        else:
            return "HOLD"

    def compute_volume_profile(self, candles: List[Dict], num_bins: int = 10) -> List[Tuple[float, float]]:
        prices = self.extract_prices(candles)
        volumes = self.extract_prices(candles, 'volume')
        
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
        high = np.array(self.extract_prices(candles, 'high'))
        low = np.array(self.extract_prices(candles, 'low'))
        close = np.array(self.extract_prices(candles))
        
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        
        return k[-1], d[-1]

    def compute_moving_average_crossover(self, candles: List[Dict], short_period: int = 50, long_period: int = 200) -> str:
        prices = self.extract_prices(candles)
        short_ma = pd.Series(prices).rolling(window=short_period).mean().iloc[-1]
        long_ma = pd.Series(prices).rolling(window=long_period).mean().iloc[-1]
        
        return "BUY" if short_ma > long_ma else "SELL" if short_ma < long_ma else "HOLD"

    def compute_atr(self, candles: List[Dict], period: int = 14) -> float:
        high = np.array(self.extract_prices(candles, 'high'))
        low = np.array(self.extract_prices(candles, 'low'))
        close = np.array(self.extract_prices(candles))
        
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        return atr[-1]

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
        prices = self.extract_prices(candles)[-period:]
        return sum(prices) / period

    def analyze_volume(self, candles: List[Dict]) -> str:
        recent_volumes = self.extract_prices(candles, 'volume')[-10:]
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        current_volume = self.extract_prices(candles, 'volume')[-1]
        
        if current_volume > avg_volume * 1.5:
            return "High"
        elif current_volume < avg_volume * 0.5:
            return "Low"
        else:
            return "Normal"
        
    def extract_prices(self, candles: List[Dict], key: str = 'close') -> List[float]:
        """
        Extract prices from candle data.

        :param candles: List of historical candle data.
        :param key: Key to extract from each candle (default is 'close').
        :return: List of prices.
        """
        return [float(candle[key]) for candle in candles]


    def detect_pullback(self, candles: List[Dict]) -> str:
        recent_prices = self.extract_prices(candles)[-5:]
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
        prices = self.extract_prices(candles)
        volumes = self.extract_prices(candles, 'volume')
        
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
        prices = self.extract_prices(candles)
        volumes = self.extract_prices(candles, 'volume')
        
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
        prices = self.extract_prices(candles)
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