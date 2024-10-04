import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from coinbaseservice import CoinbaseService
import time
import yfinance as yf
import talib
import logging
from functools import lru_cache
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from ml_model import MLSignal
from historicaldata import HistoricalData
from bitcoinpredictionmodel import BitcoinPredictionModel

@dataclass
class TechnicalAnalysisConfig:
    rsi_overbought: float = 65 # 70
    rsi_oversold: float = 35 # 30
    volatility_threshold: float = 0.03
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2
    risk_per_trade: float = 0.01
    atr_multiplier: float = 2

class TechnicalAnalysis:
    """
    A class for performing technical analysis on cryptocurrency data.

    This class provides methods for calculating various technical indicators
    and generating trading signals based on those indicators.
    """

    def __init__(self, coinbase_service: CoinbaseService, config: Optional[TechnicalAnalysisConfig] = None, candle_interval: str = 'ONE_HOUR'):
        """
        Initialize the TechnicalAnalysis class.

        Args:
            coinbase_service (CoinbaseService): An instance of the CoinbaseService class.
            config (Optional[TechnicalAnalysisConfig]): Configuration for the technical analysis.
        """
        self.coinbase_service = coinbase_service
        self.config = config or TechnicalAnalysisConfig()
        self.signal_history = []
        self.volatility_history = []
        self.logger = logging.getLogger(__name__)
        self.candle_interval = candle_interval
        self.intervals_per_day = self.calculate_intervals_per_day()
        historical_data = HistoricalData(coinbase_service.client)
        self.ml_signal = MLSignal(self.logger, historical_data)
        self.ml_signal.load_model()  # Load or train the model at initialization
        self.scaler = StandardScaler()
        self.bitcoin_prediction_model = BitcoinPredictionModel(coinbase_service)
        self.bitcoin_prediction_model.load_model()  # Load or train the model at initialization

    def calculate_intervals_per_day(self) -> int:
        interval_map = {
            "ONE_MINUTE": 1440,  # 1440 intervals in a day
            "FIVE_MINUTE": 288,   # 288 intervals in a day
            "TEN_MINUTE": 144,    # 144 intervals in a day
            "FIFTEEN_MINUTE": 96,  # 96 intervals in a day
            "THIRTY_MINUTE": 48,   # 48 intervals in a day
            "ONE_HOUR": 24,        # 24 intervals in a day
            "SIX_HOUR": 4,         # 4 intervals in a day
            "ONE_DAY": 1           # 1 interval in a day
        }
        return interval_map.get(self.candle_interval, 24)  # Default to 24 if unknown

    # RSI Methods
    @lru_cache(maxsize=100)
    def calculate_rsi(self, prices: Tuple[float, ...], period: int) -> float:
        """
        Calculate the Relative Strength Index (RSI) for a given set of prices.

        Args:
            prices (Tuple[float, ...]): A tuple of historical prices.
            period (int): The period over which to calculate the RSI.

        Returns:
            float: The calculated RSI value.
        """
        return talib.RSI(np.array(prices), timeperiod=period)[-1]

    def compute_rsi(self, product_id: str, candles: List[Dict], period: Optional[int] = None) -> float:
        """
        Compute the RSI for a given product using its historical candle data.

        :param product_id: Product identifier.
        :param candles: List of historical candle data.
        :param period: Period for RSI calculation.
        :return: RSI value.
        """
        period = period or self.config.rsi_period
        try:
            prices = tuple(self.extract_prices(candles))  # Convert to tuple for caching
            return self.calculate_rsi(prices, period)
        except IndexError as e:
            self.logger.error(f"Error computing RSI: Not enough data for product {product_id}. {e}")
        except Exception as e:
            self.logger.error(f"Error computing RSI for product {product_id}: {e}")
        return 0.0

    def compute_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        return self.calculate_rsi(tuple(prices), period)

    def evaluate_rsi_signal(self, rsi: float, volatility_std: float) -> int:
        rsi_signal = self.generate_signal(rsi, volatility_std)
        return 1 if rsi_signal == "BUY" else -1 if rsi_signal == "SELL" else 0

    # MACD Methods
    def compute_macd(self, product_id: str, candles: List[Dict]) -> Tuple[float, float, float]:
        prices = self.extract_prices(candles)
        return self.compute_macd_from_prices(prices)

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        macd, signal, histogram = talib.MACD(np.array(prices), 
                                             fastperiod=self.config.macd_fast, 
                                             slowperiod=self.config.macd_slow, 
                                             signalperiod=self.config.macd_signal)
        return macd[-1], signal[-1], histogram[-1]

    def generate_macd_signal(self, macd: float, signal: float, histogram: float) -> str:
        if macd > signal or histogram >= 0:
            return "BUY"
        elif macd < signal or histogram <= 0:
            return "SELL"
        else:
            return "HOLD"

    def evaluate_macd_signal(self, macd: float, signal: float, histogram: float) -> int:
        macd_signal = self.generate_macd_signal(macd, signal, histogram)
        return 1 if macd_signal == "BUY" else -1 if macd_signal == "SELL" else 0

    # Bollinger Bands Methods
    def compute_bollinger_bands(self, candles: List[Dict], window: Optional[int] = None, num_std: Optional[float] = None) -> Tuple[float, float, float]:
        window = window or self.config.bollinger_window
        num_std = num_std or self.config.bollinger_std
        prices = self.extract_prices(candles)
        
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

    def evaluate_bollinger_signal(self, candles: List[Dict]) -> int:
        bollinger_signal = self.generate_bollinger_bands_signal(candles)
        return 1 if bollinger_signal == "BUY" else -1 if bollinger_signal == "SELL" else 0

    # Moving Average Methods
    def exponential_moving_average(self, data: List[float], span: int) -> np.ndarray:
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    def get_moving_average(self, candles: List[Dict], period: int, ma_type: str = 'sma') -> float:
        """
        Calculate moving average for given candles and period.

        Args:
            candles (List[Dict]): List of candle data.
            period (int): Period for moving average calculation.
            ma_type (str): Type of moving average ('sma' or 'ema').

        Returns:
            float: Calculated moving average value.
        """
        prices = self.extract_prices(candles)
        adjusted_period = max(1, int(period * self.intervals_per_day / 24))
        
        if ma_type == 'sma':
            return talib.SMA(prices, timeperiod=adjusted_period)[-1]
        elif ma_type == 'ema':
            return talib.EMA(prices, timeperiod=adjusted_period)[-1]
        else:
            raise ValueError(f"Unsupported moving average type: {ma_type}")

    def calculate_sma(self, candles: List[Dict], period: int) -> float:
        return self.get_moving_average(candles, period, 'sma')

    def compute_moving_average_crossover(self, candles: List[Dict], short_period: int = 50, long_period: int = 200) -> str:
        short_ma = self.get_moving_average(candles, short_period, 'sma')
        long_ma = self.get_moving_average(candles, long_period, 'sma')
        
        return "BUY" if short_ma > long_ma else "SELL" if short_ma < long_ma else "HOLD"

    def evaluate_ma_crossover_signal(self, candles: List[Dict]) -> int:
        ma_crossover_signal = self.compute_moving_average_crossover(candles)
        return 1 if ma_crossover_signal == "BUY" else -1 if ma_crossover_signal == "SELL" else 0

    # Stochastic Oscillator Methods
    def compute_stochastic_oscillator(self, candles: List[Dict], k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        high = self.extract_prices(candles, 'high')
        low = self.extract_prices(candles, 'low')
        close = self.extract_prices(candles)
        
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
        
        return k[-1], d[-1]

    def evaluate_stochastic_signal(self, candles: List[Dict]) -> int:
        stochastic_k, stochastic_d = self.compute_stochastic_oscillator(candles)
        if stochastic_k > 80 and stochastic_d > 80:
            return -1  # Overbought condition
        elif stochastic_k < 20 and stochastic_d < 20:
            return 1  # Oversold condition
        return 0

    # Trend Analysis Methods
    def identify_trend(self, product_id: Optional[str], candles: List[Dict], window: int = 20) -> str:
        prices = self.extract_prices(candles)
        if len(prices) < window:
            return "Not enough data"
        
        # Calculate SMA for each point in the window
        sma_values = [self.get_moving_average(candles[i-window:i], window, 'sma') for i in range(window, len(candles)+1)]
        
        if len(sma_values) < 2:
            return "Not enough data"
        
        # Calculate the slope of the SMA
        slope = np.gradient(sma_values)
        
        # Determine the trend based on the recent slope
        recent_slope = slope[-5:].mean() if len(slope) >= 5 else slope.mean()
        
        if recent_slope > 0.01:  # Threshold can be adjusted
            return "Uptrend"
        elif recent_slope < -0.01:  # Threshold can be adjusted
            return "Downtrend"
        else:
            return "Sideways"

    def evaluate_trend_signal(self, candles: List[Dict]) -> int:
        price_trend = self.identify_trend(None, candles)
        return 1 if price_trend == "Uptrend" else -1 if price_trend == "Downtrend" else 0

    # Volume Analysis Methods
    def compute_volume_profile(self, candles: List[Dict], num_bins: int = 10) -> List[Tuple[float, float]]:
        prices = self.extract_prices(candles)
        volumes = self.extract_prices(candles, 'volume')
        
        min_price, max_price = np.min(prices), np.max(prices)
        bins = np.linspace(min_price, max_price, num_bins + 1)
        
        digitized = np.digitize(prices, bins)
        volume_profile = [np.sum(volumes[digitized == i]) for i in range(1, len(bins))]
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        return list(zip(bin_centers, volume_profile))

    def evaluate_volume_profile_signal(self, candles: List[Dict], current_price: float) -> int:
        volume_profile = self.compute_volume_profile(candles)
        volume_resistance = max(volume_profile, key=lambda x: x[1])[0]
        if current_price > volume_resistance:
            return 1
        elif current_price < volume_resistance:
            return -1
        return 0

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
        
    def compute_on_balance_volume(self, close_prices: List[float], volumes: List[float]) -> List[float]:
        """
        Calculate On-Balance Volume (OBV) indicator.

        :param close_prices: List of closing prices.
        :param volumes: List of volume data.
        :return: List of OBV values.
        """
        obv = [0]
        for i in range(1, len(close_prices)):
            if close_prices[i] > close_prices[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif close_prices[i] < close_prices[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])
        return obv

    # Ichimoku Cloud Methods
    def calculate_ichimoku_cloud(self, candles: List[Dict]) -> Dict[str, float]:
        highs = self.extract_prices(candles, 'high')
        lows = self.extract_prices(candles, 'low')
        
        tenkan_sen = (max(highs[-9:]) + min(lows[-9:])) / 2
        kijun_sen = (max(highs[-26:]) + min(lows[-26:])) / 2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = (max(highs[-52:]) + min(lows[-52:])) / 2
        
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b
        }

    def evaluate_ichimoku_signal(self, candles: List[Dict]) -> int:
        current_price = self.extract_prices(candles)[-1]
        ichimoku = self.calculate_ichimoku_cloud(candles)
        
        if current_price > ichimoku['senkou_span_a'] and current_price > ichimoku['senkou_span_b']:
            return 1  # Bullish
        elif current_price < ichimoku['senkou_span_a'] and current_price < ichimoku['senkou_span_b']:
            return -1  # Bearish
        return 0  # Neutral

    # Fibonacci Methods
    def calculate_fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        diff = high - low
        levels = {
            "23.6%": low + 0.236 * diff,
            "38.2%": low + 0.382 * diff,
            "50.0%": low + 0.5 * diff,
            "61.8%": low + 0.618 * diff,
            "78.6%": low + 0.786 * diff
        }
        return levels

    def evaluate_fibonacci_signal(self, candles: List[Dict]) -> int:
        prices = self.extract_prices(candles)
        high, low = max(prices), min(prices)
        current_price = prices[-1]
        fib_levels = self.calculate_fibonacci_levels(high, low)
        
        for level, price in fib_levels.items():
            if abs(current_price - price) / price < 0.01:  # Within 1% of a Fibonacci level
                return 1 if current_price > price else -1
        return 0

    # Volatility Methods
    def calculate_volatility(self, candles: List[Dict]) -> float:
        prices = self.extract_prices(candles[-20:])
        returns = np.diff(np.log(prices))
        return np.std(returns) * np.sqrt(365)

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

    def compute_atr(self, candles: List[Dict], period: int = 14) -> float:
        high = self.extract_prices(candles, 'high')
        low = self.extract_prices(candles, 'low')
        close = self.extract_prices(candles)
        
        atr = talib.ATR(high, low, close, timeperiod=period)
        
        return atr[-1]

    # Signal Generation and Analysis Methods
    def generate_signal(self, rsi: float, volatility: float) -> str:
        # Adjust RSI levels based on volatility
        if volatility > self.config.volatility_threshold:
            adjusted_overbought = self.config.rsi_overbought + 10  # More room for upside in volatile markets
            adjusted_oversold = self.config.rsi_oversold - 10  # More room for downside in volatile markets
        else:
            adjusted_overbought = self.config.rsi_overbought
            adjusted_oversold = self.config.rsi_oversold

        return "SELL" if rsi > adjusted_overbought else "BUY" if rsi < adjusted_oversold else "HOLD"

    def generate_combined_signal(self, rsi: float, macd: float, signal: float, histogram: float, 
                                 candles: List[Dict], market_conditions: Optional[str] = None) -> str:
        current_price = float(candles[-1]['close'])
        market_conditions = market_conditions or self.analyze_market_conditions(candles)
        volatility_std = self.calculate_volatility(candles)
        signal_strength = self.calculate_signal_strength(rsi, macd, signal, histogram, market_conditions, candles, volatility_std)
        final_signal = self.determine_final_signal(signal_strength, market_conditions, current_price, candles)
        return final_signal

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
            'volume': 1,
            'ichimoku': 0,
            'fibonacci': 0,
            'ml_model': 2,  # Increased weight for ML model
            'bitcoin_prediction': 3  # Assign a weight to the BitcoinPredictionModel signal
        }

        # Adjust weights for bear markets
        if market_conditions in [ "Bear Market", "Bearish"]:
            weights['rsi'] = 1
            weights['macd'] = 1
            weights['bollinger'] = 1
            weights['ma_crossover'] = 1            
            weights['stochastic'] = 1
            weights['trend'] = 1
            weights['volume_profile'] = 1
            weights['short_term_trend'] = 1
            weights['long_term_trend'] = 1
            weights['volume'] = 1            
            weights['ichimoku'] = 0
            weights['fibonacci'] = 0
            weights['ml_model'] = 0
            weights['bitcoin_prediction'] = 0

        signal_strength += weights['rsi'] * self.evaluate_rsi_signal(rsi, volatility_std)
        signal_strength += weights['macd'] * self.evaluate_macd_signal(macd, signal, histogram)
        signal_strength += weights['bollinger'] * self.evaluate_bollinger_signal(candles)
        signal_strength += weights['ma_crossover'] * self.evaluate_ma_crossover_signal(candles)
        signal_strength += weights['stochastic'] * self.evaluate_stochastic_signal(candles)
        signal_strength += weights['trend'] * self.evaluate_trend_signal(candles)
        signal_strength += weights['volume_profile'] * self.evaluate_volume_profile_signal(candles, current_price)
        signal_strength += weights['fibonacci'] * self.evaluate_fibonacci_signal(candles)
        signal_strength += weights['ichimoku'] * self.evaluate_ichimoku_signal(candles)

        # Add ML model signal
        if weights['ml_model'] > 0:
            ml_signal = self.ml_signal.predict_signal(candles)
            self.logger.debug(f"ML signal: {ml_signal}")  # Log the ML signal
            signal_strength += weights['ml_model'] * ml_signal

        # Add BitcoinPredictionModel signal
        if weights['bitcoin_prediction'] > 0:
            # Ensure the model is loaded
            bitcoin_prediction_signal = self.get_bitcoin_prediction_signal(candles)
            signal_strength += weights['bitcoin_prediction'] * bitcoin_prediction_signal

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
        
        self.logger.debug(f"Final signal strength: {signal_strength}")  # Log the final signal strength
        return signal_strength  # Return an integer

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
        short_ma = self.get_moving_average(candles, 10, 'sma')
        long_ma = self.get_moving_average(candles, 50, 'sma')
        ma_trend = "Uptrend" if short_ma > long_ma else "Downtrend"

        ma_200 = self.get_moving_average(candles, 200, 'sma')
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

    # Market Analysis Methods
    def analyze_market_conditions(self, candles: List[Dict]) -> str:
        # Convert 200 days to the appropriate number of intervals
        long_term_period = 200 * self.intervals_per_day

        # Calculate the long-term moving average
        long_term_ma = self.get_moving_average(candles, long_term_period, 'sma')

        # Analyze market conditions based on price action, volume, and volatility
        prices = self.extract_prices(candles)
        volumes = self.extract_prices(candles, 'volume')
        
        # Calculate price change and volume change
        price_change = (prices[-1] - prices[0]) / prices[0]
        volume_change = (volumes[-1] - volumes[0]) / volumes[0]
        
        # Calculate volatility (e.g., using Average True Range)
        volatility = self.compute_atr(candles)
        
        # Calculate percentage drawdown from peak
        peak_price = np.max(prices)
        current_price = prices[-1]
        drawdown = (peak_price - current_price) / peak_price

        # Calculate the shorter-term moving average (e.g., 50-day MA)
        short_term_period = 50 * self.intervals_per_day
        short_term_ma = self.get_moving_average(candles, short_term_period, 'sma')

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
            if np.all(prices[i-5:i] < prices[i]):
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
        # Adjust the volatility calculation based on candle interval
        prices = self.extract_prices(candles)
        returns = np.log(prices[1:] / prices[:-1])
        volatility = np.std(returns[-20:]) * np.sqrt(252 * (self.intervals_per_day / 24))  # Annualized volatility
        
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
        
    def detect_pullback(self, candles: List[Dict]) -> str:
        recent_prices = self.extract_prices(candles)[-5:]
        short_ma = self.get_moving_average(candles, 20, 'sma')
        long_ma = self.get_moving_average(candles, 50, 'sma')
        if recent_prices[-1] < np.min(recent_prices[:-1]) and short_ma > long_ma:
            return "Buy"
        elif recent_prices[-1] > np.max(recent_prices[:-1]) and short_ma < long_ma:
            return "Sell"
        else:
            return "Hold"

    # Position Sizing Methods
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, stop_loss_percent: float) -> float:
        max_loss_amount = account_balance * risk_per_trade
        position_size = max_loss_amount / stop_loss_percent
        return position_size

    def calculate_atr_position_size(self, balance: float, price: float, atr: float) -> float:
        risk_amount = balance * self.config.risk_per_trade
        position_size = risk_amount / (self.config.atr_multiplier * atr)
        return min(position_size * price, balance)  # Ensure we don't exceed available balance

    # Utility Methods
    def extract_prices(self, candles: List[Dict], key: str = 'close') -> np.ndarray:
        """
        Extract prices from candle data.

        :param candles: List of historical candle data.
        :param key: Key to extract from each candle (default is 'close').
        :return: NumPy array of prices.
        """
        return np.array([candle[key] for candle in candles], dtype=float)

    def get_bitcoin_prediction_signal(self, candles: List[Dict]) -> int:
        try:
            # Prepare the data for prediction
            df, X, _ = self.bitcoin_prediction_model.prepare_data(candles)
            
            # Make prediction
            prediction = self.bitcoin_prediction_model.predict(X.iloc[-1:])
            
            # Convert prediction to signal
            current_price = float(candles[-1]['close'])
            predicted_price = prediction[0]
            
            if predicted_price > current_price * 1.01:  # 1% increase
                return 1  # Buy signal
            elif predicted_price < current_price * 0.99:  # 1% decrease
                return -1  # Sell signal
            else:
                return 0  # Hold signal
        except Exception as e:
            self.logger.error(f"Error in BitcoinPredictionModel prediction: {str(e)}. Returning neutral signal.")
            return 0

# ... (any additional classes or functions)