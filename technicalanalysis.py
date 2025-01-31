import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Callable
from coinbaseservice import CoinbaseService
import time
import talib
import logging
from functools import lru_cache
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from ml_model import MLSignal
from historicaldata import HistoricalData
from bitcoinpredictionmodel import BitcoinPredictionModel
import numpy.typing as npt
from enum import Enum
from time import perf_counter
from contextlib import contextmanager

# Move these error classes to the top, before any class that uses them
class TechnicalAnalysisError(Exception):
    """Base exception class for TechnicalAnalysis errors."""
    pass

class InsufficientDataError(TechnicalAnalysisError):
    """Raised when there is not enough data for calculation."""
    pass

def validate_candles(func: Callable):
    """Decorator to validate candle data."""
    def wrapper(self, candles: List[Dict], *args, **kwargs):
        if not candles:
            raise InsufficientDataError("No candle data provided")
        if len(candles) < 2:
            raise InsufficientDataError("Insufficient candle data for analysis")
        try:
            # Validate candle structure
            required_keys = {'open', 'high', 'low', 'close', 'volume'}
            if not all(all(key in candle for key in required_keys) for candle in candles):
                raise ValueError("Invalid candle data structure")
            return func(self, candles, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {str(e)}")
            raise TechnicalAnalysisError(f"Error processing candle data: {str(e)}")
    return wrapper

# Then keep your existing enums and dataclasses
class MarketCondition(Enum):
    BULL_MARKET = "Bull Market"
    BEAR_MARKET = "Bear Market"
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"

class MarketRegime(Enum):
    LOW_VOLATILITY = "Low Volatility"
    MEDIUM_VOLATILITY = "Medium Volatility"
    HIGH_VOLATILITY = "High Volatility"

class SignalType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"

@dataclass
class TechnicalAnalysisConfig:
    # RSI parameters
    rsi_overbought: float = 70  # Changed from 65
    rsi_oversold: float = 30    # Changed from 35
    rsi_period: int = 14
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands parameters
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    
    # Volatility and risk parameters
    volatility_threshold: float = 0.02  # Changed from 0.03
    risk_per_trade: float = 0.02   # Changed from 0.01
    atr_multiplier: float = 2.5     # Changed from 2.0

@dataclass
class SignalResult:
    signal_type: SignalType
    confidence: float
    indicators: Dict[str, float]
    market_condition: MarketCondition
    timestamp: float = field(default_factory=time.time)

class TechnicalAnalysis:
    """
    A class for performing technical analysis on cryptocurrency data.

    This class provides methods for calculating various technical indicators
    and generating trading signals based on those indicators.
    """

    def __init__(self, coinbase_service: CoinbaseService, config: Optional[TechnicalAnalysisConfig] = None, candle_interval: str = 'ONE_HOUR', product_id: str = 'BTC-USDC'):
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
        self.product_id = product_id
        self.intervals_per_day = self.calculate_intervals_per_day()
        historical_data = HistoricalData(coinbase_service.client)
        self.ml_signal = MLSignal(self.logger, historical_data, product_id=self.product_id, granularity=self.candle_interval)
        self.ml_signal.load_model()
        self.scaler = StandardScaler()
        self.bitcoin_prediction_model = BitcoinPredictionModel(coinbase_service, product_id=self.product_id, granularity=self.candle_interval)
        self.bitcoin_prediction_model.load_model()
        
        # Set product-specific parameters
        self.set_product_specific_parameters()

        # Add signal stability tracking
        self.last_signal = None
        self.last_signal_time = None
        self.signal_history = []  # Store recent signals
        self.min_signal_duration = 4  # Minimum candles before signal can change
        self.trend_confirmation_period = 3  # Number of candles needed to confirm trend

    def set_product_weights(self):
        """Set product-specific signal weights."""
        base_weights = {
            'rsi': 2.5,
            'macd': 2.5,
            'bollinger': 1.5,
            'ma_crossover': 1.5,
            'stochastic': 1.0,
            'trend': 2.5,
            'volume_profile': 1.5,
            'short_term_trend': 2.5,
            'long_term_trend': 1.5,
            'volume': 1.5,
            'ichimoku': 1.5,
            'fibonacci': 1.0,
            'ml_model': 2.5,
            'bitcoin_prediction': 3.5,
            'adx': 2.0
        }

        # Product-specific adjustments
        if self.product_id == 'BTC-USDC':
            base_weights['ml_model'] = 2.5
            base_weights['bitcoin_prediction'] = 3.5

            base_weights['long_term_trend'] = 2.5
            base_weights['trend'] = 3.0
            
        elif self.product_id == 'ETH-USDC':
            base_weights['rsi'] = 3.0
            base_weights['macd'] = 3.0
            base_weights['short_term_trend'] = 3.0
            base_weights['volume'] = 2.0
            base_weights['bitcoin_prediction'] = 3.0
            
        elif self.product_id == 'SOL-USDC':
            base_weights['volume'] = 2.5
            base_weights['short_term_trend'] = 3.0
            base_weights['volatility'] = 2.0
            base_weights['bitcoin_prediction'] = 3.0
            
        elif self.product_id == 'XRP-USDC':
            base_weights['volume'] = 2.5
            base_weights['news_sentiment'] = 2.0
            base_weights['rsi'] = 2.0
            base_weights['macd'] = 2.0
            base_weights['bitcoin_prediction'] = 2.5

        return base_weights

    def set_product_specific_parameters(self):
        """Set product-specific parameters based on historical behavior and volatility."""
        if self.product_id == 'BTC-USDC':
            self.config.rsi_overbought = 70
            self.config.rsi_oversold = 30
            self.config.volatility_threshold = 0.025
            self.config.rsi_period = 14
            self.config.macd_fast = 12
            self.config.macd_slow = 26
            self.config.macd_signal = 9
            self.config.bollinger_window = 20
            self.config.bollinger_std = 2.0
            self.config.risk_per_trade = 0.02
            self.config.atr_multiplier = 2.5
            
        elif self.product_id == 'ETH-USDC':
            self.config.rsi_overbought = 75
            self.config.rsi_oversold = 25
            self.config.volatility_threshold = 0.03
            self.config.rsi_period = 12
            self.config.macd_fast = 10
            self.config.macd_slow = 24
            self.config.macd_signal = 8
            self.config.bollinger_window = 18
            self.config.bollinger_std = 2.2
            self.config.risk_per_trade = 0.025
            self.config.atr_multiplier = 2.2
            
        elif self.product_id == 'SOL-USDC':
            self.config.rsi_overbought = 75
            self.config.rsi_oversold = 25
            self.config.volatility_threshold = 0.035
            self.config.rsi_period = 10
            self.config.macd_fast = 8
            self.config.macd_slow = 21
            self.config.macd_signal = 7
            self.config.bollinger_window = 15
            self.config.bollinger_std = 2.5
            self.config.risk_per_trade = 0.015
            self.config.atr_multiplier = 3.0
            
        elif self.product_id == 'XRP-USDC':
            self.config.rsi_overbought = 80
            self.config.rsi_oversold = 20
            self.config.volatility_threshold = 0.04
            self.config.rsi_period = 10
            self.config.macd_fast = 8
            self.config.macd_slow = 21
            self.config.macd_signal = 7
            self.config.bollinger_window = 15
            self.config.bollinger_std = 2.8
            self.config.risk_per_trade = 0.015
            self.config.atr_multiplier = 3.0

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
        """Compute MACD values."""
        try:
            prices = self.extract_prices(candles)
            self.logger.debug(f"Number of prices for MACD calculation: {len(prices)}")
            
            # MACD requires at least slow period + signal period candles
            min_periods = self.config.macd_slow + self.config.macd_signal
            if len(prices) < min_periods:
                self.logger.error(f"Insufficient data for MACD calculation. Need {min_periods} periods, got {len(prices)}")
                raise InsufficientDataError(f"Need at least {min_periods} candles for MACD calculation")
            
            # Convert prices to numpy array and check for valid values
            prices_array = np.array(prices, dtype=float)
            if np.any(np.isnan(prices_array)) or np.any(np.isinf(prices_array)):
                self.logger.error("Invalid price values detected in MACD calculation")
                raise ValueError("Invalid price values in data")
            
            # Log some price statistics for debugging
            self.logger.debug(f"Price range: {min(prices_array)} to {max(prices_array)}")
            
            # Calculate EMAs
            ema_fast = talib.EMA(prices_array, timeperiod=self.config.macd_fast)
            ema_slow = talib.EMA(prices_array, timeperiod=self.config.macd_slow)
            
            # Calculate MACD line (the absolute difference between EMAs)
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = talib.EMA(macd_line, timeperiod=self.config.macd_signal)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Get the last valid values
            macd_value = float(macd_line[-1])
            signal_value = float(signal_line[-1])
            histogram_value = float(histogram[-1])
            
            # Log raw values for debugging
            self.logger.debug(f"Last price: {prices_array[-1]}")
            self.logger.debug(f"Fast EMA: {ema_fast[-1]}")
            self.logger.debug(f"Slow EMA: {ema_slow[-1]}")
            self.logger.debug(f"Raw MACD values - Last 5 entries:")
            self.logger.debug(f"MACD: {macd_line[-5:]}")
            self.logger.debug(f"Signal: {signal_line[-5:]}")
            self.logger.debug(f"Histogram: {histogram[-5:]}")
            
            # Scale the values to be more meaningful (as a percentage of price)
            price_scale = prices_array[-1] * 0.01  # 1% of current price as scale
            macd_value = macd_value / price_scale
            signal_value = signal_value / price_scale
            histogram_value = histogram_value / price_scale
            
            self.logger.info(f"Final MACD values - MACD: {macd_value:.2f}, Signal: {signal_value:.2f}, Histogram: {histogram_value:.2f}")
            return macd_value, signal_value, histogram_value
            
        except Exception as e:
            self.logger.error(f"Error computing MACD: {str(e)}")
            return 0.0, 0.0, 0.0

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
        prices = self.extract_prices(candles)
        adjusted_period = min(max(1, int(period * self.intervals_per_day / 24)), len(prices))

        self.logger.debug(f"Calculating {ma_type.upper()} with period {adjusted_period}")
        self.logger.debug(f"Number of prices: {len(prices)}")

        try:
            if ma_type == 'sma':
                return talib.SMA(prices, timeperiod=adjusted_period)[-1]
            elif ma_type == 'ema':
                return talib.EMA(prices, timeperiod=adjusted_period)[-1]
            else:
                raise ValueError(f"Unsupported moving average type: {ma_type}")
        except Exception as e:
            self.logger.error(f"Error calculating {ma_type.upper()}: {str(e)}")
            return np.mean(prices)  # Fallback to simple mean if TA-Lib calculation fails

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
        
        try:
            signal_strength = self.calculate_signal_strength(rsi, macd, signal, histogram, market_conditions, candles, volatility_std)
            final_signal = self.determine_final_signal(signal_strength, market_conditions, current_price, candles)
        except Exception as e:
            self.logger.error(f"Error generating combined signal: {str(e)}")
            final_signal = "HOLD"  # Default to HOLD if there's an error
        
        return final_signal

    def calculate_signal_strength(self, rsi: float, macd: float, signal: float, histogram: float, market_conditions: str,
                                  candles: List[Dict], volatility_std: float) -> int:
        current_price = float(candles[-1]['close'])
        signal_strength = 0

        weights = self.set_product_weights()
        # Adjust weights for bear markets - maintain some signal strength while being conservative
        if market_conditions in ["Bear Market", "Bearish"]:
            weights['rsi'] = 1.5
            weights['macd'] = 1.5
            weights['bollinger'] = 1.0
            weights['ma_crossover'] = 1.0            
            weights['stochastic'] = 0.5
            weights['trend'] = 0.5
            weights['volume_profile'] = 1.0
            weights['short_term_trend'] = 0.5
            weights['long_term_trend'] = 1.0
            weights['volume'] = 1.0            
            weights['ichimoku'] = 0.5
            weights['fibonacci'] = 0.5
            weights['ml_model'] = 0.5
            weights['bitcoin_prediction'] = 0.5

        # Adjust weights for bear markets
        # if market_conditions in [ "Bear Market", "Bearish"]:
        #     weights['rsi'] = 1
        #     weights['macd'] = 1 
        #     weights['bollinger'] = 1
        #     weights['ma_crossover'] = 1            
        #     weights['stochastic'] = 0 #1
        #     weights['trend'] = 0 #1
        #     weights['volume_profile'] = 1
        #     weights['short_term_trend'] = 0 #1
        #     weights['long_term_trend'] = 1
        #     weights['volume'] = 1            
        #     weights['ichimoku'] = 0
        #     weights['fibonacci'] = 0
        #     weights['ml_model'] = 0
        #     weights['bitcoin_prediction'] = 0


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

        if np.isnan(long_term_ma):
            self.logger.warning("Unable to calculate long-term MA. Returning 'Neutral' market condition.")
            return "Neutral"

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

    def get_combined_signal(self, candles: List[Dict]) -> SignalResult:
        """
        Generate a comprehensive trading signal with confidence level.
        """
        try:
            # Calculate all indicators
            indicators = self._calculate_all_indicators(candles)
            
            # Get market conditions
            market_condition = self.analyze_market_conditions(candles)
            
            # Calculate weighted signal
            signal_strength = self._calculate_weighted_signal(indicators, market_condition)
            
            # Calculate base confidence from signal strength
            base_confidence = min(abs(signal_strength) / 5.0, 1.0)
            
            # Adjust confidence based on market conditions
            condition_multipliers = {
                "Bull Market": 1.2,
                "Bear Market": 0.9,
                "Bullish": 1.1,
                "Bearish": 0.9,
                "Neutral": 1.0
            }
            confidence_multiplier = condition_multipliers.get(market_condition, 1.0)
            
            # Calculate final confidence
            confidence = base_confidence * confidence_multiplier
            
            # Ensure minimum confidence unless truly neutral
            if confidence < 0.1 and abs(signal_strength) > 0.1:
                confidence = 0.1
            
            # Add trend confirmation bonus
            if len(self.signal_history) >= self.trend_confirmation_period:
                recent_signals = [s['strength'] for s in self.signal_history]
                
                # Check if all recent signals are in the same direction
                if all(s > 0 for s in recent_signals) or all(s < 0 for s in recent_signals):
                    # Trend is confirmed - strengthen signal
                    confidence = min(confidence * 1.2, 1.0)
            
            # Determine signal type
            signal_type = self._determine_signal_type(signal_strength)
            
            return SignalResult(
                signal_type=signal_type,
                confidence=confidence,
                indicators=indicators,
                market_condition=MarketCondition(market_condition)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating combined signal: {str(e)}")
            # Return a low confidence neutral signal instead of zero confidence
            return SignalResult(
                signal_type=SignalType.HOLD,
                confidence=0.1,
                indicators={},
                market_condition=MarketCondition.NEUTRAL
            )

    @validate_candles
    def _calculate_all_indicators(self, candles: List[Dict]) -> Dict[str, float]:
        """Calculate all technical indicators."""
        try:
            # Get current price for normalization
            current_price = float(candles[-1]['close'])
            
            # Calculate RSI
            rsi = self.compute_rsi(self.product_id, candles)
            if np.isnan(rsi):
                rsi = 50.0  # Neutral value if calculation fails
            
            # Calculate MACD
            macd, signal, histogram = self.compute_macd(self.product_id, candles)
            if np.isnan(macd) or np.isnan(signal) or np.isnan(histogram):
                macd, signal, histogram = 0.0, 0.0, 0.0
            
            # Calculate Bollinger signal
            bollinger_signal = self.evaluate_bollinger_signal(candles)
            
            # Calculate ADX and trend
            adx, trend_direction = self.get_trend_strength(candles)
            adx_signal = 1.0 if trend_direction == "Uptrend" and adx > 25 else \
                        -1.0 if trend_direction == "Downtrend" and adx > 25 else 0.0
            
            # Calculate MA crossover
            ma_signal = self.evaluate_ma_crossover_signal(candles)
            
            # Create indicators dictionary with all values
            indicators = {
                'rsi': float(rsi),
                'macd': float(macd),
                'macd_signal': float(signal),
                'macd_histogram': float(histogram),
                'bollinger': float(bollinger_signal),
                'adx': float(adx_signal),
                'ma_crossover': float(ma_signal),
                'current_price': current_price,
                'trend_direction': trend_direction
            }
            
            # Log indicator values for debugging
            self.logger.debug(f"Calculated indicators: {indicators}")
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            # Return neutral values instead of empty dict
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0,
                'bollinger': 0.0,
                'adx': 0.0,
                'ma_crossover': 0.0,
                'current_price': float(candles[-1]['close']) if candles else 0.0,
                'trend_direction': "Unknown"
            }

    @contextmanager
    def performance_monitor(self, operation_name: str):
        """Context manager to monitor performance of operations."""
        start_time = perf_counter()
        try:
            yield
        finally:
            execution_time = perf_counter() - start_time
            self.logger.debug(f"{operation_name} took {execution_time:.4f} seconds")

    def analyze_market(self, candles: List[Dict]) -> Dict[str, Union[SignalResult, Dict]]:
        """
        Perform complete market analysis with performance monitoring.
        """
        with self.performance_monitor("Market Analysis"):
            signal_result = self.get_combined_signal(candles)
            
        with self.performance_monitor("Risk Analysis"):
            risk_metrics = self._calculate_risk_metrics(candles)
            
        return {
            'signal': signal_result,
            'risk_metrics': risk_metrics,
            'timestamp': time.time()
        }

    def _calculate_risk_metrics(self, candles: List[Dict]) -> Dict[str, float]:
        """
        Calculate risk metrics for the current market conditions.
        
        Args:
            candles: List of historical candle data
        
        Returns:
            Dictionary containing various risk metrics
        """
        try:
            current_price = float(candles[-1]['close'])
            atr = self.compute_atr(candles)
            volatility = self.calculate_volatility(candles)
            
            # Calculate risk metrics
            risk_metrics = {
                'atr': atr,
                'volatility': volatility,
                'risk_level': self._calculate_risk_level(volatility),
                'stop_loss': current_price - (atr * self.config.atr_multiplier),
                'take_profit': current_price + (atr * self.config.atr_multiplier * 1.5),
                'max_position_size': self.calculate_atr_position_size(
                    balance=10000,  # This should be passed in or stored as a property
                    price=current_price,
                    atr=atr
                )
            }
            
            return risk_metrics
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'atr': 0.0,
                'volatility': 0.0,
                'risk_level': 'high',
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'max_position_size': 0.0
            }

    def _calculate_risk_level(self, volatility: float) -> str:
        """
        Determine risk level based on volatility.
        """
        if volatility > self.config.volatility_threshold * 2:
            return 'high'
        elif volatility > self.config.volatility_threshold:
            return 'medium'
        else:
            return 'low'

    # Add these methods to the TechnicalAnalysis class, keeping all existing code

    def compute_adx(self, candles: List[Dict], period: int = 14) -> float:
        """
        Calculate the Average Directional Index (ADX)
        
        Args:
            candles: List of candle data
            period: Period for ADX calculation (default 14)
        
        Returns:
            float: ADX value between 0 and 100
        """
        try:
            # Extract price data
            high = self.extract_prices(candles, 'high')
            low = self.extract_prices(candles, 'low')
            close = self.extract_prices(candles)

            # Use TA-Lib's ADX function
            adx = talib.ADX(high, low, close, timeperiod=period)
            
            # Return the latest ADX value
            latest_adx = adx[-1]
            
            # Handle NaN values
            if np.isnan(latest_adx):
                self.logger.warning("ADX calculation returned NaN. Returning 0.")
                return 0.0
                
            return float(latest_adx)
            
        except Exception as e:
            self.logger.error(f"Error computing ADX: {str(e)}")
            return 0.0

    def evaluate_adx_signal(self, candles: List[Dict], period: int = 14) -> int:
        """
        Evaluate ADX signal strength and trend direction
        
        Args:
            candles: List of candle data
            period: Period for ADX calculation
        
        Returns:
            int: Signal strength (-1 to 1)
        """
        try:
            adx = self.compute_adx(candles, period)
            
            # Calculate DI+ and DI- using TA-Lib
            high = self.extract_prices(candles, 'high')
            low = self.extract_prices(candles, 'low')
            close = self.extract_prices(candles)
            
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)[-1]
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)[-1]
            
            # Strong trend if ADX > 25
            if adx > 25:
                if plus_di > minus_di:
                    return 1  # Strong uptrend
                elif minus_di > plus_di:
                    return -1  # Strong downtrend
                    
            # Weak trend if ADX < 20
            elif adx < 20:
                return 0  # No clear trend
                
            # Moderate trend
            else:
                if plus_di > minus_di:
                    return 0.5  # Moderate uptrend
                elif minus_di > plus_di:
                    return -0.5  # Moderate downtrend
                    
            return 0
            
        except Exception as e:
            self.logger.error(f"Error evaluating ADX signal: {str(e)}")
            return 0

    def get_trend_strength(self, candles: List[Dict]) -> Tuple[float, str]:
        """
        Get trend strength and direction using ADX and DI indicators
        
        Args:
            candles: List of candle data
        
        Returns:
            Tuple[float, str]: (ADX value, trend direction)
        """
        try:
            adx = self.compute_adx(candles)
            
            # Calculate DI+ and DI-
            high = self.extract_prices(candles, 'high')
            low = self.extract_prices(candles, 'low')
            close = self.extract_prices(candles)
            
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)[-1]
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)[-1]
            
            # Determine trend direction
            if plus_di > minus_di:
                direction = "Uptrend"
            elif minus_di > plus_di:
                direction = "Downtrend"
            else:
                direction = "Sideways"
                
            return adx, direction
            
        except Exception as e:
            self.logger.error(f"Error getting trend strength: {str(e)}")
            return 0.0, "Unknown"

    def _calculate_weighted_signal(self, indicators: Dict[str, float], market_condition: str) -> float:
        """Calculate weighted signal with stability checks."""
        try:
            # Calculate base signal
            signal_strength = self._calculate_base_signal(indicators, market_condition)
            
            # Get current time
            current_time = time.time()
            
            # Check if we have a previous signal
            if self.last_signal is not None and self.last_signal_time is not None:
                # Calculate time since last signal change
                time_since_last_signal = current_time - self.last_signal_time
                
                # If not enough time has passed, bias towards previous signal
                if time_since_last_signal < (self.min_signal_duration * self.get_candle_duration()):
                    # Bias towards previous signal but don't completely negate the new signal
                    if (signal_strength > 0 and self.last_signal < 0) or (signal_strength < 0 and self.last_signal > 0):
                        # Average with previous signal instead of reducing by half
                        signal_strength = (signal_strength + self.last_signal) / 2
                        
            # Add to signal history
            self.signal_history.append({
                'time': current_time,
                'strength': signal_strength
            })
            
            # Keep only recent history
            self.signal_history = self.signal_history[-self.trend_confirmation_period:]
            
            # Check for trend confirmation
            if len(self.signal_history) >= self.trend_confirmation_period:
                recent_signals = [s['strength'] for s in self.signal_history]
                
                # Check if all recent signals are in the same direction
                if all(s > 0 for s in recent_signals) or all(s < 0 for s in recent_signals):
                    # Trend is confirmed - strengthen signal
                    signal_strength *= 1.2
                elif any(s > 0 for s in recent_signals) and any(s < 0 for s in recent_signals):
                    # Mixed signals - reduce strength less aggressively
                    signal_strength *= 0.9
            
            # Update last signal tracking
            self.last_signal = signal_strength
            self.last_signal_time = current_time
            
            # Ensure signal strength is not zero unless truly neutral
            if abs(signal_strength) < 0.1:
                # Check if there's any directional bias
                if any(s['strength'] > 0 for s in self.signal_history[-3:]):
                    signal_strength = 0.2
                elif any(s['strength'] < 0 for s in self.signal_history[-3:]):
                    signal_strength = -0.2
            
            return signal_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating weighted signal: {str(e)}")
            # Return a small non-zero value instead of 0
            return 0.1

    def get_candle_duration(self) -> int:
        """Get candle duration in seconds."""
        durations = {
            'ONE_MINUTE': 60,
            'FIVE_MINUTE': 300,
            'FIFTEEN_MINUTE': 900,
            'THIRTY_MINUTE': 1800,
            'ONE_HOUR': 3600,
            'TWO_HOUR': 7200,
            'SIX_HOUR': 21600,
            'ONE_DAY': 86400
        }
        return durations.get(self.candle_interval, 3600)

    def _calculate_base_signal(self, indicators: Dict[str, float], market_condition: str) -> float:
        """Calculate the base signal from technical indicators."""
        try:
            signal_strength = 0.0
            indicator_count = 0
            
            # RSI Signal (-1 to 1)
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if rsi > 70:
                    signal_strength -= 1.0
                elif rsi < 30:
                    signal_strength += 1.0
                else:
                    signal_strength += (rsi - 50) / 20  # Normalized between -1 and 1
                indicator_count += 1

            # MACD Signal (-1 to 1)
            if all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
                macd = indicators['macd']
                signal = indicators['macd_signal']
                histogram = indicators['macd_histogram']
                
                # Normalize histogram to -1 to 1 range
                if abs(macd) > 0:
                    macd_strength = histogram / (abs(macd) + 0.00001)
                    macd_strength = max(min(macd_strength, 1), -1)
                    signal_strength += macd_strength
                    indicator_count += 1

            # Bollinger Bands Signal (-1 to 1)
            if 'bollinger' in indicators:
                signal_strength += indicators['bollinger']
                indicator_count += 1

            # ADX Signal (-1 to 1)
            if 'adx' in indicators:
                signal_strength += indicators['adx']
                indicator_count += 1

            # Moving Average Crossover Signal (-1 to 1)
            if 'ma_crossover' in indicators:
                signal_strength += indicators['ma_crossover']
                indicator_count += 1

            # Average the signals if we have any
            if indicator_count > 0:
                signal_strength = signal_strength / indicator_count
            
            # Apply market condition multiplier
            condition_multipliers = {
                "Bull Market": 1.2,
                "Bear Market": 0.8,
                "Bullish": 1.1,
                "Bearish": 0.9,
                "Neutral": 1.0
            }
            multiplier = condition_multipliers.get(market_condition, 1.0)
            signal_strength *= multiplier

            # Scale to final range (-10 to 10)
            signal_strength *= 5.0  # Scale from -1:1 to -5:5
            
            # Add minimum signal strength if there's a clear direction
            if 0 < signal_strength < 0.2:
                signal_strength = 0.2
            elif -0.2 < signal_strength < 0:
                signal_strength = -0.2

            self.logger.debug(f"Base signal strength calculated: {signal_strength}")
            return signal_strength

        except Exception as e:
            self.logger.error(f"Error calculating base signal: {str(e)}")
            return 0.1  # Return small non-zero value instead of 0

    def _determine_signal_type(self, signal_strength: float) -> SignalType:
        """
        Determine signal type based on signal strength.
        More balanced thresholds for consistent signals.
        
        Args:
            signal_strength: Float value between -10 and 10
            
        Returns:
            SignalType: The determined signal type
        """
        # Use more sensitive thresholds
        if signal_strength >= 3.0:  # Reduced from 7
            return SignalType.STRONG_BUY
        elif signal_strength >= 1.0:  # Reduced from 3
            return SignalType.BUY
        elif signal_strength <= -3.0:  # Changed from -7
            return SignalType.STRONG_SELL
        elif signal_strength <= -1.0:  # Changed from -3
            return SignalType.SELL
        else:
            # More sensitive to slight biases
            if signal_strength > 0.2:  # Reduced from 0.5
                return SignalType.BUY
            elif signal_strength < -0.2:  # Changed from -0.5
                return SignalType.SELL
            return SignalType.HOLD

    def detect_consolidation(self, candles: List[Dict], window: int = 20, threshold: float = 0.02) -> Dict[str, Union[bool, str]]:
        """
        Detect consolidation and analyze volume at support/resistance levels.
        """
        try:
            # Get recent prices and volumes
            prices = self.extract_prices(candles[-window:])
            highs = self.extract_prices(candles[-window:], 'high')
            lows = self.extract_prices(candles[-window:], 'low')
            closes = self.extract_prices(candles[-window:], 'close')
            volumes = self.extract_prices(candles[-window:], 'volume')
            timestamps = [candle['time'] for candle in candles[-window:]]
            
            # Calculate price ranges
            current_price = prices[-1]
            
            # Calculate average volume and volume trend
            avg_volume = sum(volumes[:-3]) / (len(volumes) - 3)
            recent_volume = sum(volumes[-3:]) / 3
            volume_increase = recent_volume > avg_volume * 1.5
            
            # Find recent swing high and low points
            swing_high = max(highs[-10:])  # Last 10 candles for recent levels
            swing_low = min(lows[-10:])
            
            # Calculate price range and determine if consolidating
            price_range = (swing_high - swing_low) / swing_low
            is_consolidating = price_range <= threshold
            
            # Initialize pattern and strength
            pattern = "None"
            strength = 0.0
            
            # Determine pattern if consolidating
            if is_consolidating:
                if current_price > swing_high and volume_increase:
                    pattern = "Breakout"
                elif current_price < swing_low and volume_increase:
                    pattern = "Breakdown"
                else:
                    pattern = "Consolidating"
                strength = 1.0 - (price_range / threshold)
            
            # Track rejection events with timestamps
            rejection_event = None
            
            # Check for resistance rejection (in last 5 candles)
            for i in range(-5, 0):
                if highs[i] >= swing_high * 0.998:  # Within 0.2% of recent high
                    if current_price < highs[i]:  # Price pulled back from high
                        # Check for confirmation
                        candles_since_rejection = abs(i)
                        confirming_candles = 0
                        volume_confirmation = False
                        avg_volume_before = np.mean(volumes[i-3:i])  # Average volume before rejection
                        
                        # Check subsequent candles for confirmation using closes
                        for j in range(i+1, 0):
                            if highs[j] < highs[i] and closes[j] < closes[i]:  # Now closes is defined
                                confirming_candles += 1
                            if volumes[j] > avg_volume_before * 1.2:  # 20% above average
                                volume_confirmation = True
                        
                        rejection_volume = volumes[i]
                        rejection_volume_ratio = rejection_volume / avg_volume
                        
                        rejection_event = {
                            'type': 'resistance',
                            'price_level': swing_high,
                            'timestamp': timestamps[i],
                            'volume': rejection_volume,
                            'volume_ratio': rejection_volume_ratio,
                            'price': current_price,
                            'distance_from_level': ((swing_high - current_price) / current_price) * 100,
                            'candles_ago': candles_since_rejection,
                            'confirming_candles': confirming_candles,
                            'volume_confirmation': volume_confirmation,
                            'avg_volume_before': avg_volume_before
                        }
                        break

            # Check for support bounce (in last 5 candles)
            if not rejection_event:
                for i in range(-5, 0):
                    if lows[i] <= swing_low * 1.002:  # Within 0.2% of recent low
                        if current_price > lows[i]:  # Price bounced from low
                            # Check for confirmation
                            candles_since_bounce = abs(i)
                            confirming_candles = 0
                            volume_confirmation = False
                            avg_volume_before = np.mean(volumes[i-3:i])  # Average volume before bounce
                            
                            # Check subsequent candles for confirmation using closes
                            for j in range(i+1, 0):
                                if lows[j] > lows[i] and closes[j] > closes[i]:  # Now closes is defined
                                    confirming_candles += 1
                                if volumes[j] > avg_volume_before * 1.2:  # 20% above average
                                    volume_confirmation = True
                            
                            rejection_volume = volumes[i]
                            rejection_volume_ratio = rejection_volume / avg_volume
                            
                            rejection_event = {
                                'type': 'support',
                                'price_level': swing_low,
                                'timestamp': timestamps[i],
                                'volume': rejection_volume,
                                'volume_ratio': rejection_volume_ratio,
                                'price': current_price,
                                'distance_from_level': ((current_price - swing_low) / current_price) * 100,
                                'candles_ago': candles_since_bounce,
                                'confirming_candles': confirming_candles,
                                'volume_confirmation': volume_confirmation,
                                'avg_volume_before': avg_volume_before
                            }
                            break
            
            return {
                'is_consolidating': is_consolidating,
                'pattern': pattern,
                'strength': strength,
                'upper_channel': swing_high,
                'lower_channel': swing_low,
                'channel_middle': (swing_high + swing_low) / 2,
                'volume_confirmed': volume_increase,
                'price_range': price_range,
                'rejection_event': rejection_event,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting consolidation: {str(e)}")
            return {
                'is_consolidating': False,
                'pattern': "Error",
                'strength': 0.0,
                'upper_channel': 0.0,
                'lower_channel': 0.0,
                'channel_middle': 0.0,
                'volume_confirmed': False,
                'price_range': 0.0,
                'rejection_event': None,
                'avg_volume': 0.0
            }
        
    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        macd, signal, histogram = talib.MACD(np.array(prices), 
                                             fastperiod=self.config.macd_fast, 
                                             slowperiod=self.config.macd_slow, 
                                             signalperiod=self.config.macd_signal)
        return macd[-1], signal[-1], histogram[-1]


    def _adjust_signal_for_consolidation(self, signal_strength: float, consolidation_info: Dict) -> float:
        """Adjust signal strength based on consolidation pattern."""
        if consolidation_info['is_consolidating']:
            if consolidation_info['pattern'] == "Breakout":
                return min(signal_strength * 1.5, 10.0)  # Amplify bullish signals
            elif consolidation_info['pattern'] == "Breakdown":
                return max(signal_strength * 1.5, -10.0)  # Amplify bearish signals
            else:
                return signal_strength * 0.5  # Reduce signal strength during consolidation
        return signal_strength

    def analyze_volume_confirmation(self, candles: List[Dict], lookback: int = 20) -> Dict[str, Union[bool, float, str]]:
        """
        Analyze volume patterns for trade confirmation.
        
        Args:
            candles: List of candle data
            lookback: Number of periods to analyze
        
        Returns:
            Dict containing volume analysis results
        """
        try:
            # Get recent data
            recent_candles = candles[-lookback:]
            volumes = self.extract_prices(recent_candles, 'volume')
            prices = self.extract_prices(recent_candles, 'close')
            
            # Calculate average volume
            avg_volume = np.mean(volumes[:-1])  # Excluding current volume
            current_volume = volumes[-1]
            
            # Calculate volume change
            volume_change = ((current_volume - avg_volume) / avg_volume) * 100
            
            # Calculate price change
            price_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100
            
            # Determine if volume is confirming price movement
            is_confirming = (price_change > 0 and volume_change > 20) or \
                           (price_change < 0 and volume_change > 20)
            
            # Calculate volume trend
            volume_sma = np.mean(volumes[-5:])  # 5-period volume SMA
            volume_trend = "Increasing" if volume_sma > avg_volume else "Decreasing"
            
            # Determine volume signal strength
            if volume_change > 100:  # Volume spike (2x average)
                strength = "Very Strong"
            elif volume_change > 50:
                strength = "Strong"
            elif volume_change > 20:
                strength = "Moderate"
            else:
                strength = "Weak"
                
            return {
                'is_confirming': is_confirming,
                'volume_change': volume_change,
                'volume_trend': volume_trend,
                'strength': strength,
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'price_change': price_change
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume confirmation: {str(e)}")
            return {
                'is_confirming': False,
                'volume_change': 0.0,
                'volume_trend': "Unknown",
                'strength': "Unknown",
                'current_volume': 0.0,
                'average_volume': 0.0,
                'price_change': 0.0
            }

# ... (any additional classes or functions)

def cache_result(ttl_seconds: int = 300):
    """Cache decorator with time-to-live."""
    def decorator(func: Callable):
        cache = {}
        
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < ttl_seconds:
                    return result
                
            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result
            
        return wrapper
    return decorator

