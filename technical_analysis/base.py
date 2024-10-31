from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
import talib
import logging

class SignalType(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING UP"
    TRENDING_DOWN = "TRENDING DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH VOLATILITY"
    LOW_VOLATILITY = "LOW VOLATILITY"

@dataclass
class TechnicalIndicatorConfig:
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR
    atr_period: int = 14
    
    # Volatility
    volatility_period: int = 20
    volatility_threshold: float = 0.02

@dataclass
class SignalResult:
    signal_type: SignalType
    confidence: float
    indicators: Dict[str, float]
    market_regime: MarketRegime
    timestamp: float

class BaseTechnicalAnalysis(ABC):
    """Base class for technical analysis implementations."""
    
    def __init__(self, config: Optional[TechnicalIndicatorConfig] = None):
        self.config = config or TechnicalIndicatorConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self, candles: List[Dict]) -> SignalResult:
        """Main analysis method to be implemented by concrete classes."""
        pass
    
    def validate_data(self, candles: List[Dict]) -> bool:
        """Validate input data structure."""
        if not candles or len(candles) < 2:
            return False
        required_keys = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        return all(all(key in candle for key in required_keys) for candle in candles)
    
    def extract_prices(self, candles: List[Dict], key: str = 'close') -> np.ndarray:
        """Extract price data from candles."""
        return np.array([float(candle[key]) for candle in candles])
    
    def calculate_rsi(self, candles: List[Dict]) -> float:
        """Calculate RSI indicator."""
        prices = self.extract_prices(candles)
        rsi = talib.RSI(prices, timeperiod=self.config.rsi_period)
        return rsi[-1]
    
    def calculate_macd(self, candles: List[Dict]) -> Tuple[float, float, float]:
        """Calculate MACD indicator."""
        prices = self.extract_prices(candles)
        macd, signal, hist = talib.MACD(
            prices,
            fastperiod=self.config.macd_fast,
            slowperiod=self.config.macd_slow,
            signalperiod=self.config.macd_signal
        )
        return macd[-1], signal[-1], hist[-1]
    
    def calculate_bollinger_bands(self, candles: List[Dict]) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        prices = self.extract_prices(candles)
        upper, middle, lower = talib.BBANDS(
            prices,
            timeperiod=self.config.bb_period,
            nbdevup=self.config.bb_std,
            nbdevdn=self.config.bb_std
        )
        return upper[-1], middle[-1], lower[-1]
    
    def calculate_atr(self, candles: List[Dict]) -> float:
        """Calculate Average True Range."""
        high = self.extract_prices(candles, 'high')
        low = self.extract_prices(candles, 'low')
        close = self.extract_prices(candles)
        atr = talib.ATR(high, low, close, timeperiod=self.config.atr_period)
        return atr[-1]
    
    def calculate_volatility(self, candles: List[Dict]) -> float:
        """Calculate price volatility."""
        prices = self.extract_prices(candles)
        returns = np.log(prices[1:] / prices[:-1])
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def identify_market_regime(self, candles: List[Dict]) -> MarketRegime:
        """Identify current market regime."""
        volatility = self.calculate_volatility(candles)
        atr = self.calculate_atr(candles)
        prices = self.extract_prices(candles)
        
        # Calculate trend using simple moving averages
        short_ma = talib.SMA(prices, timeperiod=20)[-1]
        long_ma = talib.SMA(prices, timeperiod=50)[-1]
        
        if volatility > self.config.volatility_threshold * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < self.config.volatility_threshold * 0.5:
            return MarketRegime.LOW_VOLATILITY
        elif short_ma > long_ma * 1.02:
            return MarketRegime.TRENDING_UP
        elif short_ma < long_ma * 0.98:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING 