import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd
import json
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis, SignalType, MarketCondition, TechnicalAnalysisConfig
from config import API_KEY, API_SECRET, NEWS_API_KEY
from historicaldata import HistoricalData
from coinbase.rest import RESTClient
import argparse
import time
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from newsapi import NewsApiClient

class EnhancedJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Enum and other custom types."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (datetime, timedelta)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

class AdvancedPatternType(Enum):
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    HEAD_SHOULDERS = "Head and Shoulders"
    INV_HEAD_SHOULDERS = "Inverse Head and Shoulders"
    ASCENDING_TRIANGLE = "Ascending Triangle"
    DESCENDING_TRIANGLE = "Descending Triangle"
    SYMMETRICAL_TRIANGLE = "Symmetrical Triangle"
    BULL_FLAG = "Bullish Flag"
    BEAR_FLAG = "Bearish Flag"
    CUP_HANDLE = "Cup and Handle"
    RISING_WEDGE = "Rising Wedge"
    FALLING_WEDGE = "Falling Wedge"
    NONE = "None"

@dataclass
class MarketRegime:
    type: str
    confidence: float
    volatility: float
    trend_strength: float
    mean_reversion: float

class MarketState(Enum):
    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    BREAKOUT = "Breakout"
    REVERSAL = "Reversal"
    ACCUMULATION = "Accumulation"
    DISTRIBUTION = "Distribution"

class AdvancedMarketAnalyzer:
    """
    Advanced market analyzer with enhanced features including:
    - Market regime detection
    - Sentiment analysis
    - Advanced pattern recognition
    - Machine learning-based predictions
    - Risk management
    - Multi-timeframe analysis
    """

    def __init__(
        self, 
        product_id: str = 'BTC-USDC',
        primary_interval: str = 'ONE_HOUR',
        secondary_intervals: List[str] = None
    ):
        self.product_id = product_id
        self.primary_interval = primary_interval
        self.secondary_intervals = secondary_intervals or ['FIFTEEN_MINUTE', 'SIX_HOUR', 'ONE_DAY']
        
        # Initialize services
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.client = RESTClient(API_KEY, API_SECRET)
        self.historical_data = HistoricalData(self.client)
        self.news_client = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Initialize technical analysis for multiple timeframes
        self.ta_configs = {}
        self.technical_analyzers = {}
        self._initialize_technical_analyzers()
        
        # Initialize state variables
        self.market_regimes = []
        self.sentiment_history = []
        self.volatility_history = []
        self.pattern_history = []
        self.correlation_matrix = None
        self.regime_changes = []
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize statistical measures
        self.scaler = StandardScaler()
        self.last_update = datetime.now(UTC)
        
        # Risk management parameters
        self.max_drawdown = 0.02  # 2% maximum drawdown
        self.position_sizing = 0.01  # 1% position sizing
        self.risk_free_rate = 0.03  # 3% risk-free rate
        
        # Market state tracking
        self.current_market_state = None
        self.state_transition_probabilities = self._initialize_state_transitions()

    def _initialize_technical_analyzers(self):
        """Initialize technical analysis for each timeframe."""
        for interval in [self.primary_interval] + self.secondary_intervals:
            self.ta_configs[interval] = TechnicalAnalysisConfig(
                rsi_overbought=70,
                rsi_oversold=30,
                volatility_threshold=0.02,
                risk_per_trade=0.02,
                atr_multiplier=2.5
            )
            
            self.technical_analyzers[interval] = TechnicalAnalysis(
                self.coinbase_service,
                config=self.ta_configs[interval],
                candle_interval=interval,
                product_id=self.product_id
            )

    def _setup_logging(self):
        """Configure advanced logging."""
        handler = logging.FileHandler(f'advanced_market_analyzer_{self.product_id}.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def detect_market_regime(self, candles: List[Dict]) -> MarketRegime:
        """
        Detect current market regime using multiple indicators and statistical measures.
        """
        try:
            prices = np.array([float(c['close']) for c in candles])
            returns = np.diff(np.log(prices))
            
            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Calculate trend strength using Hurst exponent
            hurst_exp = self._calculate_hurst_exponent(prices)
            
            # Perform stationarity test
            adf_result = adfuller(prices)
            mean_reversion = adf_result[0] < adf_result[4]['5%']
            
            # Determine regime type
            if hurst_exp > 0.6:  # Trending
                regime_type = "Trending"
                confidence = min((hurst_exp - 0.6) * 2.5, 1.0)
            elif mean_reversion:  # Mean-reverting
                regime_type = "Mean-Reverting"
                confidence = 1 - adf_result[1]  # Use p-value as confidence
            else:  # Random walk
                regime_type = "Random"
                confidence = 0.5
                
            return MarketRegime(
                type=regime_type,
                confidence=confidence,
                volatility=volatility,
                trend_strength=hurst_exp,
                mean_reversion=float(mean_reversion)
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime("Unknown", 0.0, 0.0, 0.0, 0.0)

    def _calculate_hurst_exponent(self, prices: np.ndarray, lags: range = range(2, 100)) -> float:
        """Calculate Hurst exponent to determine long-term memory of time series."""
        try:
            tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0] / 2.0
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {str(e)}")
            return 0.5

    def analyze_sentiment(self) -> Dict[str, Union[float, str]]:
        """
        Analyze market sentiment using news and social media data.
        """
        try:
            # Get news articles
            news = self.news_client.get_everything(
                q=self.product_id.split('-')[0],  # Search for the base currency
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            # Perform basic sentiment analysis
            sentiment_scores = []
            for article in news['articles']:
                # Simple sentiment scoring based on title keywords
                score = self._calculate_sentiment_score(article['title'])
                sentiment_scores.append(score)
            
            avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
            sentiment_std = np.std(sentiment_scores) if sentiment_scores else 0
            
            # Determine sentiment category
            if avg_sentiment > 0.5:
                category = "Bullish"
            elif avg_sentiment < -0.5:
                category = "Bearish"
            else:
                category = "Neutral"
                
            return {
                'score': avg_sentiment,
                'volatility': sentiment_std,
                'category': category,
                'confidence': 1 - (sentiment_std / 2) if sentiment_std < 2 else 0,
                'sample_size': len(sentiment_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'score': 0,
                'volatility': 0,
                'category': "Unknown",
                'confidence': 0,
                'sample_size': 0
            }

    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score for a piece of text."""
        # This is a simplified version - could be enhanced with NLP libraries
        positive_words = {'bull', 'bullish', 'up', 'surge', 'gain', 'positive', 'buy'}
        negative_words = {'bear', 'bearish', 'down', 'crash', 'loss', 'negative', 'sell'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0
            
        return (positive_count - negative_count) / total

    def get_advanced_analysis(self) -> Dict:
        """
        Perform comprehensive market analysis across multiple timeframes.
        """
        try:
            # Get data for all timeframes
            analyses = {}
            for interval in [self.primary_interval] + self.secondary_intervals:
                candles = self._get_candles(interval)
                if candles:
                    analyses[interval] = self._analyze_timeframe(candles, interval)
            
            # Detect market regime
            regime = self.detect_market_regime(analyses[self.primary_interval]['candles'])
            
            # Get sentiment analysis
            sentiment = self.analyze_sentiment()
            
            # Combine analyses into comprehensive report
            return self._generate_comprehensive_report(analyses, regime, sentiment)
            
        except Exception as e:
            self.logger.error(f"Error in advanced analysis: {str(e)}")
            return self._generate_error_response()

    def _get_candles(self, interval: str) -> List[Dict]:
        """Get candle data for specified interval with retry logic."""
        try:
            end_time = datetime.now(UTC)
            start_time = end_time - self._get_timeframe_duration(interval)
            
            candles = []
            current_end = end_time
            current_start = start_time
            
            # Fetch data in chunks if needed
            while current_start < end_time:
                chunk = self.historical_data.get_historical_data(
                    self.product_id,
                    current_start,
                    current_end,
                    interval
                )
                
                # Add debug logging
                if chunk:
                    self.logger.debug(f"First candle in chunk: {chunk[0]}")
                    self.logger.debug(f"Last candle in chunk: {chunk[-1]}")
                
                if not chunk:
                    break
                    
                candles.extend(chunk)
                current_end = current_start
                current_start = current_start - self._get_timeframe_duration(interval)
            
            # Log the amount of data received
            self.logger.info(f"Retrieved {len(candles)} candles for {interval} timeframe")
            
            if len(candles) < 100:  # Minimum required candles
                self.logger.warning(f"Insufficient data for {interval} timeframe: only {len(candles)} candles available")
                
            # Add validation for the latest candle
            if candles:
                latest_price = float(candles[-1]['close'])
                if not self._validate_price(latest_price, self.product_id):
                    self.logger.error(f"Latest candle price {latest_price} seems incorrect")
                
            return candles
            
        except Exception as e:
            self.logger.error(f"Error fetching candles for {interval}: {str(e)}")
            return []

    def _get_timeframe_duration(self, interval: str) -> timedelta:
        """Get appropriate duration for each timeframe."""
        durations = {
            'ONE_MINUTE': timedelta(days=7),      # Increased from 24 hours
            'FIVE_MINUTE': timedelta(days=14),    # Increased from 3 days
            'FIFTEEN_MINUTE': timedelta(days=30),  # Increased from 7 days
            'THIRTY_MINUTE': timedelta(days=60),   # Increased from 14 days
            'ONE_HOUR': timedelta(days=90),        # Increased from 30 days
            'SIX_HOUR': timedelta(days=180),       # Increased from 60 days
            'ONE_DAY': timedelta(days=1460)        # Increased from 240 days to 4 years
        }
        return durations.get(interval, timedelta(days=90))

    def _analyze_timeframe(self, candles: List[Dict], interval: str) -> Dict:
        """Analyze a specific timeframe."""
        if not self._check_data_sufficiency(candles):
            return {
                'candles': candles,
                'analysis': None,
                'patterns': None,
                'support_resistance': None,
                'volatility': None,
                'error': 'Insufficient data'
            }
        
        analyzer = self.technical_analyzers[interval]
        analysis = analyzer.analyze_market(candles)
        
        return {
            'candles': candles,
            'analysis': analysis,
            'patterns': self._detect_advanced_patterns(candles),
            'support_resistance': self._calculate_support_resistance(candles),
            'volatility': self._calculate_volatility(candles)
        }

    def _detect_advanced_patterns(self, candles: List[Dict]) -> Dict:
        """Detect advanced chart patterns."""
        try:
            if len(candles) < 30:  # Need at least 30 candles for pattern detection
                return {
                    'type': AdvancedPatternType.NONE,
                    'confidence': 0.0,
                    'details': {'error': 'Insufficient data for pattern detection'}
                }

            # Convert candle data to numpy arrays for easier processing
            highs = np.array([float(c['high']) for c in candles])
            lows = np.array([float(c['low']) for c in candles])
            closes = np.array([float(c['close']) for c in candles])
            volumes = np.array([float(c['volume']) for c in candles])

            patterns = []
            confidences = []

            # Double Top Detection
            if self._is_double_top(highs[-30:], closes[-30:], volumes[-30:]):
                patterns.append(AdvancedPatternType.DOUBLE_TOP)
                confidences.append(self._calculate_pattern_confidence(
                    AdvancedPatternType.DOUBLE_TOP, 
                    highs[-30:], 
                    lows[-30:], 
                    closes[-30:], 
                    volumes[-30:]
                ))

            # Double Bottom Detection
            if self._is_double_bottom(lows[-30:], closes[-30:], volumes[-30:]):
                patterns.append(AdvancedPatternType.DOUBLE_BOTTOM)
                confidences.append(self._calculate_pattern_confidence(
                    AdvancedPatternType.DOUBLE_BOTTOM,
                    highs[-30:],
                    lows[-30:],
                    closes[-30:],
                    volumes[-30:]
                ))

            # Head and Shoulders Detection
            if self._is_head_and_shoulders(highs[-50:], lows[-50:], volumes[-50:]):
                patterns.append(AdvancedPatternType.HEAD_SHOULDERS)
                confidences.append(self._calculate_pattern_confidence(
                    AdvancedPatternType.HEAD_SHOULDERS,
                    highs[-50:],
                    lows[-50:],
                    closes[-50:],
                    volumes[-50:]
                ))

            # Triangle Patterns Detection
            triangle_type = self._detect_triangle_pattern(highs[-40:], lows[-40:], volumes[-40:])
            if triangle_type != AdvancedPatternType.NONE:
                patterns.append(triangle_type)
                confidences.append(self._calculate_pattern_confidence(
                    triangle_type,
                    highs[-40:],
                    lows[-40:],
                    closes[-40:],
                    volumes[-40:]
                ))

            # Flag Patterns Detection
            flag_type = self._detect_flag_pattern(highs[-20:], lows[-20:], closes[-20:], volumes[-20:])
            if flag_type != AdvancedPatternType.NONE:
                patterns.append(flag_type)
                confidences.append(self._calculate_pattern_confidence(
                    flag_type,
                    highs[-20:],
                    lows[-20:],
                    closes[-20:],
                    volumes[-20:]
                ))

            if not patterns:
                return {
                    'type': AdvancedPatternType.NONE,
                    'confidence': 0.0,
                    'details': {'message': 'No significant patterns detected'}
                }

            # Select the pattern with highest confidence
            best_pattern_idx = np.argmax(confidences)
            best_pattern = patterns[best_pattern_idx]
            confidence = confidences[best_pattern_idx]

            # Calculate pattern metrics
            metrics = self._calculate_pattern_metrics(
                best_pattern,
                highs,
                lows,
                closes,
                volumes
            )

            return {
                'type': best_pattern,
                'confidence': confidence,
                'details': metrics
            }

        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return {
                'type': AdvancedPatternType.NONE,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }

    def _is_double_top(self, highs: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect double top pattern."""
        try:
            # Find peaks in price
            peaks = self._find_peaks(highs, distance=5)
            if len(peaks) < 2:
                return False

            # Get last two peaks
            peak1, peak2 = peaks[-2:]
            peak1_price, peak2_price = highs[peak1], highs[peak2]

            # Check if peaks are within 2% of each other
            price_diff = abs(peak1_price - peak2_price) / peak1_price
            if price_diff > 0.02:
                return False

            # Check time distance between peaks (5-30 candles)
            if not (5 <= peak2 - peak1 <= 30):
                return False

            # Check volume pattern (should decrease between peaks)
            vol_between_peaks = volumes[peak1:peak2]
            if not (np.mean(vol_between_peaks) < np.mean(volumes[peak1-3:peak1])):
                return False

            # Check for price decline after second peak
            if closes[-1] > peak2_price * 0.98:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in double top detection: {str(e)}")
            return False

    def _is_double_bottom(self, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect double bottom pattern."""
        try:
            # Find troughs in price
            troughs = self._find_troughs(lows, distance=5)
            if len(troughs) < 2:
                return False

            # Get last two troughs
            trough1, trough2 = troughs[-2:]
            trough1_price, trough2_price = lows[trough1], lows[trough2]

            # Check if troughs are within 2% of each other
            price_diff = abs(trough1_price - trough2_price) / trough1_price
            if price_diff > 0.02:
                return False

            # Check time distance between troughs (5-30 candles)
            if not (5 <= trough2 - trough1 <= 30):
                return False

            # Check volume pattern (should increase on second trough)
            if not (np.mean(volumes[trough2-2:trough2+3]) > np.mean(volumes[trough1-2:trough1+3])):
                return False

            # Check for price increase after second trough
            if closes[-1] < trough2_price * 1.02:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in double bottom detection: {str(e)}")
            return False

    def _calculate_pattern_confidence(self, pattern_type: AdvancedPatternType, 
                                    highs: np.ndarray, lows: np.ndarray, 
                                    closes: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate confidence score for detected pattern."""
        try:
            # Base confidence starts at 0.5
            confidence = 0.5

            # Volume confirmation adds up to 0.2
            vol_trend = np.polyfit(np.arange(len(volumes)), volumes, 1)[0]
            if (pattern_type in [AdvancedPatternType.DOUBLE_BOTTOM, AdvancedPatternType.BULL_FLAG] 
                and vol_trend > 0) or \
               (pattern_type in [AdvancedPatternType.DOUBLE_TOP, AdvancedPatternType.BEAR_FLAG] 
                and vol_trend < 0):
                confidence += 0.2

            # Price movement confirmation adds up to 0.2
            price_trend = np.polyfit(np.arange(len(closes)), closes, 1)[0]
            if (pattern_type in [AdvancedPatternType.DOUBLE_BOTTOM, AdvancedPatternType.BULL_FLAG] 
                and price_trend > 0) or \
               (pattern_type in [AdvancedPatternType.DOUBLE_TOP, AdvancedPatternType.BEAR_FLAG] 
                and price_trend < 0):
                confidence += 0.2

            # Pattern symmetry adds up to 0.1
            symmetry = self._calculate_pattern_symmetry(pattern_type, highs, lows)
            confidence += symmetry * 0.1

            return min(confidence, 1.0)

        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {str(e)}")
            return 0.5

    def _calculate_pattern_metrics(self, pattern_type: AdvancedPatternType,
                                 highs: np.ndarray, lows: np.ndarray,
                                 closes: np.ndarray, volumes: np.ndarray) -> Dict:
        """Calculate detailed metrics for the detected pattern."""
        try:
            current_price = closes[-1]
            atr = self._calculate_atr(highs, lows, closes)
            
            metrics = {
                'pattern_length': len(closes),
                'price_range': float(np.max(highs) - np.min(lows)),
                'volume_trend': 'Increasing' if np.polyfit(np.arange(len(volumes)), volumes, 1)[0] > 0 else 'Decreasing',
                'breakout_level': float(np.max(highs) if pattern_type in [AdvancedPatternType.DOUBLE_BOTTOM, 
                                                                         AdvancedPatternType.BULL_FLAG]
                                      else np.min(lows)),
                'stop_loss': float(current_price - atr * 2),
                'target': float(current_price + atr * 3),
                'risk_reward_ratio': 1.5
            }
            
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating pattern metrics: {str(e)}")
            return {'error': str(e)}

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            tr1 = np.abs(highs[1:] - lows[1:])
            tr2 = np.abs(highs[1:] - closes[:-1])
            tr3 = np.abs(closes[:-1] - lows[1:])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            return float(np.mean(tr[-period:]))
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0

    def _calculate_support_resistance(self, candles: List[Dict]) -> Dict:
        """Calculate support and resistance levels using multiple methods."""
        prices = np.array([float(c['close']) for c in candles])
        
        # Implementation would include various S/R calculation methods
        return {
            'support_levels': [],
            'resistance_levels': [],
            'confidence': 0.0
        }

    def _calculate_volatility(self, candles: List[Dict]) -> Dict:
        """Calculate various volatility metrics."""
        prices = np.array([float(c['close']) for c in candles])
        returns = np.diff(np.log(prices))
        
        return {
            'historical': np.std(returns) * np.sqrt(252),
            'parkinson': self._calculate_parkinson_volatility(candles),
            'garman_klass': self._calculate_garman_klass_volatility(candles)
        }

    def _calculate_parkinson_volatility(self, candles: List[Dict]) -> float:
        """Calculate Parkinson volatility using high-low prices."""
        try:
            highs = np.array([float(c['high']) for c in candles])
            lows = np.array([float(c['low']) for c in candles])
            return np.sqrt(
                1 / (4 * np.log(2)) * 
                np.mean(np.log(highs / lows) ** 2)
            ) * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"Error calculating Parkinson volatility: {str(e)}")
            return 0.0

    def _calculate_garman_klass_volatility(self, candles: List[Dict]) -> float:
        """Calculate Garman-Klass volatility."""
        try:
            opens = np.array([float(c['open']) for c in candles])
            highs = np.array([float(c['high']) for c in candles])
            lows = np.array([float(c['low']) for c in candles])
            closes = np.array([float(c['close']) for c in candles])
            
            return np.sqrt(
                0.5 * np.mean(np.log(highs / lows) ** 2) -
                (2 * np.log(2) - 1) * np.mean(np.log(closes / opens) ** 2)
            ) * np.sqrt(252)
        except Exception as e:
            self.logger.error(f"Error calculating Garman-Klass volatility: {str(e)}")
            return 0.0

    def _validate_price(self, price: float, product_id: str) -> bool:
        """Validate if the price seems reasonable for the given product."""
        # Rough price ranges for common crypto pairs
        price_ranges = {
            'BTC': (10000, 100000),  # Bitcoin typical range
            'ETH': (500, 5000),      # Ethereum typical range
            'SOL': (10, 500),        # Solana typical range
            'DOGE': (0.01, 1),       # Dogecoin typical range
            'XRP': (0.1, 5),         # Ripple typical range
        }
        
        # Get the base currency from product_id
        base_currency = product_id.split('-')[0]
        
        if base_currency in price_ranges:
            min_price, max_price = price_ranges[base_currency]
            if price < min_price or price > max_price:
                self.logger.error(
                    f"Price validation failed for {product_id}: "
                    f"Got ${price:.2f}, expected range ${min_price:.2f}-${max_price:.2f}"
                )
                return False
        return True

    def _get_current_price(self) -> Optional[float]:
        """Get current price with validation."""
        try:
            # Try to get real-time price first
            product = self.client.get_product(self.product_id)
            if product and 'price' in product:
                price = float(product['price'])
                if self._validate_price(price, self.product_id):
                    return price
            
            # Fallback to latest candle if real-time price fails
            candles = self._get_candles('ONE_MINUTE')
            if candles:
                price = float(candles[-1]['close'])
                if self._validate_price(price, self.product_id):
                    return price
                
            self.logger.error(f"Failed to get valid price for {self.product_id}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return None

    def _generate_comprehensive_report(self, analyses: Dict, regime: MarketRegime, sentiment: Dict) -> Dict:
        """Generate comprehensive analysis report."""
        try:
            # Get current price with validation
            current_price = self._get_current_price()
            
            if current_price is None:
                return self._generate_error_response("Failed to get valid current price")
            
            return {
                'timestamp': datetime.now(UTC).isoformat(),
                'product_id': self.product_id,
                'current_price': current_price,
                'market_regime': {
                    'type': regime.type,
                    'confidence': regime.confidence,
                    'volatility': regime.volatility,
                    'trend_strength': regime.trend_strength,
                    'mean_reversion': regime.mean_reversion
                },
                'sentiment': sentiment,
                'timeframe_analysis': {
                    interval: {
                        'signal': analysis['analysis']['signal'].signal_type.value,
                        'confidence': analysis['analysis']['signal'].confidence,
                        'patterns': analysis['patterns'],
                        'volatility': analysis['volatility']
                    }
                    for interval, analysis in analyses.items()
                },
                'risk_metrics': self._calculate_risk_metrics(analyses),
                'trade_recommendations': self._generate_trade_recommendations(analyses, regime, sentiment)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {str(e)}")
            return self._generate_error_response()

    def _calculate_risk_metrics(self, analyses: Dict) -> Dict:
        """Calculate comprehensive risk metrics."""
        try:
            primary_candles = analyses[self.primary_interval]['candles']
            prices = np.array([float(c['close']) for c in primary_candles])
            returns = np.diff(np.log(prices))
            
            return {
                'volatility': np.std(returns) * np.sqrt(252),
                'var_95': np.percentile(returns, 5),
                'max_drawdown': self._calculate_max_drawdown(prices),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'sortino_ratio': self._calculate_sortino_ratio(returns)
            }
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = prices[0]
        max_drawdown = 0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - (self.risk_free_rate / 252)
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - (self.risk_free_rate / 252)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)

    def _generate_trade_recommendations(self, analyses: Dict, regime: MarketRegime, sentiment: Dict) -> Dict:
        """Generate detailed trade recommendations."""
        # Implementation would include sophisticated trade recommendation logic
        return {
            'position': 'NEUTRAL',
            'confidence': 0.0,
            'entry_points': [],
            'stop_loss': 0.0,
            'take_profit': 0.0
        }

    def _generate_error_response(self, error_message: str = 'Analysis error') -> Dict:
        """Generate standardized error response."""
        return {
            'error': error_message,
            'timestamp': datetime.now(UTC).isoformat(),
            'product_id': self.product_id
        }

    def _initialize_state_transitions(self) -> Dict[MarketState, Dict[MarketState, float]]:
        """Initialize state transition probability matrix."""
        states = list(MarketState)
        return {
            state: {other_state: 1/len(states) for other_state in states}
            for state in states
        }

    def _check_data_sufficiency(self, candles: List[Dict], required_points: int = 1460) -> bool:
        """Check if we have enough data points for analysis."""
        if len(candles) < required_points:
            self.logger.warning(
                f"Insufficient data points. Have {len(candles)}, need {required_points}. "
                f"Consider using a different trading pair or timeframe."
            )
            return False
        return True

    def _find_peaks(self, data: np.ndarray, distance: int = 5) -> List[int]:
        """Find peaks in price data."""
        peaks = []
        for i in range(distance, len(data) - distance):
            if all(data[i] > data[i-j] for j in range(1, distance+1)) and \
               all(data[i] > data[i+j] for j in range(1, distance+1)):
                peaks.append(i)
        return peaks

    def _find_troughs(self, data: np.ndarray, distance: int = 5) -> List[int]:
        """Find troughs in price data."""
        troughs = []
        for i in range(distance, len(data) - distance):
            if all(data[i] < data[i-j] for j in range(1, distance+1)) and \
               all(data[i] < data[i+j] for j in range(1, distance+1)):
                troughs.append(i)
        return troughs

    def _is_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect head and shoulders pattern."""
        try:
            # Need at least 50 candles for H&S pattern
            if len(highs) < 50:
                return False

            # Find peaks
            peaks = self._find_peaks(highs, distance=5)
            if len(peaks) < 3:
                return False

            # Get last three peaks
            last_peaks = peaks[-3:]
            if len(last_peaks) < 3:
                return False

            left_shoulder = highs[last_peaks[0]]
            head = highs[last_peaks[1]]
            right_shoulder = highs[last_peaks[2]]

            # Check if head is higher than shoulders
            if not (head > left_shoulder and head > right_shoulder):
                return False

            # Check if shoulders are at similar levels (within 3%)
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff > 0.03:
                return False

            # Check spacing between peaks
            spacing1 = last_peaks[1] - last_peaks[0]
            spacing2 = last_peaks[2] - last_peaks[1]
            if abs(spacing1 - spacing2) / spacing1 > 0.5:  # Allow 50% difference in spacing
                return False

            # Check volume pattern (typically decreases)
            vol_trend = np.polyfit(np.arange(len(volumes)), volumes, 1)[0]
            if vol_trend > 0:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in head and shoulders detection: {str(e)}")
            return False

    def _detect_triangle_pattern(self, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> AdvancedPatternType:
        """Detect triangle patterns."""
        try:
            if len(highs) < 20:
                return AdvancedPatternType.NONE

            # Calculate trend lines
            high_trend = np.polyfit(np.arange(len(highs)), highs, 1)[0]
            low_trend = np.polyfit(np.arange(len(lows)), lows, 1)[0]

            # Calculate trend line angles
            high_angle = np.arctan(high_trend) * 180 / np.pi
            low_angle = np.arctan(low_trend) * 180 / np.pi

            # Check for triangle patterns
            if abs(high_angle) < 5 and low_angle > 5:
                return AdvancedPatternType.ASCENDING_TRIANGLE
            elif high_angle < -5 and abs(low_angle) < 5:
                return AdvancedPatternType.DESCENDING_TRIANGLE
            elif high_angle < -5 and low_angle > 5:
                return AdvancedPatternType.SYMMETRICAL_TRIANGLE

            return AdvancedPatternType.NONE

        except Exception as e:
            self.logger.error(f"Error in triangle pattern detection: {str(e)}")
            return AdvancedPatternType.NONE

    def _detect_flag_pattern(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> AdvancedPatternType:
        """Detect flag patterns."""
        try:
            if len(highs) < 20:
                return AdvancedPatternType.NONE

            # Calculate price trend
            price_trend = np.polyfit(np.arange(len(closes)), closes, 1)[0]
            
            # Calculate channel
            high_trend = np.polyfit(np.arange(len(highs)), highs, 1)[0]
            low_trend = np.polyfit(np.arange(len(lows)), lows, 1)[0]
            
            # Check for parallel channel
            if abs(high_trend - low_trend) < 0.0001:
                # Bullish flag
                if price_trend > 0 and np.mean(volumes[:10]) > np.mean(volumes[10:]):
                    return AdvancedPatternType.BULL_FLAG
                # Bearish flag
                elif price_trend < 0 and np.mean(volumes[:10]) > np.mean(volumes[10:]):
                    return AdvancedPatternType.BEAR_FLAG

            return AdvancedPatternType.NONE

        except Exception as e:
            self.logger.error(f"Error in flag pattern detection: {str(e)}")
            return AdvancedPatternType.NONE

    def _calculate_pattern_symmetry(self, pattern_type: AdvancedPatternType, highs: np.ndarray, lows: np.ndarray) -> float:
        """Calculate pattern symmetry score."""
        try:
            if pattern_type in [AdvancedPatternType.DOUBLE_TOP, AdvancedPatternType.DOUBLE_BOTTOM]:
                peaks = self._find_peaks(highs) if pattern_type == AdvancedPatternType.DOUBLE_TOP else self._find_troughs(lows)
                if len(peaks) >= 2:
                    # Calculate time and price symmetry
                    time_diff = peaks[-1] - peaks[-2]
                    price_diff = abs(highs[peaks[-1]] - highs[peaks[-2]]) / highs[peaks[-2]]
                    return 1.0 - min(price_diff * 10, 1.0)  # Convert price difference to symmetry score
                
            elif pattern_type == AdvancedPatternType.HEAD_SHOULDERS:
                peaks = self._find_peaks(highs)
                if len(peaks) >= 3:
                    # Calculate shoulder symmetry
                    left_shoulder = highs[peaks[-3]]
                    right_shoulder = highs[peaks[-1]]
                    symmetry = 1.0 - abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder)
                    return symmetry

            return 0.5  # Default symmetry score

        except Exception as e:
            self.logger.error(f"Error calculating pattern symmetry: {str(e)}")
            return 0.5

    def _format_analysis_output(self, analysis: Dict) -> str:
        """Format analysis results into a readable string."""
        if 'error' in analysis:
            return f"Error: {analysis['error']}"

        output = []
        output.append("=" * 80)
        output.append(f"MARKET ANALYSIS REPORT - {analysis['product_id']}")
        output.append(f"Generated at: {analysis['timestamp']}")
        output.append("=" * 80)

        # Current Price
        output.append(f"\nCurrent Price: ${analysis['current_price']:,.2f}")

        # Market Regime
        output.append("\n📊 MARKET REGIME")
        output.append("-" * 40)
        regime = analysis['market_regime']
        output.append(f"Type: {regime['type']}")
        output.append(f"Confidence: {regime['confidence']:.2%}")
        output.append(f"Volatility: {regime['volatility']:.2%}")
        output.append(f"Trend Strength: {regime['trend_strength']:.2f}")

        # Sentiment Analysis
        output.append("\n🎭 MARKET SENTIMENT")
        output.append("-" * 40)
        sentiment = analysis['sentiment']
        output.append(f"Category: {sentiment['category']}")
        output.append(f"Score: {sentiment['score']:.2f}")
        output.append(f"Confidence: {sentiment['confidence']:.2%}")

        # Timeframe Analysis
        output.append("\n⏱️ TIMEFRAME ANALYSIS")
        output.append("-" * 40)
        for interval, data in analysis['timeframe_analysis'].items():
            output.append(f"\n{interval}:")
            output.append(f"  Signal: {data['signal']}")
            output.append(f"  Confidence: {data['confidence']:.2%}")
            if data['patterns'] and data['patterns']['type'] != 'None':
                output.append(f"  Pattern: {data['patterns']['type']}")
                output.append(f"  Pattern Confidence: {data['patterns']['confidence']:.2%}")
            output.append(f"  Volatility: {data['volatility']['historical']:.2%}")

        # Risk Metrics
        if 'risk_metrics' in analysis and analysis['risk_metrics']:
            output.append("\n⚠️ RISK METRICS")
            output.append("-" * 40)
            risk = analysis['risk_metrics']
            output.append(f"Volatility: {risk.get('volatility', 0):.2%}")
            output.append(f"Value at Risk (95%): {risk.get('var_95', 0):.2%}")
            output.append(f"Max Drawdown: {risk.get('max_drawdown', 0):.2%}")
            output.append(f"Sharpe Ratio: {risk.get('sharpe_ratio', 0):.2f}")
            output.append(f"Sortino Ratio: {risk.get('sortino_ratio', 0):.2f}")

        # Trade Recommendations
        if 'trade_recommendations' in analysis:
            output.append("\n💡 TRADE RECOMMENDATIONS")
            output.append("-" * 40)
            trade = analysis['trade_recommendations']
            output.append(f"Position: {trade['position']}")
            output.append(f"Confidence: {trade['confidence']:.2%}")
            if trade['entry_points']:
                output.append(f"Entry Points: {', '.join(map(str, trade['entry_points']))}")
            if trade['stop_loss']:
                output.append(f"Stop Loss: ${trade['stop_loss']:,.2f}")
            if trade['take_profit']:
                output.append(f"Take Profit: ${trade['take_profit']:,.2f}")

        output.append("\n" + "=" * 80)
        return "\n".join(output)

def main():
    parser = argparse.ArgumentParser(description='Advanced Crypto Market Analyzer')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Product ID to analyze (e.g., BTC-USDC)')
    parser.add_argument('--interval', type=str, default='ONE_HOUR',
                      help='Primary interval for analysis (e.g., ONE_HOUR, FIFTEEN_MINUTE)')
    parser.add_argument('--json', action='store_true',
                      help='Output in JSON format instead of formatted text')
    args = parser.parse_args()

    analyzer = AdvancedMarketAnalyzer(
        product_id=args.product_id,
        primary_interval=args.interval
    )
    
    analysis = analyzer.get_advanced_analysis()
    
    if args.json:
        print(json.dumps(analysis, indent=2, cls=EnhancedJSONEncoder))
    else:
        print(analyzer._format_analysis_output(analysis))

if __name__ == "__main__":
    main() 