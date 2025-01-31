import logging
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime, timedelta, UTC
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis, SignalType, TechnicalAnalysisConfig
from config import API_KEY, API_SECRET
from historicaldata import HistoricalData
from coinbase.rest import RESTClient
import argparse
import time
import numpy as np
from enum import Enum
from functools import wraps
import logging.handlers
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Add valid choices for granularity and products
VALID_GRANULARITIES = [
    "ONE_MINUTE",
    "FIVE_MINUTE",
    "FIFTEEN_MINUTE",
    "THIRTY_MINUTE",
    "ONE_HOUR",
    "TWO_HOUR",
    "SIX_HOUR",
    "ONE_DAY"
]

VALID_PRODUCTS = [
    "BTC-USDC",
    "ETH-USDC",
    "SOL-USDC",
    "DOGE-USDC",
    "XRP-USDC",
    "ADA-USDC",
    "MATIC-USDC",
    "LINK-USDC",
    "DOT-USDC",
    "UNI-USDC"
]

class PatternType(Enum):
    DOUBLE_TOP = "Double Top"
    DOUBLE_BOTTOM = "Double Bottom"
    HEAD_SHOULDERS = "Head and Shoulders"
    INV_HEAD_SHOULDERS = "Inverse Head and Shoulders"
    TRIANGLE = "Triangle"
    WEDGE = "Wedge"
    FLAG = "Flag"
    PENNANT = "Pennant"
    CHANNEL = "Channel"
    CUP_HANDLE = "Cup and Handle"
    ROUNDING_BOTTOM = "Rounding Bottom"
    ROUNDING_TOP = "Rounding Top"
    NONE = "None"

class MarketRegime(Enum):
    TRENDING = "Trending"
    RANGING = "Ranging"
    VOLATILE = "Volatile"
    BREAKOUT = "Breakout"
    REVERSAL = "Reversal"

class MarketSentiment(Enum):
    VERY_BULLISH = "Very Bullish"
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"
    VERY_BEARISH = "Very Bearish"

class MLModel:
    """Machine Learning model for market prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None  # Initialize as None, will create when needed
        if model_path:
            self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
    def prepare_features(self, candles: List[Dict]) -> np.ndarray:
        """Prepare features for ML model."""
        try:
            df = pd.DataFrame(candles)
            
            # Ensure we have enough data
            if len(df) < 50:  # Need at least 50 candles for features
                return np.array([]).reshape(0, 6)  # Return empty array with correct shape
            
            # Convert price and volume columns to numeric
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            
            # Get numpy arrays
            prices = df['close'].values
            volumes = df['volume'].values
            
            # Initialize features list
            features = []
            
            # Price momentum (last 20 periods)
            price_momentum = np.gradient(prices[-20:])[-1] if len(prices) >= 20 else 0
            features.append(price_momentum)
            
            # Volatility (20-period standard deviation)
            volatility = np.std(prices[-20:]) if len(prices) >= 20 else 0
            features.append(volatility)
            
            # Volume momentum (last 20 periods)
            volume_momentum = np.gradient(volumes[-20:])[-1] if len(volumes) >= 20 else 0
            features.append(volume_momentum)
            
            # Relative volume (current volume vs 20-period average)
            avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            rel_volume = volumes[-1] / avg_volume if avg_volume != 0 else 1.0
            features.append(rel_volume)
            
            # Moving averages
            sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
            
            # Price vs SMA20
            price_sma20 = (prices[-1] - sma_20) / sma_20 if sma_20 != 0 else 0
            features.append(price_sma20)
            
            # SMA20 vs SMA50
            sma_ratio = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0
            features.append(sma_ratio)
            
            # Convert features to numpy array and reshape
            features_array = np.array(features, dtype=np.float64)
            return features_array.reshape(1, -1)
            
        except Exception as e:
            logging.error(f"Error preparing ML features: {str(e)}")
            return np.array([]).reshape(0, 6)  # Return empty array with correct shape
            
    def train(self, candles: List[Dict], labels: List[int]):
        """Train the ML model."""
        X = self.prepare_features(candles)
        if X.size == 0:  # Check if features array is empty
            return
            
        # Initialize scaler if not already done
        if self.scaler is None:
            self.scaler = StandardScaler()
            
        # Fit and transform the data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        
    def predict(self, candles: List[Dict]) -> float:
        """Predict market direction."""
        if self.model is None or self.scaler is None:
            return 0.0
            
        X = self.prepare_features(candles)
        if X.size == 0:  # Check if features array is empty
            return 0.0
            
        try:
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Convert probabilities to signal strength (-1 to 1)
            return (probabilities[1] - probabilities[0]) * 2 - 1
        except Exception as e:
            logging.error(f"Error in ML prediction: {str(e)}")
            return 0.0
        
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is not None and self.scaler is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, path)
            
    def load_model(self, path: str):
        """Load model from file."""
        try:
            saved = joblib.load(path)
            self.model = saved['model']
            self.scaler = saved['scaler']
            
            # Verify both model and scaler are loaded
            if self.model is None or self.scaler is None:
                raise ValueError("Failed to load model or scaler")
                
        except Exception as e:
            logging.error(f"Error loading ML model: {str(e)}")
            # Reset to initial state
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = None

class SentimentAnalyzer:
    """Analyzes market sentiment from various sources."""
    
    def __init__(self):
        self.sentiment_history = []
        self.max_history = 100
        
    def analyze_news_sentiment(self, product_id: str) -> Dict:
        """Analyze sentiment from crypto news sources."""
        try:
            # Example news API call (replace with actual API)
            news_url = f"https://cryptonews-api.com/api/v1/news?tickers={product_id}"
            # response = requests.get(news_url)
            # news = response.json()
            
            # Simulate news sentiment for now
            sentiment_scores = []
            for _ in range(5):  # Simulate 5 news articles
                score = np.random.normal(0.1, 0.3)  # Slightly bullish bias
                sentiment_scores.append(score)
            
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            return {
                'score': avg_sentiment,
                'volatility': sentiment_std,
                'confidence': 1.0 - sentiment_std,
                'source': 'news'
            }
        except Exception as e:
            logging.error(f"Error analyzing news sentiment: {str(e)}")
            return {'score': 0, 'volatility': 0, 'confidence': 0, 'source': 'news'}
            
    def analyze_social_sentiment(self, product_id: str) -> Dict:
        """Analyze sentiment from social media."""
        try:
            # Simulate social sentiment
            sentiment_scores = []
            for _ in range(10):  # Simulate 10 social posts
                score = np.random.normal(0.2, 0.4)  # More volatile than news
                sentiment_scores.append(score)
            
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_std = np.std(sentiment_scores)
            
            return {
                'score': avg_sentiment,
                'volatility': sentiment_std,
                'confidence': 1.0 - sentiment_std,
                'source': 'social'
            }
        except Exception as e:
            logging.error(f"Error analyzing social sentiment: {str(e)}")
            return {'score': 0, 'volatility': 0, 'confidence': 0, 'source': 'social'}
            
    def get_aggregated_sentiment(self, product_id: str) -> MarketSentiment:
        """Get aggregated sentiment from all sources."""
        news_sentiment = self.analyze_news_sentiment(product_id)
        social_sentiment = self.analyze_social_sentiment(product_id)
        
        # Weight the sentiments by their confidence
        total_score = (
            news_sentiment['score'] * news_sentiment['confidence'] +
            social_sentiment['score'] * social_sentiment['confidence']
        ) / (news_sentiment['confidence'] + social_sentiment['confidence'])
        
        # Store sentiment history
        self.sentiment_history.append({
            'timestamp': datetime.now(UTC),
            'score': total_score,
            'news_score': news_sentiment['score'],
            'social_score': social_sentiment['score']
        })
        
        # Keep history size in check
        if len(self.sentiment_history) > self.max_history:
            self.sentiment_history.pop(0)
        
        # Convert score to sentiment enum
        if total_score > 0.5:
            return MarketSentiment.VERY_BULLISH
        elif total_score > 0.1:
            return MarketSentiment.BULLISH
        elif total_score < -0.5:
            return MarketSentiment.VERY_BEARISH
        elif total_score < -0.1:
            return MarketSentiment.BEARISH
        else:
            return MarketSentiment.NEUTRAL

class AdaptiveWeights:
    """Dynamically adjusts indicator weights based on performance."""
    
    def __init__(self, initial_weights: Dict[str, float]):
        self.weights = initial_weights.copy()
        self.performance_history = []
        self.learning_rate = 0.1
        self.max_history = 100
        
    def update_weights(self, indicator_performances: Dict[str, float]):
        """Update weights based on indicator performance."""
        total_performance = sum(abs(p) for p in indicator_performances.values())
        if total_performance == 0:
            return
            
        # Adjust weights based on relative performance
        for indicator, performance in indicator_performances.items():
            if indicator in self.weights:
                relative_performance = performance / total_performance
                self.weights[indicator] += self.learning_rate * relative_performance
                
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
    def get_weight(self, indicator: str) -> float:
        """Get current weight for an indicator."""
        return self.weights.get(indicator, 0.0)
        
    def record_performance(self, indicator_signals: Dict[str, float], actual_movement: float):
        """Record indicator performance for future weight updates."""
        performance = {}
        for indicator, signal in indicator_signals.items():
            # Calculate how well the indicator predicted the movement
            performance[indicator] = 1.0 - abs(signal - actual_movement)
            
        self.performance_history.append({
            'timestamp': datetime.now(UTC),
            'performance': performance
        })
        
        if len(self.performance_history) > self.max_history:
            self.performance_history.pop(0)
            
        self.update_weights(performance)

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        execution_time = time.time() - start_time
        
        # Log performance metrics
        self.logger.info(f"{func.__name__} execution time: {execution_time:.2f} seconds")
        
        # Store performance metrics
        if not hasattr(self, '_performance_metrics'):
            self._performance_metrics = {}
        
        if func.__name__ not in self._performance_metrics:
            self._performance_metrics[func.__name__] = []
        
        self._performance_metrics[func.__name__].append({
            'timestamp': datetime.now(UTC),
            'execution_time': execution_time
        })
        
        # Keep only last 100 metrics
        if len(self._performance_metrics[func.__name__]) > 100:
            self._performance_metrics[func.__name__].pop(0)
        
        return result
    return wrapper

class MarketAnalyzer:
    """
    A class that analyzes market conditions and generates trading signals
    using technical analysis indicators.
    """

    def __init__(self, product_id: str = 'DOGE-USDC', candle_interval: str = 'ONE_HOUR'):
        self.product_id = product_id
        self.candle_interval = candle_interval
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.client = RESTClient(API_KEY, API_SECRET)
        self.historical_data = HistoricalData(self.client)
        
        # Setup enhanced logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            'market_analyzer.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
        # Initialize performance metrics
        self._performance_metrics = {}
        self._start_time = datetime.now(UTC)
        
        # Initialize custom technical analysis configuration
        self.ta_config = TechnicalAnalysisConfig(
            rsi_overbought=70,
            rsi_oversold=30,
            volatility_threshold=0.02,
            risk_per_trade=0.02,
            atr_multiplier=2.5
        )
        
        self.technical_analysis = TechnicalAnalysis(
            self.coinbase_service,
            config=self.ta_config,
            candle_interval=candle_interval,
            product_id=product_id
        )
        
        # Initialize ML model
        self.ml_model = MLModel()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize adaptive weights with default values
        initial_weights = {
            'rsi': 0.15,
            'macd': 0.15,
            'bollinger': 0.1,
            'adx': 0.1,
            'ma_crossover': 0.1,
            'volume': 0.1,
            'sentiment': 0.15,
            'ml_prediction': 0.15
        }
        self.adaptive_weights = AdaptiveWeights(initial_weights)
        
        self._current_candles = []
        
        # Add new attributes
        self.pattern_memory = []
        self.sentiment_scores = []
        self.volatility_history = []
        self.max_pattern_memory = 10
        
        # Initialize risk parameters
        self.base_risk = 0.02  # 2% base risk
        self.max_risk = 0.04   # 4% maximum risk
        self.min_risk = 0.01   # 1% minimum risk

        # Add indicator cache
        self._indicator_cache = {}
        self._cache_timestamp = None
        self._cache_expiry = timedelta(minutes=5)  # Cache expires after 5 minutes
        
        # Add new attributes for enhanced analysis
        self.support_resistance_levels = []
        self.pivot_points = []
        self.fibonacci_levels = []
        self.market_cycles = []
        self.order_flow_data = []
        self.liquidity_zones = []
        
        # Add market cycle tracking
        self.cycle_phases = {
            'accumulation': {'start': None, 'end': None},
            'markup': {'start': None, 'end': None},
            'distribution': {'start': None, 'end': None},
            'markdown': {'start': None, 'end': None}
        }
        
        # Add market efficiency ratio tracking
        self.market_efficiency_window = 20
        self.efficiency_ratios = []
        
        # Add order flow tracking
        self.order_flow_window = 50
        self.order_flow_history = []
        
        # Add market microstructure
        self.tick_data = []
        self.spread_history = []
        self.depth_imbalance = []
        
        # Add volatility tracking
        self.volatility_windows = [20, 50, 100]  # Multiple timeframes
        self.volatility_metrics = {
            'historical': [],
            'implied': [],
            'realized': []
        }
        
        # Add market correlation tracking
        self.correlation_assets = ['BTC-USDC', 'ETH-USDC']  # Base pairs to track
        self.correlation_history = {}
        
        # Add market breadth indicators
        self.market_breadth = {
            'advancing_declining_ratio': [],
            'new_highs_lows_ratio': [],
            'volume_breadth': []
        }
        
        # Add adaptive thresholds
        self.adaptive_thresholds = {
            'volatility': self._initialize_adaptive_threshold(0.02),  # 2% base
            'volume': self._initialize_adaptive_threshold(1.0),       # 1x base
            'momentum': self._initialize_adaptive_threshold(0.5)      # 0.5 base
        }
        
        self.logger.info(f"MarketAnalyzer initialized for {product_id} with {candle_interval} interval")
        self.logger.info("Enhanced market analyzer initialized with additional features")
        
        # Initialize alert system
        self.alert_system = AlertSystem(self.logger, market_analyzer=self)
        
        # Add default alerts
        self.alert_system.add_volatility_alert(0.05, AlertPriority.HIGH)  # 5% volatility alert
        self.alert_system.add_technical_alert('rsi', 'above', 70, AlertPriority.MEDIUM)  # RSI overbought
        self.alert_system.add_technical_alert('rsi', 'below', 30, AlertPriority.MEDIUM)  # RSI oversold

    def _initialize_adaptive_threshold(self, base_value: float) -> Dict:
        """Initialize an adaptive threshold with base value and adjustment parameters."""
        return {
            'base': base_value,
            'current': base_value,
            'history': [],
            'adjustment_factor': 0.1,
            'max_adjustment': 2.0,
            'min_adjustment': 0.5
        }

    def update_adaptive_thresholds(self):
        """Update adaptive thresholds based on market conditions."""
        try:
            if not self._current_candles:
                return
                
            # Calculate recent volatility
            returns = np.diff([c['close'] for c in self._current_candles[-20:]])
            current_volatility = np.std(returns)
            
            # Update volatility threshold
            vol_threshold = self.adaptive_thresholds['volatility']
            vol_adjustment = self._calculate_threshold_adjustment(
                current_volatility,
                vol_threshold['history'][-20:] if vol_threshold['history'] else [vol_threshold['base']]
            )
            vol_threshold['current'] = vol_threshold['base'] * vol_adjustment
            vol_threshold['history'].append(vol_threshold['current'])
            
            # Update volume threshold similarly
            volumes = [c['volume'] for c in self._current_candles[-20:]]
            avg_volume = np.mean(volumes)
            vol_threshold = self.adaptive_thresholds['volume']
            volume_adjustment = self._calculate_threshold_adjustment(
                avg_volume,
                vol_threshold['history'][-20:] if vol_threshold['history'] else [vol_threshold['base']]
            )
            vol_threshold['current'] = vol_threshold['base'] * volume_adjustment
            vol_threshold['history'].append(vol_threshold['current'])
            
            # Update momentum threshold
            momentum = self.calculate_momentum_score(self._current_candles)
            mom_threshold = self.adaptive_thresholds['momentum']
            mom_adjustment = self._calculate_threshold_adjustment(
                abs(momentum['total_score']),
                mom_threshold['history'][-20:] if mom_threshold['history'] else [mom_threshold['base']]
            )
            mom_threshold['current'] = mom_threshold['base'] * mom_adjustment
            mom_threshold['history'].append(mom_threshold['current'])
            
            self.logger.debug("Adaptive thresholds updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating adaptive thresholds: {str(e)}")

    def _calculate_threshold_adjustment(self, current_value: float, history: List[float]) -> float:
        """Calculate threshold adjustment factor based on recent market behavior."""
        if not history:
            return 1.0
            
        avg_historical = np.mean(history)
        if avg_historical == 0:
            return 1.0
            
        # Calculate relative change
        relative_change = current_value / avg_historical
        
        # Apply sigmoid function to smooth adjustment
        adjustment = 2 / (1 + np.exp(-relative_change)) - 1
        
        # Constrain adjustment
        return max(min(1.0 + adjustment, 2.0), 0.5)

    def analyze_market_cycles(self) -> Dict:
        """Analyze market cycles and identify current phase."""
        try:
            if not self._current_candles or len(self._current_candles) < 50:
                return {'phase': 'Unknown', 'confidence': 0.0}
                
            prices = np.array([c['close'] for c in self._current_candles])
            volumes = np.array([c['volume'] for c in self._current_candles])
            
            # Calculate trend metrics
            sma20 = np.mean(prices[-20:])
            sma50 = np.mean(prices[-50:])
            vol_sma20 = np.mean(volumes[-20:])
            
            # Calculate momentum and volatility
            momentum = self.calculate_momentum_score(self._current_candles)
            volatility = np.std(np.diff(np.log(prices[-20:])))
            
            # Identify cycle phase
            phase = 'Unknown'
            confidence = 0.0
            
            if prices[-1] > sma20 > sma50:  # Uptrend
                if volumes[-1] > vol_sma20 and momentum['total_score'] > 50:
                    phase = 'Markup'
                    confidence = min((momentum['total_score'] / 100) * (volumes[-1] / vol_sma20), 1.0)
                else:
                    phase = 'Accumulation'
                    confidence = 0.7
            else:  # Downtrend
                if volumes[-1] > vol_sma20 and momentum['total_score'] < -50:
                    phase = 'Markdown'
                    confidence = min(abs(momentum['total_score'] / 100) * (volumes[-1] / vol_sma20), 1.0)
                else:
                    phase = 'Distribution'
                    confidence = 0.7
            
            # Update cycle tracking
            current_time = datetime.now(UTC)
            if self.cycle_phases[phase.lower()]['start'] is None:
                self.cycle_phases[phase.lower()]['start'] = current_time
            
            # Record cycle transition
            for p in self.cycle_phases:
                if p != phase.lower() and self.cycle_phases[p]['start'] is not None and self.cycle_phases[p]['end'] is None:
                    self.cycle_phases[p]['end'] = current_time
                    self.market_cycles.append({
                        'phase': p,
                        'start': self.cycle_phases[p]['start'],
                        'end': current_time,
                        'duration': (current_time - self.cycle_phases[p]['start']).total_seconds() / 3600  # hours
                    })
            
            return {
                'phase': phase,
                'confidence': confidence,
                'metrics': {
                    'price_trend': 'Bullish' if prices[-1] > sma20 else 'Bearish',
                    'volume_trend': 'Increasing' if volumes[-1] > vol_sma20 else 'Decreasing',
                    'momentum': momentum['interpretation'],
                    'volatility': volatility
                },
                'cycle_history': self.market_cycles[-5:]  # Last 5 cycle transitions
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market cycles: {str(e)}")
            return {'phase': 'Unknown', 'confidence': 0.0}

    def analyze_order_flow(self) -> Dict:
        """Analyze order flow patterns and market microstructure."""
        try:
            if not self._current_candles or len(self._current_candles) < self.order_flow_window:
                return {'signal': 'Neutral', 'strength': 0.0}
                
            # Calculate order flow metrics
            buys = []
            sells = []
            
            for candle in self._current_candles[-self.order_flow_window:]:
                if candle['close'] > candle['open']:
                    buys.append(candle['volume'] * (candle['close'] - candle['open']))
                else:
                    sells.append(candle['volume'] * (candle['open'] - candle['close']))
            
            buy_pressure = sum(buys)
            sell_pressure = sum(sells)
            
            # Calculate order flow ratio
            total_pressure = buy_pressure + sell_pressure
            if total_pressure == 0:
                flow_ratio = 0
            else:
                flow_ratio = (buy_pressure - sell_pressure) / total_pressure
            
            # Analyze tick data if available
            tick_analysis = {}
            if self.tick_data:
                tick_analysis = {
                    'micro_trend': 'Up' if sum(1 for tick in self.tick_data[-100:] if tick > 0) > 50 else 'Down',
                    'tick_volume': len(self.tick_data),
                    'tick_volatility': np.std(self.tick_data[-100:]) if len(self.tick_data) >= 100 else 0
                }
            
            # Calculate market depth imbalance
            depth_imbalance = sum(self.depth_imbalance[-20:]) / 20 if self.depth_imbalance else 0
            
            # Determine signal
            signal = 'Neutral'
            strength = abs(flow_ratio)
            
            if flow_ratio > 0.2:
                signal = 'Buy' if depth_imbalance > 0 else 'Weak Buy'
            elif flow_ratio < -0.2:
                signal = 'Sell' if depth_imbalance < 0 else 'Weak Sell'
            
            # Update order flow history
            self.order_flow_history.append({
                'timestamp': datetime.now(UTC),
                'flow_ratio': flow_ratio,
                'buy_pressure': buy_pressure,
                'sell_pressure': sell_pressure,
                'depth_imbalance': depth_imbalance
            })
            
            # Keep history size in check
            if len(self.order_flow_history) > self.order_flow_window:
                self.order_flow_history.pop(0)
            
            return {
                'signal': signal,
                'strength': strength,
                'metrics': {
                    'flow_ratio': flow_ratio,
                    'buy_pressure': buy_pressure,
                    'sell_pressure': sell_pressure,
                    'depth_imbalance': depth_imbalance
                },
                'tick_analysis': tick_analysis,
                'history': self.order_flow_history[-5:]  # Last 5 records
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow: {str(e)}")
            return {'signal': 'Neutral', 'strength': 0.0}

    def analyze_liquidity_zones(self) -> Dict:
        """Analyze liquidity zones and identify potential support/resistance levels."""
        try:
            if not self._current_candles or len(self._current_candles) < 50:
                return {'zones': [], 'current_zone': None}
                
            prices = np.array([c['close'] for c in self._current_candles])
            volumes = np.array([c['volume'] for c in self._current_candles])
            
            # Calculate volume profile
            price_levels = np.linspace(min(prices), max(prices), 50)
            volume_profile = []
            
            for level in price_levels:
                # Find candles near this price level
                mask = (prices >= level * 0.995) & (prices <= level * 1.005)
                volume_profile.append({
                    'price': level,
                    'volume': np.sum(volumes[mask]),
                    'trades': np.sum(mask)
                })
            
            # Identify high volume nodes
            avg_volume = np.mean([v['volume'] for v in volume_profile])
            high_volume_nodes = [
                v for v in volume_profile 
                if v['volume'] > avg_volume * 1.5
            ]
            
            # Identify low volume nodes (potential breakout zones)
            low_volume_nodes = [
                v for v in volume_profile 
                if v['volume'] < avg_volume * 0.5
            ]
            
            # Calculate current price zone
            current_price = prices[-1]
            current_zone = None
            
            for node in high_volume_nodes:
                if node['price'] * 0.995 <= current_price <= node['price'] * 1.005:
                    current_zone = {
                        'type': 'High Volume',
                        'price': node['price'],
                        'volume': node['volume'],
                        'strength': node['volume'] / avg_volume
                    }
                    break
            
            if not current_zone:
                for node in low_volume_nodes:
                    if node['price'] * 0.995 <= current_price <= node['price'] * 1.005:
                        current_zone = {
                            'type': 'Low Volume',
                            'price': node['price'],
                            'volume': node['volume'],
                            'strength': avg_volume / node['volume']
                        }
                        break
            
            # Update liquidity zones
            self.liquidity_zones = {
                'high_volume_nodes': high_volume_nodes,
                'low_volume_nodes': low_volume_nodes,
                'volume_profile': volume_profile
            }
            
            return {
                'zones': {
                    'high_volume': [
                        {
                            'price': node['price'],
                            'strength': node['volume'] / avg_volume,
                            'type': 'Support' if node['price'] < current_price else 'Resistance'
                        }
                        for node in high_volume_nodes
                    ],
                    'low_volume': [
                        {
                            'price': node['price'],
                            'strength': avg_volume / node['volume'],
                            'type': 'Breakout Zone'
                        }
                        for node in low_volume_nodes
                    ]
                },
                'current_zone': current_zone,
                'volume_profile_summary': {
                    'total_volume': np.sum([v['volume'] for v in volume_profile]),
                    'price_range': {
                        'min': min(prices),
                        'max': max(prices)
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity zones: {str(e)}")
            return {'zones': [], 'current_zone': None}

    def analyze_market_efficiency(self) -> Dict:
        """Analyze market efficiency and predictability."""
        try:
            if not self._current_candles or len(self._current_candles) < self.market_efficiency_window:
                return {'ratio': 0.0, 'interpretation': 'Unknown'}
                
            prices = np.array([c['close'] for c in self._current_candles[-self.market_efficiency_window:]])
            
            # Calculate directional movement
            directional_movement = abs(prices[-1] - prices[0])
            
            # Calculate total movement
            price_changes = np.abs(np.diff(prices))
            total_movement = np.sum(price_changes)
            
            # Calculate efficiency ratio
            if total_movement == 0:
                efficiency_ratio = 0
            else:
                efficiency_ratio = directional_movement / total_movement
            
            # Store ratio
            self.efficiency_ratios.append(efficiency_ratio)
            if len(self.efficiency_ratios) > 100:  # Keep last 100 values
                self.efficiency_ratios.pop(0)
            
            # Interpret efficiency
            interpretation = 'Trending' if efficiency_ratio > 0.7 else \
                           'Ranging' if efficiency_ratio < 0.3 else \
                           'Moderately Trending'
            
            return {
                'ratio': efficiency_ratio,
                'interpretation': interpretation,
                'metrics': {
                    'directional_movement': directional_movement,
                    'total_movement': total_movement,
                    'average_efficiency': np.mean(self.efficiency_ratios[-20:])
                },
                'trend_quality': 'High' if efficiency_ratio > 0.8 else \
                               'Medium' if efficiency_ratio > 0.5 else 'Low'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market efficiency: {str(e)}")
            return {'ratio': 0.0, 'interpretation': 'Unknown'}

    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the analyzer."""
        stats = {
            'uptime': str(datetime.now(UTC) - self._start_time),
            'metrics': {}
        }
        
        for func_name, metrics in self._performance_metrics.items():
            if metrics:
                execution_times = [m['execution_time'] for m in metrics]
                stats['metrics'][func_name] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times),
                    'total_calls': len(metrics),
                    'last_execution_time': metrics[-1]['execution_time']
                }
        
        return stats

    def _get_cached_indicator(self, indicator_name: str) -> Any:
        """Get indicator from cache if valid."""
        if self._cache_timestamp is None or \
           datetime.now(UTC) - self._cache_timestamp > self._cache_expiry or \
           indicator_name not in self._indicator_cache:
            return None
        return self._indicator_cache[indicator_name]

    def _cache_indicator(self, indicator_name: str, value: Any):
        """Cache an indicator value."""
        self._indicator_cache[indicator_name] = value
        self._cache_timestamp = datetime.now(UTC)

    def _clear_indicator_cache(self):
        """Clear the indicator cache."""
        self._indicator_cache = {}
        self._cache_timestamp = None

    @performance_monitor
    def _get_historical_data_with_retry(self, start_time: datetime, end_time: datetime, product_id: str = None, max_retries: int = 3, retry_delay: float = 1.0) -> List[Dict]:
        """Get historical data with retry logic."""
        for attempt in range(max_retries):
            try:
                # Use provided product_id or fall back to self.product_id
                target_product = product_id if product_id is not None else self.product_id
                candles = self.historical_data.get_historical_data(
                    target_product,
                    start_time,
                    end_time,
                    self.candle_interval
                )
                if candles:
                    return candles
                raise ValueError("No candle data available")
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise Exception(f"Failed to get historical data after {max_retries} attempts: {str(e)}")
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff

    @performance_monitor
    def _safe_indicator_calculation(self, calc_func: callable, *args, **kwargs) -> Any:
        """Safely calculate an indicator with error handling."""
        try:
            return calc_func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error calculating indicator {calc_func.__name__}: {str(e)}")
            return None

    @performance_monitor
    def get_market_signal(self) -> Dict:
        """
        Analyze the market and generate a trading signal based on the last month of data.
        
        Returns:
            Dict containing the signal analysis results
        """
        self.logger.info(f"Starting market analysis for {self.product_id}")
        try:
            # Get candle data for the last month
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=30)
            
            # Get historical data using HistoricalData class with retry
            try:
                candles = self._get_historical_data_with_retry(start_time, end_time)
                self.logger.info(f"Retrieved {len(candles)} candles from {start_time} to {end_time}")
            except Exception as e:
                self.logger.error(f"Failed to get historical data: {str(e)}")
                return self._generate_error_response()

            # Format candles to match expected structure
            try:
                formatted_candles = self._format_candles(candles)
            except Exception as e:
                self.logger.error(f"Error formatting candles: {str(e)}")
                return self._generate_error_response()
            
            # Store the formatted candles
            self._current_candles = formatted_candles

            # Clear cache if candles have changed
            if self._current_candles != formatted_candles:
                self._clear_indicator_cache()

            # Get comprehensive market analysis with error handling
            try:
                analysis = self.technical_analysis.analyze_market(formatted_candles)
            except Exception as e:
                self.logger.error(f"Error in market analysis: {str(e)}")
                return self._generate_error_response()
            
            # Get ML model prediction
            try:
                ml_prediction = self.ml_model.predict(formatted_candles)
                ml_confidence = abs(ml_prediction)  # Higher absolute value means higher confidence
            except Exception as e:
                self.logger.error(f"Error getting ML prediction: {str(e)}")
                ml_prediction = 0.0
                ml_confidence = 0.0
                
            # Get market sentiment
            try:
                sentiment = self.sentiment_analyzer.get_aggregated_sentiment(self.product_id)
                sentiment_data = {
                    'news': self.sentiment_analyzer.analyze_news_sentiment(self.product_id),
                    'social': self.sentiment_analyzer.analyze_social_sentiment(self.product_id)
                }
            except Exception as e:
                self.logger.error(f"Error analyzing sentiment: {str(e)}")
                sentiment = MarketSentiment.NEUTRAL
                sentiment_data = {'news': {}, 'social': {}}

            # Get divergence analysis
            try:
                divergence_analysis = self.detect_divergences(formatted_candles)
            except Exception as e:
                self.logger.error(f"Error analyzing divergences: {str(e)}")
                divergence_analysis = {'divergences': [], 'error': str(e)}

            # Get additional trend information with caching and error handling
            adx_value = self._get_cached_indicator('adx')
            trend_direction = self._get_cached_indicator('trend_direction')
            if adx_value is None or trend_direction is None:
                try:
                    adx_value, trend_direction = self._safe_indicator_calculation(
                        self.technical_analysis.get_trend_strength,
                        formatted_candles
                    )
                    if adx_value is not None and trend_direction is not None:
                        self._cache_indicator('adx', adx_value)
                        self._cache_indicator('trend_direction', trend_direction)
                except Exception as e:
                    self.logger.error(f"Error calculating trend strength: {str(e)}")
                    adx_value = 0
                    trend_direction = "Unknown"
            
            # Calculate key indicators with caching and error handling
            indicators = {}
            
            # RSI calculation
            rsi = self._get_cached_indicator('rsi')
            if rsi is None:
                rsi = self._safe_indicator_calculation(
                    self.technical_analysis.compute_rsi,
                    self.product_id,
                    formatted_candles
                )
                if rsi is not None:
                    self._cache_indicator('rsi', rsi)
            indicators['rsi'] = rsi or 50  # Default to neutral if calculation fails

            # MACD calculation
            macd = self._get_cached_indicator('macd')
            signal = self._get_cached_indicator('macd_signal')
            histogram = self._get_cached_indicator('macd_histogram')
            if any(v is None for v in [macd, signal, histogram]):
                try:
                    macd, signal, histogram = self._safe_indicator_calculation(
                        self.technical_analysis.compute_macd,
                        self.product_id,
                        formatted_candles
                    )
                    if all(v is not None for v in [macd, signal, histogram]):
                        self._cache_indicator('macd', macd)
                        self._cache_indicator('macd_signal', signal)
                        self._cache_indicator('macd_histogram', histogram)
                except Exception as e:
                    self.logger.error(f"Error calculating MACD: {str(e)}")
                    macd, signal, histogram = 0, 0, 0

            # Bollinger Bands calculation
            bb_upper = self._get_cached_indicator('bb_upper')
            bb_middle = self._get_cached_indicator('bb_middle')
            bb_lower = self._get_cached_indicator('bb_lower')
            if any(v is None for v in [bb_upper, bb_middle, bb_lower]):
                try:
                    bb_upper, bb_middle, bb_lower = self._safe_indicator_calculation(
                        self.technical_analysis.compute_bollinger_bands,
                        formatted_candles
                    )
                    if all(v is not None for v in [bb_upper, bb_middle, bb_lower]):
                        self._cache_indicator('bb_upper', bb_upper)
                        self._cache_indicator('bb_middle', bb_middle)
                        self._cache_indicator('bb_lower', bb_lower)
                except Exception as e:
                    self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
                    current_price = float(formatted_candles[-1]['close'])
                    bb_upper = current_price * 1.02
                    bb_middle = current_price
                    bb_lower = current_price * 0.98

            try:
                current_price = float(formatted_candles[-1]['close'])
            except (IndexError, KeyError, ValueError) as e:
                self.logger.error(f"Error getting current price: {str(e)}")
                return self._generate_error_response()

            # Add signal stability information with enhanced detail
            signal_history = self.technical_analysis.signal_history
            total_signals = len(signal_history)
            
            if total_signals > 0:
                # Calculate the percentage of consistent signals
                positive_signals = sum(1 for s in signal_history if s['strength'] > 0)
                negative_signals = sum(1 for s in signal_history if s['strength'] < 0)
                consistency_ratio = max(positive_signals, negative_signals) / total_signals
                
                # Calculate average signal strength
                avg_strength = abs(sum(s['strength'] for s in signal_history) / total_signals)
                
                # Determine stability level with more granularity
                if total_signals >= self.technical_analysis.trend_confirmation_period:
                    if consistency_ratio >= 0.9 and avg_strength >= 0.7:
                        stability_level = "Very High"
                    elif consistency_ratio >= 0.8 and avg_strength >= 0.5:
                        stability_level = "High"
                    elif consistency_ratio >= 0.6 and avg_strength >= 0.3:
                        stability_level = "Moderate"
                    elif consistency_ratio >= 0.4:
                        stability_level = "Low"
                    else:
                        stability_level = "Very Low"
                else:
                    # For shorter history, use more lenient thresholds
                    if consistency_ratio >= 0.8 and avg_strength >= 0.6:
                        stability_level = "Emerging High"
                    elif consistency_ratio >= 0.6 and avg_strength >= 0.4:
                        stability_level = "Emerging Moderate"
                    else:
                        stability_level = "Emerging Low"
                
                signal_stability = {
                    'level': stability_level,
                    'consistency_ratio': round(consistency_ratio * 100, 1),
                    'avg_strength': round(avg_strength * 100, 1),
                    'duration': total_signals,
                    'direction': 'Bullish' if positive_signals > negative_signals else 'Bearish' if negative_signals > positive_signals else 'Neutral',
                    'confirmation_status': 'Confirmed' if total_signals >= self.technical_analysis.trend_confirmation_period else 'Developing'
                }
            else:
                signal_stability = {
                    'level': "Insufficient Data",
                    'consistency_ratio': 0.0,
                    'avg_strength': 0.0,
                    'duration': 0,
                    'direction': 'Neutral',
                    'confirmation_status': 'None'
                }
            
            # Get volume confirmation analysis
            volume_info = self._get_cached_indicator('volume_info')
            if volume_info is None:
                volume_info = self.technical_analysis.analyze_volume_confirmation(formatted_candles)
                self._cache_indicator('volume_info', volume_info)

            # Get market correlation analysis
            correlation_analysis = self.analyze_market_correlations()

            # Create detailed analysis result
            result = {
                'timestamp': datetime.now(UTC).isoformat(),
                'product_id': self.product_id,
                'current_price': current_price,
                'signal': analysis['signal'].signal_type.value,
                'position': 'LONG' if analysis['signal'].signal_type in [SignalType.STRONG_BUY, SignalType.BUY] \
                           else 'SHORT' if analysis['signal'].signal_type in [SignalType.STRONG_SELL, SignalType.SELL] \
                           else 'NEUTRAL',
                'confidence': analysis['signal'].confidence,
                'market_condition': analysis['signal'].market_condition.value,
                'risk_metrics': analysis['risk_metrics'],
                'indicators': {
                    'rsi': round(rsi, 2),
                    'macd': round(macd, 2),
                    'macd_signal': round(signal, 2),
                    'macd_histogram': round(histogram, 2),
                    'bollinger_upper': round(bb_upper, 2),
                    'bollinger_middle': round(bb_middle, 2),
                    'bollinger_lower': round(bb_lower, 2),
                    'adx': round(adx_value, 2),
                    'trend_direction': trend_direction
                },
                'recommendation': self._generate_recommendation(analysis['signal'].signal_type),
                'signal_stability': {
                    'level': signal_stability['level'],
                    'consistency': f"{signal_stability['consistency_ratio']}%",
                    'strength': f"{signal_stability['avg_strength']}%",
                    'duration': signal_stability['duration'],
                    'direction': signal_stability['direction'],
                    'confirmation_status': signal_stability['confirmation_status']
                },
                'signals_analyzed': len(self.technical_analysis.signal_history),
                'time_since_last_change': time.time() - (self.technical_analysis.last_signal_time or time.time()),
                'volume_analysis': {
                    'change': round(volume_info['volume_change'], 1),
                    'trend': volume_info['volume_trend'],
                    'strength': volume_info['strength'],
                    'is_confirming': volume_info['is_confirming'],
                    'price_change': round(volume_info['price_change'], 1)
                },
                'ml_analysis': {
                    'prediction': ml_prediction,
                    'confidence': ml_confidence,
                    'direction': 'Bullish' if ml_prediction > 0 else 'Bearish' if ml_prediction < 0 else 'Neutral',
                    'strength': abs(ml_prediction)
                },
                'sentiment_analysis': {
                    'overall': sentiment.value,
                    'news': sentiment_data['news'],
                    'social': sentiment_data['social'],
                    'history': self.sentiment_analyzer.sentiment_history[-5:]  # Last 5 sentiment records
                },
                'adaptive_weights': {
                    'current_weights': self.adaptive_weights.weights,
                    'performance_history': self.adaptive_weights.performance_history[-5:]  # Last 5 performance records
                },
                'correlation_analysis': correlation_analysis
            }

            # Add pattern recognition
            patterns = self.detect_chart_patterns(self._current_candles)
            result['patterns'] = {
                'type': patterns['type'].value,
                'confidence': patterns['confidence'],
                'target': patterns['target'],
                'stop_loss': patterns['stop_loss']
            }
            
            # Add dynamic risk calculation
            dynamic_risk = self.calculate_dynamic_risk(self._current_candles)
            result['risk_metrics']['dynamic_risk'] = dynamic_risk
            
            # Add pattern history
            result['pattern_history'] = [
                {
                    'timestamp': p['timestamp'].isoformat(),
                    'pattern': p['pattern']['type'].value,
                    'confidence': p['pattern']['confidence']
                }
                for p in self.pattern_memory[-5:]  # Last 5 patterns
            ]
            
            # Calculate probability of success
            probability = self.calculate_success_probability(
                result['indicators'],
                result['volume_analysis'],
                result['patterns']
            )
            
            result['probability_analysis'] = probability
            
            # Add momentum analysis
            momentum_analysis = self.calculate_momentum_score(formatted_candles)
            result['momentum_analysis'] = momentum_analysis
            
            # Add regime analysis
            regime_analysis = self.detect_market_regime(formatted_candles)
            result['regime_analysis'] = {
                'regime': regime_analysis['regime'].value,
                'confidence': regime_analysis['confidence'],
                'metrics': regime_analysis['metrics']
            }
            
            # Update adaptive thresholds
            self.update_adaptive_thresholds()
            
            # Add enhanced analysis
            result.update({
                'market_cycles': self.analyze_market_cycles(),
                'order_flow': self.analyze_order_flow(),
                'liquidity_zones': self.analyze_liquidity_zones(),
                'market_efficiency': self.analyze_market_efficiency(),
                'adaptive_thresholds': {
                    name: {
                        'current': threshold['current'],
                        'base': threshold['base'],
                        'adjustment': threshold['current'] / threshold['base']
                    }
                    for name, threshold in self.adaptive_thresholds.items()
                }
            })
            
            # Add divergence analysis to the result
            result['divergence_analysis'] = divergence_analysis
            
            # Update signal strength based on divergences
            if divergence_analysis.get('divergences'):
                # Adjust confidence based on highest confidence divergence
                highest_div_confidence = divergence_analysis['summary']['highest_confidence']
                result['confidence'] = (result['confidence'] + highest_div_confidence) / 2
                
                # Add divergence bias to recommendation
                div_bias = divergence_analysis['summary']['primary_bias']
                if div_bias == 'Bullish' and result['signal'] in ['BUY', 'STRONG_BUY']:
                    result['confidence'] *= 1.2  # Increase confidence when aligned
                elif div_bias == 'Bearish' and result['signal'] in ['SELL', 'STRONG_SELL']:
                    result['confidence'] *= 1.2  # Increase confidence when aligned
                else:
                    result['confidence'] *= 0.8  # Decrease confidence when conflicting

            # Add volatility regime analysis
            volatility_regime = self.analyze_volatility_regime(formatted_candles)
            
            # Add volatility regime to the result dictionary
            result['volatility_regime'] = volatility_regime

            # Add liquidity depth analysis
            try:
                liquidity_analysis = self.analyze_liquidity_depth(formatted_candles)
                result['liquidity_analysis'] = liquidity_analysis
            except Exception as e:
                self.logger.error(f"Error in liquidity analysis: {str(e)}")
                result['liquidity_analysis'] = {'status': 'error', 'error': str(e)}

            # Check for alerts
            triggered_alerts = self.alert_system.check_alerts(result)
            auto_alerts = self.alert_system.check_market_conditions(result)
            
            # Add alerts to the result
            result['alerts'] = {
                'triggered_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'message': alert.message,
                        'priority': alert.priority.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in triggered_alerts
                ],
                'auto_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'message': alert.message,
                        'priority': alert.priority.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in auto_alerts
                ],
                'alert_summary': self.alert_system.get_alert_summary()
            }
            
            return result

        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now(UTC).isoformat(),
                'product_id': self.product_id,
                'signal': 'HOLD',
                'position': 'NEUTRAL',
                'confidence': 0.0,
                'current_price': 0.0,
                'recommendation': 'Unable to generate signal due to error'
            }

    def _format_candles(self, candles: List[Dict]) -> List[Dict]:
        """Format candles to match the expected structure."""
        formatted_candles = []
        for candle in candles:
            formatted_candle = {
                'time': candle['start'],
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume'])
            }
            formatted_candles.append(formatted_candle)
        return formatted_candles

    def _generate_recommendation(self, signal_type: SignalType) -> str:
        """Generate a detailed trading recommendation including consolidation patterns and bias."""
        try:
            if not self._current_candles:
                return "No market data available for recommendation"
            
            # Get current price and indicators
            current_price = float(self._current_candles[-1]['close'])
            atr = self.technical_analysis.compute_atr(self._current_candles)
            consolidation_info = self.technical_analysis.detect_consolidation(self._current_candles)
            volume_info = self.technical_analysis.analyze_volume_confirmation(self._current_candles)
            
            # Calculate market bias
            rsi = self.technical_analysis.compute_rsi(self.product_id, self._current_candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, self._current_candles)
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(self._current_candles)
            
            # Calculate price change
            prices = self.technical_analysis.extract_prices(self._current_candles)
            price_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100 if len(prices) > 1 else 0.0
            
            # Calculate key levels
            stop_loss_atr = atr * self.ta_config.atr_multiplier
            take_profit_1r = stop_loss_atr * 1.5  # 1.5:1 reward-risk
            take_profit_2r = stop_loss_atr * 2.0  # 2:1 reward-risk
            take_profit_3r = stop_loss_atr * 3.0  # 3:1 reward-risk
            
            # Calculate support and resistance levels
            resistance_level = consolidation_info['upper_channel']
            support_level = consolidation_info['lower_channel']
            
            # Determine bias based on multiple indicators
            bias_factors = []
            
            # RSI bias
            if rsi > 50:
                bias_factors.append(("Bullish", (rsi - 50) / 50))
            else:
                bias_factors.append(("Bearish", (50 - rsi) / 50))
            
            # MACD bias
            if macd > signal:
                bias_factors.append(("Bullish", abs(histogram / macd) if macd != 0 else 0.1))
            else:
                bias_factors.append(("Bearish", abs(histogram / macd) if macd != 0 else 0.1))
            
            # Trend direction bias
            if trend_direction == "Uptrend":
                bias_factors.append(("Bullish", adx_value / 100))
            elif trend_direction == "Downtrend":
                bias_factors.append(("Bearish", adx_value / 100))
            
            # Volume trend bias
            if volume_info['volume_trend'] == "Increasing":
                if price_change > 0:
                    bias_factors.append(("Bullish", 0.3))
                else:
                    bias_factors.append(("Bearish", 0.3))
                
            # Calculate overall bias
            bullish_strength = sum(strength for direction, strength in bias_factors if direction == "Bullish")
            bearish_strength = sum(strength for direction, strength in bias_factors if direction == "Bearish")
            
            # Determine dominant bias
            if bullish_strength > bearish_strength:
                bias = f"Neutral with Bullish Bias (Strength: {(bullish_strength - bearish_strength):.1f})"
            elif bearish_strength > bullish_strength:
                bias = f"Neutral with Bearish Bias (Strength: {(bearish_strength - bullish_strength):.1f})"
            else:
                bias = "Neutral with No Clear Bias"
            
            # Define recommendations dictionary
            recommendations = {
                SignalType.STRONG_BUY: {
                    'position': 'LONG',
                    'message': f"Strong buy signal detected. Consider opening a LONG position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: LONG\n"
                              f"• Leverage: 1-2x maximum\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Losses:\n"
                              f"• Conservative: ${(current_price - stop_loss_atr):.4f} (-{(stop_loss_atr/current_price)*100:.1f}%)\n"
                              f"• Aggressive: ${(current_price - (stop_loss_atr*0.7)):.4f} (-{(stop_loss_atr*0.7/current_price)*100:.1f}%)\n\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price + take_profit_1r):.4f} (+{(take_profit_1r/current_price)*100:.1f}%)\n"
                              f"• Target 2 (2R): ${(current_price + take_profit_2r):.4f} (+{(take_profit_2r/current_price)*100:.1f}%)\n"
                              f"• Target 3 (3R): ${(current_price + take_profit_3r):.4f} (+{(take_profit_3r/current_price)*100:.1f}%)"
                },
                SignalType.STRONG_SELL: {
                    'position': 'SHORT',
                    'message': f"Strong sell signal detected. Consider opening a SHORT position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: SHORT\n"
                              f"• Leverage: 1-2x maximum\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Losses:\n"
                              f"• Conservative: ${(current_price + stop_loss_atr):.4f} (+{(stop_loss_atr/current_price)*100:.1f}%)\n"
                              f"• Aggressive: ${(current_price + (stop_loss_atr*0.7)):.4f} (+{(stop_loss_atr*0.7/current_price)*100:.1f}%)\n\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price - take_profit_1r):.4f} (-{(take_profit_1r/current_price)*100:.1f}%)\n"
                              f"• Target 2 (2R): ${(current_price - take_profit_2r):.4f} (-{(take_profit_2r/current_price)*100:.1f}%)\n"
                              f"• Target 3 (3R): ${(current_price - take_profit_3r):.4f} (-{(take_profit_3r/current_price)*100:.1f}%)"
                },
                SignalType.BUY: {
                    'position': 'LONG',
                    'message': f"Bullish conditions detected. Consider a conservative LONG position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: LONG\n"
                              f" Leverage: 1x only\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Loss: Place below support at ${(support_level - (atr * 0.5)):.4f}\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price + take_profit_1r):.4f}\n"
                              f"• Target 2 (2R): ${(current_price + take_profit_2r):.4f}"
                },
                SignalType.SELL: {
                    'position': 'SHORT',
                    'message': f"Bearish conditions detected. Consider a conservative SHORT position:\n"
                              f"• Entry Price: ${current_price:.4f}\n"
                              f"• Position Type: SHORT\n"
                              f"• Leverage: 1x only\n\n"
                              f"Support Level: ${support_level:.4f}\n"
                              f"Resistance Level: ${resistance_level:.4f}\n\n"
                              f"Stop Loss: Place above resistance at ${(resistance_level + (atr * 0.5)):.4f}\n"
                              f"Take Profit Targets:\n"
                              f"• Target 1 (1.5R): ${(current_price - take_profit_1r):.4f}\n"
                              f"• Target 2 (2R): ${(current_price - take_profit_2r):.4f}"
                },
                SignalType.HOLD: {
                    'position': 'NEUTRAL',
                    'message': f"Market conditions are neutral:\n"
                              f"• Current Price: ${current_price:.4f}\n"
                              f"• ATR: ${atr:.4f}\n\n"
                              f"Market Bias:\n"
                              f"• Current Bias: {bias}\n"
                              f"• RSI Position: {'Bullish' if rsi > 50 else 'Bearish'} ({rsi:.1f})\n"
                              f"• MACD Status: {'Bullish' if macd > signal else 'Bearish'}\n"
                              f"• Trend Direction: {trend_direction}\n"
                              f"• Volume Trend: {volume_info['volume_trend']}\n\n"
                              f"Key Levels:\n"
                              f"• Support: ${support_level:.4f}\n"
                              f"• Resistance: ${resistance_level:.4f}\n\n"
                              f"Recommendation:\n"
                              f"• Hold existing positions\n"
                              f"• Wait for price to break ${resistance_level:.4f} resistance or\n"
                              f"  ${support_level:.4f} support with volume confirmation\n"
                              f"• Prepare for potential {'bullish' if bullish_strength > bearish_strength else 'bearish'} move"
                }
            }
            
            # Add volume analysis message
            volume_message = f"\nVolume Analysis:\n" \
                            f"• Volume Change: {volume_info['volume_change']:.1f}%\n" \
                            f"• Volume Trend: {volume_info['volume_trend']}\n" \
                            f"• Signal Strength: {volume_info['strength']}\n" \
                            f"• Price Change: {volume_info['price_change']:.1f}%\n" \
                            f"• Volume Confirmation: {'Yes' if volume_info['is_confirming'] else 'No'}"
            
            # Add volume message to each recommendation
            for key in recommendations:
                recommendations[key]['message'] = recommendations[key]['message'] + volume_message
            
            rec = recommendations.get(signal_type, {
                'position': 'NEUTRAL',
                'message': "No clear trading opportunity. Wait for better setup." + volume_message
            })
            
            # Add consolidation information if relevant
            consolidation_message = ""
            if consolidation_info['is_consolidating']:
                if consolidation_info['pattern'] == "Breakout":
                    target = consolidation_info['upper_channel'] + (consolidation_info['upper_channel'] - consolidation_info['lower_channel'])
                    consolidation_message = f"\nBreakout Pattern Detected:\n" \
                                          f"• Breakout Level: ${consolidation_info['upper_channel']:.4f}\n" \
                                          f"• Volume Confirmation: {'Yes' if consolidation_info['volume_confirmed'] else 'No'}\n" \
                                          f"• Suggested Stop: ${consolidation_info['channel_middle']:.4f}\n" \
                                          f"• Target: ${target:.4f}"
                elif consolidation_info['pattern'] == "Breakdown":
                    target = consolidation_info['lower_channel'] - (consolidation_info['upper_channel'] - consolidation_info['lower_channel'])
                    consolidation_message = f"\nBreakdown Pattern Detected:\n" \
                                          f"• Breakdown Level: ${consolidation_info['lower_channel']:.4f}\n" \
                                          f"• Volume Confirmation: {'Yes' if consolidation_info['volume_confirmed'] else 'No'}\n" \
                                          f"• Suggested Stop: ${consolidation_info['channel_middle']:.4f}\n" \
                                          f"• Target: ${target:.4f}"
                else:
                    consolidation_message = f"\nConsolidation Phase Detected:\n" \
                                          f"• Upper Channel: ${consolidation_info['upper_channel']:.4f}\n" \
                                          f"• Lower Channel: ${consolidation_info['lower_channel']:.4f}\n" \
                                          f"• Channel Middle: ${consolidation_info['channel_middle']:.4f}\n" \
                                          f"• Strength: {consolidation_info['strength']*100:.1f}%"
            
            # Add rejection analysis if detected
            if consolidation_info.get('rejection_event'):
                rejection = consolidation_info['rejection_event']
                rejection_time = datetime.fromtimestamp(float(rejection['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
                
                confirmation_strength = "Strong" if rejection['confirming_candles'] >= 2 and rejection['volume_confirmation'] else \
                                      "Moderate" if rejection['confirming_candles'] >= 1 or rejection['volume_confirmation'] else \
                                      "Weak"
                
                if rejection['type'] == 'resistance':
                    consolidation_message += f"\nMost Recent Resistance Rejection:\n" \
                                           f"• Time: {rejection_time}\n" \
                                           f"• Rejection Level: ${rejection['price_level']:.4f}\n" \
                                           f"• Current Price: ${rejection['price']:.4f}\n" \
                                           f"• Distance from Level: {rejection['distance_from_level']:.1f}%\n" \
                                           f"• Rejection Volume: {rejection['volume']:.2f}\n" \
                                           f"• Volume vs Average: {rejection['volume_ratio']:.1f}x\n" \
                                           f"• Confirming Candles: {rejection['confirming_candles']}\n" \
                                           f"• Volume Confirmation: {'Yes' if rejection['volume_confirmation'] else 'No'}\n" \
                                           f"• Confirmation Strength: {confirmation_strength}"
                else:  # support rejection
                    consolidation_message += f"\nMost Recent Support Bounce:\n" \
                                           f"• Time: {rejection_time}\n" \
                                           f"• Support Level: ${rejection['price_level']:.4f}\n" \
                                           f"• Current Price: ${rejection['price']:.4f}\n" \
                                           f"• Distance from Level: {rejection['distance_from_level']:.1f}%\n" \
                                           f"• Bounce Volume: {rejection['volume']:.2f}\n" \
                                           f"• Volume vs Average: {rejection['volume_ratio']:.1f}x\n" \
                                           f"• Confirming Candles: {rejection['confirming_candles']}\n" \
                                           f"• Volume Confirmation: {'Yes' if rejection['volume_confirmation'] else 'No'}\n" \
                                           f"• Confirmation Strength: {confirmation_strength}"
            
            return f"Position: {rec['position']}\n\n{rec['message']}{consolidation_message}"
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return "Error generating recommendation"

    def _determine_signal_type(self, signal_strength: float) -> SignalType:
        """
        Determine signal type based on signal strength.
        More balanced thresholds for consistent signals.
        
        Args:
            signal_strength: Float value between -10 and 10
            
        Returns:
            SignalType: The determined signal type
        """
        if signal_strength >= 3.5:  # Reduced from 4
            return SignalType.STRONG_BUY
        elif signal_strength >= 1.2:  # Reduced from 1.5
            return SignalType.BUY
        elif signal_strength <= -3.5:  # Changed from -4
            return SignalType.STRONG_SELL
        elif signal_strength <= -1.2:  # Changed from -1.5
            return SignalType.SELL
        else:
            # Check if there's a slight bias even in the "neutral" zone
            if signal_strength > 0.3:  # Reduced from 0.5 for more sensitivity
                return SignalType.BUY
            elif signal_strength < -0.3:  # Changed from -0.5 for more sensitivity
                return SignalType.SELL
            return SignalType.HOLD

    def _calculate_base_signal(self, indicators: Dict[str, float], market_condition: str) -> float:
        """Calculate the base signal with enhanced features."""
        try:
            signal_strength = 0.0
            indicator_signals = {}
            
            # Technical indicators
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                rsi_signal = ((rsi - 50) / 50) * self.adaptive_weights.get_weight('rsi')
                indicator_signals['rsi'] = rsi_signal
                signal_strength += rsi_signal

            # MACD
            if all(k in indicators for k in ['macd', 'macd_signal', 'macd_histogram']):
                macd = indicators['macd']
                macd_signal = indicators['macd_signal']
                histogram = indicators['macd_histogram']
                
                macd_strength = (histogram / (abs(macd) + 0.00001)) * self.adaptive_weights.get_weight('macd')
                indicator_signals['macd'] = macd_strength
                signal_strength += macd_strength

            # Bollinger Bands
            if all(k in indicators for k in ['bollinger_upper', 'bollinger_middle', 'bollinger_lower']):
                current_price = indicators.get('current_price', 0)
                bb_upper = indicators['bollinger_upper']
                bb_lower = indicators['bollinger_lower']
                bb_middle = indicators['bollinger_middle']
                
                bb_signal = 0.0
                if current_price > bb_upper:
                    bb_signal = -1.0
                elif current_price < bb_lower:
                    bb_signal = 1.0
                else:
                    bb_signal = (current_price - bb_middle) / (bb_upper - bb_middle)
                
                bb_strength = bb_signal * self.adaptive_weights.get_weight('bollinger')
                indicator_signals['bollinger'] = bb_strength
                signal_strength += bb_strength

            # ADX and Trend
            if 'adx' in indicators and 'trend_direction' in indicators:
                adx = indicators['adx']
                trend = indicators['trend_direction']
                
                adx_signal = (adx / 100.0) * (1 if trend == "Uptrend" else -1 if trend == "Downtrend" else 0)
                adx_strength = adx_signal * self.adaptive_weights.get_weight('adx')
                indicator_signals['adx'] = adx_strength
                signal_strength += adx_strength

            # Volume Analysis
            if 'volume_trend' in indicators:
                volume_trend = indicators['volume_trend']
                volume_signal = 1 if volume_trend == "Increasing" else -1 if volume_trend == "Decreasing" else 0
                volume_strength = volume_signal * self.adaptive_weights.get_weight('volume')
                indicator_signals['volume'] = volume_strength
                signal_strength += volume_strength

            # Add ML model prediction
            ml_prediction = self.ml_model.predict(self._current_candles)
            ml_strength = ml_prediction * self.adaptive_weights.get_weight('ml_prediction')
            indicator_signals['ml_prediction'] = ml_strength
            signal_strength += ml_strength

            # Add sentiment analysis
            sentiment = self.sentiment_analyzer.get_aggregated_sentiment(self.product_id)
            sentiment_score = 1.0 if sentiment in [MarketSentiment.VERY_BULLISH, MarketSentiment.BULLISH] else \
                            -1.0 if sentiment in [MarketSentiment.VERY_BEARISH, MarketSentiment.BEARISH] else 0.0
            sentiment_strength = sentiment_score * self.adaptive_weights.get_weight('sentiment')
            indicator_signals['sentiment'] = sentiment_strength
            signal_strength += sentiment_strength

            # Market Condition Adjustment
            condition_multipliers = {
                "Bull Market": 1.2,
                "Bear Market": 0.8,
                "Bullish": 1.1,
                "Bearish": 0.9,
                "Neutral": 1.0
            }
            
            # Apply market condition multiplier
            multiplier = condition_multipliers.get(market_condition, 1.0)
            signal_strength *= multiplier

            # Normalize signal strength to be between -10 and 10
            signal_strength = max(min(signal_strength * 1.5, 10), -10)

            # Record indicator performances for weight adaptation
            if len(self._current_candles) >= 2:
                actual_movement = (self._current_candles[-1]['close'] - self._current_candles[-2]['close']) / self._current_candles[-2]['close']
                self.adaptive_weights.record_performance(indicator_signals, actual_movement)

            self.logger.debug(f"Base signal strength calculated: {signal_strength}")
            return signal_strength

        except Exception as e:
            self.logger.error(f"Error calculating base signal: {str(e)}")
            return 0.0

    def detect_chart_patterns(self, candles: List[Dict]) -> Dict:
        """Detect common chart patterns in the price data."""
        try:
            if len(candles) < 30:
                return {'type': PatternType.NONE, 'confidence': 0.0}
                
            prices = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            patterns = {
                'type': PatternType.NONE,
                'confidence': 0.0,
                'target': None,
                'stop_loss': None,
                'completion_percentage': 0.0,
                'invalidation_level': None
            }
            
            # Head and Shoulders Detection
            if self._is_head_and_shoulders(highs, lows):
                patterns['type'] = PatternType.HEAD_SHOULDERS
                patterns['confidence'] = 0.8
                # Target is typically the distance from head to neckline projected downward
                head_height = max(highs[-20:]) - min(lows[-20:])
                patterns['target'] = min(lows[-20:]) - head_height
                patterns['stop_loss'] = max(highs[-20:]) + (head_height * 0.1)
                
            # Inverse Head and Shoulders Detection
            elif self._is_inverse_head_and_shoulders(highs, lows):
                patterns['type'] = PatternType.INV_HEAD_SHOULDERS
                patterns['confidence'] = 0.8
                # Target is typically the distance from head to neckline projected upward
                head_depth = max(highs[-20:]) - min(lows[-20:])
                patterns['target'] = max(highs[-20:]) + head_depth
                patterns['stop_loss'] = min(lows[-20:]) - (head_depth * 0.1)
                
            # Triangle Detection
            elif self._is_triangle(highs, lows):
                patterns['type'] = PatternType.TRIANGLE
                patterns['confidence'] = 0.75
                height = max(highs[-20:]) - min(lows[-20:])
                if highs[-1] > highs[-2]:  # Ascending triangle
                    patterns['target'] = max(highs[-20:]) + height
                else:  # Descending or symmetrical triangle
                    patterns['target'] = min(lows[-20:]) - height
                patterns['stop_loss'] = lows[-1] - (height * 0.1)
                
            # Channel Detection
            elif self._is_channel(highs, lows):
                patterns['type'] = PatternType.CHANNEL
                patterns['confidence'] = 0.7
                channel_height = np.mean(highs[-10:]) - np.mean(lows[-10:])
                if prices[-1] > prices[-5]:  # Ascending channel
                    patterns['target'] = prices[-1] + channel_height
                else:  # Descending channel
                    patterns['target'] = prices[-1] - channel_height
                patterns['stop_loss'] = prices[-1] - (channel_height * 0.5)
                
            # Flag/Pennant Detection
            elif self._is_flag_or_pennant(prices, volumes):
                is_flag = abs(np.corrcoef(range(len(prices[-10:])), prices[-10:])[0, 1]) > 0.8
                patterns['type'] = PatternType.FLAG if is_flag else PatternType.PENNANT
                patterns['confidence'] = 0.7
                trend_height = abs(prices[-1] - prices[-10])
                if prices[-10] < prices[-1]:  # Bullish flag/pennant
                    patterns['target'] = prices[-1] + trend_height
                else:  # Bearish flag/pennant
                    patterns['target'] = prices[-1] - trend_height
                patterns['stop_loss'] = prices[-10]
                
            # Cup and Handle Detection
            elif self._is_cup_and_handle(prices, volumes):
                patterns['type'] = PatternType.CUP_HANDLE
                patterns['confidence'] = 0.8
                cup_depth = max(prices[-30:]) - min(prices[-30:])
                patterns['target'] = max(prices[-30:]) + cup_depth
                patterns['stop_loss'] = min(prices[-10:])
                
            # Rounding Bottom/Top Detection
            elif self._is_rounding_pattern(prices, volumes):
                is_bottom = prices[-1] > np.mean(prices[-20:])
                patterns['type'] = PatternType.ROUNDING_BOTTOM if is_bottom else PatternType.ROUNDING_TOP
                patterns['confidence'] = 0.7
                pattern_height = max(prices[-20:]) - min(prices[-20:])
                if is_bottom:
                    patterns['target'] = prices[-1] + pattern_height
                else:
                    patterns['target'] = prices[-1] - pattern_height
                patterns['stop_loss'] = min(prices[-20:]) if is_bottom else max(prices[-20:])
                
            # Double Top/Bottom Detection (existing code)
            elif self._is_double_top(highs):
                patterns['type'] = PatternType.DOUBLE_TOP
                patterns['confidence'] = 0.8
                patterns['target'] = min(lows[-30:])
                patterns['stop_loss'] = max(highs[-30:]) + (max(highs[-30:]) - min(lows[-30:])) * 0.1
                
            elif self._is_double_bottom(lows):
                patterns['type'] = PatternType.DOUBLE_BOTTOM
                patterns['confidence'] = 0.8
                patterns['target'] = max(highs[-30:])
                patterns['stop_loss'] = min(lows[-30:]) - (max(highs[-30:]) - min(lows[-30:])) * 0.1
                
            # Calculate pattern completion percentage
            if patterns['type'] != PatternType.NONE:
                patterns['completion_percentage'] = self._calculate_pattern_completion(
                    patterns['type'],
                    prices,
                    highs,
                    lows,
                    volumes
                )
                
            # Add pattern to memory
            if patterns['type'] != PatternType.NONE:
                self.pattern_memory.append({
                    'timestamp': datetime.now(UTC),
                    'pattern': patterns
                })
                
                # Maintain pattern memory size
                if len(self.pattern_memory) > self.max_pattern_memory:
                    self.pattern_memory.pop(0)
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting chart patterns: {str(e)}")
            return {'type': PatternType.NONE, 'confidence': 0.0}

    def calculate_dynamic_risk(self, candles: List[Dict]) -> float:
        """Calculate dynamic risk based on market volatility."""
        try:
            # Calculate current volatility
            returns = np.diff(np.log([c['close'] for c in candles]))
            current_vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Store volatility
            self.volatility_history.append(current_vol)
            if len(self.volatility_history) > 30:
                self.volatility_history.pop(0)
            
            # Calculate relative volatility
            avg_vol = np.mean(self.volatility_history)
            vol_ratio = current_vol / avg_vol if avg_vol != 0 else 1
            
            # Adjust risk based on volatility
            dynamic_risk = self.base_risk * (1 / vol_ratio)
            
            # Constrain risk within bounds
            return max(min(dynamic_risk, self.max_risk), self.min_risk)
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic risk: {str(e)}")
            return self.base_risk

    def _is_double_top(self, prices: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect double top pattern."""
        try:
            peaks = self._find_peaks(prices)
            if len(peaks) < 2:
                return False
                
            # Get last two peaks
            last_peaks = peaks[-2:]
            peak_prices = prices[last_peaks]
            
            # Check if peaks are within threshold
            price_diff = abs(peak_prices[0] - peak_prices[1]) / peak_prices[0]
            time_diff = last_peaks[1] - last_peaks[0]
            
            return price_diff < threshold and 5 <= time_diff <= 20
            
        except Exception as e:
            self.logger.error(f"Error detecting double top: {str(e)}")
            return False

    def _is_double_bottom(self, prices: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect double bottom pattern."""
        try:
            troughs = self._find_troughs(prices)
            if len(troughs) < 2:
                return False
                
            # Get last two troughs
            last_troughs = troughs[-2:]
            trough_prices = prices[last_troughs]
            
            # Check if troughs are within threshold
            price_diff = abs(trough_prices[0] - trough_prices[1]) / trough_prices[0]
            time_diff = last_troughs[1] - last_troughs[0]
            
            return price_diff < threshold and 5 <= time_diff <= 20
            
        except Exception as e:
            self.logger.error(f"Error detecting double bottom: {str(e)}")
            return False

    def _find_peaks(self, prices: np.ndarray, window: int = 5) -> List[int]:
        """Find peaks in price data."""
        peaks = []
        for i in range(window, len(prices) - window):
            if all(prices[i] > prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] > prices[i+j] for j in range(1, window+1)):
                peaks.append(i)
        return peaks

    def _find_troughs(self, prices: np.ndarray, window: int = 5) -> List[int]:
        """Find troughs in price data."""
        troughs = []
        for i in range(window, len(prices) - window):
            if all(prices[i] < prices[i-j] for j in range(1, window+1)) and \
               all(prices[i] < prices[i+j] for j in range(1, window+1)):
                troughs.append(i)
        return troughs

    def _generate_error_response(self) -> Dict:
        """Generate standardized error response."""
        return {
            'error': 'Analysis error',
            'timestamp': datetime.now(UTC).isoformat(),
            'product_id': self.product_id,
            'signal': 'HOLD',
            'position': 'NEUTRAL',
            'confidence': 0.0,
            'patterns': {'type': PatternType.NONE.value, 'confidence': 0.0},
            'risk_metrics': {'dynamic_risk': self.base_risk}
        }

    def calculate_success_probability(self, indicators: Dict, volume_info: Dict, patterns: Dict) -> Dict:
        """Calculate probability of success for the suggested direction with detailed move analysis."""
        try:
            probability_factors = []
            move_characteristics = {}
            
            # Enhanced trend analysis with momentum consideration
            trend_direction = indicators.get('trend_direction', 'Unknown')
            adx = indicators.get('adx', 0)
            prev_adx = indicators.get('prev_adx', 0)
            
            # Dynamic trend scoring based on ADX strength and momentum
            trend_momentum = "Accelerating" if adx > prev_adx else "Decelerating"
            trend_strength = "Strong" if adx > 25 else "Moderate" if adx > 15 else "Weak"
            
            # More dynamic trend scoring
            if trend_direction == "Uptrend":
                trend_score = min(25, (adx / 50) * 20)  # Max 25% for very strong trends
                if trend_momentum == "Accelerating":
                    trend_score *= 1.2  # 20% bonus for accelerating trends
            elif trend_direction == "Downtrend":
                trend_score = min(25, (adx / 50) * 20)
                if trend_momentum == "Accelerating":
                    trend_score *= 1.2
            else:
                trend_score = 5
            
            probability_factors.append(("Trend", trend_score))
            
            # Enhanced RSI analysis with dynamic scoring
            rsi = indicators.get('rsi', 50)
            rsi_prev = indicators.get('prev_rsi', 50)
            price = indicators.get('current_price', 0)
            price_prev = indicators.get('prev_price', price)
            
            # Calculate RSI momentum and divergence
            rsi_momentum = abs(rsi - rsi_prev) / 2
            rsi_divergence = "Bullish" if rsi > rsi_prev and price < price_prev else \
                           "Bearish" if rsi < rsi_prev and price > price_prev else "None"
            
            # Dynamic RSI scoring
            rsi_score = min(20, abs(rsi - 50) / 50 * 15)  # Base score
            if rsi_divergence != "None":
                rsi_score *= 1.3  # 30% bonus for divergence
            if rsi_momentum > 2:
                rsi_score *= 1.2  # 20% bonus for strong momentum
                
            probability_factors.append(("RSI", rsi_score))
            
            # Enhanced MACD analysis with dynamic scoring
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            histogram = indicators.get('macd_histogram', 0)
            prev_histogram = indicators.get('prev_macd_histogram', 0)
            
            # Calculate MACD momentum and strength
            macd_strength = abs(macd - macd_signal) / (abs(macd) + 0.00001)
            macd_momentum = (histogram - prev_histogram) / (abs(prev_histogram) + 0.00001)
            
            # Dynamic MACD scoring
            macd_score = min(20, macd_strength * 15)  # Base score
            if macd_momentum > 0:
                macd_score *= 1.2  # 20% bonus for positive momentum
            if abs(histogram) > abs(prev_histogram):
                macd_score *= 1.1  # 10% bonus for increasing histogram
                
            probability_factors.append(("MACD", macd_score))
            
            # Enhanced volume analysis with dynamic scoring
            volume_change = volume_info.get('volume_change', 0)
            is_confirming = volume_info.get('is_confirming', False)
            volume_trend = volume_info.get('volume_trend', 'Neutral')
            
            # Dynamic volume scoring
            volume_score = min(20, abs(volume_change) / 100 * 15)  # Base score
            if is_confirming:
                volume_score *= 1.3  # 30% bonus for confirmation
            if volume_trend == "Increasing":
                volume_score *= 1.2  # 20% bonus for increasing trend
                
            probability_factors.append(("Volume", volume_score))
            
            # Enhanced pattern analysis with completion percentage
            pattern_type = patterns.get('type', 'None')
            pattern_confidence = patterns.get('confidence', 0)
            completion = patterns.get('completion_percentage', 0)
            
            # Dynamic pattern scoring
            if pattern_type != "None":
                pattern_score = min(20, pattern_confidence * 15)  # Base score
                pattern_score *= (0.5 + completion / 200)  # Adjust based on completion
                if is_confirming:
                    pattern_score *= 1.2  # 20% bonus for volume confirmation
            else:
                pattern_score = 5
                
            probability_factors.append(("Pattern", pattern_score))
            
            # Calculate total probability with dynamic weighting
            total_score = sum(score for _, score in probability_factors)
            max_possible_score = 100  # Maximum theoretical score
            
            # Normalize to 0-100 range with sigmoid function for smoother distribution
            normalized_score = 100 / (1 + np.exp(-0.1 * (total_score - max_possible_score/2)))
            
            # Determine move quality characteristics
            move_quality = {
                'expected_speed': 'Rapid' if normalized_score > 80 else 'Moderate' if normalized_score > 60 else 'Slow',
                'expected_volatility': 'High' if volume_info.get('volume_change', 0) > 0 else 'Normal',
                'continuation_probability': f"{normalized_score:.1f}%",
                'reversal_risk': 'Low' if normalized_score > 75 else 'Moderate' if normalized_score > 50 else 'High',
                'strength_rating': 'Very Strong' if normalized_score > 85 else
                                 'Strong' if normalized_score > 70 else
                                 'Moderate' if normalized_score > 50 else 'Weak'
            }

            # Calculate confidence level with more granularity
            confidence_level = "Very High" if normalized_score >= 85 else \
                             "High" if normalized_score >= 70 else \
                             "Moderate" if normalized_score >= 50 else \
                             "Low" if normalized_score >= 30 else "Very Low"

            # Add failure points analysis
            failure_points = {
                'immediate_stop': patterns.get('stop_loss', None),
                'trend_reversal_point': patterns.get('invalidation_level', None),
                'momentum_failure_level': indicators.get('key_reversal_level', None),
                'risk_levels': {
                    'critical': patterns.get('stop_loss', None),
                    'warning': indicators.get('support_level' if trend_direction == "Uptrend" else 'resistance_level', None),
                    'alert': indicators.get('pivot_point', None)
                }
            }

            return {
                'total_probability': normalized_score,
                'confidence_level': confidence_level,
                'factors': [(factor[0], round(factor[1], 1)) for factor in probability_factors],
                'move_characteristics': move_characteristics,
                'move_quality': move_quality,
                'failure_points': failure_points
            }

        except Exception as e:
            self.logger.error(f"Error calculating success probability: {str(e)}")
            return {
                'total_probability': 0,
                'confidence_level': "Low",
                'factors': [("Error", 0)],
                'move_characteristics': {},
                'move_quality': {},
                'failure_points': {}
            }

    def calculate_momentum_score(self, candles: List[Dict]) -> Dict:
        """
        Calculate market momentum score using multiple indicators.
        Returns score between -100 (strong bearish) to +100 (strong bullish).
        """
        try:
            # Extract price data
            prices = np.array([c['close'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Calculate momentum indicators
            rsi = self.technical_analysis.compute_rsi(self.product_id, candles)
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, candles)
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(candles)
            
            # Calculate rate of change
            roc = ((prices[-1] - prices[-20]) / prices[-20]) * 100
            
            # Volume momentum
            vol_sma = np.mean(volumes[-20:])
            vol_momentum = ((volumes[-1] - vol_sma) / vol_sma) * 100
            
            # Calculate component scores
            rsi_score = ((rsi - 50) * 2)  # -100 to +100
            macd_score = (histogram / abs(macd) if abs(macd) > 0 else 0) * 100
            trend_score = adx_value * (1 if trend_direction == "Uptrend" else -1)
            roc_score = min(max(roc * 2, -100), 100)
            vol_score = min(max(vol_momentum, -100), 100)
            
            # Weight the components
            weights = {
                'rsi': 0.2,
                'macd': 0.25,
                'trend': 0.25,
                'roc': 0.2,
                'volume': 0.1
            }
            
            # Calculate final score
            momentum_score = (
                rsi_score * weights['rsi'] +
                macd_score * weights['macd'] +
                trend_score * weights['trend'] +
                roc_score * weights['roc'] +
                vol_score * weights['volume']
            )
            
            return {
                'total_score': round(momentum_score, 2),
                'components': {
                    'rsi_score': round(rsi_score, 2),
                    'macd_score': round(macd_score, 2),
                    'trend_score': round(trend_score, 2),
                    'roc_score': round(roc_score, 2),
                    'volume_score': round(vol_score, 2)
                },
                'interpretation': 'Strong Bullish' if momentum_score > 70 else
                                'Bullish' if momentum_score > 30 else
                                'Neutral' if momentum_score > -30 else
                                'Bearish' if momentum_score > -70 else
                                'Strong Bearish'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {str(e)}")
            return {'total_score': 0, 'components': {}, 'interpretation': 'Error'}

    def detect_market_regime(self, candles: List[Dict]) -> Dict:
        """
        Detect current market regime using volatility, trend strength, and price patterns.
        """
        try:
            prices = np.array([c['close'] for c in candles])
            
            # Calculate volatility
            returns = np.diff(np.log(prices))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Get trend strength
            adx_value, trend_direction = self.technical_analysis.get_trend_strength(candles)
            
            # Calculate price range
            price_range = (np.max(prices) - np.min(prices)) / np.mean(prices)
            
            # Detect regime
            regime = MarketRegime.RANGING
            confidence = 0.0
            
            if volatility > 0.04:  # High volatility threshold
                regime = MarketRegime.VOLATILE
                confidence = min((volatility - 0.04) * 10, 1.0)
            elif adx_value > 25:  # Strong trend threshold
                regime = MarketRegime.TRENDING
                confidence = min((adx_value - 25) / 75, 1.0)
            elif price_range < 0.02:  # Tight range threshold
                regime = MarketRegime.RANGING
                confidence = min((0.02 - price_range) * 50, 1.0)
            
            # Check for breakouts
            bb_upper, bb_middle, bb_lower = self.technical_analysis.compute_bollinger_bands(candles)
            if prices[-1] > bb_upper or prices[-1] < bb_lower:
                regime = MarketRegime.BREAKOUT
                confidence = min(abs(prices[-1] - bb_middle) / (bb_upper - bb_middle), 1.0)
            
            # Check for reversals
            momentum = self.calculate_momentum_score(candles)
            if abs(momentum['total_score']) > 70 and np.sign(momentum['total_score']) != np.sign(returns[-1]):
                regime = MarketRegime.REVERSAL
                confidence = min(abs(momentum['total_score']) / 100, 1.0)
            
            return {
                'regime': regime,
                'confidence': round(confidence * 100, 2),
                'metrics': {
                    'volatility': round(volatility * 100, 2),
                    'trend_strength': round(adx_value, 2),
                    'price_range': round(price_range * 100, 2)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': MarketRegime.RANGING, 'confidence': 0.0, 'metrics': {}}

    def analyze_market_correlations(self) -> Dict:
        """
        Analyze correlations between the current cryptocurrency and other major cryptocurrencies.
        Returns correlation coefficients and insights about market relationships.
        """
        try:
            # Get historical data for current product and correlation assets
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=30)
            
            correlations = {}
            current_product_data = None
            
            # Get data for current product
            try:
                current_product_data = self._get_historical_data_with_retry(start_time, end_time)
                current_prices = np.array([float(c['close']) for c in current_product_data])
            except Exception as e:
                self.logger.error(f"Error getting data for {self.product_id}: {str(e)}")
                return {'error': str(e)}
            
            # Calculate correlations with each asset
            for asset in self.correlation_assets:
                if asset == self.product_id:
                    continue
                    
                try:
                    # Get comparison asset data
                    asset_data = self._get_historical_data_with_retry(
                        start_time,
                        end_time,
                        product_id=asset
                    )
                    asset_prices = np.array([float(c['close']) for c in asset_data])
                    
                    # Ensure both arrays have the same length
                    min_length = min(len(current_prices), len(asset_prices))
                    if min_length < 2:
                        continue
                        
                    current_prices_adj = current_prices[-min_length:]
                    asset_prices_adj = asset_prices[-min_length:]
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(current_prices_adj, asset_prices_adj)[0, 1]
                    
                    # Calculate recent price movements
                    current_change = (current_prices_adj[-1] - current_prices_adj[0]) / current_prices_adj[0]
                    asset_change = (asset_prices_adj[-1] - asset_prices_adj[0]) / asset_prices_adj[0]
                    
                    # Determine correlation strength and type
                    correlation_strength = abs(correlation)
                    correlation_type = "Positive" if correlation > 0 else "Negative"
                    
                    correlations[asset] = {
                        'coefficient': correlation,
                        'strength': 'Strong' if correlation_strength > 0.7 else
                                  'Moderate' if correlation_strength > 0.4 else 'Weak',
                        'type': correlation_type,
                        'price_movement': {
                            'current_asset': f"{current_change*100:.1f}%",
                            'correlated_asset': f"{asset_change*100:.1f}%"
                        },
                        'timestamp': datetime.now(UTC).isoformat()
                    }
                    
                    # Store correlation history
                    if asset not in self.correlation_history:
                        self.correlation_history[asset] = []
                    
                    self.correlation_history[asset].append({
                        'timestamp': datetime.now(UTC),
                        'correlation': correlation,
                        'current_price_change': current_change,
                        'asset_price_change': asset_change
                    })
                    
                    # Keep history size in check
                    if len(self.correlation_history[asset]) > 100:
                        self.correlation_history[asset].pop(0)
                        
                except Exception as e:
                    self.logger.error(f"Error calculating correlation with {asset}: {str(e)}")
                    correlations[asset] = {'error': str(e)}
            
            # Calculate average correlation
            valid_correlations = [c['coefficient'] for c in correlations.values() 
                                if isinstance(c.get('coefficient'), (int, float))]
            avg_correlation = np.mean(valid_correlations) if valid_correlations else 0
            
            # Determine market independence
            independence_score = 1 - abs(avg_correlation)
            
            return {
                'correlations': correlations,
                'average_correlation': avg_correlation,
                'independence_score': independence_score,
                'interpretation': {
                    'market_independence': 'High' if independence_score > 0.7 else
                                         'Moderate' if independence_score > 0.4 else 'Low',
                    'dominant_correlation': 'Positive' if avg_correlation > 0 else 'Negative',
                    'trading_implications': [
                        "High independence suggests unique price drivers",
                        "Consider pair trading opportunities" if abs(avg_correlation) > 0.7 else
                        "Market specific factors dominate price action"
                    ]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in market correlation analysis: {str(e)}")
            return {'error': str(e)}

    def detect_divergences(self, candles: List[Dict], lookback_period: int = 20) -> Dict:
        """
        Detect and analyze regular and hidden divergences between price and indicators.
        
        Args:
            candles: List of candle data
            lookback_period: Period to look back for divergence patterns
            
        Returns:
            Dict containing divergence analysis results
        """
        try:
            if len(candles) < lookback_period:
                return {'divergences': [], 'error': 'Insufficient data'}
                
            # Extract price data
            prices = np.array([c['close'] for c in candles[-lookback_period:]])
            highs = np.array([c['high'] for c in candles[-lookback_period:]])
            lows = np.array([c['low'] for c in candles[-lookback_period:]])
            
            # Calculate indicators
            rsi = self.technical_analysis.compute_rsi(self.product_id, candles[-lookback_period:])
            macd, signal, histogram = self.technical_analysis.compute_macd(self.product_id, candles[-lookback_period:])
            
            # Ensure RSI is a numpy array
            if isinstance(rsi, (int, float)):
                rsi_values = np.array([rsi] * len(prices))
            elif isinstance(rsi, list):
                rsi_values = np.array(rsi)
            else:
                rsi_values = np.array(rsi)
                
            # Ensure MACD histogram is a numpy array
            if isinstance(histogram, (int, float)):
                macd_values = np.array([histogram] * len(prices))
            elif isinstance(histogram, list):
                macd_values = np.array(histogram)
            else:
                macd_values = np.array(histogram)
            
            divergences = []
            
            # Find price pivots
            price_highs = self._find_peaks(highs)
            price_lows = self._find_troughs(lows)
            
            # Find indicator pivots
            rsi_highs = self._find_peaks(rsi_values)
            rsi_lows = self._find_troughs(rsi_values)
            
            # Regular Bullish Divergence (Lower price lows but higher indicator lows)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if lows[price_lows[-1]] < lows[price_lows[-2]] and \
                   rsi_values[rsi_lows[-1]] > rsi_values[rsi_lows[-2]]:
                    divergences.append({
                        'type': 'Regular Bullish',
                        'indicator': 'RSI',
                        'strength': abs(rsi_values[rsi_lows[-1]] - rsi_values[rsi_lows[-2]]),
                        'price_points': [lows[price_lows[-2]], lows[price_lows[-1]]],
                        'indicator_points': [rsi_values[rsi_lows[-2]], rsi_values[rsi_lows[-1]]],
                        'confidence': min(1.0, abs(rsi_values[rsi_lows[-1]] - rsi_values[rsi_lows[-2]]) / 30)
                    })
            
            # Regular Bearish Divergence (Higher price highs but lower indicator highs)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                if highs[price_highs[-1]] > highs[price_highs[-2]] and \
                   rsi_values[rsi_highs[-1]] < rsi_values[rsi_highs[-2]]:
                    divergences.append({
                        'type': 'Regular Bearish',
                        'indicator': 'RSI',
                        'strength': abs(rsi_values[rsi_highs[-1]] - rsi_values[rsi_highs[-2]]),
                        'price_points': [highs[price_highs[-2]], highs[price_highs[-1]]],
                        'indicator_points': [rsi_values[rsi_highs[-2]], rsi_values[rsi_highs[-1]]],
                        'confidence': min(1.0, abs(rsi_values[rsi_highs[-1]] - rsi_values[rsi_highs[-2]]) / 30)
                    })
            
            # Hidden Bullish Divergence (Higher price lows but lower indicator lows)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if lows[price_lows[-1]] > lows[price_lows[-2]] and \
                   rsi_values[rsi_lows[-1]] < rsi_values[rsi_lows[-2]]:
                    divergences.append({
                        'type': 'Hidden Bullish',
                        'indicator': 'RSI',
                        'strength': abs(rsi_values[rsi_lows[-1]] - rsi_values[rsi_lows[-2]]),
                        'price_points': [lows[price_lows[-2]], lows[price_lows[-1]]],
                        'indicator_points': [rsi_values[rsi_lows[-2]], rsi_values[rsi_lows[-1]]],
                        'confidence': min(1.0, abs(rsi_values[rsi_lows[-1]] - rsi_values[rsi_lows[-2]]) / 30)
                    })
            
            # Hidden Bearish Divergence (Lower price highs but higher indicator highs)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                if highs[price_highs[-1]] < highs[price_highs[-2]] and \
                   rsi_values[rsi_highs[-1]] > rsi_values[rsi_highs[-2]]:
                    divergences.append({
                        'type': 'Hidden Bearish',
                        'indicator': 'RSI',
                        'strength': abs(rsi_values[rsi_highs[-1]] - rsi_values[rsi_highs[-2]]),
                        'price_points': [highs[price_highs[-2]], highs[price_highs[-1]]],
                        'indicator_points': [rsi_values[rsi_highs[-2]], rsi_values[rsi_highs[-1]]],
                        'confidence': min(1.0, abs(rsi_values[rsi_highs[-1]] - rsi_values[rsi_highs[-2]]) / 30)
                    })
            
            # MACD Divergences
            macd_highs = self._find_peaks(macd_values)
            macd_lows = self._find_troughs(macd_values)
            
            # Regular MACD Divergences
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                if highs[price_highs[-1]] > highs[price_highs[-2]] and \
                   macd_values[macd_highs[-1]] < macd_values[macd_highs[-2]]:
                    divergences.append({
                        'type': 'Regular Bearish',
                        'indicator': 'MACD',
                        'strength': abs(macd_values[macd_highs[-1]] - macd_values[macd_highs[-2]]),
                        'price_points': [highs[price_highs[-2]], highs[price_highs[-1]]],
                        'indicator_points': [macd_values[macd_highs[-2]], macd_values[macd_highs[-1]]],
                        'confidence': min(1.0, abs(macd_values[macd_highs[-1]] - macd_values[macd_highs[-2]]) / 0.1)
                    })
            
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                if lows[price_lows[-1]] < lows[price_lows[-2]] and \
                   macd_values[macd_lows[-1]] > macd_values[macd_lows[-2]]:
                    divergences.append({
                        'type': 'Regular Bullish',
                        'indicator': 'MACD',
                        'strength': abs(macd_values[macd_lows[-1]] - macd_values[macd_lows[-2]]),
                        'price_points': [lows[price_lows[-2]], lows[price_lows[-1]]],
                        'indicator_points': [macd_values[macd_lows[-2]], macd_values[macd_lows[-1]]],
                        'confidence': min(1.0, abs(macd_values[macd_lows[-1]] - macd_values[macd_lows[-2]]) / 0.1)
                    })
            
            # Analyze divergence significance
            for div in divergences:
                # Calculate price change percentage
                price_change = abs(div['price_points'][1] - div['price_points'][0]) / div['price_points'][0] * 100
                
                # Calculate time between points (assuming sequential indices)
                time_span = abs(price_highs[-1] - price_highs[-2] if 'Bearish' in div['type'] 
                              else price_lows[-1] - price_lows[-2])
                
                # Adjust confidence based on price change and time span
                div['confidence'] *= (1 + price_change / 100)  # More significant price changes increase confidence
                div['confidence'] *= max(0.5, min(1.0, 1 - (time_span / lookback_period)))  # Closer points are more significant
                
                # Add additional analysis
                div['analysis'] = {
                    'price_change_percent': price_change,
                    'time_span': time_span,
                    'significance': 'High' if div['confidence'] > 0.7 else 'Medium' if div['confidence'] > 0.4 else 'Low',
                    'suggested_action': 'Consider Short' if 'Bearish' in div['type'] else 'Consider Long',
                    'stop_loss': min(div['price_points']) if 'Bullish' in div['type'] else max(div['price_points'])
                }
            
            return {
                'divergences': divergences,
                'summary': {
                    'total_divergences': len(divergences),
                    'bullish_count': sum(1 for d in divergences if 'Bullish' in d['type']),
                    'bearish_count': sum(1 for d in divergences if 'Bearish' in d['type']),
                    'highest_confidence': max([d['confidence'] for d in divergences]) if divergences else 0,
                    'primary_bias': 'Bullish' if sum(1 for d in divergences if 'Bullish' in d['type']) > 
                                              sum(1 for d in divergences if 'Bearish' in d['type']) else 'Bearish'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting divergences: {str(e)}")
            return {'divergences': [], 'error': str(e)}

    def analyze_volatility_regime(self, candles: List[Dict]) -> Dict:
        """
        Analyze and classify the current volatility regime of the market.
        
        Args:
            candles: List of candle data
            
        Returns:
            Dict containing volatility regime analysis
        """
        try:
            if len(candles) < 50:  # Need sufficient data for analysis
                return {
                    'regime': 'Unknown',
                    'confidence': 0.0,
                    'metrics': {}
                }
            
            # Calculate returns and volatility metrics
            prices = np.array([c['close'] for c in candles])
            returns = np.diff(np.log(prices))
            volumes = np.array([c['volume'] for c in candles])
            
            # Calculate different volatility measures
            current_vol = np.std(returns[-20:]) * np.sqrt(252)  # 20-day volatility annualized
            long_vol = np.std(returns) * np.sqrt(252)  # Full period volatility
            
            # Calculate volatility of volatility (vol-of-vol)
            rolling_vol = []
            for i in range(20, len(returns)):
                period_vol = np.std(returns[i-20:i]) * np.sqrt(252)
                rolling_vol.append(period_vol)
            vol_of_vol = np.std(rolling_vol) if rolling_vol else 0
            
            # Calculate volatility ratios
            vol_ratio = current_vol / long_vol if long_vol != 0 else 1
            
            # Define regime thresholds
            low_vol_threshold = 0.15  # 15% annualized volatility
            high_vol_threshold = 0.40  # 40% annualized volatility
            vol_of_vol_threshold = 0.10  # 10% vol-of-vol threshold
            
            # Classify the regime
            regime = 'Normal Volatility'
            confidence = 0.7  # Base confidence
            
            # NEW: Detect market exhaustion
            # Calculate price momentum and volume trend
            price_momentum = returns[-5:].mean() if len(returns) >= 5 else 0
            volume_trend = (volumes[-5:].mean() - volumes[-10:-5].mean()) / volumes[-10:-5].mean() if len(volumes) >= 10 else 0
            
            # Calculate RSI for exhaustion detection
            rsi = self.technical_analysis.compute_rsi(self.product_id, candles)
            
            # Define exhaustion conditions
            price_exhaustion = abs(price_momentum) > np.std(returns) * 2  # Price movement 2 std devs from mean
            volume_exhaustion = abs(volume_trend) > 0.5  # 50% volume change
            rsi_exhaustion = rsi > 75 or rsi < 25  # Extreme RSI levels
            
            # Check for exhaustion pattern
            if (price_exhaustion and volume_exhaustion and rsi_exhaustion):
                regime = 'Exhaustion'
                confidence = min(1.0, (abs(price_momentum) / np.std(returns)) * 0.5 +
                               abs(volume_trend) * 0.3 +
                               (abs(rsi - 50) / 50) * 0.2)
            elif current_vol < low_vol_threshold:
                regime = 'Low Volatility'
                confidence = min(1.0, (low_vol_threshold - current_vol) / low_vol_threshold + 0.5)
            elif current_vol > high_vol_threshold:
                regime = 'High Volatility'
                confidence = min(1.0, (current_vol - high_vol_threshold) / high_vol_threshold + 0.5)
            
            # Check for volatility clustering
            if vol_of_vol > vol_of_vol_threshold and regime != 'Exhaustion':
                regime = 'Volatility Clustering'
                confidence = min(1.0, vol_of_vol / vol_of_vol_threshold)
            
            # Check for regime transition
            vol_trend = 'Increasing' if len(rolling_vol) >= 2 and rolling_vol[-1] > rolling_vol[-2] else 'Decreasing'
            
            # Calculate volatility percentile
            vol_percentile = sum(v < current_vol for v in rolling_vol) / len(rolling_vol) if rolling_vol else 0.5
            
            # NEW: Add exhaustion metrics
            exhaustion_metrics = {
                'price_momentum': price_momentum,
                'volume_trend': volume_trend,
                'rsi_level': rsi,
                'exhaustion_score': (abs(price_momentum) / np.std(returns) +
                                   abs(volume_trend) +
                                   abs(rsi - 50) / 50) / 3 if len(returns) > 0 else 0,
                'is_price_exhausted': price_exhaustion,
                'is_volume_exhausted': volume_exhaustion,
                'is_rsi_exhausted': rsi_exhaustion
            }
            
            # Prepare detailed metrics
            metrics = {
                'current_volatility': current_vol,
                'long_term_volatility': long_vol,
                'volatility_of_volatility': vol_of_vol,
                'volatility_ratio': vol_ratio,
                'volatility_percentile': vol_percentile,
                'volatility_trend': vol_trend,
                'exhaustion_metrics': exhaustion_metrics,  # NEW: Add exhaustion metrics
                'regime_thresholds': {
                    'low_volatility': low_vol_threshold,
                    'high_volatility': high_vol_threshold,
                    'vol_of_vol': vol_of_vol_threshold
                }
            }
            
            # Add trading implications
            implications = []
            if regime == 'Low Volatility':
                implications.extend([
                    'Consider mean reversion strategies',
                    'Reduce position sizes',
                    'Prepare for potential volatility expansion'
                ])
            elif regime == 'High Volatility':
                implications.extend([
                    'Focus on trend following strategies',
                    'Increase stop distances',
                    'Consider reducing leverage'
                ])
            elif regime == 'Volatility Clustering':
                implications.extend([
                    'Expect continued volatility swings',
                    'Use adaptive position sizing',
                    'Monitor for regime change signals'
                ])
            elif regime == 'Exhaustion':  # NEW: Add exhaustion implications
                implications.extend([
                    'Prepare for potential trend reversal',
                    'Tighten stop losses on existing positions',
                    'Look for confirmation of reversal patterns',
                    'Consider contrarian positions with strict risk management'
                ])
            
            return {
                'regime': regime,
                'confidence': confidence,
                'metrics': metrics,
                'implications': implications,
                'description': f"Market is in a {regime} regime with {confidence*100:.1f}% confidence. "
                             f"Current volatility is at the {vol_percentile*100:.1f}th percentile "
                             f"and is {vol_trend.lower()}."
                             + (" Showing signs of exhaustion." if regime == 'Exhaustion' else "")
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regime: {str(e)}")
            return {
                'regime': 'Unknown',
                'confidence': 0.0,
                'metrics': {},
                'implications': [],
                'description': 'Error analyzing volatility regime'
            }

    def analyze_liquidity_depth(self, candles: List[Dict]) -> Dict:
        """
        Analyze market liquidity depth and order book imbalances.
        
        Args:
            candles: List of candle data
            
        Returns:
            Dict containing liquidity analysis results
        """
        try:
            if not candles or len(candles) < 20:
                return {
                    'status': 'insufficient_data',
                    'liquidity_score': 0.0,
                    'metrics': {}
                }
            
            # Extract volume and price data
            volumes = np.array([c['volume'] for c in candles])
            prices = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            
            # Calculate volume-weighted average price (VWAP)
            typical_prices = (highs + lows + prices) / 3
            vwap = np.sum(typical_prices * volumes) / np.sum(volumes)
            
            # Calculate liquidity metrics
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            price_impact = np.abs(prices - vwap) / vwap
            
            # Calculate volume profile
            volume_profile = []
            price_levels = np.linspace(min(lows), max(highs), 20)
            for level in price_levels:
                mask = (typical_prices >= level * 0.995) & (typical_prices <= level * 1.005)
                volume_at_level = np.sum(volumes[mask])
                volume_profile.append({
                    'price_level': level,
                    'volume': volume_at_level,
                    'percentage': volume_at_level / np.sum(volumes) * 100 if np.sum(volumes) > 0 else 0
                })
            
            # Identify liquidity clusters
            clusters = []
            current_cluster = []
            for i, level in enumerate(volume_profile):
                if level['percentage'] > 5:  # Significant volume threshold
                    current_cluster.append(level)
                elif current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            if current_cluster:
                clusters.append(current_cluster)
            
            # Calculate liquidity score (0-100)
            volume_consistency = 1 - (volume_std / avg_volume)
            price_efficiency = 1 - np.mean(price_impact)
            cluster_strength = len(clusters) / len(volume_profile)
            
            liquidity_score = (
                volume_consistency * 0.4 +
                price_efficiency * 0.4 +
                cluster_strength * 0.2
            ) * 100
            
            # Identify potential liquidity gaps
            gaps = []
            for i in range(1, len(volume_profile)):
                if volume_profile[i]['percentage'] < 1 and volume_profile[i-1]['percentage'] > 5:
                    gaps.append({
                        'start_price': volume_profile[i-1]['price_level'],
                        'end_price': volume_profile[i]['price_level'],
                        'size': volume_profile[i-1]['percentage'] - volume_profile[i]['percentage']
                    })
            
            # Calculate market depth imbalance
            buy_volume = np.sum(volumes[prices > vwap])
            sell_volume = np.sum(volumes[prices < vwap])
            total_volume = buy_volume + sell_volume
            imbalance_ratio = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
            
            # NEW: Track historical market depth imbalances
            current_time = datetime.now(UTC)
            self.depth_imbalance.append({
                'timestamp': current_time,
                'imbalance_ratio': imbalance_ratio,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'vwap': vwap,
                'current_price': prices[-1]
            })
            
            # Keep depth imbalance history within limit
            if len(self.depth_imbalance) > 100:  # Keep last 100 records
                self.depth_imbalance.pop(0)
            
            # NEW: Calculate historical imbalance metrics
            if len(self.depth_imbalance) > 1:
                historical_imbalances = [d['imbalance_ratio'] for d in self.depth_imbalance]
                imbalance_mean = np.mean(historical_imbalances)
                imbalance_std = np.std(historical_imbalances)
                current_imbalance_zscore = (imbalance_ratio - imbalance_mean) / imbalance_std if imbalance_std > 0 else 0
                
                # Detect imbalance trends
                recent_imbalances = historical_imbalances[-5:]  # Last 5 periods
                imbalance_trend = 'Increasing' if np.all(np.diff(recent_imbalances) > 0) else \
                                'Decreasing' if np.all(np.diff(recent_imbalances) < 0) else \
                                'Fluctuating'
                
                # Calculate cumulative imbalance
                cumulative_imbalance = sum(historical_imbalances[-5:])  # Last 5 periods
                
                # Detect potential price pressure
                price_pressure = 'Strong Buy' if cumulative_imbalance > 2 else \
                               'Moderate Buy' if cumulative_imbalance > 1 else \
                               'Strong Sell' if cumulative_imbalance < -2 else \
                               'Moderate Sell' if cumulative_imbalance < -1 else \
                               'Neutral'
                
                historical_analysis = {
                    'mean_imbalance': imbalance_mean,
                    'imbalance_volatility': imbalance_std,
                    'current_zscore': current_imbalance_zscore,
                    'imbalance_trend': imbalance_trend,
                    'cumulative_imbalance': cumulative_imbalance,
                    'price_pressure': price_pressure,
                    'trend_strength': abs(cumulative_imbalance) / 5  # Normalized to 0-1
                }
            else:
                historical_analysis = {
                    'mean_imbalance': 0,
                    'imbalance_volatility': 0,
                    'current_zscore': 0,
                    'imbalance_trend': 'Insufficient Data',
                    'cumulative_imbalance': 0,
                    'price_pressure': 'Neutral',
                    'trend_strength': 0
                }
            
            return {
                'status': 'success',
                'liquidity_score': liquidity_score,
                'metrics': {
                    'vwap': vwap,
                    'average_volume': avg_volume,
                    'volume_consistency': volume_consistency * 100,
                    'price_efficiency': price_efficiency * 100,
                    'cluster_strength': cluster_strength * 100,
                    'imbalance_ratio': imbalance_ratio
                },
                'volume_profile': volume_profile,
                'liquidity_clusters': [
                    {
                        'price_range': (min(c['price_level'] for c in cluster),
                                      max(c['price_level'] for c in cluster)),
                        'total_volume': sum(c['volume'] for c in cluster),
                        'strength': len(cluster) / len(volume_profile)
                    }
                    for cluster in clusters
                ],
                'liquidity_gaps': gaps,
                'market_depth': {
                    'buy_side_volume': buy_volume,
                    'sell_side_volume': sell_volume,
                    'imbalance_ratio': imbalance_ratio,
                    'interpretation': 'Buy Heavy' if imbalance_ratio > 0.2 else
                                    'Sell Heavy' if imbalance_ratio < -0.2 else 'Balanced'
                },
                'historical_depth_analysis': historical_analysis,  # NEW: Added historical analysis
                'depth_imbalance_history': self.depth_imbalance[-5:]  # NEW: Last 5 records
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity depth: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'liquidity_score': 0.0,
                'metrics': {}
            }

    @performance_monitor
    def detect_manipulation_patterns(self, candles: List[Dict]) -> Dict:
        """
        Detects potential market manipulation patterns by analyzing volume and price anomalies.
        
        Args:
            candles: List of candlestick data
            
        Returns:
            Dict containing detected patterns and their confidence levels
        """
        try:
            df = pd.DataFrame(self._format_candles(candles))
            patterns = {
                'pump_and_dump': False,
                'wash_trading': False,
                'spoofing': False,
                'confidence': 0.0,
                'details': [],
                'market_condition': 'Normal'  # Default market condition
            }
            
            # Detect sudden volume spikes with price pumps
            volume_mean = df['volume'].mean()
            price_std = df['close'].pct_change().std()
            
            for i in range(1, len(df)):
                # Check for pump and dump patterns
                if (df['volume'].iloc[i] > volume_mean * 3 and  # 3x normal volume
                    df['close'].iloc[i] > df['close'].iloc[i-1] * 1.1):  # 10% price jump
                    patterns['pump_and_dump'] = True
                    patterns['market_condition'] = 'Suspicious'
                    patterns['details'].append({
                        'timestamp': df.index[i],
                        'pattern': 'Pump and Dump',
                        'volume_ratio': df['volume'].iloc[i] / volume_mean
                    })
                
                # Check for potential wash trading
                if (df['volume'].iloc[i] > volume_mean * 2 and  # High volume
                    abs(df['close'].iloc[i] - df['close'].iloc[i-1]) < price_std * 0.5):  # Small price change
                    patterns['wash_trading'] = True
                    patterns['market_condition'] = 'Suspicious'
                    patterns['details'].append({
                        'timestamp': df.index[i],
                        'pattern': 'Wash Trading',
                        'price_change': abs(df['close'].iloc[i] - df['close'].iloc[i-1])
                    })
            
            # Calculate confidence based on number and strength of patterns
            pattern_count = sum([patterns['pump_and_dump'], patterns['wash_trading'], patterns['spoofing']])
            patterns['confidence'] = min(0.95, pattern_count * 0.35)
            
            # Update market condition based on overall analysis
            if pattern_count > 1:
                patterns['market_condition'] = 'Highly Suspicious'
            elif pattern_count == 1:
                patterns['market_condition'] = 'Suspicious'
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in detect_manipulation_patterns: {str(e)}")
            return self._generate_error_response()

    def _is_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect head and shoulders pattern."""
        try:
            peaks = self._find_peaks(highs)
            if len(peaks) < 3:
                return False
                
            # Get last three peaks
            last_peaks = peaks[-3:]
            peak_prices = highs[last_peaks]
            
            # Check if middle peak (head) is higher than shoulders
            if peak_prices[1] > peak_prices[0] and peak_prices[1] > peak_prices[2]:
                # Check if shoulders are at similar levels
                shoulder_diff = abs(peak_prices[0] - peak_prices[2]) / peak_prices[0]
                if shoulder_diff < threshold:
                    # Check if neckline is relatively flat
                    troughs = self._find_troughs(lows)
                    if len(troughs) >= 2:
                        neckline_diff = abs(lows[troughs[-1]] - lows[troughs[-2]]) / lows[troughs[-2]]
                        return neckline_diff < threshold
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {str(e)}")
            return False

    def _is_inverse_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, threshold: float = 0.02) -> bool:
        """Detect inverse head and shoulders pattern."""
        try:
            troughs = self._find_troughs(lows)
            if len(troughs) < 3:
                return False
                
            # Get last three troughs
            last_troughs = troughs[-3:]
            trough_prices = lows[last_troughs]
            
            # Check if middle trough (head) is lower than shoulders
            if trough_prices[1] < trough_prices[0] and trough_prices[1] < trough_prices[2]:
                # Check if shoulders are at similar levels
                shoulder_diff = abs(trough_prices[0] - trough_prices[2]) / trough_prices[0]
                if shoulder_diff < threshold:
                    # Check if neckline is relatively flat
                    peaks = self._find_peaks(highs)
                    if len(peaks) >= 2:
                        neckline_diff = abs(highs[peaks[-1]] - highs[peaks[-2]]) / highs[peaks[-2]]
                        return neckline_diff < threshold
                        
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting inverse head and shoulders: {str(e)}")
            return False

    def _is_triangle(self, highs: np.ndarray, lows: np.ndarray, min_points: int = 5) -> bool:
        """Detect triangle patterns (ascending, descending, or symmetrical)."""
        try:
            peaks = self._find_peaks(highs[-20:])
            troughs = self._find_troughs(lows[-20:])
            
            if len(peaks) < min_points or len(troughs) < min_points:
                return False
                
            # Calculate slopes of upper and lower trend lines
            peak_slope = np.polyfit(peaks, highs[peaks], 1)[0]
            trough_slope = np.polyfit(troughs, lows[troughs], 1)[0]
            
            # Check for convergence
            return abs(peak_slope - trough_slope) < 0.1
            
        except Exception as e:
            self.logger.error(f"Error detecting triangle: {str(e)}")
            return False

    def _is_channel(self, highs: np.ndarray, lows: np.ndarray, min_points: int = 5) -> bool:
        """Detect price channel patterns."""
        try:
            peaks = self._find_peaks(highs[-20:])
            troughs = self._find_troughs(lows[-20:])
            
            if len(peaks) < min_points or len(troughs) < min_points:
                return False
                
            # Calculate slopes of upper and lower trend lines
            peak_slope = np.polyfit(peaks, highs[peaks], 1)[0]
            trough_slope = np.polyfit(troughs, lows[troughs], 1)[0]
            
            # Check for parallel lines
            return abs(peak_slope - trough_slope) < 0.01
            
        except Exception as e:
            self.logger.error(f"Error detecting channel: {str(e)}")
            return False

    def _is_flag_or_pennant(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect flag or pennant patterns."""
        try:
            # Check for strong trend before the pattern
            trend_start = prices[-20:-10]
            trend_strength = abs(trend_start[-1] - trend_start[0]) / trend_start[0]
            
            if trend_strength < 0.05:  # Require 5% move
                return False
                
            # Check for consolidation with lower volume
            consolidation = prices[-10:]
            consolidation_volume = volumes[-10:]
            trend_volume = volumes[-20:-10]
            
            volume_decline = np.mean(consolidation_volume) < np.mean(trend_volume)
            price_range = (max(consolidation) - min(consolidation)) / np.mean(consolidation)
            
            return volume_decline and price_range < 0.03  # 3% range
            
        except Exception as e:
            self.logger.error(f"Error detecting flag/pennant: {str(e)}")
            return False

    def _is_cup_and_handle(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect cup and handle pattern."""
        try:
            if len(prices) < 30:
                return False
                
            # Check for U-shaped cup
            cup_prices = prices[-30:-10]
            cup_fit = np.polyfit(range(len(cup_prices)), cup_prices, 2)
            
            # Parabola should open upward
            if cup_fit[0] > 0:
                # Check for handle (small downward drift)
                handle_prices = prices[-10:]
                handle_slope = np.polyfit(range(len(handle_prices)), handle_prices, 1)[0]
                
                return handle_slope < 0 and abs(handle_slope) < abs(cup_fit[0])
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting cup and handle: {str(e)}")
            return False

    def _is_rounding_pattern(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Detect rounding bottom or top patterns."""
        try:
            if len(prices) < 20:
                return False
                
            # Fit a quadratic function to the prices
            x = np.arange(len(prices[-20:]))
            fit = np.polyfit(x, prices[-20:], 2)
            
            # Check if the curve is significant
            return abs(fit[0]) > 0.0001
            
        except Exception as e:
            self.logger.error(f"Error detecting rounding pattern: {str(e)}")
            return False

    def _calculate_pattern_completion(self, pattern_type: PatternType, prices: np.ndarray, 
                                    highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate the completion percentage of a pattern."""
        try:
            if pattern_type in [PatternType.HEAD_SHOULDERS, PatternType.INV_HEAD_SHOULDERS]:
                # Check if neckline is broken
                neckline = np.mean(lows[-5:])
                return min(100, max(0, abs(prices[-1] - neckline) / (max(highs) - min(lows)) * 100))
                
            elif pattern_type in [PatternType.TRIANGLE, PatternType.FLAG, PatternType.PENNANT]:
                # Calculate distance to apex
                apex_distance = len(prices) - np.argmax(highs)
                return min(100, max(0, (1 - apex_distance / len(prices)) * 100))
                
            elif pattern_type == PatternType.CUP_HANDLE:
                # Calculate handle completion
                handle_range = max(prices[-10:]) - min(prices[-10:])
                cup_range = max(prices[-30:]) - min(prices[-30:])
                return min(100, max(0, (1 - handle_range / cup_range) * 100))
                
            else:
                return 50.0  # Default completion percentage
                
        except Exception as e:
            self.logger.error(f"Error calculating pattern completion: {str(e)}")
            return 0.0

    def visualize_order_book(self) -> Dict:
        """
        Create a detailed visualization and analysis of the order book.
        Returns a dictionary containing order book metrics and visualization data.
        """
        try:
            if not self._current_candles or len(self._current_candles) < 20:
                return {
                    'status': 'insufficient_data',
                    'visualization': {},
                    'metrics': {}
                }
            
            # Get current price and volume data
            current_price = float(self._current_candles[-1]['close'])
            volumes = np.array([c['volume'] for c in self._current_candles])
            highs = np.array([c['high'] for c in self._current_candles])
            lows = np.array([c['low'] for c in self._current_candles])
            
            # Calculate actual spread from recent high/low data
            recent_spread = np.mean([h - l for h, l in zip(highs[-5:], lows[-5:])])
            
            # Calculate price levels with a more realistic spread
            price_range = current_price * 0.01  # 1% range
            spread_adjustment = max(recent_spread, current_price * 0.0001)  # At least 0.01% spread
            
            # Adjust bid and ask levels to account for spread
            bid_levels = np.linspace(current_price - price_range, current_price - spread_adjustment/2, 20)
            ask_levels = np.linspace(current_price + spread_adjustment/2, current_price + price_range, 20)
            
            # Calculate cumulative volume at each level
            bid_volumes = []
            ask_volumes = []
            
            for level in bid_levels:
                mask = (np.array([c['low'] for c in self._current_candles]) >= level * 0.9995) & \
                       (np.array([c['low'] for c in self._current_candles]) <= level * 1.0005)
                bid_volumes.append(np.sum(volumes[mask]))
            
            for level in ask_levels:
                mask = (np.array([c['high'] for c in self._current_candles]) >= level * 0.9995) & \
                       (np.array([c['high'] for c in self._current_candles]) <= level * 1.0005)
                ask_volumes.append(np.sum(volumes[mask]))
            
            # Rest of the code remains the same...
            # Calculate cumulative volumes
            cum_bid_volumes = np.cumsum(bid_volumes)
            cum_ask_volumes = np.cumsum(ask_volumes)
            
            # Calculate volume thresholds more dynamically
            bid_volume_mean = np.mean(bid_volumes)
            ask_volume_mean = np.mean(ask_volumes)
            bid_volume_std = np.std(bid_volumes)
            ask_volume_std = np.std(ask_volumes)
            
            # Lower threshold for significant levels (now 1.2 standard deviations above mean)
            bid_threshold = bid_volume_mean + (1.2 * bid_volume_std)
            ask_threshold = ask_volume_mean + (1.2 * ask_volume_std)
            
            # Identify significant levels with more sensitivity
            significant_bids = []
            significant_asks = []
            
            # Find local maxima in bid volumes
            for i in range(1, len(bid_volumes)-1):
                if bid_volumes[i] > bid_volumes[i-1] and bid_volumes[i] > bid_volumes[i+1] and bid_volumes[i] > bid_threshold:
                    significant_bids.append({
                        'price': bid_levels[i],
                        'volume': bid_volumes[i],
                        'cumulative_volume': cum_bid_volumes[i],
                        'strength': (bid_volumes[i] - bid_volume_mean) / bid_volume_std
                    })
            
            # Find local maxima in ask volumes
            for i in range(1, len(ask_volumes)-1):
                if ask_volumes[i] > ask_volumes[i-1] and ask_volumes[i] > ask_volumes[i+1] and ask_volumes[i] > ask_threshold:
                    significant_asks.append({
                        'price': ask_levels[i],
                        'volume': ask_volumes[i],
                        'cumulative_volume': cum_ask_volumes[i],
                        'strength': (ask_volumes[i] - ask_volume_mean) / ask_volume_std
                    })
            
            # Ensure at least some levels are identified
            if not significant_bids:
                max_bid_idx = np.argmax(bid_volumes)
                significant_bids.append({
                    'price': bid_levels[max_bid_idx],
                    'volume': bid_volumes[max_bid_idx],
                    'cumulative_volume': cum_bid_volumes[max_bid_idx],
                    'strength': (bid_volumes[max_bid_idx] - bid_volume_mean) / bid_volume_std
                })
            
            if not significant_asks:
                max_ask_idx = np.argmax(ask_volumes)
                significant_asks.append({
                    'price': ask_levels[max_ask_idx],
                    'volume': ask_volumes[max_ask_idx],
                    'cumulative_volume': cum_ask_volumes[max_ask_idx],
                    'strength': (ask_volumes[max_ask_idx] - ask_volume_mean) / ask_volume_std
                })
            
            # Calculate order book metrics
            bid_ask_ratio = np.sum(bid_volumes) / np.sum(ask_volumes) if np.sum(ask_volumes) > 0 else 0
            spread = min(ask_levels) - max(bid_levels)  # Fixed spread calculation
            depth_score = (np.sum(cum_bid_volumes) + np.sum(cum_ask_volumes)) / (len(bid_levels) + len(ask_levels))
            
            # Calculate market pressure
            buy_pressure = np.sum(bid_volumes[-5:])
            sell_pressure = np.sum(ask_volumes[:5])
            pressure_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else float('inf')
            
            # Sort significant levels by strength
            significant_bids.sort(key=lambda x: x['strength'], reverse=True)
            significant_asks.sort(key=lambda x: x['strength'], reverse=True)
            
            # Calculate order book imbalance score (-1 to 1)
            imbalance_score = (bid_ask_ratio - 1) / (bid_ask_ratio + 1)
            
            return {
                'status': 'success',
                'current_price': current_price,
                'visualization': {
                    'bid_levels': bid_levels.tolist(),
                    'ask_levels': ask_levels.tolist(),
                    'bid_volumes': bid_volumes,
                    'ask_volumes': ask_volumes,
                    'cumulative_bid_volumes': cum_bid_volumes.tolist(),
                    'cumulative_ask_volumes': cum_ask_volumes.tolist()
                },
                'metrics': {
                    'bid_ask_ratio': bid_ask_ratio,
                    'spread': spread,
                    'depth_score': depth_score,
                    'buy_pressure': buy_pressure,
                    'sell_pressure': sell_pressure,
                    'pressure_ratio': pressure_ratio,
                    'imbalance_score': imbalance_score
                },
                'significant_levels': {
                    'support': significant_bids,
                    'resistance': significant_asks
                },
                'analysis': {
                    'market_pressure': 'Buy' if pressure_ratio > 1.2 else 'Sell' if pressure_ratio < 0.8 else 'Neutral',
                    'depth_quality': 'High' if depth_score > np.mean(volumes) else 'Low',
                    'spread_quality': 'Tight' if spread < current_price * 0.001 else 'Wide',
                    'imbalance_interpretation': 'Heavy Buy' if imbalance_score > 0.3 else
                                             'Heavy Sell' if imbalance_score < -0.3 else
                                             'Moderate Buy' if imbalance_score > 0.1 else
                                             'Moderate Sell' if imbalance_score < -0.1 else
                                             'Balanced'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error visualizing order book: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'visualization': {},
                'metrics': {}
            }

class AlertType(Enum):
    PRICE = "Price Alert"
    TECHNICAL = "Technical Alert"
    PATTERN = "Pattern Alert"
    VOLATILITY = "Volatility Alert"
    LIQUIDITY = "Liquidity Alert"
    DIVERGENCE = "Divergence Alert"
    MOMENTUM = "Momentum Alert"
    VOLUME = "Volume Alert"
    MANIPULATION = "Manipulation Alert"

class AlertPriority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class Alert:
    """Represents a single market alert."""
    
    def __init__(self, alert_type: AlertType, message: str, priority: AlertPriority,
                 timestamp: datetime = None):
        self.alert_type = alert_type
        self.message = message
        self.priority = priority
        self.timestamp = timestamp or datetime.now(UTC)
        self.is_triggered = False
        self.trigger_price = None
        self.trigger_condition = None

class AlertSystem:
    """System for managing and triggering market alerts."""
    
    def __init__(self, logger=None, market_analyzer=None):
        self.logger = logger or logging.getLogger(__name__)
        self.active_alerts = []
        self.alert_history = []
        self.market_analyzer = market_analyzer
        self.max_history = 100
        
        # Alert thresholds
        self.price_change_threshold = 0.02  # 2% price change
        self.volume_spike_threshold = 3.0    # 3x average volume
        self.volatility_threshold = 0.03     # 3% volatility
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
    def add_price_alert(self, price_level: float, direction: str, priority: AlertPriority = AlertPriority.MEDIUM) -> Alert:
        """Add a price level alert."""
        message = f"Price {'above' if direction == 'above' else 'below'} {price_level}"
        alert = Alert(AlertType.PRICE, message, priority)
        alert.trigger_price = price_level
        alert.trigger_condition = direction
        self.active_alerts.append(alert)
        return alert
        
    def add_technical_alert(self, indicator: str, condition: str, value: float,
                           priority: AlertPriority = AlertPriority.MEDIUM) -> Alert:
        """Add a technical indicator alert."""
        message = f"{indicator} {condition} {value}"
        alert = Alert(AlertType.TECHNICAL, message, priority)
        alert.trigger_condition = {'indicator': indicator, 'condition': condition, 'value': value}
        self.active_alerts.append(alert)
        return alert
        
    def add_pattern_alert(self, pattern_type: PatternType, priority: AlertPriority = AlertPriority.HIGH) -> Alert:
        """Add a pattern formation alert."""
        message = f"Pattern detected: {pattern_type.value}"
        alert = Alert(AlertType.PATTERN, message, priority)
        alert.trigger_condition = pattern_type
        self.active_alerts.append(alert)
        return alert
        
    def add_volatility_alert(self, threshold: float, priority: AlertPriority = AlertPriority.HIGH) -> Alert:
        """Add a volatility spike alert."""
        message = f"Volatility above {threshold*100}%"
        alert = Alert(AlertType.VOLATILITY, message, priority)
        alert.trigger_condition = threshold
        self.active_alerts.append(alert)
        return alert
        
    def check_alerts(self, market_data: Dict) -> List[Alert]:
        """
        Check all active alerts against current market data.
        Returns list of triggered alerts.
        """
        triggered_alerts = []
        current_price = market_data.get('current_price', 0)
        
        for alert in self.active_alerts:
            if alert.is_triggered:
                continue
                
            if alert.alert_type == AlertType.PRICE and alert.trigger_price:
                if (alert.trigger_condition == 'above' and current_price > alert.trigger_price) or \
                   (alert.trigger_condition == 'below' and current_price < alert.trigger_price):
                    alert.is_triggered = True
                    triggered_alerts.append(alert)
                    
            elif alert.alert_type == AlertType.TECHNICAL:
                condition = alert.trigger_condition
                indicator_value = market_data.get('indicators', {}).get(condition['indicator'])
                if indicator_value is not None:
                    if self._check_condition(indicator_value, condition['condition'], condition['value']):
                        alert.is_triggered = True
                        triggered_alerts.append(alert)
                        
            elif alert.alert_type == AlertType.PATTERN:
                if market_data.get('patterns', {}).get('type') == alert.trigger_condition:
                    alert.is_triggered = True
                    triggered_alerts.append(alert)
                    
            elif alert.alert_type == AlertType.VOLATILITY:
                volatility = market_data.get('volatility_regime', {}).get('metrics', {}).get('current_volatility', 0)
                if volatility > alert.trigger_condition:
                    alert.is_triggered = True
                    triggered_alerts.append(alert)
        
        # Update alert history
        for alert in triggered_alerts:
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            self.active_alerts.remove(alert)
            
        return triggered_alerts
        
    def check_market_conditions(self, market_data: Dict) -> List[Alert]:
        """
        Analyze market conditions and generate automatic alerts.
        """
        auto_alerts = []
        
        # Price change alerts
        if 'current_price' in market_data and 'previous_price' in market_data:
            price_change = (market_data['current_price'] - market_data['previous_price']) / market_data['previous_price']
            if abs(price_change) > self.price_change_threshold:
                message = f"Significant price {'increase' if price_change > 0 else 'decrease'} of {abs(price_change)*100:.1f}%"
                alert = Alert(AlertType.PRICE, message, AlertPriority.HIGH)
                auto_alerts.append(alert)
        
        # Volume spike alerts
        volume_info = market_data.get('volume_analysis', {})
        if volume_info.get('volume_change', 0) > self.volume_spike_threshold:
            message = f"Volume spike detected: {volume_info['volume_change']:.1f}x average"
            alert = Alert(AlertType.VOLUME, message, AlertPriority.HIGH)
            auto_alerts.append(alert)
        
        # Technical indicator alerts
        indicators = market_data.get('indicators', {})
        
        # RSI alerts
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi > self.rsi_overbought:
                alert = Alert(AlertType.TECHNICAL, f"RSI Overbought: {rsi:.1f}", AlertPriority.HIGH)
                auto_alerts.append(alert)
            elif rsi < self.rsi_oversold:
                alert = Alert(AlertType.TECHNICAL, f"RSI Oversold: {rsi:.1f}", AlertPriority.HIGH)
                auto_alerts.append(alert)
        
        # MACD alerts
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd is not None and macd_signal is not None:
            if macd > macd_signal and macd_signal < 0:  # Bullish crossover
                alert = Alert(AlertType.TECHNICAL, "MACD Bullish Crossover", AlertPriority.MEDIUM)
                auto_alerts.append(alert)
            elif macd < macd_signal and macd_signal > 0:  # Bearish crossover
                alert = Alert(AlertType.TECHNICAL, "MACD Bearish Crossover", AlertPriority.MEDIUM)
                auto_alerts.append(alert)
        
        # Pattern alerts
        patterns = market_data.get('patterns', {})
        if patterns.get('type') != PatternType.NONE.value:
            alert = Alert(AlertType.PATTERN, f"Pattern detected: {patterns['type']}", AlertPriority.HIGH)
            auto_alerts.append(alert)
        
        # Divergence alerts
        divergences = market_data.get('divergence_analysis', {}).get('divergences', [])
        for div in divergences:
            if div.get('confidence', 0) > 0.7:  # High confidence divergences only
                alert = Alert(AlertType.DIVERGENCE, 
                            f"High confidence {div['type']} divergence detected",
                            AlertPriority.HIGH)
                auto_alerts.append(alert)
        
        # Momentum alerts
        momentum = market_data.get('momentum_analysis', {})
        if momentum.get('total_score', 0) > 70:
            alert = Alert(AlertType.MOMENTUM, "Strong bullish momentum", AlertPriority.MEDIUM)
            auto_alerts.append(alert)
        elif momentum.get('total_score', 0) < -70:
            alert = Alert(AlertType.MOMENTUM, "Strong bearish momentum", AlertPriority.MEDIUM)
            auto_alerts.append(alert)
        
        # Liquidity alerts
        liquidity = market_data.get('liquidity_analysis', {})
        if liquidity.get('liquidity_score', 100) < 30:
            alert = Alert(AlertType.LIQUIDITY, "Low market liquidity detected", AlertPriority.HIGH)
            auto_alerts.append(alert)
        
        # Get manipulation patterns
        manipulation_patterns = self.market_analyzer.detect_manipulation_patterns(market_data.get('candles', []))
        
        if manipulation_patterns.get('pump_and_dump'):
            auto_alerts.append(Alert(
                AlertType.MANIPULATION,
                "Potential pump and dump pattern detected",
                AlertPriority.CRITICAL,
            ))
        
        if manipulation_patterns.get('wash_trading'):
            auto_alerts.append(Alert(
                AlertType.MANIPULATION,
                "Potential wash trading activity detected",
                AlertPriority.HIGH,
            ))
        
        return auto_alerts
        
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Helper method to check alert conditions."""
        if condition == 'above':
            return value > threshold
        elif condition == 'below':
            return value < threshold
        elif condition == 'equals':
            return abs(value - threshold) < 0.0001
        return False
        
    def get_alert_summary(self) -> Dict:
        """Get summary of current alerts and history."""
        return {
            'active_alerts': len(self.active_alerts),
            'triggered_alerts': len(self.alert_history),
            'alerts_by_priority': {
                priority.value: len([a for a in self.active_alerts if a.priority == priority])
                for priority in AlertPriority
            },
            'alerts_by_type': {
                alert_type.value: len([a for a in self.active_alerts if a.alert_type == alert_type])
                for alert_type in AlertType
            },
            'recent_alerts': [
                {
                    'type': alert.alert_type.value,
                    'message': alert.message,
                    'priority': alert.priority.value,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Market Analyzer')
    
    parser.add_argument(
        '--product_id',
        type=str,
        choices=VALID_PRODUCTS,
        default='DOGE-USDC',
        help='Product ID to analyze (e.g., BTC-USDC, ETH-USDC)'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=VALID_GRANULARITIES,
        default='ONE_HOUR',
        help='Candle interval granularity'
    )
    
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Run in continuous monitoring mode'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Monitoring interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--list-products',
        action='store_true',
        help='List all available products'
    )
    
    parser.add_argument(
        '--list-granularities',
        action='store_true',
        help='List all available granularities'
    )

    parser.add_argument(
        '--simple',
        action='store_true',
        help='Print only position confidence and success probability'
    )
    
    return parser.parse_args()

def list_options():
    """Print available options for products and granularities."""
    print("\nAvailable Products:")
    for product in VALID_PRODUCTS:
        print(f"  - {product}")
    
    print("\nAvailable Granularities:")
    for granularity in VALID_GRANULARITIES:
        print(f"  - {granularity}")
    print()

def run_continuous_monitoring(analyzer: MarketAnalyzer, interval: int):
    """Run the analyzer in continuous monitoring mode."""
    print(f"\n🔄 Starting continuous market monitoring (interval: {interval}s)")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_alert_time = datetime.now(UTC)
    
    try:
        while True:
            current_time = datetime.now(UTC)
            
            # Get market analysis
            analysis = analyzer.get_market_signal()
            
            # Check for new alerts
            alerts = analysis.get('alerts', {})
            new_alerts = []
            
            # Collect new triggered alerts
            if alerts.get('triggered_alerts'):
                new_alerts.extend([
                    alert for alert in alerts['triggered_alerts']
                    if datetime.fromisoformat(alert['timestamp']) > last_alert_time
                ])
            
            # Collect new automatic alerts
            if alerts.get('auto_alerts'):
                new_alerts.extend([
                    alert for alert in alerts['auto_alerts']
                    if datetime.fromisoformat(alert['timestamp']) > last_alert_time
                ])
            
            # Display new alerts
            if new_alerts:
                print(f"\n====== 🔔 New Alerts at {current_time.strftime('%Y-%m-%d %H:%M:%S')} 🔔 ======")
                for alert in new_alerts:
                    priority_emoji = '🔴' if alert['priority'] == 'Critical' else \
                                   '🟡' if alert['priority'] == 'High' else \
                                   '🟢' if alert['priority'] == 'Medium' else '⚪'
                    print(f"{priority_emoji} [{alert['priority']}] {alert['type']}: {alert['message']}")
                print("=" * 60)
                
                last_alert_time = current_time
            
            # Sleep for the specified interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n⛔ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error in continuous monitoring: {str(e)}")
        logging.error(f"Error in continuous monitoring: {str(e)}", exc_info=True)

def main():
    # Configure logging with more detail
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    args = parse_arguments()
    
    if args.list_products or args.list_granularities:
        list_options()
        return

    analyzer = MarketAnalyzer(
        product_id=args.product_id,
        candle_interval=args.granularity
    )
    
    if args.continuous:
        run_continuous_monitoring(analyzer, args.interval)
    else:
        try:
            # Single analysis run
            analysis = analyzer.get_market_signal()
            
            if args.simple:
                # Print only confidence and success probability
                print(f"\n====== 📊 Simple Market Analysis 📊 ======")
                print(f"Position: {analysis['position']}")
                print(f"Signal Confidence: {analysis['confidence']*100:.1f}%")
                print(f"Success Probability: {analysis['probability_analysis']['total_probability']:.1f}%")
                print(f"Move Quality: {analysis['probability_analysis']['move_quality']['strength_rating']}")
                print(f"Current Price: ${analysis['current_price']:.4f}")

                # Add trade suggestion
                if analysis['position'] == 'LONG':
                    entry_price = analysis['current_price']
                    stop_loss = analysis['patterns'].get('stop_loss', entry_price * 0.98)  # Default to 2% stop loss
                    target = analysis['patterns'].get('target', entry_price * 1.03)  # Default to 3% target
                    print("\n📈 Trade Suggestion:")
                    print(f"BUY at ${entry_price:.4f}")
                    print(f"Target: ${target:.4f} (+{((target-entry_price)/entry_price)*100:.1f}%)")
                    print(f"Stop Loss: ${stop_loss:.4f} (-{((entry_price-stop_loss)/entry_price)*100:.1f}%)")
                    print(f"Risk/Reward: {abs((target-entry_price)/(stop_loss-entry_price)):.2f}")
                elif analysis['position'] == 'SHORT':
                    entry_price = analysis['current_price']
                    stop_loss = analysis['patterns'].get('stop_loss', entry_price * 1.02)  # Default to 2% stop loss
                    target = analysis['patterns'].get('target', entry_price * 0.97)  # Default to 3% target
                    print("\n📉 Trade Suggestion:")
                    print(f"SELL at ${entry_price:.4f}")
                    print(f"Target: ${target:.4f} (-{((entry_price-target)/entry_price)*100:.1f}%)")
                    print(f"Stop Loss: ${stop_loss:.4f} (+{((stop_loss-entry_price)/entry_price)*100:.1f}%)")
                    print(f"Risk/Reward: {abs((entry_price-target)/(stop_loss-entry_price)):.2f}")
                else:
                    print("\n⚖️ Trade Suggestion: HOLD - No clear setup")
            else:
                # Get order book visualization
                order_book = analyzer.visualize_order_book()
                
                # Enhanced formatted output
                print("\n====== 📊 Comprehensive Market Analysis Report 📊 ======")
                print(f"🕒 Timestamp: {analysis['timestamp']}")
                print(f"💱 Product: {analysis['product_id']}")
                print(f"💵 Current Price: ${analysis['current_price']:,.4f}")
                
                # Market Overview Section
                print("\n=== 🎯 Market Overview ===")
                print(f"📍 Signal: {analysis['signal']}")
                print(f"📌 Position: {analysis['position']}")
                print(f"🎯 Confidence: {analysis['confidence']*100:.1f}%")
                print(f"📈 Market Condition: {analysis['market_condition']}")
                print(f"🔒 Signal Stability: {analysis['signal_stability']}")
                
                # Technical Indicators Section
                print("\n=== 📐 Technical Analysis Overview ===")
                indicators = analysis['indicators']
                print("Core Indicators:")
                print(f"📊 RSI: {indicators['rsi']:.2f} ({'Overbought' if indicators['rsi'] > 70 else 'Oversold' if indicators['rsi'] < 30 else 'Neutral'})")
                print(f"📈 MACD: {indicators['macd']:.4f}, Signal: {indicators['macd_signal']:.4f}, Histogram: {indicators['macd_histogram']:.4f}")
                adx_trend = 'Strong Trend' if indicators['adx'] > 25 else 'Weak Trend'
                print(f"📏 ADX: {indicators['adx']:.2f} ({adx_trend})")
                print(f"➡️ Trend Direction: {indicators['trend_direction']}")
                
                print("\nPrice Channels:")
                print(f"📊 Bollinger Bands:")
                print(f"  • Upper: ${indicators['bollinger_upper']:.4f}")
                print(f"  • Middle: ${indicators['bollinger_middle']:.4f}")
                print(f"  • Lower: ${indicators['bollinger_lower']:.4f}")
                print(f"  • Band Width: ${(indicators['bollinger_upper'] - indicators['bollinger_lower']):.4f}")
                print(f"  • Position: {'Above' if analysis['current_price'] > indicators['bollinger_middle'] else 'Below'} Middle Band")
                
                # Volume Analysis Section
                print("\n=== 📊 Volume Analysis ===")
                volume = analysis['volume_analysis']
                print(f"📈 Volume Change: {volume['change']:.1f}%")
                print(f"📊 Volume Trend: {volume['trend']}")
                print(f"💪 Volume Strength: {volume['strength']}")
                print(f"📉 Price Change: {volume['price_change']:.1f}%")
                print(f"✅ Volume Confirmation: {'Yes' if volume['is_confirming'] else 'No'}")

                # Add new enhanced analysis sections
                print("\n====== 🔄 Enhanced Market Analysis ======")

                # Market Cycles Analysis - Simplified Critical Information
                print("\n=== 🔄 Critical Market Cycle Indicators ===")
                cycles = analysis['market_cycles']
                
                # Current market phase with emoji indicators
                phase_emojis = {
                    'Accumulation': '🟢',
                    'Markup': '📈',
                    'Distribution': '🟡',
                    'Markdown': '📉',
                    'Unknown': '❓'
                }
                phase_emoji = phase_emojis.get(cycles['phase'], '❓')
                print(f"{phase_emoji} Market Phase: {cycles['phase']}")
                print(f"├─ Confidence: {cycles['confidence']*100:.1f}%")
                
                # Critical market conditions
                print("\n🎯 Market Direction:")
                print(f"├─ Price Trend: {cycles['metrics']['price_trend']}")
                print(f"└─ Momentum: {cycles['metrics']['momentum']}")
                
                # Show only significant cycle transitions
                if cycles['cycle_history']:
                    print("\n⚡ Recent Major Transition:")
                    latest_cycle = cycles['cycle_history'][-1]  # Get most recent transition
                    print(f"└─ {latest_cycle['phase'].title()} → {cycles['phase']} ({latest_cycle['duration']:.1f}h)")

                # Order Flow Analysis
                print("\n=== 📊 Order Flow Analysis ===")
                flow = analysis['order_flow']
                print(f"Signal: {flow['signal']} (Strength: {flow['strength']:.2f})")
                print("\nMetrics:")
                print(f"• Flow Ratio: {flow['metrics']['flow_ratio']:.2f}")
                print(f"• Buy Pressure: {flow['metrics']['buy_pressure']:.2f}")
                print(f"• Sell Pressure: {flow['metrics']['sell_pressure']:.2f}")
                print(f"• Depth Imbalance: {flow['metrics']['depth_imbalance']:.2f}")

                if flow['tick_analysis']:
                    print("\nTick Analysis:")
                    for key, value in flow['tick_analysis'].items():
                        print(f"• {key.replace('_', ' ').title()}: {value}")

                # Liquidity Zones Analysis
                print("\n=== 💧 Liquidity Zones Analysis ===")
                liquidity = analysis['liquidity_zones']
                
                if liquidity['current_zone']:
                    print("\nCurrent Zone:")
                    zone = liquidity['current_zone']
                    print(f"• Type: {zone['type']}")
                    print(f"• Price: ${zone['price']:.4f}")
                    print(f"• Strength: {zone['strength']:.2f}")

                print("\nHigh Volume Zones:")
                for zone in liquidity['zones']['high_volume'][:3]:  # Show top 3
                    print(f"• ${zone['price']:.4f} ({zone['type']}, Strength: {zone['strength']:.2f})")

                print("\nBreakout Zones:")
                for zone in liquidity['zones']['low_volume'][:3]:  # Show top 3
                    print(f"• ${zone['price']:.4f} (Strength: {zone['strength']:.2f})")

                # Market Efficiency Analysis
                print("\n=== 📈 Market Efficiency Analysis ===")
                efficiency = analysis['market_efficiency']
                print(f"Efficiency Ratio: {efficiency['ratio']:.2f}")
                print(f"Interpretation: {efficiency['interpretation']}")
                print(f"Trend Quality: {efficiency['trend_quality']}")
                print("\nMetrics:")
                print(f"• Directional Movement: {efficiency['metrics']['directional_movement']:.4f}")
                print(f"• Total Movement: {efficiency['metrics']['total_movement']:.4f}")
                print(f"• Average Efficiency: {efficiency['metrics']['average_efficiency']:.2f}")

                # Adaptive Thresholds
                print("\n=== 🎚️ Adaptive Thresholds ===")
                thresholds = analysis['adaptive_thresholds']
                for name, threshold in thresholds.items():
                    print(f"\n{name.title()}:")
                    print(f"• Current: {threshold['current']:.4f}")
                    print(f"• Base: {threshold['base']:.4f}")
                    print(f"• Adjustment: {threshold['adjustment']:.2f}x")

                print("\n" + "="*50)

                
                # Pattern Recognition Section
                print("\n=== 🔍 Pattern Analysis ===")
                patterns = analysis['patterns']
                print(f"📋 Current Pattern: {patterns['type']}")
                if patterns['type'] != "None":
                    print(f"🎯 Pattern Confidence: {patterns['confidence']*100:.1f}%")
                    if patterns['target']:
                        print(f"🎯 Pattern Target: ${patterns['target']:.4f}")
                    if patterns['stop_loss']:
                        print(f"🛑 Suggested Stop Loss: ${patterns['stop_loss']:.4f}")
                
                # Pattern History
                print("\n=== 📜 Recent Pattern History ===")
                for pattern in analysis['pattern_history'][-3:]:  # Show last 3 patterns
                    print(f"• {pattern['pattern']} (Confidence: {pattern['confidence']*100:.1f}%) - {pattern['timestamp']}")
                
                # Add Regime Analysis Section after Pattern History
                print("\n=== 🌊 Market Regime Analysis ===")
                regime = analysis['regime_analysis']
                print(f"📊 Current Regime: {regime['regime']}")
                print(f"🎯 Confidence: {regime['confidence']:.1f}%")
                print("\n📊 Regime Metrics:")
                print(f"• 📈 Volatility: {regime['metrics']['volatility']:.1f}%")
                print(f"• 💪 Trend Strength: {regime['metrics']['trend_strength']:.1f}")
                print(f"• 📏 Price Range: {regime['metrics']['price_range']:.1f}%")
                
                # Add Momentum Analysis Section
                print("\n=== 🚀 Momentum Analysis ===")
                momentum = analysis['momentum_analysis']
                print(f"📈 Overall Momentum: {momentum['interpretation']}")
                print(f"📊 Total Score: {momentum['total_score']:.1f}")
                print("\n📈 Component Scores:")
                for component, score in momentum['components'].items():
                    print(f"• {component.replace('_', ' ').title()}: {score:.1f}")
                
                # Risk Metrics Section
                print("\n=== ⚠️ Risk Analysis ===")
                risk = analysis['risk_metrics']
                print(f"📊 Dynamic Risk Level: {risk['dynamic_risk']*100:.1f}%")
                if 'volatility' in risk:
                    print(f"📈 Current Volatility: {risk['volatility']*100:.1f}%")
                if 'risk_reward_ratio' in risk:
                    print(f"⚖️ Risk/Reward Ratio: {risk['risk_reward_ratio']:.2f}")
                
                # Trading Recommendation Section
                print("\n=== 💡 Trading Recommendation ===")
                print(analysis['recommendation'])
                
                # Key Levels and Potential Moves
                print("\n=== 🎯 Key Levels & Potential Moves ===")
                atr = indicators.get('atr', (indicators['bollinger_upper'] - indicators['bollinger_lower']) / 4)
                current_price = analysis['current_price']
                
                print(f"📈 Potential Bullish Targets:")
                print(f"• 🎯 Conservative: ${current_price * 1.01:.4f} (+1%)")
                print(f"• 🎯 Moderate: ${current_price * 1.02:.4f} (+2%)")
                print(f"• 🎯 Aggressive: ${current_price * 1.05:.4f} (+5%)")
                
                print(f"\n📉 Potential Bearish Targets:")
                print(f"• 🎯 Conservative: ${current_price * 0.99:.4f} (-1%)")
                print(f"• 🎯 Moderate: ${current_price * 0.98:.4f} (-2%)")
                print(f"• 🎯 Aggressive: ${current_price * 0.95:.4f} (-5%)")
                
                # Add directional bias analysis with move specifics
                print("\n=== 🎯 Directional Bias & Move Analysis ===")
                bullish_points = 0
                bearish_points = 0
                
                # RSI Analysis
                if indicators['rsi'] > 50:
                    bullish_points += 1
                else:
                    bearish_points += 1
                    
                # MACD Analysis
                if indicators['macd'] > indicators['macd_signal']:
                    bullish_points += 1
                else:
                    bearish_points += 1
                    
                # Trend Direction
                if indicators['trend_direction'] == "Uptrend":
                    bullish_points += 2
                elif indicators['trend_direction'] == "Downtrend":
                    bearish_points += 2
                    
                # Volume Analysis
                if volume['is_confirming'] and volume['price_change'] > 0:
                    bullish_points += 1
                elif volume['is_confirming'] and volume['price_change'] < 0:
                    bearish_points += 1
                    
                # Price relative to Bollinger Bands
                if current_price > indicators['bollinger_middle']:
                    bullish_points += 1
                else:
                    bearish_points += 1
                    
                # Calculate confidence percentage
                total_points = bullish_points + bearish_points
                bullish_confidence = (bullish_points / total_points * 100) if total_points > 0 else 50
                bearish_confidence = (bearish_points / total_points * 100) if total_points > 0 else 50
                
                # Calculate move specifics
                atr = indicators.get('atr', (indicators['bollinger_upper'] - indicators['bollinger_lower']) / 4)
                bb_width = indicators['bollinger_upper'] - indicators['bollinger_lower']
                price_volatility = bb_width / indicators['bollinger_middle']
                
                # Define move characteristics
                move_speed = "Rapid" if price_volatility > 0.03 else "Gradual"
                move_strength = "Strong" if abs(indicators['macd']) > abs(indicators['macd_signal']) * 1.5 else "Moderate"

                # Correlation Analysis
                print("\n=== 📈 Market Correlation Analysis ===")
                correlation_analysis = analysis['correlation_analysis']
                print(f"📊 Average Correlation: {correlation_analysis['average_correlation']:.2f}")
                print(f"🎯 Market Independence: {correlation_analysis['independence_score']:.2f}")
                print("\nInterpretation:")
                for insight, description in correlation_analysis['interpretation'].items():
                    print(f"• {insight}: {description}")
                print("\nCorrelation Details:")
                for asset, details in correlation_analysis['correlations'].items():
                    print(f"\n💱 {asset}:")
                    if 'error' in details:
                        print(f"❌ Error: {details['error']}")
                    else:
                        try:
                            print(f"📊 Correlation Coefficient: {details['coefficient']:.2f}")
                            print(f"🎯 Correlation Type: {details['type']}")
                            print(f"📈 Price Movement: Current Asset vs. {asset}")
                            print(f"• Current Asset: {details['price_movement']['current_asset']}")
                            print(f"• {asset}: {details['price_movement']['correlated_asset']}")
                            print(f"📅 Last Updated: {details['timestamp']}")
                        except KeyError as e:
                            print(f"❌ Error: Missing data - {str(e)}")
                        except Exception as e:
                            print(f"❌ Error: {str(e)}")        
                
                print("")
                print("============== Move Analysis: ==============")
                if bullish_points > bearish_points:
                    print(f"BULLISH with {bullish_confidence:.1f}% confidence")
                    print("\nMove Characteristics:")
                    print(f"• Expected Move Type: {move_speed} {move_strength} Advance")
                    print(f"• Momentum: {'Accelerating' if indicators['macd_histogram'] > 0 else 'Decelerating'}")
                    print(f"• Volume Profile: {'Supporting' if volume['is_confirming'] else 'Lacking'}")
                    
                    print("\nPrice Targets:")
                    print(f"• Initial Target: ${(current_price + atr):.4f} (+{(atr/current_price)*100:.1f}%)")
                    print(f"• Secondary Target: ${indicators['bollinger_upper']:.4f} (+{((indicators['bollinger_upper']-current_price)/current_price)*100:.1f}%)")
                    print(f"• Extended Target: ${(indicators['bollinger_upper'] + atr):.4f} (+{((indicators['bollinger_upper']+atr-current_price)/current_price)*100:.1f}%)")
                    
                    print("\nSupporting Factors:")
                    if indicators['rsi'] > 50:
                        print(f"• RSI showing upward momentum ({indicators['rsi']:.1f})")
                    if indicators['macd'] > indicators['macd_signal']:
                        print(f"• MACD bullish crossover (Spread: {(indicators['macd']-indicators['macd_signal']):.4f})")
                    if indicators['trend_direction'] == "Uptrend":
                        print("• Established uptrend with higher lows")
                    if volume['is_confirming'] and volume['price_change'] > 0:
                        print(f"• Volume increased by {volume['change']:.1f}% supporting price action")
                    if current_price > indicators['bollinger_middle']:
                        print("• Price trading above BB middle band showing strength")
                        
                elif bearish_points > bullish_points:
                    print(f"BEARISH with {bearish_confidence:.1f}% confidence")
                    print("\nMove Characteristics:")
                    print(f"• Expected Move Type: {move_speed} {move_strength} Decline")
                    print(f"• Momentum: {'Accelerating' if indicators['macd_histogram'] < 0 else 'Decelerating'}")
                    print(f"• Volume Profile: {'Supporting' if volume['is_confirming'] else 'Lacking'}")
                    
                    print("\nPrice Targets:")
                    print(f"• Initial Target: ${(current_price - atr):.4f} (-{(atr/current_price)*100:.1f}%)")
                    print(f"• Secondary Target: ${indicators['bollinger_lower']:.4f} (-{((current_price-indicators['bollinger_lower'])/current_price)*100:.1f}%)")
                    print(f"• Extended Target: ${(indicators['bollinger_lower'] - atr):.4f} (-{((current_price-(indicators['bollinger_lower']-atr))/current_price)*100:.1f}%)")
                    
                    print("\nSupporting Factors:")
                    if indicators['rsi'] < 50:
                        print(f"• RSI showing downward momentum ({indicators['rsi']:.1f})")
                    if indicators['macd'] < indicators['macd_signal']:
                        print(f"• MACD bearish crossover (Spread: {(indicators['macd_signal']-indicators['macd']):.4f})")
                    if indicators['trend_direction'] == "Downtrend":
                        print("• Established downtrend with lower highs")
                    if volume['is_confirming'] and volume['price_change'] < 0:
                        print(f"• Volume increased by {volume['change']:.1f}% supporting price action")
                    if current_price < indicators['bollinger_middle']:
                        print("• Price trading below BB middle band showing weakness")
                else:
                    print("NEUTRAL - No Clear Directional Bias")
                    print("\nConsolidation Analysis:")
                    print(f"• Price Range: ${(current_price - atr):.4f} to ${(current_price + atr):.4f}")
                    print(f"• Volatility: {'High' if price_volatility > 0.03 else 'Low'} ({price_volatility*100:.1f}%)")
                    print(f"• Volume Profile: {volume['trend']} on {volume['change']:.1f}% change")
                    print("\nBreakout Levels:")
                    print(f"• Bullish Breakout Above: ${indicators['bollinger_upper']:.4f}")
                    print(f"• Bearish Breakdown Below: ${indicators['bollinger_lower']:.4f}")
                    print("\nRecommendation:")
                    print("• Consider waiting for stronger directional signals")
                    print("• Monitor for breakout of recent trading range")
                    print("• Prepare for potential volatility expansion")

                # Add Volatility Regime Analysis Section
                print("\n====== 📊 Volatility Regime Analysis 📊 ======")
                vol_regime = analysis.get('volatility_regime', {})
                if vol_regime:
                    print(f"Current Regime: {vol_regime['regime']}")
                    print(f"Confidence: {vol_regime['confidence']*100:.1f}%")
                    print(f"\nDescription: {vol_regime['description']}")
                    
                    print("\nKey Metrics:")
                    metrics = vol_regime.get('metrics', {})
                    print(f"• Current Volatility: {metrics.get('current_volatility', 0)*100:.1f}%")
                    print(f"• Long-term Volatility: {metrics.get('long_term_volatility', 0)*100:.1f}%")
                    print(f"• Volatility of Volatility: {metrics.get('volatility_of_volatility', 0)*100:.1f}%")
                    print(f"• Volatility Trend: {metrics.get('volatility_trend', 'Unknown')}")
                    print(f"• Volatility Percentile: {metrics.get('volatility_percentile', 0)*100:.1f}%")
                    
                    print("\nTrading Implications:")
                    for implication in vol_regime.get('implications', []):
                        print(f"• {implication}")

                # Add Liquidity Depth Analysis Section
                print("\n====== 📊 Liquidity Depth Analysis 📊 ======")
                liquidity_analysis = analysis.get('liquidity_analysis', {})
                
                if not liquidity_analysis:
                    print("❌ No liquidity analysis data available")
                elif liquidity_analysis.get('status') == 'success':
                    print(f"Liquidity Score: {liquidity_analysis['liquidity_score']:.1f}%")
                    print("\nMetrics:")
                    metrics = liquidity_analysis.get('metrics', {})
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"• {metric.replace('_', ' ').title()}: {value:.1f}%")
                        else:
                            print(f"• {metric.replace('_', ' ').title()}: {value}")
                            
                    print("\nVolume Profile:")
                    for profile in liquidity_analysis.get('volume_profile', []):
                        print(f"• Price Level: ${profile.get('price_level', 0):.4f}, "
                              f"Volume: {profile.get('volume', 0)}, "
                              f"Percentage: {profile.get('percentage', 0):.1f}%")
                              
                    print("\nLiquidity Clusters:")
                    for cluster in liquidity_analysis.get('liquidity_clusters', []):
                        price_range = cluster.get('price_range', (0, 0))
                        print(f"• Price Range: ${price_range[0]:.4f} to ${price_range[1]:.4f}, "
                              f"Total Volume: {cluster.get('total_volume', 0)}, "
                              f"Strength: {cluster.get('strength', 0):.1f}%")
                              
                    print("\nLiquidity Gaps:")
                    for gap in liquidity_analysis.get('liquidity_gaps', []):
                        print(f"• Start Price: ${gap.get('start_price', 0):.4f}, "
                              f"End Price: ${gap.get('end_price', 0):.4f}, "
                              f"Size: {gap.get('size', 0):.1f}%")
                              
                    print("\nMarket Depth:")
                    market_depth = liquidity_analysis.get('market_depth', {})
                    print(f"• Buy Side Volume: {market_depth.get('buy_side_volume', 0)}")
                    print(f"• Sell Side Volume: {market_depth.get('sell_side_volume', 0)}")
                    print(f"• Imbalance Ratio: {market_depth.get('imbalance_ratio', 0):.2f} "
                          f"({market_depth.get('interpretation', 'Unknown')})")
                          
                elif liquidity_analysis.get('status') == 'insufficient_data':
                    print("⚠️ Insufficient data to analyze liquidity depth")
                else:
                    error_msg = liquidity_analysis.get('error', 'Unknown error occurred')
                    print(f"❌ Error analyzing liquidity depth: {error_msg}")

                # Enhanced Probability Analysis Section
                print("\n====== 📊 Detailed Move Analysis 📊 ======")
                prob = analysis['probability_analysis']
                print(f"\n🎯 Overall Success Probability: {prob['total_probability']:.1f}%")
                print(f"📊 Confidence Level: {prob['confidence_level']}")
                
                print("\n📈 Move Quality:")
                move_quality = prob['move_quality']
                print(f"• ⚡ Expected Speed: {move_quality['expected_speed']}")
                print(f"• 📊 Expected Volatility: {move_quality['expected_volatility']}")
                print(f"• 🎯 Continuation Probability: {move_quality['continuation_probability']}")
                print(f"• ⚠️ Reversal Risk: {move_quality['reversal_risk']}")
                
                print("\n📈 Move Characteristics:")
                chars = prob['move_characteristics']
                
                print("\n📈 Trend Quality:")
                trend = chars['trend_quality']
                print(f"• 🎯 Strength: {trend['strength']}")
                print(f"• 🎯 Duration: {trend['duration']}")
                print(f"• 🎯 Momentum: {trend['momentum']}")
                
                print("\n📈 Momentum Analysis:")
                momentum = chars['momentum']
                print(f"• 🎯 Condition: {momentum['condition']}")
                print(f"• 🎯 Strength: {momentum['strength']:.2f}")
                print(f"• 🎯 Divergence: {momentum['divergence']}")
                
                print("\n📈 Volume Quality:")
                volume = chars['volume_quality']
                print(f"• 🎯 Trend: {volume['trend']}")
                print(f" 🎯 Strength: {volume['strength']}")
                print(f"• 🎯 Consistency: {volume['consistency']}")
                print(f"• 🎯 Price Alignment: {volume['price_alignment']}")
                
                print("\n📈 Pattern Analysis:")
                pattern = chars['pattern_analysis']
                print(f"• 🎯 Type: {pattern['type']}, • 🎯 Reliability: {pattern['reliability']:.2f}, • 🎯 Completion: {pattern['completion']:.1f}%")
                
                print("\n📉 Failure Points:")
                failure = prob['failure_points']
                if failure['immediate_stop']:
                    print(f"• 🛑 Immediate Stop: ${failure['immediate_stop']:.4f}")
                if failure['trend_reversal_point']:
                    print(f"• 🎯 Trend Reversal: ${failure['trend_reversal_point']:.4f}")
                if failure['momentum_failure_level']:
                    print(f"• 🎯 Momentum Failure: ${failure['momentum_failure_level']:.4f}")
                
                print("\n📈 Contributing Factors: " + ", ".join([f"• {factor}: {value:.1f}%" for factor, value in prob['factors']]))

                # Add Divergence Analysis Section
                print("\n====== 📊 Divergence Analysis 📊 ======")
                divergences = analysis.get('divergence_analysis', {})
                
                if 'error' in divergences:
                    print(f"❌ Error analyzing divergences: {divergences['error']}")
                else:
                    summary = divergences.get('summary', {})
                    print(f"Total Divergences Found: {summary.get('total_divergences', 0)}")
                    print(f"Primary Bias: {summary.get('primary_bias', 'None')}")
                    print(f"Confidence: {summary.get('highest_confidence', 0)*100:.1f}%")
                    
                    if divergences.get('divergences'):
                        print("\nDetailed Divergences:")
                        for div in divergences['divergences']:
                            print(f"\n🔍 {div['type']} ({div['indicator']})")
                            print(f"├─ Strength: {div['strength']:.4f}")
                            print(f"├─ Confidence: {div['confidence']*100:.1f}%")
                            print(f"├─ Timeframe: {div.get('timeframe', 'Current')}")
                            
                            analysis = div.get('analysis', {})
                            print(f"├─ Price Change: {analysis.get('price_change_percent', 0):.1f}%")
                            print(f"├─ Significance: {analysis.get('significance', 'Unknown')}")
                            print(f"├─ Suggested Action: {analysis.get('suggested_action', 'None')}")
                            if 'stop_loss' in analysis:
                                print(f"└─ Suggested Stop Loss: ${analysis['stop_loss']:.4f}")
                    else:
                        print("\nNo significant divergences detected in current market conditions.")

                # Add Alerts Section to the output
                print("\n====== 🔔 Market Alerts 🔔 ======")
                alerts = analysis.get('alerts', {})
                
                if alerts.get('triggered_alerts'):
                    print("\n🚨 Triggered Alerts:")
                    for alert in alerts['triggered_alerts']:
                        print(f"• [{alert['priority']}] {alert['type']}: {alert['message']}")
                        print(f"  Time: {alert['timestamp']}")
                
                if alerts.get('auto_alerts'):
                    print("\n📢 Automatic Alerts:")
                    for alert in alerts['auto_alerts']:
                        print(f"• [{alert['priority']}] {alert['type']}: {alert['message']}")
                        print(f"  Time: {alert['timestamp']}")
                
                summary = alerts.get('alert_summary', {})
                print("\n📊 Alert Summary:")
                print(f"• Active Alerts: {summary.get('active_alerts', 0)}")
                print(f"• Triggered Alerts: {summary.get('triggered_alerts', 0)}")
                
                print("\nAlert Distribution by Priority:")
                for priority, count in summary.get('alerts_by_priority', {}).items():
                    print(f"• {priority}: {count}")
                
                print("\nAlert Distribution by Type:")
                for alert_type, count in summary.get('alerts_by_type', {}).items():
                    print(f"• {alert_type}: {count}")

                # Add Order Book Analysis Section
                print("\n====== 📚 Order Book Analysis 📚 ======")
                if order_book['status'] == 'success':
                    print("\n=== 📊 Order Book Metrics ===")
                    metrics = order_book['metrics']
                    print(f"Bid/Ask Ratio: {metrics['bid_ask_ratio']:.2f}")
                    print(f"Spread: ${metrics['spread']:.4f} ({(metrics['spread']/order_book['current_price']*100):.3f}%)")
                    print(f"Market Depth Score: {metrics['depth_score']:.2f}")
                    
                    print("\n=== 💪 Market Pressure ===")
                    print(f"Buy Pressure: {metrics['buy_pressure']:.2f}")
                    print(f"Sell Pressure: {metrics['sell_pressure']:.2f}")
                    print(f"Pressure Ratio: {metrics['pressure_ratio']:.2f}")
                    print(f"Imbalance Score: {metrics['imbalance_score']:.2f}")
                    
                    print("\n=== 🎯 Significant Price Levels ===")
                    print("\nSupport Levels:")
                    for level in order_book['significant_levels']['support']:
                        print(f"• ${level['price']:.4f} (Strength: {level['strength']:.2f}x, Volume: {level['volume']:.2f})")
                    
                    print("\nResistance Levels:")
                    for level in order_book['significant_levels']['resistance']:
                        print(f"• ${level['price']:.4f} (Strength: {level['strength']:.2f}x, Volume: {level['volume']:.2f})")
                    
                    print("\n=== 📈 Order Book Analysis ===")
                    analysis = order_book['analysis']
                    print(f"Market Pressure: {analysis['market_pressure']}")
                    print(f"Depth Quality: {analysis['depth_quality']}")
                    print(f"Spread Quality: {analysis['spread_quality']}")
                    print(f"Imbalance: {analysis['imbalance_interpretation']}")
                    
                    # Add visual representation of order book depth
                    print("\n=== 📊 Order Book Depth Visualization ===")
                    max_bar_length = 50
                    max_volume = max(max(order_book['visualization']['bid_volumes']), 
                                   max(order_book['visualization']['ask_volumes']))
                    
                    print("\nAsks (Sell Orders):")
                    for i in range(len(order_book['visualization']['ask_levels'])-1, -1, -1):
                        price = order_book['visualization']['ask_levels'][i]
                        volume = order_book['visualization']['ask_volumes'][i]
                        bar_length = int((volume / max_volume) * max_bar_length)
                        print(f"${price:.4f} | {'█' * bar_length}{' ' * (max_bar_length - bar_length)} | {volume:.2f}")
                    
                    print("\nBids (Buy Orders):")
                    for i in range(len(order_book['visualization']['bid_levels'])):
                        price = order_book['visualization']['bid_levels'][i]
                        volume = order_book['visualization']['bid_volumes'][i]
                        bar_length = int((volume / max_volume) * max_bar_length)
                        print(f"${price:.4f} | {'█' * bar_length}{' ' * (max_bar_length - bar_length)} | {volume:.2f}")
                    
                elif order_book['status'] == 'insufficient_data':
                    print("⚠️ Insufficient data for order book analysis")
                else:
                    print(f"❌ Error analyzing order book: {order_book.get('error', 'Unknown error')}")

        except KeyboardInterrupt:
            print("\n\n⛔ Analysis stopped by user")
        except Exception as e:
            print(f"\n❌ Error during market analysis: {str(e)}")
            logging.error(f"Error during market analysis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 