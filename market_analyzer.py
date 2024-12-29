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
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
    def prepare_features(self, candles: List[Dict]) -> np.ndarray:
        """Prepare features for ML model."""
        df = pd.DataFrame(candles)
        
        # Technical indicators as features
        features = []
        prices = df['close'].values
        volumes = df['volume'].values
        
        # Price-based features
        features.append(np.gradient(prices))  # Price momentum
        features.append(np.std(prices[-20:]))  # Volatility
        
        # Volume-based features
        features.append(np.gradient(volumes))  # Volume momentum
        features.append(volumes[-1] / np.mean(volumes[-20:]))  # Relative volume
        
        # Trend features
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        features.append((prices[-1] - sma_20) / sma_20)  # Price vs SMA20
        features.append((sma_20 - sma_50) / sma_50)  # SMA20 vs SMA50
        
        return np.array(features).reshape(1, -1)
        
    def train(self, candles: List[Dict], labels: List[int]):
        """Train the ML model."""
        X = self.prepare_features(candles)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        
    def predict(self, candles: List[Dict]) -> float:
        """Predict market direction."""
        if self.model is None:
            return 0.0
            
        X = self.prepare_features(candles)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Convert probabilities to signal strength (-1 to 1)
        return (probabilities[1] - probabilities[0]) * 2 - 1
        
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is not None:
            joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
            
    def load_model(self, path: str):
        """Load model from file."""
        try:
            saved = joblib.load(path)
            self.model = saved['model']
            self.scaler = saved['scaler']
        except Exception as e:
            logging.error(f"Error loading ML model: {str(e)}")

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

            # Add signal stability information
            signal_stability = "High" if len(self.technical_analysis.signal_history) >= self.technical_analysis.trend_confirmation_period and \
                                  all(s['strength'] > 0 for s in self.technical_analysis.signal_history) or \
                                  all(s['strength'] < 0 for s in self.technical_analysis.signal_history) else "Low"
            
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
                'position': 'LONG' if analysis['signal'].signal_type in [SignalType.STRONG_BUY, SignalType.BUY] 
                           else 'SHORT' if analysis['signal'].signal_type in [SignalType.STRONG_SELL, SignalType.SELL]
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
                'signal_stability': signal_stability,
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
        More sensitive thresholds for more frequent signals.
        
        Args:
            signal_strength: Float value between -10 and 10
            
        Returns:
            SignalType: The determined signal type
        """
        if signal_strength >= 4:  # Reduced from 7
            return SignalType.STRONG_BUY
        elif signal_strength >= 1.5:  # Reduced from 3
            return SignalType.BUY
        elif signal_strength <= -4:  # Changed from -7
            return SignalType.STRONG_SELL
        elif signal_strength <= -1.5:  # Changed from -3
            return SignalType.SELL
        else:
            # Check if there's a slight bias even in the "neutral" zone
            if signal_strength > 0.5:  # Slight bullish bias
                return SignalType.BUY
            elif signal_strength < -0.5:  # Slight bearish bias
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
                "Bull Market": 1.3,
                "Bear Market": 0.7,
                "Bullish": 1.2,
                "Bearish": 0.8,
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
            prices = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            
            patterns = {
                'type': PatternType.NONE,
                'confidence': 0.0,
                'target': None,
                'stop_loss': None
            }
            
            # Double Top Detection
            if self._is_double_top(highs[-30:]):
                patterns['type'] = PatternType.DOUBLE_TOP
                patterns['confidence'] = 0.8
                patterns['target'] = min(lows[-30:])
                patterns['stop_loss'] = max(highs[-30:]) + (max(highs[-30:]) - min(lows[-30:])) * 0.1
            
            # Double Bottom Detection
            elif self._is_double_bottom(lows[-30:]):
                patterns['type'] = PatternType.DOUBLE_BOTTOM
                patterns['confidence'] = 0.8
                patterns['target'] = max(highs[-30:])
                patterns['stop_loss'] = min(lows[-30:]) - (max(highs[-30:]) - min(lows[-30:])) * 0.1
            
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
            
            # Trend alignment (0-20%)
            trend_direction = indicators.get('trend_direction', 'Unknown')
            adx = indicators.get('adx', 0)
            prev_adx = indicators.get('prev_adx', 0)
            
            # Enhanced trend analysis with momentum consideration
            trend_momentum = "Accelerating" if adx > prev_adx else "Decelerating"
            trend_strength = "Strong" if adx > 25 else "Moderate" if adx > 15 else "Weak"
            
            if trend_direction == "Uptrend":
                trend_score = 20 if adx > 25 else 15 if adx > 15 else 10
                # Add momentum bonus
                if trend_momentum == "Accelerating":
                    trend_score += 5
                probability_factors.append(("Trend", trend_score))
                move_characteristics['trend_quality'] = {
                    'strength': trend_strength,
                    'duration': 'Established' if adx > 30 else 'Developing',
                    'momentum': trend_momentum,
                    'consistency': 'High' if adx > prev_adx else 'Moderate'
                }
            elif trend_direction == "Downtrend":
                trend_score = 20 if adx > 25 else 15 if adx > 15 else 10
                # Add momentum bonus
                if trend_momentum == "Accelerating":
                    trend_score += 5
                probability_factors.append(("Trend", trend_score))
                move_characteristics['trend_quality'] = {
                    'strength': trend_strength,
                    'duration': 'Established' if adx > 30 else 'Developing',
                    'momentum': trend_momentum,
                    'consistency': 'High' if adx > prev_adx else 'Moderate'
                }
            else:
                probability_factors.append(("Trend", 5))
                move_characteristics['trend_quality'] = {
                    'strength': 'Weak',
                    'duration': 'Undefined',
                    'momentum': 'Neutral',
                    'consistency': 'Low'
                }

            # Enhanced RSI analysis with divergence detection (0-15%)
            rsi = indicators.get('rsi', 50)
            rsi_prev = indicators.get('prev_rsi', 50)
            price = indicators.get('current_price', 0)
            price_prev = indicators.get('prev_price', price)
            
            # Detect RSI divergence
            rsi_divergence = "Bullish" if rsi > rsi_prev and price < price_prev else \
                           "Bearish" if rsi < rsi_prev and price > price_prev else "None"
            
            rsi_momentum = {
                'condition': 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral',
                'strength': abs(rsi - 50) / 50,
                'divergence': rsi_divergence,
                'trend': 'Bullish' if rsi > rsi_prev else 'Bearish' if rsi < rsi_prev else 'Neutral'
            }
            
            rsi_score = 15 if (rsi > 70 or rsi < 30) else \
                       10 if (rsi > 60 or rsi < 40) else \
                       5
            
            # Add divergence bonus
            if rsi_divergence != "None":
                rsi_score += 5
                
            probability_factors.append(("RSI", rsi_score))
            move_characteristics['momentum'] = rsi_momentum

            # Enhanced MACD analysis with trend confirmation (0-15%)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            histogram = indicators.get('macd_histogram', 0)
            prev_histogram = indicators.get('prev_macd_histogram', 0)
            
            macd_analysis = {
                'crossover_type': 'Bullish' if macd > macd_signal else 'Bearish' if macd < macd_signal else 'None',
                'histogram_strength': abs(histogram) / abs(macd) if macd != 0 else 0,
                'momentum_quality': 'Increasing' if histogram > 0 and histogram > prev_histogram else
                                  'Decreasing' if histogram < 0 and histogram < prev_histogram else 'Neutral',
                'trend_alignment': 'Aligned' if (macd > macd_signal and trend_direction == "Uptrend") or
                                             (macd < macd_signal and trend_direction == "Downtrend") else 'Divergent'
            }
            
            macd_score = 0
            if macd != 0 or macd_signal != 0:
                macd_diff = abs(macd - macd_signal)
                macd_score = min(15, (macd_diff / (abs(macd_signal) + 0.00001)) * 15)
                # Add trend alignment bonus
                if macd_analysis['trend_alignment'] == 'Aligned':
                    macd_score += 5
                    
            probability_factors.append(("MACD", macd_score))
            move_characteristics['macd_analysis'] = macd_analysis

            # Enhanced volume analysis with trend confirmation (0-20%)
            volume_change = volume_info.get('volume_change', 0)
            is_confirming = volume_info.get('is_confirming', False)
            volume_trend = volume_info.get('volume_trend', 'Neutral')
            
            volume_quality = {
                'trend': volume_trend,
                'strength': 'Strong' if abs(volume_change) > 50 else 'Moderate' if abs(volume_change) > 20 else 'Weak',
                'consistency': 'High' if is_confirming else 'Low',
                'price_alignment': 'Confirmed' if is_confirming else 'Divergent',
                'trend_support': 'Strong' if volume_trend == "Increasing" and is_confirming else
                               'Weak' if volume_trend == "Decreasing" else 'Neutral'
            }
            
            volume_score = 0
            if is_confirming:
                volume_score = min(20, abs(volume_change) / 5)
                # Add trend support bonus
                if volume_quality['trend_support'] == 'Strong':
                    volume_score += 5
            else:
                volume_score = 5
                
            probability_factors.append(("Volume", volume_score))
            move_characteristics['volume_quality'] = volume_quality

            # Enhanced pattern recognition with failure points (0-15%)
            pattern_type = patterns.get('type', 'None')
            pattern_confidence = patterns.get('confidence', 0)
            
            pattern_analysis = {
                'type': pattern_type,
                'reliability': pattern_confidence,
                'completion': patterns.get('completion_percentage', 0),
                'failure_points': {
                    'immediate': patterns.get('stop_loss', None),
                    'pattern_invalidation': patterns.get('invalidation_level', None)
                },
                'confirmation_status': 'Confirmed' if is_confirming and pattern_confidence > 0.7 else
                                     'Partial' if pattern_confidence > 0.5 else 'Unconfirmed'
            }
            
            pattern_score = 0
            if pattern_type != "None":
                pattern_score = min(15, pattern_confidence * 15)
                # Add confirmation bonus
                if pattern_analysis['confirmation_status'] == 'Confirmed':
                    pattern_score += 5
                elif pattern_analysis['confirmation_status'] == 'Partial':
                    pattern_score += 2
                    
            probability_factors.append(("Pattern", pattern_score))
            move_characteristics['pattern_analysis'] = pattern_analysis

            # Enhanced market condition analysis (0-15%)
            market_condition = indicators.get('market_condition', 'Unknown')
            volatility = indicators.get('volatility', 'Normal')
            
            market_context = {
                'condition': market_condition,
                'volatility': volatility,
                'liquidity': 'High' if volume_info.get('volume_change', 0) > 0 else 'Normal',
                'support_resistance_proximity': indicators.get('price_level_proximity', 'Far'),
                'market_phase': 'Accumulation' if market_condition == "Bull Market" and volume_trend == "Increasing" else
                               'Distribution' if market_condition == "Bear Market" and volume_trend == "Decreasing" else
                               'Transition'
            }
            
            market_score = 15 if market_condition in ["Bull Market", "Bear Market"] else \
                         10 if market_condition in ["Bullish", "Bearish"] else 5
                         
            # Add market phase bonus
            if market_context['market_phase'] in ['Accumulation', 'Distribution']:
                market_score += 5
                
            probability_factors.append(("Market", market_score))
            move_characteristics['market_context'] = market_context

            # Calculate total probability with weighted factors
            total_probability = sum(factor[1] for factor in probability_factors)
            total_probability = max(0, min(100, total_probability))
            
            # Determine move quality characteristics
            move_quality = {
                'expected_speed': 'Rapid' if total_probability > 80 else 'Moderate' if total_probability > 60 else 'Slow',
                'expected_volatility': 'High' if volatility == 'High' else 'Normal',
                'continuation_probability': f"{total_probability:.1f}%",
                'reversal_risk': 'Low' if total_probability > 75 else 'Moderate' if total_probability > 50 else 'High',
                'strength_rating': 'Very Strong' if total_probability > 85 else
                                 'Strong' if total_probability > 70 else
                                 'Moderate' if total_probability > 50 else 'Weak'
            }

            # Calculate confidence level with more granularity
            confidence_level = "Very High" if total_probability >= 85 else \
                             "High" if total_probability >= 70 else \
                             "Moderate" if total_probability >= 50 else \
                             "Low" if total_probability >= 30 else "Very Low"

            # Add failure points analysis
            failure_points = {
                'immediate_stop': pattern_analysis['failure_points']['immediate'],
                'trend_reversal_point': pattern_analysis['failure_points']['pattern_invalidation'],
                'momentum_failure_level': indicators.get('key_reversal_level', None),
                'risk_levels': {
                    'critical': patterns.get('stop_loss', None),
                    'warning': indicators.get('support_level' if trend_direction == "Uptrend" else 'resistance_level', None),
                    'alert': indicators.get('pivot_point', None)
                }
            }

            return {
                'total_probability': total_probability,
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
        '--list-products',
        action='store_true',
        help='List all available products'
    )
    
    parser.add_argument(
        '--list-granularities',
        action='store_true',
        help='List all available granularities'
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
    
    try:
        # Get market analysis
        analysis = analyzer.get_market_signal()
        
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
        print("\n=== 📐 Technical Indicators ===")
        indicators = analysis['indicators']
        print(f"📊 RSI: {indicators['rsi']:.2f} ({'Overbought' if indicators['rsi'] > 70 else 'Oversold' if indicators['rsi'] < 30 else 'Neutral'})")
        print(f"📈 MACD: {indicators['macd']:.4f}, MACD Signal: {indicators['macd_signal']:.4f}, MACD Histogram: {indicators['macd_histogram']:.4f}")
        adx_trend = 'Strong Trend' if indicators['adx'] > 25 else 'Weak Trend'
        print(f"📏 ADX: {indicators['adx']:.2f} ({adx_trend})")
        print(f"➡️ Trend Direction: {indicators['trend_direction']}")
        
        # Bollinger Bands
        print("\n=== 📉 Price Channels ===")
        print(f"📊 Bollinger Upper: ${indicators['bollinger_upper']:.4f}, Bollinger Middle: ${indicators['bollinger_middle']:.4f}, Bollinger Lower: ${indicators['bollinger_lower']:.4f}")
        
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
        
        print("Move Analysis:")
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
        print(f"���� 🎯 Strength: {volume['strength']}")
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

    except Exception as e:
        logging.error(f"Error running market analysis: {str(e)}", exc_info=True)
        return

if __name__ == "__main__":
    main() 