import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"

@dataclass
class RegimeFeatures:
    """Features used for regime clustering."""
    volatility: float  # Price volatility
    trend_strength: float  # Trend strength (ADX-like)
    price_momentum: float  # Price momentum
    volume_trend: float  # Volume trend
    mean_reversion: float  # Mean reversion tendency
    range_bound: float  # Range-bound tendency

class RegimeClusterer:
    """Uses K-means clustering to identify market regimes."""
    
    def __init__(self, n_clusters: int = 6):
        """
        Initialize the regime clusterer.
        
        Args:
            n_clusters: Number of clusters to identify (default: 6)
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.cluster_to_regime = {}
        self.is_fitted = False
        
    def calculate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate features for regime detection."""
        features = []
        
        # Need at least 100 candles for reliable feature calculation
        if len(df) < 100:
            return np.array([])
        
        # Calculate volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=14).mean()
        
        # Calculate trend strength (ADX-like)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = true_range
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        
        # Calculate price momentum (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate volume trend
        volume_sma = df['volume'].rolling(window=20).mean()
        volume_trend = df['volume'] / volume_sma
        
        # Calculate mean reversion (Hurst exponent)
        def hurst(ts):
            lags = range(2, 100)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:].values, ts[:-lag].values))) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        
        hurst_exp = df['close'].rolling(window=100).apply(hurst)
        
        # Calculate range-bound tendency
        range_ratio = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close']
        
        # Combine features
        features = np.column_stack([
            atr / df['close'],  # Normalized ATR
            adx / 100,  # Normalized ADX
            rsi / 100,  # Normalized RSI
            volume_trend,
            hurst_exp,
            range_ratio
        ])
        
        # Remove any rows with NaN values
        features = features[~np.isnan(features).any(axis=1)]
        
        return features
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit the K-means model to historical data."""
        features = self.calculate_features(df)
        
        if len(features) == 0:
            self.is_fitted = False
            return
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit K-means
        self.kmeans.fit(features_scaled)
        
        # Map clusters to regimes
        cluster_centers = self.kmeans.cluster_centers_
        
        # Sort clusters by trend strength (ADX component)
        trend_strength = cluster_centers[:, 1]  # ADX is the second feature
        cluster_order = np.argsort(trend_strength)
        
        # Map clusters to regimes
        self.cluster_to_regime = {
            cluster_order[0]: MarketRegime.RANGING,  # Lowest trend strength
            cluster_order[1]: MarketRegime.VOLATILE,
            cluster_order[2]: MarketRegime.REVERSAL,
            cluster_order[3]: MarketRegime.BREAKOUT,
            cluster_order[4]: MarketRegime.TRENDING_DOWN,
            cluster_order[5]: MarketRegime.TRENDING_UP  # Highest trend strength
        }
        
        self.is_fitted = True
    
    def predict_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Predict the current market regime."""
        if not self.is_fitted:
            return MarketRegime.RANGING, 0.0
        
        features = self.calculate_features(df)
        
        if len(features) == 0:
            return MarketRegime.RANGING, 0.0
        
        # Get the last feature vector
        last_features = features[-1:]
        
        # Scale features
        last_features_scaled = self.scaler.transform(last_features)
        
        # Predict cluster
        cluster = self.kmeans.predict(last_features_scaled)[0]
        
        # Calculate confidence based on distance to cluster center
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - last_features_scaled, axis=1)
        min_distance = distances[cluster]
        confidence = 1 / (1 + min_distance)  # Convert distance to confidence score
        
        return self.cluster_to_regime[cluster], confidence

def detect_market_regime(df: pd.DataFrame) -> Tuple[MarketRegime, float]:
    """
    Detect the current market regime using K-means clustering.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Tuple of (MarketRegime, confidence_score)
    """
    clusterer = RegimeClusterer()
    clusterer.fit(df)
    return clusterer.predict_regime(df) 