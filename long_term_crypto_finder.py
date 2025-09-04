#!/usr/bin/env python3
"""
Long-Term Crypto Opportunity Finder

This program analyzes cryptocurrencies to find the best long-term investment opportunities
by evaluating multiple factors including technical indicators, fundamental metrics,
risk assessment, and market sentiment.

Author: Crypto Finance Toolkit
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
import logging
import time
from typing import Dict, List, Optional, Tuple
import json
import argparse
from dataclasses import dataclass
from enum import Enum
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET
from historicaldata import HistoricalData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('long_term_crypto_finder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM_LOW = "MEDIUM_LOW"
    MEDIUM = "MEDIUM"
    MEDIUM_HIGH = "MEDIUM_HIGH"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class CryptoMetrics:
    """Data class to hold comprehensive crypto metrics."""
    symbol: str
    name: str
    current_price: float
    market_cap: float
    market_cap_rank: int
    volume_24h: float
    price_change_24h: float
    price_change_7d: float
    price_change_30d: float
    ath_price: float
    ath_date: str
    atl_price: float
    atl_date: str
    volatility_30d: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    rsi_14: float
    macd_signal: str
    bb_position: str
    trend_strength: float
    momentum_score: float
    fundamental_score: float
    technical_score: float
    risk_score: float
    overall_score: float
    risk_level: RiskLevel
    # Trading levels
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    position_size_percentage: float

class LongTermCryptoFinder:
    """
    A comprehensive tool for finding long-term cryptocurrency investment opportunities using Coinbase API.
    """

    def __init__(self, min_market_cap: int = 100000000, max_results: int = 20):
        """
        Initialize the crypto finder.

        Args:
            min_market_cap: Minimum market capitalization to consider (default: $100M)
            max_results: Maximum number of results to return
        """
        # Initialize Coinbase service
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.historical_data = HistoricalData(self.coinbase_service.client)

        # List of major cryptocurrencies available on Coinbase (verified product IDs)
        self.major_cryptos = [
            "BTC-USDC", "ETH-USDC", "ADA-USDC", "SOL-USDC", "DOT-USDC",
            "LINK-USDC", "UNI-USDC", "AAVE-USDC", "SUSHI-USDC", "COMP-USDC",
            "MKR-USDC", "YFI-USDC", "BAL-USDC", "MATIC-USDC", "AVAX-USDC"
        ]

        self.min_market_cap = min_market_cap
        self.max_results = max_results

        # Rate limiting for Coinbase API
        self.request_delay = 0.5  # 0.5 second delay between requests
        self.last_request_time = 0

        logger.info("Long-Term Crypto Finder initialized with Coinbase API")

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make API request with rate limiting and retry logic."""
        for attempt in range(max_retries):
            try:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time

                if time_since_last < self.request_delay:
                    time.sleep(self.request_delay - time_since_last)

                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                self.last_request_time = time.time()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error: {str(e)}")
                    return None
            except Exception as e:
                logger.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

        return None

    def get_cryptocurrencies_to_analyze(self) -> List[Dict]:
        """
        Get cryptocurrencies to analyze using Coinbase products.

        Returns:
            List of cryptocurrency data with basic info
        """
        logger.info("Fetching cryptocurrencies for analysis using Coinbase")

        crypto_data = []
        for product_id in self.major_cryptos[:15]:  # Limit to 15 for analysis
            try:
                # Get current price from Coinbase
                current_time = datetime.now(UTC)
                start_time = current_time - timedelta(hours=1)

                candles = self.historical_data.get_historical_data(
                    product_id,
                    start_time,
                    current_time,
                    "ONE_HOUR"
                )

                if candles and len(candles) > 0:
                    current_price = float(candles[-1]['close'])

                    # Create a simplified data structure
                    crypto_info = {
                        'product_id': product_id,
                        'symbol': product_id.split('-')[0],
                        'name': self._get_crypto_name(product_id.split('-')[0]),
                        'current_price': current_price,
                        'price_change_24h': self._calculate_price_change(candles),
                        'volume_24h': self._calculate_volume(candles),
                        'market_cap': self._estimate_market_cap(product_id.split('-')[0], current_price)
                    }

                    crypto_data.append(crypto_info)
                    logger.debug(f"Retrieved data for {product_id}: ${current_price:.2f}")

                time.sleep(self.request_delay)

            except Exception as e:
                logger.warning(f"Failed to get data for {product_id}: {str(e)}")
                continue

        logger.info(f"Retrieved {len(crypto_data)} cryptocurrencies for analysis")
        return crypto_data

    def _get_crypto_name(self, symbol: str) -> str:
        """Get full name for cryptocurrency symbol."""
        name_map = {
            'BTC': 'Bitcoin',
            'ETH': 'Ethereum',
            'ADA': 'Cardano',
            'SOL': 'Solana',
            'DOT': 'Polkadot',
            'LINK': 'Chainlink',
            'UNI': 'Uniswap',
            'AAVE': 'Aave',
            'SUSHI': 'SushiSwap',
            'COMP': 'Compound',
            'MKR': 'Maker',
            'YFI': 'Yearn Finance',
            'BAL': 'Balancer',
            'MATIC': 'Polygon',
            'AVAX': 'Avalanche'
        }
        return name_map.get(symbol, symbol)

    def _calculate_price_change(self, candles: List[Dict]) -> float:
        """Calculate 24h price change from candles."""
        if len(candles) < 24:
            return 0.0

        current_price = float(candles[-1]['close'])
        price_24h_ago = float(candles[0]['close'])

        return ((current_price - price_24h_ago) / price_24h_ago) * 100

    def _calculate_volume(self, candles: List[Dict]) -> float:
        """Calculate 24h volume from candles."""
        if not candles:
            return 0.0

        total_volume = sum(float(candle.get('volume', 0)) for candle in candles)
        return total_volume

    def _estimate_market_cap(self, symbol: str, price: float) -> float:
        """Estimate market cap (simplified - would need actual circulating supply)."""
        # This is a rough estimate - in production you'd get actual supply data
        supply_estimates = {
            'BTC': 19500000,  # Approximate circulating supply
            'ETH': 120000000,
            'ADA': 35000000000,
            'SOL': 400000000,
            'DOT': 1200000000,
            'LINK': 500000000,
            'UNI': 1000000000,
            'AAVE': 16000000,
            'SUSHI': 250000000,
            'COMP': 8000000,
            'MKR': 1000000,
            'YFI': 36600,
            'BAL': 50000000,
            'MATIC': 10000000000,
            'AVAX': 400000000
        }

        supply = supply_estimates.get(symbol, 1000000)
        return price * supply

    def get_historical_data(self, product_id: str, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a cryptocurrency using Coinbase.

        Args:
            product_id: Coinbase product ID (e.g., "BTC-USDC")
            days: Number of days of historical data

        Returns:
            DataFrame with historical price data
        """
        logger.debug(f"Fetching {days} days of historical data for {product_id}")

        try:
            # Calculate time range
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=days)

            # Get daily candles from Coinbase
            candles = self.historical_data.get_historical_data(
                product_id,
                start_time,
                end_time,
                "ONE_DAY"
            )

            if not candles or len(candles) < 30:
                logger.warning(f"Insufficient historical data for {product_id}: {len(candles) if candles else 0} candles")
                return None

            # Convert to DataFrame
            df_data = []
            for candle in candles:
                df_data.append({
                    'timestamp': datetime.fromtimestamp(int(candle['start']), UTC),
                    'price': float(candle['close']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'open': float(candle['open']),
                    'volume': float(candle.get('volume', 0))
                })

            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {product_id}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators for analysis.

        Args:
            df: DataFrame with price data

        Returns:
            Dictionary of technical indicators
        """
        try:
            prices = df['price'].values
            returns = np.diff(prices) / prices[:-1]

            # Basic volatility (30-day)
            volatility_30d = np.std(returns[-30:]) * np.sqrt(365) if len(returns) >= 30 else 0

            # Sharpe ratio (annualized)
            risk_free_rate = 0.03  # 3% annual risk-free rate
            excess_returns = returns - (risk_free_rate / 365)
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if len(excess_returns) > 0 else 0

            # Sortino ratio
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(365) if len(downside_returns) > 0 else 0

            # Maximum drawdown
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            max_drawdown = np.min(drawdown)

            # RSI (simplified 14-day)
            rsi_period = 14
            if len(prices) >= rsi_period:
                gains = np.maximum(np.diff(prices), 0)
                losses = np.maximum(-np.diff(prices), 0)

                avg_gain = np.mean(gains[-rsi_period:])
                avg_loss = np.mean(losses[-rsi_period:])

                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50

            # MACD signal (simplified)
            if len(prices) >= 26:
                ema_12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
                ema_26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
                macd = ema_12 - ema_26

                if macd > 0:
                    macd_signal = "BULLISH"
                elif macd < -0.1:
                    macd_signal = "BEARISH"
                else:
                    macd_signal = "NEUTRAL"
            else:
                macd_signal = "INSUFFICIENT_DATA"

            # Bollinger Bands position
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                std_20 = np.std(prices[-20:])
                upper_band = sma_20 + (2 * std_20)
                lower_band = sma_20 - (2 * std_20)
                current_price = prices[-1]

                if current_price > upper_band:
                    bb_position = "OVERBOUGHT"
                elif current_price < lower_band:
                    bb_position = "OVERSOLD"
                else:
                    bb_position = "NEUTRAL"
            else:
                bb_position = "INSUFFICIENT_DATA"

            # Trend strength (using linear regression slope)
            if len(prices) >= 30:
                x = np.arange(len(prices[-30:]))
                slope = np.polyfit(x, prices[-30:], 1)[0]
                trend_strength = slope / np.mean(prices[-30:]) * 100  # Percentage change per day
            else:
                trend_strength = 0

            return {
                'volatility_30d': volatility_30d,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'rsi_14': rsi,
                'macd_signal': macd_signal,
                'bb_position': bb_position,
                'trend_strength': trend_strength
            }

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {
                'volatility_30d': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'rsi_14': 50,
                'macd_signal': 'ERROR',
                'bb_position': 'ERROR',
                'trend_strength': 0
            }

    def calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Calculate momentum score based on recent performance.

        Args:
            df: DataFrame with price data

        Returns:
            Momentum score (0-100)
        """
        try:
            if len(df) < 30:
                return 50  # Neutral score for insufficient data

            prices = df['price'].values

            # Recent performance (last 30 days vs previous 30 days)
            recent_30d = prices[-30:]
            previous_30d = prices[-60:-30] if len(prices) >= 60 else prices[:30]

            recent_return = (recent_30d[-1] - recent_30d[0]) / recent_30d[0]
            previous_return = (previous_30d[-1] - previous_30d[0]) / previous_30d[0] if len(previous_30d) > 1 else 0

            # Momentum score based on acceleration
            if recent_return > 0 and recent_return > previous_return:
                momentum = 75 + (recent_return - previous_return) * 25  # Strong upward momentum
            elif recent_return > 0:
                momentum = 60 + recent_return * 20  # Steady upward
            elif recent_return > previous_return:
                momentum = 40 + (recent_return - previous_return) * 30  # Improving but negative
            else:
                momentum = 20 + recent_return * 40  # Weak or declining

            return max(0, min(100, momentum))

        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return 50

    def calculate_fundamental_score(self, coin_data: Dict) -> float:
        """
        Calculate fundamental score based on various metrics.

        Args:
            coin_data: Dictionary with coin data from CoinGecko

        Returns:
            Fundamental score (0-100)
        """
        try:
            score = 0

            # Market cap score (larger cap = higher score, but not too large)
            market_cap = coin_data.get('market_cap', 0)
            if market_cap > 1000000000:  # > $1B
                market_cap_score = 80
            elif market_cap > 500000000:  # > $500M
                market_cap_score = 90
            elif market_cap > 100000000:  # > $100M
                market_cap_score = 100
            else:
                market_cap_score = 70

            score += market_cap_score * 0.3

            # Volume score (higher volume = better liquidity)
            volume_24h = coin_data.get('total_volume', 0)
            market_cap = coin_data.get('market_cap', 1)
            volume_to_market_cap = volume_24h / market_cap

            if volume_to_market_cap > 0.1:  # Very high volume
                volume_score = 100
            elif volume_to_market_cap > 0.05:
                volume_score = 90
            elif volume_to_market_cap > 0.02:
                volume_score = 80
            elif volume_to_market_cap > 0.01:
                volume_score = 70
            else:
                volume_score = 50

            score += volume_score * 0.25

            # Price stability score (lower volatility = higher score)
            price_change_7d = abs(coin_data.get('price_change_percentage_7d_in_currency', 0))
            if price_change_7d < 10:
                stability_score = 90
            elif price_change_7d < 25:
                stability_score = 70
            elif price_change_7d < 50:
                stability_score = 50
            else:
                stability_score = 30

            score += stability_score * 0.25

            # ATH distance score (further from ATH = better buying opportunity)
            current_price = coin_data.get('current_price', 0)
            ath_price = coin_data.get('ath', 0)
            if ath_price > 0:
                ath_distance = (current_price / ath_price) * 100
                if ath_distance < 20:  # Very close to ATH
                    ath_score = 30
                elif ath_distance < 50:
                    ath_score = 60
                elif ath_distance < 80:
                    ath_score = 80
                else:
                    ath_score = 100  # Deep correction = buying opportunity
            else:
                ath_score = 50

            score += ath_score * 0.2

            return min(100, score)

        except Exception as e:
            logger.error(f"Error calculating fundamental score: {str(e)}")
            return 50

    def calculate_risk_score(self, metrics: Dict) -> Tuple[float, RiskLevel]:
        """
        Calculate risk score and determine risk level.

        Args:
            metrics: Dictionary of technical and fundamental metrics

        Returns:
            Tuple of (risk_score, risk_level)
        """
        try:
            risk_score = 0

            # Volatility risk
            volatility = metrics.get('volatility_30d', 0)
            if volatility < 0.5:  # Low volatility
                vol_risk = 20
            elif volatility < 1.0:
                vol_risk = 40
            elif volatility < 1.5:
                vol_risk = 60
            elif volatility < 2.0:
                vol_risk = 80
            else:
                vol_risk = 100

            risk_score += vol_risk * 0.4

            # Drawdown risk
            max_drawdown = abs(metrics.get('max_drawdown', 0))
            if max_drawdown < 0.2:  # < 20% drawdown
                dd_risk = 20
            elif max_drawdown < 0.4:
                dd_risk = 50
            elif max_drawdown < 0.6:
                dd_risk = 70
            else:
                dd_risk = 90

            risk_score += dd_risk * 0.3

            # Sharpe ratio risk (lower Sharpe = higher risk)
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 2:  # Excellent risk-adjusted returns
                sharpe_risk = 10
            elif sharpe > 1:
                sharpe_risk = 30
            elif sharpe > 0.5:
                sharpe_risk = 50
            elif sharpe > 0:
                sharpe_risk = 70
            else:
                sharpe_risk = 90

            risk_score += sharpe_risk * 0.3

            # Determine risk level
            if risk_score < 25:
                risk_level = RiskLevel.LOW
            elif risk_score < 40:
                risk_level = RiskLevel.MEDIUM_LOW
            elif risk_score < 55:
                risk_level = RiskLevel.MEDIUM
            elif risk_score < 70:
                risk_level = RiskLevel.MEDIUM_HIGH
            elif risk_score < 85:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.VERY_HIGH

            return risk_score, risk_level

        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 50, RiskLevel.MEDIUM

    def analyze_cryptocurrency(self, coin_data: Dict) -> Optional[CryptoMetrics]:
        """
        Perform comprehensive analysis of a single cryptocurrency using Coinbase data.

        Args:
            coin_data: Dictionary with coin data from Coinbase

        Returns:
            CryptoMetrics object with complete analysis
        """
        try:
            product_id = coin_data['product_id']
            symbol = coin_data['symbol']
            name = coin_data['name']

            logger.info(f"Analyzing {symbol} ({name})")

            # Get historical data
            historical_df = self.get_historical_data(product_id, days=365)
            if historical_df is None or len(historical_df) < 30:
                logger.warning(f"Insufficient historical data for {symbol}")
                return None

            # Calculate technical indicators
            technical_metrics = self.calculate_technical_indicators(historical_df)

            # Calculate momentum score
            momentum_score = self.calculate_momentum_score(historical_df)

            # Calculate fundamental score
            fundamental_score = self.calculate_fundamental_score(coin_data)

            # Calculate technical score (weighted combination of indicators)
            technical_score = self._calculate_technical_score(technical_metrics, momentum_score)

            # Calculate risk
            risk_score, risk_level = self.calculate_risk_score({
                **technical_metrics,
                'fundamental_score': fundamental_score
            })

            # Calculate overall score (risk-adjusted)
            overall_score = (
                technical_score * 0.4 +
                fundamental_score * 0.4 +
                momentum_score * 0.2
            ) * (1 - risk_score / 200)  # Risk adjustment

            # Calculate additional price changes from historical data
            price_changes = self._calculate_price_changes_from_history(historical_df)

            # Calculate trading levels (entry, stop loss, take profit)
            trading_levels = self.calculate_trading_levels(
                historical_df,
                coin_data.get('current_price', 0),
                technical_metrics
            )

            return CryptoMetrics(
                symbol=symbol,
                name=name,
                current_price=coin_data.get('current_price', 0),
                market_cap=coin_data.get('market_cap', 0),
                market_cap_rank=0,  # Not available from Coinbase basic data
                volume_24h=coin_data.get('volume_24h', 0),
                price_change_24h=coin_data.get('price_change_24h', 0),
                price_change_7d=price_changes.get('7d', 0),
                price_change_30d=price_changes.get('30d', 0),
                ath_price=price_changes.get('ath', 0),
                ath_date='',  # Not available from basic Coinbase data
                atl_price=price_changes.get('atl', 0),
                atl_date='',  # Not available from basic Coinbase data
                volatility_30d=technical_metrics['volatility_30d'],
                sharpe_ratio=technical_metrics['sharpe_ratio'],
                sortino_ratio=technical_metrics['sortino_ratio'],
                max_drawdown=technical_metrics['max_drawdown'],
                rsi_14=technical_metrics['rsi_14'],
                macd_signal=technical_metrics['macd_signal'],
                bb_position=technical_metrics['bb_position'],
                trend_strength=technical_metrics['trend_strength'],
                momentum_score=momentum_score,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                risk_score=risk_score,
                overall_score=overall_score,
                risk_level=risk_level,
                # Trading levels
                entry_price=trading_levels['entry_price'],
                stop_loss_price=trading_levels['stop_loss_price'],
                take_profit_price=trading_levels['take_profit_price'],
                risk_reward_ratio=trading_levels['risk_reward_ratio'],
                position_size_percentage=trading_levels['position_size_percentage']
            )

        except Exception as e:
            logger.error(f"Error analyzing {coin_data.get('symbol', 'unknown')}: {str(e)}")
            return None

    def _calculate_price_changes_from_history(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price changes from historical data."""
        try:
            if len(df) < 30:
                return {'7d': 0, '30d': 0, 'ath': 0, 'atl': 0}

            current_price = df['price'].iloc[-1]

            # 7-day change
            if len(df) >= 7:
                price_7d_ago = df['price'].iloc[-7]
                change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100
            else:
                change_7d = 0

            # 30-day change
            if len(df) >= 30:
                price_30d_ago = df['price'].iloc[-30]
                change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
            else:
                change_30d = 0

            # ATH and ATL
            ath_price = df['high'].max()
            atl_price = df['low'].min()

            return {
                '7d': change_7d,
                '30d': change_30d,
                'ath': ath_price,
                'atl': atl_price
            }

        except Exception as e:
            logger.error(f"Error calculating price changes: {str(e)}")
            return {'7d': 0, '30d': 0, 'ath': 0, 'atl': 0}

    def calculate_trading_levels(self, df: pd.DataFrame, current_price: float, technical_metrics: Dict) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and take profit levels based on technical analysis.

        Args:
            df: Historical price data
            current_price: Current market price
            technical_metrics: Technical indicators

        Returns:
            Dictionary with trading levels
        """
        try:
            # Entry price (use current price as entry)
            entry_price = current_price

            # Stop Loss calculation based on multiple methods
            stop_loss_methods = []

            # Method 1: ATR-based stop loss (2x ATR below entry)
            atr = technical_metrics.get('atr', df['price'].pct_change().std() * 100)
            if atr > 0:
                atr_stop = entry_price * (1 - atr / 100 * 2)
                stop_loss_methods.append(atr_stop)

            # Method 2: Recent low support (10-day low)
            if len(df) >= 10:
                recent_low = df['low'].tail(10).min()
                support_stop = recent_low * 0.98  # 2% below recent low
                stop_loss_methods.append(support_stop)

            # Method 3: Percentage-based stop (5-8% depending on volatility)
            volatility = technical_metrics.get('volatility_30d', 0.5)
            if volatility < 0.3:
                pct_stop = entry_price * 0.92  # 8% stop for low volatility
            elif volatility < 0.6:
                pct_stop = entry_price * 0.94  # 6% stop for medium volatility
            else:
                pct_stop = entry_price * 0.95  # 5% stop for high volatility
            stop_loss_methods.append(pct_stop)

            # Choose the most conservative stop loss (highest price)
            stop_loss_price = max(stop_loss_methods) if stop_loss_methods else entry_price * 0.95

            # Take Profit calculation
            take_profit_methods = []

            # Method 1: Reward-to-risk ratio (3:1)
            risk_amount = entry_price - stop_loss_price
            rr_take_profit = entry_price + (risk_amount * 3)
            take_profit_methods.append(rr_take_profit)

            # Method 2: Recent high resistance
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                resistance_tp = recent_high * 1.02  # 2% above recent high
                take_profit_methods.append(resistance_tp)

            # Method 3: ATH-based target (scaled based on distance from ATH)
            ath_price = df['high'].max()
            ath_distance = (ath_price - current_price) / current_price

            if ath_distance < 0.2:  # Within 20% of ATH
                ath_tp = ath_price * 1.05  # 5% above ATH
            elif ath_distance < 0.5:  # Within 50% of ATH
                ath_tp = ath_price * 1.10  # 10% above ATH
            else:  # Far from ATH
                ath_tp = entry_price * 2.0  # 100% return target
            take_profit_methods.append(ath_tp)

            # Choose the most conservative take profit (lowest price)
            take_profit_price = min(take_profit_methods) if take_profit_methods else entry_price * 1.50

            # Calculate risk-reward ratio
            risk = entry_price - stop_loss_price
            reward = take_profit_price - entry_price
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Calculate position size percentage based on risk
            if risk_reward_ratio >= 3:
                position_size_percentage = 2.0  # 2% of portfolio for good R:R
            elif risk_reward_ratio >= 2:
                position_size_percentage = 1.5  # 1.5% for decent R:R
            else:
                position_size_percentage = 1.0  # 1% for lower R:R

            return {
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size_percentage': position_size_percentage
            }

        except Exception as e:
            logger.error(f"Error calculating trading levels: {str(e)}")
            # Return conservative defaults
            return {
                'entry_price': current_price,
                'stop_loss_price': current_price * 0.95,
                'take_profit_price': current_price * 1.50,
                'risk_reward_ratio': 2.0,
                'position_size_percentage': 1.0
            }

    def _calculate_technical_score(self, technical_metrics: Dict, momentum_score: float) -> float:
        """Calculate technical score from various indicators."""
        try:
            score = 0

            # RSI score
            rsi = technical_metrics['rsi_14']
            if 30 <= rsi <= 70:  # Neutral zone
                rsi_score = 70
            elif rsi < 30:  # Oversold
                rsi_score = 90
            else:  # Overbought
                rsi_score = 50

            score += rsi_score * 0.2

            # MACD score
            macd = technical_metrics['macd_signal']
            if macd == 'BULLISH':
                macd_score = 80
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:
                macd_score = 40

            score += macd_score * 0.2

            # Bollinger Bands score
            bb = technical_metrics['bb_position']
            if bb == 'OVERSOLD':
                bb_score = 85
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:  # OVERBOUGHT
                bb_score = 45

            score += bb_score * 0.15

            # Trend strength score
            trend = technical_metrics['trend_strength']
            if trend > 0.5:  # Strong upward trend
                trend_score = 90
            elif trend > 0.1:
                trend_score = 70
            elif trend > -0.1:
                trend_score = 50
            else:
                trend_score = 30

            score += trend_score * 0.25

            # Momentum score (already calculated)
            score += momentum_score * 0.2

            return min(100, score)

        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 50

    def find_best_opportunities(self, limit: int = 15) -> List[CryptoMetrics]:
        """
        Find the best long-term cryptocurrency opportunities using Coinbase.

        Args:
            limit: Number of cryptocurrencies to analyze

        Returns:
            List of CryptoMetrics objects sorted by overall score
        """
        logger.info("Starting comprehensive crypto analysis using Coinbase...")

        # Get cryptocurrencies to analyze
        crypto_list = self.get_cryptocurrencies_to_analyze()
        if not crypto_list:
            logger.error("Failed to retrieve cryptocurrency list")
            return []

        # Analyze each cryptocurrency
        analyzed_cryptos = []
        for i, coin_data in enumerate(crypto_list[:limit]):
            logger.info(f"Analyzing {i+1}/{min(len(crypto_list), limit)}: {coin_data['symbol']} ({coin_data['name']})")

            metrics = self.analyze_cryptocurrency(coin_data)
            if metrics:
                analyzed_cryptos.append(metrics)

            # Small delay to avoid overwhelming the API
            time.sleep(1.0)

        # Sort by overall score (descending)
        analyzed_cryptos.sort(key=lambda x: x.overall_score, reverse=True)

        # Return top results
        top_results = analyzed_cryptos[:self.max_results]

        logger.info(f"Analysis complete. Found {len(top_results)} top opportunities.")
        return top_results

    def print_results(self, results: List[CryptoMetrics]):
        """
        Print formatted analysis results.

        Args:
            results: List of CryptoMetrics to display
        """
        print("\n" + "="*100)
        print("LONG-TERM CRYPTO OPPORTUNITIES ANALYSIS")
        print("="*100)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total cryptocurrencies analyzed: {len(results)}")
        print("="*100)

        for i, crypto in enumerate(results, 1):
            print(f"\n{i}. {crypto.symbol} ({crypto.name})")
            print("-" * 50)
            print(f"Price: ${crypto.current_price:.6f}")
            print(f"Market Cap: ${crypto.market_cap:,.0f} (Rank #{crypto.market_cap_rank})")
            print(f"24h Volume: ${crypto.volume_24h:,.0f}")
            print(f"24h Change: {crypto.price_change_24h:.2f}%")
            print(f"7d Change: {crypto.price_change_7d:.2f}%")
            print(f"30d Change: {crypto.price_change_30d:.2f}%")
            print(f"ATH: ${crypto.ath_price:.2f} (Date: {crypto.ath_date[:10] if crypto.ath_date else 'N/A'})")
            print(f"ATL: ${crypto.atl_price:.6f} (Date: {crypto.atl_date[:10] if crypto.atl_date else 'N/A'})")
            print(f"Volatility (30d): {crypto.volatility_30d:.3f}")
            print(f"Sharpe Ratio: {crypto.sharpe_ratio:.2f}")
            print(f"Sortino Ratio: {crypto.sortino_ratio:.2f}")
            print(f"Max Drawdown: {crypto.max_drawdown:.2f}")
            print(f"RSI (14): {crypto.rsi_14:.1f}")
            print(f"MACD Signal: {crypto.macd_signal}")
            print(f"BB Position: {crypto.bb_position}")
            print(f"Trend Strength: {crypto.trend_strength:.2f}% per day")
            print(f"Momentum Score: {crypto.momentum_score:.1f}/100")
            print(f"Fundamental Score: {crypto.fundamental_score:.1f}/100")
            print(f"Technical Score: {crypto.technical_score:.1f}/100")
            print(f"Risk Score: {crypto.risk_score:.1f}/100")
            print(f"Risk Level: {crypto.risk_level.value}")
            print(f"Overall Score: {crypto.overall_score:.2f}/100")

            # Trading Levels Section
            print("")
            print("💼 TRADING LEVELS (LONG POSITION):")
            print(f"Entry Price: ${crypto.entry_price:.6f}")
            print(f"Stop Loss: ${crypto.stop_loss_price:.6f}")
            print(f"Take Profit: ${crypto.take_profit_price:.6f}")
            print(f"Risk:Reward Ratio: {crypto.risk_reward_ratio:.1f}:1")
            print(f"Recommended Position Size: {crypto.position_size_percentage:.1f}% of portfolio")

def main():
    """Main function to run the crypto opportunity finder."""
    parser = argparse.ArgumentParser(description='Find the best long-term cryptocurrency opportunities')
    parser.add_argument('--limit', type=int, default=50,
                       help='Number of top cryptocurrencies to analyze (default: 50)')
    parser.add_argument('--min-market-cap', type=int, default=100000000,
                       help='Minimum market cap in USD (default: $100M)')
    parser.add_argument('--max-results', type=int, default=20,
                       help='Maximum number of results to display (default: 20)')
    parser.add_argument('--output', type=str, choices=['console', 'json'],
                       help='Output format (default: console)')

    args = parser.parse_args()

    # Initialize the finder
    finder = LongTermCryptoFinder(
        min_market_cap=args.min_market_cap,
        max_results=args.max_results
    )

    # Find opportunities
    results = finder.find_best_opportunities(limit=args.limit)

    if not results:
        print("No opportunities found. Please check your internet connection and try again.")
        return

    # Output results
    if args.output == 'json':
        # Convert results to dictionaries for JSON serialization
        json_results = []
        for crypto in results:
            crypto_dict = {
                'symbol': crypto.symbol,
                'name': crypto.name,
                'current_price': crypto.current_price,
                'market_cap': crypto.market_cap,
                'market_cap_rank': crypto.market_cap_rank,
                'volume_24h': crypto.volume_24h,
                'price_change_24h': crypto.price_change_24h,
                'price_change_7d': crypto.price_change_7d,
                'price_change_30d': crypto.price_change_30d,
                'ath_price': crypto.ath_price,
                'ath_date': crypto.ath_date,
                'atl_price': crypto.atl_price,
                'atl_date': crypto.atl_date,
                'volatility_30d': crypto.volatility_30d,
                'sharpe_ratio': crypto.sharpe_ratio,
                'sortino_ratio': crypto.sortino_ratio,
                'max_drawdown': crypto.max_drawdown,
                'rsi_14': crypto.rsi_14,
                'macd_signal': crypto.macd_signal,
                'bb_position': crypto.bb_position,
                'trend_strength': crypto.trend_strength,
                'momentum_score': crypto.momentum_score,
                'fundamental_score': crypto.fundamental_score,
                'technical_score': crypto.technical_score,
                'risk_score': crypto.risk_score,
                'overall_score': crypto.overall_score,
                'risk_level': crypto.risk_level.value
            }
            json_results.append(crypto_dict)

        print(json.dumps(json_results, indent=2))
    else:
        finder.print_results(results)

if __name__ == "__main__":
    main()
