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
from typing import Dict, List, Optional, Tuple, Union
import json
import argparse
from dataclasses import dataclass
from enum import Enum
import os
import sys
from pathlib import Path
from functools import wraps, lru_cache
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import hashlib
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET
from historicaldata import HistoricalData

# Configuration management
@dataclass
class CryptoFinderConfig:
    """Configuration class for the crypto finder."""
    min_market_cap: int = 100000000  # $100M default
    max_results: int = 20
    max_workers: int = 4
    request_delay: float = 0.5  # seconds
    cache_ttl: int = 300  # 5 minutes
    risk_free_rate: float = 0.03  # 3% annual
    analysis_days: int = 365
    rsi_period: int = 14
    atr_period: int = 14
    stochastic_period: int = 14
    williams_period: int = 14
    cci_period: int = 20
    adx_period: int = 14
    bb_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    @classmethod
    def from_env(cls) -> 'CryptoFinderConfig':
        """Create configuration from environment variables."""
        return cls(
            min_market_cap=int(os.getenv('CRYPTO_MIN_MARKET_CAP', '100000000')),
            max_results=int(os.getenv('CRYPTO_MAX_RESULTS', '20')),
            max_workers=int(os.getenv('CRYPTO_MAX_WORKERS', '4')),
            request_delay=float(os.getenv('CRYPTO_REQUEST_DELAY', '0.5')),
            cache_ttl=int(os.getenv('CRYPTO_CACHE_TTL', '300')),
            risk_free_rate=float(os.getenv('CRYPTO_RISK_FREE_RATE', '0.03')),
            analysis_days=int(os.getenv('CRYPTO_ANALYSIS_DAYS', '365')),
            rsi_period=int(os.getenv('CRYPTO_RSI_PERIOD', '14')),
            atr_period=int(os.getenv('CRYPTO_ATR_PERIOD', '14')),
            stochastic_period=int(os.getenv('CRYPTO_STOCHASTIC_PERIOD', '14')),
            williams_period=int(os.getenv('CRYPTO_WILLIAMS_PERIOD', '14')),
            cci_period=int(os.getenv('CRYPTO_CCI_PERIOD', '20')),
            adx_period=int(os.getenv('CRYPTO_ADX_PERIOD', '14')),
            bb_period=int(os.getenv('CRYPTO_BB_PERIOD', '20')),
            macd_fast=int(os.getenv('CRYPTO_MACD_FAST', '12')),
            macd_slow=int(os.getenv('CRYPTO_MACD_SLOW', '26')),
            macd_signal=int(os.getenv('CRYPTO_MACD_SIGNAL', '9'))
        )
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'min_market_cap': self.min_market_cap,
            'max_results': self.max_results,
            'max_workers': self.max_workers,
            'request_delay': self.request_delay,
            'cache_ttl': self.cache_ttl,
            'risk_free_rate': self.risk_free_rate,
            'analysis_days': self.analysis_days,
            'rsi_period': self.rsi_period,
            'atr_period': self.atr_period,
            'stochastic_period': self.stochastic_period,
            'williams_period': self.williams_period,
            'cci_period': self.cci_period,
            'adx_period': self.adx_period,
            'bb_period': self.bb_period,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal
        }

# Configure enhanced logging with file rotation
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        RotatingFileHandler(
            log_dir / 'long_term_crypto_finder.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Custom exception classes
class CryptoAnalysisError(Exception):
    """Base exception for crypto analysis errors."""
    pass

class APIRateLimitError(CryptoAnalysisError):
    """Exception raised when API rate limit is exceeded."""
    pass

class InsufficientDataError(CryptoAnalysisError):
    """Exception raised when insufficient data is available for analysis."""
    pass

class DataValidationError(CryptoAnalysisError):
    """Exception raised when data validation fails."""
    pass

# Decorator for error handling
def handle_errors(default_return=None, log_errors=True):
    """Decorator to handle common errors in crypto analysis methods."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except APIRateLimitError as e:
                if log_errors:
                    logger.error(f"API rate limit exceeded in {func.__name__}: {str(e)}")
                raise
            except InsufficientDataError as e:
                if log_errors:
                    logger.warning(f"Insufficient data in {func.__name__}: {str(e)}")
                return default_return
            except DataValidationError as e:
                if log_errors:
                    logger.error(f"Data validation failed in {func.__name__}: {str(e)}")
                return default_return
            except Exception as e:
                if log_errors:
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator

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

    def __init__(self, config: Optional[CryptoFinderConfig] = None):
        """
        Initialize the crypto finder.

        Args:
            config: Configuration object. If None, will use environment variables or defaults.
        """
        # Load configuration
        self.config = config or CryptoFinderConfig.from_env()
        
        # Initialize Coinbase service
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)
        self.historical_data = HistoricalData(self.coinbase_service.client)

        # List of major cryptocurrencies available on Coinbase (verified product IDs)
        self.major_cryptos = [
            "BTC-USDC", "ETH-USDC", "ADA-USDC", "SOL-USDC", "DOT-USDC",
            "LINK-USDC", "UNI-USDC", "AAVE-USDC", "SUSHI-USDC", "COMP-USDC",
            "MKR-USDC", "YFI-USDC", "BAL-USDC", "MATIC-USDC", "AVAX-USDC"
        ]

        # Rate limiting for Coinbase API
        self.request_delay = self.config.request_delay
        self.last_request_time = 0

        # Caching system
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = self.config.cache_ttl
        self._cache_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self.max_workers = min(self.config.max_workers, len(self.major_cryptos))

        logger.info(f"Long-Term Crypto Finder initialized with Coinbase API")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        # Validate API credentials
        self._validate_api_credentials()

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make API request with enhanced rate limiting and retry logic."""
        if not url or not isinstance(url, str):
            raise DataValidationError("Invalid URL provided")
            
        for attempt in range(max_retries):
            try:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time

                if time_since_last < self.request_delay:
                    sleep_time = self.request_delay - time_since_last
                    logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)

                # Add request headers for better API compatibility
                headers = {
                    'User-Agent': 'CryptoFinanceToolkit/1.0',
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip, deflate'
                }
                
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers,
                    timeout=15,  # Increased timeout
                    verify=True  # Ensure SSL verification
                )
                
                # Log response details for debugging
                logger.debug(f"API request to {url} returned status {response.status_code}")
                
                response.raise_for_status()
                self.last_request_time = time.time()
                
                # Validate response is valid JSON
                try:
                    data = response.json()
                    if not isinstance(data, (dict, list)):
                        raise DataValidationError("API response is not a valid JSON object or array")
                    return data
                except json.JSONDecodeError as e:
                    raise DataValidationError(f"Invalid JSON response: {str(e)}")

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit exceeded
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                elif e.response.status_code >= 500:  # Server errors
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Server error {e.response.status_code}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"HTTP error {e.response.status_code}: {str(e)}")
                    raise APIRateLimitError(f"HTTP {e.response.status_code}: {str(e)}")
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff for timeouts
                    continue
                raise APIRateLimitError(f"Request timeout after {max_retries} attempts")
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise APIRateLimitError(f"Connection failed after {max_retries} attempts")
                
            except Exception as e:
                logger.error(f"Unexpected API request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise CryptoAnalysisError(f"API request failed after {max_retries} attempts: {str(e)}")

                return None

    def _validate_api_credentials(self) -> None:
        """Validate that API credentials are properly configured."""
        try:
            if not API_KEY or not API_SECRET:
                raise DataValidationError("API credentials not found in config")
            
            # Test API connection with a simple request
            test_url = "https://api.coinbase.com/v2/time"
            response = requests.get(test_url, timeout=5)
            response.raise_for_status()
            
            logger.info("API credentials validated successfully")
            
        except Exception as e:
            logger.error(f"API credential validation failed: {str(e)}")
            raise DataValidationError(f"Invalid API configuration: {str(e)}")
    
    def _validate_crypto_data(self, data: Dict) -> bool:
        """Validate cryptocurrency data structure and values."""
        required_fields = ['product_id', 'symbol', 'name', 'current_price']
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field '{field}' in crypto data")
                return False
        
        # Validate price is positive
        if not isinstance(data.get('current_price'), (int, float)) or data['current_price'] <= 0:
            logger.warning(f"Invalid price value: {data.get('current_price')}")
            return False
            
        # Validate symbol format
        symbol = data.get('symbol', '')
        if not symbol or not isinstance(symbol, str) or len(symbol) < 2:
            logger.warning(f"Invalid symbol format: {symbol}")
            return False
            
        return True
    
    def _sanitize_price(self, price: Union[str, float, int]) -> float:
        """Sanitize and validate price values."""
        try:
            if isinstance(price, str):
                price = float(price.replace(',', '').replace('$', ''))
            elif not isinstance(price, (int, float)):
                raise ValueError(f"Invalid price type: {type(price)}")
                
            if price < 0:
                raise ValueError(f"Negative price not allowed: {price}")
                
            return float(price)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Price sanitization failed: {str(e)}")
            return 0.0
    
    def _get_cache_key(self, data: str) -> str:
        """Generate a cache key from data string."""
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Retrieve data from cache if valid."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if not cache_file.exists():
                return None

            # Check if cache is still valid
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age > self.cache_ttl:
                cache_file.unlink()  # Remove expired cache
                return None
                
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
            return None
    
    def _set_cached_data(self, cache_key: str, data: Dict) -> None:
        """Store data in cache."""
        try:
            with self._cache_lock:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")
    
    def _make_cached_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with caching."""
        # Create cache key from URL and params
        cache_data = f"{url}:{json.dumps(params or {}, sort_keys=True)}"
        cache_key = self._get_cache_key(cache_data)
        
        # Try to get from cache first
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for {url}")
            return cached_data
        
        # Make API request
        data = self._make_request(url, params)
        if data is not None:
            self._set_cached_data(cache_key, data)
            logger.debug(f"Cached data for {url}")
        
        return data
    
    def _process_single_crypto(self, product_id: str, index: int, total: int) -> Optional[Dict]:
        """Process a single cryptocurrency in parallel."""
        try:
            logger.info(f"Processing {index+1}/{total}: {product_id}")
            
            # Get current price from Coinbase
            current_time = datetime.now(UTC)
            start_time = current_time - timedelta(hours=1)

            candles = self.historical_data.get_historical_data(
                product_id,
                start_time,
                current_time,
                "ONE_HOUR"
                )

            if not candles or len(candles) == 0:
                logger.warning(f"No candle data available for {product_id}")
                return None
            
            # Validate candle data
            latest_candle = candles[-1]
            required_candle_fields = ['close', 'high', 'low', 'open']
            for field in required_candle_fields:
                if field not in latest_candle:
                    logger.warning(f"Missing {field} in candle data for {product_id}")
                    return None
                    
            current_price = self._sanitize_price(latest_candle['close'])
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {product_id}: {current_price}")
                return None

            # Get accurate ATH/ATL data from CoinGecko
            ath_data = self._get_ath_atl_from_coingecko(product_id.split('-')[0])

            # Create a validated data structure
            crypto_info = {
                'product_id': product_id,
                'symbol': product_id.split('-')[0],
                'name': self._get_crypto_name(product_id.split('-')[0]),
                'current_price': current_price,
                'price_change_24h': self._calculate_price_change(candles),
                'volume_24h': self._calculate_volume(candles),
                'market_cap': self._estimate_market_cap(product_id.split('-')[0], current_price),
                'ath_price': self._sanitize_price(ath_data.get('ath', 0)),
                'ath_date': ath_data.get('ath_date', ''),
                'atl_price': self._sanitize_price(ath_data.get('atl', 0)),
                'atl_date': ath_data.get('atl_date', '')
            }

            # Validate the complete crypto info
            if self._validate_crypto_data(crypto_info):
                logger.debug(f"Successfully processed {product_id}: ${current_price:.2f}")
                return crypto_info
            else:
                logger.warning(f"Data validation failed for {product_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to process {product_id}: {str(e)}")
            return None

    @handle_errors(default_return=[], log_errors=True)
    def get_cryptocurrencies_to_analyze(self) -> List[Dict]:
        """
        Get cryptocurrencies to analyze using Coinbase products with enhanced ATH/ATL data.

        Returns:
            List of cryptocurrency data with basic info
        """
        logger.info("Fetching cryptocurrencies for analysis using Coinbase with parallel processing")

        crypto_data = []
        failed_products = []
        products_to_process = self.major_cryptos[:15]  # Limit to 15 for analysis
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_product = {
                executor.submit(self._process_single_crypto, product_id, i, len(products_to_process)): product_id
                for i, product_id in enumerate(products_to_process)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_product):
                product_id = future_to_product[future]
                try:
                    result = future.result()
                    if result is not None:
                        crypto_data.append(result)
                    else:
                        failed_products.append(product_id)
                except Exception as e:
                            logger.error(f"Exception in parallel processing for {product_id}: {str(e)}")
                            failed_products.append(product_id)

        if failed_products:
            logger.warning(f"Failed to retrieve data for {len(failed_products)} products: {failed_products}")
            
        logger.info(f"Successfully retrieved {len(crypto_data)} cryptocurrencies for analysis")
        
        if len(crypto_data) == 0:
            raise InsufficientDataError("No valid cryptocurrency data retrieved")
            
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
            risk_free_rate = self.config.risk_free_rate
            excess_returns = returns - (risk_free_rate / 365)
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365) if len(excess_returns) > 0 else 0

            # Sortino ratio
            downside_returns = excess_returns[excess_returns < 0]
            sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(365) if len(downside_returns) > 0 else 0

            # Maximum drawdown
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            max_drawdown = np.min(drawdown)

            # RSI (configurable period)
            rsi_period = self.config.rsi_period
            if len(prices) >= rsi_period:
                gains = np.maximum(np.diff(prices), 0)
                losses = np.maximum(-np.diff(prices), 0)

                avg_gain = np.mean(gains[-rsi_period:])
                avg_loss = np.mean(losses[-rsi_period:])

                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50

            # MACD signal (configurable)
            if len(prices) >= self.config.macd_slow:
                ema_fast = pd.Series(prices).ewm(span=self.config.macd_fast).mean().iloc[-1]
                ema_slow = pd.Series(prices).ewm(span=self.config.macd_slow).mean().iloc[-1]
                macd = ema_fast - ema_slow

                if macd > 0:
                    macd_signal = "BULLISH"
                elif macd < -0.1:
                    macd_signal = "BEARISH"
                else:
                    macd_signal = "NEUTRAL"
            else:
                macd_signal = "INSUFFICIENT_DATA"

            # Bollinger Bands position
            if len(prices) >= self.config.bb_period:
                sma_bb = np.mean(prices[-self.config.bb_period:])
                std_bb = np.std(prices[-self.config.bb_period:])
                upper_band = sma_bb + (2 * std_bb)
                lower_band = sma_bb - (2 * std_bb)
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

            # Additional advanced indicators
            atr = self._calculate_atr(df, self.config.atr_period) if len(df) >= self.config.atr_period else 0
            stochastic = self._calculate_stochastic(df, self.config.stochastic_period) if len(df) >= self.config.stochastic_period else 50
            williams_r = self._calculate_williams_r(df, self.config.williams_period) if len(df) >= self.config.williams_period else -50
            cci = self._calculate_cci(df, self.config.cci_period) if len(df) >= self.config.cci_period else 0
            adx = self._calculate_adx(df, self.config.adx_period) if len(df) >= self.config.adx_period else 0
            obv = self._calculate_obv(df) if len(df) >= 2 else 0
            
            # Volume indicators
            volume_sma = df['volume'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else 0
            current_volume = df['volume'].iloc[-1] if len(df) > 0 else 0
            volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1

            return {
                'volatility_30d': volatility_30d,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'rsi_14': rsi,
                'macd_signal': macd_signal,
                'bb_position': bb_position,
                'trend_strength': trend_strength,
                'atr': atr,
                'stochastic': stochastic,
                'williams_r': williams_r,
                'cci': cci,
                'adx': adx,
                'obv': obv,
                'volume_ratio': volume_ratio
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
                'trend_strength': 0,
                'atr': 0,
                'stochastic': 50,
                'williams_r': -50,
                'cci': 0,
                'adx': 0,
                'obv': 0,
                'volume_ratio': 1
            }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)."""
        try:
            if len(df) < period + 1:
                return 0.0
                
            high = df['high'].values
            low = df['low'].values
            close = df['price'].values
            
            tr_list = []
            for i in range(1, len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr_list.append(max(tr1, tr2, tr3))
            
            if len(tr_list) >= period:
                return np.mean(tr_list[-period:])
            return 0.0
            
        except Exception as e:
            logger.warning(f"ATR calculation error: {str(e)}")
            return 0.0
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> float:
        """Calculate Stochastic Oscillator %K."""
        try:
            if len(df) < k_period:
                return 50.0
                
            high = df['high'].values[-k_period:]
            low = df['low'].values[-k_period:]
            close = df['price'].values[-1]
            
            lowest_low = np.min(low)
            highest_high = np.max(high)
            
            if highest_high == lowest_low:
                return 50.0
                
            k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            return k_percent
            
        except Exception as e:
            logger.warning(f"Stochastic calculation error: {str(e)}")
            return 50.0
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R."""
        try:
            if len(df) < period:
                return -50.0
                
            high = df['high'].values[-period:]
            low = df['low'].values[-period:]
            close = df['price'].values[-1]
            
            highest_high = np.max(high)
            lowest_low = np.min(low)
            
            if highest_high == lowest_low:
                return -50.0
                
            williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
            return williams_r
            
        except Exception as e:
            logger.warning(f"Williams %R calculation error: {str(e)}")
            return -50.0
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Commodity Channel Index (CCI)."""
        try:
            if len(df) < period:
                return 0.0
                
            high = df['high'].values[-period:]
            low = df['low'].values[-period:]
            close = df['price'].values[-period:]
            
            # Typical Price
            tp = (high + low + close) / 3
            
            # Simple Moving Average of TP
            sma_tp = np.mean(tp)
            
            # Mean Deviation
            mean_dev = np.mean(np.abs(tp - sma_tp))
            
            if mean_dev == 0:
                return 0.0
                
            cci = (tp[-1] - sma_tp) / (0.015 * mean_dev)
            return cci
            
        except Exception as e:
            logger.warning(f"CCI calculation error: {str(e)}")
            return 0.0
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX) - simplified version."""
        try:
            if len(df) < period + 1:
                return 0.0
                
            high = df['high'].values
            low = df['low'].values
            close = df['price'].values
            
            # Calculate True Range and Directional Movement
            tr_list = []
            dm_plus_list = []
            dm_minus_list = []
            
            for i in range(1, len(high)):
                # True Range
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr_list.append(max(tr1, tr2, tr3))
                
                # Directional Movement
                dm_plus = high[i] - high[i-1] if high[i] - high[i-1] > low[i-1] - low[i] else 0
                dm_minus = low[i-1] - low[i] if low[i-1] - low[i] > high[i] - high[i-1] else 0
                
                dm_plus_list.append(max(dm_plus, 0))
                dm_minus_list.append(max(dm_minus, 0))
            
            if len(tr_list) >= period:
                # Smoothed averages
                atr = np.mean(tr_list[-period:])
                di_plus = np.mean(dm_plus_list[-period:]) / atr * 100 if atr > 0 else 0
                di_minus = np.mean(dm_minus_list[-period:]) / atr * 100 if atr > 0 else 0
                
                # ADX calculation
                dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) > 0 else 0
                return dx
                
            return 0.0
            
        except Exception as e:
            logger.warning(f"ADX calculation error: {str(e)}")
            return 0.0
    
    def _calculate_obv(self, df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume (OBV)."""
        try:
            if len(df) < 2:
                return 0.0
                
            close = df['price'].values
            volume = df['volume'].values
            
            obv = 0
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv += volume[i]
                elif close[i] < close[i-1]:
                    obv -= volume[i]
                # If close[i] == close[i-1], OBV remains unchanged
            
            return obv
            
        except Exception as e:
            logger.warning(f"OBV calculation error: {str(e)}")
            return 0.0

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
                ath_price=coin_data.get('ath_price', price_changes.get('ath', 0)),
                ath_date=coin_data.get('ath_date', ''),
                atl_price=coin_data.get('atl_price', price_changes.get('atl', 0)),
                atl_date=coin_data.get('atl_date', ''),
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

    def _get_ath_atl_from_coingecko(self, symbol: str) -> Dict[str, Union[float, str]]:
        """
        Get accurate ATH/ATL data from CoinGecko API.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'btc', 'eth')

        Returns:
            Dictionary with ATH/ATL data
        """
        try:
            # Map Coinbase symbols to CoinGecko IDs
            coingecko_id_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOT': 'polkadot',
                'LINK': 'chainlink',
                'UNI': 'uniswap',
                'AAVE': 'aave',
                'SUSHI': 'sushi',
                'COMP': 'compound-governance-token',
                'MKR': 'maker',
                'YFI': 'yearn-finance',
                'BAL': 'balancer',
                'MATIC': 'matic-network',
                'AVAX': 'avalanche-2'
            }

            coingecko_id = coingecko_id_map.get(symbol.upper(), symbol.lower())

            # Fetch data from CoinGecko with caching
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
            try:
                data = self._make_cached_request(url)
            except Exception as e:
                logger.warning(f"Failed to fetch CoinGecko data for {symbol}: {str(e)}")
                return {
                    'ath': 0,
                    'ath_date': '',
                    'atl': 0,
                    'atl_date': ''
                }
            
            if data is None:
                logger.warning(f"No data returned from CoinGecko for {symbol}")
                return {
                    'ath': 0,
                    'ath_date': '',
                    'atl': 0,
                    'atl_date': ''
                }

            # Extract ATH/ATL data
            market_data = data.get('market_data', {})

            ath_price = market_data.get('ath', {}).get('usd', 0)
            ath_date = market_data.get('ath_date', {}).get('usd', '')
            atl_price = market_data.get('atl', {}).get('usd', 0)
            atl_date = market_data.get('atl_date', {}).get('usd', '')

            return {
                'ath': float(ath_price) if ath_price else 0,
                'ath_date': ath_date[:10] if ath_date else '',
                'atl': float(atl_price) if atl_price else 0,
                'atl_date': atl_date[:10] if atl_date else ''
            }

        except Exception as e:
            logger.warning(f"Failed to get ATH/ATL data for {symbol}: {str(e)}")
            # Return fallback data
            return {
                'ath': 0,
                'ath_date': '',
                'atl': 0,
                'atl_date': ''
            }

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
        top_results = analyzed_cryptos[:self.config.max_results]

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
            print(f"ATH: ${crypto.ath_price:.2f} (Date: {crypto.ath_date or 'N/A'})")
            print(f"ATL: ${crypto.atl_price:.6f} (Date: {crypto.atl_date or 'N/A'})")
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

    # Create configuration
    config = CryptoFinderConfig(
        min_market_cap=args.min_market_cap,
        max_results=args.max_results
    )
    
    # Initialize the finder
    finder = LongTermCryptoFinder(config=config)

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
