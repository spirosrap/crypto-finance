#!/usr/bin/env python3
"""
Long-Term Crypto Opportunity Finder

This program analyzes cryptocurrencies to find the best long-term investment opportunities
by evaluating multiple factors including technical indicators, fundamental metrics,
risk assessment, and market sentiment.

Author: Crypto Finance Toolkit
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:  # Py3.11+
    from datetime import UTC  # type: ignore
except Exception:  # Py<=3.10
    from datetime import timezone as _tz  # type: ignore
    UTC = _tz.utc  # type: ignore
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
    # New configurable options
    side: str = "both"  # one of: long, short, both
    unique_by_symbol: bool = False  # keep only the best side per symbol
    min_overall_score: float = 0.0  # filter out weak candidates
    offline: bool = False  # avoid external HTTP where possible
    symbols: Optional[List[str]] = None  # analyze explicit symbols if provided
    top_per_side: Optional[int] = None  # when set, cap results per side
    quotes: Optional[List[str]] = None  # preferred quote currencies, e.g., ["USDC","USD","USDT"]
    
    @classmethod
    def from_env(cls) -> 'CryptoFinderConfig':
        """Create configuration from environment variables."""
        # Parse optional list of symbols from env (comma-separated)
        symbols_env = os.getenv('CRYPTO_SYMBOLS')
        symbols_list: Optional[List[str]] = None
        if symbols_env:
            symbols_list = [s.strip().upper() for s in symbols_env.split(',') if s.strip()]
        quotes_env = os.getenv('CRYPTO_QUOTES')
        quotes_list: Optional[List[str]] = None
        if quotes_env:
            quotes_list = [q.strip().upper() for q in quotes_env.split(',') if q.strip()]

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
            macd_signal=int(os.getenv('CRYPTO_MACD_SIGNAL', '9')),
            side=os.getenv('CRYPTO_SIDE', 'both').lower(),
            unique_by_symbol=os.getenv('CRYPTO_UNIQUE_BY_SYMBOL', '0') in ('1', 'true', 'True'),
            min_overall_score=float(os.getenv('CRYPTO_MIN_SCORE', '0')),
            offline=os.getenv('CRYPTO_OFFLINE', '0') in ('1', 'true', 'True'),
            symbols=symbols_list,
            top_per_side=int(os.getenv('CRYPTO_TOP_PER_SIDE')) if os.getenv('CRYPTO_TOP_PER_SIDE') else None
            ,quotes=quotes_list
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
            'macd_signal': self.macd_signal,
            'side': self.side,
            'unique_by_symbol': self.unique_by_symbol,
            'min_overall_score': self.min_overall_score,
            'top_per_side': self.top_per_side,
            'quotes': self.quotes,
            'offline': self.offline
        }

# Configure enhanced logging with file rotation
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import uuid

def _setup_logging() -> logging.Logger:
    """Configure module-level logging without hijacking root handlers."""
    base_logs_dir = Path('logs') / 'long_term_crypto_finder'
    base_logs_dir.mkdir(parents=True, exist_ok=True)

    log_level_str = os.getenv('CRYPTO_FINDER_LOG_LEVEL', 'INFO').upper()
    log_to_console = os.getenv('CRYPTO_FINDER_LOG_TO_CONSOLE', '1') not in ('0', 'false', 'False')
    histdata_verbose = os.getenv('CRYPTO_FINDER_HISTDATA_VERBOSE', '0') in ('1', 'true', 'True')
    backups = int(os.getenv('CRYPTO_FINDER_LOG_RETENTION', '14'))

    level = getattr(logging, log_level_str, logging.INFO)
    run_id = uuid.uuid4().hex[:8]

    file_handler = TimedRotatingFileHandler(
        base_logs_dir / 'long_term_crypto_finder.log', when='midnight', backupCount=backups, encoding='utf-8'
    )
    size_handler = RotatingFileHandler(
        base_logs_dir / 'long_term_crypto_finder.size.log', maxBytes=10 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )

    class _ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.run_id = run_id
            return True

    context_filter = _ContextFilter()
    file_handler.addFilter(context_filter)
    size_handler.addFilter(context_filter)

    fmt = '%(asctime)s - %(run_id)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    size_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.propagate = False

    # Prepare console handler if requested
    console = None
    if log_to_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console.addFilter(context_filter)

    # Add handlers only once to avoid duplication on re-imports
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(size_handler)
        if console is not None:
            logger.addHandler(console)

    # Tune verbosity of noisy modules
    logging.getLogger('historicaldata').setLevel(logging.INFO if histdata_verbose else logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    logger.info(f"Logging initialized (level={log_level_str}, run_id={run_id}, retention={backups}d)")
    logger.info(f"Logs directory: {base_logs_dir}")
    return logger

# Initialize logging once for this module
logger = _setup_logging()


def _finite(x: Union[float, int], default: float = 0.0) -> float:
    """Coerce non-finite numbers to a default float for JSON safety."""
    try:
        xv = float(x)
        return xv if np.isfinite(xv) else float(default)
    except Exception:
        return float(default)

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

class PositionSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class CryptoMetrics:
    """Data class to hold comprehensive crypto metrics."""
    symbol: str
    name: str
    # Position side for which trading levels and scores are computed
    position_side: str
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
    # Data timestamp for integrity
    data_timestamp_utc: str = ""

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
        
        # Load API credentials from environment (fallback to config.py only if needed)
        self.api_key, self.api_secret = self._load_api_credentials()

        # Initialize Coinbase service
        self.coinbase_service = CoinbaseService(self.api_key, self.api_secret)
        self.historical_data = HistoricalData(self.coinbase_service.client)

        # HTTP session with retries + backoff
        self._sess = requests.Session()
        self._sess.headers.update({
            'User-Agent': 'CryptoFinanceToolkit/1.0',
            'Accept': 'application/json',
        })
        self._sess.mount(
            'https://',
            HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]))
        )
        # Thread-safety: serialize request timing updates
        self._req_lock = threading.Lock()
        # Thread-local session holder
        self._tls = threading.local()

        # Dynamic product mapping (symbol -> product_id) populated lazily
        self._symbol_to_product: Dict[str, str] = {}
        # Lazy CoinGecko list cache in-memory
        self._coingecko_list: Optional[List[Dict[str, str]]] = None
        # Fast lookup index for CoinGecko symbol->record
        self._cg_index: Optional[Dict[str, Dict[str, str]]] = None
        self._cg_id_cache: Dict[str, str] = {}  # memo for symbol -> coingecko_id

        # Rate limiting for Coinbase API
        self.request_delay = self.config.request_delay
        self.last_request_time = 0

        # Caching system
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = self.config.cache_ttl
        self._cache_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self.max_workers = max(1, self.config.max_workers)

        # Behavior flags
        self.offline = bool(self.config.offline)
        # Normalize side
        self.side = (self.config.side or 'both').lower()
        if self.side not in ('long', 'short', 'both'):
            self.side = 'both'

        logger.info(f"Long-Term Crypto Finder initialized with Coinbase API")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        # Validate API credentials
        self._validate_api_credentials()

    def _throttle(self) -> None:
        """Apply a simple global throttle for outbound calls (thread-safe, monotonic clock)."""
        with self._req_lock:
            now = time.monotonic()
            last = getattr(self, "_last_tick", 0.0)
            wait = max(0.0, self.request_delay - (now - last))
            if wait > 0:
                time.sleep(wait)
            self._last_tick = time.monotonic()

    def _load_api_credentials(self) -> Tuple[Optional[str], Optional[str]]:
        """Load API credentials from environment or optional .env; fallback to config.py if present.

        Returns:
            Tuple of (api_key, api_secret) which may be None if not configured.
        """
        # Attempt to load from .env if python-dotenv is available
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            # Best-effort: ignore if dotenv is not installed
            pass

        env_key = os.getenv('API_KEY')
        env_secret = os.getenv('API_SECRET')

        if env_key and env_secret:
            return env_key, env_secret

        # Fallback to config.py if available
        try:
            from config import API_KEY as CFG_API_KEY, API_SECRET as CFG_API_SECRET  # type: ignore

            return CFG_API_KEY, CFG_API_SECRET
        except Exception:
            logger.warning("API credentials not found in environment or config.py; public endpoints may still work.")
            return None, None

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Make API request via shared Session with retries and thread-safe throttle."""
        if not url or not isinstance(url, str):
            raise DataValidationError("Invalid URL provided")
            
        for attempt in range(max_retries):
            try:
                # Global throttle
                self._throttle()
                # Use a thread-local session to avoid cross-thread state issues
                sess = getattr(self._tls, 'session', None)
                if sess is None:
                    sess = requests.Session()
                    sess.headers.update(self._sess.headers)
                    sess.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
                    self._tls.session = sess
                response = sess.get(url, params=params, timeout=15, verify=True)

                # Log response details for debugging
                logger.debug(f"API request to {url} returned status {response.status_code}")
                
                response.raise_for_status()
                
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
                    # Validation or client errors: surface server message and do not retry
                    server_msg = None
                    try:
                        server_msg = e.response.json()
                    except Exception:
                        server_msg = e.response.text
                    logger.error(f"HTTP {e.response.status_code} error: {server_msg}")
                    raise DataValidationError(f"HTTP {e.response.status_code}: {server_msg}")
                    
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


    def _validate_api_credentials(self) -> None:
        """Lightweight presence check; do not block public/offline usage."""
        if getattr(self, 'offline', False) or os.getenv("CRYPTO_PUBLIC_ONLY") in ("1", "true", "True"):
            logger.info("Public-only/offline: skipping API credential presence check")
            return
        if not self.api_key or not self.api_secret:
            logger.warning("API credentials not found; continuing with public endpoints only")
            return
        logger.info("API credentials present (no on-start auth call performed)")
    
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

    def _fetch_usdc_products(self) -> Dict[str, Dict[str, str]]:
        """Fetch Coinbase products and return mapping symbol -> {product_id, base_name} filtered by preferred quotes, status online.

        Gracefully falls back to a small static set if auth/public endpoint fails.
        """
        try:
            data = self._make_cached_request(
                "https://api.coinbase.com/api/v3/brokerage/products",
                {"limit": 250}
            ) or {}
            products = data.get("products", []) if isinstance(data, dict) else []
        except Exception:
            products = []
        result: Dict[str, Dict[str, str]] = {}
        # Determine preferred quotes from config/env
        quotes = None
        try:
            quotes = [q.strip().upper() for q in (self.config.quotes or []) if q]
        except Exception:
            quotes = None
        if not quotes:
            env_q = os.getenv('CRYPTO_QUOTES', 'USDC')
            quotes = [q.strip().upper() for q in env_q.split(',') if q.strip()]
        quotes_set = set(quotes or ['USDC'])
        for p in products:
            if p.get("status") == "online" and (p.get("quote_currency") or '').upper() in quotes_set:
                base = (p.get("base_currency") or "").upper()
                if base:
                    result[base] = {
                        "product_id": p.get("product_id"),
                        "base_name": p.get("base_name", base)
                    }
        if not result:
            for sym in ["BTC", "ETH", "SOL", "ADA", "MATIC", "AVAX", "DOT", "LINK"]:
                result[sym] = {"product_id": f"{sym}-USDC", "base_name": sym}
        self._symbol_to_product = {k: v["product_id"] for k, v in result.items() if v.get("product_id")}
        return result

    def _load_coingecko_list(self) -> List[Dict[str, str]]:
        """Load CoinGecko coins list (id,symbol,name) with disk JSON cache."""
        if self._coingecko_list is not None:
            return self._coingecko_list

        cache_path = self.cache_dir / "coingecko_coins_list.json"
        # Try cache first with its own longer TTL (e.g., 7 days)
        long_ttl = float(os.getenv('CRYPTO_COINGECKO_LIST_TTL_SEC', str(7 * 24 * 3600)))
        try:
            if cache_path.exists():
                with open(cache_path, 'r', encoding='utf-8') as f:
                    envelope = json.load(f)
                if isinstance(envelope, dict) and envelope.get('v') == 1:
                    ts = float(envelope.get('ts', 0))
                    data = envelope.get('data', []) or []
                    if long_ttl <= 0 or (time.time() - ts) <= long_ttl:
                        self._coingecko_list = data
                        return self._coingecko_list
        except Exception:
            pass

        if self.offline:
            self._coingecko_list = []
            return self._coingecko_list

        # Fetch and cache
        try:
            url = "https://api.coingecko.com/api/v3/coins/list"
            data = self._make_request(url)
            if isinstance(data, list):
                self._coingecko_list = [{
                    'id': str(x.get('id','')),
                    'symbol': str(x.get('symbol','')).upper(),
                    'name': str(x.get('name',''))
                } for x in data]
                # Build fast index: SYMBOL -> {id, symbol, name}
                try:
                    self._cg_index = {item['symbol'].upper(): item for item in self._coingecko_list if item.get('symbol')}
                except Exception:
                    self._cg_index = {}
                envelope = {'v': 1, 'ts': time.time(), 'data': self._coingecko_list}
                self._atomic_write_json(cache_path, envelope)
            else:
                self._coingecko_list = []
        except Exception as e:
            logger.warning(f"Failed to fetch CoinGecko list: {e}")
            self._coingecko_list = []
        return self._coingecko_list

    def _coingecko_id_for_symbol(self, symbol: str) -> Optional[str]:
        """Resolve CoinGecko id from symbol. Prefer highest-market-cap id on collisions."""
        sym = (symbol or '').upper()
        # memo hit
        if sym in self._cg_id_cache:
            return self._cg_id_cache[sym]
        lst = self._load_coingecko_list()
        # fast path via index (exact symbol match)
        if self._cg_index and sym in self._cg_index:
            cid = self._cg_index[sym].get('id')
            if cid:
                self._cg_id_cache[sym] = cid
                return cid
        candidates = []
        for item in lst:
            try:
                if (item.get('symbol') or '').upper() == sym and item.get('id'):
                    candidates.append(item['id'])
            except Exception:
                continue
        if not candidates:
            # Fallback hard map for common assets
            cid = {
                'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano', 'SOL': 'solana', 'DOT': 'polkadot',
                'LINK': 'chainlink', 'UNI': 'uniswap', 'AAVE': 'aave', 'SUSHI': 'sushi', 'COMP': 'compound-governance-token',
                'MKR': 'maker', 'YFI': 'yearn-finance', 'BAL': 'balancer', 'MATIC': 'matic-network', 'AVAX': 'avalanche-2'
            }.get(sym)
            if cid:
                self._cg_id_cache[sym] = cid
            return cid
        if len(candidates) == 1 or getattr(self, 'offline', False):
            self._cg_id_cache[sym] = candidates[0]
            return self._cg_id_cache[sym]
        # When online, query markets to pick the largest by market cap
        try:
            markets = self._cg_markets(candidates)
            if isinstance(markets, list) and markets:
                markets.sort(key=lambda m: float(m.get('market_cap') or 0), reverse=True)
                cid = str(markets[0].get('id') or candidates[0])
                self._cg_id_cache[sym] = cid
                return cid
        except Exception:
            pass
        self._cg_id_cache[sym] = candidates[0]
        return self._cg_id_cache[sym]

    def _cg_markets(self, ids: List[str]) -> List[Dict]:
        """Fetch CoinGecko markets snapshot for given ids (batched)."""
        if not ids:
            return []
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(ids[:250]),
            "price_change_percentage": "7d,30d"
        }
        data = self._make_cached_request(url, params)
        return data if isinstance(data, list) else []
    
    def _get_cache_key(self, data: str) -> str:
        """Generate a cache key from data string."""
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_cached_data(self, cache_key: str) -> Optional[Union[Dict, List]]:
        """Retrieve data from JSON cache if valid (portable format)."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if not cache_file.exists():
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                envelope = json.load(f)

            if not isinstance(envelope, dict) or envelope.get('v') != 1:
                # Unknown cache version; ignore
                return None

            ts = float(envelope.get('ts', 0))
            if self.cache_ttl > 0 and (time.time() - ts) > self.cache_ttl:
                try:
                    cache_file.unlink()
                except Exception:
                    pass
                return None

            return envelope.get('data')
        except Exception as e:
            logger.warning(f"Cache read error: {str(e)}")
            return None
    
    def _set_cached_data(self, cache_key: str, data: Union[Dict, List]) -> None:
        """Store data in JSON cache using atomic write (portable and safe)."""
        try:
            with self._cache_lock:
                cache_file = self.cache_dir / f"{cache_key}.json"
                envelope = {'v': 1, 'ts': time.time(), 'data': data}
                self._atomic_write_json(cache_file, envelope)
        except Exception as e:
            logger.warning(f"Cache write error: {str(e)}")

    # -------- Persistence helpers (atomic write + file lock) --------
    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        """Atomically write bytes to a file with an exclusive, best-effort lock."""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{int(time.time()*1000)}")
        lock_path = path.with_suffix(path.suffix + '.lock')

        # Acquire a best-effort lock on lock_path
        try:
            import fcntl  # type: ignore

            lock_fd = open(lock_path, 'w')
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                with open(tmp, 'wb') as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            finally:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_fd.close()
        except Exception:
            # Fallback: no fcntl available; still write atomically via os.replace
            with open(tmp, 'wb') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)

    def _atomic_write_json(self, path: Path, obj: Union[Dict, List]) -> None:
        """Atomically write a JSON object (utf-8, indented)."""
        data = json.dumps(obj, indent=2).encode('utf-8')
        self._atomic_write_bytes(Path(path), data)
    
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
        
        # If offline, don't attempt network
        if getattr(self, 'offline', False):
            logger.info(f"Offline mode: skipping network request for {url}")
            return None

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
            
            # Get last 24 hours of hourly candles from Coinbase
            # This supports accurate 24h change and 24h volume calculations
            current_time = datetime.now(UTC)
            start_time = current_time - timedelta(hours=24)

            # Throttle Coinbase hourly candles request
            self._throttle()
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
            # Data timestamp from latest candle
            try:
                ts_raw = latest_candle.get('start') or latest_candle.get('time') or 0
                # normalize int and handle ms vs s
                ts = int(ts_raw)
                # if looks like milliseconds (>= 10^12), convert to seconds
                if ts >= 10**12:
                    ts //= 1000
            except Exception:
                ts = 0
            try:
                data_ts = datetime.fromtimestamp(ts, UTC).strftime('%Y-%m-%d %H:%M:%SZ') if ts else ''
            except Exception:
                data_ts = ''
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {product_id}: {current_price}")
                return None

            # Get accurate ATH/ATL and market snapshot from CoinGecko
            cg = self._get_ath_atl_from_coingecko(product_id.split('-')[0])

            # Create a validated data structure
            real_mc = float(cg.get('market_cap_usd', 0) or 0.0) > 0.0
            crypto_info = {
                'product_id': product_id,
                'symbol': product_id.split('-')[0],
                'name': self._get_crypto_name(product_id.split('-')[0]),
                'current_price': current_price,
                'price_change_24h': self._calculate_price_change(candles),
                # Prefer CoinGecko USD volume; fall back to estimated USD from candles
                'volume_24h': (
                    self._sanitize_price(cg.get('total_volume_usd', 0))
                    or self._calculate_usd_volume(candles)
                ),
                # Prefer CoinGecko market cap if available (estimate only for display, not filtering)
                'market_cap': self._sanitize_price(cg.get('market_cap_usd', 0)) or self._estimate_market_cap(product_id.split('-')[0], current_price),
                'market_cap_is_real': real_mc,
                'market_cap_rank': int(cg.get('market_cap_rank', 0) or 0),
                'total_volume': self._sanitize_price(cg.get('total_volume_usd', 0)),
                'ath_price': self._sanitize_price(cg.get('ath', 0)),
                'ath_date': cg.get('ath_date', ''),
                'atl_price': self._sanitize_price(cg.get('atl', 0)),
                'atl_date': cg.get('atl_date', ''),
                'price_change_percentage_7d_in_currency': float(cg.get('price_change_7d_pct', 0) or 0),
                'price_change_percentage_30d_in_currency': float(cg.get('price_change_30d_pct', 0) or 0),
                'data_timestamp_utc': data_ts
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
    def get_cryptocurrencies_to_analyze(self, limit: Optional[int] = None, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Get cryptocurrencies to analyze using Coinbase products with enhanced ATH/ATL data.

        Returns:
            List of cryptocurrency data with basic info
        """
        logger.info("Fetching cryptocurrencies for analysis using Coinbase with parallel processing")

        # Build product list dynamically from Coinbase (online + USDC)
        prod_map = self._fetch_usdc_products()  # symbol -> {product_id, base_name}
        if symbols:
            product_ids = []
            for sym in symbols:
                pid = (prod_map.get(sym.upper()) or {}).get('product_id')
                if pid:
                    product_ids.append(pid)
                else:
                    logger.info(f"Skipping {sym.upper()} — not listed as USDC/online on Coinbase")
        else:
            product_ids = [info['product_id'] for info in prod_map.values()]

        # Apply limit if given
        max_to_process = limit if (isinstance(limit, int) and limit > 0) else len(product_ids)
        products_to_process = product_ids[:max_to_process]

        crypto_data: List[Dict] = []
        failed_products: List[str] = []
        
        # Use ThreadPoolExecutor for parallel processing (cap pool to workload)
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(products_to_process))) as executor:
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
            
        # Filter by minimum market cap if specified (only accept real CoinGecko market caps)
        min_mc = float(self.config.min_market_cap or 0)
        if min_mc > 0:
            pre_filter_count = len(crypto_data)
            filtered: List[Dict] = []
            for c in crypto_data:
                if not c.get('market_cap_is_real', False):
                    continue
                if float(c.get('market_cap', 0) or 0) >= min_mc:
                    filtered.append(c)
            crypto_data = filtered
            logger.info(f"Applied market cap filter (real >= ${min_mc:,.0f}): {len(crypto_data)}/{pre_filter_count} remaining")

        logger.info(f"Successfully retrieved {len(crypto_data)} cryptocurrencies for analysis")
        
        if len(crypto_data) == 0:
            raise InsufficientDataError("No valid cryptocurrency data retrieved")
            
        return crypto_data

    def _get_crypto_name(self, symbol: str) -> str:
        """Get readable asset name using CoinGecko index; fallback to symbol."""
        sym = (symbol or '').upper()
        # ensure list/index loaded
        _ = self._load_coingecko_list()
        item = None
        if self._cg_index:
            item = self._cg_index.get(sym)
        nm = (item or {}).get('name') if item else None
        return nm or sym

    def _calculate_price_change(self, candles: List[Dict]) -> float:
        """Calculate ~24h price change from hourly candles."""
        if not candles:
            return 0.0
        current_price = float(candles[-1]['close'])
        if len(candles) >= 24:
            price_ago = float(candles[-24]['close'])
        else:
            price_ago = float(candles[0]['close'])
        return ((current_price - price_ago) / price_ago) * 100 if price_ago else 0.0

    def _calculate_volume(self, candles: List[Dict]) -> float:
        """Calculate 24h volume from candles."""
        if not candles:
            return 0.0

        total_volume = sum(float(candle.get('volume', 0)) for candle in candles)
        return total_volume

    def _calculate_usd_volume(self, candles: List[Dict]) -> float:
        """Estimate 24h USD volume from candles using Decimal for precise accounting.

        Sums base volume × typical price per candle.
        """
        if not candles:
            return 0.0
        from decimal import Decimal, getcontext, ROUND_DOWN

        getcontext().prec = 28
        total = Decimal('0')
        for c in candles:
            try:
                base_vol = Decimal(str(c.get('volume', 0) or '0'))
                o = Decimal(str(c.get('open', 0) or '0'))
                h = Decimal(str(c.get('high', 0) or '0'))
                l = Decimal(str(c.get('low', 0) or '0'))
                cl = Decimal(str(c.get('close', 0) or '0'))
                # Typical price; prefer average of O/H/L/C if available
                typical = (o + h + l + cl) / Decimal(4) if (o != 0 or h != 0 or l != 0 or cl != 0) else cl
                total += base_vol * typical
            except Exception:
                # Fallback to close × volume if any parsing issues
                try:
                    base_vol = Decimal(str(c.get('volume', 0) or '0'))
                    cl = Decimal(str(c.get('close', 0) or '0'))
                    total += base_vol * cl
                except Exception:
                    continue
        # Return as float for compatibility with downstream dataclass
        return float(total)

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

            # Get daily candles from Coinbase (LRU cached wrapper)
            candles = list(self._cached_candles(
                product_id,
                "ONE_DAY",
                start_time.isoformat(),
                end_time.isoformat()
            ))

            if not candles or len(candles) < 30:
                logger.warning(f"Insufficient historical data for {product_id}: {len(candles) if candles else 0} candles")
                return None

            # Convert to DataFrame
            df_data = []
            for candle in candles:
                df_data.append({
                    'timestamp': (lambda ts_raw: datetime.fromtimestamp((int(ts_raw)//1000 if int(ts_raw) >= 10**12 else int(ts_raw)), UTC))(candle.get('start') or candle.get('time') or 0),
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

    @lru_cache(maxsize=512)
    def _cached_candles(self, product_id: str, granularity: str, start_iso: str, end_iso: str) -> Tuple[dict, ...]:
        """LRU-cached candles for given (product_id, granularity, start, end)."""
        try:
            start_time = datetime.fromisoformat(start_iso)
            end_time = datetime.fromisoformat(end_iso)
            # Throttle Coinbase historical fetches to avoid burst rate limits
            self._throttle()
            data = self.historical_data.get_historical_data(product_id, start_time, end_time, granularity)
            return tuple(data or [])
        except Exception as e:
            logger.warning(f"Cached candles retrieval failed for {product_id}: {e}")
            return tuple()

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate technical indicators for analysis.

        Args:
            df: DataFrame with price data

        Returns:
            Dictionary of technical indicators
        """
        try:
            # Basic cleaning to handle NaNs/Infs at head/tail
            df = df.copy()
            df[['price', 'high', 'low', 'open', 'volume']] = df[['price', 'high', 'low', 'open', 'volume']].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            df['volume'] = df['volume'].fillna(0.0)

            prices = df['price'].values
            returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([])

            # Daily vol (last 30 returns), annualized vol
            if len(returns) >= 30:
                daily_vol = float(np.std(returns[-30:], ddof=1))
            else:
                daily_vol = 0.0
            volatility_30d = daily_vol * float(np.sqrt(365))

            # Sharpe ratio (annualized) using sample std
            risk_free_rate = float(self.config.risk_free_rate)
            if len(returns) >= 2:
                mean_daily = float(np.mean(returns) - risk_free_rate / 365.0)
                std_daily = float(np.std(returns, ddof=1))
                std_daily = max(std_daily, 1e-8)
                sharpe_ratio = (mean_daily * 365.0 / (std_daily * np.sqrt(365.0))) if std_daily > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # Sortino ratio (downside std, ddof=1)
            if len(returns) >= 2:
                excess = returns - risk_free_rate / 365.0
                downside = excess[excess < 0]
                if len(downside) >= 2:
                    down_std = float(np.std(downside, ddof=1))
                    down_std = max(down_std, 1e-8)
                    sortino_ratio = (float(np.mean(excess)) * 365.0 / (down_std * np.sqrt(365.0))) if down_std > 0 else 0.0
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = 0.0

            # Maximum drawdown
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            max_drawdown = np.min(drawdown)

            # RSI (Wilder)
            rsi_period = int(self.config.rsi_period)
            rsi = self._rsi_wilder(prices, rsi_period)

            # MACD + signal + histogram classification and fresh cross
            macd_fast = int(self.config.macd_fast)
            macd_slow = int(self.config.macd_slow)
            macd_sig = int(self.config.macd_signal)
            if len(prices) >= macd_slow + macd_sig:
                macd_series = self._ema(prices, macd_fast) - self._ema(prices, macd_slow)
                signal_series = pd.Series(macd_series).ewm(span=macd_sig, adjust=False).mean().to_numpy()
                hist_series = macd_series - signal_series
                macd_line = float(macd_series[-1])
                signal_line = float(signal_series[-1])
                hist = float(hist_series[-1])
                macd_signal = "BULLISH" if hist > 0 else "BEARISH" if hist < 0 else "NEUTRAL"
                macd_cross = False
                if len(hist_series) >= 2:
                    macd_cross = (np.sign(hist_series[-1]) != np.sign(hist_series[-2]))
            else:
                macd_line = 0.0
                signal_line = 0.0
                hist = 0.0
                macd_signal = "INSUFFICIENT_DATA"
                macd_cross = False

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
                'daily_vol_30d': daily_vol,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'rsi_14': rsi,
                'macd_signal': macd_signal,
                'macd': float(macd_line),
                'macd_signal_line': float(signal_line),
                'macd_hist': float(hist),
                'macd_cross': bool(macd_cross),
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
                'daily_vol_30d': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'rsi_14': 50,
                'macd_signal': 'ERROR',
                'macd': 0.0,
                'macd_signal_line': 0.0,
                'macd_hist': 0.0,
                'macd_cross': False,
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
        """Calculate Average True Range (ATR) using Wilder smoothing."""
        try:
            if len(df) < period + 1:
                return 0.0

            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            close = df['price'].to_numpy()

            tr = np.r_[0.0, np.maximum.reduce([
                high[1:] - low[1:],
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            ])]
            atr = np.empty_like(tr)
            atr[:period] = np.nan
            atr[period] = tr[1:period + 1].sum()
            for i in range(period + 1, len(tr)):
                atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
            val = float(atr[-1] / period) if np.isfinite(atr[-1]) else 0.0
            return max(0.0, val)
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
        """Calculate ADX using Wilder smoothing (DI+/DI− and smoothed DX)."""
        try:
            if len(df) < period + 2:
                return 0.0

            high = df['high'].to_numpy()
            low = df['low'].to_numpy()
            close = df['price'].to_numpy()

            tr = np.zeros(len(close))
            dm_plus = np.zeros(len(close))
            dm_minus = np.zeros(len(close))

            for i in range(1, len(close)):
                up_move = high[i] - high[i - 1]
                down_move = low[i - 1] - low[i]
                dm_plus[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
                dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i - 1])
                tr3 = abs(low[i] - close[i - 1])
                tr[i] = max(tr1, tr2, tr3)

            # Wilder smoothing
            atr = np.zeros(len(close))
            sm_dm_plus = np.zeros(len(close))
            sm_dm_minus = np.zeros(len(close))

            atr[period] = np.sum(tr[1:period + 1])
            sm_dm_plus[period] = np.sum(dm_plus[1:period + 1])
            sm_dm_minus[period] = np.sum(dm_minus[1:period + 1])
            for i in range(period + 1, len(close)):
                atr[i] = atr[i - 1] - (atr[i - 1] / period) + tr[i]
                sm_dm_plus[i] = sm_dm_plus[i - 1] - (sm_dm_plus[i - 1] / period) + dm_plus[i]
                sm_dm_minus[i] = sm_dm_minus[i - 1] - (sm_dm_minus[i - 1] / period) + dm_minus[i]

            with np.errstate(divide='ignore', invalid='ignore'):
                di_plus = 100.0 * (sm_dm_plus / atr)
                di_minus = 100.0 * (sm_dm_minus / atr)
                dx = 100.0 * np.abs(di_plus - di_minus) / (di_plus + di_minus)

            # First ADX value is average of first 'period' DX values starting at index period+1
            adx = np.zeros(len(close))
            start = period * 2
            if len(close) <= start or not np.isfinite(np.nanmean(dx[period + 1:start + 1])):
                return 0.0
            adx[start] = np.nanmean(dx[period + 1:start + 1])
            for i in range(start + 1, len(close)):
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

            val = float(adx[-1])
            return max(0.0, min(100.0, val)) if np.isfinite(val) else 0.0
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

    # --- Indicator helpers (Wilder RSI, MACD) ---
    def _rsi_wilder(self, prices: np.ndarray, period: int = 14) -> float:
        """Wilder's RSI implementation with smoothing."""
        try:
            if prices is None or len(prices) <= period:
                return 50.0
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            # If not enough deltas, return neutral
            if len(gains) < period:
                return 50.0
            avg_gain = gains[:period].mean()
            avg_loss = losses[:period].mean()
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return float(max(0.0, min(100.0, rsi)))
        except Exception:
            return 50.0

    def _ema(self, x: np.ndarray, span: int) -> np.ndarray:
        return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()

    def _macd_signal(self, prices: np.ndarray, fast=12, slow=26, signal=9) -> Tuple[float, float, float, str]:
        try:
            if prices is None or len(prices) < slow + signal:
                return 0.0, 0.0, 0.0, "INSUFFICIENT_DATA"
            macd_line = self._ema(prices, fast) - self._ema(prices, slow)
            signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().to_numpy()
            hist = macd_line[-1] - signal_line[-1]
            label = "BULLISH" if hist > 0 else "BEARISH" if hist < 0 else "NEUTRAL"
            return float(macd_line[-1]), float(signal_line[-1]), float(hist), label
        except Exception:
            return 0.0, 0.0, 0.0, "ERROR"

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
            previous_30d = prices[-60:-30] if len(prices) >= 60 else prices[: max(1, len(prices) // 2)]

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

            # Market cap score (larger cap = higher score)
            market_cap = (
                coin_data.get('market_cap')
                or coin_data.get('market_cap_usd')
                or 0
            )
            if market_cap > 1000000000:  # > $1B
                market_cap_score = 100
            elif market_cap > 500000000:  # > $500M
                market_cap_score = 90
            elif market_cap > 100000000:  # > $100M
                market_cap_score = 80
            else:
                market_cap_score = 70

            score += market_cap_score * 0.3

            # Volume score (higher volume = better liquidity)
            volume_24h = (
                coin_data.get('total_volume')
                or coin_data.get('total_volume_usd')
                or coin_data.get('volume_24h')
                or 0
            )
            mc = market_cap if market_cap and market_cap > 0 else 1
            volume_to_market_cap = float(volume_24h) / float(mc)

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
            price_change_7d = abs(
                coin_data.get('price_change_7d', 0)
                or coin_data.get('price_change_percentage_7d_in_currency', 0)
            )
            if price_change_7d < 10:
                stability_score = 90
            elif price_change_7d < 25:
                stability_score = 70
            elif price_change_7d < 50:
                stability_score = 50
            else:
                stability_score = 30

            score += stability_score * 0.25

            # ATH gap score (further below ATH = better entry)
            current_price = coin_data.get('current_price', 0) or coin_data.get('current_price_usd', 0)
            ath_price = coin_data.get('ath', 0) or coin_data.get('ath_price', 0)
            if ath_price and ath_price > 0:
                ath_gap_pct = max(0.0, ((ath_price - current_price) / ath_price) * 100.0)
                if ath_gap_pct < 20:
                    ath_score = 30   # close to ATH → worse entry
                elif ath_gap_pct < 50:
                    ath_score = 60
                elif ath_gap_pct < 80:
                    ath_score = 80
                else:
                    ath_score = 100  # deep discount
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
            historical_df = self.get_historical_data(product_id, days=self.config.analysis_days)
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

            # Data timestamp (UTC) from last candle
            try:
                ts_idx = historical_df.index.max()
                ts_str = ts_idx.strftime('%Y-%m-%d %H:%M:%SZ') if ts_idx is not None else ''
            except Exception:
                ts_str = ''

            return CryptoMetrics(
                symbol=symbol,
                name=name,
                position_side=PositionSide.LONG.value,
                current_price=coin_data.get('current_price', 0),
                market_cap=coin_data.get('market_cap', 0),
                market_cap_rank=int(coin_data.get('market_cap_rank', 0) or 0),
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
                position_size_percentage=trading_levels['position_size_percentage'],
                data_timestamp_utc=ts_str
            )

        except Exception as e:
            logger.error(f"Error analyzing {coin_data.get('symbol', 'unknown')}: {str(e)}")
            return None

    def _build_long_metrics(self, coin_data: Dict, df: pd.DataFrame, technical_metrics: Dict, momentum_score: float, price_changes: Dict[str, float]) -> Optional[CryptoMetrics]:
        """Build a CryptoMetrics object for the LONG side using precomputed data."""
        try:
            fundamental_score = self.calculate_fundamental_score(coin_data)
            technical_score = self._calculate_technical_score(technical_metrics, momentum_score)
            risk_score, risk_level = self.calculate_risk_score({**technical_metrics, 'fundamental_score': fundamental_score})
            overall_score = (
                technical_score * 0.4 + fundamental_score * 0.4 + momentum_score * 0.2
            ) * (1 - risk_score / 200)
            trading_levels = self.calculate_trading_levels(df, coin_data.get('current_price', 0), technical_metrics)
            try:
                ts_idx = df.index.max()
                ts_str = ts_idx.strftime('%Y-%m-%d %H:%M:%SZ') if ts_idx is not None else ''
            except Exception:
                ts_str = ''

            return CryptoMetrics(
                symbol=coin_data['symbol'],
                name=coin_data['name'],
                position_side=PositionSide.LONG.value,
                current_price=coin_data.get('current_price', 0),
                market_cap=coin_data.get('market_cap', 0),
                market_cap_rank=int(coin_data.get('market_cap_rank', 0) or 0),
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
                entry_price=trading_levels['entry_price'],
                stop_loss_price=trading_levels['stop_loss_price'],
                take_profit_price=trading_levels['take_profit_price'],
                risk_reward_ratio=trading_levels['risk_reward_ratio'],
                position_size_percentage=trading_levels['position_size_percentage'],
                data_timestamp_utc=ts_str
            )
        except Exception as e:
            logger.error(f"Failed to build LONG metrics for {coin_data.get('symbol', 'unknown')}: {e}")
            return None

    def _build_short_metrics(self, coin_data: Dict, df: pd.DataFrame, technical_metrics: Dict, long_momentum: float, price_changes: Dict[str, float]) -> Optional[CryptoMetrics]:
        """Build a CryptoMetrics object for the SHORT side using precomputed data."""
        try:
            short_tech_score = self._calculate_technical_score_short(technical_metrics, long_momentum)
            fundamental_score = self.calculate_fundamental_score(coin_data)
            risk_score, risk_level = self.calculate_risk_score({**technical_metrics, 'fundamental_score': fundamental_score})
            short_overall = (
                short_tech_score * 0.4 + fundamental_score * 0.4 + (max(0.0, min(100.0, 100.0 - long_momentum))) * 0.2
            ) * (1 - risk_score / 200)
            short_levels = self.calculate_short_trading_levels(df, coin_data.get('current_price', 0), technical_metrics)
            try:
                ts_idx = df.index.max()
                ts_str = ts_idx.strftime('%Y-%m-%d %H:%M:%SZ') if ts_idx is not None else ''
            except Exception:
                ts_str = ''

            return CryptoMetrics(
                symbol=coin_data['symbol'],
                name=coin_data['name'],
                position_side=PositionSide.SHORT.value,
                current_price=coin_data.get('current_price', 0),
                market_cap=coin_data.get('market_cap', 0),
                market_cap_rank=int(coin_data.get('market_cap_rank', 0) or 0),
                volume_24h=coin_data.get('volume_24h', 0),
                price_change_24h=coin_data.get('price_change_24h', 0),
                price_change_7d=price_changes.get('7d', 0),
                price_change_30d=price_changes.get('30d', 0),
                ath_price=coin_data.get('ath_price', 0),
                ath_date=coin_data.get('ath_date', ''),
                atl_price=coin_data.get('atl_price', 0),
                atl_date=coin_data.get('atl_date', ''),
                volatility_30d=technical_metrics['volatility_30d'],
                sharpe_ratio=technical_metrics['sharpe_ratio'],
                sortino_ratio=technical_metrics['sortino_ratio'],
                max_drawdown=technical_metrics['max_drawdown'],
                rsi_14=technical_metrics['rsi_14'],
                macd_signal=technical_metrics['macd_signal'],
                bb_position=technical_metrics['bb_position'],
                trend_strength=technical_metrics['trend_strength'],
                momentum_score=max(0.0, min(100.0, 100.0 - long_momentum)),
                fundamental_score=fundamental_score,
                technical_score=short_tech_score,
                risk_score=risk_score,
                overall_score=short_overall,
                risk_level=risk_level,
                entry_price=short_levels['entry_price'],
                stop_loss_price=short_levels['stop_loss_price'],
                take_profit_price=short_levels['take_profit_price'],
                risk_reward_ratio=short_levels['risk_reward_ratio'],
                position_size_percentage=short_levels['position_size_percentage'],
                data_timestamp_utc=ts_str
            )
        except Exception as e:
            logger.error(f"Failed to build SHORT metrics for {coin_data.get('symbol', 'unknown')}: {e}")
            return None

    def _calculate_price_changes_from_history(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price changes from historical data."""
        try:
            if len(df) < 30:
                return {'7d': 0, '30d': 0, 'ath': 0, 'atl': 0}

            current_price = df['price'].iloc[-1]

            # 7-day change
            if len(df) >= 7:
                price_7d_ago = float(df['price'].iloc[-7])
                change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100 if price_7d_ago != 0 else 0.0
            else:
                change_7d = 0.0

            # 30-day change
            if len(df) >= 30:
                price_30d_ago = float(df['price'].iloc[-30])
                change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100 if price_30d_ago != 0 else 0.0
            else:
                change_30d = 0.0

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

    def _get_ath_atl_from_coingecko(self, symbol: str) -> Dict[str, Union[float, str, int]]:
        """
        Get accurate ATH/ATL data from CoinGecko API.

        Args:
            symbol: Cryptocurrency symbol (e.g., 'btc', 'eth')

        Returns:
            Dictionary with ATH/ATL data
        """
        try:
            coingecko_id = self._coingecko_id_for_symbol(symbol) or symbol.lower()

            # Fetch data from CoinGecko with caching (slim payload to reduce rate caps)
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
            params = {
                'localization': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false',
                'sparkline': 'false'
            }
            try:
                data = self._make_cached_request(url, params)
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

            # Extract market data
            market_data = data.get('market_data', {})

            ath_price = market_data.get('ath', {}).get('usd', 0)
            ath_date = market_data.get('ath_date', {}).get('usd', '')
            atl_price = market_data.get('atl', {}).get('usd', 0)
            atl_date = market_data.get('atl_date', {}).get('usd', '')
            current_price = market_data.get('current_price', {}).get('usd', 0)
            market_cap = market_data.get('market_cap', {}).get('usd', 0)
            total_volume = market_data.get('total_volume', {}).get('usd', 0)
            price_change_7d = market_data.get('price_change_percentage_7d_in_currency', {}).get('usd', 0)
            price_change_30d = market_data.get('price_change_percentage_30d_in_currency', {}).get('usd', 0)
            market_cap_rank = data.get('market_cap_rank', 0) or 0
            
            return {
                'ath': float(ath_price) if ath_price else 0,
                'ath_date': ath_date[:10] if ath_date else '',
                'atl': float(atl_price) if atl_price else 0,
                'atl_date': atl_date[:10] if atl_date else '',
                'current_price_usd': float(current_price) if current_price else 0,
                'market_cap_usd': float(market_cap) if market_cap else 0,
                'total_volume_usd': float(total_volume) if total_volume else 0,
                'price_change_7d_pct': float(price_change_7d) if price_change_7d else 0,
                'price_change_30d_pct': float(price_change_30d) if price_change_30d else 0,
                'market_cap_rank': int(market_cap_rank)
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

            # Method 1: ATR-based stop loss (2x ATR below entry). ATR is in price units.
            atr_raw = float(technical_metrics.get('atr', 0.0))
            if atr_raw > 0 and entry_price > 0:
                atr_pct = atr_raw / entry_price * 100.0
            else:
                atr_pct = float(df['price'].pct_change().std(ddof=1) * 100.0) if len(df) >= 2 else 0.0
            if np.isfinite(atr_pct) and atr_pct > 0:
                atr_stop = entry_price * (1 - 2.0 * atr_pct / 100.0)
                stop_loss_methods.append(atr_stop)

            # Method 2: Recent low support (10-day low)
            if len(df) >= 10:
                recent_low = df['low'].tail(10).min()
                support_stop = recent_low * 0.98  # 2% below recent low
                stop_loss_methods.append(support_stop)

            # Method 3: Percentage-based stop via DAILY vol thresholds
            daily_vol = float(technical_metrics.get('daily_vol_30d', 0.0) or 0.0)
            if daily_vol < 0.02:
                pct = 0.08
            elif daily_vol < 0.04:
                pct = 0.06
            else:
                pct = 0.05
            stop_loss_methods.append(entry_price * (1 - pct))

            # Choose the most conservative stop loss (highest price)
            stop_loss_price = max(stop_loss_methods) if stop_loss_methods else entry_price * 0.95
            # Clamp invariants for long positions
            stop_loss_price = min(stop_loss_price, entry_price * 0.999)

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
            ath_distance = max((ath_price - current_price) / current_price, 0.0)

            if ath_distance < 0.2:  # Within 20% of ATH
                ath_tp = ath_price * 1.05  # 5% above ATH
            elif ath_distance < 0.5:  # Within 50% of ATH
                ath_tp = ath_price * 1.10  # 10% above ATH
            else:  # Far from ATH
                ath_tp = entry_price * 2.0  # 100% return target
            take_profit_methods.append(ath_tp)

            # Choose the most conservative take profit (lowest price), clamp above entry
            tp_raw = min(take_profit_methods) if take_profit_methods else entry_price * 1.50
            take_profit_price = max(entry_price * 1.01, tp_raw)
            # Clamp invariants for long positions
            take_profit_price = max(take_profit_price, entry_price * 1.001)

            # Calculate risk-reward ratio with floor and cap, include ATR-based floor
            raw_risk = entry_price - stop_loss_price
            risk = max(entry_price * 1e-3, raw_risk, (atr_raw * 0.25) if atr_raw > 0 else 0.0)
            reward = max(0.0, take_profit_price - entry_price)
            risk_reward_ratio = min(10.0, (reward / risk) if risk > 0 else 0.0)

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

    def calculate_short_trading_levels(self, df: pd.DataFrame, current_price: float, technical_metrics: Dict) -> Dict[str, float]:
        """
        Calculate entry, stop loss, and take profit levels for a SHORT position.

        Mirrors the long calculation but inverted for short bias.
        """
        try:
            entry_price = current_price

            # Stop Loss (above entry for shorts)
            stop_loss_methods = []

            atr_raw = float(technical_metrics.get('atr', 0.0))
            if atr_raw > 0 and entry_price > 0:
                atr_pct = atr_raw / entry_price * 100.0
            else:
                atr_pct = float(df['price'].pct_change().std(ddof=1) * 100.0) if len(df) >= 2 else 0.0
            if np.isfinite(atr_pct) and atr_pct > 0:
                atr_stop = entry_price * (1 + 2.0 * atr_pct / 100.0)
                stop_loss_methods.append(atr_stop)

            if len(df) >= 10:
                recent_high = df['high'].tail(10).max()
                resistance_stop = recent_high * 1.02  # 2% above recent high
                stop_loss_methods.append(resistance_stop)

            daily_vol = float(technical_metrics.get('daily_vol_30d', 0.0) or 0.0)
            if daily_vol < 0.02:
                pct = 0.08
            elif daily_vol < 0.04:
                pct = 0.06
            else:
                pct = 0.05
            stop_loss_methods.append(entry_price * (1 + pct))

            # For shorts choose the tightest stop (lowest above entry)
            stop_loss_price = min(stop_loss_methods) if stop_loss_methods else entry_price * 1.05
            # Clamp invariants for short positions
            stop_loss_price = max(stop_loss_price, entry_price * 1.001)

            # Take Profit (below entry for shorts)
            take_profit_methods = []

            risk_amount = stop_loss_price - entry_price
            rr_take_profit = entry_price - (risk_amount * 3)
            take_profit_methods.append(rr_take_profit)

            if len(df) >= 20:
                recent_low = df['low'].tail(20).min()
                support_tp = recent_low * 0.98  # 2% below recent low
                take_profit_methods.append(support_tp)

            atl_price = df['low'].min()
            atl_distance = (current_price - atl_price) / current_price if current_price > 0 else 0
            if atl_distance < 0.2:  # Within 20% of ATL
                atl_tp = atl_price * 0.95  # 5% below ATL
            elif atl_distance < 0.5:
                atl_tp = atl_price * 0.90  # 10% below ATL
            else:
                atl_tp = entry_price * 0.5  # 50% drawdown target when far from ATL
            take_profit_methods.append(atl_tp)

            # Conservative take profit (closest to entry for shorts => highest price), clamp below entry
            tp_raw = max(take_profit_methods) if take_profit_methods else entry_price * 0.85
            take_profit_price = min(entry_price * 0.99, tp_raw)
            # Clamp invariants for short positions
            take_profit_price = min(take_profit_price, entry_price * 0.999)

            raw_risk = stop_loss_price - entry_price
            risk = max(entry_price * 1e-3, raw_risk, (atr_raw * 0.25) if atr_raw > 0 else 0.0)
            reward = max(0.0, entry_price - take_profit_price)
            risk_reward_ratio = min(10.0, (reward / risk) if risk > 0 else 0.0)

            if risk_reward_ratio >= 3:
                position_size_percentage = 2.0
            elif risk_reward_ratio >= 2:
                position_size_percentage = 1.5
            else:
                position_size_percentage = 1.0

            return {
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_reward_ratio': risk_reward_ratio,
                'position_size_percentage': position_size_percentage
            }
        except Exception as e:
            logger.error(f"Error calculating short trading levels: {str(e)}")
            return {
                'entry_price': current_price,
                'stop_loss_price': current_price * 1.05,
                'take_profit_price': current_price * 0.85,
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

            # MACD cross small bonus/penalty
            macd_cross = bool(technical_metrics.get('macd_cross', False))
            if macd_cross:
                macd_hist = float(technical_metrics.get('macd_hist', 0.0) or 0.0)
                score += 5 if macd_hist > 0 else -5

            return min(100, max(0, score))

        except Exception as e:
            logger.error(f"Error calculating technical score: {str(e)}")
            return 50

    def _calculate_technical_score_short(self, technical_metrics: Dict, momentum_score_long: float) -> float:
        """Calculate technical score for SHORT bias by inverting bullish signals."""
        try:
            score = 0

            # RSI: overbought better for shorts
            rsi = technical_metrics.get('rsi_14', 50)
            if rsi >= 70:
                rsi_score = 90
            elif rsi >= 30:
                rsi_score = 70
            else:  # oversold -> worse for shorts
                rsi_score = 45
            score += rsi_score * 0.2

            # MACD: bearish preferred
            macd = technical_metrics.get('macd_signal', 'NEUTRAL')
            if macd == 'BEARISH':
                macd_score = 85
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:  # BULLISH
                macd_score = 40
            score += macd_score * 0.2

            # Bollinger Bands: overbought preferred for shorts
            bb = technical_metrics.get('bb_position', 'NEUTRAL')
            if bb == 'OVERBOUGHT':
                bb_score = 85
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:  # OVERSOLD -> worse for shorts
                bb_score = 45
            score += bb_score * 0.15

            # Trend strength: negative trend favored for shorts
            trend = technical_metrics.get('trend_strength', 0)
            if trend < -0.5:
                trend_score = 90
            elif trend < -0.1:
                trend_score = 70
            elif trend < 0.1:
                trend_score = 50
            else:
                trend_score = 30
            score += trend_score * 0.25

            # Momentum: derive short momentum from long momentum
            short_momentum = max(0.0, min(100.0, 100.0 - momentum_score_long))
            score += short_momentum * 0.2

            return min(100, score)
        except Exception as e:
            logger.error(f"Error calculating short technical score: {str(e)}")
            return 50

    def find_best_opportunities(self, limit: int = 15) -> List[CryptoMetrics]:
        """
        Find the best long-term cryptocurrency opportunities using Coinbase.
        Now evaluates both LONG and SHORT candidates and ranks across both.

        Args:
            limit: Number of cryptocurrencies to analyze

        Returns:
            List of CryptoMetrics objects sorted by overall score
        """
        logger.info("Starting comprehensive crypto analysis using Coinbase...")

        # Get cryptocurrencies to analyze (respect limit and optional symbols)
        crypto_list = self.get_cryptocurrencies_to_analyze(limit=limit, symbols=self.config.symbols)
        if not crypto_list:
            logger.error("Failed to retrieve cryptocurrency list")
            return []

        # Analyze each cryptocurrency and include applicable side(s)
        analyzed_candidates: List[CryptoMetrics] = []
        for i, coin_data in enumerate(crypto_list[:limit]):
            logger.info(f"Analyzing {i+1}/{min(len(crypto_list), limit)}: {coin_data['symbol']} ({coin_data['name']})")
            product_id = coin_data['product_id']
            df = self.get_historical_data(product_id, days=self.config.analysis_days)
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient historical data for {coin_data['symbol']}")
                continue
            tech = self.calculate_technical_indicators(df)
            mom = self.calculate_momentum_score(df)
            chg = self._calculate_price_changes_from_history(df)

            if self.side in ('long', 'both'):
                long_metrics = self._build_long_metrics(coin_data, df, tech, mom, chg)
                if long_metrics and getattr(long_metrics, 'risk_reward_ratio', 0.0) >= 2.0:
                    analyzed_candidates.append(long_metrics)

            if self.side in ('short', 'both'):
                short_metrics = self._build_short_metrics(coin_data, df, tech, mom, chg)
                if short_metrics and getattr(short_metrics, 'risk_reward_ratio', 0.0) >= 2.0:
                    analyzed_candidates.append(short_metrics)

        # Filter by minimum score if requested
        min_score = float(self.config.min_overall_score or 0)
        if min_score > 0:
            analyzed_candidates = [c for c in analyzed_candidates if c and float(c.overall_score) >= min_score]

        # Unique by symbol: keep best side per symbol
        if self.config.unique_by_symbol:
            best_by_symbol: Dict[str, CryptoMetrics] = {}
            for c in analyzed_candidates:
                if not c:
                    continue
                prev = best_by_symbol.get(c.symbol)
                if (prev is None) or (c.overall_score > prev.overall_score):
                    best_by_symbol[c.symbol] = c
            analyzed_candidates = list(best_by_symbol.values())

        # Optional per-side cap
        tps = self.config.top_per_side
        if tps and tps > 0:
            longs = [c for c in analyzed_candidates if getattr(c, 'position_side', '') == 'LONG']
            shorts = [c for c in analyzed_candidates if getattr(c, 'position_side', '') == 'SHORT']
            longs.sort(key=lambda x: x.overall_score, reverse=True)
            shorts.sort(key=lambda x: x.overall_score, reverse=True)
            analyzed_candidates = longs[:tps] + shorts[:tps]

        # Sort by overall score (descending)
        analyzed_candidates.sort(key=lambda x: x.overall_score, reverse=True)

        # Return top results across selected side(s)
        top_results = analyzed_candidates[: self.config.max_results]

        long_count = sum(1 for x in top_results if getattr(x, 'position_side', '') == 'LONG')
        short_count = sum(1 for x in top_results if getattr(x, 'position_side', '') == 'SHORT')
        logger.info(
            f"Analysis complete. Found {len(top_results)} top opportunities (LONG={long_count}, SHORT={short_count})."
        )
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
        # Use UTC for deterministic timestamps
        print(f"Generated on (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}Z")
        print(f"Total opportunities listed: {len(results)}")
        print("="*100)

        for i, crypto in enumerate(results, 1):
            print(f"\n{i}. {crypto.symbol} ({crypto.name}) — {crypto.position_side}")
            print("-" * 50)
            if getattr(crypto, 'data_timestamp_utc', ''):
                print(f"Data Timestamp (UTC): {crypto.data_timestamp_utc}")
            print(f"Price: ${crypto.current_price:.6f}")
            rank_str = f"#{crypto.market_cap_rank}" if getattr(crypto, 'market_cap_rank', 0) else "N/A"
            print(f"Market Cap: ${crypto.market_cap:,.0f} (Rank {rank_str})")
            print(f"24h Volume: ${crypto.volume_24h:,.0f}")
            print(f"24h Change: {crypto.price_change_24h:.2f}%")
            print(f"7d Change: {crypto.price_change_7d:.2f}%")
            print(f"30d Change: {crypto.price_change_30d:.2f}%")
            print(f"ATH: ${crypto.ath_price:.2f} (Date: {crypto.ath_date or 'N/A'})")
            print(f"ATL: ${crypto.atl_price:.6f} (Date: {crypto.atl_date or 'N/A'})")
            print(f"Volatility (30d, ann.): {crypto.volatility_30d*100:.1f}%")
            print(f"Sharpe Ratio: {crypto.sharpe_ratio:.2f}")
            print(f"Sortino Ratio: {crypto.sortino_ratio:.2f}")
            # Display drawdown with its actual sign
            print(f"Max Drawdown: {crypto.max_drawdown * 100:.2f}%")
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
            print(f"💼 TRADING LEVELS ({crypto.position_side}):")
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
    parser.add_argument('--output', type=str, choices=['console', 'json'], default='console',
                       help='Output format (default: console)')
    parser.add_argument('--side', type=str, choices=['long', 'short', 'both'], default='both',
                       help='Which side(s) to evaluate (default: both)')
    parser.add_argument('--unique-by-symbol', action='store_true',
                       help='Keep only the best side per symbol')
    parser.add_argument('--min-score', type=float, default=0.0,
                       help='Filter out candidates below this overall score')
    parser.add_argument('--save', type=str,
                       help='Optional path to save results (json or csv)')
    parser.add_argument('--symbols', type=str,
                       help='Comma-separated list of symbols to analyze (e.g., BTC,ETH,SOL)')
    parser.add_argument('--top-per-side', type=int,
                       help='Cap results per side before final sorting')
    parser.add_argument('--max-workers', type=int,
                       help='Override worker threads for parallel fetch')
    parser.add_argument('--offline', action='store_true',
                       help='Avoid external HTTP where possible (use cache only)')
    parser.add_argument('--quotes', type=str,
                       help='Preferred quote currencies (comma-separated), e.g., USDC,USD,USDT')
    parser.add_argument('--risk-free-rate', type=float,
                       help='Override annual risk-free rate (e.g., 0.03 for 3%)')

    args = parser.parse_args()

    # Create configuration
    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    quotes_list = None
    if args.quotes:
        quotes_list = [q.strip().upper() for q in args.quotes.split(',') if q.strip()]

    config = CryptoFinderConfig(
        min_market_cap=args.min_market_cap,
        max_results=args.max_results,
        side=args.side,
        unique_by_symbol=bool(args.unique_by_symbol),
        min_overall_score=float(args.min_score or 0.0),
        offline=bool(args.offline),
        symbols=symbols_list,
        top_per_side=args.top_per_side,
        max_workers=args.max_workers or CryptoFinderConfig.from_env().max_workers,
        quotes=quotes_list,
        risk_free_rate=args.risk_free_rate if args.risk_free_rate is not None else CryptoFinderConfig.from_env().risk_free_rate
    )
    
    # Initialize the finder
    finder = LongTermCryptoFinder(config=config)

    # Find opportunities
    results = finder.find_best_opportunities(limit=args.limit)

    if not results:
        print("No opportunities found. Please check your internet connection and try again.")
        return

    # Output results
    if args.output == 'json' or (args.save and args.save.lower().endswith('.json')):
        # Convert results to dictionaries for JSON serialization
        json_results = []
        for crypto in results:
            crypto_dict = {
                'symbol': crypto.symbol,
                'name': crypto.name,
                'position_side': getattr(crypto, 'position_side', 'LONG'),
                'current_price': _finite(crypto.current_price),
                'market_cap': _finite(crypto.market_cap),
                'market_cap_rank': int(getattr(crypto, 'market_cap_rank', 0) or 0),
                'volume_24h': _finite(crypto.volume_24h),
                'price_change_24h': _finite(crypto.price_change_24h),
                'price_change_7d': _finite(crypto.price_change_7d),
                'price_change_30d': _finite(crypto.price_change_30d),
                'ath_price': _finite(crypto.ath_price),
                'ath_date': crypto.ath_date,
                'atl_price': _finite(crypto.atl_price),
                'atl_date': crypto.atl_date,
                'volatility_30d': _finite(crypto.volatility_30d),
                'sharpe_ratio': _finite(crypto.sharpe_ratio),
                'sortino_ratio': _finite(crypto.sortino_ratio),
                'max_drawdown': _finite(crypto.max_drawdown),
                'rsi_14': _finite(crypto.rsi_14),
                'macd_signal': crypto.macd_signal,
                'bb_position': crypto.bb_position,
                'trend_strength': _finite(crypto.trend_strength),
                'momentum_score': _finite(crypto.momentum_score),
                'fundamental_score': _finite(crypto.fundamental_score),
                'technical_score': _finite(crypto.technical_score),
                'risk_score': _finite(crypto.risk_score),
                'overall_score': _finite(crypto.overall_score),
                'risk_level': crypto.risk_level.value,
                'entry_price': _finite(crypto.entry_price),
                'stop_loss_price': _finite(crypto.stop_loss_price),
                'take_profit_price': _finite(crypto.take_profit_price),
                'risk_reward_ratio': _finite(crypto.risk_reward_ratio),
                'position_size_percentage': _finite(crypto.position_size_percentage),
                'data_timestamp_utc': getattr(crypto, 'data_timestamp_utc', '')
            }
            json_results.append(crypto_dict)
        if args.save and args.save.lower().endswith('.json'):
            # Atomic JSON save
            finder._atomic_write_json(Path(args.save), json_results)
            print(f"Saved {len(json_results)} results to {args.save}")
        if args.output == 'json':
            print(json.dumps(json_results, indent=2))
    else:
        finder.print_results(results)

        # Optional CSV saving when not using --output json
        if args.save and args.save.lower().endswith('.csv'):
            import csv
            fieldnames = [
                'symbol', 'name', 'position_side', 'current_price', 'market_cap', 'market_cap_rank',
                'volume_24h', 'price_change_24h', 'price_change_7d', 'price_change_30d',
                'ath_price', 'ath_date', 'atl_price', 'atl_date',
                'volatility_30d', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'rsi_14',
                'macd_signal', 'bb_position', 'trend_strength', 'momentum_score', 'fundamental_score',
                'technical_score', 'risk_score', 'overall_score', 'risk_level',
                'entry_price', 'stop_loss_price', 'take_profit_price', 'risk_reward_ratio',
                'position_size_percentage', 'data_timestamp_utc'
            ]
            # Atomic CSV save: write to temp and replace
            tmp_path = Path(args.save + f".tmp.{os.getpid()}.{int(time.time()*1000)}")
            final_path = Path(args.save)
            with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for crypto in results:
                    writer.writerow({
                        'symbol': crypto.symbol,
                        'name': crypto.name,
                        'position_side': getattr(crypto, 'position_side', 'LONG'),
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
                        'risk_level': crypto.risk_level.value,
                        'entry_price': crypto.entry_price,
                        'stop_loss_price': crypto.stop_loss_price,
                        'take_profit_price': crypto.take_profit_price,
                        'risk_reward_ratio': crypto.risk_reward_ratio,
                        'position_size_percentage': crypto.position_size_percentage,
                        'data_timestamp_utc': getattr(crypto, 'data_timestamp_utc', '')
                    })
            os.replace(tmp_path, final_path)
            print(f"Saved {len(results)} results to {args.save}")

if __name__ == "__main__":
    main()
