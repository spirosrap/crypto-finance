#!/usr/bin/env python3
"""
Long-Term Crypto Opportunity Finder

This program analyzes cryptocurrencies to find the best long-term investment opportunities
by evaluating multiple factors including technical indicators, fundamental metrics,
risk assessment, and market sentiment.

Author: Crypto Finance Toolkit
"""

import io
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
from typing import Callable, Dict, List, Optional, Set, TextIO, Tuple, Union
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
import hashlib
from coinbaseservice import CoinbaseService
from historicaldata import HistoricalData
from llm_scoring import LLMScorer, build_llm_payload

# Configuration management
@dataclass
class CryptoFinderConfig:
    """Configuration class for the crypto finder."""
    min_market_cap: int = 100000000  # $100M default
    max_results: int = 20
    max_workers: int = 4
    request_delay: float = 0.5  # seconds
    cache_ttl: int = 300  # 5 minutes
    force_refresh_candles: bool = False  # bypass candle caches when true
    risk_free_rate: float = 0.03  # 3% annual
    analysis_days: int = 365
    min_volume_24h: float = 0.0
    min_volume_market_cap_ratio: float = 0.0
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
    max_risk_level: Optional[str] = None  # highest risk level to include (e.g., "MEDIUM")
    risk_reward_weight: float = 0.15  # influence of risk/reward alignment
    trend_weight: float = 0.10  # influence of trend alignment
    use_openai_scoring: bool = False
    openai_model: str = "gpt-5-mini"
    openai_weight: float = 0.25
    openai_max_candidates: int = 12
    openai_temperature: Optional[float] = None
    openai_sleep_seconds: float = 0.0
    report_position_notional: float = 1000.0
    report_leverage: float = 50.0

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
        max_risk_level_env = os.getenv('CRYPTO_MAX_RISK_LEVEL')
        max_risk_level = max_risk_level_env.strip() if max_risk_level_env else None

        def _float_env(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logging.getLogger(__name__).warning(
                    "Invalid float value '%s' for %s; using default %.3f",
                    raw,
                    name,
                    default,
                )
                return default

        def _optional_float_env(name: str) -> Optional[float]:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return None
            try:
                return float(raw)
            except ValueError:
                logging.getLogger(__name__).warning(
                    "Invalid float value '%s' for %s; ignoring override",
                    raw,
                    name,
                )
                return None

        return cls(
            min_market_cap=int(os.getenv('CRYPTO_MIN_MARKET_CAP', '100000000')),
            max_results=int(os.getenv('CRYPTO_MAX_RESULTS', '20')),
            max_workers=int(os.getenv('CRYPTO_MAX_WORKERS', '4')),
            request_delay=float(os.getenv('CRYPTO_REQUEST_DELAY', '0.5')),
            cache_ttl=int(os.getenv('CRYPTO_CACHE_TTL', '300')),
            force_refresh_candles=os.getenv('CRYPTO_FORCE_REFRESH_CANDLES', '0').lower() in ('1', 'true', 't', 'yes', 'y'),
            risk_free_rate=_float_env('CRYPTO_RISK_FREE_RATE', 0.03),
            analysis_days=int(os.getenv('CRYPTO_ANALYSIS_DAYS', '365')),
            min_volume_24h=_float_env('CRYPTO_MIN_VOLUME_24H', 0.0),
            min_volume_market_cap_ratio=_float_env('CRYPTO_MIN_VMC_RATIO', 0.0),
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
            min_overall_score=_float_env('CRYPTO_MIN_SCORE', 0.0),
            offline=os.getenv('CRYPTO_OFFLINE', '0') in ('1', 'true', 'True'),
            symbols=symbols_list,
            top_per_side=int(os.getenv('CRYPTO_TOP_PER_SIDE')) if os.getenv('CRYPTO_TOP_PER_SIDE') else None,
            quotes=quotes_list,
            max_risk_level=max_risk_level,
            risk_reward_weight=_float_env('CRYPTO_RR_WEIGHT', 0.15),
            trend_weight=_float_env('CRYPTO_TREND_WEIGHT', 0.10),
            use_openai_scoring=os.getenv('CRYPTO_USE_OPENAI_SCORING', '0').lower() in ('1', 'true', 't', 'yes', 'y'),
            openai_model=os.getenv('CRYPTO_OPENAI_MODEL', 'gpt-5-mini'),
            openai_weight=_float_env('CRYPTO_OPENAI_WEIGHT', 0.25),
            openai_max_candidates=int(os.getenv('CRYPTO_OPENAI_MAX_CANDIDATES', '12') or '12'),
            openai_temperature=_optional_float_env('CRYPTO_OPENAI_TEMPERATURE'),
            openai_sleep_seconds=_float_env('CRYPTO_OPENAI_SLEEP_SECONDS', 0.0),
            report_position_notional=_float_env('CRYPTO_REPORT_NOTIONAL', 1000.0),
            report_leverage=_float_env('CRYPTO_REPORT_LEVERAGE', 50.0),
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'min_market_cap': self.min_market_cap,
            'max_results': self.max_results,
            'max_workers': self.max_workers,
            'request_delay': self.request_delay,
            'cache_ttl': self.cache_ttl,
            'force_refresh_candles': self.force_refresh_candles,
            'risk_free_rate': self.risk_free_rate,
            'analysis_days': self.analysis_days,
            'min_volume_24h': self.min_volume_24h,
            'min_volume_market_cap_ratio': self.min_volume_market_cap_ratio,
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
            'offline': self.offline,
            'max_risk_level': self.max_risk_level.upper() if isinstance(self.max_risk_level, str) else self.max_risk_level,
            'risk_reward_weight': self.risk_reward_weight,
            'trend_weight': self.trend_weight,
            'use_openai_scoring': self.use_openai_scoring,
            'openai_model': self.openai_model,
            'openai_weight': self.openai_weight,
            'openai_max_candidates': self.openai_max_candidates,
            'openai_temperature': self.openai_temperature,
            'openai_sleep_seconds': self.openai_sleep_seconds,
            'report_position_notional': self.report_position_notional,
            'report_leverage': self.report_leverage,
        }

# Configure enhanced logging with file rotation
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import uuid

def _setup_logging() -> logging.Logger:
    """Configure module-level logging without hijacking root handlers."""
    log_subdir = os.getenv('CRYPTO_FINDER_LOG_SUBDIR', 'long_term_crypto_finder').strip() or 'long_term_crypto_finder'
    base_logs_dir = Path('logs') / log_subdir
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

    logger_name = os.getenv('CRYPTO_FINDER_LOGGER_NAME', __name__).strip() or __name__
    logger = logging.getLogger(logger_name)
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
    llm_score: float = 0.0
    llm_confidence: str = ""
    llm_reason: str = ""
    llm_adjustment: float = 0.0

class LongTermCryptoFinder:
    """Main class for finding long-term crypto opportunities with comprehensive risk analysis."""
    REPORT_TITLE = "LONG-TERM CRYPTO OPPORTUNITIES ANALYSIS"
    FINDER_LABEL = "Long-Term Crypto Finder"

    # Configuration constants for risk calculation
    DEFAULT_TECH_WEIGHT = 0.45
    DEFAULT_FUND_WEIGHT = 0.35
    DEFAULT_MOM_WEIGHT = 0.20
    DEFAULT_SIGMOID_K = 1.0
    DEFAULT_RISK_LAMBDA = 1.2
    DEFAULT_VOL_FLOOR_USD = 50000
    DEFAULT_MCAP_FLOOR_USD = 1000000
    DEFAULT_RR_WEIGHT = 0.15
    DEFAULT_TREND_WEIGHT = 0.10
    DEFAULT_RR_SIGMOID_CENTER = 2.0
    DEFAULT_RR_SIGMOID_K = 1.1
    DEFAULT_TREND_ALIGN_SCALE = 1.5  # percent per day
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'LOW': 0.25,
        'MEDIUM_LOW': 0.40,
        'MEDIUM': 0.55,
        'MEDIUM_HIGH': 0.70,
        'HIGH': 0.85
    }
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
        # Auth presence flag (used to prefer authenticated product paths)
        self._has_auth = bool(self.api_key and self.api_secret)

        # Initialize Coinbase service
        self.coinbase_service = CoinbaseService(self.api_key, self.api_secret)
        self.historical_data = HistoricalData(self.coinbase_service.client)
        self.historical_data.set_force_refresh(bool(self.config.force_refresh_candles))
        if getattr(self.config, 'force_refresh_candles', False):
            try:
                self._cached_candles.cache_clear()  # type: ignore[attr-defined]
            except AttributeError:
                pass

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
        self._cb_lock = threading.Lock()
        self._cb_sem = threading.Semaphore(int(os.getenv("CRYPTO_CB_CONCURRENCY", "3")))
        self._cg_lock = threading.Lock()
        # Thread-local session holder
        self._tls = threading.local()

        # Dynamic product mapping (symbol -> product_id) populated lazily
        self._symbol_to_product: Dict[str, str] = {}
        # Lazy CoinGecko list cache in-memory
        self._coingecko_list: Optional[List[Dict[str, str]]] = None
        # Fast lookup index for CoinGecko symbol->record
        self._cg_index: Optional[Dict[str, Dict[str, str]]] = None
        self._cg_id_cache: Dict[str, str] = {}  # memo for symbol -> coingecko_id
        self._cg_neg_cache: set[str] = set()    # memo for symbols with no match

        # Rate limiting for Coinbase API
        self.request_delay = self.config.request_delay
        self.last_request_time = 0

        # Caching system
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = self.config.cache_ttl
        self._cache_lock = threading.Lock()
        
        # Thread pool for parallel processing
        self.max_workers = max(1, self.config.max_workers or (os.cpu_count() or 4))

        # Behavior flags
        self.offline = bool(self.config.offline)
        # Normalize side
        self.side = (self.config.side or 'both').lower()
        if self.side not in ('long', 'short', 'both'):
            self.side = 'both'

        self._max_risk_level = self._parse_risk_level(self.config.max_risk_level)

        # Weight overrides for setup quality adjustments (clamped to reasonable bounds)
        self.risk_reward_weight = float(
            np.clip(getattr(self.config, 'risk_reward_weight', self.DEFAULT_RR_WEIGHT), 0.0, 0.5)
        )
        self.trend_weight = float(
            np.clip(getattr(self.config, 'trend_weight', self.DEFAULT_TREND_WEIGHT), 0.0, 0.5)
        )

        self.llm_scorer: Optional[LLMScorer] = None
        if getattr(self.config, 'use_openai_scoring', False):
            try:
                llm_weight = float(np.clip(getattr(self.config, 'openai_weight', 0.25), 0.0, 1.0))
                llm_max_candidates = int(max(1, getattr(self.config, 'openai_max_candidates', 12) or 12))

                llm_temp_raw = getattr(self.config, 'openai_temperature', None)
                llm_temperature: Optional[float]
                if llm_temp_raw is None:
                    llm_temperature = None
                else:
                    llm_temperature = float(max(0.0, llm_temp_raw))

                llm_sleep = float(max(0.0, getattr(self.config, 'openai_sleep_seconds', 0.0)))
                self.llm_scorer = LLMScorer(
                    model=getattr(self.config, 'openai_model', 'gpt-5-mini') or 'gpt-5-mini',
                    weight=llm_weight,
                    max_candidates=llm_max_candidates,
                    temperature=llm_temperature,
                    sleep_seconds=llm_sleep,
                )
                if not getattr(self.llm_scorer, 'enabled', False):
                    self.llm_scorer = None
            except Exception as exc:
                logger.warning("Disabling OpenAI scoring due to initialisation error: %s", exc)
                self.llm_scorer = None

        finder_label = getattr(self, 'FINDER_LABEL', 'Long-Term Crypto Finder')
        logger.info(f"{finder_label} initialized with Coinbase API")
        logger.info(f"Configuration: {self.config.to_dict()}")

        # Counters for diagnostics
        self._vol_fallbacks = 0
        self._vol_total = 0
        
        # Validate API credentials
        self._validate_api_credentials()

    def _normalize_ts(self, t) -> int:
        """Return epoch seconds from seconds/ms/us or ISO string; 0 on failure."""
        try:
            # numeric path: s, ms, µs
            ts = int(str(t))
            if ts >= 10**15:
                ts //= 10**9
            elif ts >= 10**12:
                ts //= 10**3
            return ts
        except Exception:
            pass
        try:
            from dateutil import parser  # type: ignore
            return int(parser.isoparse(str(t)).timestamp())
        except Exception:
            try:
                return int(datetime.fromisoformat(str(t).replace('Z', '+00:00')).timestamp())
            except Exception:
                return 0

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

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Union[Dict, List]]:
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
                resp = getattr(e, 'response', None)
                code = getattr(resp, 'status_code', None)
                if code == 429:  # Rate limit exceeded
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                elif isinstance(code, int) and code >= 500:  # Server errors
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Server error {code}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    # Validation or client errors: surface server message and do not retry
                    server_msg = None
                    try:
                        server_msg = resp.json() if resp is not None else None
                    except Exception:
                        server_msg = getattr(resp, 'text', str(e))
                    logger.error(f"HTTP {code or 'N/A'} error: {server_msg}")
                    raise DataValidationError(f"HTTP {code or 'N/A'}: {server_msg}")
                    
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
        
    # --- Shared helpers (risk/position sizing/ranks) ---
    def _position_size_percentage(self, entry_price: float, stop_loss_price: float, atr_raw: float) -> float:
        """Account-currency position sizing with ATR floor and fee/slippage add-on.

        Uses env vars: CRYPTO_ACCOUNT_EQUITY_USD, CRYPTO_RISK_PER_TRADE, CRYPTO_POS_CAP_PCT,
        CRYPTO_ATR_SL_MULT, CRYPTO_FEE_BPS, CRYPTO_SLIPPAGE_BPS.
        """
        try:
            fee_bps = float(os.getenv("CRYPTO_FEE_BPS", "10"))
            slp_bps = float(os.getenv("CRYPTO_SLIPPAGE_BPS", "10"))
            fee_add = (fee_bps + slp_bps) / 10000.0 * entry_price
            atr_floor_mult = float(os.getenv("CRYPTO_ATR_SL_MULT", "0.5"))
            dist = max(
                entry_price * 1e-3,
                abs(entry_price - stop_loss_price),
                (atr_raw * atr_floor_mult) if atr_raw and atr_raw > 0 else 0.0,
                fee_add,
            )
            risk_pct = float(os.getenv("CRYPTO_RISK_PER_TRADE", "0.01"))  # 1% default
            cap_pct = float(os.getenv("CRYPTO_POS_CAP_PCT", "5.0"))  # cap percent of equity
            equity_env = os.getenv("CRYPTO_ACCOUNT_EQUITY_USD")
            if equity_env is not None:
                try:
                    eq = max(1.0, float(equity_env))
                    risk_usd = max(0.0, risk_pct * eq)
                    units = risk_usd / dist if dist > 0 else 0.0
                    pos_value = units * entry_price
                    return float(min(cap_pct, (pos_value / eq) * 100.0))
                except Exception:
                    pass
            # Fallback: percentage-of-portfolio approximation
            risk_frac = max(1e-6, dist / max(entry_price, 1e-9))
            return float(np.clip((risk_pct / risk_frac) * 100.0, 0.5, cap_pct))
        except Exception:
            return 1.0

    def _risk_reward_ratio(self, entry_price: float, stop_loss_price: float, take_profit_price: float, atr_raw: float, is_long: bool) -> float:
        """Compute risk/reward ratio with ATR floor and fee/slippage add-on.

        - dist incorporates min tick (0.1%), ATR floor, and fees/slippage.
        - reward is direction-aware: long uses (TP - entry), short uses (entry - TP).
        - clamps to [0, 10].
        """
        try:
            fee_bps = float(os.getenv("CRYPTO_FEE_BPS", "10"))
            slp_bps = float(os.getenv("CRYPTO_SLIPPAGE_BPS", "10"))
            fee_add = (fee_bps + slp_bps) / 10000.0 * entry_price
            atr_floor_mult = float(os.getenv("CRYPTO_ATR_SL_MULT", "0.5"))

            raw_risk = abs(entry_price - stop_loss_price)
            dist = max(entry_price * 1e-3, raw_risk, (atr_raw * atr_floor_mult) if atr_raw and atr_raw > 0 else 0.0, fee_add)

            if is_long:
                reward = max(0.0, take_profit_price - entry_price)
            else:
                reward = max(0.0, entry_price - take_profit_price)

            return float(min(10.0, (reward / dist) if dist > 0 else 0.0))
        except Exception:
            return 0.0

    def _apply_risk_haircut(self, analyzed_candidates: List["CryptoMetrics"]) -> None:
        """Compute cross-sectional ranks and continuous risk haircut in-place for candidates."""
        if not analyzed_candidates:
            return
        
        try:
            # Calculate component rankings
            r_tech, r_fund, r_mom = self._calculate_component_rankings(analyzed_candidates)
            
            # Calculate risk components
            risk_vol, risk_dd, risk_sh, risk_liq = self._calculate_risk_components(analyzed_candidates)
            
            # Combine risks and apply haircut
            risk_norm = self._normalize_risk_components(risk_vol, risk_dd, risk_sh, risk_liq)
            haircut = self._calculate_risk_haircut(risk_norm)
            
            # Apply final scoring
            self._apply_final_scoring(analyzed_candidates, r_tech, r_fund, r_mom, haircut, risk_norm)
            
        except Exception as e:
            logger.error(f"Error in risk haircut calculation: {e}")
            # Fallback: assign default scores
            self._apply_fallback_scoring(analyzed_candidates)

    def _calculate_component_rankings(self, candidates: List["CryptoMetrics"]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate normalized rankings for technical, fundamental, and momentum scores."""
        tech_scores = [self._safe_float(c.technical_score) for c in candidates]
        fund_scores = [self._safe_float(c.fundamental_score) for c in candidates]
        mom_scores = [self._safe_float(c.momentum_score) for c in candidates]
        
        r_tech = self._rank01(tech_scores, higher_is_better=True)
        r_fund = self._rank01(fund_scores, higher_is_better=True)
        r_mom = self._rank01(mom_scores, higher_is_better=True)
        
        return r_tech, r_fund, r_mom

    def _rank01(self, vals: List[float], higher_is_better: bool = True) -> List[float]:
        """Convert values to normalized rankings between 0 and 1."""
        n = len(vals)
        if n <= 1:
            return [0.5] * n
            
        try:
            arr = np.array(vals, dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Sort indices based on higher_is_better flag
            order = np.argsort(-arr if higher_is_better else arr)
            ranks = np.empty(n, dtype=float)
            ranks[order] = np.arange(1, n + 1, dtype=float)
            
            # Normalize to [0, 1]
            return list((ranks - 1.0) / max(1.0, n - 1.0))
            
        except Exception as e:
            logger.warning(f"Error in ranking calculation: {e}")
            return [0.5] * n

    def _calculate_risk_components(self, candidates: List["CryptoMetrics"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate individual risk components: volatility, drawdown, sharpe, and liquidity."""
        # Volatility risk
        vol_data = np.array([abs(self._safe_float(c.volatility_30d)) for c in candidates], dtype=float)
        vol_data = np.nan_to_num(vol_data, nan=0.0, posinf=0.0, neginf=0.0)
        risk_vol = self._calculate_sigmoid_risk(vol_data, invert=False)
        
        # Drawdown risk
        dd_data = np.array([abs(self._safe_float(c.max_drawdown)) for c in candidates], dtype=float)
        dd_data = np.nan_to_num(dd_data, nan=0.0, posinf=0.0, neginf=0.0)
        risk_dd = self._calculate_sigmoid_risk(dd_data, invert=False)
        
        # Sharpe risk (inverted - lower sharpe = higher risk)
        sharpe_data = np.array([self._safe_float(c.sharpe_ratio) for c in candidates], dtype=float)
        sharpe_data = np.nan_to_num(sharpe_data, nan=0.0, posinf=0.0, neginf=0.0)
        risk_sh = self._calculate_sigmoid_risk(sharpe_data, invert=True)
        
        # Liquidity risk
        risk_liq = self._calculate_liquidity_risk(candidates)
        
        return risk_vol, risk_dd, risk_sh, risk_liq

    def _calculate_sigmoid_risk(self, data: np.ndarray, invert: bool = False) -> np.ndarray:
        """Calculate sigmoid-based risk using z-score normalization."""
        if data.size == 0:
            return np.array([])
            
        try:
            # Calculate z-scores
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            
            if std_val == 0:
                z_scores = np.zeros_like(data)
            else:
                z_scores = (data - mean_val) / std_val
            
            # Apply sigmoid with configurable steepness
            k = float(os.getenv('CRYPTO_RISK_SIGMOID_K', str(self.DEFAULT_SIGMOID_K)))
            sigmoid_input = -z_scores if invert else z_scores
            risk = 1.0 / (1.0 + np.exp(-k * sigmoid_input))
            
            return np.clip(risk, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in sigmoid risk calculation: {e}")
            return np.zeros_like(data)

    def _calculate_liquidity_risk(self, candidates: List["CryptoMetrics"]) -> np.ndarray:
        """Calculate liquidity risk based on volume-to-market-cap ratio."""
        vol_floor = float(os.getenv('CRYPTO_LIQ_VOL_FLOOR_USD', str(self.DEFAULT_VOL_FLOOR_USD)))
        mc_floor = float(os.getenv('CRYPTO_LIQ_MCAP_FLOOR_USD', str(self.DEFAULT_MCAP_FLOOR_USD)))
        
        inv_vmc_values = []
        for c in candidates:
            mc = self._safe_float(c.market_cap)
            vol_usd = self._safe_float(c.volume_24h)
            
            if mc > 0 or vol_usd > 0:
                vmc_ratio = max(vol_usd, vol_floor) / max(mc, mc_floor)
                inv_vmc = 1.0 / max(vmc_ratio, 1e-9)
            else:
                inv_vmc = 0.0
                
            inv_vmc_values.append(inv_vmc)
        
        if not inv_vmc_values:
            return np.array([])
            
        try:
            inv_vmc_array = np.array(inv_vmc_values, dtype=float)
            inv_vmc_array = np.nan_to_num(inv_vmc_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            if inv_vmc_array.size == 0:
                return np.array([])
            
            # Handle infinite values more robustly
            finite_mask = np.isfinite(inv_vmc_array)
            if not np.any(finite_mask):
                return np.zeros_like(inv_vmc_array)
            
            # Normalize to [0, 1]
            min_val = np.min(inv_vmc_array[finite_mask])
            max_val = np.max(inv_vmc_array[finite_mask])
            
            if max_val > min_val:
                risk_liq = (inv_vmc_array - min_val) / (max_val - min_val)
            else:
                risk_liq = np.zeros_like(inv_vmc_array)
                
            return np.clip(risk_liq, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in liquidity risk calculation: {e}")
            return np.zeros(len(candidates))

    def _normalize_risk_components(self, risk_vol: np.ndarray, risk_dd: np.ndarray, 
                                 risk_sh: np.ndarray, risk_liq: np.ndarray) -> np.ndarray:
        """Combine and normalize risk components."""
        if risk_vol.size == 0:
            return np.array([])
            
        try:
            # Equal weight combination of risk components
            combined_risk = (risk_vol + risk_dd + risk_sh + risk_liq) / 4.0
            return np.clip(combined_risk, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in risk normalization: {e}")
            return np.zeros_like(risk_vol)

    def _calculate_risk_haircut(self, risk_norm: np.ndarray) -> np.ndarray:
        """Calculate risk haircut using exponential decay."""
        if risk_norm.size == 0:
            return np.array([])
            
        try:
            lambda_val = float(os.getenv('CRYPTO_RISK_LAMBDA', str(self.DEFAULT_RISK_LAMBDA)))
            haircut = np.exp(-lambda_val * risk_norm)
            return np.clip(haircut, 0.0, 1.0)
            
        except Exception as e:
            logger.warning(f"Error in haircut calculation: {e}")
            return np.ones_like(risk_norm)

    def _normalize_setup_components(
        self, candidates: List["CryptoMetrics"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize risk/reward and trend-alignment components for scoring adjustments."""
        if not candidates:
            return np.array([], dtype=float), np.array([], dtype=float)

        try:
            rr_values = np.array(
                [self._safe_float(getattr(c, 'risk_reward_ratio', 0.0)) for c in candidates],
                dtype=float,
            )
            rr_values = np.nan_to_num(rr_values, nan=0.0, posinf=10.0, neginf=0.0)

            rr_center = float(os.getenv('CRYPTO_RR_SIGMOID_CENTER', str(self.DEFAULT_RR_SIGMOID_CENTER)))
            rr_center = float(np.clip(rr_center, 0.5, 5.0))
            rr_k = float(os.getenv('CRYPTO_RR_SIGMOID_K', str(self.DEFAULT_RR_SIGMOID_K)))
            rr_k = float(np.clip(rr_k, 0.1, 5.0))
            rr_norm = 1.0 / (1.0 + np.exp(-rr_k * (rr_values - rr_center)))

            trend_bias: List[float] = []
            for candidate in candidates:
                raw_trend = self._safe_float(getattr(candidate, 'trend_strength', 0.0))
                side = (getattr(candidate, 'position_side', 'LONG') or 'LONG').upper()
                direction = 1.0 if side == 'LONG' else -1.0
                trend_bias.append(direction * raw_trend)

            trend_array = np.array(trend_bias, dtype=float)
            trend_array = np.nan_to_num(trend_array, nan=0.0, posinf=0.0, neginf=0.0)

            trend_scale = float(os.getenv('CRYPTO_TREND_ALIGN_SCALE', str(self.DEFAULT_TREND_ALIGN_SCALE)))
            trend_scale = float(max(0.1, trend_scale))
            trend_norm = 0.5 + 0.5 * np.tanh(trend_array / trend_scale)

            return np.clip(rr_norm, 0.0, 1.0), np.clip(trend_norm, 0.0, 1.0)

        except Exception as exc:
            logger.warning(f"Error normalizing setup components: {exc}")
            neutral = np.full(len(candidates), 0.5, dtype=float)
            return neutral.copy(), neutral

    def _apply_final_scoring(self, candidates: List["CryptoMetrics"], r_tech: List[float], 
                           r_fund: List[float], r_mom: List[float], haircut: np.ndarray, 
                           risk_norm: np.ndarray) -> None:
        """Apply final scoring with weights and risk adjustment."""
        # Use validated parameters
        params = self._validate_risk_parameters()
        tech_weight = params['tech_weight']
        fund_weight = params['fund_weight']
        mom_weight = params['mom_weight']
        
        try:
            # Calculate weighted score
            weighted_score = (
                tech_weight * np.array(r_tech, dtype=float)
                + fund_weight * np.array(r_fund, dtype=float)
                + mom_weight * np.array(r_mom, dtype=float)
            )
            weighted_score = np.nan_to_num(weighted_score, nan=0.0, posinf=0.0, neginf=0.0)

            effective_haircut = haircut if haircut.size else np.ones_like(weighted_score)
            base_score = weighted_score * effective_haircut

            rr_norm, trend_norm = self._normalize_setup_components(candidates)
            setup_adjustment = np.zeros_like(base_score)

            if rr_norm.size:
                setup_adjustment += self.risk_reward_weight * (rr_norm - 0.5)
            if trend_norm.size:
                setup_adjustment += self.trend_weight * (trend_norm - 0.5)

            enhanced_score = np.clip(base_score + setup_adjustment, 0.0, 1.0)
            final_scores = 100.0 * enhanced_score

            # Assign scores and risk levels
            for idx, candidate in enumerate(candidates):
                candidate.overall_score = float(max(0.0, min(100.0, final_scores[idx])))
                candidate.risk_score = float(np.clip(risk_norm[idx], 0.0, 1.0)) * 100.0
                candidate.risk_level = self._determine_risk_level(risk_norm[idx])
                if rr_norm.size:
                    setattr(candidate, 'risk_reward_bias', float(np.clip(rr_norm[idx], 0.0, 1.0)))
                if trend_norm.size:
                    setattr(candidate, 'trend_alignment', float(np.clip(trend_norm[idx], 0.0, 1.0)))
                setattr(candidate, 'score_adjustment_setup', float(setup_adjustment[idx]))
                
        except Exception as e:
            logger.error(f"Error in final scoring: {e}")
            self._apply_fallback_scoring(candidates)

    def _refine_scores_with_llm(self, candidates: List["CryptoMetrics"]) -> int:
        """Optionally refine top candidate scores using an OpenAI model."""
        if not candidates or not getattr(self, 'llm_scorer', None):
            return 0

        try:
            ranked = sorted(
                candidates,
                key=lambda c: getattr(c, 'overall_score', 0.0),
                reverse=True,
            )

            subset = []
            seen: set[str] = set()
            for candidate in ranked:
                candidate_id = f"{candidate.symbol}:{getattr(candidate, 'position_side', 'LONG')}"
                if candidate_id in seen:
                    continue
                payload = build_llm_payload(candidate)
                payload['base_score'] = float(getattr(candidate, 'overall_score', 0.0))
                payload['risk_level'] = getattr(candidate.risk_level, 'value', '')
                subset.append((candidate_id, candidate, payload))
                seen.add(candidate_id)
                if len(subset) >= self.llm_scorer.max_candidates:
                    break

            if not subset:
                return 0

            payloads = [payload for _, _, payload in subset]
            results = self.llm_scorer.score_candidates(payloads)  # type: ignore[union-attr]
            if not results:
                return 0

            adjusted = 0
            for candidate_id, candidate, _payload in subset:
                outcome = results.get(candidate_id)
                if not outcome:
                    continue
                orig_score = float(getattr(candidate, 'overall_score', 0.0))
                llm_score = float(outcome.get('llm_score', orig_score))
                combined = self.llm_scorer.combine_scores(orig_score, llm_score)
                setattr(candidate, 'overall_score_pre_llm', orig_score)
                candidate.llm_score = llm_score
                candidate.llm_confidence = str(outcome.get('confidence', ''))
                candidate.llm_reason = str(outcome.get('reason', ''))
                candidate.llm_adjustment = combined - orig_score
                candidate.overall_score = combined
                adjusted += 1
            return adjusted
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.warning(f"OpenAI scoring step skipped due to error: {exc}")
            return 0

    def _determine_risk_level(self, risk_value: float) -> RiskLevel:
        """Determine risk level based on normalized risk value using class thresholds."""
        risk_value = float(np.clip(risk_value, 0.0, 1.0))
        
        if risk_value < self.RISK_THRESHOLDS['LOW']:
            return RiskLevel.LOW
        elif risk_value < self.RISK_THRESHOLDS['MEDIUM_LOW']:
            return RiskLevel.MEDIUM_LOW
        elif risk_value < self.RISK_THRESHOLDS['MEDIUM']:
            return RiskLevel.MEDIUM
        elif risk_value < self.RISK_THRESHOLDS['MEDIUM_HIGH']:
            return RiskLevel.MEDIUM_HIGH
        elif risk_value < self.RISK_THRESHOLDS['HIGH']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _apply_fallback_scoring(self, candidates: List["CryptoMetrics"]) -> None:
        """Apply fallback scoring when main calculation fails."""
        logger.warning("Applying fallback scoring due to calculation errors")
        
        for candidate in candidates:
            candidate.overall_score = 50.0  # Neutral score
            candidate.risk_score = 50.0     # Medium risk
            candidate.risk_level = RiskLevel.MEDIUM

    def _safe_float(self, value: Union[float, int, None]) -> float:
        """Safely convert value to float with proper error handling."""
        if value is None:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _validate_risk_parameters(self) -> Dict[str, float]:
        """Validate and return risk calculation parameters with bounds checking."""
        params = {}
        
        # Validate weights sum to 1.0
        tech_weight = self.DEFAULT_TECH_WEIGHT
        fund_weight = self.DEFAULT_FUND_WEIGHT
        mom_weight = self.DEFAULT_MOM_WEIGHT
        
        weight_sum = tech_weight + fund_weight + mom_weight
        if abs(weight_sum - 1.0) > 1e-6:
            logger.warning(f"Weights don't sum to 1.0: {weight_sum}, normalizing")
            tech_weight /= weight_sum
            fund_weight /= weight_sum
            mom_weight /= weight_sum
        
        params['tech_weight'] = tech_weight
        params['fund_weight'] = fund_weight
        params['mom_weight'] = mom_weight
        
        # Validate other parameters
        params['sigmoid_k'] = max(0.1, min(10.0, self.DEFAULT_SIGMOID_K))
        params['risk_lambda'] = max(0.1, min(5.0, self.DEFAULT_RISK_LAMBDA))
        params['vol_floor'] = max(1000, self.DEFAULT_VOL_FLOOR_USD)
        params['mcap_floor'] = max(100000, self.DEFAULT_MCAP_FLOOR_USD)
        
        return params

    def _parse_risk_level(self, value: Optional[Union[str, RiskLevel]]) -> Optional[RiskLevel]:
        """Normalize user-supplied risk level sentinel to the RiskLevel enum."""
        if value is None:
            return None
        if isinstance(value, RiskLevel):
            return value
        try:
            candidate = str(value).strip().upper()
            if candidate in RiskLevel.__members__:
                return RiskLevel[candidate]
            for level in RiskLevel:
                if level.value == candidate:
                    return level
        except Exception:
            pass
        logger.warning(f"Ignoring unsupported risk level value: {value}")
        return None

    def _filter_by_max_risk_level(self, candidates: List["CryptoMetrics"]) -> List["CryptoMetrics"]:
        """Filter candidates whose risk level exceeds the configured maximum."""
        if not candidates:
            return candidates
        max_level = getattr(self, '_max_risk_level', None)
        if max_level is None:
            return candidates

        order = {level: idx for idx, level in enumerate(RiskLevel)}
        max_idx = order.get(max_level, len(order) - 1)
        filtered: List[CryptoMetrics] = []
        dropped = 0
        for candidate in candidates:
            level_attr = getattr(candidate, 'risk_level', None)
            if isinstance(level_attr, RiskLevel):
                level = level_attr
            else:
                level = self._parse_risk_level(level_attr)
            if level is None:
                level = RiskLevel.VERY_HIGH
            if order.get(level, len(order)) <= max_idx:
                filtered.append(candidate)
            else:
                dropped += 1
        if dropped:
            logger.info(
                f"Filtered {dropped} candidates exceeding risk level {max_level.name}; "
                f"{len(filtered)} remain."
            )
        return filtered
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
            if price is None:
                return 0.0
            if isinstance(price, float) and not np.isfinite(price):
                return 0.0
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
        products: List = []
        # Prefer authenticated product list when keys are present to avoid public 401 noise
        if getattr(self, "_has_auth", False):
            try:
                from coinbase.rest import products as cb_products  # type: ignore
                auth_resp = cb_products.get_products(self.coinbase_service.client)
                if isinstance(auth_resp, dict):
                    products = auth_resp.get("products", []) or []
                else:
                    products = getattr(auth_resp, "products", []) or []
            except Exception as e:
                logger.warning(f"Authenticated products fetch failed: {e}")
                products = []
        # If no auth or auth fetch empty, try public endpoint
        if not products:
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
            env_q = os.getenv('CRYPTO_QUOTES', 'USDC,USD,USDT')
            quotes = [q.strip().upper() for q in env_q.split(',') if q.strip()]
        quotes_set = set(quotes or ['USDC'])
        kept = 0
        total_online = 0
        order = {q: i for i, q in enumerate(quotes)}
        # If still empty, try public best bid/ask to derive available product_ids
        if not products:
            try:
                from coinbase.rest import products as cb_products  # type: ignore
                pbooks = cb_products.get_best_bid_ask(self.coinbase_service.client)
                books = []
                if isinstance(pbooks, dict):
                    books = pbooks.get('pricebooks', []) or []
                else:
                    books = getattr(pbooks, 'pricebooks', []) or []
                derived: Dict[str, Dict[str, str]] = {}
                for b in books:
                    pid = b.get('product_id') if isinstance(b, dict) else getattr(b, 'product_id', '')
                    if not pid or '-' not in pid:
                        continue
                    base, quote = pid.split('-', 1)
                    base = str(base or '').upper()
                    quote = str(quote or '').upper()
                    if quote not in quotes_set:
                        continue
                    cur = derived.get(base)
                    if (not cur) or (order.get(quote, 1e9) < order.get(cur.get('quote', 'ZZZ'), 1e9)):
                        derived[base] = {"product_id": pid, "base_name": base, "quote": quote}
                if derived:
                    result.update(derived)
            except Exception as e:
                logger.warning(f"Failed deriving products from best bid/ask: {e}")

        def _pget(obj, key: str):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _is_online(prod) -> bool:
            st = str(_pget(prod, "status") or '').lower()
            if st:
                return st == "online"
            td = _pget(prod, "trading_disabled")
            co = _pget(prod, "cancel_only")
            try:
                return (td is False) and (co is False)
            except Exception:
                return False

        for p in products:
            online = _is_online(p)
            if online:
                total_online += 1
            q = str((_pget(p, "quote_currency") or _pget(p, "quote_currency_id") or '')).upper()
            if online and q in quotes_set:
                base = str((_pget(p, "base_currency") or _pget(p, "base_currency_id") or '')).upper()
                if base:
                    cur = result.get(base)
                    candidate = {
                        "product_id": _pget(p, "product_id"),
                        "base_name": _pget(p, "base_name") or base,
                        "quote": q
                    }
                    if (not cur) or (order.get(q, 1e9) < order.get(cur.get("quote", "ZZZ"), 1e9)):
                        result[base] = candidate
                        kept += 1
        if not result:
            for sym in ["BTC", "ETH", "SOL", "ADA", "MATIC", "AVAX", "DOT", "LINK"]:
                result[sym] = {"product_id": f"{sym}-USDC", "base_name": sym}
        self._symbol_to_product = {k: v["product_id"] for k, v in result.items() if v.get("product_id")}
        skipped = max(0, total_online - kept)
        logger.info(f"Coinbase products filtered by quotes {sorted(quotes_set)}: kept {kept}, skipped {skipped}")
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
                        try:
                            self._cg_index = {item['symbol'].upper(): item for item in self._coingecko_list if item.get('symbol')}
                        except Exception:
                            self._cg_index = {}
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
        # memo hit (thread-safe)
        with self._cg_lock:
            if sym in self._cg_id_cache:
                return self._cg_id_cache[sym]
            if sym in self._cg_neg_cache:
                return None
        lst = self._load_coingecko_list()
        # fast path via index (exact symbol match)
        if self._cg_index and sym in self._cg_index:
            cid = self._cg_index[sym].get('id')
            if cid:
                with self._cg_lock:
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
                with self._cg_lock:
                    self._cg_id_cache[sym] = cid
                return cid
            with self._cg_lock:
                self._cg_neg_cache.add(sym)
            logger.debug(f"No CoinGecko id found for symbol {sym}")
            return None
        if len(candidates) == 1 or getattr(self, 'offline', False):
            with self._cg_lock:
                self._cg_id_cache[sym] = candidates[0]
                return self._cg_id_cache[sym]
        # When online, query markets to pick the largest by market cap
        try:
            markets = self._cg_markets(candidates)
            if isinstance(markets, list) and markets:
                markets.sort(key=lambda m: float(m.get('market_cap') or 0), reverse=True)
                cid = str(markets[0].get('id') or candidates[0])
                with self._cg_lock:
                    self._cg_id_cache[sym] = cid
                return cid
        except Exception:
            pass
        logger.debug(f"Ambiguous CoinGecko id for {sym}; picking first candidate")
        with self._cg_lock:
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

    def _prefetch_cg_markets(self, ids: List[str]) -> Dict[str, Dict]:
        """Prefetch markets data for ids and shape to our fields."""
        market_map: Dict[str, Dict] = {}
        try:
            data = self._cg_markets(ids)
            for m in data or []:
                cid = str(m.get('id', '') or '')
                if not cid:
                    continue
                market_map[cid] = {
                    'ath': m.get('ath'),
                    'ath_date': (m.get('ath_date') or '')[:10],
                    'atl': m.get('atl'),
                    'atl_date': (m.get('atl_date') or '')[:10],
                    'current_price_usd': m.get('current_price'),
                    'market_cap_usd': m.get('market_cap'),
                    'total_volume_usd': m.get('total_volume'),
                    'price_change_7d_pct': m.get('price_change_percentage_7d_in_currency') or 0,
                    'price_change_30d_pct': m.get('price_change_percentage_30d_in_currency') or 0,
                    'market_cap_rank': m.get('market_cap_rank'),
                    'last_updated': m.get('last_updated') or ''
                }
        except Exception:
            return {}
        return market_map

    # -------- Coinbase market cap (best-effort) --------
    def _extract_market_cap_from_cb_asset(self, rec: Dict) -> float:
        """Try to pull a USD market cap from a Coinbase asset record."""
        if not isinstance(rec, dict):
            return 0.0
        # common fields across potential CB asset endpoints
        for key in (
            'market_cap', 'market_cap_usd', 'market_cap_in_usd',
            # nested metrics paths
        ):
            try:
                val = rec.get(key)
                if val is not None:
                    return float(val)
            except Exception:
                continue
        # Sometimes nested in 'metrics' or 'latest'
        for parent in ('metrics', 'latest'):
            try:
                inner = rec.get(parent) or {}
                for key in ('market_cap', 'market_cap_usd', 'market_cap_in_usd'):
                    val = inner.get(key)
                    if val is not None:
                        return float(val)
            except Exception:
                continue
        return 0.0

    def _get_market_cap_from_coinbase(self, symbol: str) -> float:
        """Best-effort public Coinbase market cap for a symbol; returns 0.0 if unavailable."""
        if not symbol:
            return 0.0
        if getattr(self, 'offline', False):
            return 0.0
        sym = symbol.upper()
        # Try summary endpoint first
        try:
            url = "https://api.coinbase.com/v2/assets/summary"
            params = {"base": "USD", "limit": 500}
            data = self._make_cached_request(url, params)  # type: ignore
            if isinstance(data, dict):
                arr = data.get('data') or data.get('assets') or []
            elif isinstance(data, list):
                arr = data
            else:
                arr = []
            for rec in arr:
                try:
                    rec_sym = (rec.get('symbol') or rec.get('asset_symbol') or '').upper()
                    if rec_sym == sym:
                        mc = self._extract_market_cap_from_cb_asset(rec)
                        if mc and mc > 0:
                            return float(mc)
                except Exception:
                    continue
        except Exception:
            pass
        # Try search endpoint
        try:
            url = "https://api.coinbase.com/v2/assets/search"
            params = {"base": "USD", "query": sym, "limit": 50}
            data = self._make_cached_request(url, params)  # type: ignore
            arr = []
            if isinstance(data, dict):
                arr = data.get('data') or data.get('assets') or []
            elif isinstance(data, list):
                arr = data
            for rec in arr:
                try:
                    rec_sym = (rec.get('symbol') or rec.get('asset_symbol') or '').upper()
                    if rec_sym == sym:
                        mc = self._extract_market_cap_from_cb_asset(rec)
                        if mc and mc > 0:
                            return float(mc)
                except Exception:
                    continue
        except Exception:
            pass
        return 0.0

    def _prefetch_cb_assets_summary(self) -> Dict[str, Dict[str, Union[float, int, str]]]:
        """Fetch Coinbase assets summary once and build a map: SYMBOL -> fields.

        Returns a dictionary keyed by uppercased symbol containing:
          - market_cap_usd: float
          - market_cap_rank: int (best-effort)
          - name: str (if available)
        """
        if getattr(self, 'offline', False):
            return {}
        try:
            url = "https://api.coinbase.com/v2/assets/summary"
            params = {"base": "USD", "limit": 500}
            data = self._make_cached_request(url, params)
            records: List[Dict] = []
            if isinstance(data, dict):
                records = data.get('data') or data.get('assets') or []
            elif isinstance(data, list):
                records = data
            result: Dict[str, Dict[str, Union[float, int, str]]] = {}
            for rec in records or []:
                try:
                    sym = (rec.get('symbol') or rec.get('asset_symbol') or '').upper()
                    if not sym:
                        continue
                    mc = self._extract_market_cap_from_cb_asset(rec)
                    rank = int(rec.get('market_cap_rank') or rec.get('rank') or 0)
                    name = rec.get('name') or rec.get('asset_name') or ''
                    result[sym] = {
                        'market_cap_usd': float(mc) if mc is not None else 0.0,
                        'market_cap_rank': rank,
                        'name': str(name or ''),
                    }
                except Exception:
                    continue
            logger.info(f"Prefetched Coinbase assets summary for {len(result)} symbols")
            return result
        except Exception as e:
            logger.warning(f"Failed to prefetch Coinbase assets summary: {e}")
            return {}
    
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
        path.parent.mkdir(parents=True, exist_ok=True)
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
    
    def _make_cached_request(self, url: str, params: Optional[Dict] = None) -> Optional[Union[Dict, List]]:
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
            with self._cb_sem:
                candles = self.historical_data.get_historical_data(
                    product_id,
                    start_time,
                    current_time,
                    "ONE_HOUR",
                    force_refresh=self.config.force_refresh_candles,
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
            ts = self._normalize_ts(latest_candle.get('start') or latest_candle.get('time'))
            try:
                data_ts = datetime.fromtimestamp(ts, UTC).strftime('%Y-%m-%d %H:%M:%SZ') if ts else ''
            except Exception:
                data_ts = ''
            
            if current_price <= 0:
                logger.warning(f"Invalid price for {product_id}: {current_price}")
                return None

            # Get accurate ATH/ATL and market snapshot (prefer prefetch from markets)
            base_sym = product_id.split('-')[0]
            cg: Dict[str, Union[float, str, int]] = {}
            try:
                cid = self._coingecko_id_for_symbol(base_sym)
                pre = getattr(self, '_cg_prefetch_map', {}) or {}
                if cid and cid in pre:
                    cg = pre[cid]
                else:
                    cg = self._get_ath_atl_from_coingecko(base_sym)
            except Exception:
                cg = self._get_ath_atl_from_coingecko(base_sym)

            # Create a validated data structure
            # Prefer Coinbase market cap to avoid CoinGecko limitations
            cb_mc = self._get_market_cap_from_coinbase(base_sym)
            cg_mc_val = float(cg.get('market_cap_usd', 0) or 0.0)
            if cb_mc and cb_mc > 0:
                mc_value = cb_mc
                real_mc = True
            else:
                mc_value = cg_mc_val
                real_mc = cg_mc_val > 0.0
            # Liquidity: favor CoinGecko USD 24h volume if present and fresh; otherwise fallback to candles
            cg_vol_usd = self._sanitize_price(cg.get('total_volume_usd', 0)) if isinstance(cg, dict) else 0.0
            last_upd = ''
            try:
                last_upd = str(cg.get('last_updated') or '') if isinstance(cg, dict) else ''
            except Exception:
                last_upd = ''
            stale_sec = float(os.getenv('CRYPTO_CG_MARKETS_STALE_SEC', str(12 * 3600)))
            now_ts = time.time()
            upd_ts = self._normalize_ts(last_upd) if last_upd else 0
            cg_is_stale = (upd_ts == 0) or ((now_ts - float(upd_ts)) > stale_sec)
            vol_source = 'coingecko'
            if (cg_vol_usd is None) or (cg_vol_usd <= 0) or cg_is_stale:
                # fallback to candle-derived USD volume
                cg_vol_usd = self._calculate_usd_volume(candles)
                vol_source = 'candles'
                try:
                    self._vol_fallbacks += 1
                except Exception:
                    pass
            try:
                self._vol_total += 1
            except Exception:
                pass
            crypto_info = {
                'product_id': product_id,
                'symbol': product_id.split('-')[0],
                'name': self._get_crypto_name(product_id.split('-')[0]),
                'current_price': current_price,
                'price_change_24h': self._calculate_price_change(candles),
                # Prefer CoinGecko USD volume; fall back to estimated USD from candles
                'volume_24h': cg_vol_usd,
                'volume_24h_source': vol_source,
                # Prefer Coinbase/CG market cap; fallback to estimate only for display
                'market_cap': float(mc_value) if mc_value else self._estimate_market_cap(product_id.split('-')[0], current_price),
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

    def _filter_by_liquidity(self, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates by minimum 24h volume and volume/market-cap ratio."""

        min_volume = float(getattr(self.config, 'min_volume_24h', 0.0) or 0.0)
        min_ratio = float(getattr(self.config, 'min_volume_market_cap_ratio', 0.0) or 0.0)

        if min_volume <= 0 and min_ratio <= 0:
            return candidates

        allow_est_mc = os.getenv('CRYPTO_ALLOW_EST_MC', '0') in ('1', 'true', 'True')

        kept: List[Dict] = []
        dropped_low_volume = 0
        dropped_low_ratio = 0
        dropped_missing_mc = 0

        for record in candidates:
            try:
                volume_usd = float(record.get('volume_24h') or 0.0)
            except Exception:
                volume_usd = 0.0

            if min_volume > 0 and volume_usd < min_volume:
                dropped_low_volume += 1
                continue

            if min_ratio > 0:
                try:
                    market_cap = float(record.get('market_cap') or 0.0)
                except Exception:
                    market_cap = 0.0

                has_real_mc = bool(record.get('market_cap_is_real', False))
                if market_cap <= 0 or not np.isfinite(market_cap):
                    dropped_missing_mc += 1
                    continue

                if not has_real_mc and not allow_est_mc:
                    dropped_missing_mc += 1
                    continue

                ratio = volume_usd / market_cap if market_cap > 0 else 0.0
                if ratio < min_ratio:
                    dropped_low_ratio += 1
                    continue

            kept.append(record)

        logger.info(
            "Applied liquidity filter (min_volume=%s, min_vmc_ratio=%s): kept %s/%s (dropped_low_vol=%s, dropped_low_ratio=%s, dropped_missing_mc=%s)",
            f"${min_volume:,.0f}" if min_volume > 0 else 'n/a',
            min_ratio if min_ratio > 0 else 'n/a',
            len(kept),
            len(candidates),
            dropped_low_volume,
            dropped_low_ratio,
            dropped_missing_mc,
        )

        return kept

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

        # Optional early prefilter by Coinbase market cap before any candle pulls
        min_mc_prefilter = float(self.config.min_market_cap or 0)
        products_prefiltered = product_ids
        if min_mc_prefilter > 0:
            mc_map = self._prefetch_cb_assets_summary()
            if not mc_map:
                logger.warning(
                    "Coinbase assets summary returned 0 symbols; skipping market-cap prefilter to avoid false negatives. "
                    "Set CRYPTO_PREFILTER_KEEP_UNKNOWN=0 to enforce drop of unknowns."
                )
                products_prefiltered = product_ids
            else:
                kept, dropped_below, dropped_unknown = [], 0, 0
                keep_unknown = os.getenv('CRYPTO_PREFILTER_KEEP_UNKNOWN', '0') in ('1', 'true', 'True')
                for pid in product_ids:
                    try:
                        base = pid.split('-')[0].upper()
                        rec = mc_map.get(base)
                        mc = float((rec or {}).get('market_cap_usd') or 0.0)
                        if mc >= min_mc_prefilter:
                            kept.append(pid)
                        else:
                            if rec is None:
                                if keep_unknown:
                                    kept.append(pid)
                                else:
                                    dropped_unknown += 1
                            else:
                                dropped_below += 1
                    except Exception:
                        dropped_unknown += 1
                products_prefiltered = kept
                logger.info(
                    f"Prefiltered by Coinbase market cap >= ${min_mc_prefilter:,.0f}: kept {len(kept)}, "
                    f"dropped_below {dropped_below}, dropped_unknown {dropped_unknown}"
                )

        # Apply limit if given (post-prefilter)
        max_to_process = limit if (isinstance(limit, int) and limit > 0) else len(products_prefiltered)
        products_to_process = products_prefiltered[:max_to_process]

        # If nothing to process, exit early to avoid ThreadPoolExecutor(0) error
        if not products_to_process:
            logger.warning("No products to process after symbol selection and market-cap prefilter.")
            return []

        crypto_data: List[Dict] = []
        failed_products: List[str] = []
        
        # Prefetch CoinGecko markets snapshot for all symbols to reduce per-coin hits
        base_syms = [pid.split('-')[0] for pid in products_to_process]
        cg_ids: List[str] = []
        if not self.offline:
            for s in base_syms:
                cid = self._coingecko_id_for_symbol(s)
                if cid:
                    cg_ids.append(cid)
        self._cg_prefetch_map = self._prefetch_cg_markets(cg_ids) if cg_ids else {}

        # Use ThreadPoolExecutor for parallel processing (cap pool to workload)
        pool_workers = max(1, min(self.max_workers, len(products_to_process)))
        with ThreadPoolExecutor(max_workers=pool_workers) as executor:
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
            
        # Filter by minimum market cap if specified (real MC by default; estimated MC optional with separate threshold)
        min_mc = float(self.config.min_market_cap or 0)
        allow_est = os.getenv('CRYPTO_ALLOW_EST_MC', '0') in ('1','true','True')
        min_est_mc = float(os.getenv('CRYPTO_MIN_EST_MC', '0') or 0)
        if min_mc > 0:
            pre_filter_count = len(crypto_data)
            filtered: List[Dict] = []
            dropped_no_mc = 0
            dropped_below_real = 0
            dropped_below_est = 0
            for c in crypto_data:
                mc_val = float(c.get('market_cap', 0) or 0)
                is_real = bool(c.get('market_cap_is_real', False))
                if is_real:
                    if mc_val >= min_mc:
                        filtered.append(c)
                    else:
                        dropped_below_real += 1
                else:
                    if not allow_est:
                        dropped_no_mc += 1
                        continue
                    if mc_val >= min_est_mc:
                        filtered.append(c)
                    else:
                        dropped_below_est += 1
            crypto_data = filtered
            logger.info(
                f"Applied market cap filter (real >= ${min_mc:,.0f}, est >= ${min_est_mc:,.0f} if allowed): "
                f"{len(crypto_data)}/{pre_filter_count} remaining; drops — no_mc:{dropped_no_mc}, real_below:{dropped_below_real}, est_below:{dropped_below_est}"
            )

        crypto_data = self._filter_by_liquidity(crypto_data)

        # Log liquidity volume fallback rate (CG -> candles)
        try:
            if self._vol_total > 0:
                pct_fb = 100.0 * float(self._vol_fallbacks) / float(self._vol_total)
                logger.info(
                    f"Liquidity volume source: fallback_to_candles={self._vol_fallbacks}/{self._vol_total} ({pct_fb:.1f}%)"
                )
        except Exception:
            pass

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
                base_vol = Decimal(str(c.get('volume', 0) or c.get('volume_in_base', 0) or c.get('volumeBase', 0) or '0'))
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
                    base_vol = Decimal(str(c.get('volume', 0) or c.get('volume_in_base', 0) or c.get('volumeBase', 0) or '0'))
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
                ts_i = self._normalize_ts(candle.get('start') or candle.get('time'))
                ts_dt = datetime.fromtimestamp(ts_i, UTC) if ts_i else None
                df_data.append({
                    'timestamp': ts_dt,
                    'price': float(candle['close']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'open': float(candle['open']),
                    'volume': float(candle.get('volume', 0))
                })

            df = pd.DataFrame(df_data)
            df = df.dropna(subset=['timestamp'])
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
            with self._cb_sem:
                data = self.historical_data.get_historical_data(
                    product_id,
                    start_time,
                    end_time,
                    granularity,
                    force_refresh=self.config.force_refresh_candles,
                )
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
            # Basic cleaning to handle NaNs/Infs at head/tail (use ffill/bfill to avoid deprecated 'method' kw)
            df = df.copy()
            cols = ['price', 'high', 'low', 'open', 'volume']
            df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
            df[cols] = df[cols].ffill().bfill()
            df['volume'] = df['volume'].fillna(0.0)

            prices = df['price'].values
            # Simple returns
            rets_raw = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([])

            # Winsorize returns to reduce outlier impact
            def _winsorize(a: np.ndarray, p: float = 0.01) -> np.ndarray:
                if a.size == 0:
                    return a
                lo, hi = np.quantile(a, [p, 1 - p])
                return np.clip(a, lo, hi)

            returns = _winsorize(rets_raw, 0.01)

            # Daily vol using >=90 bars if available; guard with EWMA when short
            if len(returns) >= 2:
                if len(returns) >= 90:
                    win_rets = returns[-90:]
                elif len(returns) >= 30:
                    win_rets = returns[-30:]
                else:
                    win_rets = returns
                std_win = float(np.std(win_rets, ddof=1)) if len(win_rets) >= 2 else 0.0
                ewma_std = float(pd.Series(returns).ewm(span=20, adjust=False).std().iloc[-1]) if len(returns) >= 5 else std_win
                daily_vol = max(std_win, ewma_std) if np.isfinite(ewma_std) else std_win
            else:
                daily_vol = 0.0
            volatility_30d = daily_vol * float(np.sqrt(365))

            # Sharpe ratio (annualized), >=90-bar window when possible, with winsorized returns
            risk_free_rate = float(self.config.risk_free_rate)
            if len(returns) >= 2:
                if len(returns) >= 90:
                    sr_rets = returns[-90:]
                else:
                    sr_rets = returns
                mean_daily = float(np.mean(sr_rets) - risk_free_rate / 365.0)
                std_daily = float(np.std(sr_rets, ddof=1))
                std_daily = max(std_daily, 1e-8)
                sharpe_ratio = (mean_daily * 365.0 / (std_daily * np.sqrt(365.0))) if std_daily > 0 else 0.0
            else:
                sharpe_ratio = 0.0

            # Sortino ratio (downside std, ddof=1), same >=90 window preference
            if len(returns) >= 2:
                sr_excess = returns[-90:] if len(returns) >= 90 else returns
                excess = sr_excess - risk_free_rate / 365.0
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
                    eps = 1e-9
                    macd_cross = (
                        np.sign(hist_series[-1]) != np.sign(hist_series[-2]) and
                        (abs(hist_series[-1]) > 3*eps or abs(hist_series[-2]) > 3*eps)
                    )
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

            # Trend strength (normalized slope over 60 bars)
            if len(prices) >= 60:
                x = np.arange(60.0)
                p = prices[-60:]
                slope = np.polyfit(x, p, 1)[0]
                trend_strength = float(slope / max(np.mean(p), 1e-8)) * 100.0
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
            val = float(np.nan_to_num(atr[-1] / period, nan=0.0, posinf=0.0, neginf=0.0)) if np.isfinite(atr[-1]) else 0.0
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

            val = float(np.nan_to_num(adx[-1], nan=0.0, posinf=0.0, neginf=0.0))
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
            if avg_loss == 0:
                if avg_gain == 0:
                    return 50.0
                return 100.0
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
        Momentum via slope t-stat of log price over 60–90 bars.
        Returns a 0–100 score using a logistic map of the t-stat.
        """
        try:
            n = len(df)
            if n < 60:
                return 50.0
            win = 90 if n >= 90 else 60
            y = np.log(df['price'].values[-win:])
            x = np.arange(win, dtype=float)
            x_mean = x.mean()
            y_mean = y.mean()
            Sxx = np.sum((x - x_mean) ** 2)
            if Sxx <= 0:
                return 50.0
            b = np.sum((x - x_mean) * (y - y_mean)) / Sxx
            y_hat = y_mean + b * (x - x_mean)
            resid = y - y_hat
            dof = max(win - 2, 1)
            s2 = float(np.sum(resid ** 2) / dof)
            se_b = float(np.sqrt(s2 / Sxx)) if Sxx > 0 else 0.0
            t = float(b / se_b) if se_b > 0 else 0.0
            score = 100.0 / (1.0 + np.exp(-t / 2.0))
            return float(max(0.0, min(100.0, score)))
        except Exception as e:
            logger.error(f"Error calculating momentum score (t-stat): {str(e)}")
            return 50.0

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
            if not coin_data.get('market_cap_is_real', False):
                market_cap = 0
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
            real_mc = market_cap if market_cap and market_cap > 0 else None
            volume_to_market_cap = (float(volume_24h) / float(real_mc)) if real_mc else 0.0

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

            # Optionally cap volume score when market cap is not real
            if not real_mc:
                volume_score = min(volume_score, 60)
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

            # Volatility risk (annualized fraction). Broader bands for crypto.
            volatility = metrics.get('volatility_30d', 0)
            if volatility < 0.6:  # <60%
                vol_risk = 20
            elif volatility < 1.0:  # <100%
                vol_risk = 40
            elif volatility < 1.5:  # <150%
                vol_risk = 60
            elif volatility < 2.0:  # <200%
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

            # Overall score is computed cross-sectionally in find_best_opportunities
            overall_score = 0.0

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
            # Overall score is computed cross-sectionally in find_best_opportunities
            overall_score = 0.0
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
            # Overall score is computed cross-sectionally in find_best_opportunities
            short_overall = 0.0
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
                'tickers': 'false',
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
            if atr_raw > 0:
                stop_loss_methods.append(entry_price - 2.0 * atr_raw)

            # Method 2: Recent low support (10-day low)
            if len(df) >= 10:
                recent_low = df['low'].tail(10).min()
                support_stop = recent_low * 0.98  # 2% below recent low
                stop_loss_methods.append(support_stop)

            # Method 3: Percentage-based stop via DAILY vol thresholds
            daily_vol = float(technical_metrics.get('daily_vol_30d', 0.0) or 0.0)
            if daily_vol >= 0.04:
                pct = 0.08
            elif daily_vol >= 0.02:
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

            # Risk-reward via shared helper
            risk_reward_ratio = self._risk_reward_ratio(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                atr_raw=atr_raw,
                is_long=True,
            )

            # Position sizing via shared helper
            position_size_percentage = self._position_size_percentage(entry_price, stop_loss_price, atr_raw)

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
            if atr_raw > 0:
                stop_loss_methods.append(entry_price + 2.0 * atr_raw)

            if len(df) >= 10:
                recent_high = df['high'].tail(10).max()
                resistance_stop = recent_high * 1.02  # 2% above recent high
                stop_loss_methods.append(resistance_stop)

            daily_vol = float(technical_metrics.get('daily_vol_30d', 0.0) or 0.0)
            if daily_vol >= 0.04:
                pct = 0.08
            elif daily_vol >= 0.02:
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

            # Risk-reward via shared helper
            risk_reward_ratio = self._risk_reward_ratio(
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                atr_raw=atr_raw,
                is_long=False,
            )

            position_size_percentage = self._position_size_percentage(entry_price, stop_loss_price, atr_raw)

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
        """Technical score with fixed weights.

        Weights: RSI 0.25, MACD 0.05, BB 0.20, Trend 0.30, Momentum 0.20.
        MACD cross: ±5 as a tie-breaker.
        """
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

            score += rsi_score * 0.25

            # MACD score
            macd = technical_metrics['macd_signal']
            if macd == 'BULLISH':
                macd_score = 80
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:
                macd_score = 40

            # Down-weight MACD to reduce overlap with trend/momentum
            score += macd_score * 0.05

            # Bollinger Bands score
            bb = technical_metrics['bb_position']
            if bb == 'OVERSOLD':
                bb_score = 85
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:  # OVERBOUGHT
                bb_score = 45

            score += bb_score * 0.20

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

            score += trend_score * 0.30

            # Momentum contribution (t-stat mapped 0–100)
            score += max(0.0, min(100.0, float(momentum_score))) * 0.20

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
        """Technical score for SHORT bias (inverted) with fixed weights.

        Weights: RSI 0.25 (overbought better), MACD 0.05 (bearish better),
        BB 0.20 (overbought better), Trend 0.30 (more negative better),
        Momentum 0.20 (use 100 - long momentum). MACD cross: ±5 inverted.
        """
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
            score += rsi_score * 0.25

            # MACD: bearish preferred
            macd = technical_metrics.get('macd_signal', 'NEUTRAL')
            if macd == 'BEARISH':
                macd_score = 85
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:  # BULLISH
                macd_score = 40
            score += macd_score * 0.05

            # Bollinger Bands: overbought preferred for shorts
            bb = technical_metrics.get('bb_position', 'NEUTRAL')
            if bb == 'OVERBOUGHT':
                bb_score = 85
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:  # OVERSOLD -> worse for shorts
                bb_score = 45
            score += bb_score * 0.20

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
            score += trend_score * 0.30

            # Short momentum from long momentum
            short_mom = max(0.0, min(100.0, 100.0 - float(momentum_score_long)))
            score += short_mom * 0.20

            # MACD cross small bonus/penalty (inverted for shorts)
            macd_cross = bool(technical_metrics.get('macd_cross', False))
            if macd_cross:
                macd_hist = float(technical_metrics.get('macd_hist', 0.0) or 0.0)
                score += 5 if macd_hist < 0 else -5

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

        def _analyze_one(coin: Dict) -> List[CryptoMetrics]:
            try:
                symbol = coin.get('symbol', 'UNKNOWN')
                name = coin.get('name', symbol)
                logger.info(f"Analyzing {symbol} ({name})")
                product_id = coin['product_id']
                df = self.get_historical_data(product_id, days=self.config.analysis_days)
                if df is None or len(df) < 30:
                    logger.warning(f"Insufficient historical data for {symbol}")
                    return []
                tech = self.calculate_technical_indicators(df)
                mom = self.calculate_momentum_score(df)
                chg = self._calculate_price_changes_from_history(df)

                results: List[CryptoMetrics] = []
                if self.side in ('long', 'both'):
                    long_metrics = self._build_long_metrics(coin, df, tech, mom, chg)
                    if long_metrics and getattr(long_metrics, 'risk_reward_ratio', 0.0) >= 2.0:
                        results.append(long_metrics)
                if self.side in ('short', 'both'):
                    short_metrics = self._build_short_metrics(coin, df, tech, mom, chg)
                    if short_metrics and getattr(short_metrics, 'risk_reward_ratio', 0.0) >= 2.0:
                        results.append(short_metrics)
                return results
            except Exception as e:
                try:
                    logger.error(f"Analyze failed for {coin.get('symbol','?')}: {e}")
                except Exception:
                    logger.error(f"Analyze failed for unknown coin: {e}")
                return []

        # Parallelize analysis to speed up runs, respecting API throttles via internal semaphores
        coins_to_process = crypto_list[:limit]
        pool_workers = max(1, min(self.max_workers, len(coins_to_process)))
        total_coins = len(coins_to_process)
        completed_count = 0
        error_count = 0
        failed_symbols = []
        
        if pool_workers == 1:
            # Keep ordering deterministic in single-thread path with progress reporting
            logger.info(f"Analyzing {total_coins} cryptocurrencies (single-threaded)")
            for i, coin in enumerate(coins_to_process, 1):
                try:
                    result = _analyze_one(coin)
                    analyzed_candidates.extend(result)
                    completed_count += 1
                    if i % 5 == 0 or i == total_coins:  # Report every 5 coins or at end
                        progress_pct = (i / total_coins) * 100
                        logger.info(f"Progress: {i}/{total_coins} ({progress_pct:.1f}%) - {len(analyzed_candidates)} candidates found")
                except Exception as e:
                    error_count += 1
                    symbol = coin.get('symbol', 'UNKNOWN')
                    failed_symbols.append(symbol)
                    logger.error(f"Failed to analyze {symbol}: {e}")
        else:
            logger.info(f"Analyzing {total_coins} cryptocurrencies using {pool_workers} workers")
            with ThreadPoolExecutor(max_workers=pool_workers) as ex:
                # Submit all futures with symbol mapping for better error reporting
                future_to_symbol = {ex.submit(_analyze_one, coin): coin.get('symbol', 'UNKNOWN') for coin in coins_to_process}
                
                for fut in as_completed(future_to_symbol):
                    symbol = future_to_symbol[fut]
                    try:
                        result = fut.result() or []
                        analyzed_candidates.extend(result)
                        completed_count += 1
                        
                        # Report progress every 5 completions or at end
                        if completed_count % 5 == 0 or completed_count == total_coins:
                            progress_pct = (completed_count / total_coins) * 100
                            logger.info(f"Progress: {completed_count}/{total_coins} ({progress_pct:.1f}%) - {len(analyzed_candidates)} candidates found")
                            
                    except Exception as e:
                        error_count += 1
                        failed_symbols.append(symbol)
                        logger.error(f"Worker error analyzing {symbol}: {e}")
        
        # Summary of analysis results
        if error_count > 0:
            logger.warning(f"Analysis completed with {error_count} errors. Failed symbols: {', '.join(failed_symbols[:10])}" + 
                          (f" and {len(failed_symbols) - 10} more..." if len(failed_symbols) > 10 else ""))
        else:
            logger.info(f"Analysis completed successfully for all {completed_count} cryptocurrencies")
        
        logger.info(f"Found {len(analyzed_candidates)} total candidates (long/short positions) for ranking")

        # Cross-sectional ranks and continuous risk re-scaling
        if analyzed_candidates:
            self._apply_risk_haircut(analyzed_candidates)
            llm_adjusted = self._refine_scores_with_llm(analyzed_candidates)
            if llm_adjusted:
                logger.info("OpenAI scoring refined %s candidates", llm_adjusted)

        # Filter by minimum score if requested (apply after rank-based recompute)
        min_score = float(self.config.min_overall_score or 0)
        if min_score > 0:
            analyzed_candidates = [c for c in analyzed_candidates if c and float(c.overall_score) >= min_score]

        analyzed_candidates = self._filter_by_max_risk_level(analyzed_candidates)

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
        # Telemetry: echo liquidity volume fallback ratio at end of run
        try:
            if getattr(self, '_vol_total', 0) > 0:
                pct_fb = 100.0 * float(getattr(self, '_vol_fallbacks', 0)) / float(getattr(self, '_vol_total', 0))
                logger.info(
                    f"Liquidity volume source (run): fallback_to_candles={self._vol_fallbacks}/{self._vol_total} ({pct_fb:.1f}%)"
                )
        except Exception:
            pass

        logger.info(
            f"Analysis complete. Found {len(top_results)} top opportunities (LONG={long_count}, SHORT={short_count})."
        )
        return top_results

    def _macd_short_phrase(self, crypto: CryptoMetrics) -> str:
        """Create a concise MACD phrase tailored to the trade side."""
        side = (getattr(crypto, 'position_side', 'LONG') or 'LONG').upper()
        macd_raw = (getattr(crypto, 'macd_signal', '') or '').strip().upper()

        if not macd_raw:
            return 'MACD flat'

        if macd_raw == 'BULLISH':
            return 'bullish MACD support' if side == 'LONG' else 'bullish MACD fade'
        if macd_raw == 'BEARISH':
            return 'bearish MACD rebound' if side == 'LONG' else 'bearish MACD momentum'
        if macd_raw == 'NEUTRAL':
            return 'MACD neutral'

        return f"{macd_raw.lower()} MACD"

    def _format_short_summary(self, crypto: CryptoMetrics, index: int) -> str:
        """Render a short single-line summary for quick scanning."""
        side_lower = (getattr(crypto, 'position_side', 'LONG') or 'LONG').lower()

        score = getattr(crypto, 'overall_score', float('nan'))
        score_text = f"{score:.2f}" if np.isfinite(score) else "n/a"

        rr = getattr(crypto, 'risk_reward_ratio', float('nan'))
        rr_text = f"{rr:.1f}× RR" if np.isfinite(rr) else "RR n/a"

        rsi = getattr(crypto, 'rsi_14', float('nan'))
        rsi_text = f"RSI {rsi:.0f}" if np.isfinite(rsi) else "RSI n/a"

        macd_text = self._macd_short_phrase(crypto)

        risk_level = getattr(crypto, 'risk_level', RiskLevel.MEDIUM)
        risk_text = f"risk {risk_level.value.replace('_', ' ').lower()}"

        trend = getattr(crypto, 'trend_strength', float('nan'))
        trend_text = ""
        if np.isfinite(trend) and abs(trend) >= 0.1:
            trend_text = f"trend {trend:+.2f}%/d"

        segments = [rr_text, rsi_text, macd_text, risk_text]
        if trend_text:
            segments.append(trend_text)

        detail = "; ".join(filter(None, segments))
        return f"{index}. Summary: {crypto.symbol} {side_lower} – score {score_text}, {detail}."

    def _compute_position_pnl(self, crypto: CryptoMetrics) -> Optional[Dict[str, float]]:
        """Return dollarised TP/SL projections for a standardised position size."""

        try:
            notional = float(getattr(self.config, 'report_position_notional', 1000.0))
        except Exception:
            notional = 1000.0

        try:
            leverage = float(getattr(self.config, 'report_leverage', 50.0))
        except Exception:
            leverage = 50.0

        if not np.isfinite(notional) or notional <= 0:
            return None

        entry_price = float(getattr(crypto, 'entry_price', float('nan')))
        take_profit_price = float(getattr(crypto, 'take_profit_price', float('nan')))
        stop_loss_price = float(getattr(crypto, 'stop_loss_price', float('nan')))

        if not np.isfinite(entry_price) or entry_price <= 0:
            return None
        if not np.isfinite(take_profit_price) or not np.isfinite(stop_loss_price):
            return None

        side = (getattr(crypto, 'position_side', 'LONG') or 'LONG').upper()
        side_factor = -1.0 if side == 'SHORT' else 1.0

        tp_return = side_factor * ((take_profit_price / entry_price) - 1.0)
        sl_return = side_factor * ((stop_loss_price / entry_price) - 1.0)

        if not np.isfinite(tp_return) or not np.isfinite(sl_return):
            return None

        tp_pnl = notional * tp_return
        sl_pnl = notional * sl_return

        if not np.isfinite(tp_pnl) or not np.isfinite(sl_pnl):
            return None

        effective_leverage = leverage if leverage and leverage > 0 else 1.0
        margin_required = notional / effective_leverage

        quantity = notional / entry_price

        return {
            'notional': notional,
            'leverage': effective_leverage,
            'margin': margin_required,
            'tp_pnl': tp_pnl,
            'sl_pnl': sl_pnl,
            'tp_return': tp_return,
            'sl_return': sl_return,
            'quantity': quantity,
            'tp_price_move': side_factor * (take_profit_price - entry_price),
            'sl_price_move': side_factor * (stop_loss_price - entry_price),
        }

    def print_results(self, results: List[CryptoMetrics], stream: Optional[TextIO] = None):
        """Print formatted analysis results to the desired stream."""

        destination = stream or sys.stdout

        def _write(text: str = "") -> None:
            print(text, file=destination)

        _write()
        _write("=" * 100)
        title = getattr(self, 'REPORT_TITLE', 'LONG-TERM CRYPTO OPPORTUNITIES ANALYSIS')
        _write(title)
        _write("=" * 100)
        # Use UTC for deterministic timestamps
        _write(f"Generated on (UTC): {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}Z")
        _write(f"Total opportunities listed: {len(results)}")
        _write("=" * 100)

        short_summaries: List[str] = []

        for i, crypto in enumerate(results, 1):
            _write()
            _write(f"{i}. {crypto.symbol} ({crypto.name}) — {crypto.position_side}")
            _write("-" * 50)
            if getattr(crypto, 'data_timestamp_utc', ''):
                _write(f"Data Timestamp (UTC): {crypto.data_timestamp_utc}")
            _write(f"Price: ${crypto.current_price:.6f}")
            rank_str = f"#{crypto.market_cap_rank}" if getattr(crypto, 'market_cap_rank', 0) else "N/A"
            _write(f"Market Cap: ${crypto.market_cap:,.0f} (Rank {rank_str})")
            _write(f"24h Volume: ${crypto.volume_24h:,.0f}")
            _write(f"24h Change: {crypto.price_change_24h:.2f}%")
            _write(f"7d Change: {crypto.price_change_7d:.2f}%")
            _write(f"30d Change: {crypto.price_change_30d:.2f}%")
            _write(f"ATH: ${crypto.ath_price:.2f} (Date: {crypto.ath_date or 'N/A'})")
            _write(f"ATL: ${crypto.atl_price:.6f} (Date: {crypto.atl_date or 'N/A'})")
            _write(f"Volatility (30d, ann.): {crypto.volatility_30d*100:.1f}%")
            _write(f"Sharpe Ratio: {crypto.sharpe_ratio:.2f}")
            _write(f"Sortino Ratio: {crypto.sortino_ratio:.2f}")
            # Display drawdown with its actual sign
            _write(f"Max Drawdown: {crypto.max_drawdown * 100:.2f}%")
            _write(f"RSI (14): {crypto.rsi_14:.1f}")
            _write(f"MACD Signal: {crypto.macd_signal}")
            _write(f"BB Position: {crypto.bb_position}")
            _write(f"Trend Strength: {crypto.trend_strength:.2f}% per day")
            _write(f"Momentum Score: {crypto.momentum_score:.1f}/100")
            _write(f"Fundamental Score: {crypto.fundamental_score:.1f}/100")
            _write(f"Technical Score: {crypto.technical_score:.1f}/100")
            _write(f"Risk Score: {crypto.risk_score:.1f}/100")
            _write(f"Risk Level: {crypto.risk_level.value}")
            _write(f"Overall Score: {crypto.overall_score:.2f}/100")
            if getattr(crypto, 'llm_score', 0.0):
                conf = getattr(crypto, 'llm_confidence', '') or 'N/A'
                _write(f"LLM Score: {crypto.llm_score:.2f}/100 (confidence: {conf})")
                reason = getattr(crypto, 'llm_reason', '')
                if reason:
                    _write(f"LLM Insight: {reason}")

            # Trading Levels Section
            _write()
            _write(f"💼 TRADING LEVELS ({crypto.position_side}):")
            _write(f"Entry Price: ${crypto.entry_price:.6f}")
            _write(f"Stop Loss: ${crypto.stop_loss_price:.6f}")
            _write(f"Take Profit: ${crypto.take_profit_price:.6f}")
            _write(f"Risk:Reward Ratio: {crypto.risk_reward_ratio:.1f}:1")
            _write(f"Recommended Position Size: {crypto.position_size_percentage:.1f}% of portfolio")

            pnl_profile = self._compute_position_pnl(crypto)
            if pnl_profile:
                leverage_text = f"{pnl_profile['leverage']:.1f}x" if pnl_profile['leverage'] != 1 else "1x"
                _write(
                    f"Std. Position Notional: ${pnl_profile['notional']:,.2f} at {leverage_text} leverage (margin ${pnl_profile['margin']:,.2f})"
                )
                _write(
                    f"Potential TP P&L: ${pnl_profile['tp_pnl']:,.2f} ({pnl_profile['tp_return']*100:.2f}%)"
                )
                _write(
                    f"Potential SL P&L: ${pnl_profile['sl_pnl']:,.2f} ({pnl_profile['sl_return']*100:.2f}%)"
                )

            short_summaries.append(self._format_short_summary(crypto, i))

        if short_summaries:
            _write()
            _write("Short-Line Summaries")
            _write("-" * 50)
            for line in short_summaries:
                _write(line)

PROFILE_PRESETS = {
    "default": {},
    "wide": {
        "limit": 400,
        "max_results": 20,
        "max_workers": 12,
        "analysis_days": 90,
    },
}


def _positive_int(value: str) -> int:
    """Argparse type that enforces a strictly positive integer."""

    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - argparse manages TypeError
        raise argparse.ArgumentTypeError(f"Expected integer, received '{value}'") from exc

    if converted <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")

    return converted


def _positive_float(value: str) -> float:
    """Argparse type that enforces a positive float."""

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Expected number, received '{value}'") from exc

    if converted <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive number")

    return converted


def make_risk_level_validator(valid_levels: Set[str]) -> Callable[[str], str]:
    """Return a validator that normalises and checks risk level choices."""

    normalized_levels = {level.strip().upper() for level in valid_levels}

    def _validator(value: str) -> str:
        normalised = value.strip().upper()
        if normalised not in normalized_levels:
            choices = ', '.join(sorted(normalized_levels))
            raise argparse.ArgumentTypeError(
                f"Invalid risk level '{value}'. Choose from: {choices}."
            )
        return normalised

    return _validator


def build_cli_parser(
    env_defaults: CryptoFinderConfig,
    default_limit: int,
    profile_default: str,
    default_max_risk: Optional[str],
    risk_level_type: Callable[[str], str],
    profile_presets: Optional[Dict[str, Dict[str, int]]] = None,
) -> argparse.ArgumentParser:
    """Construct the CLI parser so tests can reuse it with injected defaults."""

    presets = profile_presets or PROFILE_PRESETS
    parser = argparse.ArgumentParser(
        description='Find the best long-term cryptocurrency opportunities'
    )
    parser.add_argument(
        '--profile',
        choices=sorted(presets.keys()),
        default=profile_default,
        help=f"Preset bundle of frequently used parameters (default: {profile_default})",
    )
    parser.add_argument(
        '--plain-output',
        type=Path,
        help='Write a formatted text report to this path (excludes log lines)',
    )
    parser.add_argument(
        '--suppress-console-logs',
        action='store_true',
        help='Disable console log handler for cleaner stdout piping',
    )
    parser.add_argument(
        '--limit',
        type=_positive_int,
        default=None,
        help=(
            "Number of top cryptocurrencies to analyze "
            f"(default: {default_limit}; profile may override)"
        ),
    )
    parser.add_argument(
        '--min-market-cap',
        type=_positive_int,
        default=env_defaults.min_market_cap,
        help=f"Minimum market cap in USD (default: ${env_defaults.min_market_cap:,})",
    )
    parser.add_argument(
        '--min-volume',
        type=_positive_float,
        default=env_defaults.min_volume_24h,
        help=(
            "Minimum 24h USD volume required (default: "
            f"${env_defaults.min_volume_24h:,.0f})"
        ),
    )
    parser.add_argument(
        '--max-results',
        type=_positive_int,
        default=None,
        help=(
            "Maximum number of results to display "
            f"(default: {env_defaults.max_results}; profile may override)"
        ),
    )
    parser.add_argument(
        '--output',
        type=str,
        choices=['console', 'json'],
        default='console',
        help='Output format (default: console)',
    )
    parser.add_argument(
        '--side',
        type=str,
        choices=['long', 'short', 'both'],
        default=env_defaults.side,
        help=f"Which side(s) to evaluate (default: {env_defaults.side})",
    )
    parser.add_argument(
        '--unique-by-symbol',
        action=argparse.BooleanOptionalAction,
        default=env_defaults.unique_by_symbol,
        help='Keep only the best side per symbol',
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=env_defaults.min_overall_score,
        help=f"Filter out candidates below this overall score (default: {env_defaults.min_overall_score})",
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Optional path to save results (json or csv)',
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to analyze (e.g., BTC,ETH,SOL)',
    )
    parser.add_argument(
        '--top-per-side',
        type=_positive_int,
        default=env_defaults.top_per_side,
        help='Cap results per side before final sorting',
    )
    parser.add_argument(
        '--max-workers',
        type=_positive_int,
        default=None,
        help=(
            "Override worker threads for parallel fetch "
            f"(default: {env_defaults.max_workers}; profile may override)"
        ),
    )
    parser.add_argument(
        '--offline',
        action=argparse.BooleanOptionalAction,
        default=env_defaults.offline,
        help='Avoid external HTTP where possible (use cache only)',
    )
    parser.add_argument(
        '--force-refresh',
        action=argparse.BooleanOptionalAction,
        default=env_defaults.force_refresh_candles,
        help='Force fresh candle downloads instead of using cache (default: %(default)s)',
    )
    parser.add_argument(
        '--quotes',
        type=str,
        help='Preferred quote currencies (comma-separated), e.g., USDC,USD,USDT',
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=env_defaults.risk_free_rate,
        help=f"Override annual risk-free rate (default: {env_defaults.risk_free_rate})",
    )
    parser.add_argument(
        '--analysis-days',
        type=_positive_int,
        default=None,
        help=(
            "Number of daily bars to pull "
            f"(default: {env_defaults.analysis_days}; profile may override)"
        ),
    )
    parser.add_argument(
        '--min-vmc-ratio',
        type=_positive_float,
        default=env_defaults.min_volume_market_cap_ratio,
        help=(
            "Minimum volume-to-market-cap ratio (e.g., 0.03 for 3%) "
            f"(default: {env_defaults.min_volume_market_cap_ratio})"
        ),
    )
    max_risk_help = 'Highest risk level to include (e.g., LOW, MEDIUM, HIGH)'
    if default_max_risk:
        max_risk_help += f" (default: {default_max_risk})"
    parser.add_argument(
        '--max-risk-level',
        type=risk_level_type,
        default=default_max_risk,
        help=max_risk_help,
    )
    parser.add_argument(
        '--use-openai-scoring',
        action=argparse.BooleanOptionalAction,
        default=env_defaults.use_openai_scoring,
        help='Blend scores with OpenAI model output (default: %(default)s; override env/CRYPTO_* if set)',
    )
    parser.add_argument(
        '--openai-weight',
        type=float,
        default=None,
        help='Blend weight for OpenAI score (0-1); defaults to env or 0.25 when enabled',
    )
    parser.add_argument(
        '--openai-model',
        type=str,
        default=None,
        help=f"Override OpenAI model name (default: {env_defaults.openai_model})",
    )
    parser.add_argument(
        '--openai-max-candidates',
        type=_positive_int,
        default=None,
        help=f"Limit number of candidates sent to OpenAI (default: {env_defaults.openai_max_candidates})",
    )
    openai_temp_default = (
        env_defaults.openai_temperature
        if env_defaults.openai_temperature is not None
        else 'model default'
    )
    parser.add_argument(
        '--openai-temperature',
        type=float,
        default=None,
        help=f"Temperature for OpenAI call (default: {openai_temp_default})",
    )
    parser.add_argument(
        '--openai-sleep-seconds',
        type=float,
        default=None,
        help=f"Optional pause between OpenAI calls (default: {env_defaults.openai_sleep_seconds})",
    )

    return parser


def main():
    """Main function to run the crypto opportunity finder."""
    env_defaults = CryptoFinderConfig.from_env()

    valid_risk_levels = {level.name for level in RiskLevel}
    risk_level_type = make_risk_level_validator(valid_risk_levels)

    default_max_risk: Optional[str] = None
    if env_defaults.max_risk_level:
        try:
            default_max_risk = risk_level_type(str(env_defaults.max_risk_level))
        except argparse.ArgumentTypeError as exc:
            logger.warning(f"Ignoring invalid CRYPTO_MAX_RISK_LEVEL value: {exc}")

    default_limit = int(os.getenv('CRYPTO_DEFAULT_LIMIT', '50'))
    profile_default = os.getenv('CRYPTO_FINDER_PROFILE', 'default')
    if profile_default not in PROFILE_PRESETS:
        profile_default = 'default'

    parser = build_cli_parser(
        env_defaults,
        default_limit=default_limit,
        profile_default=profile_default,
        default_max_risk=default_max_risk,
        risk_level_type=risk_level_type,
    )
    args = parser.parse_args()

    profile_overrides = PROFILE_PRESETS.get(args.profile, {})
    final_limit = args.limit if args.limit is not None else profile_overrides.get('limit', default_limit)
    final_max_results = (
        args.max_results if args.max_results is not None else profile_overrides.get('max_results', env_defaults.max_results)
    )
    final_max_workers = (
        args.max_workers if args.max_workers is not None else profile_overrides.get('max_workers', env_defaults.max_workers)
    )
    final_analysis_days = (
        args.analysis_days if args.analysis_days is not None else profile_overrides.get('analysis_days', env_defaults.analysis_days)
    )

    if args.suppress_console_logs:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.StreamHandler):
                logger.removeHandler(handler)

    config = env_defaults
    config.min_market_cap = args.min_market_cap
    config.max_results = final_max_results
    config.side = args.side
    config.unique_by_symbol = args.unique_by_symbol
    config.min_overall_score = float(args.min_score or 0.0)
    config.offline = args.offline
    config.top_per_side = args.top_per_side
    config.max_workers = final_max_workers
    config.risk_free_rate = args.risk_free_rate
    config.analysis_days = final_analysis_days
    config.min_volume_24h = args.min_volume
    config.min_volume_market_cap_ratio = args.min_vmc_ratio
    config.max_risk_level = args.max_risk_level if args.max_risk_level is not None else config.max_risk_level
    config.force_refresh_candles = args.force_refresh
    config.use_openai_scoring = args.use_openai_scoring
    if args.openai_weight is not None:
        config.openai_weight = float(args.openai_weight)
    if args.openai_model:
        config.openai_model = args.openai_model
    if args.openai_max_candidates is not None:
        config.openai_max_candidates = int(args.openai_max_candidates)
    if args.openai_temperature is not None:
        config.openai_temperature = float(args.openai_temperature)
    if args.openai_sleep_seconds is not None:
        config.openai_sleep_seconds = float(args.openai_sleep_seconds)

    if config.offline and config.force_refresh_candles:
        logger.warning("Force refresh disabled because offline mode is enabled; using cached candles only.")
        config.force_refresh_candles = False

    config.symbols = None
    if args.symbols:
        config.symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]

    config.quotes = None
    if args.quotes:
        config.quotes = [q.strip().upper() for q in args.quotes.split(',') if q.strip()]

    # Initialize the finder
    finder = LongTermCryptoFinder(config=config)

    # Find opportunities
    results = finder.find_best_opportunities(limit=final_limit)

    if not results:
        print("No opportunities found. Please check your internet connection and try again.")
        return

    def save_plain_report(path: Path, content: str, notify: bool = True, status_stream: TextIO = sys.stdout) -> None:
        tmp_path = Path(f"{path}.tmp.{os.getpid()}.{int(time.time()*1000)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            handle.write(content)
        os.replace(tmp_path, path)
        if notify:
            print(f"Saved {len(results)} results to {path}", file=status_stream)

    # Output results
    if args.output == 'json' or (args.save and args.save.lower().endswith('.json')):
        # Convert results to dictionaries for JSON serialization
        json_results = []
        for crypto in results:
            pnl_profile = finder._compute_position_pnl(crypto)
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
                'data_timestamp_utc': getattr(crypto, 'data_timestamp_utc', ''),
                'position_notional_usd': _finite(pnl_profile['notional'], 0.0) if pnl_profile else 0.0,
                'position_leverage': _finite(pnl_profile['leverage'], 0.0) if pnl_profile else 0.0,
                'position_margin_usd': _finite(pnl_profile['margin'], 0.0) if pnl_profile else 0.0,
                'take_profit_pnl_usd': _finite(pnl_profile['tp_pnl'], 0.0) if pnl_profile else 0.0,
                'stop_loss_pnl_usd': _finite(pnl_profile['sl_pnl'], 0.0) if pnl_profile else 0.0,
                'take_profit_return_pct': _finite(pnl_profile['tp_return'] * 100.0, 0.0) if pnl_profile else 0.0,
                'stop_loss_return_pct': _finite(pnl_profile['sl_return'] * 100.0, 0.0) if pnl_profile else 0.0,
            }
            json_results.append(crypto_dict)
        if args.save and args.save.lower().endswith('.json'):
            # Atomic JSON save
            finder._atomic_write_json(Path(args.save), json_results)
            if args.output == 'json':
                print(f"Saved {len(json_results)} results to {args.save}", file=sys.stderr)
            else:
                print(f"Saved {len(json_results)} results to {args.save}")
        if args.output == 'json':
            print(json.dumps({
                'version': '1.0',
                'config': finder.config.to_dict(),
                'generated_utc': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ'),
                'results': json_results
            }, indent=2))
        if args.plain_output:
            buffer = io.StringIO()
            finder.print_results(results, stream=buffer)
            save_plain_report(
                args.plain_output,
                buffer.getvalue(),
                notify=args.output != 'json',
                status_stream=sys.stderr if args.output == 'json' else sys.stdout
            )
    else:
        if args.plain_output:
            buffer = io.StringIO()
            finder.print_results(results, stream=buffer)
            report_text = buffer.getvalue()
            print(report_text, end='')
            save_plain_report(args.plain_output, report_text)
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
                'position_size_percentage', 'data_timestamp_utc', 'position_notional_usd',
                'position_leverage', 'position_margin_usd', 'take_profit_pnl_usd',
                'stop_loss_pnl_usd', 'take_profit_return_pct', 'stop_loss_return_pct'
            ]
            # Atomic CSV save: write to temp and replace
            tmp_path = Path(args.save + f".tmp.{os.getpid()}.{int(time.time()*1000)}")
            final_path = Path(args.save)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for crypto in results:
                    pnl_profile = finder._compute_position_pnl(crypto)
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
                        'data_timestamp_utc': getattr(crypto, 'data_timestamp_utc', ''),
                        'position_notional_usd': pnl_profile['notional'] if pnl_profile else 0.0,
                        'position_leverage': pnl_profile['leverage'] if pnl_profile else 0.0,
                        'position_margin_usd': pnl_profile['margin'] if pnl_profile else 0.0,
                        'take_profit_pnl_usd': pnl_profile['tp_pnl'] if pnl_profile else 0.0,
                        'stop_loss_pnl_usd': pnl_profile['sl_pnl'] if pnl_profile else 0.0,
                        'take_profit_return_pct': pnl_profile['tp_return'] * 100.0 if pnl_profile else 0.0,
                        'stop_loss_return_pct': pnl_profile['sl_return'] * 100.0 if pnl_profile else 0.0,
                    })
            os.replace(tmp_path, final_path)
            print(f"Saved {len(results)} results to {final_path}")
        elif args.save:
            print("Tip: use a .csv or .json extension in --save to write the file.")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
