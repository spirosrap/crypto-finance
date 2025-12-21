from __future__ import annotations

from typing import List, Optional

from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None
    return value  # type: ignore[return-value]


def _none_if_empty(value):
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


class CryptoFinderSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CRYPTO_", case_sensitive=False, extra="ignore")

    min_market_cap: int = 100_000_000
    max_results: int = 20
    max_workers: int = 4
    request_delay: float = 0.5
    cache_ttl: int = 300
    force_refresh_candles: bool = False
    risk_free_rate: float = 0.03
    analysis_days: int = 365
    intraday_granularity: str = "ONE_HOUR"
    intraday_lookback_days: int = 14
    intraday_resample: str = "4H"
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
    side: str = "both"
    unique_by_symbol: bool = False
    min_overall_score: float = 0.0
    offline: bool = False
    symbols: Optional[List[str]] = None
    top_per_side: Optional[int] = None
    quotes: Optional[List[str]] = None
    max_risk_level: Optional[str] = None
    risk_reward_weight: float = 0.15
    trend_weight: float = 0.10
    use_openai_scoring: bool = False
    openai_model: str = "gpt-5-mini"
    openai_weight: float = 0.25
    openai_max_candidates: int = 12
    openai_temperature: Optional[float] = None
    openai_sleep_seconds: float = 0.0
    report_position_notional: float = 1000.0
    report_leverage: float = 50.0
    incremental_cache: bool = False
    incremental_cache_path: str = "cache/incremental_metrics.json"
    exchange_backend: str = "ccxt"
    ccxt_exchange_id: str = "coinbaseadvanced"
    max_atr_usd: Optional[float] = None
    max_atr_bps: Optional[float] = None
    max_spread_margin_pct: Optional[float] = None

    @field_validator("symbols", "quotes", mode="before")
    @classmethod
    def _split_lists(cls, value):
        items = _split_csv(value)
        if items is None:
            return None
        return [str(item).strip().upper() for item in items if str(item).strip()]

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value):
        if value is None:
            return value
        side = str(value).strip().lower()
        if side not in {"long", "short", "both"}:
            raise ValueError("side must be one of: long, short, both")
        return side

    @field_validator("intraday_granularity", "intraday_resample", mode="before")
    @classmethod
    def _upper_intraday(cls, value):
        if value is None:
            return value
        return str(value).strip().upper()

    @field_validator("max_risk_level", mode="before")
    @classmethod
    def _upper_risk(cls, value):
        if value is None:
            return None
        return str(value).strip().upper()

    @field_validator(
        "max_atr_usd",
        "max_atr_bps",
        "max_spread_margin_pct",
        "openai_temperature",
        mode="before",
    )
    @classmethod
    def _none_if_empty_str(cls, value):
        return _none_if_empty(value)

    @field_validator("exchange_backend", mode="before")
    @classmethod
    def _lower_exchange_backend(cls, value):
        if value is None:
            return value
        backend = str(value).strip().lower()
        if backend not in {"ccxt", "coinbase"}:
            raise ValueError("exchange_backend must be 'ccxt' or 'coinbase'")
        return backend

    @field_validator("ccxt_exchange_id", mode="before")
    @classmethod
    def _strip_exchange_id(cls, value):
        if value is None:
            return value
        return str(value).strip()


class ShortTermSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHORT_", case_sensitive=False, extra="ignore")

    analysis_days: Optional[int] = None
    min_market_cap: Optional[int] = None
    max_results: Optional[int] = None
    request_delay: Optional[float] = None
    max_workers: Optional[int] = None
    side: Optional[str] = None
    unique_by_symbol: Optional[bool] = None
    min_overall_score: Optional[float] = None
    top_per_side: Optional[int] = None
    risk_free_rate: float = 0.01
    force_refresh_candles: bool = True
    min_volume_24h: Optional[float] = None
    min_volume_market_cap_ratio: Optional[float] = None
    intraday_granularity: Optional[str] = None
    intraday_lookback_days: Optional[int] = None
    intraday_resample: Optional[str] = None
    rsi_period: int = 7
    atr_period: int = 7
    stochastic_period: int = 10
    williams_period: int = 10
    cci_period: int = 14
    adx_period: int = 14
    bb_period: int = 14
    macd_fast: int = 8
    macd_slow: int = 21
    macd_signal: int = 5
    max_atr_usd: Optional[float] = None
    max_atr_bps: Optional[float] = None
    max_risk_level: Optional[str] = None
    use_openai_scoring: Optional[bool] = None
    openai_model: Optional[str] = None
    openai_weight: Optional[float] = None
    openai_max_candidates: Optional[int] = None
    openai_temperature: Optional[float] = None
    openai_sleep_seconds: Optional[float] = None
    report_position_notional: Optional[float] = None
    report_leverage: Optional[float] = None
    max_spread_margin_pct: Optional[float] = None
    incremental_cache: Optional[bool] = None
    incremental_cache_path: Optional[str] = None
    default_limit: int = 30
    finder_profile: str = "default"

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value):
        if value is None:
            return value
        side = str(value).strip().lower()
        if side not in {"long", "short", "both"}:
            raise ValueError("side must be one of: long, short, both")
        return side

    @field_validator("intraday_granularity", "intraday_resample", mode="before")
    @classmethod
    def _upper_intraday(cls, value):
        if value is None:
            return value
        return str(value).strip().upper()

    @field_validator("max_risk_level", mode="before")
    @classmethod
    def _upper_risk(cls, value):
        if value is None:
            return None
        return str(value).strip().upper()

    @field_validator(
        "max_atr_usd",
        "max_atr_bps",
        "max_spread_margin_pct",
        "openai_temperature",
        mode="before",
    )
    @classmethod
    def _none_if_empty_str(cls, value):
        return _none_if_empty(value)

    @field_validator("finder_profile", mode="before")
    @classmethod
    def _normalize_profile(cls, value):
        if value is None:
            return value
        return str(value).strip()


__all__ = ["CryptoFinderSettings", "ShortTermSettings", "ValidationError"]
