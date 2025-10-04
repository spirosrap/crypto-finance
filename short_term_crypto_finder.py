#!/usr/bin/env python3
"""Short-Term Crypto Opportunity Finder

Tailored variant of ``long_term_crypto_finder`` that narrows the analysis
window, emphasises fast-moving technical signals, and tightens the
risk/target framework for short-horizon trades.

Usage mirrors ``long_term_crypto_finder.py`` but the metrics, scoring, and
trading levels are tuned for swing/short-term setups (≈days to weeks).
"""

from __future__ import annotations

import os

os.environ.setdefault("CRYPTO_FINDER_LOG_SUBDIR", "short_term_crypto_finder")
os.environ.setdefault("CRYPTO_FINDER_LOGGER_NAME", "short_term_crypto_finder")

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TextIO

import numpy as np
import pandas as pd

from long_term_crypto_finder import (
    CryptoFinderConfig,
    CryptoMetrics,
    LongTermCryptoFinder,
    RiskLevel,
    UTC,
    _finite,
)


GRANULARITY_TO_HOURS = {
    'ONE_MINUTE': 1.0 / 60.0,
    'FIVE_MINUTE': 5.0 / 60.0,
    'TEN_MINUTE': 10.0 / 60.0,
    'FIFTEEN_MINUTE': 15.0 / 60.0,
    'THIRTY_MINUTE': 0.5,
    'ONE_HOUR': 1.0,
    'TWO_HOUR': 2.0,
    'THREE_HOUR': 3.0,
    'FOUR_HOUR': 4.0,
    'SIX_HOUR': 6.0,
    'ONE_DAY': 24.0,
}


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def _setup_logger() -> logging.Logger:
    name = "short_term_crypto_finder"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logs_dir = Path("logs") / name
    logs_dir.mkdir(parents=True, exist_ok=True)

    level = os.getenv("SHORT_FINDER_LOG_LEVEL", "INFO").upper()
    level_value = getattr(logging, level, logging.INFO)
    logger.setLevel(level_value)

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    file_handler = TimedRotatingFileHandler(
        logs_dir / "short_term_crypto_finder.log",
        when="midnight",
        backupCount=int(os.getenv("SHORT_FINDER_LOG_RETENTION", "7")),
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if os.getenv("SHORT_FINDER_LOG_TO_CONSOLE", "1") not in ("0", "false", "False"):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    logger.info("Short-term finder logging initialised")
    return logger


logger = _setup_logger()


# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------


def _env_override(key: str, default, cast):
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except Exception:
        logger.warning(f"Invalid value for {key!r}: {raw!r}; falling back to {default!r}")
        return default


def _env_flag(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y"}


def build_short_term_config() -> CryptoFinderConfig:
    """Create a CryptoFinderConfig tuned for shorter-term trading."""

    cfg = CryptoFinderConfig.from_env()

    cfg.analysis_days = _env_override("SHORT_ANALYSIS_DAYS", 120, int)
    cfg.min_market_cap = _env_override("SHORT_MIN_MARKET_CAP", max(cfg.min_market_cap, 50_000_000), int)
    cfg.max_results = _env_override("SHORT_MAX_RESULTS", cfg.max_results, int)
    cfg.request_delay = _env_override("SHORT_REQUEST_DELAY", cfg.request_delay, float)
    cfg.max_workers = _env_override("SHORT_MAX_WORKERS", cfg.max_workers, int)
    cfg.side = _env_override("SHORT_SIDE", cfg.side, str).lower()
    cfg.unique_by_symbol = _env_flag("SHORT_UNIQUE_BY_SYMBOL", bool(cfg.unique_by_symbol))
    cfg.min_overall_score = _env_override("SHORT_MIN_SCORE", max(cfg.min_overall_score, 20.0), float)
    cfg.top_per_side = _env_override(
        "SHORT_TOP_PER_SIDE",
        cfg.top_per_side if cfg.top_per_side is not None else 10,
        int,
    )
    cfg.risk_free_rate = _env_override("SHORT_RISK_FREE_RATE", 0.01, float)
    cfg.force_refresh_candles = _env_flag("SHORT_FORCE_REFRESH_CANDLES", True)
    cfg.min_volume_24h = _env_override("SHORT_MIN_VOLUME_24H", cfg.min_volume_24h, float)
    cfg.min_volume_market_cap_ratio = _env_override(
        "SHORT_MIN_VMC_RATIO",
        cfg.min_volume_market_cap_ratio,
        float,
    )
    cfg.intraday_granularity = _env_override(
        "SHORT_INTRADAY_GRANULARITY",
        cfg.intraday_granularity,
        str,
    ).upper()
    cfg.intraday_lookback_days = max(
        3,
        _env_override(
            "SHORT_INTRADAY_LOOKBACK_DAYS",
            max(cfg.intraday_lookback_days, 7),
            int,
        ),
    )
    cfg.intraday_resample = _env_override(
        "SHORT_INTRADAY_RESAMPLE",
        cfg.intraday_resample,
        str,
    ).upper()

    # Faster indicator defaults
    cfg.rsi_period = _env_override("SHORT_RSI_PERIOD", 7, int)
    cfg.atr_period = _env_override("SHORT_ATR_PERIOD", 7, int)
    cfg.stochastic_period = _env_override("SHORT_STOCH_PERIOD", 10, int)
    cfg.williams_period = _env_override("SHORT_WILLIAMS_PERIOD", 10, int)
    cfg.cci_period = _env_override("SHORT_CCI_PERIOD", 14, int)
    cfg.adx_period = _env_override("SHORT_ADX_PERIOD", 14, int)
    cfg.bb_period = _env_override("SHORT_BB_PERIOD", 14, int)
    cfg.macd_fast = _env_override("SHORT_MACD_FAST", 8, int)
    cfg.macd_slow = _env_override("SHORT_MACD_SLOW", 21, int)
    cfg.macd_signal = _env_override("SHORT_MACD_SIGNAL", 5, int)

    max_risk_env = os.getenv("SHORT_MAX_RISK_LEVEL")
    if max_risk_env:
        cfg.max_risk_level = max_risk_env.strip().upper()

    cfg.use_openai_scoring = _env_flag("SHORT_USE_OPENAI_SCORING", bool(cfg.use_openai_scoring))
    openai_model_override = os.getenv("SHORT_OPENAI_MODEL")
    if openai_model_override:
        cfg.openai_model = openai_model_override.strip()
    cfg.openai_weight = _env_override("SHORT_OPENAI_WEIGHT", cfg.openai_weight, float)
    cfg.openai_max_candidates = _env_override("SHORT_OPENAI_MAX_CANDIDATES", cfg.openai_max_candidates, int)
    cfg.openai_temperature = _env_override("SHORT_OPENAI_TEMPERATURE", cfg.openai_temperature, float)
    cfg.openai_sleep_seconds = _env_override("SHORT_OPENAI_SLEEP_SECONDS", cfg.openai_sleep_seconds, float)
    cfg.report_position_notional = _env_override("SHORT_REPORT_NOTIONAL", cfg.report_position_notional, float)
    cfg.report_leverage = _env_override("SHORT_REPORT_LEVERAGE", cfg.report_leverage, float)

    return cfg


# -----------------------------------------------------------------------------
# Finder subclass with short-term logic tweaks
# -----------------------------------------------------------------------------


class ShortTermCryptoFinder(LongTermCryptoFinder):
    REPORT_TITLE = "SHORT-TERM CRYPTO OPPORTUNITIES ANALYSIS"
    FINDER_LABEL = "Short-Term Crypto Finder"

    MOMENTUM_MIN_BARS = 20
    MOMENTUM_MAX_BARS = 45
    ATR_STOP_MULT_LONG = 1.3
    ATR_STOP_MULT_SHORT = 1.3
    RR_TARGET = 2.2

    def __init__(self, config: Optional[CryptoFinderConfig] = None):
        cfg = config or build_short_term_config()
        super().__init__(config=cfg)
        logger.info("Short-term finder configured: %s", cfg.to_dict())

    # ------------------------------------------------------------------
    # Intraday enrichment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _granularity_hours(granularity: str) -> float:
        return float(GRANULARITY_TO_HOURS.get(granularity.upper(), 1.0))

    @staticmethod
    def _window_return(series: pd.Series, bars: int) -> float:
        if bars <= 0 or len(series) < 2:
            return 0.0
        if len(series) <= bars:
            prev = float(series.iloc[0])
        else:
            prev = float(series.iloc[-(bars + 1)])
        last = float(series.iloc[-1])
        if prev <= 0 or last <= 0:
            return 0.0
        return (last / prev) - 1.0

    @staticmethod
    def _derive_intraday_metrics(df: pd.DataFrame, candle_hours: float) -> Dict[str, float]:
        if df.empty:
            return {}

        candle_hours = max(candle_hours, 1e-6)
        returns = df['close'].pct_change().dropna()
        bars_per_6h = max(int(round(6.0 / candle_hours)), 1)
        bars_per_24h = max(int(round(24.0 / candle_hours)), bars_per_6h)

        vol6 = float(np.std(returns.tail(bars_per_6h), ddof=0)) if len(returns) >= 1 else 0.0
        vol24 = float(np.std(returns.tail(bars_per_24h), ddof=0)) if len(returns) >= 1 else vol6

        metrics = {
            'intraday_return_6h': ShortTermCryptoFinder._window_return(df['close'], bars_per_6h),
            'intraday_return_24h': ShortTermCryptoFinder._window_return(df['close'], bars_per_24h),
            'intraday_volatility_6h': vol6,
            'intraday_volatility_24h': vol24,
            'intraday_volume_24h': float(df['volume'].tail(bars_per_24h).sum()),
        }

        return metrics

    def _fetch_intraday_candles(self, product_id: str) -> Optional[pd.DataFrame]:
        granularity = getattr(self.config, 'intraday_granularity', 'ONE_HOUR') or 'ONE_HOUR'
        lookback_days = max(1, int(getattr(self.config, 'intraday_lookback_days', 14)))

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(days=lookback_days)

        candles = list(
            self._cached_candles(
                product_id,
                granularity,
                start_time.isoformat(),
                end_time.isoformat(),
            )
        )

        if not candles:
            return None

        rows: List[Dict[str, float]] = []
        for candle in candles:
            ts = self._normalize_ts(candle.get('start') or candle.get('time'))
            if not ts:
                continue
            rows.append(
                {
                    'timestamp': datetime.fromtimestamp(ts, UTC),
                    'open': float(candle.get('open', 0.0)),
                    'high': float(candle.get('high', 0.0)),
                    'low': float(candle.get('low', 0.0)),
                    'close': float(candle.get('close', 0.0)),
                    'volume': float(candle.get('volume', 0.0)),
                }
            )

        if not rows:
            return None

        intraday_df = pd.DataFrame(rows)
        intraday_df.sort_values('timestamp', inplace=True)
        intraday_df.set_index('timestamp', inplace=True)

        resample_rule = str(getattr(self.config, 'intraday_resample', '') or '').strip()
        if resample_rule:
            try:
                rule_for_pd = resample_rule.lower()
                resampled = intraday_df.resample(rule_for_pd).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                }).dropna()
                if not resampled.empty and len(resampled) >= 3:
                    intraday_df = resampled
            except Exception as exc:
                logger.debug("Intraday resample %s failed for %s: %s", resample_rule, product_id, exc)

        return intraday_df

    def _load_intraday_summary(self, product_id: str) -> Dict[str, float]:
        try:
            intraday_df = self._fetch_intraday_candles(product_id)
            if intraday_df is None or intraday_df.empty:
                return {}

            candle_hours = self._granularity_hours(getattr(self.config, 'intraday_granularity', 'ONE_HOUR'))
            if len(intraday_df.index) >= 2:
                last_delta = (intraday_df.index[-1] - intraday_df.index[-2]).total_seconds() / 3600.0
                if last_delta > 0:
                    candle_hours = max(candle_hours, last_delta)
            summary = self._derive_intraday_metrics(intraday_df, candle_hours)
            if not summary:
                return {}
            return summary
        except Exception as exc:
            logger.debug("Failed to load intraday summary for %s: %s", product_id, exc)
            return {}

    def get_historical_data(self, product_id: str, days: int = 120) -> Optional[pd.DataFrame]:  # type: ignore[override]
        base_df = super().get_historical_data(product_id, days)
        if base_df is None or base_df.empty:
            return base_df

        summary = self._load_intraday_summary(product_id)
        if not summary:
            return base_df

        df = base_df.copy()
        for key, value in summary.items():
            df[key] = np.nan
            df.iloc[-1, df.columns.get_loc(key)] = value
            df[key] = df[key].ffill()

        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:  # type: ignore[override]
        """Augment base indicators with short-horizon impulse context."""

        metrics = super().calculate_technical_indicators(df)

        try:
            if df.empty:
                metrics.setdefault('volume_thrust_3_15', 1.0)
                metrics.setdefault('impulse_3v10', 0.0)
                metrics.setdefault('continuation_10v21', 0.0)
                metrics.setdefault('return_3d', 0.0)
                metrics.setdefault('return_10d', 0.0)
                metrics.setdefault('return_21d', 0.0)
                metrics.setdefault('up_day_ratio_7', 0.5)
                metrics.setdefault('breakout_distance_5d', 0.0)
                metrics.setdefault('breakdown_distance_5d', 0.0)
                return metrics

            prices = df['price'].astype(float)
            volumes = df['volume'].astype(float)

            def _pct_change(window: int) -> float:
                if len(prices) < window + 1:
                    return 0.0
                past_price = float(prices.iloc[-(window + 1)])
                current_price = float(prices.iloc[-1])
                if past_price <= 0:
                    return 0.0
                return (current_price / past_price) - 1.0

            return_3d = _pct_change(3)
            return_10d = _pct_change(10) if len(prices) >= 11 else return_3d
            return_21d = _pct_change(21) if len(prices) >= 22 else return_10d

            impulse = return_3d - 0.5 * return_10d
            continuation = return_10d - 0.5 * return_21d

            short_vol = float(volumes.tail(3).mean()) if len(volumes) else 0.0
            mid_vol = float(volumes.tail(15).mean()) if len(volumes) else short_vol
            volume_thrust = 1.0
            if mid_vol > 0:
                volume_thrust = float(np.clip(short_vol / mid_vol, 0.1, 10.0))

            up_moves = prices.pct_change().tail(6)
            up_ratio = float((up_moves > 0).sum()) / max(len(up_moves), 1)

            current_price = float(prices.iloc[-1])
            recent_high = float(df['high'].tail(5).max()) if len(df) >= 5 else float(df['high'].iloc[-1])
            recent_low = float(df['low'].tail(5).min()) if len(df) >= 5 else float(df['low'].iloc[-1])

            breakout_distance = 0.0
            if recent_high > 0:
                breakout_distance = float(np.clip(current_price / recent_high - 1.0, -1.0, 1.0))

            breakdown_distance = 0.0
            if current_price > 0 and recent_low > 0:
                breakdown_distance = float(np.clip(recent_low / current_price - 1.0, -1.0, 1.0))

            metrics['return_3d'] = return_3d
            metrics['return_10d'] = return_10d
            metrics['return_21d'] = return_21d
            metrics['impulse_3v10'] = impulse
            metrics['continuation_10v21'] = continuation
            metrics['volume_thrust_3_15'] = volume_thrust
            metrics['up_day_ratio_7'] = float(np.clip(up_ratio, 0.0, 1.0))
            metrics['breakout_distance_5d'] = breakout_distance
            metrics['breakdown_distance_5d'] = breakdown_distance

            if 'intraday_return_6h' in df.columns:
                metrics['intraday_return_6h'] = float(df['intraday_return_6h'].iloc[-1] or 0.0)
            if 'intraday_return_24h' in df.columns:
                metrics['intraday_return_24h'] = float(df['intraday_return_24h'].iloc[-1] or 0.0)
            if 'intraday_volatility_6h' in df.columns:
                metrics['intraday_volatility_6h'] = float(df['intraday_volatility_6h'].iloc[-1] or 0.0)
            if 'intraday_volatility_24h' in df.columns:
                metrics['intraday_volatility_24h'] = float(df['intraday_volatility_24h'].iloc[-1] or 0.0)
            intraday_volume_sum = float(df['intraday_volume_24h'].iloc[-1]) if 'intraday_volume_24h' in df.columns else 0.0
            if intraday_volume_sum > 0.0:
                metrics['intraday_volume_24h'] = intraday_volume_sum
                last_daily_volume = float(df['volume'].iloc[-1] or 0.0)
                if last_daily_volume > 0:
                    ratio = intraday_volume_sum / last_daily_volume
                    metrics['intraday_volume_ratio'] = float(np.clip(ratio, 0.0, 12.0))

        except Exception as exc:
            logger.warning("Failed to enrich short-term indicators: %s", exc)

        return metrics

    # ------------------------------------------------------------------
    # Momentum: use 20-45 day slope for quicker signal capture
    # ------------------------------------------------------------------
    def calculate_momentum_score(self, df: pd.DataFrame) -> float:  # type: ignore[override]
        try:
            n = len(df)
            if n < self.MOMENTUM_MIN_BARS:
                return 50.0

            win = min(max(self.MOMENTUM_MIN_BARS, n), self.MOMENTUM_MAX_BARS)
            tail = df['price'].tail(win).to_numpy(dtype=float)
            if np.any(tail <= 0):
                tail = np.clip(tail, 1e-8, None)

            y = np.log(tail)
            x = np.arange(len(y), dtype=float)
            x_mean = x.mean()
            y_mean = y.mean()
            sxx = float(np.sum((x - x_mean) ** 2))
            if sxx <= 0:
                return 50.0
            beta = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
            resid = y - (y_mean + beta * (x - x_mean))
            dof = max(len(y) - 2, 1)
            s2 = float(np.sum(resid ** 2) / dof)
            se_beta = float(np.sqrt(s2 / sxx)) if sxx > 0 else 0.0
            t_stat = float(beta / se_beta) if se_beta > 0 else 0.0

            score = 100.0 / (1.0 + np.exp(-t_stat / 1.5))
            return float(np.clip(score, 0.0, 100.0))
        except Exception as exc:
            logger.error(f"Short-term momentum calculation failed: {exc}")
            return 50.0

    # ------------------------------------------------------------------
    # Technical scores favour faster oscillators and volume surges
    # ------------------------------------------------------------------
    def _calculate_technical_score(self, technical_metrics: Dict, momentum_score: float) -> float:  # type: ignore[override]
        try:
            score = 0.0

            rsi = float(technical_metrics.get('rsi_14', 50.0))
            if 42 <= rsi <= 58:
                rsi_score = 78
            elif rsi < 42:
                rsi_score = 92
            elif rsi <= 68:
                rsi_score = 65
            else:
                rsi_score = 48
            score += rsi_score * 0.20

            macd = technical_metrics.get('macd_signal', 'NEUTRAL')
            if macd == 'BULLISH':
                macd_score = 85
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:
                macd_score = 40
            score += macd_score * 0.12

            bb = technical_metrics.get('bb_position', 'NEUTRAL')
            if bb == 'OVERSOLD':
                bb_score = 85
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:
                bb_score = 45
            score += bb_score * 0.12

            trend = float(technical_metrics.get('trend_strength', 0.0))
            if trend > 0.9:
                trend_score = 90
            elif trend > 0.25:
                trend_score = 76
            elif trend > -0.05:
                trend_score = 58
            else:
                trend_score = 38
            score += trend_score * 0.12

            adx = float(technical_metrics.get('adx', 0.0))
            adx_score = 55.0 + 35.0 * np.tanh((adx - 22.0) / 12.0)
            score += adx_score * 0.08

            vol_ratio = float(technical_metrics.get('volume_ratio', 1.0))
            if vol_ratio >= 1.8:
                volume_score = 90
            elif vol_ratio >= 1.25:
                volume_score = 75
            elif vol_ratio >= 0.95:
                volume_score = 58
            else:
                volume_score = 40
            score += volume_score * 0.08

            volume_thrust = float(technical_metrics.get('volume_thrust_3_15', 1.0) or 1.0)
            vol_thrust_score = 55.0 + 35.0 * np.tanh((volume_thrust - 1.0) / 0.5)
            score += vol_thrust_score * 0.04

            impulse_val = float(technical_metrics.get('impulse_3v10', 0.0) or 0.0)
            continuation_val = float(technical_metrics.get('continuation_10v21', 0.0) or 0.0)
            impulse_combo = impulse_val + 0.5 * continuation_val
            impulse_score = 55.0 + 35.0 * np.tanh(impulse_combo / 0.05)
            score += impulse_score * 0.06

            score += np.clip(momentum_score, 0.0, 100.0) * 0.18

            intraday_ret6 = float(technical_metrics.get('intraday_return_6h', 0.0) or 0.0)
            score += float(np.clip(5.0 * np.tanh(intraday_ret6 / 0.015), -5.0, 5.0))

            intraday_ret24 = float(technical_metrics.get('intraday_return_24h', 0.0) or 0.0)
            score += float(np.clip(4.0 * np.tanh(intraday_ret24 / 0.03), -4.0, 4.0))

            intraday_vol6 = float(technical_metrics.get('intraday_volatility_6h', 0.0) or 0.0)
            if intraday_vol6 > 0:
                vol_target = 0.02
                vol_score = 3.5 * np.tanh((vol_target - intraday_vol6) / 0.015)
                score += float(vol_score)

            intraday_volume_ratio = float(technical_metrics.get('intraday_volume_ratio', 1.0) or 1.0)
            score += float(np.clip(4.5 * np.tanh((intraday_volume_ratio - 1.0) / 0.35), -5.0, 5.0))

            if technical_metrics.get('macd_cross'):
                hist = float(technical_metrics.get('macd_hist', 0.0) or 0.0)
                score += 4 if hist > 0 else -4

            breakout_bonus = float(technical_metrics.get('breakout_distance_5d', 0.0) or 0.0)
            if breakout_bonus > 0:
                score += float(np.clip(breakout_bonus * 400.0, 0.0, 6.0))

            return float(np.clip(score, 0.0, 100.0))
        except Exception as exc:
            logger.error(f"Short-term technical score failed: {exc}")
            return 50.0

    def _calculate_technical_score_short(self, technical_metrics: Dict, momentum_score_long: float) -> float:  # type: ignore[override]
        try:
            score = 0.0

            rsi = float(technical_metrics.get('rsi_14', 50.0))
            if rsi >= 70:
                rsi_score = 94
            elif rsi >= 55:
                rsi_score = 76
            elif rsi >= 40:
                rsi_score = 58
            else:
                rsi_score = 40
            score += rsi_score * 0.20

            macd = technical_metrics.get('macd_signal', 'NEUTRAL')
            if macd == 'BEARISH':
                macd_score = 88
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:
                macd_score = 42
            score += macd_score * 0.12

            bb = technical_metrics.get('bb_position', 'NEUTRAL')
            if bb == 'OVERBOUGHT':
                bb_score = 90
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:
                bb_score = 45
            score += bb_score * 0.12

            trend = float(technical_metrics.get('trend_strength', 0.0))
            if trend < -0.9:
                trend_score = 90
            elif trend < -0.25:
                trend_score = 74
            elif trend < 0.05:
                trend_score = 56
            else:
                trend_score = 35
            score += trend_score * 0.12

            adx = float(technical_metrics.get('adx', 0.0))
            adx_score = 55.0 + 35.0 * np.tanh((adx - 22.0) / 12.0)
            score += adx_score * 0.08

            vol_ratio = float(technical_metrics.get('volume_ratio', 1.0))
            if vol_ratio >= 1.8:
                volume_score = 88
            elif vol_ratio >= 1.25:
                volume_score = 72
            elif vol_ratio >= 0.95:
                volume_score = 55
            else:
                volume_score = 42
            score += volume_score * 0.08

            volume_thrust = float(technical_metrics.get('volume_thrust_3_15', 1.0) or 1.0)
            vol_thrust_score = 55.0 + 35.0 * np.tanh((volume_thrust - 1.0) / 0.5)
            score += vol_thrust_score * 0.04

            impulse_val = float(technical_metrics.get('impulse_3v10', 0.0) or 0.0)
            continuation_val = float(technical_metrics.get('continuation_10v21', 0.0) or 0.0)
            impulse_combo = -(impulse_val + 0.5 * continuation_val)
            impulse_score = 55.0 + 35.0 * np.tanh(impulse_combo / 0.05)
            breakdown_bonus = float(technical_metrics.get('breakdown_distance_5d', 0.0) or 0.0)
            if breakdown_bonus > 0:
                impulse_score += float(np.clip(breakdown_bonus * 400.0, 0.0, 6.0))
            score += impulse_score * 0.06

            short_momentum = np.clip(100.0 - momentum_score_long, 0.0, 100.0)
            score += short_momentum * 0.18

            intraday_ret6 = float(technical_metrics.get('intraday_return_6h', 0.0) or 0.0)
            score += float(np.clip(5.0 * np.tanh((-intraday_ret6) / 0.015), -5.0, 5.0))

            intraday_ret24 = float(technical_metrics.get('intraday_return_24h', 0.0) or 0.0)
            score += float(np.clip(4.0 * np.tanh((-intraday_ret24) / 0.03), -4.0, 4.0))

            intraday_vol6 = float(technical_metrics.get('intraday_volatility_6h', 0.0) or 0.0)
            if intraday_vol6 > 0:
                vol_target = 0.02
                vol_score = 3.5 * np.tanh((intraday_vol6 - vol_target) / 0.015)
                score += float(vol_score)

            intraday_volume_ratio = float(technical_metrics.get('intraday_volume_ratio', 1.0) or 1.0)
            score += float(np.clip(4.5 * np.tanh((intraday_volume_ratio - 1.0) / 0.35), -5.0, 5.0))

            if technical_metrics.get('macd_cross'):
                hist = float(technical_metrics.get('macd_hist', 0.0) or 0.0)
                score += 4 if hist < 0 else -4

            return float(np.clip(score, 0.0, 100.0))
        except Exception as exc:
            logger.error(f"Short-term short-score failed: {exc}")
            return 50.0

    # ------------------------------------------------------------------
    # Trading levels: tighter ATR stops and nearer swing targets
    # ------------------------------------------------------------------
    def calculate_trading_levels(self, df: pd.DataFrame, current_price: float, technical_metrics: Dict) -> Dict[str, float]:  # type: ignore[override]
        try:
            entry_price = current_price
            atr_raw = float(technical_metrics.get('atr', 0.0) or 0.0)

            stop_candidates = []
            if atr_raw > 0:
                stop_candidates.append(entry_price - self.ATR_STOP_MULT_LONG * atr_raw)
            if len(df) >= 6:
                swing_low = float(df['low'].tail(6).min())
                stop_candidates.append(swing_low * 0.99)

            daily_vol = float(technical_metrics.get('daily_vol_30d', 0.0) or 0.0)
            pct_buffer = 0.035 if daily_vol <= 0.03 else 0.045
            stop_candidates.append(entry_price * (1 - pct_buffer))

            stop_loss_price = max(stop_candidates) if stop_candidates else entry_price * 0.97
            stop_loss_price = min(stop_loss_price, entry_price * 0.998)

            risk_amount = max(entry_price - stop_loss_price, entry_price * 0.001)
            tp_candidates = [entry_price + risk_amount * self.RR_TARGET]

            if len(df) >= 10:
                swing_high = float(df['high'].tail(10).max())
                tp_candidates.append(swing_high * 1.01)

            if len(df) >= 30:
                recent_close = float(df['price'].tail(30).max())
                tp_candidates.append(max(recent_close * 1.02, entry_price + risk_amount))

            take_profit_price = max(entry_price * 1.01, min(tp_candidates))
            take_profit_price = max(take_profit_price, entry_price * 1.001)

            rr = self._risk_reward_ratio(entry_price, stop_loss_price, take_profit_price, atr_raw, is_long=True)
            pos_pct = self._position_size_percentage(entry_price, stop_loss_price, atr_raw)

            return {
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_reward_ratio': rr,
                'position_size_percentage': pos_pct,
            }
        except Exception as exc:
            logger.error(f"Short-term trading levels failed: {exc}")
            return {
                'entry_price': current_price,
                'stop_loss_price': current_price * 0.97,
                'take_profit_price': current_price * 1.20,
                'risk_reward_ratio': 2.0,
                'position_size_percentage': 1.0,
            }

    def calculate_short_trading_levels(self, df: pd.DataFrame, current_price: float, technical_metrics: Dict) -> Dict[str, float]:  # type: ignore[override]
        try:
            entry_price = current_price
            atr_raw = float(technical_metrics.get('atr', 0.0) or 0.0)

            stop_candidates = []
            if atr_raw > 0:
                stop_candidates.append(entry_price + self.ATR_STOP_MULT_SHORT * atr_raw)
            if len(df) >= 6:
                swing_high = float(df['high'].tail(6).max())
                stop_candidates.append(swing_high * 1.01)

            daily_vol = float(technical_metrics.get('daily_vol_30d', 0.0) or 0.0)
            pct_buffer = 0.035 if daily_vol <= 0.03 else 0.045
            stop_candidates.append(entry_price * (1 + pct_buffer))

            stop_loss_price = min(stop_candidates) if stop_candidates else entry_price * 1.03
            stop_loss_price = max(stop_loss_price, entry_price * 1.001)

            risk_amount = max(stop_loss_price - entry_price, entry_price * 0.001)
            tp_candidates = [entry_price - risk_amount * self.RR_TARGET]

            if len(df) >= 10:
                swing_low = float(df['low'].tail(10).min())
                tp_candidates.append(swing_low * 0.99)

            if len(df) >= 30:
                recent_close = float(df['price'].tail(30).min())
                tp_candidates.append(min(recent_close * 0.98, entry_price - risk_amount))

            take_profit_price = min(entry_price * 0.99, max(tp_candidates))
            take_profit_price = min(take_profit_price, entry_price * 0.999)

            rr = self._risk_reward_ratio(entry_price, stop_loss_price, take_profit_price, atr_raw, is_long=False)
            pos_pct = self._position_size_percentage(entry_price, stop_loss_price, atr_raw)

            return {
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'risk_reward_ratio': rr,
                'position_size_percentage': pos_pct,
            }
        except Exception as exc:
            logger.error(f"Short-term short trading levels failed: {exc}")
            return {
                'entry_price': current_price,
                'stop_loss_price': current_price * 1.03,
                'take_profit_price': current_price * 0.80,
                'risk_reward_ratio': 2.0,
                'position_size_percentage': 1.0,
            }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


PROFILE_PRESETS = {
    "default": {},
    "wide": {
        "limit": 400,
        "max_results": 20,
        "max_workers": 12,
        "analysis_days": 90,
    },
    "focused_llm": {
        "limit": 200,
        "top_per_side": 5,
        "use_openai_scoring": True,
        "min_volume_24h": 5_000_000,
        "min_volume_market_cap_ratio": 0.03,
        "intraday_lookback_days": 20,
        "unique_by_symbol": True,
        "max_risk_level": "MEDIUM",
    },
}


def _print_profile_presets(
    presets: Dict[str, Dict[str, Any]],
    default_profile: str,
    stream: TextIO = sys.stdout,
) -> None:
    """Display profile presets in a concise, human-friendly format."""

    header = f"Available profiles (default: {default_profile})"
    print(header, file=stream)

    if not presets:
        print("  (no presets defined)", file=stream)
        return

    default_marker = default_profile.strip().lower()

    for name in sorted(presets):
        overrides = presets[name]
        marker = "*" if name.strip().lower() == default_marker else "-"
        if overrides:
            entries = ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))
        else:
            entries = "(inherits environment defaults)"
        print(f" {marker} {name}: {entries}", file=stream)

    print(" * marks the default profile.", file=stream)


def _positive_int(value: str) -> int:
    """Argparse type that enforces a strictly positive integer."""

    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - argparse already handles TypeError
        raise argparse.ArgumentTypeError(f"Expected integer, received '{value}'") from exc

    if converted <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer")

    return converted


def _positive_float(value: str) -> float:
    """Argparse type that ensures a positive float."""

    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"Expected number, received '{value}'") from exc

    if converted <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive number")

    return converted


def make_risk_level_validator(valid_levels: Set[str]) -> Callable[[str], str]:
    """Create an argparse-compatible validator for risk level strings."""

    normalized_levels = {level.strip().upper() for level in valid_levels}

    def _validator(value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in normalized_levels:
            choices = ', '.join(sorted(normalized_levels))
            raise argparse.ArgumentTypeError(
                f"Invalid risk level '{value}'. Choose from: {choices}."
            )
        return normalized

    return _validator


def build_cli_parser(
    env_defaults: CryptoFinderConfig,
    default_limit: int,
    profile_default: str,
    default_max_risk: Optional[str],
    risk_level_type: Callable[[str], str],
    profile_presets: Optional[Dict[str, Dict[str, Any]]] = None,
) -> argparse.ArgumentParser:
    """Return the CLI parser with defaults injected for easier testing."""

    presets = profile_presets or PROFILE_PRESETS
    parser = argparse.ArgumentParser(
        description='Find high-conviction short-term cryptocurrency trades'
    )
    parser.add_argument(
        '--profile',
        choices=sorted(presets.keys()),
        default=profile_default,
        help=f"Preset bundle of frequently used parameters (default: {profile_default})",
    )
    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='Show available profile presets and exit',
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
            "Number of products to evaluate before ranking "
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
            "Maximum number of setups to display "
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
        help=f"Which trade direction(s) to include (default: {env_defaults.side})",
    )
    parser.add_argument(
        '--unique-by-symbol',
        action=argparse.BooleanOptionalAction,
        default=env_defaults.unique_by_symbol,
        help='Keep only the best direction per symbol',
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=env_defaults.min_overall_score,
        help=f"Discard setups below this overall score (default: {env_defaults.min_overall_score})",
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Optional path (.json or .csv) to persist results',
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to analyse (e.g., BTC,ETH,SOL)',
    )
    parser.add_argument(
        '--top-per-side',
        type=_positive_int,
        default=env_defaults.top_per_side,
        help='Cap the number of long and short setups before final merge',
    )
    parser.add_argument(
        '--max-workers',
        type=_positive_int,
        default=None,
        help=(
            "Override worker threads for API fetches "
            f"(default: {env_defaults.max_workers}; profile may override)"
        ),
    )
    parser.add_argument(
        '--offline',
        action=argparse.BooleanOptionalAction,
        default=env_defaults.offline,
        help='Use cache only; skip network requests where possible',
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
            "Number of daily bars for the swing window "
            f"(default: {env_defaults.analysis_days}; profile may override)"
        ),
    )
    parser.add_argument(
        '--intraday-lookback-days',
        type=_positive_int,
        default=env_defaults.intraday_lookback_days,
        help=(
            "Intraday history window (days) for hourly analytics "
            f"(default: {env_defaults.intraday_lookback_days})"
        ),
    )
    parser.add_argument(
        '--intraday-granularity',
        type=str,
        default=env_defaults.intraday_granularity,
        help=(
            "Coinbase candle granularity for intraday features (e.g., ONE_HOUR, THIRTY_MINUTE) "
            f"(default: {env_defaults.intraday_granularity})"
        ),
    )
    parser.add_argument(
        '--intraday-resample',
        type=str,
        default=env_defaults.intraday_resample,
        help=(
            "Resample rule for derived intraday metrics (pandas offset alias) "
            f"(default: {env_defaults.intraday_resample})"
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
    max_risk_help = 'Highest risk tier to allow (e.g., LOW, MEDIUM_LOW, MEDIUM)'
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
        help=f"Blend weight for OpenAI score (0-1); default {env_defaults.openai_weight}",
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

    st_openai_temp_default = (
        env_defaults.openai_temperature
        if env_defaults.openai_temperature is not None
        else 'model default'
    )
    parser.add_argument(
        '--openai-temperature',
        type=float,
        default=None,
        help=f"Temperature for OpenAI call (default: {st_openai_temp_default})",
    )
    parser.add_argument(
        '--openai-sleep-seconds',
        type=float,
        default=None,
        help=f"Optional pause between OpenAI calls (default: {env_defaults.openai_sleep_seconds})",
    )

    return parser


def main() -> None:
    env_defaults = build_short_term_config()

    valid_risk_levels = {level.name for level in RiskLevel}
    risk_level_type = make_risk_level_validator(valid_risk_levels)

    default_max_risk: Optional[str] = None
    if env_defaults.max_risk_level:
        try:
            default_max_risk = risk_level_type(str(env_defaults.max_risk_level))
        except argparse.ArgumentTypeError as exc:
            logger.warning(f"Ignoring invalid SHORT_MAX_RISK_LEVEL value: {exc}")

    default_limit = _env_override("SHORT_DEFAULT_LIMIT", 30, int)
    profile_default = os.getenv("SHORT_FINDER_PROFILE", "default")
    profile_presets: Dict[str, Dict[str, Any]] = PROFILE_PRESETS
    if profile_default not in profile_presets:
        profile_default = "default"

    raw_args = sys.argv[1:]

    parser = build_cli_parser(
        env_defaults,
        default_limit=default_limit,
        profile_default=profile_default,
        default_max_risk=default_max_risk,
        risk_level_type=risk_level_type,
        profile_presets=profile_presets,
    )
    args = parser.parse_args()

    if args.list_profiles:
        _print_profile_presets(profile_presets, profile_default)
        return

    explicit_flags = {
        token.split('=', 1)[0]
        for token in raw_args
        if token.startswith('--')
    }

    profile_overrides = profile_presets.get(args.profile, {})
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
    config.min_overall_score = float(args.min_score or 0.0)
    config.offline = args.offline
    top_per_side = args.top_per_side
    if '--top-per-side' not in explicit_flags and 'top_per_side' in profile_overrides:
        top_per_side = int(profile_overrides['top_per_side'])
    config.top_per_side = top_per_side
    config.max_workers = final_max_workers
    config.risk_free_rate = args.risk_free_rate
    config.analysis_days = final_analysis_days
    intraday_lookback = args.intraday_lookback_days
    if '--intraday-lookback-days' not in explicit_flags and 'intraday_lookback_days' in profile_overrides:
        intraday_lookback = int(profile_overrides['intraday_lookback_days'])
    config.intraday_lookback_days = intraday_lookback
    config.intraday_granularity = args.intraday_granularity.upper()
    config.intraday_resample = args.intraday_resample.upper()
    min_volume = args.min_volume
    if '--min-volume' not in explicit_flags and 'min_volume_24h' in profile_overrides:
        min_volume = float(profile_overrides['min_volume_24h'])
    config.min_volume_24h = min_volume

    min_vmc_ratio = args.min_vmc_ratio
    if '--min-vmc-ratio' not in explicit_flags and 'min_volume_market_cap_ratio' in profile_overrides:
        min_vmc_ratio = float(profile_overrides['min_volume_market_cap_ratio'])
    config.min_volume_market_cap_ratio = min_vmc_ratio

    if '--max-risk-level' in explicit_flags:
        config.max_risk_level = args.max_risk_level
    elif 'max_risk_level' in profile_overrides:
        try:
            config.max_risk_level = risk_level_type(str(profile_overrides['max_risk_level']))
        except argparse.ArgumentTypeError as exc:
            logger.warning("Ignoring invalid max_risk_level preset for %s: %s", args.profile, exc)
    else:
        config.max_risk_level = args.max_risk_level if args.max_risk_level is not None else config.max_risk_level

    config.force_refresh_candles = args.force_refresh

    unique_flag_set = '--unique-by-symbol' in explicit_flags or '--no-unique-by-symbol' in explicit_flags
    if not unique_flag_set and 'unique_by_symbol' in profile_overrides:
        config.unique_by_symbol = bool(profile_overrides['unique_by_symbol'])
    else:
        config.unique_by_symbol = args.unique_by_symbol

    openai_flag_set = '--use-openai-scoring' in explicit_flags or '--no-use-openai-scoring' in explicit_flags
    if not openai_flag_set and 'use_openai_scoring' in profile_overrides:
        config.use_openai_scoring = bool(profile_overrides['use_openai_scoring'])
    else:
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

    finder = ShortTermCryptoFinder(config=config)
    results = finder.find_best_opportunities(limit=final_limit)

    if not results:
        print("No short-term opportunities found. Adjust filters or broaden the symbol universe.")
        return

    def save_plain_report(path: Path, content: str, notify: bool = True, status_stream: TextIO = sys.stdout) -> None:
        tmp_path = Path(f"{path}.tmp.{os.getpid()}.{int(datetime.now().timestamp()*1000)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            handle.write(content)
        os.replace(tmp_path, path)
        if notify:
            print(f"Saved {len(results)} results to {path}", file=status_stream)

    if args.output == 'json' or (args.save and args.save.lower().endswith('.json')):
        json_results = []
        for crypto in results:
            pnl_profile = finder._compute_position_pnl(crypto)
            json_results.append({
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
            })

        if args.save and args.save.lower().endswith('.json'):
            finder._atomic_write_json(Path(args.save), json_results)
            if args.output == 'json':
                print(f"Saved {len(json_results)} results to {args.save}", file=sys.stderr)
            else:
                print(f"Saved {len(json_results)} results to {args.save}")

        if args.output == 'json':
            print(
                json.dumps(
                    {
                        'version': '1.0',
                        'config': finder.config.to_dict(),
                        'generated_utc': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%SZ'),
                        'results': json_results,
                    },
                    indent=2,
                )
            )
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

            tmp_path = Path(args.save + f".tmp.{os.getpid()}.{int(datetime.now().timestamp()*1000)}")
            final_path = Path(args.save)
            final_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
