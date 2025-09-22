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
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, List, Optional, TextIO

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
            elif rsi <= 65:
                rsi_score = 65
            else:
                rsi_score = 45
            score += rsi_score * 0.25

            macd = technical_metrics.get('macd_signal', 'NEUTRAL')
            if macd == 'BULLISH':
                macd_score = 85
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:
                macd_score = 40
            score += macd_score * 0.15

            bb = technical_metrics.get('bb_position', 'NEUTRAL')
            if bb == 'OVERSOLD':
                bb_score = 85
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:
                bb_score = 45
            score += bb_score * 0.15

            trend = float(technical_metrics.get('trend_strength', 0.0))
            if trend > 0.8:
                trend_score = 88
            elif trend > 0.2:
                trend_score = 72
            elif trend > -0.1:
                trend_score = 55
            else:
                trend_score = 35
            score += trend_score * 0.15

            vol_ratio = float(technical_metrics.get('volume_ratio', 1.0))
            if vol_ratio >= 1.8:
                volume_score = 90
            elif vol_ratio >= 1.3:
                volume_score = 75
            elif vol_ratio >= 1.0:
                volume_score = 60
            else:
                volume_score = 45
            score += volume_score * 0.10

            score += np.clip(momentum_score, 0.0, 100.0) * 0.20

            if technical_metrics.get('macd_cross'):
                hist = float(technical_metrics.get('macd_hist', 0.0) or 0.0)
                score += 4 if hist > 0 else -4

            return float(np.clip(score, 0.0, 100.0))
        except Exception as exc:
            logger.error(f"Short-term technical score failed: {exc}")
            return 50.0

    def _calculate_technical_score_short(self, technical_metrics: Dict, momentum_score_long: float) -> float:  # type: ignore[override]
        try:
            score = 0.0

            rsi = float(technical_metrics.get('rsi_14', 50.0))
            if rsi >= 68:
                rsi_score = 92
            elif rsi >= 55:
                rsi_score = 75
            elif rsi >= 40:
                rsi_score = 58
            else:
                rsi_score = 40
            score += rsi_score * 0.25

            macd = technical_metrics.get('macd_signal', 'NEUTRAL')
            if macd == 'BEARISH':
                macd_score = 85
            elif macd == 'NEUTRAL':
                macd_score = 60
            else:
                macd_score = 40
            score += macd_score * 0.15

            bb = technical_metrics.get('bb_position', 'NEUTRAL')
            if bb == 'OVERBOUGHT':
                bb_score = 88
            elif bb == 'NEUTRAL':
                bb_score = 60
            else:
                bb_score = 45
            score += bb_score * 0.15

            trend = float(technical_metrics.get('trend_strength', 0.0))
            if trend < -0.8:
                trend_score = 88
            elif trend < -0.2:
                trend_score = 74
            elif trend < 0.1:
                trend_score = 55
            else:
                trend_score = 35
            score += trend_score * 0.15

            vol_ratio = float(technical_metrics.get('volume_ratio', 1.0))
            if vol_ratio >= 1.8:
                volume_score = 88
            elif vol_ratio >= 1.3:
                volume_score = 72
            elif vol_ratio >= 1.0:
                volume_score = 55
            else:
                volume_score = 45
            score += volume_score * 0.10

            short_momentum = np.clip(100.0 - momentum_score_long, 0.0, 100.0)
            score += short_momentum * 0.20

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
}


def main() -> None:
    env_defaults = build_short_term_config()

    valid_risk_levels = {level.name for level in RiskLevel}

    def risk_level_type(value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in valid_risk_levels:
            raise argparse.ArgumentTypeError(
                f"Invalid risk level '{value}'. Choose from: {', '.join(sorted(valid_risk_levels))}."
            )
        return normalized

    default_max_risk: Optional[str] = None
    if env_defaults.max_risk_level:
        try:
            default_max_risk = risk_level_type(str(env_defaults.max_risk_level))
        except argparse.ArgumentTypeError as exc:
            logger.warning(f"Ignoring invalid SHORT_MAX_RISK_LEVEL value: {exc}")

    default_limit = _env_override("SHORT_DEFAULT_LIMIT", 30, int)
    profile_default = os.getenv("SHORT_FINDER_PROFILE", "default")
    if profile_default not in PROFILE_PRESETS:
        profile_default = "default"

    parser = argparse.ArgumentParser(description='Find high-conviction short-term cryptocurrency trades')
    parser.add_argument('--profile', choices=sorted(PROFILE_PRESETS.keys()), default=profile_default,
                        help=f"Preset bundle of frequently used parameters (default: {profile_default})")
    parser.add_argument('--plain-output', type=Path,
                        help='Write a formatted text report to this path (excludes log lines)')
    parser.add_argument('--suppress-console-logs', action='store_true',
                        help='Disable console log handler for cleaner stdout piping')
    parser.add_argument('--limit', type=int, default=None,
                        help=f"Number of products to evaluate before ranking (default: {default_limit}; profile may override)")
    parser.add_argument('--min-market-cap', type=int, default=env_defaults.min_market_cap,
                        help=f"Minimum market cap in USD (default: ${env_defaults.min_market_cap:,})")
    parser.add_argument('--max-results', type=int, default=None,
                        help=f"Maximum number of setups to display (default: {env_defaults.max_results}; profile may override)")
    parser.add_argument('--output', type=str, choices=['console', 'json'], default='console',
                        help='Output format (default: console)')
    parser.add_argument('--side', type=str, choices=['long', 'short', 'both'], default=env_defaults.side,
                        help=f"Which trade direction(s) to include (default: {env_defaults.side})")
    parser.add_argument('--unique-by-symbol', action='store_true', default=env_defaults.unique_by_symbol,
                        help='Keep only the best direction per symbol')
    parser.add_argument('--min-score', type=float, default=env_defaults.min_overall_score,
                        help=f"Discard setups below this overall score (default: {env_defaults.min_overall_score})")
    parser.add_argument('--save', type=str,
                        help='Optional path (.json or .csv) to persist results')
    parser.add_argument('--symbols', type=str,
                        help='Comma-separated list of symbols to analyse (e.g., BTC,ETH,SOL)')
    parser.add_argument('--top-per-side', type=int, default=env_defaults.top_per_side,
                        help='Cap the number of long and short setups before final merge')
    parser.add_argument('--max-workers', type=int, default=None,
                        help=f"Override worker threads for API fetches (default: {env_defaults.max_workers}; profile may override)")
    parser.add_argument('--offline', action='store_true', default=env_defaults.offline,
                        help='Use cache only; skip network requests where possible')
    parser.add_argument('--quotes', type=str,
                        help='Preferred quote currencies (comma-separated), e.g., USDC,USD,USDT')
    parser.add_argument('--risk-free-rate', type=float, default=env_defaults.risk_free_rate,
                        help=f"Override annual risk-free rate (default: {env_defaults.risk_free_rate})")
    parser.add_argument('--analysis-days', type=int, default=None,
                        help=f"Number of daily bars for the swing window (default: {env_defaults.analysis_days}; profile may override)")
    parser.add_argument('--max-risk-level', type=risk_level_type, default=default_max_risk,
                        help='Highest risk tier to allow (e.g., LOW, MEDIUM_LOW, MEDIUM)')

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
    config.unique_by_symbol = bool(args.unique_by_symbol)
    config.min_overall_score = float(args.min_score or 0.0)
    config.offline = bool(args.offline)
    config.top_per_side = args.top_per_side
    config.max_workers = final_max_workers
    config.risk_free_rate = args.risk_free_rate
    config.analysis_days = final_analysis_days
    config.max_risk_level = args.max_risk_level if args.max_risk_level is not None else config.max_risk_level

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
                'position_size_percentage', 'data_timestamp_utc'
            ]

            tmp_path = Path(args.save + f".tmp.{os.getpid()}.{int(datetime.now().timestamp()*1000)}")
            final_path = Path(args.save)
            final_path.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp_path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                        'data_timestamp_utc': getattr(crypto, 'data_timestamp_utc', ''),
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
