#!/usr/bin/env python3
"""
Quick snapshot helper for one or more symbols.

Fetches the raw finder metrics for the requested symbols (LONG and SHORT)
and prints entry/stop/TP plus risk/reward even if they wouldn't clear the
normal filters.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
try:
    from rich import box
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    RICH_CONSOLE = Console(color_system=None, force_terminal=False, width=120)
except Exception:
    RICH_AVAILABLE = False
    RICH_CONSOLE = None
    Table = None
    box = None

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXCLUDED_PERPS_PATH = REPO_ROOT / "config" / "excluded_perps.txt"
LIVE_CLOSED_LOG_PATH = REPO_ROOT / "trade_logs" / "watchdog_closed_positions.csv"
PAPER_CLOSED_LOG_PATH = REPO_ROOT / "trade_logs" / "paper_finder_closed_positions.csv"
RANGE_BREAK_STATUS_PATH = REPO_ROOT / "logs" / "range_break_status.json"
DAILY_STOP_HISTORY_PATH = REPO_ROOT / "logs" / "daily_stop_history.json"
_LOGGER = logging.getLogger(__name__)
_BTC_SYMBOLS = ["BTC-USD", "BTC-USDC", "BTC-USDT", "BTCUSD"]
_SUPPRESSION_COUNTS = {
    "total": 0,
    "regime_bullish": 0,
    "regime_bearish": 0,
    "regime_neutral": 0,
}

from short_term_crypto_finder import (
    PROFILE_PRESETS,
    ShortTermCryptoFinder,
    build_short_term_config,
)
from coinbaseservice import CoinbaseService
from perp_support import canonical_perp_symbol, perp_price_multiplier
from trading.risk_thresholds import load_risk_thresholds
from watchdog_close_old_positions import (
    _extract_entry_price,
    _extract_mark_price,
    _extract_unrealized_pnl,
    _get_portfolio_uuid,
)

try:
    from credentials import get_perps_credentials
except ModuleNotFoundError:
    try:  # pragma: no cover - fallback for older deployments
        from config import API_KEY_PERPS, API_SECRET_PERPS  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - no config available
        API_KEY_PERPS = ""  # type: ignore
        API_SECRET_PERPS = ""  # type: ignore

    def get_perps_credentials() -> Tuple[str, str]:
        return (API_KEY_PERPS or "", API_SECRET_PERPS or "")  # type: ignore[arg-type]

API_KEY_PERPS, API_SECRET_PERPS = get_perps_credentials()


def apply_profile_overrides(cfg, profile: str) -> None:
    preset = PROFILE_PRESETS.get(profile, {})
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _fmt(value, precision: int = 2) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return "n/a"


def _apply_risk_threshold_overrides(args: argparse.Namespace, argv: List[str]) -> None:
    overrides = load_risk_thresholds()
    if not overrides:
        return
    mapping = {
        "daily_stop_pct": ("daily_stop_pct", "--daily-stop-pct"),
        "daily_stop_usd": ("daily_stop_usd", "--daily-stop-usd"),
        "daily_stop_equity": ("daily_stop_equity", "--daily-stop-equity"),
        "range_break_symbol": ("range_break_symbol", "--range-break-symbol"),
        "range_break_days": ("range_break_days", "--range-break-days"),
        "range_break_atr_mult": ("range_break_atr_mult", "--range-break-atr-mult"),
        "range_break_confirmed_only": ("range_break_confirmed_only", "--range-break-confirmed-only"),
        "baseline_max_open": ("baseline_max_open", "--baseline-max-open"),
        "baseline_max_per_cluster": ("baseline_max_per_cluster", "--baseline-max-per-cluster"),
        "baseline_atr_mult": ("baseline_atr_mult", "--baseline-atr-mult"),
        "baseline_rr": ("baseline_rr", "--baseline-rr"),
        "baseline_atr_mode": ("baseline_atr_mode", "--baseline-atr-mode"),
        "baseline_expiry": ("baseline_expiry", "--baseline-expiry"),
        "baseline_position_pct": ("baseline_position_pct", "--baseline-position-pct"),
        "baseline_portfolio_usd": ("baseline_portfolio_usd", "--baseline-portfolio-usd"),
        "baseline_position_usd": ("baseline_position_usd", "--baseline-position-usd"),
        "baseline_live_position_usd": ("baseline_live_position_usd", "--baseline-live-position-usd"),
        "baseline_leverage": ("baseline_leverage", "--baseline-leverage"),
    }
    for key, (attr, flag) in mapping.items():
        if key not in overrides:
            continue
        if flag in argv:
            continue
        value = overrides.get(key)
        if value is None:
            continue
        setattr(args, attr, value)


def _fmt_usd_compact(value: Optional[float]) -> str:
    """Format a USD-ish number as 12.3K/4.5M/6.7B/1.2T."""
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except Exception:
        return "n/a"
    num = abs(num)
    if not (num >= 0):  # NaN
        return "n/a"
    for unit, denom in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if num >= denom:
            return f"{num / denom:.2f}{unit}"
    return f"{num:.0f}"


def _fmt_mult(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except Exception:
        return "n/a"
    if num >= 100:
        return f"x{num:.0f}"
    if num >= 10:
        return f"x{num:.1f}"
    return f"x{num:.2f}"


def _price_precision(entry: float) -> int:
    if entry < 1:
        return 6
    if entry < 10:
        return 4
    if entry < 1000:
        return 3
    return 2


def _fmt_price(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _load_excluded_perps(path: Path = EXCLUDED_PERPS_PATH) -> tuple[set[str], set[str]]:
    """Return (excluded_products, excluded_symbols) from config/excluded_perps.txt."""
    if not path.exists():
        return set(), set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return set(), set()
    excluded_products: set[str] = set()
    excluded_symbols: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        excluded_products.add(upper)
        base = upper.split("-")[0]
        candidates = {base}
        canonical_base = canonical_perp_symbol(base)
        if canonical_base:
            candidates.add(canonical_base)
        for candidate in candidates:
            if candidate:
                excluded_symbols.add(candidate.upper())
    return excluded_products, excluded_symbols


def _metric_score(metric: object) -> Optional[float]:
    if not metric:
        return None
    # In gate-scan we don't compute cross-sectional overall_score, so prefer
    # technical/momentum scores and only fall back to overall_score if > 0.
    for attr in ("technical_score", "momentum_score", "overall_score"):
        try:
            raw = getattr(metric, attr)
        except Exception:
            raw = None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if pd.isna(val):
            continue
        if attr == "overall_score" and val <= 0:
            continue
        return val
    return None


def _score_passes(score: Optional[float], min_score: float) -> bool:
    if min_score <= 0:
        return True
    if score is None:
        return False
    return score >= min_score


def _load_closed_trade_logs(
    paths: Iterable[Path],
    lookback_days: Optional[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "product_id" not in df.columns or "profit_loss" not in df.columns:
            continue
        cols = [c for c in ("product_id", "position_side", "profit_loss", "closed_at") if c in df.columns]
        df = df[cols].copy()
        df["product_id"] = df["product_id"].astype(str).str.upper()
        if "position_side" in df.columns:
            df["position_side"] = df["position_side"].astype(str).str.upper()
        df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
        df = df[df["profit_loss"].notna()]
        if "closed_at" in df.columns:
            df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce", utc=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["product_id", "position_side", "profit_loss", "closed_at"])
    df = pd.concat(frames, ignore_index=True)
    if lookback_days and "closed_at" in df.columns:
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        df = df[df["closed_at"] >= cutoff]
    return df


def _compute_perf_exclusions(
    df: pd.DataFrame,
    *,
    min_trades: int,
    drop_worst_frac: float,
    min_expectancy: Optional[float],
) -> tuple[set[str], dict]:
    if df.empty:
        return set(), {"eligible": 0, "excluded": 0}
    grouped = df.groupby("product_id").agg(
        trades=("profit_loss", "size"),
        expectancy=("profit_loss", "mean"),
    )
    grouped = grouped[grouped["trades"] >= max(1, int(min_trades or 0))]
    excluded: set[str] = set()
    if grouped.empty:
        return excluded, {"eligible": 0, "excluded": 0}
    if min_expectancy is not None:
        excluded |= set(grouped[grouped["expectancy"] < min_expectancy].index)
    frac = float(drop_worst_frac or 0.0)
    if frac > 0:
        if frac > 1:
            frac = frac / 100.0
        drop_n = int(len(grouped) * frac)
        if drop_n > 0:
            worst = grouped.sort_values("expectancy").head(drop_n)
            excluded |= set(worst.index)
    meta = {
        "eligible": int(len(grouped)),
        "excluded": int(len(excluded)),
        "min_trades": int(min_trades),
        "drop_worst_frac": float(drop_worst_frac),
        "min_expectancy": min_expectancy,
    }
    return excluded, meta


def _baseline_levels(
    *,
    side: str,
    entry: float,
    atr_raw: float,
    atr_mult: float,
    rr: float,
    atr_mode: str,
    finder: ShortTermCryptoFinder,
) -> tuple[float, float, float]:
    atr_eff = float(atr_raw)
    if atr_mode == "clipped":
        atr_eff = finder._cap_atr_value(atr_eff, entry)
    risk = atr_eff * float(atr_mult)
    if risk <= 0:
        return atr_eff, 0.0, 0.0
    if side == "LONG":
        stop = entry - risk
        tp = entry + risk * rr
    else:
        stop = entry + risk
        tp = entry - risk * rr
    return atr_eff, float(stop), float(tp)


def _load_open_perp_symbols() -> set[str]:
    if not API_KEY_PERPS or not API_SECRET_PERPS:
        return set()
    try:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    except Exception:
        return set()
    portfolio_uuid = _get_portfolio_uuid(cb)
    if not portfolio_uuid:
        return set()
    try:
        positions_response = cb.client.list_perps_positions(portfolio_uuid=portfolio_uuid)
    except Exception:
        return set()
    positions_raw = []
    if isinstance(positions_response, dict):
        positions_raw = positions_response.get("positions", []) or []
    else:
        positions_raw = getattr(positions_response, "positions", []) or []
    symbols: set[str] = set()
    for pos in positions_raw:
        pos_dict = pos if isinstance(pos, dict) else pos.to_dict()
        symbol = pos_dict.get("symbol") or pos_dict.get("product_id")
        if not symbol:
            continue
        base = canonical_perp_symbol(symbol)
        if base:
            base = base.split("-")[0]
        else:
            base = str(symbol).split("-")[0]
        if base:
            symbols.add(base.upper())
    return symbols


def _daily_pnl_today(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return None
    if "closed_at" not in df.columns or "profit_loss" not in df.columns:
        return None
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
    df = df.dropna(subset=["closed_at", "profit_loss"])
    today = datetime.now(timezone.utc).date()
    pnl_today = df.loc[df["closed_at"].dt.date == today, "profit_loss"].sum()
    try:
        return float(pnl_today)
    except Exception:
        return None


def _combine_pnl(closed_pnl: Optional[float], open_pnl: Optional[float]) -> Optional[float]:
    if closed_pnl is None and open_pnl is None:
        return None
    total = 0.0
    if closed_pnl is not None:
        total += closed_pnl
    if open_pnl is not None:
        total += open_pnl
    return total


def _load_open_paper_pnl(path: Path = Path("trade_logs/paper_finder_open_positions.csv")) -> Optional[float]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper() == "OPEN"]
    if df.empty:
        return 0.0
    if "unrealized_pnl" in df.columns:
        df["unrealized_pnl"] = pd.to_numeric(df["unrealized_pnl"], errors="coerce")
        pnl = df["unrealized_pnl"].sum(min_count=1)
        try:
            return float(pnl)
        except Exception:
            return None
    return None


def _load_open_live_pnl() -> Optional[float]:
    if not API_KEY_PERPS or not API_SECRET_PERPS:
        return None
    try:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    except Exception:
        return None
    portfolio_uuid = _get_portfolio_uuid(cb)
    if not portfolio_uuid:
        return None
    try:
        positions_response = cb.client.list_perps_positions(portfolio_uuid=portfolio_uuid)
    except Exception:
        return None

    def _from_currency(container, default: float = 0.0) -> Optional[float]:
        if container is None:
            return None
        if isinstance(container, dict):
            value = container.get("value")
            try:
                return float(value)
            except (TypeError, ValueError):
                return default
        try:
            return float(container)
        except (TypeError, ValueError):
            return default

    summary_obj = positions_response.get("summary") if isinstance(positions_response, dict) else getattr(positions_response, "summary", None)
    if summary_obj is not None:
        summary_total = _from_currency(summary_obj.get("aggregated_pnl") if isinstance(summary_obj, dict) else getattr(summary_obj, "aggregated_pnl", None))
        if summary_total is not None:
            return summary_total

    positions_raw = []
    if isinstance(positions_response, dict):
        positions_raw = positions_response.get("positions", []) or []
    else:
        positions_raw = getattr(positions_response, "positions", []) or []

    total = 0.0
    seen_any = False
    for pos in positions_raw:
        pos_dict = pos if isinstance(pos, dict) else pos.to_dict()
        try:
            net_size = float(pos_dict.get("net_size", 0) or 0.0)
        except Exception:
            net_size = 0.0
        side_raw = str(pos_dict.get("position_side") or "").upper()
        if "SHORT" in side_raw:
            net_size = -abs(net_size)
        else:
            net_size = abs(net_size)
        entry = _extract_entry_price(pos_dict)
        mark = _extract_mark_price(pos_dict)
        pnl = _extract_unrealized_pnl(pos_dict, net_size, entry, mark)
        if pnl is None:
            continue
        total += float(pnl)
        seen_any = True
    return total if seen_any else None


def _daily_stop_status(
    *,
    pnl_today: Optional[float],
    equity: float,
    stop_pct: float,
    stop_usd: float,
) -> tuple[bool, str]:
    if pnl_today is None:
        return False, "Daily stop: n/a (no closed trades yet)"
    thresholds = []
    if stop_pct and stop_pct > 0:
        thresholds.append(equity * (stop_pct / 100.0))
    if stop_usd and stop_usd > 0:
        thresholds.append(stop_usd)
    if not thresholds:
        return False, f"Daily stop: off (today P/L {pnl_today:+.2f})"
    threshold = min(thresholds)
    triggered = pnl_today <= -threshold
    pct_threshold = equity * (stop_pct / 100.0) if stop_pct and stop_pct > 0 else None
    pct_txt = f"{pct_threshold:.2f}" if pct_threshold is not None else "n/a"
    usd_txt = f"{stop_usd:.2f}" if stop_usd and stop_usd > 0 else "n/a"
    status = "ACTIVE" if triggered else "OK"
    reason = f"{pnl_today:+.2f} <= -{threshold:.2f}" if triggered else f"{pnl_today:+.2f} > -{threshold:.2f}"
    return triggered, f"Daily stop ({status}): {reason} (pct={pct_txt}, usd={usd_txt})"


def _load_daily_stop_history(path: Path = DAILY_STOP_HISTORY_PATH) -> dict:
    if not path.exists():
        return {"live": {"stops": [], "pause_until": None}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"live": {"stops": [], "pause_until": None}}
    if not isinstance(data, dict):
        return {"live": {"stops": [], "pause_until": None}}
    return data


def _load_range_break_status(path: Path = RANGE_BREAK_STATUS_PATH) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_stop_dates(values: List[str]) -> List[date]:
    dates: List[date] = []
    for val in values:
        try:
            dates.append(date.fromisoformat(str(val)))
        except Exception:
            continue
    return dates


def _count_recent(stops: List[date], today: date, window_days: int) -> int:
    if window_days <= 0:
        return 0
    cutoff = today - timedelta(days=window_days - 1)
    return sum(1 for d in stops if d >= cutoff)


def _live_pause_status(
    *,
    streak_window_days: int,
    streak_count: int,
    warn_window_days: int,
    warn_count: int,
    escalate_window_days: int,
    escalate_count: int,
) -> dict:
    history = _load_daily_stop_history()
    live = history.get("live", {}) if isinstance(history.get("live", {}), dict) else {}
    stops_raw = live.get("stops", [])
    if not isinstance(stops_raw, list):
        stops_raw = []
    stops = _parse_stop_dates([str(val) for val in stops_raw])

    today = datetime.now(timezone.utc).date()
    max_window = max(streak_window_days, warn_window_days, escalate_window_days, 1)
    cutoff = today - timedelta(days=max_window - 1)
    stops = [d for d in stops if d >= cutoff]

    pause_until = None
    existing_pause = live.get("pause_until")
    if existing_pause:
        try:
            pause_until = date.fromisoformat(str(existing_pause))
        except Exception:
            pause_until = None
    if pause_until is not None and pause_until < today:
        pause_until = None

    streak_hits = _count_recent(stops, today, streak_window_days)
    warn_hits = _count_recent(stops, today, warn_window_days)
    escalate_hits = _count_recent(stops, today, escalate_window_days)
    pause_active = pause_until is not None and pause_until >= today

    return {
        "streak_hits": streak_hits,
        "warn_hits": warn_hits,
        "escalate_hits": escalate_hits,
        "pause_until": pause_until,
        "pause_active": pause_active,
        "streak_window_days": streak_window_days,
        "warn_window_days": warn_window_days,
        "escalate_window_days": escalate_window_days,
        "streak_count": streak_count,
        "warn_count": warn_count,
        "escalate_count": escalate_count,
    }


def _range_break_check(
    *,
    df: pd.DataFrame,
    atr: float,
    days: int,
    atr_mult: float,
    confirmed_only: bool,
) -> Optional[dict]:
    if df is None or df.empty:
        return None
    if atr <= 0:
        return None
    lookback = max(int(days), 1)
    if len(df) < lookback + 1:
        return None
    prev = df.iloc[-(lookback + 1):-1]
    if prev.empty:
        return None
    try:
        range_high = float(prev["high"].max())
        range_low = float(prev["low"].min())
        close = float(df["price"].iloc[-1])
    except Exception:
        return None
    confirmed_close = None
    try:
        if len(df) >= 2:
            confirmed_close = float(df["price"].iloc[-2])
    except Exception:
        confirmed_close = None
    buffer = float(atr) * float(atr_mult)
    trigger_source = "confirmed" if confirmed_only else "intraday"
    if confirmed_only and confirmed_close is None:
        trigger_price = close
        breakout = False
        breakdown = False
    else:
        trigger_price = confirmed_close if confirmed_only else close
        breakout = trigger_price > range_high + buffer
        breakdown = trigger_price < range_low - buffer
    triggered = breakout or breakdown
    direction = "breakout" if breakout else "breakdown" if breakdown else "inside"
    overage = (
        (trigger_price - range_high - buffer)
        if breakout
        else (range_low - buffer - trigger_price)
        if breakdown
        else 0.0
    )
    confirmed_inside = None
    if confirmed_close is not None:
        confirmed_inside = (range_low - buffer) <= confirmed_close <= (range_high + buffer)
    return {
        "range_high": range_high,
        "range_low": range_low,
        "close": close,
        "confirmed_close": confirmed_close,
        "trigger_source": trigger_source,
        "trigger_price": float(trigger_price),
        "atr": float(atr),
        "buffer": buffer,
        "direction": direction,
        "triggered": bool(triggered),
        "overage": float(overage),
        "confirmed_inside": confirmed_inside,
        "days": lookback,
        "atr_mult": float(atr_mult),
    }


def _load_open_paper_symbols(path: Path = Path("trade_logs/paper_finder_open_positions.csv")) -> List[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    symbols: List[str] = []
    status_col = "status" if "status" in df.columns else None
    for _, row in df.iterrows():
        if status_col and str(row.get(status_col, "")).upper() != "OPEN":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            product = str(row.get("product_id") or "").strip().upper()
            if product:
                symbol = product.split("-")[0]
        if symbol:
            symbols.append(symbol)
    return symbols


def _true_range_last(df: pd.DataFrame) -> Optional[float]:
    """Compute the last candle true range (TR1) using high/low and previous close."""
    if df is None or len(df) < 2:
        return None
    try:
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        prev_close = float(df["price"].iloc[-2])
        return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    except Exception:
        return None


def _vol_regime_ratios(
    atr7: float, atr21: float, tr1: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """Return (ATR7/ATR21, TR1/ATR7) ratios to help judge volatility regime."""
    atr7_to_21 = (atr7 / atr21) if (atr21 and atr21 > 0) else None
    tr1_to_atr7 = (tr1 / atr7) if (tr1 is not None and atr7 > 0) else None
    return atr7_to_21, tr1_to_atr7


def _detect_btc_regime(
    finder: "ShortTermCryptoFinder",
    ema_period: int = 20,
    confirm_days: int = 2,
    buffer_bps: float = 0.0,
) -> str:
    """Detect BTC trend regime based on daily close vs EMA."""
    if ema_period <= 0 or confirm_days <= 0:
        return "neutral"
    df = None
    used_symbol = None
    days_needed = max(ema_period + confirm_days + 5, 35)
    for btc_symbol in _BTC_SYMBOLS:
        try:
            candidate = finder.get_historical_data(btc_symbol, days=days_needed)
        except Exception:
            continue
        if candidate is not None and len(candidate) >= ema_period + confirm_days:
            df = candidate
            used_symbol = btc_symbol
            break

    if df is None or len(df) < ema_period + confirm_days:
        _LOGGER.warning("Regime detection: could not fetch BTC daily candles; defaulting to neutral")
        return "neutral"

    try:
        df = df.sort_index()
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_convert("UTC")

        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        last_ts = df.index[-1]
        try:
            last_date = last_ts.date()
        except Exception:
            last_date = None
        if last_date == now_utc.date():
            df = df.iloc[:-1]
            _LOGGER.debug("Regime detection: dropped incomplete candle for %s", now_utc.date())
            if len(df) < ema_period + confirm_days:
                _LOGGER.warning("Regime detection: insufficient confirmed candles after dropping incomplete")
                return "neutral"

        close_col = "close" if "close" in df.columns else "price"
        closes = df[close_col].astype(float)
        df["ema"] = closes.ewm(span=ema_period, adjust=False).mean()

        buffer_mult = buffer_bps / 10000.0
        ema_upper = df["ema"] * (1 + buffer_mult)
        ema_lower = df["ema"] * (1 - buffer_mult)
        recent = df.tail(confirm_days)
        recent_closes = recent[close_col]
        recent_upper = ema_upper.tail(confirm_days)
        recent_lower = ema_lower.tail(confirm_days)
        above_count = (recent_closes > recent_upper).sum()
        below_count = (recent_closes < recent_lower).sum()

        if above_count == confirm_days:
            _LOGGER.info(
                "Regime detection (%s): BULLISH - %s closes above EMA%s (+%sbps)",
                used_symbol,
                confirm_days,
                ema_period,
                buffer_bps,
            )
            return "bullish"
        if below_count == confirm_days:
            _LOGGER.info(
                "Regime detection (%s): BEARISH - %s closes below EMA%s (-%sbps)",
                used_symbol,
                confirm_days,
                ema_period,
                buffer_bps,
            )
            return "bearish"
        _LOGGER.info(
            "Regime detection (%s): NEUTRAL - mixed signals in last %s closes",
            used_symbol,
            confirm_days,
        )
        return "neutral"
    except Exception as exc:
        _LOGGER.error("Regime detection error: %s", exc)
        return "neutral"


def _print_side(label: str, metric) -> None:
    if not metric:
        print(f"{label}: n/a")
        return

    def _price_prec(entry: float) -> int:
        try:
            val = float(entry)
        except Exception:
            return 4
        if val < 1:
            return 6
        if val < 10:
            return 4
        if val < 1000:
            return 3
        return 2

    price_prec = _price_prec(metric.entry_price)
    print(f"{label}: RR={_fmt(metric.risk_reward_ratio, 2)}  "
          f"entry={_fmt(metric.entry_price, price_prec)}  "
          f"SL={_fmt(metric.stop_loss_price, price_prec)}  "
          f"TP={_fmt(metric.take_profit_price, price_prec)}  "
          f"RSI14={_fmt(metric.rsi_14, 1)}  "
          f"trend={_fmt(metric.trend_strength, 3)}%/d  "
          f"mom={_fmt(metric.momentum_score, 2)}")


def _format_gates(
    long_m,
    short_m,
    tech: Dict,
    rr_target: float,
    atr_cap_usd: Optional[float],
    atr_cap_bps: Optional[float],
    price: Optional[float],
    rr_drivers_long: Optional[str] = None,
    rr_drivers_short: Optional[str] = None,
) -> tuple[str, str, str, Optional[str]]:
    atr_raw = float(tech.get("atr") or 0.0)
    atr_note = ""
    # Tiered dynamic bps (mirror finder logic)
    def _dyn_bps(p: float) -> float:
        if p >= 20000:
            return 325.0
        if p >= 2000:
            return 350.0
        if p >= 200:
            return 400.0
        return 450.0
    caps: List[float] = []
    if atr_cap_usd and atr_cap_usd > 0:
        caps.append(float(atr_cap_usd))
    if price and price > 0:
        tier_bps = _dyn_bps(price)
        eff_bps = None
        if atr_cap_bps and atr_cap_bps > 0:
            eff_bps = min(float(atr_cap_bps), tier_bps)
        else:
            eff_bps = tier_bps
        caps.append(float(price) * float(eff_bps) / 10000.0)
    if caps:
        cap_val = min(caps)
        headroom = cap_val - atr_raw
        if price and price > 0:
            headroom_bps = headroom / price * 10_000
            cap_bps = cap_val / price * 10_000
            if headroom_bps > 5000:  # >50% of price
                atr_note = " | ATR cap not binding"
            else:
                atr_note = f" | ATR headroom to cap: {headroom:+.2f} ({headroom_bps:+.0f} bps; cap={cap_bps:.0f} bps)"
        else:
            atr_note = f" | ATR headroom to cap: {headroom:+.2f}"
    def _rr_gap(m) -> str:
        if not m:
            return "n/a"
        try:
            rr = float(m.risk_reward_ratio)
            gap = rr_target - rr
            return f"{rr:.2f} (needs {gap:+.2f} to hit {rr_target:.1f})"
        except Exception:
            return "n/a"
    def _dist(m) -> str:
        if not m:
            return "n/a"
        try:
            entry = float(m.entry_price)
            sl = float(m.stop_loss_price)
            tp = float(m.take_profit_price)
            risk_pct = abs(entry - sl) / entry * 100
            reward_pct = abs(tp - entry) / entry * 100
            return f"risk {risk_pct:.2f}%, reward {reward_pct:.2f}%"
        except Exception:
            return "n/a"
    def _dist_atr(m) -> str:
        if not m or atr_raw <= 0:
            return "n/a"
        try:
            entry = float(m.entry_price)
            sl = float(m.stop_loss_price)
            tp = float(m.take_profit_price)
            sl_mult = abs(entry - sl) / atr_raw
            tp_mult = abs(tp - entry) / atr_raw
            return f"SL {sl_mult:.2f}x ATR, TP {tp_mult:.2f}x ATR"
        except Exception:
            return "n/a"
    gates_line = f"Gates: ATR7={_fmt(atr_raw, 2)}{atr_note} | RR long { _rr_gap(long_m) } | RR short { _rr_gap(short_m) }"
    dist_line = f"Distances: long {_dist(long_m)} | short {_dist(short_m)}"
    atr_line = f"ATR multiples: long {_dist_atr(long_m)} | short {_dist_atr(short_m)}"
    rr_line = None
    if rr_drivers_long or rr_drivers_short:
        rr_line = f"RR drivers: long {rr_drivers_long or 'n/a'} | short {rr_drivers_short or 'n/a'}"
    return gates_line, dist_line, atr_line, rr_line


def _print_gates(
    long_m,
    short_m,
    tech: Dict,
    rr_target: float,
    atr_cap_usd: Optional[float],
    atr_cap_bps: Optional[float],
    price: Optional[float],
    rr_drivers_long: Optional[str] = None,
    rr_drivers_short: Optional[str] = None,
) -> None:
    gates_line, dist_line, atr_line, rr_line = _format_gates(
        long_m,
        short_m,
        tech,
        rr_target,
        atr_cap_usd,
        atr_cap_bps,
        price,
        rr_drivers_long=rr_drivers_long,
        rr_drivers_short=rr_drivers_short,
    )
    print(gates_line)
    print(dist_line)
    print(atr_line)
    if rr_line:
        print(rr_line)


def _gate_scan_sort_key(row: Dict[str, object]) -> Tuple[float, float]:
    rr_gap = row.get("rr_gap")
    headroom = row.get("headroom_bps")
    try:
        rr_val = float(rr_gap) if rr_gap is not None else None
    except Exception:
        rr_val = None
    try:
        headroom_val = float(headroom) if headroom is not None else None
    except Exception:
        headroom_val = None
    return (rr_val if rr_val is not None else 1e9,
            -(headroom_val if headroom_val is not None else -1e9))


def _select_balanced_rows(
    rows: List[Dict[str, object]],
    top: int,
    regime: str = "neutral",
    tilt_pct: float = 70.0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if top <= 0 or not rows:
        return [], {
            "longs": 0,
            "shorts": 0,
            "long_slots": 0,
            "short_slots": 0,
            "min_required": 0,
            "selected_longs": 0,
            "selected_shorts": 0,
            "regime": regime,
            "suppressed": False,
        }
    longs = [row for row in rows if str(row.get("best_side") or "").upper() == "LONG"]
    shorts = [row for row in rows if str(row.get("best_side") or "").upper() == "SHORT"]
    longs.sort(key=_gate_scan_sort_key)
    shorts.sort(key=_gate_scan_sort_key)
    if regime == "bullish":
        long_slots = int(top * (tilt_pct / 100.0))
        short_slots = top - long_slots
    elif regime == "bearish":
        long_slots = top - int(top * (tilt_pct / 100.0))
        short_slots = int(top * (tilt_pct / 100.0))
    else:
        long_slots = top // 2
        short_slots = top // 2
    min_required_per_side = min(long_slots, short_slots)
    if min_required_per_side and (len(longs) < min_required_per_side or len(shorts) < min_required_per_side):
        _LOGGER.info(
            "Gate-scan SUPPRESSED (reversal protection): regime=%s, available L=%s/S=%s, required min=%s per side",
            regime,
            len(longs),
            len(shorts),
            min_required_per_side,
        )
        return [], {
            "longs": len(longs),
            "shorts": len(shorts),
            "long_slots": long_slots,
            "short_slots": short_slots,
            "min_required": min_required_per_side,
            "regime": regime,
            "suppressed": True,
            "suppression_reason": "insufficient_balance",
        }
    selected: List[Dict[str, object]] = []
    if long_slots:
        selected.extend(longs[:long_slots])
    if short_slots:
        selected.extend(shorts[:short_slots])
    remaining = top - len(selected)
    if remaining > 0:
        remainder_pool = longs[long_slots:] + shorts[short_slots:]
        remainder_pool.sort(key=_gate_scan_sort_key)
        selected.extend(remainder_pool[:remaining])
    selected.sort(key=_gate_scan_sort_key)
    final_longs = sum(1 for r in selected if str(r.get("best_side") or "").upper() == "LONG")
    final_shorts = len(selected) - final_longs
    return selected, {
        "longs": len(longs),
        "shorts": len(shorts),
        "long_slots": long_slots,
        "short_slots": short_slots,
        "min_required": min_required_per_side,
        "selected_longs": final_longs,
        "selected_shorts": final_shorts,
        "regime": regime,
        "suppressed": False,
    }


def _rr_driver_line(
    finder: ShortTermCryptoFinder,
    df: pd.DataFrame,
    tech: Dict,
    coin: Dict,
    metric,
    is_long: bool,
    rr_target: float,
    fee_slp_bps: float,
) -> str:
    if not metric:
        return "n/a"
    try:
        entry = float(metric.entry_price)
        sl = float(metric.stop_loss_price)
        tp = float(metric.take_profit_price)
    except Exception:
        return "n/a"
    if entry <= 0:
        return "n/a"

    price = float(coin.get("current_price") or entry)
    atr_raw = float(tech.get("atr") or 0.0)
    atr_raw = finder._cap_atr_value(atr_raw, price) if price > 0 else atr_raw
    vol_ratio = finder._volatility_ratio(tech)
    range_pos = finder._range_position(tech)
    min_risk = entry * (0.0008 + max(vol_ratio - 1.0, 0.0) * 0.0006)
    raw_risk = abs(entry - sl)
    risk_amount = max(raw_risk, min_risk)

    if is_long:
        tp_rr = entry + risk_amount * rr_target
        tp_candidates = [tp_rr]
        if len(df) >= 10:
            swing_high = float(df["high"].tail(10).max())
            tp_candidates.append(swing_high * 1.01)
        if len(df) >= 30:
            recent_close = float(df["price"].tail(30).max())
            tp_candidates.append(max(recent_close * 1.02, entry + risk_amount))
        intraday_high = float(tech.get("intraday_high_lookback", 0.0) or 0.0)
        if intraday_high > 0:
            tp_candidates.append(max(intraday_high * 0.998, entry + risk_amount * 0.8))
        base_tp = min(tp_candidates)
        if range_pos > 0.85:
            base_tp = min(base_tp, entry * 1.03)
        tp_final = max(entry * 1.005, base_tp)
        tp_final = max(tp_final, entry * 1.001)
    else:
        tp_rr = entry - risk_amount * rr_target
        tp_candidates = [tp_rr]
        if len(df) >= 10:
            swing_low = float(df["low"].tail(10).min())
            tp_candidates.append(swing_low * 0.99)
        if len(df) >= 30:
            recent_close = float(df["price"].tail(30).min())
            tp_candidates.append(min(recent_close * 0.98, entry - risk_amount))
        intraday_low = float(tech.get("intraday_low_lookback", 0.0) or 0.0)
        if intraday_low > 0:
            tp_candidates.append(min(intraday_low * 1.002, entry - risk_amount * 0.8))
        base_tp = max(tp_candidates)
        if range_pos < 0.15:
            base_tp = max(base_tp, entry * 0.97)
        tp_final = min(entry * 0.995, base_tp)
        tp_final = min(tp_final, entry * 0.999)

    tp_driver = "tp_rr"
    if abs(tp_final - tp_rr) / entry > 1e-4:
        tp_driver = "tp_clamp"

    atr_floor_mult = float(os.getenv("CRYPTO_ATR_SL_MULT", "0.5"))
    fee_add = (float(fee_slp_bps) / 10000.0) * entry if fee_slp_bps else 0.0
    dist_components = {
        "raw": raw_risk,
        "tick": entry * 1e-3,
        "atr_floor": (atr_raw * atr_floor_mult) if atr_raw > 0 else 0.0,
        "fee": fee_add,
    }
    risk_driver = max(dist_components.items(), key=lambda x: x[1])[0]
    return f"{tp_driver}, risk={risk_driver}"


def _dynamic_bps(price: float) -> float:
    if price >= 20000:
        return 325.0
    if price >= 2000:
        return 350.0
    if price >= 200:
        return 400.0
    return 450.0


def _effective_atr_cap(price: float, atr_cap_usd: Optional[float], atr_cap_bps: Optional[float]) -> Optional[float]:
    caps: List[float] = []
    if atr_cap_usd and atr_cap_usd > 0:
        caps.append(float(atr_cap_usd))
    if price and price > 0:
        dyn_bps = _dynamic_bps(price)
        eff_bps = None
        if atr_cap_bps and atr_cap_bps > 0:
            eff_bps = min(float(atr_cap_bps), dyn_bps)
        else:
            eff_bps = dyn_bps
        caps.append(float(price) * float(eff_bps) / 10000.0)
    return min(caps) if caps else None


def snapshot_symbols(symbols: Iterable[str], profile: str, disable_liquidity: bool) -> None:
    cfg = build_short_term_config()
    apply_profile_overrides(cfg, profile)
    cfg.symbols = list(symbols)
    cfg.force_refresh_candles = True
    if disable_liquidity:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0

    finder = ShortTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=None, symbols=cfg.symbols)
    if not coins:
        print("No symbols retrieved (check connectivity or liquidity filters).")
        return

    excluded_products, excluded_symbols = _load_excluded_perps()
    for coin in coins:
        product_id = coin["product_id"]
        symbol_raw = str(coin.get("symbol") or "").upper()
        base_symbol = symbol_raw.split("-")[0] if symbol_raw else str(product_id).split("-")[0].upper()
        if product_id in excluded_products or base_symbol in excluded_symbols:
            continue
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            print(f"{coin['symbol']}: no historical data")
            continue

        tech = finder.calculate_technical_indicators(df)
        mom = finder.calculate_momentum_score(df)
        chg = finder._calculate_price_changes_from_history(df)
        long_m = finder._build_long_metrics(coin, df, tech, mom, chg)
        short_m = finder._build_short_metrics(coin, df, tech, mom, chg)

        vol = coin.get('volume_24h')
        mc = coin.get('market_cap')
        cg_warn = False
        if (vol is None or vol == 0) or (mc is None or mc == 0):
            cg_warn = True
        ts = coin.get("data_timestamp_utc") or getattr(long_m, "data_timestamp_utc", "")

        atr21 = finder._calculate_atr(df, period=21) if len(df) >= 22 else 0.0
        atr7 = float(tech.get("atr") or 0.0)
        tr1 = _true_range_last(df)
        atr7_to_21, tr1_to_atr7 = _vol_regime_ratios(atr7, atr21, tr1)
        fee_slp_bps = finder._effective_fee_bps(coin)
        rr_target_long = finder._dynamic_rr_target(tech, is_long=True)
        rr_target_short = finder._dynamic_rr_target(tech, is_long=False)
        rr_driver_long = _rr_driver_line(
            finder,
            df,
            tech,
            coin,
            long_m,
            True,
            rr_target_long,
            fee_slp_bps,
        )
        rr_driver_short = _rr_driver_line(
            finder,
            df,
            tech,
            coin,
            short_m,
            False,
            rr_target_short,
            fee_slp_bps,
        )
        gates_line, dist_line, atr_line, rr_line = _format_gates(
            long_m,
            short_m,
            tech,
            rr_target=2.0,
            atr_cap_usd=getattr(cfg, "max_atr_usd", None),
            atr_cap_bps=getattr(cfg, "max_atr_bps", None),
            price=coin.get("current_price"),
            rr_drivers_long=rr_driver_long,
            rr_drivers_short=rr_driver_short,
        )
        vol_regime_line = (
            f"ATR21={_fmt(atr21, 2)}  ATR7/ATR21={_fmt(atr7_to_21, 2)}  TR1/ATR7={_fmt(tr1_to_atr7, 2)}"
        )

        if RICH_AVAILABLE and RICH_CONSOLE is not None:
            console = RICH_CONSOLE
            console.print("=" * 80)
            console.print(f"{coin['symbol']} ({coin.get('name','n/a')})  product={product_id}")
            overview = Table(title="Overview", box=box.ASCII, show_header=False)
            overview.add_column("Field", style="bold")
            overview.add_column("Value")
            overview.add_row("Price", _fmt(coin.get("current_price"), 2))
            overview.add_row("Vol24h", _fmt(vol, 0))
            overview.add_row("MCAP", _fmt(mc, 0))
            overview.add_row("Rank", str(coin.get("market_cap_rank", "n/a")))
            if ts:
                overview.add_row("Data TS (UTC)", ts)
            overview.add_row(
                "ATR/Vol",
                f"ATR7={_fmt(tech.get('atr'), 2)}  daily_vol_30d={_fmt(tech.get('daily_vol_30d'), 4)}  "
                f"intraday_range_pos={_fmt(tech.get('intraday_range_position'), 3)}  "
                f"intraday_vol_6h={_fmt(tech.get('intraday_volatility_6h'), 4)}  "
                f"spread_bps={_fmt(coin.get('spread_bps'), 3)}",
            )
            overview.add_row("Vol regime", vol_regime_line)
            overview.add_row("Gates", gates_line.replace("Gates: ", ""))
            overview.add_row("Distances", dist_line.replace("Distances: ", ""))
            overview.add_row("ATR multiples", atr_line.replace("ATR multiples: ", ""))
            if rr_line:
                overview.add_row("RR drivers", rr_line.replace("RR drivers: ", ""))
            console.print(overview)

            sides = Table(title="Sides", box=box.ASCII)
            sides.add_column("Side")
            sides.add_column("RR", justify="right")
            sides.add_column("Entry", justify="right")
            sides.add_column("SL", justify="right")
            sides.add_column("TP", justify="right")
            sides.add_column("RSI14", justify="right")
            sides.add_column("Trend %/d", justify="right")
            sides.add_column("Mom", justify="right")

            def _add_side_row(label: str, metric) -> None:
                if not metric:
                    sides.add_row(label, "n/a", "-", "-", "-", "-", "-", "-")
                    return
                def _price_prec(entry: float) -> int:
                    try:
                        val = float(entry)
                    except Exception:
                        return 4
                    if val < 1:
                        return 6
                    if val < 10:
                        return 4
                    if val < 1000:
                        return 3
                    return 2
                price_prec = _price_prec(metric.entry_price)
                sides.add_row(
                    label,
                    _fmt(metric.risk_reward_ratio, 2),
                    _fmt(metric.entry_price, price_prec),
                    _fmt(metric.stop_loss_price, price_prec),
                    _fmt(metric.take_profit_price, price_prec),
                    _fmt(metric.rsi_14, 1),
                    _fmt(metric.trend_strength, 3),
                    _fmt(metric.momentum_score, 2),
                )

            _add_side_row("LONG", long_m)
            _add_side_row("SHORT", short_m)
            console.print(sides)
            if cg_warn:
                console.print("WARNING: Missing/zero volume or market cap (CoinGecko/MC feed unavailable); liquidity checks may be incomplete.")
            console.print()
        else:
            print("=" * 80)
            print(f"{coin['symbol']} ({coin.get('name','n/a')})  product={product_id}")
            print(f"Price={_fmt(coin.get('current_price'), 2)}  "
                  f"Vol24h={_fmt(vol, 0)}  "
                  f"MCAP={_fmt(mc, 0)}  "
                  f"Rank={coin.get('market_cap_rank', 'n/a')}")
            if cg_warn:
                print("WARNING: Missing/zero volume or market cap (CoinGecko/MC feed unavailable); liquidity checks may be incomplete.")
            if ts:
                print(f"Data TS (UTC): {ts}")

            print(f"ATR7={_fmt(tech.get('atr'), 2)}  daily_vol_30d={_fmt(tech.get('daily_vol_30d'), 4)}  "
                  f"intraday_range_pos={_fmt(tech.get('intraday_range_position'), 3)}  "
                  f"intraday_vol_6h={_fmt(tech.get('intraday_volatility_6h'), 4)}  "
                  f"spread_bps={_fmt(coin.get('spread_bps'), 3)}")
            print(f"Vol regime: {vol_regime_line}")
            _print_gates(
                long_m,
                short_m,
                tech,
                rr_target=2.0,
                atr_cap_usd=getattr(cfg, "max_atr_usd", None),
                atr_cap_bps=getattr(cfg, "max_atr_bps", None),
                price=coin.get("current_price"),
                rr_drivers_long=rr_driver_long,
                rr_drivers_short=rr_driver_short,
            )
            _print_side("LONG", long_m)
            _print_side("SHORT", short_m)
            print()


def gate_scan(
    profile: str,
    disable_liquidity: bool,
    top: int,
    rr_target: float,
    scan_limit: Optional[int],
    baseline_commands: bool,
    baseline_paper_command: bool,
    baseline_portfolio_usd: Optional[float],
    baseline_position_pct: float,
    baseline_position_usd: Optional[float],
    baseline_live_position_usd: Optional[float],
    baseline_atr_mult: float,
    baseline_rr: float,
    baseline_atr_mode: str,
    baseline_partial_tp_rr: float,
    baseline_partial_tp_pct: float,
    baseline_partial_tp_move_sl: bool,
    baseline_limit_bps: float,
    baseline_leverage: Optional[float],
    baseline_expiry: str,
    baseline_include_open: bool,
    baseline_max_open: int,
    baseline_max_per_cluster: int,
    daily_stop_pct: float,
    daily_stop_usd: float,
    daily_stop_equity: float,
    range_break_symbol: str,
    range_break_days: int,
    range_break_atr_mult: float,
    range_break_confirmed_only: bool,
    balanced: bool,
    regime_tilt: bool,
    regime_ema_period: int,
    regime_confirm_days: int,
    regime_tilt_pct: float,
    regime_buffer_bps: float,
    perf_filter: bool,
    perf_source: str,
    perf_lookback_days: Optional[int],
    perf_min_trades: int,
    perf_drop_worst: float,
    perf_min_expectancy: Optional[float],
    min_score_long: float,
    min_score_short: float,
) -> None:
    cfg = build_short_term_config()
    apply_profile_overrides(cfg, profile)
    cfg.symbols = None  # scan the profile universe
    cfg.force_refresh_candles = True
    if disable_liquidity:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0

    finder = ShortTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=scan_limit, symbols=None)
    if not coins:
        print("No symbols retrieved (check connectivity or liquidity filters).")
        return

    if regime_tilt and not balanced:
        print("Regime tilt requested without --balanced; ignoring tilt.")
        regime_tilt = False

    regime = "neutral"
    if regime_tilt:
        regime = _detect_btc_regime(
            finder,
            ema_period=regime_ema_period,
            confirm_days=regime_confirm_days,
            buffer_bps=regime_buffer_bps,
        )
        tilt_pct_display = regime_tilt_pct if regime != "neutral" else 50.0
        buffer_note = f", {regime_buffer_bps:.0f}bps buffer" if regime_buffer_bps > 0 else ""
        print(
            "Regime: {regime} (BTC vs {ema} EMA, {days}d confirm{buffer}) -> {long_pct:.0f}/{short_pct:.0f} tilt".format(
                regime=regime.upper(),
                ema=regime_ema_period,
                days=regime_confirm_days,
                buffer=buffer_note,
                long_pct=tilt_pct_display,
                short_pct=100 - tilt_pct_display,
            )
        )

    perf_excluded: set[str] = set()
    perf_meta: Optional[dict] = None
    if perf_filter:
        sources: List[Path] = []
        source_key = (perf_source or "paper").lower()
        if source_key in {"paper", "both"}:
            sources.append(PAPER_CLOSED_LOG_PATH)
        if source_key in {"live", "both"}:
            sources.append(LIVE_CLOSED_LOG_PATH)
        perf_df = _load_closed_trade_logs(sources, perf_lookback_days)
        perf_excluded, perf_meta = _compute_perf_exclusions(
            perf_df,
            min_trades=perf_min_trades,
            drop_worst_frac=perf_drop_worst,
            min_expectancy=perf_min_expectancy,
        )
        if perf_meta and perf_meta.get("eligible", 0) == 0:
            print("Performance filter: no eligible trade history; skipping.")
        else:
            print(
                "Performance filter: excluded {excluded}/{eligible} products "
                "(min_trades={min_trades}, drop_worst={drop_worst_frac}, min_expectancy={min_expectancy}).".format(
                    excluded=len(perf_excluded),
                    eligible=perf_meta.get("eligible") if perf_meta else 0,
                    min_trades=perf_min_trades,
                    drop_worst_frac=perf_drop_worst,
                    min_expectancy="n/a" if perf_min_expectancy is None else f"{perf_min_expectancy:.2f}",
                )
            )
    if min_score_long > 0 or min_score_short > 0:
        print(
            "Score gates: min LONG score={long_score:.1f}, min SHORT score={short_score:.1f}.".format(
                long_score=min_score_long,
                short_score=min_score_short,
            )
        )

    live_daily_closed = _daily_pnl_today(LIVE_CLOSED_LOG_PATH)
    paper_daily_closed = _daily_pnl_today(PAPER_CLOSED_LOG_PATH)
    live_open_pnl = _load_open_live_pnl()
    paper_open_pnl = _load_open_paper_pnl()
    live_daily_pnl = _combine_pnl(live_daily_closed, live_open_pnl)
    paper_daily_pnl = _combine_pnl(paper_daily_closed, paper_open_pnl)
    live_stop, live_stop_msg = _daily_stop_status(
        pnl_today=live_daily_pnl,
        equity=daily_stop_equity,
        stop_pct=daily_stop_pct,
        stop_usd=daily_stop_usd,
    )
    paper_stop, paper_stop_msg = _daily_stop_status(
        pnl_today=paper_daily_pnl,
        equity=daily_stop_equity,
        stop_pct=daily_stop_pct,
        stop_usd=daily_stop_usd,
    )
    thresholds = load_risk_thresholds()
    streak_window_days = int(thresholds.get("daily_stop_streak_window_days", 7) or 7)
    streak_count = int(thresholds.get("daily_stop_streak_count", 3) or 3)
    warn_window_days = int(thresholds.get("daily_stop_warn_window_days", 14) or 14)
    warn_count = int(thresholds.get("daily_stop_warn_count", 5) or 5)
    escalate_window_days = int(thresholds.get("daily_stop_escalate_window_days", 21) or 21)
    escalate_count = int(thresholds.get("daily_stop_escalate_count", 7) or 7)
    live_pause_status = _live_pause_status(
        streak_window_days=streak_window_days,
        streak_count=streak_count,
        warn_window_days=warn_window_days,
        warn_count=warn_count,
        escalate_window_days=escalate_window_days,
        escalate_count=escalate_count,
    )
    range_break_info: Optional[dict] = None

    range_break_msg = "Range break: n/a"
    range_break_active = False
    range_break_latched = False
    range_break_latched_since = None
    range_break_symbol_upper = range_break_symbol.upper()
    if baseline_commands or baseline_paper_command:
        print(f"Daily stop (live): {live_stop_msg}")
        print(f"Daily stop (paper): {paper_stop_msg}")
        if live_pause_status["pause_active"]:
            pause_until = live_pause_status["pause_until"]
            pause_until_txt = pause_until.isoformat() if pause_until else "n/a"
            print(
                "Live pause (stop streak): ACTIVE until "
                f"{pause_until_txt} ({live_pause_status['streak_hits']}/{live_pause_status['streak_count']} in "
                f"{live_pause_status['streak_window_days']}d)"
            )
        else:
            print(
                "Live pause (stop streak): OK "
                f"({live_pause_status['streak_hits']}/{live_pause_status['streak_count']} in "
                f"{live_pause_status['streak_window_days']}d)"
            )
        if live_pause_status["warn_count"] > 0 and live_pause_status["warn_hits"] >= live_pause_status["warn_count"]:
            print(
                "Live stop warning: "
                f"{live_pause_status['warn_hits']}/{live_pause_status['warn_count']} in "
                f"{live_pause_status['warn_window_days']}d (consider reducing size 50%)"
            )
        if (
            live_pause_status["escalate_count"] > 0
            and live_pause_status["escalate_hits"] >= live_pause_status["escalate_count"]
        ):
            print(
                "Live stop escalation: "
                f"{live_pause_status['escalate_hits']}/{live_pause_status['escalate_count']} in "
                f"{live_pause_status['escalate_window_days']}d (tighten filters or paper-only)"
            )

    min_volume = float(getattr(cfg, "min_volume_24h", 0.0) or 0.0)
    min_ratio = float(getattr(cfg, "min_volume_market_cap_ratio", 0.0) or 0.0)
    major_symbols = {
        "BTC", "ETH", "SOL", "XRP", "USDT", "USDC",
        "ADA", "AVAX", "LINK", "DOGE", "LTC", "DOT", "MATIC",
    }
    baseline_exempt_symbols = {"BTC", "ETH"}
    memecoin_symbols = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "FARTCOIN", "PENGU"}
    excluded_products, excluded_symbols = _load_excluded_perps()

    def _spread_cap_bps(volume_usd: float) -> float:
        """Heuristic 'acceptable' spread cap for reporting (not a hard gate)."""
        if volume_usd >= 1e9:
            return 3.0
        if volume_usd >= 1e8:
            return 5.0
        return 10.0

    def _cluster_label(symbol: str) -> str:
        base = (symbol or "").upper()
        if base.startswith("1000"):
            base = base[4:]
        if base in memecoin_symbols:
            return "memecoins"
        if base in major_symbols:
            return "majors"
        return "alts"

    def _cluster_counts(symbols: Iterable[str]) -> dict[str, int]:
        counts = {"majors": 0, "memecoins": 0, "alts": 0}
        for sym in symbols:
            counts[_cluster_label(sym)] += 1
        return counts

    rows = []
    perf_skipped = 0
    for coin in coins:
        product_id = str(coin["product_id"])
        product_upper = product_id.upper()
        symbol = str(coin.get("symbol") or "").upper()
        base_symbol = symbol.split("-")[0] if symbol else str(product_id).split("-")[0].upper()
        if product_upper in excluded_products or base_symbol in excluded_symbols:
            continue
        if perf_excluded and product_upper in perf_excluded:
            perf_skipped += 1
            continue
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            continue
        tech = finder.calculate_technical_indicators(df)
        mom = finder.calculate_momentum_score(df)
        chg = finder._calculate_price_changes_from_history(df)
        long_m = finder._build_long_metrics(coin, df, tech, mom, chg)
        short_m = finder._build_short_metrics(coin, df, tech, mom, chg)
        long_score = _metric_score(long_m)
        short_score = _metric_score(short_m)
        price = float(coin.get("current_price") or 0.0)
        cap = _effective_atr_cap(price, getattr(cfg, "max_atr_usd", None), getattr(cfg, "max_atr_bps", None))
        atr = float(tech.get("atr") or 0.0)
        atr21 = finder._calculate_atr(df, period=21) if len(df) >= 22 else 0.0
        tr1 = _true_range_last(df)
        atr7_to_21, tr1_to_atr7 = _vol_regime_ratios(atr, atr21, tr1)
        atr_bps = (atr / price * 10_000) if price > 0 else None
        cap_bps = (cap / price * 10_000) if (cap and price > 0) else None
        headroom_bps = (cap_bps - atr_bps) if (cap_bps is not None and atr_bps is not None) else None

        if not range_break_info and symbol == range_break_symbol.upper():
            range_break_info = _range_break_check(
                df=df,
                atr=atr,
                days=range_break_days,
                atr_mult=range_break_atr_mult,
                confirmed_only=range_break_confirmed_only,
            )

        try:
            vol24h = float(coin.get("volume_24h") or 0.0)
        except Exception:
            vol24h = 0.0
        try:
            market_cap = float(coin.get("market_cap") or 0.0)
        except Exception:
            market_cap = 0.0
        ratio = (vol24h / market_cap) if market_cap > 0 else None
        ratio_gap_pp = ((ratio - min_ratio) * 100.0) if (ratio is not None and min_ratio > 0) else None
        vol_mult = (vol24h / min_volume) if (min_volume > 0 and vol24h >= 0) else None

        spread_bps = None
        try:
            spread_bps = float(coin.get("spread_bps")) if coin.get("spread_bps") is not None else None
        except Exception:
            spread_bps = None
        spread_cap = _spread_cap_bps(vol24h) if vol24h > 0 else 10.0
        spread_headroom = (spread_cap - spread_bps) if (spread_bps is not None and spread_bps > 0) else None

        def _rr_gap(m):
            if not m:
                return None
            try:
                rr = float(m.risk_reward_ratio)
                return max(0.0, rr_target - rr)
            except Exception:
                return None
        gaps = []
        if _score_passes(long_score, min_score_long):
            gaps.append(("LONG", _rr_gap(long_m), getattr(long_m, "risk_reward_ratio", None), long_score))
        if _score_passes(short_score, min_score_short):
            gaps.append(("SHORT", _rr_gap(short_m), getattr(short_m, "risk_reward_ratio", None), short_score))
        gaps = [g for g in gaps if g[1] is not None]
        if not gaps:
            continue
        gaps.sort(key=lambda x: x[1])
        best_side, best_gap, best_rr, best_score = gaps[0]
        atr_ratio = (atr_bps / cap_bps) if (atr_bps is not None and cap_bps) else None
        vmc_exempt = (coin.get("symbol") or "").upper() in major_symbols
        baseline_vmc_exempt = (coin.get("symbol") or "").upper() in baseline_exempt_symbols
        baseline_atr_ok = atr_ratio is not None and atr_ratio <= 1.5
        baseline_vmc_ok = (min_ratio <= 0) or (ratio is not None and ratio >= min_ratio) or baseline_vmc_exempt
        baseline_spread_ok = spread_headroom is not None and spread_headroom >= 0
        baseline_pass = baseline_atr_ok and baseline_vmc_ok and baseline_spread_ok
        rr_pass = best_rr is not None and best_rr >= rr_target

        rows.append({
            "symbol": coin["symbol"],
            "product": product_upper,
            "best_side": best_side,
            "rr_gap": best_gap,
            "rr": best_rr,
            "headroom_bps": headroom_bps,
            "atr_bps": atr_bps,
            "cap_bps": cap_bps,
            "atr_ratio": atr_ratio,
            "price": price,
            "atr_raw": atr,
            "atr7_to_21": atr7_to_21,
            "tr1_to_atr7": tr1_to_atr7,
            "vol24h": vol24h,
            "market_cap": market_cap,
            "vmc_ratio": ratio,
            "vmc_gap_pp": ratio_gap_pp,
            "vol_mult": vol_mult,
            "spread_bps": spread_bps,
            "spread_cap_bps": spread_cap,
            "spread_headroom_bps": spread_headroom,
            "vmc_exempt": vmc_exempt,
            "volume_source": str(coin.get("volume_24h_source") or "").strip(),
            "baseline_pass": baseline_pass,
            "rr_pass": rr_pass,
            "score": best_score,
            "long_score": long_score,
            "short_score": short_score,
        })

    if perf_excluded:
        print(f"Performance filter skipped {perf_skipped} symbols from the scan universe.")

    rows.sort(key=_gate_scan_sort_key)
    top_rows = rows[:top]
    balance_meta: Optional[Dict[str, object]] = None
    balance_note = None
    if balanced:
        top_rows, balance_meta = _select_balanced_rows(
            rows,
            top,
            regime=regime if regime_tilt else "neutral",
            tilt_pct=regime_tilt_pct,
        )
        if balance_meta.get("suppressed"):
            if regime_tilt and regime != "neutral":
                balance_note = (
                    "Regime-tilted gate-scan ({regime}): LONG={longs}, SHORT={shorts}, "
                    "min_required={min_required}. Insufficient balance; output suppressed (reversal protection).".format(
                        **balance_meta
                    )
                )
            else:
                balance_note = (
                    "Balanced gate-scan: LONG={longs}, SHORT={shorts}, "
                    "min_required={min_required}. Insufficient balance; output suppressed.".format(
                        **balance_meta
                    )
                )
        else:
            if regime_tilt and regime != "neutral":
                balance_note = (
                    "Regime-tilted gate-scan ({regime}): selected LONG={selected_longs}, SHORT={selected_shorts} "
                    "(target {long_slots}/{short_slots}, available {longs}/{shorts}).".format(**balance_meta)
                )
            else:
                balance_note = (
                    "Balanced gate-scan: LONG={longs}, SHORT={shorts}, selected={selected_longs}/{selected_shorts}.".format(
                        **balance_meta
                    )
                )

    def _write_range_break_status() -> None:
        if range_break_info is None:
            return
        RANGE_BREAK_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": range_break_symbol_upper,
            **range_break_info,
        }
        try:
            RANGE_BREAK_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    if range_break_info is not None:
        triggered_now = bool(range_break_info.get("triggered"))
        confirmed_inside = range_break_info.get("confirmed_inside")
        prev_status = _load_range_break_status() or {}
        prev_latched = bool(prev_status.get("latched"))
        prev_direction = str(prev_status.get("direction") or "")
        prev_latched_since = prev_status.get("latched_since")

        if triggered_now:
            range_break_latched = True
            if not prev_latched_since:
                range_break_latched_since = datetime.now(timezone.utc).isoformat()
            else:
                range_break_latched_since = prev_latched_since
        elif prev_latched:
            if confirmed_inside is True:
                range_break_latched = False
                range_break_latched_since = None
            else:
                range_break_latched = True
                range_break_latched_since = prev_latched_since

        if range_break_latched and not triggered_now and prev_direction in {"breakout", "breakdown"}:
            range_break_info["direction"] = prev_direction

        range_break_info["latched"] = range_break_latched
        range_break_info["latched_since"] = range_break_latched_since

        range_break_active = triggered_now or range_break_latched
        direction = range_break_info.get("direction", "inside")
        range_high = range_break_info.get("range_high")
        range_low = range_break_info.get("range_low")
        close = range_break_info.get("close")
        buffer = range_break_info.get("buffer")
        overage = range_break_info.get("overage", 0.0)
        confirmed_close = range_break_info.get("confirmed_close")
        trigger_source = range_break_info.get("trigger_source", "intraday")
        confirmed_txt = f"{confirmed_close:.2f}" if confirmed_close is not None else "n/a"
        mode_txt = f"{trigger_source}"
        state = "ACTIVE (latched)" if range_break_latched else "ACTIVE"
        if range_break_active:
            range_break_msg = (
                f"Range break {state}: {range_break_symbol_upper} {range_break_info['days']}d {direction} "
                f"({mode_txt}) | "
                f"close={close:.2f} range={range_low:.2f}-{range_high:.2f} buffer={buffer:.2f} "
                f"confirmed_close={confirmed_txt}"
            )
            if triggered_now:
                range_break_msg += f" over={overage:.2f}"
        else:
            range_break_msg = (
                f"Range break OK: {range_break_symbol_upper} {range_break_info['days']}d inside ({mode_txt}) | "
                f"close={close:.2f} range={range_low:.2f}-{range_high:.2f} buffer={buffer:.2f} "
                f"confirmed_close={confirmed_txt}"
            )
    print(range_break_msg)
    if balance_note:
        print(balance_note)
    if balance_meta and balance_meta.get("suppressed"):
        _SUPPRESSION_COUNTS["total"] += 1
        regime_key = f"regime_{balance_meta.get('regime')}"
        if regime_key in _SUPPRESSION_COUNTS:
            _SUPPRESSION_COUNTS[regime_key] += 1
        _LOGGER.info(
            "Gate-scan suppressed (total=%s): %s regime, L=%s, S=%s",
            _SUPPRESSION_COUNTS["total"],
            balance_meta.get("regime"),
            balance_meta.get("longs"),
            balance_meta.get("shorts"),
        )
        _write_range_break_status()
        return
    print(f"Top {len(top_rows)} closest to RR {rr_target}:")

    def _leverage_text(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        try:
            num = float(value)
        except Exception:
            return None
        if num.is_integer():
            return f"{int(num)}"
        return f"{num:.2f}"

    def _baseline_size_usd() -> Optional[float]:
        if baseline_live_position_usd is not None and baseline_live_position_usd > 0:
            return float(baseline_live_position_usd)
        if baseline_position_usd is not None and baseline_position_usd > 0:
            return float(baseline_position_usd)
        if baseline_portfolio_usd is not None and baseline_portfolio_usd > 0:
            return float(baseline_portfolio_usd) * (baseline_position_pct / 100.0)
        fallback = getattr(cfg, "report_position_notional", None)
        if fallback:
            return float(fallback)
        return None

    def _print_baseline_commands() -> None:
        if not baseline_commands or not top_rows:
            return
        if live_stop:
            print("Commands: suppressed (daily stop active for live trades).")
            return
        if range_break_active:
            print("Commands: suppressed (range break active).")
            return
        open_symbols = _load_open_perp_symbols() if (baseline_max_open or baseline_max_per_cluster or not baseline_include_open) else set()
        skip_symbols = set() if baseline_include_open else set(open_symbols)
        leverage_text = _leverage_text(
            baseline_leverage if baseline_leverage is not None else getattr(cfg, "report_leverage", None)
        )
        size_usd = _baseline_size_usd()
        if size_usd is None or size_usd <= 0:
            print("Commands: no size available (set --baseline-live-position-usd, --baseline-position-usd, or --baseline-portfolio-usd).")
            return

        initial_total_open = len(open_symbols)
        initial_cluster_counts = _cluster_counts(open_symbols)
        total_open = initial_total_open
        cluster_counts = dict(initial_cluster_counts)
        commands: List[str] = []
        skipped_open: List[str] = []
        skipped_capacity: List[str] = []
        skipped_cluster: List[str] = []
        for row in top_rows:
            if not row.get("baseline_pass"):
                continue
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            if symbol in skip_symbols:
                skipped_open.append(symbol)
                continue
            if baseline_max_open > 0 and total_open >= baseline_max_open:
                skipped_capacity.append(symbol)
                continue
            cluster = _cluster_label(symbol)
            if baseline_max_per_cluster > 0 and cluster_counts.get(cluster, 0) >= baseline_max_per_cluster:
                skipped_cluster.append(symbol)
                continue
            side = str(row.get("best_side") or "").upper()
            entry = float(row.get("price") or 0.0)
            atr_raw = float(row.get("atr_raw") or 0.0)
            if entry <= 0 or atr_raw <= 0:
                continue
            price_mult = perp_price_multiplier(symbol)
            entry *= price_mult
            atr_raw *= price_mult
            _, stop, tp = _baseline_levels(
                side=side,
                entry=entry,
                atr_raw=atr_raw,
                atr_mult=baseline_atr_mult,
                rr=baseline_rr,
                atr_mode=baseline_atr_mode,
                finder=finder,
            )
            if stop <= 0 or tp <= 0:
                continue
            precision = _price_precision(entry)
            tp_txt = _fmt_price(tp, precision)
            sl_txt = _fmt_price(stop, precision)
            partial_tp_txt = None
            if baseline_partial_tp_rr and baseline_partial_tp_pct:
                risk = abs(entry - stop)
                if risk > 0:
                    if side == "LONG":
                        partial_tp = entry + risk * baseline_partial_tp_rr
                    else:
                        partial_tp = entry - risk * baseline_partial_tp_rr
                    if partial_tp > 0:
                        partial_tp_txt = _fmt_price(partial_tp, precision)
            base_symbol = canonical_perp_symbol(symbol) or symbol
            product = f"{base_symbol}-PERP-INTX"
            side_ccxt = "BUY" if side == "LONG" else "SELL"
            cmd = f"python ccxt_trade_perp.py --product {product} --side {side_ccxt} --size {size_usd:.2f}"
            if leverage_text:
                cmd += f" --leverage {leverage_text}"
            cmd += f" --tp {tp_txt} --sl {sl_txt} --expiry {baseline_expiry}"
            if baseline_limit_bps and baseline_limit_bps > 0:
                limit_price = entry
                offset = entry * (baseline_limit_bps / 10000.0)
                if side == "LONG":
                    limit_price = entry + offset
                elif side == "SHORT":
                    limit_price = entry - offset
                limit_price = max(limit_price, 0.0)
                if limit_price > 0:
                    limit_txt = _fmt_price(limit_price, precision)
                    cmd += f" --limit {limit_txt}"
            if partial_tp_txt and baseline_partial_tp_pct:
                cmd += f" --tp1 {partial_tp_txt} --tp1-pct {baseline_partial_tp_pct:.2f}"
            commands.append(cmd)
            total_open += 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        if baseline_max_open > 0 or baseline_max_per_cluster > 0:
            print(
                "Open capacity: total {total}/{max_total} | majors {majors}/{max_cluster} | memecoins {memecoins}/{max_cluster} | alts {alts}/{max_cluster}".format(
                    total=initial_total_open,
                    max_total=baseline_max_open,
                    majors=initial_cluster_counts.get("majors", 0),
                    memecoins=initial_cluster_counts.get("memecoins", 0),
                    alts=initial_cluster_counts.get("alts", 0),
                    max_cluster=baseline_max_per_cluster,
                )
            )
            if commands:
                print(
                    "Projected capacity (after commands): total {total}/{max_total} | majors {majors}/{max_cluster} | memecoins {memecoins}/{max_cluster} | alts {alts}/{max_cluster}".format(
                        total=total_open,
                        max_total=baseline_max_open,
                        majors=cluster_counts.get("majors", 0),
                        memecoins=cluster_counts.get("memecoins", 0),
                        alts=cluster_counts.get("alts", 0),
                        max_cluster=baseline_max_per_cluster,
                    )
                )
        if skipped_open:
            print(f"Open positions detected (skipping): {', '.join(sorted(set(skipped_open)))}")
        if skipped_capacity:
            print(f"Max open reached (skipping): {', '.join(sorted(set(skipped_capacity)))}")
        if skipped_cluster:
            print(f"Cluster cap reached (skipping): {', '.join(sorted(set(skipped_cluster)))}")
        if commands:
            print("Commands:")
            print("\n".join(commands))
        else:
            print("Commands: none (no baseline-pass symbols without open positions).")

    def _print_baseline_paper_command() -> None:
        if not baseline_paper_command or not top_rows:
            return
        if paper_stop:
            print("Paper command: suppressed (daily stop active for paper trades).")
            return
        if range_break_active:
            print("Paper command: suppressed (range break active).")
            return
        open_paper_list = _load_open_paper_symbols() if (baseline_max_open or baseline_max_per_cluster or not baseline_include_open) else []
        skip_paper = set() if baseline_include_open else set(open_paper_list)

        symbol_specs: List[str] = []
        skipped_paper: List[str] = []
        skipped_capacity: List[str] = []
        skipped_cluster: List[str] = []
        initial_total_open = len(open_paper_list)
        initial_cluster_counts = _cluster_counts(open_paper_list)
        total_open = initial_total_open
        cluster_counts = dict(initial_cluster_counts)
        for row in top_rows:
            if not row.get("baseline_pass"):
                continue
            symbol = str(row.get("symbol") or "").upper()
            if not symbol:
                continue
            if symbol in skip_paper:
                skipped_paper.append(symbol)
                continue
            if baseline_max_open > 0 and total_open >= baseline_max_open:
                skipped_capacity.append(symbol)
                continue
            cluster = _cluster_label(symbol)
            if baseline_max_per_cluster > 0 and cluster_counts.get(cluster, 0) >= baseline_max_per_cluster:
                skipped_cluster.append(symbol)
                continue
            side = str(row.get("best_side") or "").upper()
            if side not in {"LONG", "SHORT"}:
                continue
            symbol_specs.append(f"{symbol}:{side}")
            total_open += 1
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        if baseline_max_open > 0 or baseline_max_per_cluster > 0:
            print(
                "Paper open capacity: total {total}/{max_total} | majors {majors}/{max_cluster} | memecoins {memecoins}/{max_cluster} | alts {alts}/{max_cluster}".format(
                    total=initial_total_open,
                    max_total=baseline_max_open,
                    majors=initial_cluster_counts.get("majors", 0),
                    memecoins=initial_cluster_counts.get("memecoins", 0),
                    alts=initial_cluster_counts.get("alts", 0),
                    max_cluster=baseline_max_per_cluster,
                )
            )
            if symbol_specs:
                print(
                    "Paper projected capacity (after command): total {total}/{max_total} | majors {majors}/{max_cluster} | memecoins {memecoins}/{max_cluster} | alts {alts}/{max_cluster}".format(
                        total=total_open,
                        max_total=baseline_max_open,
                        majors=cluster_counts.get("majors", 0),
                        memecoins=cluster_counts.get("memecoins", 0),
                        alts=cluster_counts.get("alts", 0),
                        max_cluster=baseline_max_per_cluster,
                    )
                )
        if skipped_paper:
            print(f"Open paper positions detected (skipping): {', '.join(sorted(set(skipped_paper)))}")
        if skipped_capacity:
            print(f"Paper max open reached (skipping): {', '.join(sorted(set(skipped_capacity)))}")
        if skipped_cluster:
            print(f"Paper cluster cap reached (skipping): {', '.join(sorted(set(skipped_cluster)))}")

        if not symbol_specs:
            print("Paper command: none (no baseline-pass symbols without open positions).")
            return

        cmd_parts = [
            "python scripts/baseline_finder_from_snapshot.py",
            f"--symbols {','.join(symbol_specs)}",
            f"--profile {profile}",
            f"--atr-mult {baseline_atr_mult}",
            f"--rr {baseline_rr}",
            f"--atr-mode {baseline_atr_mode}",
            "--open-paper",
        ]
        if baseline_portfolio_usd is not None and baseline_portfolio_usd > 0:
            cmd_parts.append(f"--portfolio-usd {baseline_portfolio_usd}")
        if baseline_position_pct is not None and baseline_position_pct > 0:
            cmd_parts.append(f"--position-pct {baseline_position_pct}")
        if baseline_position_usd is not None and baseline_position_usd > 0:
            cmd_parts.append(f"--fixed-position-usd {baseline_position_usd}")
        if baseline_leverage is not None:
            cmd_parts.append(f"--leverage {baseline_leverage}")
        if baseline_partial_tp_rr and baseline_partial_tp_rr > 0:
            cmd_parts.append(f"--partial-tp-rr {baseline_partial_tp_rr}")
        if baseline_partial_tp_pct and baseline_partial_tp_pct > 0:
            cmd_parts.append(f"--partial-tp-pct {baseline_partial_tp_pct}")
        if baseline_partial_tp_move_sl:
            cmd_parts.append("--partial-tp-move-sl")
        print("Paper command:")
        print(" ".join(cmd_parts))
    if RICH_AVAILABLE and RICH_CONSOLE is not None and top_rows:
        table = Table(title="Gate Scan (Short-Term)", box=box.ASCII, show_lines=False)
        table.add_column("Symbol")
        table.add_column("Side")
        table.add_column("RR", justify="right")
        table.add_column("Gap", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("ATR bps", justify="right")
        table.add_column("ATR cap", justify="left", no_wrap=True, overflow="ellipsis")
        table.add_column("Base", justify="center")
        table.add_column("RR", justify="center")
        table.add_column("Vol24h", justify="right")
        table.add_column("VMC", justify="right")
        table.add_column("Spr", justify="right")
        table.add_column("Regime", justify="right")

        for row in top_rows:
            hr = row["headroom_bps"]
            atr_txt = f"{row['atr_bps']:.0f}" if row['atr_bps'] is not None else "n/a"
            if hr is None:
                atr_gate_txt = "n/a"
            elif hr < 0:
                atr_gate_txt = f"clipped -{abs(hr):.0f}"
            else:
                atr_gate_txt = f"+{hr:.0f} head"

            vol_txt = _fmt_usd_compact(row.get("vol24h"))
            liq_mult = _fmt_mult(row.get("vol_mult")) if min_volume > 0 else "n/a"
            vol_cell = f"{vol_txt} ({liq_mult})"

            vmc_ratio = row.get("vmc_ratio")
            if vmc_ratio is None:
                vmc_cell = "n/a"
            else:
                vmc_txt = f"{vmc_ratio * 100.0:.2f}%"
                vmc_gap = row.get("vmc_gap_pp")
                if min_ratio > 0 and vmc_gap is not None:
                    vmc_cell = f"{vmc_txt} ({vmc_gap:+.2f}pp)"
                else:
                    vmc_cell = vmc_txt

            spr = row.get("spread_bps")
            if spr is None or spr <= 0:
                spr_cell = "n/a"
            else:
                spr_cap = float(row.get("spread_cap_bps") or 0.0)
                spr_hr = row.get("spread_headroom_bps")
                spr_hr_txt = f"{spr_hr:+.2f}" if spr_hr is not None else "n/a"
                spr_cell = f"{spr:.2f} ({spr_hr_txt}/{spr_cap:.0f})"

            regime = f"{_fmt(row.get('atr7_to_21'), 2)}/{_fmt(row.get('tr1_to_atr7'), 2)}"

            score_val = row.get("score")
            score_txt = f"{score_val:.1f}" if isinstance(score_val, (int, float)) else "n/a"
            table.add_row(
                row["symbol"],
                row["best_side"],
                f"{row['rr']:.2f}",
                f"{row['rr_gap']:.2f}",
                score_txt,
                atr_txt,
                atr_gate_txt,
                "Y" if row.get("baseline_pass") else "N",
                "Y" if row.get("rr_pass") else "N",
                vol_cell,
                vmc_cell,
                spr_cell,
                regime,
            )

        RICH_CONSOLE.print(table)
        _print_baseline_commands()
        _print_baseline_paper_command()
        _write_range_break_status()
        return

    for row in top_rows:
        hr = row["headroom_bps"]
        cap_txt = f"{row['cap_bps']:.0f} bps" if row['cap_bps'] is not None else "n/a"
        atr_txt = f"{row['atr_bps']:.0f} bps" if row['atr_bps'] is not None else "n/a"
        if hr is None:
            atr_gate_txt = "ATR cap n/a"
        elif hr < 0:
            atr_gate_txt = f"ATR CLIPPED (over cap by {abs(hr):.0f} bps)"
        else:
            atr_gate_txt = f"ATR within cap (+{hr:.0f} bps headroom)"

        # Liquidity + spread context (informational)
        vol_txt = _fmt_usd_compact(row.get("vol24h"))
        liq_mult = _fmt_mult(row.get("vol_mult")) if min_volume > 0 else "n/a"
        vmc_ratio = row.get("vmc_ratio")
        if vmc_ratio is None:
            vmc_txt = "n/a"
            vmc_gap_txt = "n/a"
        else:
            vmc_txt = f"{vmc_ratio * 100.0:.2f}%"
            if min_ratio > 0 and row.get("vmc_gap_pp") is not None:
                vmc_gap_txt = f"{row['vmc_gap_pp']:+.2f}pp"
                if row.get("vmc_exempt") and row["vmc_gap_pp"] < 0:
                    vmc_gap_txt += " (exempt)"
            else:
                vmc_gap_txt = "n/a"

        spr = row.get("spread_bps")
        if spr is None or spr <= 0:
            spr_txt = "n/a"
        else:
            spr_cap = float(row.get("spread_cap_bps") or 0.0)
            spr_hr = row.get("spread_headroom_bps")
            spr_hr_txt = f"{spr_hr:+.2f} bps" if spr_hr is not None else "n/a"
            spr_txt = f"{spr:.2f} bps ({spr_hr_txt}; cap={spr_cap:.0f})"

        base_txt = "pass" if row.get("baseline_pass") else "fail"
        rr_txt = "pass" if row.get("rr_pass") else "fail"
        score_txt = _fmt(row.get("score"), 1) if row.get("score") is not None else "n/a"
        print(f"{row['symbol']} ({row['product']}) {row['best_side']} "
              f"RR={row['rr']:.2f} (gap {row['rr_gap']:.2f}) score={score_txt} | "
              f"ATR {atr_txt}, cap {cap_txt}, {atr_gate_txt} | "
              f"base={base_txt}, rr={rr_txt} | "
              f"liq vol={vol_txt} ({liq_mult} vs min={_fmt_usd_compact(min_volume)}) "
              f"vmc={vmc_txt} ({vmc_gap_txt} vs {min_ratio * 100:.1f}%) | "
              f"spr={spr_txt} | "
              f"ATR7/ATR21={_fmt(row.get('atr7_to_21'), 2)} TR1/ATR7={_fmt(row.get('tr1_to_atr7'), 2)}")

    _print_baseline_commands()
    _print_baseline_paper_command()
    _write_range_break_status()

def main() -> None:
    parser = argparse.ArgumentParser(description="Print snapshot metrics for specific symbols.")
    parser.add_argument(
        "--symbols",
        required=False,
        help="Comma-separated symbols (e.g., BTC,ETH).",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=sorted(PROFILE_PRESETS.keys()),
        help="Finder profile to apply (default: default).",
    )
    parser.add_argument(
        "--no-liquidity-filter",
        action="store_true",
        help="Disable liquidity filters (min_volume, volume/market-cap ratio).",
    )
    parser.add_argument(
        "--gate-scan",
        action="store_true",
        help="Scan entire profile universe and print top symbols closest to RR/ATR gates (ignores --symbols).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many symbols to show in gate-scan mode (default: 15).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Require a balanced LONG/SHORT mix in gate-scan output (needs >= floor(top/2) per side).",
    )
    parser.add_argument(
        "--regime-tilt",
        action="store_true",
        help="Tilt long/short allocation based on BTC regime (70/30 split favoring trend direction).",
    )
    parser.add_argument(
        "--regime-ema-period",
        type=int,
        default=20,
        help="EMA period for regime detection (default: 20).",
    )
    parser.add_argument(
        "--regime-confirm-days",
        type=int,
        default=2,
        help="Consecutive days required to confirm regime (default: 2).",
    )
    parser.add_argument(
        "--regime-tilt-pct",
        type=float,
        default=70.0,
        help="Percentage allocation for favored side when regime active (default: 70).",
    )
    parser.add_argument(
        "--regime-buffer-bps",
        type=float,
        default=0.0,
        help="Buffer in basis points to reduce whipsaw (e.g., 20 = 0.2%%). Default: 0 (no buffer).",
    )
    parser.add_argument(
        "--min-score-long",
        type=float,
        default=0.0,
        help="Minimum technical/overall score required for LONG candidates (default: 0).",
    )
    parser.add_argument(
        "--min-score-short",
        type=float,
        default=0.0,
        help="Minimum technical/overall score required for SHORT candidates (default: 0).",
    )
    parser.add_argument(
        "--rr-target",
        type=float,
        default=2.0,
        help="RR target used for gate-scan gap calculations (default: 2.0).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Limit how many symbols to scan in gate-scan mode (e.g., 100 or 200). Default scans full profile universe.",
    )
    parser.add_argument(
        "--perf-filter",
        action="store_true",
        help="Exclude underperforming products based on closed-trade logs.",
    )
    parser.add_argument(
        "--perf-source",
        choices=("paper", "live", "both"),
        default="paper",
        help="Closed-trade logs to use for performance filtering (default: paper).",
    )
    parser.add_argument(
        "--perf-lookback-days",
        type=int,
        default=120,
        help="Lookback window (days) for performance filter (default: 120).",
    )
    parser.add_argument(
        "--perf-min-trades",
        type=int,
        default=5,
        help="Minimum trades per product to be eligible for performance filtering (default: 5).",
    )
    parser.add_argument(
        "--perf-drop-worst",
        type=float,
        default=0.0,
        help="Drop worst-performing fraction of products (0-1 or percent). Default: 0 (disabled).",
    )
    parser.add_argument(
        "--perf-min-expectancy",
        type=float,
        default=None,
        help="Exclude products with expectancy below this (default: disabled).",
    )
    parser.add_argument(
        "--baseline-commands",
        action="store_true",
        help="Print ccxt_trade_perp command lines for baseline-pass symbols in gate-scan mode.",
    )
    parser.add_argument(
        "--baseline-paper-command",
        action="store_true",
        help="Print baseline_finder_from_snapshot command line to open paper trades for baseline-pass symbols.",
    )
    parser.add_argument(
        "--baseline-portfolio-usd",
        type=float,
        default=None,
        help="Portfolio size used to compute baseline position size.",
    )
    parser.add_argument(
        "--baseline-position-pct",
        type=float,
        default=5.0,
        help="Position size percent of portfolio for baseline commands (default: 5).",
    )
    parser.add_argument(
        "--baseline-position-usd",
        type=float,
        default=None,
        help="Fixed USD size per trade for baseline commands (overrides portfolio/pct).",
    )
    parser.add_argument(
        "--baseline-live-position-usd",
        type=float,
        default=None,
        help="Fixed USD size for live baseline commands only (paper command still uses portfolio/pct unless fixed-position-usd is set).",
    )
    parser.add_argument(
        "--baseline-atr-mult",
        type=float,
        default=0.8,
        help="ATR multiple for baseline SL distance (default: 0.8).",
    )
    parser.add_argument(
        "--baseline-rr",
        type=float,
        default=1.5,
        help="Baseline RR target for command generation (default: 1.5).",
    )
    parser.add_argument(
        "--baseline-atr-mode",
        choices=["raw", "clipped"],
        default="clipped",
        help="Use raw or capped ATR for baseline commands (default: clipped).",
    )
    parser.add_argument(
        "--baseline-limit-bps",
        type=float,
        default=0.0,
        help="Marketable limit offset (bps) for live baseline commands (default: 0 disables).",
    )
    parser.add_argument(
        "--baseline-partial-tp-rr",
        type=float,
        default=0.0,
        help="RR multiple for partial take-profit (paper only, default: 0 disabled).",
    )
    parser.add_argument(
        "--baseline-partial-tp-pct",
        type=float,
        default=0.0,
        help="Percent of position to close at partial take-profit (paper only, default: 0 disabled).",
    )
    parser.add_argument(
        "--baseline-partial-tp-move-sl",
        action="store_true",
        help="After partial take-profit, move paper stop-loss to entry (paper only).",
    )
    parser.add_argument(
        "--baseline-leverage",
        type=float,
        default=None,
        help="Leverage for baseline commands (default uses profile leverage).",
    )
    parser.add_argument(
        "--baseline-expiry",
        type=str,
        default="30d",
        help="Expiry string for baseline commands (default: 30d).",
    )
    parser.add_argument(
        "--baseline-max-open",
        type=int,
        default=10,
        help="Max open positions allowed before commands are suppressed (default: 10; set 0 to disable).",
    )
    parser.add_argument(
        "--baseline-max-per-cluster",
        type=int,
        default=3,
        help="Max open positions per cluster (majors/memecoins/alts) (default: 3; set 0 to disable).",
    )
    parser.add_argument(
        "--baseline-include-open",
        action="store_true",
        help="Include symbols that already have open live positions when printing commands.",
    )
    parser.add_argument(
        "--daily-stop-pct",
        type=float,
        default=2.0,
        help="Daily stop loss percent of equity for gate-scan commands (default: 2).",
    )
    parser.add_argument(
        "--daily-stop-usd",
        type=float,
        default=20.0,
        help="Daily stop loss USD for gate-scan commands (default: 20).",
    )
    parser.add_argument(
        "--daily-stop-equity",
        type=float,
        default=1000.0,
        help="Equity baseline used for daily stop percent (default: 1000).",
    )
    parser.add_argument(
        "--range-break-symbol",
        type=str,
        default="BTC",
        help="Symbol used to detect range breaks for the circuit breaker (default: BTC).",
    )
    parser.add_argument(
        "--range-break-days",
        type=int,
        default=7,
        help="Lookback days for range-break high/low (default: 7).",
    )
    parser.add_argument(
        "--range-break-atr-mult",
        type=float,
        default=0.5,
        help="ATR multiple for range-break buffer (default: 0.5).",
    )
    parser.add_argument(
        "--range-break-confirmed-only",
        action="store_true",
        default=None,
        help="Trigger range-breaks only on confirmed daily closes (default: false).",
    )
    args = parser.parse_args()
    _apply_risk_threshold_overrides(args, sys.argv[1:])

    if args.gate_scan:
        gate_scan(
            profile=args.profile,
            disable_liquidity=args.no_liquidity_filter,
            top=args.top,
            rr_target=args.rr_target,
            scan_limit=args.scan_limit,
            baseline_commands=args.baseline_commands,
            baseline_paper_command=args.baseline_paper_command,
            baseline_portfolio_usd=args.baseline_portfolio_usd,
            baseline_position_pct=args.baseline_position_pct,
            baseline_position_usd=args.baseline_position_usd,
            baseline_live_position_usd=args.baseline_live_position_usd,
            baseline_atr_mult=args.baseline_atr_mult,
            baseline_rr=args.baseline_rr,
            baseline_atr_mode=args.baseline_atr_mode,
            baseline_partial_tp_rr=args.baseline_partial_tp_rr,
            baseline_partial_tp_pct=args.baseline_partial_tp_pct,
            baseline_partial_tp_move_sl=args.baseline_partial_tp_move_sl,
            baseline_limit_bps=args.baseline_limit_bps,
            baseline_leverage=args.baseline_leverage,
            baseline_expiry=args.baseline_expiry,
            baseline_include_open=args.baseline_include_open,
            baseline_max_open=args.baseline_max_open,
            baseline_max_per_cluster=args.baseline_max_per_cluster,
            daily_stop_pct=args.daily_stop_pct,
            daily_stop_usd=args.daily_stop_usd,
            daily_stop_equity=args.daily_stop_equity,
            range_break_symbol=args.range_break_symbol,
            range_break_days=args.range_break_days,
            range_break_atr_mult=args.range_break_atr_mult,
            range_break_confirmed_only=bool(args.range_break_confirmed_only),
            balanced=args.balanced,
            regime_tilt=args.regime_tilt,
            regime_ema_period=args.regime_ema_period,
            regime_confirm_days=args.regime_confirm_days,
            regime_tilt_pct=args.regime_tilt_pct,
            regime_buffer_bps=args.regime_buffer_bps,
            perf_filter=args.perf_filter,
            perf_source=args.perf_source,
            perf_lookback_days=args.perf_lookback_days,
            perf_min_trades=args.perf_min_trades,
            perf_drop_worst=args.perf_drop_worst,
            perf_min_expectancy=args.perf_min_expectancy,
            min_score_long=args.min_score_long,
            min_score_short=args.min_score_short,
        )
        return

    if not args.symbols:
        parser.error("No symbols provided after parsing.")
    syms = _parse_symbols(args.symbols)
    if not syms:
        parser.error("No symbols provided after parsing.")
    snapshot_symbols(syms, profile=args.profile, disable_liquidity=args.no_liquidity_filter)


if __name__ == "__main__":
    main()
