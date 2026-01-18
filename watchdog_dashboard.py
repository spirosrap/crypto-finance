#!/usr/bin/env python3
"""
Streamlit dashboard for watchdog logs.

Launch with:
    streamlit run watchdog_dashboard.py
"""

from __future__ import annotations

import json
import html
import math
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
try:  # pragma: no cover - optional dependency
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover - optional dependency
    st_autorefresh = None  # type: ignore[assignment]

from coinbaseservice import CoinbaseService
from fills_pnl import compute_round_trip_cycles, fetch_fills
from trading.risk_thresholds import load_risk_thresholds

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

from watchdog_utils import filter_by_date, select_count_window, select_last
from watchdog_close_old_positions import (
    _extract_entry_price,
    _extract_mark_price,
    _extract_position_open_time,
    _extract_symbol_and_size,
    _extract_unrealized_pnl,
    _get_portfolio_uuid,
)
from watchdog_metrics import build_snapshot
try:
    from trading.equity_report import build_equity_figure
except Exception as exc:  # pragma: no cover - optional dependency
    build_equity_figure = None  # type: ignore[assignment]
    EQUITY_REPORT_IMPORT_ERROR = exc


UTC = timezone.utc
AUTO_REFRESH_DEFAULT_SEC = 60
RISK_THRESHOLDS = load_risk_thresholds()
DAILY_STOP_PCT = float(RISK_THRESHOLDS.get("daily_stop_pct", 2.0) or 0.0)
DAILY_STOP_USD = float(RISK_THRESHOLDS.get("daily_stop_usd", 20.0) or 0.0)
PAPER_CLOSED_CSV = Path("trade_logs/paper_finder_closed_positions.csv")
PAPER_OPEN_CSV = Path("trade_logs/paper_finder_open_positions.csv")
STATE_PATH = Path("cache/watchdog_dashboard_state.json")
RANGE_BREAK_STATUS_PATH = REPO_ROOT / "logs" / "range_break_status.json"
LIVE_SNAPSHOT_PATH = REPO_ROOT / "logs" / "live_snapshot.json"
LOG_HEARTBEATS = (
    ("Gate scan", REPO_ROOT / "logs" / "gate_scan_paper.log", 4 * 60 * 60),
    ("Paper update", REPO_ROOT / "logs" / "paper_finder_update.log", 5 * 60),
    ("Fill poll", REPO_ROOT / "logs" / "watchdog_close_update.log", 5 * 60),
    ("Live snapshot", REPO_ROOT / "logs" / "live_snapshot_update.log", 5 * 60),
)
LOG_HEARTBEAT_STALE_MULTIPLIER = 3.0


API_KEY_PERPS, API_SECRET_PERPS = get_perps_credentials()


def load_watchdog_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    if "opened_at" in df.columns:
        df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
    if "profit_loss_pct" in df.columns:
        df["profit_loss_pct"] = pd.to_numeric(df["profit_loss_pct"], errors="coerce")
    df = df.dropna(subset=["closed_at"]).sort_values("closed_at")
    return df


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_order_payload(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, dict):
        if "order" in payload and isinstance(payload["order"], dict):
            return payload["order"]
        return payload
    order = getattr(payload, "order", None)
    if isinstance(order, dict):
        return order
    try:
        data = vars(payload)
    except TypeError:
        return {}
    if isinstance(data, dict):
        if "order" in data and isinstance(data["order"], dict):
            return data["order"]
        return data
    return {}


def _extract_order_configuration(order: Dict[str, Any]) -> Dict[str, Any]:
    if not order:
        return {}
    config = order.get("order_configuration")
    if isinstance(config, dict):
        return config
    info = order.get("info")
    if isinstance(info, dict):
        nested = info.get("order_configuration")
        if isinstance(nested, dict):
            return nested
    return {}


def _select_bracket_target(config: Dict[str, Any], reason: str) -> Optional[float]:
    for key in ("trigger_bracket_gtd", "trigger_bracket_gtc", "trigger_bracket_ioc"):
        bracket = config.get(key)
        if not isinstance(bracket, dict):
            continue
        tp_price = _coerce_float(bracket.get("limit_price"))
        sl_price = _coerce_float(bracket.get("stop_trigger_price"))
        normalized = (reason or "").lower()
        if normalized in {"stop_loss", "expired_breakeven"} and sl_price is not None:
            return sl_price
        if normalized in {"take_profit", "partial_take"} and tp_price is not None:
            return tp_price
        return tp_price if tp_price is not None else sl_price
    return None


def _select_limit_target(config: Dict[str, Any]) -> Optional[float]:
    for key in ("limit_limit_gtd", "limit_limit_gtc", "limit_limit_ioc"):
        limit_cfg = config.get(key)
        if isinstance(limit_cfg, dict):
            price = _coerce_float(limit_cfg.get("limit_price"))
            if price is not None:
                return price
    return None


def _select_stop_limit_target(config: Dict[str, Any], reason: str) -> Optional[float]:
    for key in ("stop_limit_stop_limit_gtd", "stop_limit_stop_limit_gtc", "stop_limit_stop_limit_ioc"):
        stop_cfg = config.get(key)
        if not isinstance(stop_cfg, dict):
            continue
        stop_price = _coerce_float(stop_cfg.get("stop_price") or stop_cfg.get("stop_trigger_price"))
        limit_price = _coerce_float(stop_cfg.get("limit_price"))
        normalized = (reason or "").lower()
        if normalized in {"stop_loss", "expired_breakeven"} and stop_price is not None:
            return stop_price
        return limit_price if limit_price is not None else stop_price
    return None


def _extract_target_price(order_payload: Any, reason: str) -> Optional[float]:
    order = _normalize_order_payload(order_payload)
    config = _extract_order_configuration(order)
    if not config:
        return None
    target = _select_bracket_target(config, reason)
    if target is not None:
        return target
    target = _select_stop_limit_target(config, reason)
    if target is not None:
        return target
    return _select_limit_target(config)


def _compute_exit_slippage(
    exit_price: float,
    target_price: float,
    position_side: str,
) -> Tuple[float, float]:
    """
    Return (slippage_price, slippage_bps). Positive means worse than target.
    """
    side = (position_side or "").upper()
    if side == "SHORT":
        slippage_price = exit_price - target_price
    else:
        slippage_price = target_price - exit_price
    slippage_bps = (slippage_price / target_price * 10000.0) if target_price else 0.0
    return slippage_price, slippage_bps


def _fetch_live_fills(limit: int) -> List[Dict[str, Any]]:
    api_key, api_secret = get_perps_credentials()
    if not api_key or not api_secret:
        raise RuntimeError("Missing perps API credentials for Coinbase fills.")
    cb = CoinbaseService(api_key=api_key, api_secret=api_secret)
    return fetch_fills(cb, limit=limit)


def _flatten_cycles(cycles_by_product: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for product_id, cycles in cycles_by_product.items():
        for cycle in cycles:
            record = dict(cycle)
            record["product_id"] = product_id
            flattened.append(record)
    return flattened


def build_exit_slippage_table(
    trades: pd.DataFrame,
    fetch_order: Callable[[str], Any],
    max_orders: int = 200,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    required_cols = {"order_id", "exit_price", "net_size", "position_side"}
    if not required_cols.issubset(set(trades.columns)):
        return pd.DataFrame()

    working = trades.copy()
    working["exit_price"] = pd.to_numeric(working["exit_price"], errors="coerce")
    working["net_size"] = pd.to_numeric(working["net_size"], errors="coerce")
    working = working.dropna(subset=["order_id", "exit_price", "net_size"])
    working = working[working["order_id"].astype(str).str.len() > 0]
    if working.empty:
        return pd.DataFrame()

    if max_orders and len(working) > max_orders:
        working = working.sort_values("closed_at").tail(max_orders)

    rows: List[Dict[str, Any]] = []
    for _, row in working.iterrows():
        order_id = str(row.get("order_id") or "").strip()
        if not order_id:
            continue
        order_payload = fetch_order(order_id)
        target_price = _extract_target_price(order_payload, str(row.get("closure_reason") or ""))
        if target_price is None:
            continue
        exit_price = _coerce_float(row.get("exit_price"))
        if exit_price is None:
            continue
        slippage_price, slippage_bps = _compute_exit_slippage(
            exit_price,
            float(target_price),
            str(row.get("position_side") or ""),
        )
        size = abs(float(row.get("net_size") or 0.0))
        slippage_usd = slippage_price * size
        rows.append(
            {
                "product_id": row.get("product_id"),
                "closed_at": row.get("closed_at"),
                "closure_reason": row.get("closure_reason"),
                "position_side": row.get("position_side"),
                "exit_price": exit_price,
                "target_price": float(target_price),
                "slippage_price": slippage_price,
                "slippage_bps": slippage_bps,
                "slippage_usd": slippage_usd,
                "order_id": order_id,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("closed_at").reset_index(drop=True)


def collapse_trade_rows(df: pd.DataFrame, include_partial_only: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    trades = df.copy()
    trades["closed_at"] = pd.to_datetime(trades["closed_at"], utc=True, errors="coerce")
    if "profit_loss" in trades.columns:
        trades["profit_loss"] = pd.to_numeric(trades["profit_loss"], errors="coerce")
    else:
        trades["profit_loss"] = np.nan

    if "opened_at" in trades.columns:
        trades["opened_at"] = pd.to_datetime(trades["opened_at"], utc=True, errors="coerce")
    else:
        trades["opened_at"] = pd.NaT

    if "position_side" not in trades.columns:
        trades["position_side"] = trades.get("side", "")

    group_time = trades["opened_at"].where(trades["opened_at"].notna(), trades["closed_at"])
    trades["_group_time"] = pd.to_datetime(group_time, utc=True, errors="coerce")

    if "net_size" in trades.columns:
        trades["_abs_size"] = pd.to_numeric(trades["net_size"], errors="coerce").abs()
    else:
        trades["_abs_size"] = np.nan
    if "profit_loss_pct" in trades.columns:
        trades["_pl_pct"] = pd.to_numeric(trades["profit_loss_pct"], errors="coerce")
    else:
        trades["_pl_pct"] = np.nan
    trades["_pl_pct_weighted"] = trades["_pl_pct"] * trades["_abs_size"]

    grouped = trades.groupby(["product_id", "position_side", "_group_time"], dropna=False)
    aggregated = grouped.agg(
        closed_at=("closed_at", "max"),
        opened_at=("opened_at", "min"),
        profit_loss=("profit_loss", "sum"),
        abs_size=("_abs_size", lambda s: s.sum(min_count=1)),
        pl_pct_weight=("_pl_pct_weighted", lambda s: s.sum(min_count=1)),
        has_non_partial=("closure_reason", lambda s: (s.astype(str).str.lower() != "partial_take").any()),
    ).reset_index()

    aggregated["profit_loss_pct"] = np.where(
        (aggregated["abs_size"].notna()) & (aggregated["abs_size"] > 0),
        aggregated["pl_pct_weight"] / aggregated["abs_size"],
        np.nan,
    )
    aggregated = aggregated.drop(columns=["abs_size", "pl_pct_weight"])
    if include_partial_only:
        aggregated["partial_only"] = ~aggregated["has_non_partial"]
        aggregated = aggregated.drop(columns=["has_non_partial"])
    else:
        aggregated = aggregated[aggregated["has_non_partial"]].drop(columns=["has_non_partial"])
    aggregated = aggregated.sort_values("closed_at").reset_index(drop=True)
    return aggregated


def filter_paper_trade_batches(df: pd.DataFrame, include_partials: bool = False) -> pd.DataFrame:
    if df.empty or "opened_at" not in df.columns or "closed_at" not in df.columns:
        return df

    trades = df.copy()
    trades["opened_at"] = pd.to_datetime(trades["opened_at"], utc=True, errors="coerce")
    trades["closed_at"] = pd.to_datetime(trades["closed_at"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["opened_at", "closed_at"])
    if trades.empty:
        return df

    trades["_closed_date"] = trades["closed_at"].dt.date
    trades["_opened_batch"] = trades["opened_at"].dt.floor("15min")

    counts = (
        trades.groupby(["_closed_date", "_opened_batch"]).size().reset_index(name="count")
    )
    if counts.empty:
        return trades.drop(columns=["_closed_date", "_opened_batch"])

    counts = counts.sort_values(["_closed_date", "count", "_opened_batch"])
    dominant = counts.groupby("_closed_date").tail(1)
    keep_keys = set(zip(dominant["_closed_date"], dominant["_opened_batch"]))

    mask = trades.apply(lambda row: (row["_closed_date"], row["_opened_batch"]) in keep_keys, axis=1)
    if include_partials:
        if "partial_only" in trades.columns:
            partial_mask = trades["partial_only"].fillna(False)
        elif "closure_reason" in trades.columns:
            partial_mask = trades["closure_reason"].astype(str).str.lower().eq("partial_take")
        else:
            partial_mask = False
        if not isinstance(partial_mask, pd.Series):
            partial_mask = pd.Series([False] * len(trades), index=trades.index)
        mask = mask | partial_mask
    filtered = trades.loc[mask].drop(columns=["_closed_date", "_opened_batch"])
    return filtered.sort_values("closed_at").reset_index(drop=True)


def load_range_break_status() -> Optional[dict]:
    if not RANGE_BREAK_STATUS_PATH.exists():
        return None
    try:
        data = json.loads(RANGE_BREAK_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _load_live_snapshot() -> Optional[dict]:
    if not LIVE_SNAPSHOT_PATH.exists():
        return None
    try:
        data = json.loads(LIVE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _save_live_snapshot(df: pd.DataFrame, total_unrealized: float, usdc_balance: Optional[float]) -> None:
    try:
        LIVE_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        if df.empty:
            positions = []
        else:
            df_copy = df.copy()
            if "opened_at" in df_copy.columns:
                df_copy["opened_at"] = df_copy["opened_at"].apply(
                    lambda v: v.isoformat() if isinstance(v, datetime) else (str(v) if v is not None else None)
                )
            positions = df_copy.to_dict(orient="records")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "positions": positions,
            "total_unrealized": total_unrealized,
            "usdc_balance": usdc_balance,
        }
        LIVE_SNAPSHOT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        return


def _prepare_open_positions_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "opened_at" in df.columns:
        df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce", utc=True)
    numeric_cols = [
        "net_size",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
        "notional",
        "hours_open",
        "time_left_hours",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hours_open" in df.columns:
        df["hours_open"] = df["hours_open"].round(2)
        df["hours_open_display"] = df["hours_open"].apply(_format_hours_minutes)
    if "time_left_hours" in df.columns:
        df["time_left_display"] = df["time_left_hours"].apply(_format_hours_minutes)
    if "notional" in df.columns:
        df = df.sort_values("notional", ascending=False)
    return df


def _load_dashboard_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _save_dashboard_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state))
    except Exception:
        return


def _parse_date_value(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _serialize_date_value(value: Optional[date]) -> Optional[str]:
    if not value:
        return None
    return value.isoformat()


def _format_hours_minutes(hours_val: Optional[float]) -> str:
    if hours_val is None:
        return "n/a"
    try:
        hours_float = float(hours_val)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(hours_float):
        return "n/a"
    total_minutes = int(abs(hours_float) * 60)
    days, rem = divmod(total_minutes, 60 * 24)
    hours, minutes = divmod(rem, 60)
    prefix = "-" if hours_val < 0 else ""
    if days > 0:
        return f"{prefix}{days}d {hours:02d}h"
    return f"{prefix}{hours:02d}h {minutes:02d}m"


def _time_until_next_utc_midnight() -> str:
    now = datetime.now(UTC)
    next_midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    remaining = next_midnight - now
    seconds = max(0.0, remaining.total_seconds())
    return _format_hours_minutes(seconds / 3600.0)


def _render_status_box(
    container: st.delta_generator.DeltaGenerator,
    title: str,
    details: Optional[str],
    tone: str,
    margin_left: float = 0.0,
    margin_right: float = 0.0,
    details_min_height_em: float = 0.0,
) -> None:
    palette = {
        "ok": {
            "bg": "#e7f6e7",
            "border": "#c7eac7",
            "text": "#1b5e20",
        },
        "bad": {
            "bg": "#fdecea",
            "border": "#f5c6cb",
            "text": "#7a1f1f",
        },
        "warn": {
            "bg": "#fff4ce",
            "border": "#ffe8a1",
            "text": "#8a6d3b",
        },
    }
    style = palette.get(tone, palette["warn"])
    safe_title = html.escape(title)
    details_html = ""
    if details:
        safe_details = html.escape(details).replace("\n", "<br>")
        min_height_style = (
            f" min-height:{details_min_height_em:.2f}em;" if details_min_height_em else ""
        )
        details_html = (
            f"<div style=\"font-size:0.85em; margin-top:0.35rem; color:{style['text']};"
            f" line-height:1.2;{min_height_style}\">"
            f"{safe_details}</div>"
        )
    margin_total = margin_left + margin_right
    width_style = f" width: calc(100% - {margin_total}rem);" if margin_total else ""
    margin_style = ""
    if margin_left:
        margin_style += f" margin-left:{margin_left}rem;"
    if margin_right:
        margin_style += f" margin-right:{margin_right}rem;"
    container.markdown(
        "<div style=\""
        f"background:{style['bg']}; border:1px solid {style['border']}; "
        f"border-radius:0.5rem; padding:0.6rem 0.9rem;{margin_style}{width_style}\">"
        f"<div style=\"font-weight:600; color:{style['text']};\">{safe_title}</div>"
        f"{details_html}</div>",
        unsafe_allow_html=True,
    )


def _format_daily_stop_detail(
    total_pnl_today: float,
    daily_stop_threshold: float,
    closed_txt: str,
    open_txt: str,
    pct_threshold: Optional[float],
    usd_threshold: Optional[float],
    reset_in: Optional[str] = None,
) -> str:
    pct_val = f"{DAILY_STOP_PCT:.0f}%" if DAILY_STOP_PCT > 0 else "n/a"
    usd_val = f"{usd_threshold:.0f}" if usd_threshold is not None else "n/a"
    detail = (
        f"{total_pnl_today:+.2f} vs -{daily_stop_threshold:.2f} | "
        f"closed {closed_txt} open {open_txt} | pct/usd {pct_val}/{usd_val}"
    )
    if reset_in:
        detail += f" | reset {reset_in}"
    return detail


def _collect_log_heartbeats() -> list[dict[str, object]]:
    now = datetime.now(UTC)
    results: list[dict[str, object]] = []
    for label, path, expected_seconds in LOG_HEARTBEATS:
        if not path.exists():
            results.append(
                {
                    "label": label,
                    "path": path,
                    "status": "missing",
                    "age_seconds": math.inf,
                    "expected_seconds": expected_seconds,
                    "stale": True,
                }
            )
            continue
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except (OSError, ValueError):
            results.append(
                {
                    "label": label,
                    "path": path,
                    "status": "unknown",
                    "age_seconds": math.inf,
                    "expected_seconds": expected_seconds,
                    "stale": True,
                }
            )
            continue
        age_seconds = (now - mtime).total_seconds()
        stale = age_seconds > expected_seconds * LOG_HEARTBEAT_STALE_MULTIPLIER
        results.append(
            {
                "label": label,
                "path": path,
                "status": "stale" if stale else "ok",
                "age_seconds": age_seconds,
                "expected_seconds": expected_seconds,
                "stale": stale,
            }
        )
    return results


def _format_heartbeat_line(heartbeats: list[dict[str, object]]) -> Optional[str]:
    if not heartbeats:
        return None
    parts = []
    for hb in heartbeats:
        label = str(hb.get("label"))
        age_seconds = hb.get("age_seconds", math.inf)
        if isinstance(age_seconds, (int, float)) and math.isfinite(age_seconds):
            age_display = _format_hours_minutes(age_seconds / 3600.0)
        else:
            age_display = "n/a"
        status = str(hb.get("status", ""))
        suffix = f" ({status})" if status not in {"ok", ""} else ""
        parts.append(f"{label} {age_display}{suffix}")
    return f"Log heartbeats: {' | '.join(parts)}"


def load_open_positions() -> Tuple[pd.DataFrame, float]:
    if not API_KEY_PERPS or not API_SECRET_PERPS:
        return pd.DataFrame(), 0.0

    try:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    except Exception:
        return pd.DataFrame(), 0.0

    portfolio_uuid = _get_portfolio_uuid(cb)
    if not portfolio_uuid:
        return pd.DataFrame(), 0.0

    try:
        positions_response = cb.client.list_perps_positions(portfolio_uuid=portfolio_uuid)
    except Exception:
        return pd.DataFrame(), 0.0

    positions_raw = []
    if isinstance(positions_response, dict):
        positions_raw = positions_response.get("positions", []) or []
    else:
        positions_raw = getattr(positions_response, "positions", []) or []

    def _from_currency(container: dict, default: float = 0.0) -> float:
        if not container:
            return default
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

    summary_total = 0.0
    summary_obj = positions_response.get("summary") if isinstance(positions_response, dict) else getattr(positions_response, "summary", None)
    if isinstance(summary_obj, dict):
        summary_total = _from_currency(summary_obj.get("aggregated_pnl"))
    elif summary_obj is not None:
        summary_total = _from_currency(getattr(summary_obj, "aggregated_pnl", None))

    rows = []
    for pos in positions_raw:
        pos_dict = pos if isinstance(pos, dict) else pos.to_dict()
        symbol = pos_dict.get("symbol") or pos_dict.get("product_id")
        try:
            size = float(pos_dict.get("net_size", 0) or 0)
        except Exception:
            size = 0.0
        side_raw = (pos_dict.get("position_side") or "").upper()
        side_label = "SHORT" if "SHORT" in side_raw else "LONG"
        if side_label == "SHORT":
            size = -abs(size)
        else:
            size = abs(size)

        if not symbol or abs(size) <= 0:
            continue

        entry = _from_currency(pos_dict.get("entry_vwap")) or _from_currency(pos_dict.get("vwap"))
        mark = _from_currency(pos_dict.get("mark_price"), entry)
        pnl = _from_currency(pos_dict.get("aggregated_pnl"))
        if pnl == 0.0:
            pnl = _from_currency(pos_dict.get("unrealized_pnl"))
        if entry and mark and size:
            computed_pnl = (mark - entry) * size
            if abs(computed_pnl - pnl) > max(1.0, abs(pnl) * 0.1):
                pnl = computed_pnl

        notional = _from_currency(pos_dict.get("position_notional"))
        leverage = pos_dict.get("leverage", "")

        query_size = abs(size)
        all_orders: list[dict] = []
        cursor = None
        while True:
            try:
                orders_response = cb.client.list_orders(
                    portfolio_uuid=portfolio_uuid,
                    product_id=symbol,
                    order_status="FILLED",
                    limit=200,
                    cursor=cursor,
                )
            except Exception:
                orders_response = None

            if isinstance(orders_response, dict):
                orders_raw = orders_response.get("orders", []) or []
                has_next = bool(orders_response.get("has_next"))
                cursor = orders_response.get("cursor")
            else:
                orders_raw = getattr(orders_response, "orders", []) or []
                has_next = bool(getattr(orders_response, "has_next", False))
                cursor = getattr(orders_response, "cursor", None)

            for raw_order in orders_raw:
                if isinstance(raw_order, dict):
                    all_orders.append(raw_order)
                else:
                    order_dict = getattr(raw_order, "to_dict", None)
                    all_orders.append(order_dict() if callable(order_dict) else {})

            if not has_next:
                break

        def _parse_order_time(raw_time: Optional[str]) -> Optional[datetime]:
            if not raw_time:
                return None
            try:
                return datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
            except Exception:
                return None

        def _order_time_dt(order: dict) -> Optional[datetime]:
            return _parse_order_time(order.get("created_time") or order.get("completion_time"))

        def _order_signed_size(order: dict) -> float:
            side_order = (order.get("side") or "").upper()
            try:
                filled = float(order.get("filled_size") or 0)
            except Exception:
                return 0.0
            if side_order == "BUY":
                return filled
            if side_order == "SELL":
                return -filled
            return 0.0

        entry_time = None
        target = abs(size)
        tolerance = max(1e-6, target * 1e-6)
        orders_sorted = sorted(all_orders, key=lambda o: _order_time_dt(o) or datetime.min.replace(tzinfo=UTC), reverse=True)
        running = 0.0
        for order in orders_sorted:
            dt = _order_time_dt(order)
            if not dt:
                continue
            signed = _order_signed_size(order)
            if signed == 0:
                continue
            running += signed
            entry_time = dt
            if abs(running) >= (target - tolerance):
                break

        opened_dt = None
        if entry_time:
            try:
                opened_dt = datetime.fromisoformat(str(entry_time).replace("Z", "+00:00"))
            except Exception:
                opened_dt = None

        hours_open = (
            (datetime.now(UTC) - opened_dt).total_seconds() / 3600.0
            if opened_dt is not None
            else None
        )

        expiry_hours = 24.0
        time_left_hours = (expiry_hours - hours_open) if hours_open is not None else None

        rows.append(
            {
                "product_id": symbol,
                "side": side_label,
                "net_size": size,
                "entry_price": entry,
                "mark_price": mark,
                "unrealized_pnl": pnl,
                "notional": notional if notional else abs(entry * size),
                "hours_open": hours_open,
                "time_left_hours": time_left_hours,
                "opened_at": opened_dt,
                "leverage": leverage,
            }
        )

    if not rows:
        return pd.DataFrame(), summary_total

    df = _prepare_open_positions_df(pd.DataFrame(rows))
    total_unrealized = summary_total if summary_total or summary_total == 0.0 else float(df["unrealized_pnl"].sum())
    return df, total_unrealized


def load_perp_usdc_balance() -> Optional[float]:
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
        portfolio = cb.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
    except Exception:
        return None

    breakdown = None
    if isinstance(portfolio, dict):
        breakdown = portfolio.get("breakdown")
    else:
        breakdown = getattr(portfolio, "breakdown", None)
    if breakdown is None:
        breakdown = portfolio

    if not isinstance(breakdown, dict):
        if hasattr(breakdown, "__dict__"):
            breakdown = vars(breakdown)
        else:
            return None

    balances = breakdown.get("portfolio_balances")
    if balances is None and isinstance(breakdown.get("breakdown"), dict):
        balances = breakdown["breakdown"].get("portfolio_balances")
    if balances is None:
        return None

    if not isinstance(balances, dict):
        if hasattr(balances, "__dict__"):
            balances = vars(balances)
        else:
            return None

    def _value_from(container: dict, key: str) -> Optional[float]:
        entry = container.get(key)
        if not entry:
            return None
        if isinstance(entry, dict):
            value = entry.get("value")
        else:
            value = getattr(entry, "value", None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    available = _value_from(balances, "available_balance")
    if available is not None:
        return available
    return _value_from(balances, "total_balance")


def load_paper_open_positions(path: Path) -> Tuple[pd.DataFrame, float]:
    if not path.exists():
        return pd.DataFrame(), 0.0
    df = pd.read_csv(path)
    date_cols = ["opened_at", "expires_at", "last_price_at"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    numeric_cols = [
        "entry_price",
        "last_price",
        "take_profit",
        "stop_loss",
        "position_usd",
        "unrealized_pnl",
        "unrealized_pct",
        "finder_score",
        "finder_rank",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    total_unrealized = float(df["unrealized_pnl"].sum()) if "unrealized_pnl" in df.columns else 0.0
    return df, total_unrealized


def apply_filters(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    start_count: int,
    end_count: int,
    last: int,
    symbols: Optional[list[str]] = None,
) -> pd.DataFrame:
    filtered = filter_by_date(df, start, end)
    if filtered.empty:
        return filtered

    if symbols:
        filtered = filtered[filtered["product_id"].isin(symbols)]
        if filtered.empty:
            return filtered

    filtered = select_count_window(
        filtered,
        start_count=start_count,
        end_count=end_count,
        ordering_col="closed_at",
    )
    filtered = select_last(filtered, last, ordering_col="closed_at")
    return filtered


def build_daily_equity(
    trades: pd.DataFrame,
    starting_equity: float,
    *,
    metrics_trades: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    pnl_trades = trades.sort_values("closed_at")
    pnl_trades["date"] = pnl_trades["closed_at"].dt.date

    metrics_source = metrics_trades if metrics_trades is not None else pnl_trades
    metrics_source = metrics_source.sort_values("closed_at")
    metrics_source["date"] = metrics_source["closed_at"].dt.date

    daily_pnl = pnl_trades.groupby("date")["profit_loss"].sum(min_count=1).fillna(0.0)
    trade_counts = metrics_source.groupby("date").size()
    daily = pd.DataFrame({"daily_pnl": daily_pnl, "trades": trade_counts})
    daily["trades"] = daily["trades"].fillna(0).astype(int)
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()
    daily["equity"] = starting_equity + daily["cum_pnl"]

    # Anchor a baseline point so the curve starts at the starting equity.
    if not daily.empty:
        first_date = daily.index.min()
        baseline_date = first_date - pd.Timedelta(days=1)
        baseline_row = pd.DataFrame(
            {
                "daily_pnl": [0.0],
                "trades": [0],
                "cum_pnl": [0.0],
                "equity": [starting_equity],
            },
            index=[baseline_date],
        )
        daily = pd.concat([baseline_row, daily]).sort_index()
        daily.index.name = "date"

    prev_equity = np.concatenate(([starting_equity], daily["equity"].to_numpy()[:-1]))
    prev_equity = np.where(prev_equity == 0, np.nan, prev_equity)
    daily_return_pct = np.divide(daily["daily_pnl"], prev_equity, where=~np.isnan(prev_equity)) * 100.0
    daily["daily_return_pct"] = np.nan_to_num(daily_return_pct)

    rolling_peak = daily["equity"].cummax()
    drawdown = daily["equity"] - rolling_peak
    drawdown_values = drawdown.to_numpy()
    peak_values = rolling_peak.replace(0, np.nan).to_numpy()
    drawdown_pct = np.divide(drawdown_values, peak_values, out=np.zeros_like(drawdown_values), where=~np.isnan(peak_values)) * 100.0
    drawdown_pct = np.nan_to_num(drawdown_pct)
    daily["drawdown"] = drawdown_values
    daily["drawdown_pct"] = drawdown_pct

    profit_loss_series = metrics_source["profit_loss"]
    valid_profit_loss = profit_loss_series.dropna()
    valid_trade_count = len(valid_profit_loss)

    gross_profit = valid_profit_loss[valid_profit_loss > 0].sum()
    gross_loss = valid_profit_loss[valid_profit_loss < 0].sum()
    gross_loss_abs = abs(gross_loss)
    if valid_trade_count == 0:
        profit_factor = float("nan")
    elif gross_loss_abs > 0:
        profit_factor = gross_profit / gross_loss_abs
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    if "profit_loss_pct" in metrics_source.columns:
        profit_loss_pct_series = pd.to_numeric(metrics_source["profit_loss_pct"], errors="coerce").dropna()
    else:
        profit_loss_pct_series = pd.Series(dtype=float)
    if not profit_loss_pct_series.empty:
        nonzero_profit_loss_pct = profit_loss_pct_series[profit_loss_pct_series != 0]
        if not nonzero_profit_loss_pct.empty:
            median_profit_loss_pct = float(nonzero_profit_loss_pct.median())
        else:
            median_profit_loss_pct = float(profit_loss_pct_series.median())
        avg_pct = float(profit_loss_pct_series.mean())
        wins_pct = profit_loss_pct_series[profit_loss_pct_series > 0]
        losses_pct = profit_loss_pct_series[profit_loss_pct_series < 0]
        avg_win_pct = float(wins_pct.mean()) if not wins_pct.empty else float("nan")
        avg_loss_pct = float(losses_pct.mean()) if not losses_pct.empty else float("nan")
        win_rate_pct_series = len(wins_pct) / len(profit_loss_pct_series)
        avg_loss_abs = abs(avg_loss_pct) if not math.isnan(avg_loss_pct) else 0.0
        expectancy_pct = win_rate_pct_series * (avg_win_pct if not math.isnan(avg_win_pct) else 0.0) - (
            1 - win_rate_pct_series
        ) * avg_loss_abs
    else:
        median_profit_loss_pct = float("nan")
        avg_pct = float("nan")
        avg_win_pct = float("nan")
        avg_loss_pct = float("nan")
        expectancy_pct = float("nan")

    wins = int((valid_profit_loss > 0).sum())
    losses = int((valid_profit_loss < 0).sum())
    breakevens = int((valid_profit_loss == 0).sum())
    win_rate_pct = float(wins / valid_trade_count * 100.0) if valid_trade_count else 0.0
    expectancy = float(valid_profit_loss.mean()) if valid_trade_count else 0.0
    nonzero_profit_loss = valid_profit_loss[valid_profit_loss != 0]
    if not nonzero_profit_loss.empty:
        median_profit_loss = float(nonzero_profit_loss.median())
    elif not valid_profit_loss.empty:
        median_profit_loss = float(valid_profit_loss.median())
    else:
        median_profit_loss = float("nan")
    std_dev_trade = float(valid_profit_loss.std(ddof=0)) if valid_trade_count else float("nan")

    recent_trade_window = min(20, valid_trade_count)
    if recent_trade_window > 0:
        recent_profit_losses = valid_profit_loss.tail(recent_trade_window)
        recent_trade_expectancy = float(recent_profit_losses.mean())
        recent_trade_win_rate_pct = float((recent_profit_losses > 0).mean() * 100.0)
    else:
        recent_trade_expectancy = float("nan")
        recent_trade_win_rate_pct = float("nan")

    recent_expectancy_delta = (
        recent_trade_expectancy - expectancy
        if recent_trade_window > 0 and valid_trade_count > 0
        else float("nan")
    )
    recent_win_rate_delta = (
        recent_trade_win_rate_pct - win_rate_pct
        if recent_trade_window > 0 and valid_trade_count > 0
        else float("nan")
    )

    ending_equity = float(daily["equity"].iloc[-1]) if not daily.empty else starting_equity
    total_return_pct = ((ending_equity - starting_equity) / starting_equity * 100.0) if starting_equity else 0.0

    metrics = {
        "trades": int(len(metrics_source)),
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate_pct": win_rate_pct,
        "expectancy": expectancy,
        "expectancy_pct": expectancy_pct,
        "profit_factor": float(profit_factor),
        "median_profit_loss": median_profit_loss,
        "median_profit_loss_pct": median_profit_loss_pct,
        "avg_pct": avg_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "metric_glossary": {
            "profit_loss": "USD P&L per trade (drives equity curve).",
            "profit_loss_pct": "% P&L per trade (trade quality).",
            "expectancy": "Average profit_loss (USD).",
            "expectancy_pct": "Average profit_loss_pct (%).",
            "profit_factor": "Total wins ÷ total losses (USD).",
            "win_rate_pct": "Percent of trades with positive profit_loss.",
            "avg_pct": "Average profit_loss_pct across trades.",
            "avg_win_pct": "Average profit_loss_pct for winning trades.",
            "avg_loss_pct": "Average profit_loss_pct for losing trades.",
            "median_profit_loss": "Median USD P&L per trade (less skewed by outliers).",
            "median_profit_loss_pct": "Median % P&L per trade.",
            "max_drawdown": "Worst peak-to-trough equity drop (USD).",
            "max_drawdown_pct": "Worst peak-to-trough equity drop (%).",
            "sharpe_ratio": "Risk-adjusted return of daily equity (higher is better; noisy on small samples).",
            "total_return_pct": "Total return from starting to ending equity (%).",
        },
        "best_day": float(daily["daily_pnl"].max()) if not daily.empty else 0.0,
        "worst_day": float(daily["daily_pnl"].min()) if not daily.empty else 0.0,
        "ending_equity": ending_equity,
        "starting_equity": float(starting_equity),
        "total_return_pct": float(total_return_pct),
        "max_drawdown": float(drawdown.min()) if not daily.empty else 0.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()) if not daily.empty else 0.0,
        "std_dev_trade": std_dev_trade,
        "std_dev_drawdown": float(drawdown.std(ddof=0)) if len(daily) else 0.0,
        "recent_trade_window": int(recent_trade_window),
        "recent_trade_expectancy": float(recent_trade_expectancy),
        "recent_trade_expectancy_delta": float(recent_expectancy_delta),
        "recent_trade_win_rate_pct": float(recent_trade_win_rate_pct),
        "recent_trade_win_rate_delta": float(recent_win_rate_delta),
    }

    daily_returns = daily["daily_return_pct"] / 100.0
    if len(daily_returns) >= 2:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std(ddof=1)
        sharpe = (mean_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0.0
    elif len(daily_returns) == 1:
        sharpe = float("inf") if daily_returns.iloc[0] > 0 else 0.0
    else:
        sharpe = 0.0
    metrics["sharpe_ratio"] = float(sharpe)

    lookback_days = min(7, len(daily_returns))
    if lookback_days >= 2:
        recent_returns = daily_returns.tail(lookback_days)
        recent_mean = recent_returns.mean()
        recent_std = recent_returns.std(ddof=1)
        if recent_std > 0:
            recent_sharpe = (recent_mean / recent_std) * math.sqrt(365)
        else:
            recent_sharpe = float("inf") if recent_mean > 0 else 0.0
    elif lookback_days == 1:
        val = daily_returns.tail(1).iloc[0]
        recent_sharpe = float("inf") if val > 0 else 0.0
    else:
        recent_sharpe = float("nan")

    metrics["recent_sharpe_ratio"] = float(recent_sharpe)
    if math.isfinite(metrics["sharpe_ratio"]) and math.isfinite(recent_sharpe):
        metrics["recent_sharpe_delta"] = float(recent_sharpe - metrics["sharpe_ratio"])
    else:
        metrics["recent_sharpe_delta"] = float("nan")

    degradation_reasons = []
    recent_window = metrics["recent_trade_window"]
    recent_expectancy = metrics["recent_trade_expectancy"]
    overall_expectancy = metrics["expectancy"]
    if recent_window >= 5 and not math.isnan(recent_expectancy):
        threshold_exp = max(abs(overall_expectancy) * 0.25, 0.1)
        if overall_expectancy > 0 and recent_expectancy < 0:
            degradation_reasons.append("Recent expectancy turned negative while overall expectancy is positive.")
        elif recent_expectancy < overall_expectancy - threshold_exp:
            diff = overall_expectancy - recent_expectancy
            degradation_reasons.append(f"Recent expectancy is {diff:.2f} below overall.")

    recent_win_rate = metrics["recent_trade_win_rate_pct"]
    overall_win_rate = metrics["win_rate_pct"]
    if recent_window >= 5 and not math.isnan(recent_win_rate):
        if recent_win_rate < overall_win_rate - 10:
            diff = overall_win_rate - recent_win_rate
            degradation_reasons.append(f"Recent win rate dropped by {diff:.1f} percentage points.")

    overall_sharpe = metrics["sharpe_ratio"]
    if math.isfinite(recent_sharpe) and math.isfinite(overall_sharpe):
        if overall_sharpe > 0 and recent_sharpe <= 0:
            degradation_reasons.append("Recent Sharpe fell to zero or negative while overall Sharpe is positive.")
        elif recent_sharpe < overall_sharpe - 0.5:
            diff = overall_sharpe - recent_sharpe
            degradation_reasons.append(f"Recent Sharpe is {diff:.2f} below the overall level.")

    metrics["recent_degradation_flag"] = bool(degradation_reasons)
    metrics["recent_degradation_reasons"] = degradation_reasons

    if daily.index.name is None:
        daily.index.name = "date"
    daily_reset = daily.reset_index()
    return daily_reset, metrics


def main() -> None:
    st.set_page_config(page_title="Watchdog Daily Equity", layout="wide")
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] > div:first-child {
                padding-top: 0.5rem;
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
            }

            h1:first-child {
                margin-top: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Watchdog Daily Equity Dashboard")

    def _normalize_source_choice(raw: object) -> str | None:
        if raw is None:
            return None
        value = str(raw).strip().lower()
        if value in {"live", "live watchdog"}:
            return "Live Watchdog"
        if value in {"paper", "paper finder"}:
            return "Paper Finder"
        return None

    def _get_source_from_query() -> str | None:
        try:
            params = st.query_params
            if "source" in params:
                raw = params.get("source")
                if isinstance(raw, list):
                    raw = raw[0] if raw else None
                return _normalize_source_choice(raw)
        except Exception:
            params = st.experimental_get_query_params()
            raw = params.get("source")
            if isinstance(raw, list):
                raw = raw[0] if raw else None
            return _normalize_source_choice(raw)
        return None

    def _set_source_query(choice: str) -> None:
        value = "paper" if choice == "Paper Finder" else "live"
        try:
            st.query_params["source"] = value
        except Exception:
            st.experimental_set_query_params(source=value)

    source_from_query = _get_source_from_query()
    if "log_source" not in st.session_state:
        st.session_state["log_source"] = source_from_query or "Live Watchdog"

    def _persist_log_source() -> None:
        _set_source_query(st.session_state["log_source"])

    source_choice = st.sidebar.selectbox(
        "Log source",
        options=("Live Watchdog", "Paper Finder"),
        index=("Live Watchdog", "Paper Finder").index(st.session_state["log_source"]),
        key="log_source",
        on_change=_persist_log_source,
    )
    source_mode = "paper" if source_choice == "Paper Finder" else "live"
    if "dashboard_state" not in st.session_state:
        st.session_state["dashboard_state"] = _load_dashboard_state()

    refresh_defaults = {}
    if isinstance(st.session_state.get("dashboard_state"), dict):
        refresh_defaults = st.session_state["dashboard_state"].get("refresh", {}) or {}
    auto_refresh_default = bool(refresh_defaults.get("enabled", False))
    refresh_interval_default = int(refresh_defaults.get("interval_sec", AUTO_REFRESH_DEFAULT_SEC))
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=auto_refresh_default)
    st.sidebar.caption("Runs a Streamlit rerun (same as pressing “R”); does not reload the browser page.")
    refresh_interval = st.sidebar.number_input(
        "Refresh interval (sec)",
        min_value=30,
        max_value=600,
        value=refresh_interval_default,
        step=30,
        disabled=not auto_refresh,
    )
    if auto_refresh:
        if st_autorefresh is None:
            st.sidebar.warning("Install streamlit-autorefresh to enable auto-refresh without reloading the page.")
        else:
            st_autorefresh(interval=int(refresh_interval * 1000), key="watchdog_autorefresh")
    if "paper_update_log" not in st.session_state:
        st.session_state["paper_update_log"] = ""
    if "live_update_log" not in st.session_state:
        st.session_state["live_update_log"] = ""
    if "live_positions_df" not in st.session_state:
        st.session_state["live_positions_df"] = pd.DataFrame()
    if "live_total_unrealized" not in st.session_state:
        st.session_state["live_total_unrealized"] = 0.0
    if "live_usdc_balance" not in st.session_state:
        st.session_state["live_usdc_balance"] = None
    if "live_snapshot_ts" not in st.session_state:
        st.session_state["live_snapshot_ts"] = ""
    if "live_fills_cache" not in st.session_state:
        st.session_state["live_fills_cache"] = None
    if "live_fills_loaded_at" not in st.session_state:
        st.session_state["live_fills_loaded_at"] = ""
    if "live_slippage_df" not in st.session_state:
        st.session_state["live_slippage_df"] = None
    if source_mode == "live":
        snapshot = _load_live_snapshot()
        if snapshot:
            snap_ts = str(snapshot.get("timestamp") or "")
            cached_df = pd.DataFrame(snapshot.get("positions", []))
            st.session_state["live_positions_df"] = _prepare_open_positions_df(cached_df)
            try:
                st.session_state["live_total_unrealized"] = float(snapshot.get("total_unrealized", 0.0))
            except (TypeError, ValueError):
                st.session_state["live_total_unrealized"] = 0.0
            st.session_state["live_usdc_balance"] = snapshot.get("usdc_balance")
            if snap_ts:
                st.session_state["live_snapshot_ts"] = snap_ts
    if source_mode == "paper":
        if st.sidebar.button("Update paper trades"):
            with st.spinner("Updating paper trades..."):
                conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
                if conda_exe:
                    cmd = [
                        conda_exe,
                        "run",
                        "-n",
                        "trade",
                        "python",
                        str(REPO_ROOT / "paper_finder_simulator.py"),
                        "update",
                    ]
                else:
                    cmd = [sys.executable, str(REPO_ROOT / "paper_finder_simulator.py"), "update"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=REPO_ROOT,
                )
            output = (result.stdout or "") + (result.stderr or "")
            st.session_state["paper_update_log"] = output.strip()
            if result.returncode != 0:
                st.sidebar.error("Paper update failed.")
            else:
                st.sidebar.success("Paper trades updated.")
        if st.session_state["paper_update_log"]:
            st.sidebar.code(st.session_state["paper_update_log"])
    else:
        if st.sidebar.button("Update live data"):
            with st.spinner("Updating live fills and positions..."):
                conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
                if conda_exe:
                    cmd = [
                        conda_exe,
                        "run",
                        "-n",
                        "trade",
                        "python",
                        str(REPO_ROOT / "watchdog_close_old_positions.py"),
                        "--log-fills",
                        "--skip-close",
                        "--fills-limit",
                        "500",
                    ]
                else:
                    cmd = [
                        sys.executable,
                        str(REPO_ROOT / "watchdog_close_old_positions.py"),
                        "--log-fills",
                        "--skip-close",
                        "--fills-limit",
                        "500",
                    ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    cwd=REPO_ROOT,
                )
            output = (result.stdout or "") + (result.stderr or "")
            st.session_state["live_update_log"] = output.strip()
            if result.returncode != 0:
                st.sidebar.error("Live fills update failed.")
            else:
                st.sidebar.success("Live fills updated.")
            live_df, live_unreal = load_open_positions()
            st.session_state["live_positions_df"] = _prepare_open_positions_df(live_df)
            st.session_state["live_total_unrealized"] = live_unreal
            st.session_state["live_usdc_balance"] = load_perp_usdc_balance()
            st.session_state["live_snapshot_ts"] = datetime.now(timezone.utc).isoformat()
            _save_live_snapshot(
                st.session_state["live_positions_df"],
                st.session_state["live_total_unrealized"],
                st.session_state["live_usdc_balance"],
            )
        if st.session_state["live_update_log"]:
            st.sidebar.code(st.session_state["live_update_log"])
    default_csv = PAPER_CLOSED_CSV if source_mode == "paper" else Path("trade_logs/watchdog_closed_positions.csv")
    csv_candidates = sorted(Path("trade_logs").glob("*.csv"))
    csv_options = [str(path) for path in csv_candidates]
    default_csv_str = str(default_csv)
    if default_csv_str not in csv_options:
        csv_options.insert(0, default_csv_str)
    csv_options.append("Custom...")
    default_index = csv_options.index(default_csv_str) if default_csv_str in csv_options else 0
    selected_csv = st.sidebar.selectbox("CSV file", options=csv_options, index=default_index)
    if selected_csv == "Custom...":
        csv_path = st.sidebar.text_input("CSV path", default_csv_str).strip()
    else:
        csv_path = selected_csv
    data_path = Path(csv_path)

    today = date.today()
    persisted_filters = {}
    if isinstance(st.session_state.get("dashboard_state"), dict):
        persisted_filters = (
            st.session_state["dashboard_state"]
            .get("date_filters", {})
            .get(source_mode, {})
        )
    persisted_preset = persisted_filters.get("preset")
    if persisted_preset not in ("Custom", "Last 7 days", "Last 30 days", "Year to date"):
        persisted_preset = "Custom"
    persisted_start = _parse_date_value(persisted_filters.get("start"))
    persisted_end = _parse_date_value(persisted_filters.get("end"))

    date_preset = st.sidebar.selectbox(
        "Date preset",
        options=("Custom", "Last 7 days", "Last 30 days", "Year to date"),
        index=("Custom", "Last 7 days", "Last 30 days", "Year to date").index(persisted_preset),
        help="Quickly apply a rolling time window; choose Custom to rely on the manual date inputs below.",
    )

    start_date_input = st.sidebar.date_input("Start date", value=persisted_start)
    end_date_input = st.sidebar.date_input("End date", value=persisted_end)
    if isinstance(end_date_input, list):
        end_date_input = end_date_input[0] if end_date_input else None

    persisted_counts = {}
    if isinstance(st.session_state.get("dashboard_state"), dict):
        persisted_counts = (
            st.session_state["dashboard_state"]
            .get("count_filters", {})
            .get(source_mode, {})
        )
    start_count_default = int(persisted_counts.get("start", 0) or 0)
    end_count_default = int(persisted_counts.get("end", 0) or 0)
    tail_last_default = int(persisted_counts.get("last", 0) or 0)

    start_count = st.sidebar.number_input("Start count (1-based)", min_value=0, value=start_count_default, step=1)
    end_count = st.sidebar.number_input("End count (inclusive, 0 for none)", min_value=0, value=end_count_default, step=1)
    tail_last = st.sidebar.number_input("Last N trades (0 to ignore)", min_value=0, value=tail_last_default, step=1)
    persisted_starting = (
        st.session_state["dashboard_state"].get("starting_equity", {}).get(source_mode)
        if isinstance(st.session_state.get("dashboard_state"), dict)
        else None
    )
    starting_equity_default = (
        float(persisted_starting)
        if persisted_starting is not None
        else 1000.0
    )
    starting_equity = st.sidebar.number_input(
        "Starting equity",
        min_value=0.0,
        value=starting_equity_default,
        step=100.0,
        key=f"starting_equity_{source_mode}",
    )
    view_defaults = {}
    if isinstance(st.session_state.get("dashboard_state"), dict):
        view_defaults = (
            st.session_state["dashboard_state"]
            .get("view_options", {})
            .get(source_mode, {})
        )
    include_partials_default = bool(view_defaults.get("include_partials", False))
    include_partials = st.sidebar.checkbox(
        "Include partial exits",
        value=include_partials_default,
        help="Include partial_take-only closures in metrics/filters before a full close.",
    )
    if isinstance(st.session_state.get("dashboard_state"), dict):
        existing_filters = st.session_state["dashboard_state"].get("date_filters", {})
        updated_filters = {
            "preset": date_preset,
            "start": _serialize_date_value(start_date_input),
            "end": _serialize_date_value(end_date_input),
        }
        if existing_filters.get(source_mode) != updated_filters:
            existing_filters[source_mode] = updated_filters
            st.session_state["dashboard_state"]["date_filters"] = existing_filters
            _save_dashboard_state(st.session_state["dashboard_state"])
        existing_counts = st.session_state["dashboard_state"].get("count_filters", {})
        updated_counts = {"start": int(start_count), "end": int(end_count), "last": int(tail_last)}
        if existing_counts.get(source_mode) != updated_counts:
            existing_counts[source_mode] = updated_counts
            st.session_state["dashboard_state"]["count_filters"] = existing_counts
            _save_dashboard_state(st.session_state["dashboard_state"])
        existing = st.session_state["dashboard_state"].get("starting_equity", {})
        if existing.get(source_mode) != starting_equity:
            existing[source_mode] = float(starting_equity)
            st.session_state["dashboard_state"]["starting_equity"] = existing
            _save_dashboard_state(st.session_state["dashboard_state"])
        existing_views = st.session_state["dashboard_state"].get("view_options", {})
        updated_views = {"include_partials": bool(include_partials)}
        if existing_views.get(source_mode) != updated_views:
            existing_views[source_mode] = updated_views
            st.session_state["dashboard_state"]["view_options"] = existing_views
            _save_dashboard_state(st.session_state["dashboard_state"])
        refresh_state = st.session_state["dashboard_state"].get("refresh", {})
        refresh_updated = {
            "enabled": bool(auto_refresh),
            "interval_sec": int(refresh_interval),
        }
        if refresh_state != refresh_updated:
            st.session_state["dashboard_state"]["refresh"] = refresh_updated
            _save_dashboard_state(st.session_state["dashboard_state"])

    if st.sidebar.button("Reset filters"):
        if isinstance(st.session_state.get("dashboard_state"), dict):
            existing_filters = st.session_state["dashboard_state"].get("date_filters", {})
            if source_mode in existing_filters:
                existing_filters.pop(source_mode, None)
                st.session_state["dashboard_state"]["date_filters"] = existing_filters
            existing_counts = st.session_state["dashboard_state"].get("count_filters", {})
            if source_mode in existing_counts:
                existing_counts.pop(source_mode, None)
                st.session_state["dashboard_state"]["count_filters"] = existing_counts
            _save_dashboard_state(st.session_state["dashboard_state"])
        st.experimental_rerun()

    try:
        trades_df = load_watchdog_csv(data_path)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    trade_view_df = collapse_trade_rows(trades_df, include_partial_only=include_partials)
    if source_mode == "paper":
        trade_view_df = filter_paper_trade_batches(trade_view_df, include_partials=include_partials)

    count_source_df = trades_df if source_mode == "paper" else trade_view_df
    column_map = {col.strip().lower(): col for col in count_source_df.columns}
    max_trade_count = None
    for key in ("count", "trade_count", "trade_number", "trade_index"):
        if key in column_map:
            series = pd.to_numeric(count_source_df[column_map[key]], errors='coerce')
            if series.notna().any():
                max_trade_count = int(series.max())
                break
    if max_trade_count is None:
        max_trade_count = int(count_source_df.shape[0])
    max_trade_count = max(1, max_trade_count)

    effective_start_count = start_count if start_count > 0 else 0
    effective_end_count = end_count if end_count > 0 else 0
    effective_tail_last = tail_last if tail_last > 0 else 0

    warnings = []
    if effective_start_count > max_trade_count:
        effective_start_count = 0
        warnings.append("Start count exceeds available trades and was ignored.")
    if effective_end_count > max_trade_count:
        effective_end_count = max_trade_count
        warnings.append("End count capped to available trades.")
    if effective_tail_last > max_trade_count:
        effective_tail_last = max_trade_count
        warnings.append("Last N trades capped to available trades.")

    if warnings:
        st.sidebar.caption(" ".join(warnings) + f" (max {max_trade_count})")

    try:
        pipeline_snapshot = build_snapshot(trade_view_df, include_unrealized=False)
    except Exception as exc:
        pipeline_snapshot = None
        st.sidebar.caption(f"Metrics snapshot unavailable: {exc}")

    available_products = sorted(trade_view_df["product_id"].dropna().unique())
    selected_products = st.sidebar.multiselect(
        "Products (leave empty for all)",
        options=available_products,
        default=[],
        help="Filter summary and charts to selected instruments.",
    )

    start_str = start_date_input.isoformat() if start_date_input else None
    end_str = end_date_input.isoformat() if end_date_input else None
    if date_preset != "Custom":
        if date_preset == "Last 7 days":
            preset_start = today - timedelta(days=6)
            preset_end = today + timedelta(days=1)
        elif date_preset == "Last 30 days":
            preset_start = today - timedelta(days=29)
            preset_end = today + timedelta(days=1)
        elif date_preset == "Year to date":
            preset_start = date(today.year, 1, 1)
            preset_end = today + timedelta(days=1)
        else:
            preset_start = start_date_input or today
            preset_end = (end_date_input or today) + timedelta(days=1)

        start_str = preset_start.isoformat()
        end_str = preset_end.isoformat()
    filter_start_count = int(effective_start_count)
    filter_end_count = int(effective_end_count)
    count_filters_active = any(
        value > 0
        for value in (filter_start_count, filter_end_count, int(effective_tail_last))
    )

    if source_mode == "paper":
        pnl_filtered = apply_filters(
            trades_df,
            start_str,
            end_str,
            filter_start_count,
            filter_end_count,
            int(effective_tail_last),
            symbols=selected_products or None,
        )
        filtered = collapse_trade_rows(pnl_filtered, include_partial_only=include_partials)
        filtered = filter_paper_trade_batches(filtered, include_partials=include_partials)
    else:
        filtered = apply_filters(
            trade_view_df,
            start_str,
            end_str,
            filter_start_count,
            filter_end_count,
            int(effective_tail_last),
            symbols=selected_products or None,
        )

        pnl_filtered = apply_filters(
            trades_df,
            start_str,
            end_str,
            0,
            0,
            0,
            symbols=selected_products or None,
        )

        if count_filters_active and not filtered.empty and not pnl_filtered.empty:
            filtered_keys = filtered.copy()
            filtered_keys["_group_time"] = filtered_keys["opened_at"].where(
                filtered_keys["opened_at"].notna(), filtered_keys["closed_at"]
            )
            filtered_key_set = set(
                zip(
                    filtered_keys["product_id"],
                    filtered_keys["position_side"],
                    filtered_keys["_group_time"],
                )
            )

            pnl_keys = pnl_filtered.copy()
            pnl_keys["_group_time"] = pnl_keys["opened_at"].where(
                pnl_keys["opened_at"].notna(), pnl_keys["closed_at"]
            )
            pnl_key_series = list(
                zip(
                    pnl_keys["product_id"],
                    pnl_keys["position_side"],
                    pnl_keys["_group_time"],
                )
            )
            key_match = pd.Series(pnl_key_series, index=pnl_filtered.index).isin(filtered_key_set)

            min_close = filtered["closed_at"].min()
            max_close = filtered["closed_at"].max()
            if key_match.any():
                mask = key_match
            else:
                if pd.notna(min_close) and pd.notna(max_close):
                    mask = pnl_filtered["closed_at"].between(min_close, max_close)
                else:
                    mask = key_match

            pnl_filtered = pnl_filtered.loc[mask].copy()

    if filtered.empty and pnl_filtered.empty:
        st.warning("No trades match the selected filters.")
        st.stop()

    equity_source = pnl_filtered
    if count_filters_active and not filtered.empty:
        equity_source = filtered

    daily, metrics = build_daily_equity(
        equity_source,
        float(starting_equity),
        metrics_trades=filtered,
    )

    summary_notes = ["Visualise equity, drawdowns, and daily performance derived from watchdog logs."]
    if selected_products:
        summary_notes.append(f"Products: {', '.join(selected_products)}")
    if date_preset != "Custom":
        summary_notes.append(f"Preset window: {date_preset}")
    elif start_str or end_str:
        summary_notes.append("Custom date window")
    if summary_notes:
        st.caption(" | ".join(summary_notes))

    daily_pnl_today = None
    open_pnl_today = None
    try:
        today = datetime.now(UTC).date()
        daily_pnl_today = trades_df.loc[trades_df["closed_at"].dt.date == today, "profit_loss"].sum()
        daily_pnl_today = float(daily_pnl_today)
    except Exception:
        daily_pnl_today = None
    if source_mode == "paper":
        _, open_pnl_today = load_paper_open_positions(PAPER_OPEN_CSV)
    else:
        open_pnl_today = st.session_state.get("live_total_unrealized", 0.0)
    try:
        open_pnl_today = float(open_pnl_today) if open_pnl_today is not None else None
    except (TypeError, ValueError):
        open_pnl_today = None

    total_pnl_today = None
    if daily_pnl_today is not None or open_pnl_today is not None:
        total_pnl_today = (daily_pnl_today or 0.0) + (open_pnl_today or 0.0)

    pct_threshold = float(starting_equity) * (DAILY_STOP_PCT / 100.0) if DAILY_STOP_PCT > 0 else None
    usd_threshold = DAILY_STOP_USD if DAILY_STOP_USD > 0 else None
    thresholds = [t for t in (pct_threshold, usd_threshold) if t is not None]
    daily_stop_threshold = min(thresholds) if thresholds else None

    daily_cols = st.columns([0.9, 0.02, 1.08])
    daily_cols[1].markdown("&nbsp;", unsafe_allow_html=True)
    if total_pnl_today is None or daily_stop_threshold is None:
        daily_cols[0].info("Daily stop: n/a (no closed trades yet).")
    else:
        closed_txt = f"{daily_pnl_today:+.2f}" if daily_pnl_today is not None else "n/a"
        open_txt = f"{open_pnl_today:+.2f}" if open_pnl_today is not None else "n/a"
        if total_pnl_today <= -daily_stop_threshold:
            reset_in = _time_until_next_utc_midnight()
            reason = _format_daily_stop_detail(
                total_pnl_today,
                daily_stop_threshold,
                closed_txt,
                open_txt,
                pct_threshold,
                usd_threshold,
                reset_in,
            )
            _render_status_box(
                daily_cols[0],
                "Daily stop ACTIVE",
                reason,
                tone="bad",
                margin_right=0.05,
                details_min_height_em=1.6,
            )
        else:
            reason = _format_daily_stop_detail(
                total_pnl_today,
                daily_stop_threshold,
                closed_txt,
                open_txt,
                pct_threshold,
                usd_threshold,
            )
            _render_status_box(
                daily_cols[0],
                "Daily stop OK",
                reason,
                tone="ok",
                margin_right=0.05,
                details_min_height_em=1.6,
            )
    daily_cols[0].markdown("<div style=\"margin-bottom:0.6rem;\"></div>", unsafe_allow_html=True)

    range_slot = daily_cols[2]
    range_status = load_range_break_status()
    if range_status is None:
        range_slot.info("Range break: n/a (run gate-scan).")
    else:
        symbol = str(range_status.get("symbol", "n/a"))
        try:
            days = int(range_status.get("days", 0))
        except (TypeError, ValueError):
            days = 0
        try:
            atr_mult = float(range_status.get("atr_mult", 0.0))
        except (TypeError, ValueError):
            atr_mult = 0.0
        try:
            close = float(range_status.get("close", 0.0))
        except (TypeError, ValueError):
            close = 0.0
        try:
            confirmed_close = float(range_status.get("confirmed_close", 0.0))
        except (TypeError, ValueError):
            confirmed_close = 0.0
        try:
            range_low = float(range_status.get("range_low", 0.0))
        except (TypeError, ValueError):
            range_low = 0.0
        try:
            range_high = float(range_status.get("range_high", 0.0))
        except (TypeError, ValueError):
            range_high = 0.0
        try:
            buffer = float(range_status.get("buffer", 0.0))
        except (TypeError, ValueError):
            buffer = 0.0
        direction = str(range_status.get("direction", "inside"))
        latched = bool(range_status.get("latched"))
        triggered = bool(range_status.get("triggered")) or latched
        state = "latched" if latched else "live"
        msg = (
            f"{symbol} {days}d {direction} ({state}) | close={close:.2f} confirmed={confirmed_close:.2f} "
            f"range={range_low:.2f}-{range_high:.2f} buffer={buffer:.2f} (ATRx{atr_mult:.2f})"
        )
        if triggered:
            _render_status_box(
                range_slot,
                "Range break ACTIVE",
                msg,
                tone="bad",
                margin_left=0.05,
                details_min_height_em=1.6,
            )
        else:
            _render_status_box(
                range_slot,
                "Range break OK",
                msg,
                tone="ok",
                margin_left=0.05,
                details_min_height_em=1.6,
            )
    range_slot.markdown("<div style=\"margin-bottom:0.6rem;\"></div>", unsafe_allow_html=True)

    with st.expander("Metric glossary", expanded=False):
        glossary = metrics.get("metric_glossary", {})
        if glossary:
            for key, desc in glossary.items():
                st.markdown(f"- `{key}`: {desc}")

    if source_mode == "paper":
        open_positions_df, total_unrealized = load_paper_open_positions(PAPER_OPEN_CSV)
    else:
        open_positions_df = _prepare_open_positions_df(
            st.session_state.get("live_positions_df", pd.DataFrame())
        )
        total_unrealized = st.session_state.get("live_total_unrealized", 0.0)

    if pipeline_snapshot is not None:
        if total_unrealized is not None:
            try:
                pipeline_snapshot.unrealized_pnl = float(total_unrealized)
            except (TypeError, ValueError):
                pass
        pipeline_snapshot.open_positions = int(len(open_positions_df))

        health_col1, health_col2, health_col3 = st.columns([2, 1, 1])
        if source_mode == "paper":
            heartbeats = _collect_log_heartbeats()
            heartbeat_line = _format_heartbeat_line(heartbeats)
            title = "Paper simulator metrics — No live routing."
            _render_status_box(health_col1, title, heartbeat_line, tone="ok")
        elif pipeline_snapshot.health_level == "ok":
            heartbeats = _collect_log_heartbeats()
            heartbeat_line = _format_heartbeat_line(heartbeats)
            _render_status_box(health_col1, "Pipeline health: OK", heartbeat_line, tone="ok")
        else:
            msg = "Pipeline attention required"
            if pipeline_snapshot.health_reasons:
                msg += " — " + "; ".join(pipeline_snapshot.health_reasons)
            heartbeats = _collect_log_heartbeats()
            details_line = _format_heartbeat_line(heartbeats)
            _render_status_box(health_col1, msg, details_line, tone="warn")

        trades_24h = pipeline_snapshot.trades_last_24h
        health_col2.metric("Trades (24h)", trades_24h)

        age_seconds = pipeline_snapshot.latest_closed_age_seconds
        if math.isfinite(age_seconds):
            age_value = age_seconds / 3600.0
            age_display = f"{age_value:.1f} h"
        else:
            age_display = "n/a"
        health_col3.metric("Latest close age", age_display)

        # heartbeat already shown in the status box above

    open_positions_count = int(len(open_positions_df))
    position_label = "open position" if open_positions_count == 1 else "open positions"
    summary_total_value = float(total_unrealized) if total_unrealized is not None else 0.0
    pos_label = "live" if source_mode == "live" else "paper"
    if source_mode == "live":
        usdc_balance = st.session_state.get("live_usdc_balance")
    else:
        usdc_balance = None
    exp_label_text = f"({open_positions_count}) Open positions ({pos_label}) | P/L {summary_total_value:+.2f}"
    if usdc_balance is not None:
        exp_label_text = f"{exp_label_text} | USDC {usdc_balance:,.2f}"
    label_color = None
    if open_positions_count > 0:
        label_color = "green" if total_unrealized >= 0 else "red"

    expander = st.expander(exp_label_text, expanded=False)
    if label_color:
        st.markdown(
            f"""
            <style>
            div[data-testid="stExpander"]:first-of-type button div:first-child span {{
                color: {label_color};
                font-weight: 600;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    with expander:
        if open_positions_df.empty:
            msg = "No open INTX positions detected." if source_mode == "live" else "No open paper trades logged."
            st.caption(msg)
        else:
            if source_mode == "paper":
                if "expires_at" in open_positions_df.columns:
                    now = datetime.now(UTC)

                    def _minutes_left(ts: pd.Timestamp) -> int:
                        if pd.isna(ts):
                            return 0
                        delta = ts - now
                        return int(delta.total_seconds() // 60)

                    def _format_time_left(ts: pd.Timestamp) -> str:
                        if pd.isna(ts):
                            return "n/a"
                        delta = ts - now
                        raw_seconds = int(delta.total_seconds())
                        if raw_seconds < 0:
                            prefix = "expired "
                            total_sec = abs(raw_seconds)
                        else:
                            prefix = ""
                            total_sec = raw_seconds
                        days, rem = divmod(total_sec, 86400)
                        hours, rem = divmod(rem, 3600)
                        minutes, _ = divmod(rem, 60)
                        if days > 0:
                            return f"{prefix}{days}d {hours:02d}h {minutes:02d}m"
                        return f"{prefix}{hours:02d}h {minutes:02d}m"

                    open_positions_df = open_positions_df.copy()
                    open_positions_df["expires_in_minutes"] = open_positions_df["expires_at"].apply(_minutes_left)
                    open_positions_df["expires_in"] = open_positions_df["expires_at"].apply(_format_time_left)
                    open_positions_df = open_positions_df.sort_values("expires_in_minutes")

                paper_cols = [
                    "symbol",
                    "position_side",
                    "position_usd",
                    "entry_price",
                    "last_price",
                    "take_profit",
                    "stop_loss",
                    "unrealized_pnl",
                    "unrealized_pct",
                    "expires_in",
                ]
                existing = [col for col in paper_cols if col in open_positions_df.columns]
                display_df = open_positions_df[existing].copy()
                display_df = display_df.rename(
                    columns={
                        "symbol": "Symbol",
                        "position_side": "Side",
                        "position_usd": "Size (USD)",
                        "entry_price": "Entry",
                        "last_price": "Last Price",
                        "take_profit": "Take Profit",
                        "stop_loss": "Stop Loss",
                        "unrealized_pnl": "Unrealized",
                        "unrealized_pct": "Unrealized %",
                        "expires_in": "Time Left",
                    }
                )
                st.dataframe(
                    display_df.style.format(
                        {
                            "Size (USD)": "{:.2f}",
                            "Entry": "{:.4f}",
                            "Last Price": "{:.4f}",
                            "Take Profit": "{:.4f}",
                            "Stop Loss": "{:.4f}",
                            "Unrealized": "{:+.2f}",
                            "Unrealized %": "{:+.4f}%",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                columns_to_use = [
                    "product_id",
                    "side",
                    "net_size",
                    "notional",
                    "entry_price",
                    "mark_price",
                    "unrealized_pnl",
                    "time_left_display",
                ]
                display_df = open_positions_df[columns_to_use].copy()
                display_df = display_df.rename(
                    columns={
                        "product_id": "Product",
                        "side": "Side",
                        "net_size": "Quantity",
                        "notional": "Size (USD)",
                        "entry_price": "Entry",
                        "mark_price": "Mark",
                        "unrealized_pnl": "Unrealized",
                        "time_left_display": "Time Left",
                    }
                )
                st.dataframe(
                    display_df.style.format(
                        {
                            "Quantity": "{:.4f}",
                            "Entry": "{:.4f}",
                            "Mark": "{:.4f}",
                            "Unrealized": "{:+.2f}",
                            "Size (USD)": "{:.2f}",
                        }
                    ),
                    use_container_width=True,
                )

    def format_compact(value: float) -> str:
        if value is None:
            return "—"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        if math.isnan(numeric):
            return "—"
        if math.isinf(numeric):
            return "∞" if numeric > 0 else "-∞"
        abs_value = abs(numeric)
        if abs_value >= 1:
            return f"{numeric:.2f}"
        if abs_value >= 0.01:
            return f"{numeric:.4f}"
        return f"{numeric:.6f}"

    def format_compact_pct(value: float) -> str:
        formatted = format_compact(value)
        if formatted in {"—"}:
            return formatted
        if formatted in {"∞", "-∞"}:
            return f"{formatted}%"
        return f"{formatted}%"

    def format_delta(value: float, suffix: str = "") -> Optional[str]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric):
            return None
        if math.isinf(numeric):
            sign = "∞" if numeric > 0 else "-∞"
            return f"{sign}{suffix}"
        abs_value = abs(numeric)
        if abs_value >= 1:
            delta_str = f"{numeric:+.2f}"
        elif abs_value >= 0.01:
            delta_str = f"{numeric:+.4f}"
        else:
            delta_str = f"{numeric:+.6f}"
        return f"{delta_str}{suffix}"

    summary_row1 = st.columns(4)
    summary_row1[0].metric("Trades", metrics["trades"])
    summary_row1[1].metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
    summary_row1[2].metric("Expectancy", f"{metrics['expectancy']:.2f}")
    summary_row1[3].metric("Sharpe (daily)", "∞" if math.isinf(metrics["sharpe_ratio"]) else f"{metrics['sharpe_ratio']:.2f}")

    summary_row2 = st.columns(4)
    profit_factor_value = metrics["profit_factor"]
    if math.isnan(profit_factor_value):
        profit_factor_display = "—"
    elif math.isinf(profit_factor_value):
        profit_factor_display = "∞"
    else:
        profit_factor_display = f"{profit_factor_value:.2f}"
    summary_row2[0].metric("Profit factor", profit_factor_display)

    median_pl_value = metrics["median_profit_loss"]
    summary_row2[1].metric("Median trade P/L", format_compact(median_pl_value))

    median_pct = metrics.get("median_profit_loss_pct", float("nan"))
    summary_row2[2].metric("Median trade P/L %", format_compact_pct(median_pct))
    summary_row2[3].metric("Ending equity", f"{metrics['ending_equity']:.2f}")

    summary_row3 = st.columns(4)
    summary_row3[0].metric("Best day", f"{metrics['best_day']:.2f}")
    summary_row3[1].metric("Worst day", f"{metrics['worst_day']:.2f}")
    summary_row3[2].metric("Max drawdown", f"{metrics['max_drawdown']:.2f}")
    summary_row3[3].metric("Max drawdown %", f"{metrics['max_drawdown_pct']:.2f}%")

    recent_window = metrics.get("recent_trade_window", 0)
    recent_cols = st.columns(3)

    recent_expectancy = metrics.get("recent_trade_expectancy", float("nan"))
    delta_expectancy = metrics.get("recent_trade_expectancy_delta", float("nan"))
    expectancy_label = "Recent expectancy" if recent_window <= 0 else f"Recent expectancy (last {recent_window} trades)"
    recent_cols[0].metric(
        expectancy_label,
        format_compact(recent_expectancy),
        delta=format_delta(delta_expectancy),
    )

    recent_win_rate = metrics.get("recent_trade_win_rate_pct", float("nan"))
    delta_win_rate = metrics.get("recent_trade_win_rate_delta", float("nan"))
    recent_cols[1].metric(
        "Recent win rate",
        format_compact_pct(recent_win_rate),
        delta=format_delta(delta_win_rate, suffix="%"),
    )

    recent_sharpe = metrics.get("recent_sharpe_ratio", float("nan"))
    delta_sharpe = metrics.get("recent_sharpe_delta", float("nan"))
    recent_cols[2].metric(
        "Recent Sharpe (7 days)",
        format_compact(recent_sharpe),
        delta=format_delta(delta_sharpe),
    )

    if metrics.get("recent_degradation_flag"):
        reasons = metrics.get("recent_degradation_reasons") or []
        joined_reasons = " ".join(reasons) if reasons else "Recent performance is weaker than the broader sample."
        st.warning(f"Recent degradation detected: {joined_reasons}")

    if source_mode == "live":
        st.markdown("---")
        with st.expander("Fill costs (fees & slippage)", expanded=False):
            if not API_KEY_PERPS or not API_SECRET_PERPS:
                st.info("Perps API keys not configured — cannot fetch Coinbase fills.")
            else:
                fills_limit = st.number_input(
                    "Fills to fetch",
                    min_value=100,
                    max_value=2000,
                    value=500,
                    step=100,
                    help="Pull recent fills to compute actual fees and fill-based PnL.",
                    key="fills_limit",
                )
                if st.button("Fetch Coinbase fills", key="fetch_fills"):
                    try:
                        st.session_state["live_fills_cache"] = _fetch_live_fills(int(fills_limit))
                        st.session_state["live_fills_loaded_at"] = datetime.now(UTC).isoformat()
                    except Exception as exc:
                        st.session_state["live_fills_cache"] = None
                        st.session_state["live_fills_loaded_at"] = ""
                        st.error(f"Failed to fetch fills: {exc}")

                fills_cache = st.session_state.get("live_fills_cache")
                if fills_cache:
                    cycles_by_product = compute_round_trip_cycles(fills_cache)
                    cycles = _flatten_cycles(cycles_by_product)

                    if selected_products:
                        cycles = [c for c in cycles if c.get("product_id") in selected_products]

                    start_dt = pd.to_datetime(start_str, utc=True) if start_str else None
                    end_dt = pd.to_datetime(end_str, utc=True) if end_str else None

                    def _cycle_end_time(cycle: Dict[str, Any]) -> Optional[datetime]:
                        raw = cycle.get("end_time")
                        if isinstance(raw, datetime):
                            return raw
                        if raw:
                            return pd.to_datetime(raw, utc=True, errors="coerce")
                        return None

                    filtered_cycles: List[Dict[str, Any]] = []
                    for cycle in cycles:
                        end_time = _cycle_end_time(cycle)
                        if end_time is None:
                            continue
                        if start_dt and end_time < start_dt:
                            continue
                        if end_dt and end_time >= end_dt:
                            continue
                        filtered_cycles.append(cycle)

                    if not filtered_cycles:
                        st.info("No fill cycles found in the selected window.")
                    else:
                        total_fees = sum(float(c.get("fees", 0.0)) for c in filtered_cycles)
                        gross_pnl = sum(float(c.get("gross_pnl", 0.0)) for c in filtered_cycles)
                        net_pnl = sum(float(c.get("realized_pnl", 0.0)) for c in filtered_cycles)
                        fee_pct = (total_fees / gross_pnl * 100.0) if gross_pnl else 0.0

                        fee_cols = st.columns(4)
                        fee_cols[0].metric("Fees (fills)", f"{total_fees:.2f}")
                        fee_cols[1].metric("Gross PnL (fills)", f"{gross_pnl:.2f}")
                        fee_cols[2].metric("Net PnL (fills)", f"{net_pnl:.2f}")
                        fee_cols[3].metric("Fee % of gross", f"{fee_pct:.2f}%")
                        loaded_at = st.session_state.get("live_fills_loaded_at")
                        if loaded_at:
                            st.caption(f"Cycles: {len(filtered_cycles)} | Loaded at {loaded_at}")

                st.markdown("**Exit slippage vs order target**")
                slippage_limit = st.number_input(
                    "Max exit orders to evaluate",
                    min_value=50,
                    max_value=400,
                    value=200,
                    step=50,
                    help="Uses closing order IDs from the filtered trades.",
                    key="slippage_limit",
                )

                if st.button("Compute exit slippage", key="compute_slippage"):
                    cb = CoinbaseService(api_key=API_KEY_PERPS, api_secret=API_SECRET_PERPS)
                    order_cache: Dict[str, Any] = {}

                    def _fetch_order(order_id: str) -> Dict[str, Any]:
                        cached = order_cache.get(order_id)
                        if cached is not None:
                            return cached
                        try:
                            response = cb.client.get_order(order_id=order_id)
                        except Exception:
                            response = {}
                        order_cache[order_id] = response
                        return response

                    slippage_df = build_exit_slippage_table(
                        pnl_filtered,
                        _fetch_order,
                        max_orders=int(slippage_limit),
                    )
                    st.session_state["live_slippage_df"] = slippage_df

                slippage_df = st.session_state.get("live_slippage_df")
                if isinstance(slippage_df, pd.DataFrame) and not slippage_df.empty:
                    total_slip = float(slippage_df["slippage_usd"].sum())
                    avg_slip = float(slippage_df["slippage_usd"].mean())
                    avg_bps = float(slippage_df["slippage_bps"].mean())
                    slip_cols = st.columns(3)
                    slip_cols[0].metric("Exit slippage (total)", f"{total_slip:+.2f}")
                    slip_cols[1].metric("Exit slippage (avg)", f"{avg_slip:+.2f}")
                    slip_cols[2].metric("Exit slippage (avg bps)", f"{avg_bps:+.2f}")
                    st.dataframe(
                        slippage_df.sort_values("closed_at", ascending=False).head(50),
                        use_container_width=True,
                    )
                else:
                    st.caption("No slippage data yet. Use “Compute exit slippage” to pull order targets.")

    st.markdown("---")
    st.subheader("Charts")
    chart_df = daily.set_index("date")
    st.line_chart(chart_df["equity"], height=280, use_container_width=True)
    st.area_chart(chart_df["drawdown"], height=280, use_container_width=True)
    st.bar_chart(chart_df["daily_pnl"], height=280, use_container_width=True)

    st.markdown("---")
    st.subheader("Shareable equity report")
    if build_equity_figure is None:
        st.info("Install plotly + kaleido to enable shareable equity reports.")
    else:
        report_title_default = f"{source_choice} Equity Report"
        report_title = st.text_input("Report title", value=report_title_default)
        report_fig = build_equity_figure(daily, metrics, report_title)
        st.plotly_chart(report_fig, use_container_width=True)

        report_timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        html_bytes = report_fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button(
            label="Download report (HTML)",
            data=html_bytes,
            file_name=f"equity_report_{report_timestamp}.html",
            mime="text/html",
        )

        try:
            png_bytes = report_fig.to_image(format="png", scale=2)
            st.download_button(
                label="Download report (PNG)",
                data=png_bytes,
                file_name=f"equity_report_{report_timestamp}.png",
                mime="image/png",
            )
        except Exception:
            st.info("PNG export requires kaleido (`pip install kaleido`).")

    st.markdown("---")
    st.subheader("Daily table")
    st.dataframe(chart_df, use_container_width=True)

    csv_bytes = chart_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download daily table as CSV",
        data=csv_bytes,
        file_name=f"watchdog_daily_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
