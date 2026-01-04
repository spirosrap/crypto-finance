#!/usr/bin/env python3
"""Enforce the daily stop using closed + open P/L for live and paper trades."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coinbaseservice import CoinbaseService
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


UTC = timezone.utc
LIVE_CLOSED_LOG_PATH = REPO_ROOT / "trade_logs" / "watchdog_closed_positions.csv"
PAPER_CLOSED_LOG_PATH = REPO_ROOT / "trade_logs" / "paper_finder_closed_positions.csv"
PAPER_OPEN_LOG_PATH = REPO_ROOT / "trade_logs" / "paper_finder_open_positions.csv"

API_KEY_PERPS, API_SECRET_PERPS = get_perps_credentials()


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


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
    if df.empty:
        return 0.0
    today = datetime.now(UTC).date()
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


def _load_open_paper_pnl(path: Path = PAPER_OPEN_LOG_PATH) -> Optional[float]:
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


def _threshold(equity: float, stop_pct: float, stop_usd: float) -> Optional[float]:
    thresholds = []
    if stop_pct and stop_pct > 0:
        thresholds.append(equity * (stop_pct / 100.0))
    if stop_usd and stop_usd > 0:
        thresholds.append(stop_usd)
    if not thresholds:
        return None
    return min(thresholds)


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def main() -> int:
    _load_dotenv_if_available()
    thresholds = load_risk_thresholds()
    stop_pct = float(thresholds.get("daily_stop_pct", 2.0) or 0.0)
    stop_usd = float(thresholds.get("daily_stop_usd", 20.0) or 0.0)
    stop_equity = float(thresholds.get("daily_stop_equity", 1000.0) or 0.0)
    daily_stop_threshold = _threshold(stop_equity, stop_pct, stop_usd)

    live_closed = _daily_pnl_today(LIVE_CLOSED_LOG_PATH)
    paper_closed = _daily_pnl_today(PAPER_CLOSED_LOG_PATH)
    live_open = _load_open_live_pnl()
    paper_open = _load_open_paper_pnl()
    live_total = _combine_pnl(live_closed, live_open)
    paper_total = _combine_pnl(paper_closed, paper_open)

    print(
        f"Daily stop threshold: {daily_stop_threshold:.2f} (pct={stop_pct:.2f}, usd={stop_usd:.2f}, equity={stop_equity:.2f})"
        if daily_stop_threshold is not None
        else "Daily stop threshold: n/a"
    )
    print(f"Live P/L today: closed={live_closed:+.2f} open={live_open:+.2f} total={live_total:+.2f}" if live_total is not None else "Live P/L today: n/a")
    print(f"Paper P/L today: closed={paper_closed:+.2f} open={paper_open:+.2f} total={paper_total:+.2f}" if paper_total is not None else "Paper P/L today: n/a")

    run_live = os.environ.get("RUN_LIVE", "1") == "1"
    run_paper = os.environ.get("RUN_PAPER", "1") == "1"

    triggered_live = daily_stop_threshold is not None and live_total is not None and live_total <= -daily_stop_threshold
    triggered_paper = daily_stop_threshold is not None and paper_total is not None and paper_total <= -daily_stop_threshold

    if triggered_live and run_live:
        print("Daily stop ACTIVE (live): closing live positions.")
        _run([sys.executable, str(REPO_ROOT / "close_positions.py")])
    if triggered_paper and run_paper:
        print("Daily stop ACTIVE (paper): closing paper positions.")
        _run([sys.executable, str(REPO_ROOT / "paper_finder_simulator.py"), "close", "--all", "--reason", "daily_stop"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
