#!/usr/bin/env python3
"""Enforce the daily stop using closed + open P/L for live and paper trades."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
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
HISTORY_PATH = REPO_ROOT / "logs" / "daily_stop_history.json"


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
    dedup_cols = [col for col in ("closed_at", "product_id", "closure_reason", "net_size", "profit_loss") if col in df.columns]
    if dedup_cols:
        # partial closes sometimes emit duplicate rows (same time/product/pnl); avoid double-counting
        df = df.drop_duplicates(subset=dedup_cols)
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


def _has_open_paper_positions(path: Path = PAPER_OPEN_LOG_PATH) -> bool:
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if df.empty:
        return False
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper() == "OPEN"]
    return not df.empty


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


def _has_open_live_positions() -> Optional[bool]:
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

    if isinstance(positions_response, dict):
        positions_raw = positions_response.get("positions", []) or []
    else:
        positions_raw = getattr(positions_response, "positions", []) or []

    for pos in positions_raw:
        pos_dict = pos if isinstance(pos, dict) else pos.to_dict()
        try:
            net_size = float(pos_dict.get("net_size", 0) or 0.0)
        except Exception:
            net_size = 0.0
        if abs(net_size) > 0:
            return True
    return False


def _threshold(equity: float, stop_pct: float, stop_usd: float) -> Optional[float]:
    thresholds = []
    if stop_pct and stop_pct > 0:
        thresholds.append(equity * (stop_pct / 100.0))
    if stop_usd and stop_usd > 0:
        thresholds.append(stop_usd)
    if not thresholds:
        return None
    return min(thresholds)


def _load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {"live": {"stops": [], "pause_until": None}}
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"live": {"stops": [], "pause_until": None}}
    if not isinstance(data, dict):
        return {"live": {"stops": [], "pause_until": None}}
    return data


def _save_history(data: dict) -> None:
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        return


def _parse_dates(values: list[str]) -> list[date]:
    dates: list[date] = []
    for val in values:
        try:
            dates.append(date.fromisoformat(str(val)))
        except Exception:
            continue
    return dates


def _count_recent(stops: list[date], today: date, window_days: int) -> int:
    if window_days <= 0:
        return 0
    cutoff = today - timedelta(days=window_days - 1)
    return sum(1 for d in stops if d >= cutoff)


def _update_live_stop_history(
    *,
    triggered: bool,
    today: date,
    streak_window_days: int,
    streak_count: int,
    pause_days: int,
    warn_window_days: int,
    warn_count: int,
    escalate_window_days: int,
    escalate_count: int,
) -> dict:
    history = _load_history()
    live = history.get("live", {}) if isinstance(history.get("live", {}), dict) else {}
    stops_raw = live.get("stops", [])
    if not isinstance(stops_raw, list):
        stops_raw = []
    stops = _parse_dates([str(val) for val in stops_raw])

    if triggered and today not in stops:
        stops.append(today)

    max_window = max(streak_window_days, warn_window_days, escalate_window_days, 1)
    cutoff = today - timedelta(days=max_window - 1)
    stops = [d for d in stops if d >= cutoff]
    stops.sort()

    pause_until = None
    existing_pause = live.get("pause_until")
    if existing_pause:
        try:
            pause_until = date.fromisoformat(str(existing_pause))
        except Exception:
            pause_until = None

    streak_hits = _count_recent(stops, today, streak_window_days)
    warn_hits = _count_recent(stops, today, warn_window_days)
    escalate_hits = _count_recent(stops, today, escalate_window_days)

    if streak_count > 0 and streak_hits >= streak_count:
        candidate = today + timedelta(days=max(pause_days, 1))
        if pause_until is None or candidate > pause_until:
            pause_until = candidate

    if pause_until is not None and pause_until < today:
        pause_until = None

    history["live"] = {
        "stops": [d.isoformat() for d in stops],
        "pause_until": pause_until.isoformat() if pause_until else None,
    }
    _save_history(history)

    return {
        "streak_hits": streak_hits,
        "warn_hits": warn_hits,
        "escalate_hits": escalate_hits,
        "pause_until": pause_until,
        "streak_window_days": streak_window_days,
        "warn_window_days": warn_window_days,
        "escalate_window_days": escalate_window_days,
        "streak_count": streak_count,
        "warn_count": warn_count,
        "escalate_count": escalate_count,
    }


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def main() -> int:
    _load_dotenv_if_available()
    thresholds = load_risk_thresholds()
    stop_pct = float(thresholds.get("daily_stop_pct", 2.0) or 0.0)
    stop_usd = float(thresholds.get("daily_stop_usd", 20.0) or 0.0)
    stop_equity = float(thresholds.get("daily_stop_equity", 1000.0) or 0.0)
    streak_window_days = int(thresholds.get("daily_stop_streak_window_days", 7) or 7)
    streak_count = int(thresholds.get("daily_stop_streak_count", 3) or 3)
    pause_days = int(thresholds.get("daily_stop_pause_days", 3) or 3)
    warn_window_days = int(thresholds.get("daily_stop_warn_window_days", 14) or 14)
    warn_count = int(thresholds.get("daily_stop_warn_count", 5) or 5)
    escalate_window_days = int(thresholds.get("daily_stop_escalate_window_days", 21) or 21)
    escalate_count = int(thresholds.get("daily_stop_escalate_count", 7) or 7)
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

    today = datetime.now(UTC).date()
    live_status = _update_live_stop_history(
        triggered=bool(triggered_live),
        today=today,
        streak_window_days=streak_window_days,
        streak_count=streak_count,
        pause_days=pause_days,
        warn_window_days=warn_window_days,
        warn_count=warn_count,
        escalate_window_days=escalate_window_days,
        escalate_count=escalate_count,
    )

    pause_until = live_status.get("pause_until")
    if pause_until:
        print(
            f"Live pause active: {live_status['streak_hits']}/{streak_count} stops in "
            f"{streak_window_days}d (pause until {pause_until})."
        )
    else:
        print(
            f"Live stop streak: {live_status['streak_hits']}/{streak_count} stops in "
            f"{streak_window_days}d."
        )
    if warn_count > 0 and live_status["warn_hits"] >= warn_count:
        print(
            f"Warning: {live_status['warn_hits']}/{warn_count} daily stops in "
            f"{warn_window_days}d. Consider reducing size by 50%."
        )
    if escalate_count > 0 and live_status["escalate_hits"] >= escalate_count:
        print(
            f"Escalation: {live_status['escalate_hits']}/{escalate_count} daily stops in "
            f"{escalate_window_days}d. Tighten filters or switch to paper-only."
        )

    if triggered_live and run_live:
        has_live_positions = _has_open_live_positions()
        if has_live_positions is False:
            print("Daily stop ACTIVE (live): no open positions detected; skip close.")
        else:
            if has_live_positions is None:
                print("Daily stop ACTIVE (live): open positions unknown; closing to be safe.")
            else:
                print("Daily stop ACTIVE (live): closing live positions.")
            _run([sys.executable, str(REPO_ROOT / "close_positions.py")])
    if triggered_paper and run_paper:
        if _has_open_paper_positions():
            print("Daily stop ACTIVE (paper): closing paper positions.")
            _run([sys.executable, str(REPO_ROOT / "paper_finder_simulator.py"), "close", "--all", "--reason", "daily_stop"])
        else:
            print("Daily stop ACTIVE (paper): no open paper trades to close.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
