#!/usr/bin/env python3
"""
Daily equity drill-down for watchdog trade logs.

This helper reads `watchdog_closed_positions.csv`, applies the same
filters used by the other watchdog utilities (date window, count
window, tail selection), and produces a daily equity table plus
aggregate stats (max drawdown, Sharpe, per-trade variance, etc.).

Usage examples:
    python watchdog_daily_equity.py --start-date 2025-10-01
    python watchdog_daily_equity.py --start-date 2025-10-01 --start-count 101 --end-count 200
    python watchdog_daily_equity.py --start-date 2025-10-01 --last 40 --json --output daily_equity.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_CSV = Path("trade_logs") / "watchdog_closed_positions.csv"
DEFAULT_START_DATE = "2025-10-01"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily equity summary from watchdog logs")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to watchdog_closed_positions.csv")
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Inclusive UTC date (YYYY-MM-DD) for closed_at filter (default 2025-10-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional exclusive UTC date upper bound",
    )
    parser.add_argument(
        "--start-count",
        type=int,
        default=0,
        help="Start counting trades at this 1-based index after filters (default 1)",
    )
    parser.add_argument(
        "--end-count",
        type=int,
        default=0,
        help="Stop counting trades at this 1-based index (inclusive). 0 means no upper bound.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only analyse the most recent N trades after other filters",
    )
    parser.add_argument(
        "--starting-equity",
        type=float,
        default=1000.0,
        help="Starting equity used to build the cumulative curve (default 1000)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable tables",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the daily table as CSV",
    )
    return parser.parse_args()


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    return df


def _filter_date(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    if start_date:
        start = pd.to_datetime(start_date, utc=True)
        df = df[df["closed_at"] >= start]
    if end_date:
        end = pd.to_datetime(end_date, utc=True)
        df = df[df["closed_at"] < end]
    return df


def _select_count_window(df: pd.DataFrame, start_count: int, end_count: int) -> pd.DataFrame:
    if df.empty:
        return df
    start = start_count if start_count and start_count > 0 else None
    end = end_count if end_count and end_count > 0 else None
    if start is None and end is None:
        return df
    if start is None:
        start = 1
    if end is not None and end < start:
        return df.iloc[0:0]
    ordered = df.sort_values("closed_at")
    start_idx = start - 1
    if start_idx >= len(ordered):
        return ordered.iloc[0:0]
    if end is not None:
        return ordered.iloc[start_idx:end]
    return ordered.iloc[start_idx:]


def _select_last(df: pd.DataFrame, last: int) -> pd.DataFrame:
    if last <= 0 or df.empty:
        return df
    ordered = df.sort_values("closed_at")
    return ordered.tail(last)


@dataclass
class DailySummary:
    trades: int
    wins: int
    losses: int
    breakevens: int
    win_rate_pct: float
    expectancy_currency: float
    max_drawdown: float
    max_drawdown_pct: float
    best_day: float
    worst_day: float
    sharpe_ratio: float
    ending_equity: float
    std_dev_currency: float
    std_dev_drawdown: float


def _build_daily_equity(
    df: pd.DataFrame, starting_equity: float
) -> tuple[pd.DataFrame, DailySummary]:
    df = df.sort_values("closed_at").copy()
    profits = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0.0)
    df["profit_loss"] = profits

    df["date"] = df["closed_at"].dt.date
    daily = df.groupby("date")["profit_loss"].agg(["sum", "count"]).rename(columns={"sum": "daily_pnl", "count": "trades"})
    daily.reset_index(inplace=True)
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()
    equity = starting_equity + daily["cum_pnl"]
    daily["equity"] = equity

    prev_equity = np.concatenate(([starting_equity], equity.to_numpy()[:-1]))
    prev_equity = np.where(prev_equity == 0, np.nan, prev_equity)
    daily_return_pct = np.divide(daily["daily_pnl"], prev_equity, where=~np.isnan(prev_equity)) * 100.0
    daily_return_pct = np.nan_to_num(daily_return_pct)
    daily["daily_return_pct"] = daily_return_pct

    rolling_peak = equity.cummax()
    drawdown = equity - rolling_peak
    drawdown_pct = (drawdown / rolling_peak.replace(0, np.nan)) * 100.0
    drawdown_pct = drawdown_pct.fillna(0.0)
    daily["drawdown"] = drawdown
    daily["drawdown_pct"] = drawdown_pct

    total_trades = len(df)
    wins = int((profits > 0).sum())
    losses = int((profits < 0).sum())
    breakevens = total_trades - wins - losses
    win_rate = float(wins / total_trades * 100.0) if total_trades else 0.0
    expectancy = float(profits.mean()) if total_trades else 0.0
    best_day = float(daily["daily_pnl"].max()) if not daily.empty else 0.0
    worst_day = float(daily["daily_pnl"].min()) if not daily.empty else 0.0
    max_dd = float(drawdown.min()) if len(daily) else 0.0
    max_dd_pct = float(drawdown_pct.min()) if len(daily) else 0.0
    ending_equity = float(equity.iloc[-1]) if len(daily) else starting_equity
    std_dev_currency = float(profits.std(ddof=0)) if total_trades else 0.0
    std_dev_drawdown = float(drawdown.std(ddof=0)) if len(daily) else 0.0

    daily_returns_dec = daily_return_pct / 100.0
    if len(daily_returns_dec) >= 2:
        mean_ret = float(daily_returns_dec.mean())
        std_ret = float(daily_returns_dec.std(ddof=1))
        sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(365)) if std_ret > 0 else 0.0
    elif len(daily_returns_dec) == 1:
        sharpe_ratio = float("inf") if daily_returns_dec.iloc[0] > 0 else 0.0
    else:
        sharpe_ratio = 0.0

    summary = DailySummary(
        trades=total_trades,
        wins=wins,
        losses=losses,
        breakevens=breakevens,
        win_rate_pct=win_rate,
        expectancy_currency=expectancy,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        best_day=best_day,
        worst_day=worst_day,
        sharpe_ratio=sharpe_ratio,
        ending_equity=ending_equity,
        std_dev_currency=std_dev_currency,
        std_dev_drawdown=std_dev_drawdown,
    )
    return daily, summary


def _prepare_trade_subset(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = _filter_date(df, args.start_date, args.end_date)
    filtered = filtered.dropna(subset=["closed_at"]).sort_values("closed_at")
    filtered = _select_count_window(filtered, args.start_count, args.end_count)
    filtered = _select_last(filtered, args.last)
    return filtered


def _to_json(daily: pd.DataFrame, summary: DailySummary, args: argparse.Namespace) -> str:
    payload = {
        "meta": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "start_count": int(max(args.start_count, 1)) if args.start_count > 0 or args.end_count > 0 else 0,
            "end_count": int(args.end_count) if args.end_count > 0 else 0,
            "last": int(args.last or 0),
            "count_window_applied": bool(args.start_count > 0 or args.end_count > 0),
            "trades_considered": summary.trades,
            "starting_equity": args.starting_equity,
            "ending_equity": summary.ending_equity,
        },
        "summary": {
            "trades": summary.trades,
            "wins": summary.wins,
            "losses": summary.losses,
            "breakevens": summary.breakevens,
            "win_rate_pct": summary.win_rate_pct,
            "expectancy_currency": summary.expectancy_currency,
            "max_drawdown": summary.max_drawdown,
            "max_drawdown_pct": summary.max_drawdown_pct,
            "best_day": summary.best_day,
            "worst_day": summary.worst_day,
            "sharpe_ratio": summary.sharpe_ratio,
            "std_dev_currency": summary.std_dev_currency,
            "std_dev_drawdown": summary.std_dev_drawdown,
        },
        "daily": json.loads(
            daily.to_json(orient="records", date_format="iso")
        ),
    }
    return json.dumps(payload, indent=2)


def main() -> None:
    args = _parse_args()
    df = _load_dataframe(args.csv)
    subset = _prepare_trade_subset(df, args)

    if subset.empty:
        print("No trades match the requested filters.")
        return

    daily, summary = _build_daily_equity(subset, float(args.starting_equity))

    if args.output:
        daily.to_csv(args.output, index=False)

    if args.json:
        print(_to_json(daily, summary, args))
        return

    window_line = f"closed_at >= {args.start_date}" if args.start_date else "full history"
    if args.end_date:
        window_line += f" and < {args.end_date}"
    if args.start_count > 0 or args.end_count > 0:
        start = max(args.start_count, 1)
        window_line += f" | count {start}-{args.end_count if args.end_count > 0 else '∞'}"
    if args.last > 0:
        window_line += f" | last {args.last} trades"

    print("Daily Equity Report")
    print(f"Source: {args.csv}")
    print(f"Window: {window_line}")
    print()
    print("Summary")
    print(f"  trades: {summary.trades}")
    print(f"  wins / losses / breakevens: {summary.wins} / {summary.losses} / {summary.breakevens}")
    print(f"  win_rate_pct: {summary.win_rate_pct:.2f}")
    print(f"  expectancy ($/trade): {summary.expectancy_currency:.2f}")
    print(f"  std_dev ($/trade): {summary.std_dev_currency:.2f}")
    print(f"  max_drawdown: {summary.max_drawdown:.2f} ({summary.max_drawdown_pct:.2f}%)")
    print(f"  std_dev_drawdown: {summary.std_dev_drawdown:.2f}")
    print(f"  best_day / worst_day: {summary.best_day:.2f} / {summary.worst_day:.2f}")
    print(f"  ending_equity: {summary.ending_equity:.2f}")
    print(f"  sharpe_ratio (daily): {summary.sharpe_ratio:.2f}")
    print()
    print(daily.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    if args.output:
        print()
        print(f"Daily table written to: {args.output}")


if __name__ == "__main__":
    main()
