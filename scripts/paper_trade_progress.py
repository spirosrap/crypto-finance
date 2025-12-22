#!/usr/bin/env python3
"""Summarize paper-trade progress toward a target sample size."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def _expectancy(pnl_pct: pd.Series) -> Optional[float]:
    pnl_pct = pd.to_numeric(pnl_pct, errors="coerce").dropna()
    if pnl_pct.empty:
        return None
    wins = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct < 0]
    win_rate = len(wins) / len(pnl_pct)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.abs().mean() if len(losses) else 0.0
    return win_rate * avg_win - (1 - win_rate) * avg_loss


def _format_last_ts(df: pd.DataFrame) -> str:
    if df.empty or "closed_at" not in df.columns:
        return "n/a"
    try:
        ts = pd.to_datetime(df["closed_at"], utc=True, errors="coerce").max()
    except Exception:
        return "n/a"
    if pd.isna(ts):
        return "n/a"
    return ts.strftime("%Y-%m-%d %H:%M:%SZ")


def _closure_rate(df: pd.DataFrame, reason: str) -> float:
    if df.empty or "closure_reason" not in df.columns:
        return 0.0
    total = len(df)
    if total == 0:
        return 0.0
    reasons = df["closure_reason"].fillna("").astype(str).str.lower()
    if reason == "expired":
        return float(reasons.str.startswith("expired").sum()) / total
    return float((reasons == reason).sum()) / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-trade progress and expectancy.")
    parser.add_argument(
        "--closed",
        type=Path,
        default=Path("trade_logs/paper_finder_closed_positions.csv"),
        help="Paper closed trades CSV (default: trade_logs/paper_finder_closed_positions.csv).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=100,
        help="Target number of trades (default: 100).",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only analyze the last N trades (0 = all).",
    )
    args = parser.parse_args()

    df = _read_csv(args.closed)
    if df.empty:
        print(f"No closed paper trades found at {args.closed}")
        return

    if args.last and args.last > 0:
        df = df.tail(args.last)

    total = len(df)
    progress = min(1.0, total / max(args.target, 1))
    pnl_pct = pd.to_numeric(df.get("profit_loss_pct"), errors="coerce")
    win_rate = float((pnl_pct > 0).mean()) if not pnl_pct.empty else 0.0
    avg_pct = float(pnl_pct.mean()) if not pnl_pct.empty else 0.0
    expectancy = _expectancy(pnl_pct)

    expiry_rate = _closure_rate(df, "expired")
    tp_rate = _closure_rate(df, "take_profit")
    sl_rate = _closure_rate(df, "stop_loss")

    print(f"Paper trades: {total}/{args.target} ({progress * 100:.1f}% toward target)")
    print(f"Win rate: {_pct(win_rate)} | Avg %: {avg_pct:.3f}% | Expectancy: {expectancy:.3f}%" if expectancy is not None else "Expectancy: n/a")
    print(f"TP: {_pct(tp_rate)} | SL: {_pct(sl_rate)} | Expired: {_pct(expiry_rate)}")
    print(f"Last closed: {_format_last_ts(df)}")


if __name__ == "__main__":
    main()
