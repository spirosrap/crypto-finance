#!/usr/bin/env python3
"""Watchdog Reporting helper for closed positions.

Features:
- Filters watchdog_closed_positions.csv by closed_at date (default start 2025-10-01 UTC).
- Computes headline metrics (PnL, win/loss counts, duration stats).
- Breaks down PnL by closure_reason, product, and position_side.
- Buckets trades by holding-duration bands for quick stop-loss diagnostics.
- Surfaces top winners/losers and daily PnL tallies for the filtered window.

Usage example:
python watchdog_reporting.py --start-date 2025-10-01 --top-n 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

DEFAULT_CSV = Path("trade_logs") / "watchdog_closed_positions.csv"
DEFAULT_START_DATE = "2025-10-01"
DEFAULT_DURATION_BOUNDS = (12.0, 24.0)


@dataclass
class HeadlineMetrics:
    trades: int
    total_pnl: float
    avg_pnl: float
    median_pnl: float
    win_rate_pct: float
    loss_rate_pct: float
    breakeven_pct: float
    positive_days: int
    negative_days: int
    flat_days: int
    best_day: float
    worst_day: float
    avg_duration_h: float
    median_duration_h: float
    avg_r_multiple: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    peak_equity: float
    valley_equity: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize watchdog closed trades")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to watchdog_closed_positions.csv")
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Inclusive UTC date (YYYY-MM-DD) to filter closed_at",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional exclusive UTC date upper bound",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top/bottom trades and products to display",
    )
    parser.add_argument(
        "--duration-bounds",
        type=float,
        nargs="*",
        default=DEFAULT_DURATION_BOUNDS,
        help="Breakpoints in hours for duration buckets (ascending order)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable tables",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only analyze the most recent N trades (after date filters)",
    )
    return parser.parse_args()


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    if "opened_at" in df.columns:
        df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")
        df["duration_hours"] = (df["closed_at"] - df["opened_at"]).dt.total_seconds() / 3600.0
    else:
        df["duration_hours"] = pd.NA
    return df


def _filter_date(df: pd.DataFrame, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    start = pd.to_datetime(start_date, utc=True)
    mask = df["closed_at"] >= start
    if end_date:
        end = pd.to_datetime(end_date, utc=True)
        mask &= df["closed_at"] < end
    return df.loc[mask].copy()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _headline_metrics(df: pd.DataFrame) -> HeadlineMetrics:
    profits = _safe_numeric(df["profit_loss"]).fillna(0.0)
    wins = profits > 0
    losses = profits < 0
    breakevens = profits == 0
    total = len(profits)
    win_rate = float(wins.mean() * 100.0) if total else 0.0
    loss_rate = float(losses.mean() * 100.0) if total else 0.0
    be_rate = float(breakevens.mean() * 100.0) if total else 0.0
    daily = profits.groupby(df["closed_at"].dt.date).sum()
    pos_days = int((daily > 0).sum())
    neg_days = int((daily < 0).sum())
    flat_days = int((daily == 0).sum())
    best_day = float(daily.max()) if not daily.empty else 0.0
    worst_day = float(daily.min()) if not daily.empty else 0.0
    duration = df["duration_hours"] if "duration_hours" in df.columns else pd.Series(dtype=float)

    avg_loss_abs = float(profits[profits < 0].abs().mean()) if (losses.any()) else 0.0
    if avg_loss_abs > 0:
        r_multiple = profits / avg_loss_abs
        avg_r_multiple = float(r_multiple.mean())
        avg_win_r = float(r_multiple[r_multiple > 0].mean()) if (r_multiple > 0).any() else 0.0
        avg_loss_r = float((-r_multiple[r_multiple < 0]).mean()) if (r_multiple < 0).any() else 0.0
    else:
        avg_r_multiple = 0.0
        avg_win_r = 0.0
        avg_loss_r = 0.0

    win_rate_dec = wins.mean() if total else 0.0
    loss_rate_dec = losses.mean() if total else 0.0
    expectancy_r = (win_rate_dec * avg_win_r) - (loss_rate_dec * avg_loss_r)

    returns = _safe_numeric(df.get("profit_loss_pct", pd.Series(dtype=float))).dropna() / 100.0
    if len(returns) >= 2:
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        sharpe_ratio = float(mean_ret / std_ret) if std_ret > 0 else 0.0
        downside = returns[returns < 0]
        downside_std = float((downside.pow(2).mean()) ** 0.5) if not downside.empty else 0.0
        sortino_ratio = float(mean_ret / downside_std) if downside_std > 0 else 0.0
    elif len(returns) == 1:
        sharpe_ratio = float("inf") if returns.iloc[0] > 0 else 0.0
        sortino_ratio = sharpe_ratio
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

    initial_equity = 1000.0
    equity_curve = initial_equity + profits.cumsum()
    rolling_peak = equity_curve.cummax()
    drawdowns = equity_curve - rolling_peak
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0
    peak_equity = float(rolling_peak.max()) if not rolling_peak.empty else initial_equity
    valley_equity = float(equity_curve.loc[drawdowns.idxmin()]) if not drawdowns.empty else initial_equity
    drawdown_pct_series = (equity_curve / rolling_peak) - 1.0
    max_drawdown_pct = float(drawdown_pct_series.min() * 100.0) if not drawdown_pct_series.empty else 0.0

    return HeadlineMetrics(
        trades=total,
        total_pnl=float(profits.sum()),
        avg_pnl=float(profits.mean()) if total else 0.0,
        median_pnl=float(profits.median()) if total else 0.0,
        win_rate_pct=win_rate,
        loss_rate_pct=loss_rate,
        breakeven_pct=be_rate,
        positive_days=pos_days,
        negative_days=neg_days,
        flat_days=flat_days,
        best_day=best_day,
        worst_day=worst_day,
        avg_duration_h=float(duration.mean()) if not duration.empty else 0.0,
        median_duration_h=float(duration.median()) if not duration.empty else 0.0,
        avg_r_multiple=avg_r_multiple,
        expectancy_r=float(expectancy_r),
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        peak_equity=peak_equity,
        valley_equity=valley_equity,
    )


def _duration_labels(bounds: Sequence[float]) -> List[str]:
    sorted_bounds = sorted(set(bounds))
    labels: List[str] = []
    lower = 0.0
    for bound in sorted_bounds:
        if bound <= lower:
            continue
        labels.append(f"{lower:.0f}-{bound:.0f}h")
        lower = bound
    tail_label = f">={lower:.0f}h" if sorted_bounds else ">=0h"
    labels.append(tail_label)
    return labels


def _assign_duration_band(df: pd.DataFrame, bounds: Sequence[float]) -> pd.Series:
    if df["duration_hours"].isna().all():
        return pd.Series(["unknown"] * len(df), index=df.index)
    sorted_bounds = sorted(set(bounds))
    bins = [-float("inf"), *sorted_bounds, float("inf")]
    labels = _duration_labels(sorted_bounds)
    return pd.cut(df["duration_hours"], bins=bins, labels=labels, right=False)


def _group_stats(df: pd.DataFrame, by: str) -> pd.DataFrame:
    prof = _safe_numeric(df["profit_loss"])
    enriched = df.assign(_profit=prof, _is_win=(prof > 0).astype(float))
    grouped = enriched.groupby(by, observed=False)
    summary = grouped["_profit"].agg(count="count", sum="sum", mean="mean")
    win_rates = grouped["_is_win"].mean().mul(100.0)
    summary["win_rate_pct"] = win_rates
    return summary.sort_values("sum", ascending=False)


def _top_bottom_trades(df: pd.DataFrame, n: int) -> dict:
    cols = ["closed_at", "product_id", "position_side", "profit_loss", "duration_hours", "closure_reason"]
    top = df.nlargest(n, "profit_loss")[cols]
    bottom = df.nsmallest(n, "profit_loss")[cols]
    return {"top": top.reset_index(drop=True), "bottom": bottom.reset_index(drop=True)}


def _daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(df["closed_at"].dt.date)
    return grouped["profit_loss"].agg(["sum", "count"]).rename(columns={"sum": "daily_pnl", "count": "trades"})


def _serialize_df(df: pd.DataFrame) -> List[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _select_last(df: pd.DataFrame, last: int) -> pd.DataFrame:
    if last <= 0 or df.empty:
        return df
    if "closed_at" in df.columns:
        ordered = df.sort_values("closed_at")
        return ordered.tail(last)
    return df.tail(last)


def _text_table(title: str, df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    if df.empty:
        return f"{title}: (no data)"
    return f"{title}\n" + df.to_string(float_format=lambda x: format(x, floatfmt))


def main() -> None:
    args = _parse_args()
    df = _load_dataframe(args.csv)
    filtered = _filter_date(df, args.start_date, args.end_date)
    if filtered.empty:
        print("No trades after the requested start date.")
        return
    if int(args.last or 0) > 0:
        filtered = _select_last(filtered, int(args.last))
        if filtered.empty:
            print("No trades available after applying --last filter.")
            return
    filtered["duration_hours"] = filtered["duration_hours"].astype(float)
    filtered["profit_loss"] = _safe_numeric(filtered["profit_loss"]).fillna(0.0)
    duration_band = _assign_duration_band(filtered, args.duration_bounds)
    filtered["duration_band"] = duration_band

    headline = _headline_metrics(filtered)
    closure = _group_stats(filtered, "closure_reason")
    products = _group_stats(filtered, "product_id")
    sides = _group_stats(filtered, "position_side")
    duration_summary = _group_stats(filtered, "duration_band")
    daily = _daily_summary(filtered)
    tops = _top_bottom_trades(filtered, args.top_n)
    product_best = products.head(args.top_n)
    product_worst = products.tail(args.top_n).iloc[::-1]

    if args.json:
        payload_meta = {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "last": int(args.last or 0),
            "trades_considered": int(len(filtered)),
        }
        payload = {
            "meta": payload_meta,
            "headline": asdict(headline),
            "closure_reason": _serialize_df(closure.reset_index()),
            "products": _serialize_df(products.reset_index()),
            "position_side": _serialize_df(sides.reset_index()),
            "duration_bands": _serialize_df(duration_summary.reset_index()),
            "daily": _serialize_df(daily.reset_index()),
            "top_trades": _serialize_df(tops["top"]),
            "bottom_trades": _serialize_df(tops["bottom"]),
        }
        print(json.dumps(payload, indent=2))
        return

    print("Watchdog Reporting")
    print(f"Source: {args.csv}")
    window_line = f"Window: closed_at >= {args.start_date}"
    if args.end_date:
        window_line += f" and < {args.end_date}"
    if int(args.last or 0) > 0:
        window_line += f" | last {int(args.last)} trades"
    print(window_line)
    print("\nHeadline Metrics")
    for key, value in asdict(headline).items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    sections = [
        ("Closure Reason", closure),
        ("Best Products", product_best),
        ("Lagging Products", product_worst),
        ("Position Side", sides),
        ("Duration Bands", duration_summary),
        ("Daily PnL", daily.tail(max(args.top_n, 7))),
        ("Top Trades", tops["top"]),
        ("Bottom Trades", tops["bottom"]),
    ]
    for title, frame in sections:
        print()
        print(_text_table(title, frame))


if __name__ == "__main__":
    main()
