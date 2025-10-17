#!/usr/bin/env python3
"""
Streamlit dashboard for watchdog logs.

Launch with:
    streamlit run watchdog_dashboard.py
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


UTC = timezone.utc


def load_watchdog_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["closed_at"]).sort_values("closed_at")
    return df


def apply_filters(
    df: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
    start_count: int,
    end_count: int,
    last: int,
) -> pd.DataFrame:
    filtered = df.copy()
    if start:
        filtered = filtered[filtered["closed_at"] >= pd.to_datetime(start, utc=True)]
    if end:
        filtered = filtered[filtered["closed_at"] < pd.to_datetime(end, utc=True)]

    if filtered.empty:
        return filtered

    if start_count > 0 or end_count > 0:
        start_idx = max(start_count - 1, 0)
        end_idx = None if end_count <= 0 else end_count
        filtered = filtered.iloc[start_idx:end_idx]

    if last and last > 0:
        filtered = filtered.tail(last)

    return filtered


def build_daily_equity(
    trades: pd.DataFrame,
    starting_equity: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    trades = trades.sort_values("closed_at")
    trades["date"] = trades["closed_at"].dt.date

    daily = trades.groupby("date")["profit_loss"].agg(["sum", "count"]).rename(columns={"sum": "daily_pnl", "count": "trades"})
    daily["cum_pnl"] = daily["daily_pnl"].cumsum()
    daily["equity"] = starting_equity + daily["cum_pnl"]

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

    metrics = {
        "trades": int(len(trades)),
        "wins": int((trades["profit_loss"] > 0).sum()),
        "losses": int((trades["profit_loss"] < 0).sum()),
        "breakevens": int((trades["profit_loss"] == 0).sum()),
        "win_rate_pct": float((trades["profit_loss"] > 0).mean() * 100) if len(trades) else 0.0,
        "expectancy": float(trades["profit_loss"].mean()) if len(trades) else 0.0,
        "best_day": float(daily["daily_pnl"].max()) if not daily.empty else 0.0,
        "worst_day": float(daily["daily_pnl"].min()) if not daily.empty else 0.0,
        "ending_equity": float(daily["equity"].iloc[-1]) if not daily.empty else starting_equity,
        "max_drawdown": float(drawdown.min()) if not daily.empty else 0.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()) if not daily.empty else 0.0,
        "std_dev_trade": float(trades["profit_loss"].std(ddof=0)) if len(trades) else 0.0,
        "std_dev_drawdown": float(drawdown.std(ddof=0)) if len(daily) else 0.0,
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

    daily_reset = daily.reset_index()
    return daily_reset, metrics


def main() -> None:
    st.set_page_config(page_title="Watchdog Daily Equity", layout="wide")
    st.title("Watchdog Daily Equity Dashboard")
    st.caption("Visualise equity, drawdowns, and daily performance derived from watchdog logs.")

    default_csv = Path("trade_logs/watchdog_closed_positions.csv")
    csv_path = st.sidebar.text_input("CSV path", str(default_csv))
    csv_path = csv_path.strip()
    data_path = Path(csv_path)

    start_date_default = date(2025, 10, 1)
    start_date_input = st.sidebar.date_input("Start date", start_date_default)
    end_date_input = st.sidebar.date_input("End date", value=None)
    if isinstance(end_date_input, list):
        end_date_input = end_date_input[0] if end_date_input else None

    start_count = st.sidebar.number_input("Start count (1-based)", min_value=0, value=0, step=1)
    end_count = st.sidebar.number_input("End count (inclusive, 0 for none)", min_value=0, value=0, step=1)
    tail_last = st.sidebar.number_input("Last N trades (0 to ignore)", min_value=0, value=0, step=1)
    starting_equity = st.sidebar.number_input("Starting equity", min_value=0.0, value=1000.0, step=100.0)

    if st.sidebar.button("Reset filters"):
        st.experimental_rerun()

    try:
        trades_df = load_watchdog_csv(data_path)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    start_str = start_date_input.isoformat() if start_date_input else None
    end_str = end_date_input.isoformat() if end_date_input else None
    filtered = apply_filters(trades_df, start_str, end_str, int(start_count), int(end_count), int(tail_last))

    if filtered.empty:
        st.warning("No trades match the selected filters.")
        st.stop()

    daily, metrics = build_daily_equity(filtered, float(starting_equity))

    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades", metrics["trades"])
    c2.metric("Win rate", f"{metrics['win_rate_pct']:.1f}%")
    c3.metric("Expectancy", f"{metrics['expectancy']:.2f}")
    c4.metric("Sharpe (daily)", "∞" if math.isinf(metrics["sharpe_ratio"]) else f"{metrics['sharpe_ratio']:.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Best day", f"{metrics['best_day']:.2f}")
    c6.metric("Worst day", f"{metrics['worst_day']:.2f}")
    c7.metric("Ending equity", f"{metrics['ending_equity']:.2f}")
    c8.metric("Max drawdown", f"{metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")

    st.markdown("---")
    st.subheader("Charts")
    chart_df = daily.set_index("date")
    st.line_chart(chart_df["equity"], height=280, width='stretch')
    st.area_chart(chart_df["drawdown"], height=280, width='stretch')
    st.bar_chart(chart_df["daily_pnl"], height=280, width='stretch')

    st.markdown("---")
    st.subheader("Daily table")
    st.dataframe(chart_df, width='stretch')

    csv_bytes = chart_df.to_csv().encode("utf-8")
    st.download_button(
        label="Download daily table as CSV",
        data=csv_bytes,
        file_name=f"watchdog_daily_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
