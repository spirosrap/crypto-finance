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

from watchdog_utils import filter_by_date, select_count_window, select_last


UTC = timezone.utc


def load_watchdog_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
    if "profit_loss_pct" in df.columns:
        df["profit_loss_pct"] = pd.to_numeric(df["profit_loss_pct"], errors="coerce")
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
    filtered = filter_by_date(df, start, end)
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
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    trades = trades.sort_values("closed_at")
    trades["date"] = trades["closed_at"].dt.date

    daily_pnl = trades.groupby("date")["profit_loss"].sum(min_count=1).fillna(0.0)
    trade_counts = trades.groupby("date").size()
    daily = pd.DataFrame({"daily_pnl": daily_pnl, "trades": trade_counts})
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

    profit_loss_series = trades["profit_loss"]
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

    profit_loss_pct_series = trades["profit_loss_pct"] if "profit_loss_pct" in trades.columns else None
    if profit_loss_pct_series is not None:
        valid_profit_loss_pct = profit_loss_pct_series.dropna()
        nonzero_profit_loss_pct = valid_profit_loss_pct[valid_profit_loss_pct != 0]
        if not nonzero_profit_loss_pct.empty:
            median_profit_loss_pct = float(nonzero_profit_loss_pct.median())
        elif not valid_profit_loss_pct.empty:
            median_profit_loss_pct = float(valid_profit_loss_pct.median())
        else:
            median_profit_loss_pct = float("nan")
    else:
        median_profit_loss_pct = float("nan")

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

    metrics = {
        "trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate_pct": win_rate_pct,
        "expectancy": expectancy,
        "profit_factor": float(profit_factor),
        "median_profit_loss": median_profit_loss,
        "median_profit_loss_pct": median_profit_loss_pct,
        "best_day": float(daily["daily_pnl"].max()) if not daily.empty else 0.0,
        "worst_day": float(daily["daily_pnl"].min()) if not daily.empty else 0.0,
        "ending_equity": float(daily["equity"].iloc[-1]) if not daily.empty else starting_equity,
        "max_drawdown": float(drawdown.min()) if not daily.empty else 0.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()) if not daily.empty else 0.0,
        "std_dev_trade": std_dev_trade,
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
    if math.isnan(median_pl_value):
        median_pl_display = "—"
    elif abs(median_pl_value) >= 1:
        median_pl_display = f"{median_pl_value:.2f}"
    elif abs(median_pl_value) >= 0.01:
        median_pl_display = f"{median_pl_value:.4f}"
    else:
        median_pl_display = f"{median_pl_value:.6f}"
    summary_row2[1].metric("Median trade P/L", median_pl_display)

    median_pct = metrics.get("median_profit_loss_pct", float("nan"))
    if math.isnan(median_pct):
        median_pct_display = "—"
    elif abs(median_pct) >= 1:
        median_pct_display = f"{median_pct:.2f}%"
    elif abs(median_pct) >= 0.01:
        median_pct_display = f"{median_pct:.4f}%"
    else:
        median_pct_display = f"{median_pct:.6f}%"
    summary_row2[2].metric("Median trade P/L %", median_pct_display)
    summary_row2[3].metric("Ending equity", f"{metrics['ending_equity']:.2f}")

    summary_row3 = st.columns(4)
    summary_row3[0].metric("Best day", f"{metrics['best_day']:.2f}")
    summary_row3[1].metric("Worst day", f"{metrics['worst_day']:.2f}")
    summary_row3[2].metric("Max drawdown", f"{metrics['max_drawdown']:.2f}")
    summary_row3[3].metric("Max drawdown %", f"{metrics['max_drawdown_pct']:.2f}%")

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
