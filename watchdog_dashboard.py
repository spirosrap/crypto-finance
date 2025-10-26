#!/usr/bin/env python3
"""
Streamlit dashboard for watchdog logs.

Launch with:
    streamlit run watchdog_dashboard.py
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
from watchdog_utils import filter_by_date, select_count_window, select_last
from watchdog_close_old_positions import (
    _extract_entry_price,
    _extract_mark_price,
    _extract_position_open_time,
    _extract_symbol_and_size,
    _extract_unrealized_pnl,
    _get_portfolio_uuid,
)


UTC = timezone.utc
AUTO_REFRESH_SECONDS = 10


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

        net_progress = 0.0
        entry_time = None
        tolerance = max(1e-6, abs(size) * 1e-6)
        for order in sorted(all_orders, key=lambda o: o.get("created_time", "")):
            side_order = (order.get("side") or "").upper()
            try:
                filled = float(order.get("filled_size") or 0)
            except Exception:
                filled = 0.0
            created_raw = order.get("created_time") or order.get("completion_time")
            if side_order == "BUY":
                net_progress += filled
            elif side_order == "SELL":
                net_progress -= filled

            if abs(net_progress) < tolerance:
                entry_time = None
                continue

            if entry_time is None and created_raw:
                entry_time = created_raw

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
                "opened_at": opened_dt,
                "leverage": leverage,
            }
        )

    if not rows:
        return pd.DataFrame(), summary_total

    df = pd.DataFrame(rows)
    df["opened_at"] = pd.to_datetime(df["opened_at"], errors="coerce")
    numeric_cols = [
        "net_size",
        "entry_price",
        "mark_price",
        "unrealized_pnl",
        "notional",
        "hours_open",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["hours_open"] = df["hours_open"].round(2)
    df = df.sort_values("notional", ascending=False)
    total_unrealized = summary_total if summary_total or summary_total == 0.0 else float(df["unrealized_pnl"].sum())
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

    daily_reset = daily.reset_index()
    return daily_reset, metrics


def main() -> None:
    st.set_page_config(page_title="Watchdog Daily Equity", layout="wide")
    components.html(
        f"""
        <script>
        const watchdogInterval = {AUTO_REFRESH_SECONDS * 1000};
        setTimeout(() => {{
            window.parent.postMessage({{type: 'streamlit:rerun'}}, '*');
        }}, watchdogInterval);
        </script>
        """,
        height=0,
        key="watchdog-live-refresh",
    )
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
    st.caption("Visualise equity, drawdowns, and daily performance derived from watchdog logs.")

    default_csv = Path("trade_logs/watchdog_closed_positions.csv")
    csv_path = st.sidebar.text_input("CSV path", str(default_csv))
    csv_path = csv_path.strip()
    data_path = Path(csv_path)

    today = date.today()
    date_preset = st.sidebar.selectbox(
        "Date preset",
        options=("Custom", "Last 7 days", "Last 30 days", "Year to date"),
        index=0,
        help="Quickly apply a rolling time window; choose Custom to rely on the manual date inputs below.",
    )

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

    available_products = sorted(trades_df["product_id"].dropna().unique())
    selected_products = st.sidebar.multiselect(
        "Products (leave empty for all)",
        options=available_products,
        default=[],
        help="Filter summary and charts to selected instruments.",
    )

    batch_split = st.sidebar.number_input(
        "Trade split index",
        min_value=1,
        value=93,
        step=1,
        help="Index (1-based) separating the legacy allocation from scaled trades.",
    )
    batch_options = (
        "All trades",
        f"Trades 1–{batch_split + 1}",
        f"Trades {batch_split + 2}+",
    )
    batch_choice = st.sidebar.radio(
        "Trade batch",
        options=batch_options,
        index=0,
        help="Quick toggle between the initial batch of trades and the scaled batch.",
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
    filter_start_count = int(start_count)
    filter_end_count = int(end_count)
    if batch_choice == batch_options[1]:
        filter_start_count = 1
        filter_end_count = int(batch_split + 1)
    elif batch_choice == batch_options[2]:
        filter_start_count = int(batch_split + 2)
        filter_end_count = 0

    filtered = apply_filters(
        trades_df,
        start_str,
        end_str,
        filter_start_count,
        filter_end_count,
        int(tail_last),
        symbols=selected_products or None,
    )

    if filtered.empty:
        st.warning("No trades match the selected filters.")
        st.stop()

    daily, metrics = build_daily_equity(filtered, float(starting_equity))

    st.subheader("Summary")
    summary_notes = []
    if selected_products:
        summary_notes.append(f"Products: {', '.join(selected_products)}")
    if date_preset != "Custom":
        summary_notes.append(f"Preset window: {date_preset}")
    elif start_str or end_str:
        summary_notes.append("Custom date window")
    if batch_choice != batch_options[0]:
        summary_notes.append(batch_choice)
    if not filtered.empty:
        window_start = filtered["closed_at"].min()
        window_end = filtered["closed_at"].max()
        if window_start is not None and window_end is not None:
            summary_notes.append(
                f"Data range: {window_start.date().isoformat()} → {window_end.date().isoformat()}"
            )
    if summary_notes:
        st.caption(" | ".join(summary_notes))

    open_positions_df, total_unrealized = load_open_positions()
    exp_label_text = "Open positions (live)"
    label_color = None
    if not open_positions_df.empty:
        label_color = "green" if total_unrealized >= 0 else "red"
        exp_label_text += f" | P/L {total_unrealized:+.2f}"

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
            st.caption("No open INTX positions detected.")
        else:
            columns_to_use = [
                "product_id",
                "side",
                "net_size",
                "entry_price",
                "mark_price",
                "unrealized_pnl",
                "hours_open",
            ]
            display_df = open_positions_df[columns_to_use].copy()
            display_df = display_df.rename(
                columns={
                    "product_id": "Product",
                    "side": "Side",
                    "net_size": "Size",
                    "entry_price": "Entry",
                    "mark_price": "Mark",
                    "unrealized_pnl": "Unrealized",
                    "hours_open": "Hours",
                }
            )
            st.dataframe(
                display_df.style.format(
                    {
                        "Size": "{:.4f}",
                        "Entry": "{:.4f}",
                        "Mark": "{:.4f}",
                        "Unrealized": "{:+.2f}",
                        "Hours": "{:.2f}",
                    }
                ),
                width="stretch",
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
