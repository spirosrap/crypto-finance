"""Build shareable equity reports from closed trade logs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_closed_positions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    if "closed_at" not in df.columns:
        raise ValueError("CSV missing required column 'closed_at'")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    for col in ("profit_loss", "profit_loss_pct", "mae", "mfe"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["closed_at"]).sort_values("closed_at")
    return df


def build_daily_equity(
    trades: pd.DataFrame,
    starting_equity: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if trades.empty:
        empty_daily = pd.DataFrame(
            columns=["date", "daily_pnl", "trades", "cum_pnl", "equity", "daily_return_pct", "drawdown", "drawdown_pct"]
        )
        metrics = {
            "trades": 0,
            "win_rate_pct": 0.0,
            "expectancy": 0.0,
            "expectancy_pct": 0.0,
            "avg_pct": float("nan"),
            "avg_win_pct": float("nan"),
            "avg_loss_pct": float("nan"),
            "profit_factor": float("nan"),
            "ending_equity": starting_equity,
            "starting_equity": starting_equity,
            "total_return_pct": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": float("nan"),
        }
        return empty_daily, metrics

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

    profit_loss_pct_series = trades["profit_loss_pct"] if "profit_loss_pct" in trades.columns else pd.Series(dtype=float)
    profit_loss_pct_series = pd.to_numeric(profit_loss_pct_series, errors="coerce").dropna()
    avg_pct = float(profit_loss_pct_series.mean()) if not profit_loss_pct_series.empty else float("nan")
    wins_pct = profit_loss_pct_series[profit_loss_pct_series > 0]
    losses_pct = profit_loss_pct_series[profit_loss_pct_series < 0]
    avg_win_pct = float(wins_pct.mean()) if not wins_pct.empty else float("nan")
    avg_loss_pct = float(losses_pct.mean()) if not losses_pct.empty else float("nan")
    win_rate_pct = float((valid_profit_loss > 0).mean() * 100.0) if valid_trade_count else 0.0
    expectancy = float(valid_profit_loss.mean()) if valid_trade_count else 0.0
    if len(profit_loss_pct_series):
        win_rate_pct_series = len(wins_pct) / len(profit_loss_pct_series)
        avg_win = float(wins_pct.mean()) if len(wins_pct) else 0.0
        avg_loss = abs(float(losses_pct.mean())) if len(losses_pct) else 0.0
        expectancy_pct = win_rate_pct_series * avg_win - (1 - win_rate_pct_series) * avg_loss
    else:
        expectancy_pct = 0.0

    daily_returns = daily["daily_return_pct"] / 100.0
    if len(daily_returns) >= 2:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std(ddof=1)
        sharpe = (mean_ret / std_ret) * math.sqrt(365) if std_ret > 0 else 0.0
    elif len(daily_returns) == 1:
        sharpe = float("inf") if daily_returns.iloc[0] > 0 else 0.0
    else:
        sharpe = float("nan")

    ending_equity = float(daily["equity"].iloc[-1]) if not daily.empty else starting_equity
    total_return_pct = ((ending_equity - starting_equity) / starting_equity * 100.0) if starting_equity else 0.0

    metrics = {
        "trades": int(len(trades)),
        "win_rate_pct": win_rate_pct,
        "expectancy": expectancy,
        "expectancy_pct": expectancy_pct,
        "avg_pct": avg_pct,
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": float(profit_factor),
        "ending_equity": ending_equity,
        "starting_equity": starting_equity,
        "total_return_pct": total_return_pct,
        "max_drawdown": float(drawdown.min()) if not daily.empty else 0.0,
        "max_drawdown_pct": float(daily["drawdown_pct"].min()) if not daily.empty else 0.0,
        "sharpe_ratio": float(sharpe),
    }

    daily_reset = daily.reset_index()
    return daily_reset, metrics


def _format_metrics_block(metrics: Dict[str, float]) -> str:
    win_rate = metrics.get("win_rate_pct", float("nan"))
    expectancy_pct = metrics.get("expectancy_pct", float("nan"))
    avg_pct = metrics.get("avg_pct", float("nan"))
    profit_factor = metrics.get("profit_factor", float("nan"))
    sharpe = metrics.get("sharpe_ratio", float("nan"))
    max_dd_pct = metrics.get("max_drawdown_pct", float("nan"))
    total_return_pct = metrics.get("total_return_pct", float("nan"))

    if math.isnan(profit_factor):
        profit_factor_display = "n/a"
    elif math.isinf(profit_factor):
        profit_factor_display = "inf"
    else:
        profit_factor_display = f"{profit_factor:.2f}"

    if math.isinf(sharpe):
        sharpe_display = "inf"
    elif math.isnan(sharpe):
        sharpe_display = "n/a"
    else:
        sharpe_display = f"{sharpe:.2f}"

    return (
        f"Win rate: {win_rate:.1f}% | Expectancy: {expectancy_pct:+.2f}% | Avg trade: {avg_pct:+.2f}%<br>"
        f"Profit factor: {profit_factor_display} | Sharpe (daily): {sharpe_display} | Max DD: {max_dd_pct:.2f}%<br>"
        f"Total return: {total_return_pct:+.2f}%"
    )


def build_equity_figure(
    daily: pd.DataFrame,
    metrics: Dict[str, float],
    title: str,
) -> go.Figure:
    if daily.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=title,
            height=520,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        fig.add_annotation(
            text="No closed trades available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="#666"),
        )
        return fig

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("Equity Curve", "Drawdown (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["equity"],
            mode="lines+markers",
            line=dict(color="#2a9d8f", width=2.5),
            marker=dict(size=5, color="#2a9d8f"),
            name="Equity",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=daily["date"],
            y=daily["drawdown_pct"],
            marker_color="#e76f51",
            name="Drawdown",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, xanchor="left", y=0.98, yanchor="top"),
        height=720,
        margin=dict(l=50, r=50, t=160, b=60),
        showlegend=False,
        font=dict(size=13),
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    fig.add_annotation(
        text=_format_metrics_block(metrics),
        x=0.98,
        y=1.18,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="left",
        font=dict(size=12, color="#222"),
        bordercolor="#e0e0e0",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.9)",
        xanchor="right",
        yanchor="top",
    )

    return fig
