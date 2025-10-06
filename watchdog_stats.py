#!/usr/bin/env python3
"""
Watchdog Stats — expectancy, win rate, drawdown (percent), average R

Reads `trade_logs/watchdog_closed_positions.csv` and computes summary metrics:
  - Win rate (% of trades with positive PnL; breakeven treated as 0)
  - Expectancy (mean PnL per trade, in currency units)
  - Max drawdown percent (from cumulative PnL equity curve)
  - Average R (mean R-multiple per trade), where 1R is derived from losses

R basis options:
  - avg_loss (default): 1R = average absolute loss size across losing trades
  - median_loss: 1R = median absolute loss size across losing trades
  - fixed: 1R = --risk-dollar

Notes/assumptions:
  - Uses `profit_loss` column from the CSV (already includes breakeven adjustment
    when `closure_reason` is `expired_breakeven`).
  - Trades without numeric PnL are ignored.
  - If there are no losing trades and no fixed risk provided, Average R is not available.
  - Provide `--ending-equity` to infer the starting equity as ending equity minus
    cumulative PnL for the selected trades; otherwise supply `--starting-equity`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("trade_logs") / "watchdog_closed_positions.csv"


@dataclass(frozen=True)
class StatsResult:
    total_trades: int
    wins: int
    losses: int
    breakevens: int
    win_rate_pct: float
    expectancy_currency: float
    max_drawdown_pct: float
    average_r: Optional[float]
    expectancy_r: Optional[float]
    starting_equity_used: float
    ending_equity: Optional[float]


def _safe_numeric(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)


def _equity_curve_and_max_dd_pct(pnls: np.ndarray, starting_equity: float) -> Tuple[np.ndarray, float]:
    # Ensure a positive baseline to avoid division by ~0 issues and seed the equity curve
    base = float(starting_equity) if starting_equity and starting_equity > 0 else 1.0
    pnls = np.asarray(pnls, dtype=float)
    equity = base + pnls.cumsum()
    if equity.size == 0:
        return equity, 0.0

    # Include the baseline before the first trade so initial drawdowns are captured.
    equity_with_base = np.concatenate(([base], equity))
    running_peak = np.maximum.accumulate(equity_with_base)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(
            running_peak != 0.0,
            (equity_with_base - running_peak) / running_peak * 100.0,
            0.0,
        )
    max_dd_pct = float(np.min(dd_pct)) if dd_pct.size else 0.0
    return equity, max_dd_pct


def _derive_r_denominator(losses: np.ndarray, r_basis: str, fixed_risk: Optional[float]) -> Optional[float]:
    if r_basis == "fixed":
        if fixed_risk is None or fixed_risk <= 0:
            return None
        return float(fixed_risk)
    abs_losses = np.abs(losses)
    if abs_losses.size == 0:
        return None
    if r_basis == "median_loss":
        return float(np.median(abs_losses))
    # default avg_loss
    return float(np.mean(abs_losses))


def compute_metrics(
    df: pd.DataFrame,
    r_basis: str = "avg_loss",
    fixed_risk: Optional[float] = None,
    starting_equity: float = 1000.0,
    ending_equity: Optional[float] = None,
) -> StatsResult:
    pnls = _safe_numeric(df.get("profit_loss"))
    if isinstance(df, pd.DataFrame) and ("closure_reason" in df.columns):
        reasons = df["closure_reason"].astype(str).str.lower()
    else:
        reasons = pd.Series([""] * len(df), dtype=str)

    # Keep rows with finite PnL
    mask = pnls.replace([np.inf, -np.inf], np.nan).notna()
    pnls = pnls[mask].astype(float)
    reasons = reasons[mask] if len(reasons) == len(df) else pd.Series([""] * len(pnls))

    # Treat explicit breakeven as zero (already expected in CSV)
    be_mask = reasons.eq("expired_breakeven")
    pnls = pnls.where(~be_mask, 0.0)

    total = int(len(pnls))

    ending_equity_val = float(ending_equity) if ending_equity is not None else None
    resolved_start_equity = float(starting_equity)
    if ending_equity_val is not None:
        resolved_start_equity = ending_equity_val - float(pnls.sum())
        if resolved_start_equity <= 0:
            raise ValueError(
                "Computed starting equity from ending equity is <= 0. Check the selected trades or provide --starting-equity explicitly."
            )

    if resolved_start_equity <= 0:
        raise ValueError("starting_equity must be positive")

    if total == 0:
        return StatsResult(0, 0, 0, 0, 0.0, 0.0, 0.0, None, None, resolved_start_equity, ending_equity_val)

    wins_mask = pnls > 0
    losses_mask = pnls < 0
    be_count = int((pnls == 0).sum())
    wins = int(wins_mask.sum())
    losses = int(losses_mask.sum())
    win_rate_pct = float(wins / total * 100.0)

    expectancy_currency = float(pnls.mean())

    # Drawdown percent from cumulative equity curve with baseline capital
    _, max_dd_pct = _equity_curve_and_max_dd_pct(pnls.to_numpy(), resolved_start_equity)

    # Average R via derived risk denominator
    denom = _derive_r_denominator(pnls[losses_mask].to_numpy(), r_basis, fixed_risk)
    average_r: Optional[float]
    expectancy_r: Optional[float]
    if denom is None or denom <= 0:
        average_r = None
        expectancy_r = None
    else:
        r_values = pnls.to_numpy() / float(denom)
        average_r = float(np.mean(r_values))
        expectancy_r = average_r

    return StatsResult(
        total_trades=total,
        wins=wins,
        losses=losses,
        breakevens=be_count,
        win_rate_pct=win_rate_pct,
        expectancy_currency=expectancy_currency,
        max_drawdown_pct=max_dd_pct,
        average_r=average_r,
        expectancy_r=expectancy_r,
        starting_equity_used=resolved_start_equity,
        ending_equity=ending_equity_val,
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # Fall back to Python csv via pandas engine if needed
        return pd.read_csv(path, engine="python")


def _select_last_trades(df: pd.DataFrame, last: int) -> pd.DataFrame:
    if df is None or df.empty or not last or last <= 0:
        return df
    if "closed_at" in df.columns:
        try:
            s = pd.to_datetime(df["closed_at"], errors="coerce")
            df = df.assign(_closed_dt=s).sort_values("_closed_dt", kind="stable")
            out = df.tail(last).drop(columns=["_closed_dt"])  # keep original cols
            return out
        except Exception:
            pass
    # Fallback: rely on file order
    return df.tail(last)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute expectancy, win rate, drawdown %, and average R from watchdog CSV")
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to watchdog_closed_positions.csv")
    ap.add_argument("--r-basis", type=str, choices=["avg_loss", "median_loss", "fixed"], default="avg_loss",
                    help="How to define 1R. avg_loss=mean loss size; median_loss=median loss size; fixed=--risk-dollar")
    ap.add_argument("--risk-dollar", type=float, default=None, help="Fixed dollar risk per trade when r-basis=fixed")
    ap.add_argument("--starting-equity", type=float, default=1000.0, help="Starting equity for drawdown calc (default 1000)")
    ap.add_argument("--ending-equity", type=float, default=None,
                    help="Optional ending equity; if provided, starting equity is inferred as ending equity minus cumulative PnL")
    ap.add_argument("--last", type=int, default=0, help="Only analyze the most recent N trades (by closed_at if present)")
    ap.add_argument("--json", action="store_true", help="Output metrics as JSON instead of pretty text")

    args = ap.parse_args()
    df = _load_csv(Path(args.csv))
    if int(args.last or 0) > 0:
        df = _select_last_trades(df, int(args.last))
    try:
        res = compute_metrics(
            df,
            r_basis=str(args.r_basis),
            fixed_risk=args.risk_dollar,
            starting_equity=float(args.starting_equity),
            ending_equity=(None if args.ending_equity is None else float(args.ending_equity)),
        )
    except ValueError as exc:
        ap.error(str(exc))

    if args.json:
        print(json.dumps({
            "total_trades": res.total_trades,
            "wins": res.wins,
            "losses": res.losses,
            "breakevens": res.breakevens,
            "win_rate_pct": round(res.win_rate_pct, 4),
            "expectancy_currency": round(res.expectancy_currency, 6),
            "max_drawdown_pct": round(res.max_drawdown_pct, 4),
            "average_r": (None if res.average_r is None else round(res.average_r, 6)),
            "expectancy_r": (None if res.expectancy_r is None else round(res.expectancy_r, 6)),
            "r_basis": args.r_basis,
            "risk_dollar": args.risk_dollar,
            "csv": str(Path(args.csv)),
            "starting_equity_used": round(res.starting_equity_used, 6),
            "ending_equity": (None if res.ending_equity is None else round(res.ending_equity, 6)),
        }, indent=2))
        return

    print("\n=== Watchdog Performance Metrics ===")
    scope = f" (last {int(args.last)} trades)" if int(args.last or 0) > 0 else ""
    print(f"Source CSV: {Path(args.csv)}{scope}")
    print(f"Trades: {res.total_trades}  Wins: {res.wins}  Losses: {res.losses}  Breakevens: {res.breakevens}")
    print(f"Win Rate: {res.win_rate_pct:.2f}%")
    print(f"Expectancy ($/trade): {res.expectancy_currency:.2f}")
    start_label = res.starting_equity_used
    if res.ending_equity is not None:
        print(f"Max Drawdown: {res.max_drawdown_pct:.2f}% (inferred start=${start_label:.2f}, end=${res.ending_equity:.2f})")
    else:
        print(f"Max Drawdown: {res.max_drawdown_pct:.2f}% (start=${start_label:.2f})")
    if res.average_r is None:
        print(f"Average R: n/a (no losses or no risk basis)")
    else:
        print(f"Average R: {res.average_r:.3f}  (basis={args.r_basis}{' fixed='+str(args.risk_dollar) if args.r_basis=='fixed' else ''})")


if __name__ == "__main__":
    main()
