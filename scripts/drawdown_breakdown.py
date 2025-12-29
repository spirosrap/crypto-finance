#!/usr/bin/env python3
"""Summarize recent drawdowns for paper and live trade logs."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from short_term_crypto_finder import ShortTermCryptoFinder, build_short_term_config, PROFILE_PRESETS


def apply_profile_overrides(cfg, profile: str) -> None:
    preset = PROFILE_PRESETS.get(profile, {})
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def atr_series(df: pd.DataFrame, period: int = 7) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if len(df) < period + 1:
        return pd.Series([0.0] * len(df), index=df.index)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["price"].to_numpy(dtype=float)
    tr = np.r_[0.0, np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])]
    atr = np.empty_like(tr)
    atr[:] = np.nan
    atr[period] = tr[1:period + 1].sum()
    for idx in range(period + 1, len(tr)):
        atr[idx] = atr[idx - 1] - atr[idx - 1] / period + tr[idx]
    out = pd.Series(atr / period, index=df.index)
    return out.fillna(0.0)


def _dynamic_bps(price: float) -> float:
    if price >= 20000:
        return 325.0
    if price >= 2000:
        return 350.0
    if price >= 200:
        return 400.0
    return 450.0


def effective_cap_bps(price: float, atr_cap_usd: Optional[float], atr_cap_bps: Optional[float]) -> Optional[float]:
    caps: list[float] = []
    if atr_cap_usd and atr_cap_usd > 0:
        caps.append(atr_cap_usd / price * 10_000)
    tier = _dynamic_bps(price)
    if atr_cap_bps and atr_cap_bps > 0:
        caps.append(min(atr_cap_bps, tier))
    else:
        caps.append(tier)
    return min(caps) if caps else None


def spread_cap_bps(volume_usd: float) -> float:
    if volume_usd >= 1e9:
        return 3.0
    if volume_usd >= 1e8:
        return 5.0
    return 10.0


def bucket_atr_ratio(ratio: float | None) -> str:
    if ratio is None or not np.isfinite(ratio):
        return "n/a"
    if ratio <= 1.0:
        return "<=1.0"
    if ratio <= 1.25:
        return "1.0-1.25"
    if ratio <= 1.5:
        return "1.25-1.5"
    if ratio <= 2.0:
        return "1.5-2.0"
    return ">2.0"


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["opened_at"] = pd.to_datetime(df["opened_at"], utc=True, errors="coerce")
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["base"] = df["product_id"].astype(str).str.split("-").str[0]
    return df


def annotate_metrics(df: pd.DataFrame, finder: ShortTermCryptoFinder, coin_map: dict, cfg) -> pd.DataFrame:
    atr_cache: dict[str, pd.Series] = {}
    for base in df["base"].unique():
        product_id = f"{base}-USDC"
        hist = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if hist is None or hist.empty:
            continue
        atr_cache[base] = atr_series(hist, period=cfg.atr_period)

    def _atr_at_entry(row: pd.Series) -> float:
        atr_ser = atr_cache.get(row["base"])
        if atr_ser is None or atr_ser.empty or pd.isna(row["opened_at"]):
            return np.nan
        idx = atr_ser.index.get_indexer([row["opened_at"]], method="pad")
        if idx.size == 0 or idx[0] < 0:
            return np.nan
        return float(atr_ser.iloc[idx[0]])

    df = df.copy()
    df["atr_entry"] = df.apply(_atr_at_entry, axis=1)
    df["atr_bps_entry"] = (df["atr_entry"] / df["entry_price"]) * 10_000
    df["cap_bps_entry"] = df["entry_price"].apply(
        lambda p: effective_cap_bps(float(p), getattr(cfg, "max_atr_usd", None), getattr(cfg, "max_atr_bps", None))
    )
    df["atr_ratio_entry"] = df["atr_bps_entry"] / df["cap_bps_entry"]
    df["atr_bucket"] = df["atr_ratio_entry"].apply(bucket_atr_ratio)

    spreads = []
    spread_caps = []
    spread_headroom = []
    for _, row in df.iterrows():
        coin = coin_map.get(row["base"])
        if not coin:
            spreads.append(np.nan)
            spread_caps.append(np.nan)
            spread_headroom.append(np.nan)
            continue
        spread_bps = coin.get("spread_bps")
        vol = coin.get("volume_24h") or 0.0
        cap = spread_cap_bps(float(vol)) if vol else np.nan
        spreads.append(float(spread_bps) if spread_bps is not None else np.nan)
        spread_caps.append(cap)
        if spread_bps is None or cap != cap:
            spread_headroom.append(np.nan)
        else:
            spread_headroom.append(cap - float(spread_bps))
    df["spread_bps_now"] = spreads
    df["spread_cap_now"] = spread_caps
    df["spread_headroom_now"] = spread_headroom
    df["spread_ok_now"] = df["spread_headroom_now"] >= 0
    return df


def summarize(label: str, df: pd.DataFrame, hours: float) -> None:
    df = df.dropna(subset=["closed_at"])
    if df.empty:
        print(f"\n== {label} ==\nNo rows with valid closed_at.")
        return
    max_ts = df["closed_at"].max()
    window_start = max_ts - timedelta(hours=hours)
    recent = df.loc[df["closed_at"] >= window_start].copy()
    print(f"\n== {label} | last {hours:.0f}h ({window_start} -> {max_ts}) ==")
    if recent.empty:
        print("No rows in window.")
        return
    wins = (recent["profit_loss"] > 0).sum()
    losses = (recent["profit_loss"] < 0).sum()
    flats = (recent["profit_loss"] == 0).sum()
    print(f"rows={len(recent)} win={wins} loss={losses} flat={flats}")
    print(f"pnl_sum={recent['profit_loss'].sum():.2f} avg={recent['profit_loss'].mean():.2f} med={recent['profit_loss'].median():.2f}")
    print("closure_reason counts:")
    print(recent["closure_reason"].value_counts().to_string())

    print("\nTop loss symbols (last window):")
    sym = recent.groupby("product_id")["profit_loss"].sum().sort_values().head(5)
    print(sym.to_string())

    print("\nLosses by close hour (UTC, last window):")
    recent["close_hour"] = recent["closed_at"].dt.hour
    loss_hours = recent.loc[recent["profit_loss"] < 0].groupby("close_hour")["profit_loss"].sum().sort_values()
    print(loss_hours.to_string())

    print("\nClosure reason vs ATR bucket (entry):")
    print(pd.crosstab(recent["closure_reason"], recent["atr_bucket"]).to_string())

    print("\nStop-loss rate by ATR bucket (entry):")
    stop_rate = recent.assign(stop=lambda d: d["closure_reason"].eq("stop_loss")).groupby("atr_bucket")["stop"].mean()
    print(stop_rate.sort_index().to_string())

    print("\nStop-loss rate by current spread OK?:")
    spread_rate = recent.assign(stop=lambda d: d["closure_reason"].eq("stop_loss")).groupby("spread_ok_now")["stop"].mean()
    print(spread_rate.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Breakdown recent drawdown behavior for paper and live logs.")
    parser.add_argument("--paper", default="trade_logs/paper_finder_closed_positions.csv", help="Paper closed trades CSV.")
    parser.add_argument("--live", default="trade_logs/watchdog_closed_positions.csv", help="Live closed trades CSV.")
    parser.add_argument("--hours", type=float, default=24.0, help="Lookback window in hours (default: 24).")
    parser.add_argument("--profile", default="focused_no_llm_100", choices=sorted(PROFILE_PRESETS.keys()))
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh of candles (slower).")
    parser.add_argument("--no-liquidity-filter", action="store_true", help="Disable liquidity filters for coin lookup.")
    args = parser.parse_args()

    cfg = build_short_term_config()
    apply_profile_overrides(cfg, args.profile)
    if args.no_liquidity_filter:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0
    cfg.force_refresh_candles = bool(args.force_refresh)

    finder = ShortTermCryptoFinder(config=cfg)

    paper_df = load_trades(Path(args.paper))
    live_df = load_trades(Path(args.live))

    symbols = sorted(set(paper_df["base"]).union(set(live_df["base"])))
    cfg.symbols = symbols
    coins = finder.get_cryptocurrencies_to_analyze(limit=None, symbols=symbols)
    coin_map = {str(c.get("symbol") or "").upper(): c for c in coins}

    paper_df = annotate_metrics(paper_df, finder, coin_map, cfg)
    live_df = annotate_metrics(live_df, finder, coin_map, cfg)

    print("Note: spread_ok_now is based on current spreads, not entry-time spreads.")
    summarize("paper", paper_df, args.hours)
    summarize("live", live_df, args.hours)


if __name__ == "__main__":
    main()
