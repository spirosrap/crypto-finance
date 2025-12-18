#!/usr/bin/env python3
"""
Quick snapshot helper for one or more symbols using LongTermCryptoFinder.

This script is intended for multi-day to multi-week swing ideas. It uses the
long-term finder data/indicators (daily candles, ATR(14) by default, Sharpe,
drawdown, etc) rather than the short-term intraday metrics printed by
`scripts/symbol_snapshot.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from long_term_crypto_finder import (  # noqa: E402
    PROFILE_PRESETS,
    CryptoFinderConfig,
    LongTermCryptoFinder,
)


def apply_profile_overrides(cfg: CryptoFinderConfig, profile: str) -> None:
    preset = PROFILE_PRESETS.get(profile, {})
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _fmt(value, precision: int = 2) -> str:
    try:
        return f"{float(value):.{precision}f}"
    except Exception:
        return "n/a"


def _fmt_usd_compact(value: Optional[float]) -> str:
    """Format a USD-ish number as 12.3K/4.5M/6.7B/1.2T."""
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except Exception:
        return "n/a"
    num = abs(num)
    if not (num >= 0):  # NaN
        return "n/a"
    for unit, denom in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if num >= denom:
            return f"{num / denom:.2f}{unit}"
    return f"{num:.0f}"


def _price_precision(entry: float) -> int:
    """Return a reasonable decimal precision for printing prices."""
    try:
        val = float(entry)
    except Exception:
        return 4
    if val < 1:
        return 6
    if val < 10:
        return 4
    if val < 1000:
        return 3
    return 2


def _format_side(label: str, metric) -> str:
    if not metric:
        return f"{label}: n/a"

    price_prec = _price_precision(getattr(metric, "entry_price", 0.0))
    risk_level = getattr(metric, "risk_level", None) or "n/a"
    return (
        f"{label}: RR={_fmt(getattr(metric, 'risk_reward_ratio', None), 2)}  "
        f"entry={_fmt(getattr(metric, 'entry_price', None), price_prec)}  "
        f"SL={_fmt(getattr(metric, 'stop_loss_price', None), price_prec)}  "
        f"TP={_fmt(getattr(metric, 'take_profit_price', None), price_prec)}  "
        f"RSI={_fmt(getattr(metric, 'rsi_14', None), 1)}  "
        f"trend={_fmt(getattr(metric, 'trend_strength', None), 3)}%/d  "
        f"mom={_fmt(getattr(metric, 'momentum_score', None), 2)}  "
        f"risk={risk_level}"
    )


def snapshot_symbols(symbols: Iterable[str], profile: str, disable_liquidity: bool) -> None:
    cfg = CryptoFinderConfig.from_env()
    apply_profile_overrides(cfg, profile)
    cfg.symbols = list(symbols)
    cfg.force_refresh_candles = True
    if disable_liquidity:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0

    finder = LongTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=max(len(cfg.symbols), 1), symbols=cfg.symbols)
    if not coins:
        print("No symbols retrieved (check connectivity or liquidity filters).")
        return

    for coin in coins:
        product_id = coin["product_id"]
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            print(f"{coin.get('symbol', product_id)}: no historical data")
            continue

        tech = finder.calculate_technical_indicators(df)
        mom = finder.calculate_momentum_score(df)
        chg = finder._calculate_price_changes_from_history(df)
        long_m = finder._build_long_metrics(coin, df, tech, mom, chg)
        short_m = finder._build_short_metrics(coin, df, tech, mom, chg)

        price = float(coin.get("current_price") or 0.0)
        atr = float(tech.get("atr") or 0.0)
        atr_pct = (atr / price * 100.0) if price > 0 else 0.0

        print("=" * 80)
        print(f"{coin.get('symbol', 'n/a')} ({coin.get('name', 'n/a')})  product={product_id}")
        print(
            f"Price={_fmt(price, 2)}  "
            f"Vol24h={_fmt_usd_compact(coin.get('volume_24h'))}  "
            f"MCAP={_fmt_usd_compact(coin.get('market_cap'))}  "
            f"Rank={coin.get('market_cap_rank', 'n/a')}"
        )
        ts = coin.get("data_timestamp_utc") or getattr(long_m, "data_timestamp_utc", "")
        if ts:
            print(f"Data TS (UTC): {ts}")

        print(
            f"ATR{cfg.atr_period}={_fmt(atr, 2)} ({_fmt(atr_pct, 2)}%)  "
            f"daily_vol_30d={_fmt(tech.get('daily_vol_30d'), 4)}  "
            f"Sharpe={_fmt(tech.get('sharpe_ratio'), 2)}  "
            f"MaxDD={_fmt(float(tech.get('max_drawdown') or 0.0) * 100.0, 2)}%  "
            f"spread_bps={_fmt(coin.get('spread_bps'), 3)}"
        )
        print(_format_side("LONG", long_m))
        print(_format_side("SHORT", short_m))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print snapshot metrics for specific symbols using the long-term finder."
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma-separated symbols (e.g., BTC,ETH,SOL).",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=sorted(PROFILE_PRESETS.keys()),
        help="Finder profile to apply (default: default).",
    )
    parser.add_argument(
        "--no-liquidity-filter",
        action="store_true",
        help="Disable liquidity filters (min_volume, volume/market-cap ratio).",
    )
    args = parser.parse_args()

    syms = _parse_symbols(args.symbols)
    if not syms:
        parser.error("No symbols provided after parsing.")
    snapshot_symbols(syms, profile=args.profile, disable_liquidity=args.no_liquidity_filter)


if __name__ == "__main__":
    main()

