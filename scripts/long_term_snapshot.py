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


STABLE_SYMBOLS = {
    "USDT",
    "USDC",
    "USD1",
    "EURC",
    "DAI",
    "TUSD",
    "USDP",
    "GUSD",
    "LUSD",
    "USDD",
    "FRAX",
    "USDE",
}


def apply_profile_overrides(cfg: CryptoFinderConfig, profile: str) -> None:
    preset = PROFILE_PRESETS.get(profile, {})
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _profile_limit(profile: str) -> Optional[int]:
    raw = (PROFILE_PRESETS.get(profile) or {}).get("limit")
    try:
        val = int(raw)
    except Exception:
        return None
    return val if val > 0 else None


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


def _is_stable_symbol(symbol: str) -> bool:
    return symbol.strip().upper() in STABLE_SYMBOLS


def _risk_rank(risk_level: str) -> int:
    order = {
        "LOW": 0,
        "MEDIUM_LOW": 1,
        "MEDIUM": 2,
        "MEDIUM_HIGH": 3,
        "HIGH": 4,
        "VERY_HIGH": 5,
    }
    return int(order.get((risk_level or "").strip().upper(), 99))


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


def gate_scan(
    profile: str,
    disable_liquidity: bool,
    top: int,
    rr_target: float,
    scan_limit: Optional[int],
    include_stables: bool,
) -> None:
    cfg = CryptoFinderConfig.from_env()
    apply_profile_overrides(cfg, profile)
    cfg.symbols = None
    cfg.force_refresh_candles = True
    if disable_liquidity:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0

    effective_limit = scan_limit if scan_limit is not None else _profile_limit(profile)

    finder = LongTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=effective_limit, symbols=None)
    if not coins:
        print("No symbols retrieved (check connectivity or liquidity filters).")
        return

    rows = []
    for coin in coins:
        sym = str(coin.get("symbol") or "").upper()
        if (not include_stables) and _is_stable_symbol(sym):
            continue

        product_id = coin["product_id"]
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            continue

        tech = finder.calculate_technical_indicators(df)
        mom = finder.calculate_momentum_score(df)
        chg = finder._calculate_price_changes_from_history(df)
        long_m = finder._build_long_metrics(coin, df, tech, mom, chg)
        short_m = finder._build_short_metrics(coin, df, tech, mom, chg)

        candidates = []
        if long_m is not None:
            try:
                rr = float(long_m.risk_reward_ratio)
                candidates.append(("LONG", rr, max(0.0, rr_target - rr), str(long_m.risk_level or "n/a")))
            except Exception:
                pass
        if short_m is not None:
            try:
                rr = float(short_m.risk_reward_ratio)
                candidates.append(("SHORT", rr, max(0.0, rr_target - rr), str(short_m.risk_level or "n/a")))
            except Exception:
                pass
        if not candidates:
            continue

        best_side, best_rr, best_gap, best_risk = sorted(candidates, key=lambda t: (t[2], -t[1]))[0]

        try:
            price = float(coin.get("current_price") or 0.0)
        except Exception:
            price = 0.0
        atr = float(tech.get("atr") or 0.0)
        atr_pct = (atr / price * 100.0) if price > 0 else 0.0

        try:
            vol24h = float(coin.get("volume_24h") or 0.0)
        except Exception:
            vol24h = 0.0
        try:
            market_cap = float(coin.get("market_cap") or 0.0)
        except Exception:
            market_cap = 0.0
        vmc_ratio = (vol24h / market_cap) if (vol24h > 0 and market_cap > 0) else None

        spread_bps = None
        try:
            spread_bps = float(coin.get("spread_bps")) if coin.get("spread_bps") is not None else None
        except Exception:
            spread_bps = None

        rows.append(
            {
                "symbol": sym,
                "product": product_id,
                "best_side": best_side,
                "rr": best_rr,
                "rr_gap": best_gap,
                "risk": best_risk,
                "risk_rank": _risk_rank(best_risk),
                "atr_pct": atr_pct,
                "daily_vol": float(tech.get("daily_vol_30d") or 0.0),
                "sharpe": float(tech.get("sharpe_ratio") or 0.0),
                "max_dd": float(tech.get("max_drawdown") or 0.0),
                "vol24h": vol24h,
                "vmc_ratio": vmc_ratio,
                "spread_bps": spread_bps,
            }
        )

    rows.sort(
        key=lambda r: (
            r["rr_gap"],
            r["risk_rank"],
            (r["spread_bps"] if (r["spread_bps"] is not None and r["spread_bps"] > 0) else 1e9),
            -r["vol24h"],
        )
    )
    print(f"Top {min(top, len(rows))} closest to RR {rr_target}:")
    for row in rows[:top]:
        spr = row["spread_bps"]
        spr_txt = "n/a" if spr is None or spr <= 0 else f"{spr:.2f} bps"
        vmc = row["vmc_ratio"]
        vmc_txt = "n/a" if vmc is None else f"{vmc * 100.0:.2f}%"
        print(
            f"{row['symbol']} ({row['product']}) {row['best_side']} "
            f"RR={row['rr']:.2f} (gap {row['rr_gap']:.2f}) | "
            f"ATR{cfg.atr_period}={row['atr_pct']:.2f}% | risk={row['risk']} | "
            f"vol24h={_fmt_usd_compact(row['vol24h'])} vmc={vmc_txt} | "
            f"Sharpe={row['sharpe']:.2f} DD={row['max_dd']*100.0:.1f}% | spr={spr_txt}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print snapshot metrics for specific symbols using the long-term finder."
    )
    parser.add_argument(
        "--symbols",
        required=False,
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
    parser.add_argument(
        "--gate-scan",
        action="store_true",
        help="Scan the profile universe and print top symbols closest to RR target (ignores --symbols).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many symbols to show in gate-scan mode (default: 15).",
    )
    parser.add_argument(
        "--rr-target",
        type=float,
        default=2.0,
        help="RR target used for gate-scan gap calculations (default: 2.0).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=None,
        help="Limit how many symbols to scan in gate-scan mode (e.g., 100 or 200). Default uses profile limit (if any).",
    )
    parser.add_argument(
        "--include-stables",
        action="store_true",
        help="Include stablecoins in gate-scan output (default: excluded).",
    )
    args = parser.parse_args()

    if args.gate_scan:
        gate_scan(
            profile=args.profile,
            disable_liquidity=args.no_liquidity_filter,
            top=args.top,
            rr_target=args.rr_target,
            scan_limit=args.scan_limit,
            include_stables=args.include_stables,
        )
        return

    if not args.symbols:
        parser.error("No symbols provided after parsing.")
    syms = _parse_symbols(args.symbols)
    if not syms:
        parser.error("No symbols provided after parsing.")
    snapshot_symbols(syms, profile=args.profile, disable_liquidity=args.no_liquidity_filter)


if __name__ == "__main__":
    main()
