#!/usr/bin/env python3
"""
Quick snapshot helper for one or more symbols.

Fetches the raw finder metrics for the requested symbols (LONG and SHORT)
and prints entry/stop/TP plus risk/reward even if they wouldn't clear the
normal filters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from short_term_crypto_finder import (
    PROFILE_PRESETS,
    ShortTermCryptoFinder,
    build_short_term_config,
)


def apply_profile_overrides(cfg, profile: str) -> None:
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


def _print_side(label: str, metric) -> None:
    if not metric:
        print(f"{label}: n/a")
        return

    def _price_prec(entry: float) -> int:
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

    price_prec = _price_prec(metric.entry_price)
    print(f"{label}: RR={_fmt(metric.risk_reward_ratio, 2)}  "
          f"entry={_fmt(metric.entry_price, price_prec)}  "
          f"SL={_fmt(metric.stop_loss_price, price_prec)}  "
          f"TP={_fmt(metric.take_profit_price, price_prec)}  "
          f"RSI14={_fmt(metric.rsi_14, 1)}  "
          f"trend={_fmt(metric.trend_strength, 3)}%/d  "
          f"mom={_fmt(metric.momentum_score, 2)}")


def _print_gates(
    long_m,
    short_m,
    tech: Dict,
    rr_target: float,
    atr_cap_usd: Optional[float],
    atr_cap_bps: Optional[float],
    price: Optional[float],
) -> None:
    atr_raw = float(tech.get("atr") or 0.0)
    atr_note = ""
    # Tiered dynamic bps (mirror finder logic)
    def _dyn_bps(p: float) -> float:
        if p >= 20000:
            return 325.0
        if p >= 2000:
            return 350.0
        if p >= 200:
            return 400.0
        return 450.0
    caps: List[float] = []
    if atr_cap_usd and atr_cap_usd > 0:
        caps.append(float(atr_cap_usd))
    if price and price > 0:
        tier_bps = _dyn_bps(price)
        eff_bps = None
        if atr_cap_bps and atr_cap_bps > 0:
            eff_bps = min(float(atr_cap_bps), tier_bps)
        else:
            eff_bps = tier_bps
        caps.append(float(price) * float(eff_bps) / 10000.0)
    if caps:
        cap_val = min(caps)
        headroom = cap_val - atr_raw
        if price and price > 0:
            headroom_bps = headroom / price * 10_000
            cap_bps = cap_val / price * 10_000
            if headroom_bps > 5000:  # >50% of price
                atr_note = " | ATR cap not binding"
            else:
                atr_note = f" | ATR headroom to cap: {headroom:+.2f} ({headroom_bps:+.0f} bps; cap={cap_bps:.0f} bps)"
        else:
            atr_note = f" | ATR headroom to cap: {headroom:+.2f}"
    def _rr_gap(m) -> str:
        if not m:
            return "n/a"
        try:
            rr = float(m.risk_reward_ratio)
            gap = rr_target - rr
            return f"{rr:.2f} (needs {gap:+.2f} to hit {rr_target:.1f})"
        except Exception:
            return "n/a"
    def _dist(m) -> str:
        if not m:
            return "n/a"
        try:
            entry = float(m.entry_price)
            sl = float(m.stop_loss_price)
            tp = float(m.take_profit_price)
            risk_pct = abs(entry - sl) / entry * 100
            reward_pct = abs(tp - entry) / entry * 100
            return f"risk {risk_pct:.2f}%, reward {reward_pct:.2f}%"
        except Exception:
            return "n/a"
    print(f"Gates: ATR7={_fmt(atr_raw, 2)}{atr_note} | RR long { _rr_gap(long_m) } | RR short { _rr_gap(short_m) }")
    print(f"Distances: long {_dist(long_m)} | short {_dist(short_m)}")


def _dynamic_bps(price: float) -> float:
    if price >= 20000:
        return 325.0
    if price >= 2000:
        return 350.0
    if price >= 200:
        return 400.0
    return 450.0


def _effective_atr_cap(price: float, atr_cap_usd: Optional[float], atr_cap_bps: Optional[float]) -> Optional[float]:
    caps: List[float] = []
    if atr_cap_usd and atr_cap_usd > 0:
        caps.append(float(atr_cap_usd))
    if price and price > 0:
        dyn_bps = _dynamic_bps(price)
        eff_bps = None
        if atr_cap_bps and atr_cap_bps > 0:
            eff_bps = min(float(atr_cap_bps), dyn_bps)
        else:
            eff_bps = dyn_bps
        caps.append(float(price) * float(eff_bps) / 10000.0)
    return min(caps) if caps else None


def snapshot_symbols(symbols: Iterable[str], profile: str, disable_liquidity: bool) -> None:
    cfg = build_short_term_config()
    apply_profile_overrides(cfg, profile)
    cfg.symbols = list(symbols)
    cfg.force_refresh_candles = True
    if disable_liquidity:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0

    finder = ShortTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=None, symbols=cfg.symbols)
    if not coins:
        print("No symbols retrieved (check connectivity or liquidity filters).")
        return

    for coin in coins:
        product_id = coin["product_id"]
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            print(f"{coin['symbol']}: no historical data")
            continue

        tech = finder.calculate_technical_indicators(df)
        mom = finder.calculate_momentum_score(df)
        chg = finder._calculate_price_changes_from_history(df)
        long_m = finder._build_long_metrics(coin, df, tech, mom, chg)
        short_m = finder._build_short_metrics(coin, df, tech, mom, chg)

        print("=" * 80)
        print(f"{coin['symbol']} ({coin.get('name','n/a')})  product={product_id}")
        vol = coin.get('volume_24h')
        mc = coin.get('market_cap')
        cg_warn = False
        if (vol is None or vol == 0) or (mc is None or mc == 0):
            cg_warn = True
        print(f"Price={_fmt(coin.get('current_price'), 2)}  "
              f"Vol24h={_fmt(vol, 0)}  "
              f"MCAP={_fmt(mc, 0)}  "
              f"Rank={coin.get('market_cap_rank', 'n/a')}")
        if cg_warn:
            print("WARNING: Missing/zero volume or market cap (CoinGecko/MC feed unavailable); liquidity checks may be incomplete.")
        ts = coin.get("data_timestamp_utc") or getattr(long_m, "data_timestamp_utc", "")
        if ts:
            print(f"Data TS (UTC): {ts}")
        print(f"ATR7={_fmt(tech.get('atr'), 2)}  daily_vol_30d={_fmt(tech.get('daily_vol_30d'), 4)}  "
              f"intraday_range_pos={_fmt(tech.get('intraday_range_position'), 3)}  "
              f"intraday_vol_6h={_fmt(tech.get('intraday_volatility_6h'), 4)}  "
              f"spread_bps={_fmt(coin.get('spread_bps'), 3)}")
        _print_gates(
            long_m,
            short_m,
            tech,
            rr_target=2.0,
            atr_cap_usd=getattr(cfg, "max_atr_usd", None),
            atr_cap_bps=getattr(cfg, "max_atr_bps", None),
            price=coin.get("current_price"),
        )
        _print_side("LONG", long_m)
        _print_side("SHORT", short_m)
        print()


def gate_scan(
    profile: str,
    disable_liquidity: bool,
    top: int,
    rr_target: float,
    scan_limit: Optional[int],
) -> None:
    cfg = build_short_term_config()
    apply_profile_overrides(cfg, profile)
    cfg.symbols = None  # scan the profile universe
    cfg.force_refresh_candles = True
    if disable_liquidity:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0

    finder = ShortTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=scan_limit, symbols=None)
    if not coins:
        print("No symbols retrieved (check connectivity or liquidity filters).")
        return

    rows = []
    for coin in coins:
        product_id = coin["product_id"]
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            continue
        tech = finder.calculate_technical_indicators(df)
        mom = finder.calculate_momentum_score(df)
        chg = finder._calculate_price_changes_from_history(df)
        long_m = finder._build_long_metrics(coin, df, tech, mom, chg)
        short_m = finder._build_short_metrics(coin, df, tech, mom, chg)
        price = float(coin.get("current_price") or 0.0)
        cap = _effective_atr_cap(price, getattr(cfg, "max_atr_usd", None), getattr(cfg, "max_atr_bps", None))
        atr = float(tech.get("atr") or 0.0)
        atr_bps = (atr / price * 10_000) if price > 0 else None
        cap_bps = (cap / price * 10_000) if (cap and price > 0) else None
        headroom_bps = (cap_bps - atr_bps) if (cap_bps is not None and atr_bps is not None) else None
        def _rr_gap(m):
            if not m:
                return None
            try:
                rr = float(m.risk_reward_ratio)
                return max(0.0, rr_target - rr)
            except Exception:
                return None
        gaps = [
            ("LONG", _rr_gap(long_m), getattr(long_m, "risk_reward_ratio", None)),
            ("SHORT", _rr_gap(short_m), getattr(short_m, "risk_reward_ratio", None)),
        ]
        gaps = [g for g in gaps if g[1] is not None]
        if not gaps:
            continue
        gaps.sort(key=lambda x: x[1])
        best_side, best_gap, best_rr = gaps[0]
        rows.append({
            "symbol": coin["symbol"],
            "product": product_id,
            "best_side": best_side,
            "rr_gap": best_gap,
            "rr": best_rr,
            "headroom_bps": headroom_bps,
            "atr_bps": atr_bps,
            "cap_bps": cap_bps,
        })

    rows.sort(key=lambda r: (r["rr_gap"] if r["rr_gap"] is not None else 1e9,
                             -(r["headroom_bps"] if r["headroom_bps"] is not None else -1e9)))
    print(f"Top {min(top, len(rows))} closest to RR {rr_target}:")
    for row in rows[:top]:
        hr = row["headroom_bps"]
        hr_txt = "n/a"
        if hr is not None:
            hr_txt = f"{hr:+.0f} bps"
        cap_txt = f"{row['cap_bps']:.0f} bps" if row['cap_bps'] is not None else "n/a"
        atr_txt = f"{row['atr_bps']:.0f} bps" if row['atr_bps'] is not None else "n/a"
        print(f"{row['symbol']} ({row['product']}) {row['best_side']} RR={row['rr']:.2f} (gap {row['rr_gap']:.2f}) | "
              f"ATR {atr_txt}, cap {cap_txt}, headroom {hr_txt}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Print snapshot metrics for specific symbols.")
    parser.add_argument(
        "--symbols",
        required=False,
        help="Comma-separated symbols (e.g., BTC,ETH).",
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
        help="Scan entire profile universe and print top symbols closest to RR/ATR gates (ignores --symbols).",
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
        help="Limit how many symbols to scan in gate-scan mode (e.g., 100 or 200). Default scans full profile universe.",
    )
    args = parser.parse_args()

    if args.gate_scan:
        gate_scan(
            profile=args.profile,
            disable_liquidity=args.no_liquidity_filter,
            top=args.top,
            rr_target=args.rr_target,
            scan_limit=args.scan_limit,
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
