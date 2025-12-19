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

import pandas as pd

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


def _fmt_mult(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    try:
        num = float(value)
    except Exception:
        return "n/a"
    if num >= 100:
        return f"x{num:.0f}"
    if num >= 10:
        return f"x{num:.1f}"
    return f"x{num:.2f}"


def _true_range_last(df: pd.DataFrame) -> Optional[float]:
    """Compute the last candle true range (TR1) using high/low and previous close."""
    if df is None or len(df) < 2:
        return None
    try:
        high = float(df["high"].iloc[-1])
        low = float(df["low"].iloc[-1])
        prev_close = float(df["price"].iloc[-2])
        return float(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    except Exception:
        return None


def _vol_regime_ratios(
    atr7: float, atr21: float, tr1: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """Return (ATR7/ATR21, TR1/ATR7) ratios to help judge volatility regime."""
    atr7_to_21 = (atr7 / atr21) if (atr21 and atr21 > 0) else None
    tr1_to_atr7 = (tr1 / atr7) if (tr1 is not None and atr7 > 0) else None
    return atr7_to_21, tr1_to_atr7


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
    def _dist_atr(m) -> str:
        if not m or atr_raw <= 0:
            return "n/a"
        try:
            entry = float(m.entry_price)
            sl = float(m.stop_loss_price)
            tp = float(m.take_profit_price)
            sl_mult = abs(entry - sl) / atr_raw
            tp_mult = abs(tp - entry) / atr_raw
            return f"SL {sl_mult:.2f}x ATR, TP {tp_mult:.2f}x ATR"
        except Exception:
            return "n/a"
    print(f"Gates: ATR7={_fmt(atr_raw, 2)}{atr_note} | RR long { _rr_gap(long_m) } | RR short { _rr_gap(short_m) }")
    print(f"Distances: long {_dist(long_m)} | short {_dist(short_m)}")
    print(f"ATR multiples: long {_dist_atr(long_m)} | short {_dist_atr(short_m)}")


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

        atr21 = finder._calculate_atr(df, period=21) if len(df) >= 22 else 0.0
        atr7 = float(tech.get("atr") or 0.0)
        tr1 = _true_range_last(df)
        atr7_to_21, tr1_to_atr7 = _vol_regime_ratios(atr7, atr21, tr1)
        print(f"ATR7={_fmt(tech.get('atr'), 2)}  daily_vol_30d={_fmt(tech.get('daily_vol_30d'), 4)}  "
              f"intraday_range_pos={_fmt(tech.get('intraday_range_position'), 3)}  "
              f"intraday_vol_6h={_fmt(tech.get('intraday_volatility_6h'), 4)}  "
              f"spread_bps={_fmt(coin.get('spread_bps'), 3)}")
        print(
            f"Vol regime: ATR21={_fmt(atr21, 2)}  ATR7/ATR21={_fmt(atr7_to_21, 2)}  TR1/ATR7={_fmt(tr1_to_atr7, 2)}"
        )
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

    min_volume = float(getattr(cfg, "min_volume_24h", 0.0) or 0.0)
    min_ratio = float(getattr(cfg, "min_volume_market_cap_ratio", 0.0) or 0.0)
    major_symbols = {
        "BTC", "ETH", "SOL", "XRP", "USDT", "USDC",
        "ADA", "AVAX", "LINK", "DOGE", "LTC", "DOT", "MATIC",
    }

    def _spread_cap_bps(volume_usd: float) -> float:
        """Heuristic 'acceptable' spread cap for reporting (not a hard gate)."""
        if volume_usd >= 1e9:
            return 3.0
        if volume_usd >= 1e8:
            return 5.0
        return 10.0

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
        atr21 = finder._calculate_atr(df, period=21) if len(df) >= 22 else 0.0
        tr1 = _true_range_last(df)
        atr7_to_21, tr1_to_atr7 = _vol_regime_ratios(atr, atr21, tr1)
        atr_bps = (atr / price * 10_000) if price > 0 else None
        cap_bps = (cap / price * 10_000) if (cap and price > 0) else None
        headroom_bps = (cap_bps - atr_bps) if (cap_bps is not None and atr_bps is not None) else None

        try:
            vol24h = float(coin.get("volume_24h") or 0.0)
        except Exception:
            vol24h = 0.0
        try:
            market_cap = float(coin.get("market_cap") or 0.0)
        except Exception:
            market_cap = 0.0
        ratio = (vol24h / market_cap) if market_cap > 0 else None
        ratio_gap_pp = ((ratio - min_ratio) * 100.0) if (ratio is not None and min_ratio > 0) else None
        vol_mult = (vol24h / min_volume) if (min_volume > 0 and vol24h >= 0) else None

        spread_bps = None
        try:
            spread_bps = float(coin.get("spread_bps")) if coin.get("spread_bps") is not None else None
        except Exception:
            spread_bps = None
        spread_cap = _spread_cap_bps(vol24h) if vol24h > 0 else 10.0
        spread_headroom = (spread_cap - spread_bps) if (spread_bps is not None and spread_bps > 0) else None

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
            "atr7_to_21": atr7_to_21,
            "tr1_to_atr7": tr1_to_atr7,
            "vol24h": vol24h,
            "market_cap": market_cap,
            "vmc_ratio": ratio,
            "vmc_gap_pp": ratio_gap_pp,
            "vol_mult": vol_mult,
            "spread_bps": spread_bps,
            "spread_cap_bps": spread_cap,
            "spread_headroom_bps": spread_headroom,
            "vmc_exempt": (coin.get("symbol") or "").upper() in major_symbols,
            "volume_source": str(coin.get("volume_24h_source") or "").strip(),
        })

    rows.sort(key=lambda r: (r["rr_gap"] if r["rr_gap"] is not None else 1e9,
                             -(r["headroom_bps"] if r["headroom_bps"] is not None else -1e9)))
    print(f"Top {min(top, len(rows))} closest to RR {rr_target}:")
    for row in rows[:top]:
        hr = row["headroom_bps"]
        cap_txt = f"{row['cap_bps']:.0f} bps" if row['cap_bps'] is not None else "n/a"
        atr_txt = f"{row['atr_bps']:.0f} bps" if row['atr_bps'] is not None else "n/a"
        if hr is None:
            atr_gate_txt = "ATR cap n/a"
        elif hr < 0:
            atr_gate_txt = f"ATR CLIPPED (over cap by {abs(hr):.0f} bps)"
        else:
            atr_gate_txt = f"ATR within cap (+{hr:.0f} bps headroom)"

        # Liquidity + spread context (informational)
        vol_txt = _fmt_usd_compact(row.get("vol24h"))
        liq_mult = _fmt_mult(row.get("vol_mult")) if min_volume > 0 else "n/a"
        vmc_ratio = row.get("vmc_ratio")
        if vmc_ratio is None:
            vmc_txt = "n/a"
            vmc_gap_txt = "n/a"
        else:
            vmc_txt = f"{vmc_ratio * 100.0:.2f}%"
            if min_ratio > 0 and row.get("vmc_gap_pp") is not None:
                vmc_gap_txt = f"{row['vmc_gap_pp']:+.2f}pp"
                if row.get("vmc_exempt") and row["vmc_gap_pp"] < 0:
                    vmc_gap_txt += " (exempt)"
            else:
                vmc_gap_txt = "n/a"

        spr = row.get("spread_bps")
        if spr is None or spr <= 0:
            spr_txt = "n/a"
        else:
            spr_cap = float(row.get("spread_cap_bps") or 0.0)
            spr_hr = row.get("spread_headroom_bps")
            spr_hr_txt = f"{spr_hr:+.2f} bps" if spr_hr is not None else "n/a"
            spr_txt = f"{spr:.2f} bps ({spr_hr_txt}; cap={spr_cap:.0f})"

        print(f"{row['symbol']} ({row['product']}) {row['best_side']} RR={row['rr']:.2f} (gap {row['rr_gap']:.2f}) | "
              f"ATR {atr_txt}, cap {cap_txt}, {atr_gate_txt} | "
              f"liq vol={vol_txt} ({liq_mult} vs min={_fmt_usd_compact(min_volume)}) "
              f"vmc={vmc_txt} ({vmc_gap_txt} vs {min_ratio * 100:.1f}%) | "
              f"spr={spr_txt} | "
              f"ATR7/ATR21={_fmt(row.get('atr7_to_21'), 2)} TR1/ATR7={_fmt(row.get('tr1_to_atr7'), 2)}")

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
