#!/usr/bin/env python3
"""
Build a baseline finder file from snapshot-style metrics.

This helper lets you pick symbols (and sides) from symbol_snapshot output,
compute baseline ATR-based SL/TP levels, and optionally open paper trades.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPEN_PAPER_CSV = REPO_ROOT / "trade_logs" / "paper_finder_open_positions.csv"

from short_term_crypto_finder import PROFILE_PRESETS, ShortTermCryptoFinder, build_short_term_config


def apply_profile_overrides(cfg, profile: str) -> None:
    preset = PROFILE_PRESETS.get(profile, {})
    for key, value in preset.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def _normalize_side(raw: str) -> str:
    side = raw.strip().upper()
    if side in {"L", "LONG"}:
        return "LONG"
    if side in {"S", "SHORT"}:
        return "SHORT"
    if side in {"B", "BOTH", "ALL"}:
        return "BOTH"
    raise ValueError(f"Unsupported side '{raw}' (expected LONG/SHORT/BOTH)")


def _parse_symbol_specs(raw: str, default_side: str) -> Tuple[List[str], Dict[str, List[str]]]:
    specs: Dict[str, List[str]] = {}
    order: List[str] = []
    default = _normalize_side(default_side)
    parts = [p for p in re.split(r"[,\s]+", raw.strip()) if p]
    for part in parts:
        if ":" in part:
            symbol, side_raw = part.split(":", 1)
            side = _normalize_side(side_raw)
        else:
            symbol, side = part, default
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        if symbol not in specs:
            specs[symbol] = []
            order.append(symbol)
        if side == "BOTH":
            for s in ("LONG", "SHORT"):
                if s not in specs[symbol]:
                    specs[symbol].append(s)
        else:
            if side not in specs[symbol]:
                specs[symbol].append(side)
    return order, specs


def _price_precision(entry: float) -> int:
    if entry < 1:
        return 6
    if entry < 10:
        return 4
    if entry < 1000:
        return 3
    return 2


def _fmt_price(value: float, precision: int) -> str:
    return f"{value:.{precision}f}"


def _baseline_levels(
    *,
    side: str,
    entry: float,
    atr_raw: float,
    atr_mult: float,
    rr: float,
    atr_mode: str,
    finder: ShortTermCryptoFinder,
) -> Tuple[float, float, float]:
    atr_eff = float(atr_raw)
    if atr_mode == "clipped":
        atr_eff = finder._cap_atr_value(atr_eff, entry)
    risk = atr_eff * float(atr_mult)
    if risk <= 0:
        return atr_eff, 0.0, 0.0
    if side == "LONG":
        stop = entry - risk
        tp = entry + risk * rr
    else:
        stop = entry + risk
        tp = entry - risk * rr
    return atr_eff, float(stop), float(tp)


def _format_block(
    *,
    rank: int,
    coin: Dict[str, object],
    side: str,
    entry: float,
    stop: float,
    tp: float,
    rr: float,
    atr_raw: float,
    atr_eff: float,
    atr_mode: str,
    atr_mult: float,
    pos_pct: float,
) -> str:
    symbol = str(coin.get("symbol", ""))
    name = str(coin.get("name", "n/a"))
    ts = str(coin.get("data_timestamp_utc") or "")
    price_prec = _price_precision(entry)
    rr_actual = abs(tp - entry) / abs(entry - stop) if stop != entry else 0.0
    lines: List[str] = []
    lines.append(f"{rank}. {symbol} ({name}) - {side}")
    lines.append("-" * 50)
    if ts:
        lines.append(f"Data Timestamp (UTC): {ts}")
    lines.append(f"TRADING LEVELS ({side}):")
    lines.append(f"Entry Price: ${_fmt_price(entry, price_prec)}")
    lines.append(f"Stop Loss: ${_fmt_price(stop, price_prec)}")
    lines.append(f"Take Profit: ${_fmt_price(tp, price_prec)}")
    lines.append(f"Risk:Reward Ratio: {rr_actual:.2f}:1 (target {rr:.2f})")
    lines.append(f"Baseline ATR: raw={_fmt_price(atr_raw, price_prec)} "
                 f"eff={_fmt_price(atr_eff, price_prec)} mode={atr_mode} mult={atr_mult:.2f}")
    lines.append(f"Recommended Position Size: {pos_pct:.1f}% of portfolio")
    lines.append("")
    return "\n".join(lines)


def _write_finder_file(path: Path, blocks: Sequence[str]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    header = [
        "=" * 100,
        "SHORT-TERM BASELINE OPPORTUNITIES (SNAPSHOT)",
        "=" * 100,
        f"Generated on (UTC): {ts}",
        f"Total opportunities listed: {len(blocks)}",
        "=" * 100,
        "",
    ]
    content = "\n".join(header + list(blocks)).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")


def _open_paper_trades(
    *,
    out_path: Path,
    symbols: Iterable[str],
    portfolio_usd: Optional[float],
    fixed_position_usd: Optional[float],
    default_position_pct: float,
    leverage: Optional[float],
    expiry_hours: float,
    tag: Optional[str],
    notes: Optional[str],
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "paper_finder_simulator.py"),
        "open",
        "--finder-output",
        str(out_path),
        "--symbols",
        *list(symbols),
        "--default-position-pct",
        str(default_position_pct),
        "--expiry-hours",
        str(expiry_hours),
    ]
    if portfolio_usd is not None:
        cmd.extend(["--portfolio-usd", f"{portfolio_usd:.2f}"])
    if fixed_position_usd is not None:
        cmd.extend(["--fixed-position-usd", f"{fixed_position_usd:.2f}"])
    if leverage is not None:
        cmd.extend(["--leverage", f"{leverage:.2f}"])
    if tag:
        cmd.extend(["--tag", tag])
    if notes:
        cmd.extend(["--notes", notes])
    if dry_run:
        cmd.append("--dry-run")

    subprocess.run(cmd, check=False)


def _load_open_paper_pairs(path: Path = OPEN_PAPER_CSV) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty:
        return set()
    status_col = "status" if "status" in df.columns else None
    pairs: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        if status_col and str(row.get(status_col, "")).upper() != "OPEN":
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            product = str(row.get("product_id") or "").strip().upper()
            if product:
                symbol = product.split("-")[0]
        side = str(row.get("position_side") or "").strip().upper()
        if symbol and side:
            pairs.add((symbol, side))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a baseline finder file from snapshot symbols (optional: open paper trades)."
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Comma/space-separated symbols. Optional side suffix (e.g., BTC:SHORT,SUPER:LONG).",
    )
    parser.add_argument(
        "--side",
        default="long",
        help="Default side for symbols without suffix: long, short, or both (default: long).",
    )
    parser.add_argument(
        "--profile",
        default="focused_no_llm_100",
        choices=sorted(PROFILE_PRESETS.keys()),
        help="Finder profile to apply (default: focused_no_llm_100).",
    )
    parser.add_argument(
        "--out",
        default="finder_short_baseline.txt",
        help="Output finder file (default: finder_short_baseline.txt).",
    )
    parser.add_argument("--rr", type=float, default=2.0, help="Baseline RR target (default: 2.0).")
    parser.add_argument("--atr-mult", type=float, default=1.3, help="ATR multiple for SL distance (default: 1.3).")
    parser.add_argument(
        "--atr-mode",
        choices=["raw", "clipped"],
        default="clipped",
        help="Use raw ATR or capped ATR (default: clipped).",
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=5.0,
        help="Recommended position size percent in the finder block (default: 5.0).",
    )
    parser.add_argument(
        "--no-liquidity-filter",
        action="store_true",
        help="Disable liquidity filters (min volume and volume/market-cap ratio).",
    )
    parser.add_argument(
        "--skip-spread-gate",
        action="store_true",
        help="Skip the spread-margin gate for these symbols.",
    )
    parser.add_argument(
        "--max-spread-margin-pct",
        type=float,
        default=None,
        help="Override spread gate max margin percent (default uses config).",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=None,
        help="Leverage used for spread-gate margin calc and paper trade metadata.",
    )
    parser.add_argument(
        "--open-paper",
        action="store_true",
        help="Open the generated finder blocks in paper_finder_simulator.py.",
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Include symbols that already have open paper trades (default: skip duplicates).",
    )
    parser.add_argument(
        "--portfolio-usd",
        type=float,
        default=None,
        help="Portfolio size for paper trade sizing.",
    )
    parser.add_argument(
        "--fixed-position-usd",
        type=float,
        default=None,
        help="Fixed USD size per trade for paper trades.",
    )
    parser.add_argument(
        "--expiry-hours",
        type=float,
        default=24.0,
        help="Expiry horizon for paper trades (default: 24h).",
    )
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for paper trades.")
    parser.add_argument("--notes", type=str, default=None, help="Optional note for paper trades.")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run when opening paper trades.")
    args = parser.parse_args()

    order, symbol_sides = _parse_symbol_specs(args.symbols, args.side)
    if not order:
        raise SystemExit("No symbols parsed from --symbols input.")

    cfg = build_short_term_config()
    apply_profile_overrides(cfg, args.profile)
    cfg.symbols = list(order)
    cfg.force_refresh_candles = True
    if args.no_liquidity_filter:
        cfg.min_volume_24h = 0
        cfg.min_volume_market_cap_ratio = 0
    if args.max_spread_margin_pct is not None:
        cfg.max_spread_margin_pct = args.max_spread_margin_pct
    leverage_hint = args.leverage
    if leverage_hint is not None:
        cfg.report_leverage = leverage_hint
    else:
        leverage_hint = getattr(cfg, "report_leverage", None)
    if args.skip_spread_gate:
        cfg.max_spread_margin_pct = None

    finder = ShortTermCryptoFinder(config=cfg)
    coins = finder.get_cryptocurrencies_to_analyze(limit=None, symbols=cfg.symbols)
    if not coins:
        raise SystemExit("No symbols retrieved (check connectivity or liquidity/spread filters).")

    coin_map = {str(c.get("symbol", "")).upper(): c for c in coins}
    blocks: List[str] = []
    added_symbols: set[str] = set()
    rank = 1
    open_pairs = _load_open_paper_pairs() if (args.open_paper and not args.include_open) else set()
    skipped_pairs: set[tuple[str, str]] = set()
    for symbol in order:
        coin = coin_map.get(symbol)
        if not coin:
            print(f"{symbol}: not available after filters; try --no-liquidity-filter or --skip-spread-gate")
            continue
        product_id = str(coin.get("product_id") or "")
        df = finder.get_historical_data(product_id, days=cfg.analysis_days)
        if df is None or df.empty:
            print(f"{symbol}: no historical data")
            continue
        tech = finder.calculate_technical_indicators(df)
        entry = float(coin.get("current_price") or tech.get("price") or df["price"].iloc[-1])
        atr_raw = float(tech.get("atr") or 0.0)
        for side in symbol_sides.get(symbol, []):
            if open_pairs and (symbol, side) in open_pairs:
                skipped_pairs.add((symbol, side))
                continue
            atr_eff, stop, tp = _baseline_levels(
                side=side,
                entry=entry,
                atr_raw=atr_raw,
                atr_mult=args.atr_mult,
                rr=args.rr,
                atr_mode=args.atr_mode,
                finder=finder,
            )
            if stop <= 0 or tp <= 0:
                print(f"{symbol} {side}: invalid baseline levels (ATR={atr_raw:.4f})")
                continue
            blocks.append(
                _format_block(
                    rank=rank,
                    coin=coin,
                    side=side,
                    entry=entry,
                    stop=stop,
                    tp=tp,
                    rr=args.rr,
                    atr_raw=atr_raw,
                    atr_eff=atr_eff,
                    atr_mode=args.atr_mode,
                    atr_mult=args.atr_mult,
                    pos_pct=args.position_pct,
                )
            )
            rank += 1
            added_symbols.add(symbol)

    if skipped_pairs:
        skipped_txt = ", ".join(sorted([f"{sym}:{side}" for sym, side in skipped_pairs]))
        print(f"Skipping open paper positions: {skipped_txt}")

    out_path = Path(args.out)
    _write_finder_file(out_path, blocks)
    print(f"Wrote {len(blocks)} baseline entries to {out_path}")

    if args.open_paper and blocks:
        symbols_for_open = [sym for sym in order if sym in added_symbols]
        _open_paper_trades(
            out_path=out_path,
            symbols=symbols_for_open,
            portfolio_usd=args.portfolio_usd,
            fixed_position_usd=args.fixed_position_usd,
            default_position_pct=args.position_pct,
            leverage=leverage_hint,
            expiry_hours=args.expiry_hours,
            tag=args.tag,
            notes=args.notes,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
