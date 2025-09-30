#!/usr/bin/env python3
"""
Select Top-5 Trades From Finder Output (2 longs, 2 shorts, + next best)

Reads text output produced by short/long-term finder scripts, ranks trades by a
"score" parsed from the text, and constructs up to 5 trade commands for
`trade_btc_perp.py`:
  - Best 2 LONGs by score
  - Best 2 SHORTs by score
  - Next best remaining by score (any side)

Behaves like `add_position_from_finder.py` otherwise (dry-run by default; use
`--execute` to place orders via CoinbaseService). Threaded with `--expiry` to
set GTD for bracket orders.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

# Reuse helpers from trade_btc_perp
try:
    from trade_btc_perp import get_price_precision, round_to_precision, calculate_base_size
except Exception:
    def get_price_precision(product_id: str) -> float:
        return {
            'BTC-PERP-INTX': 1.0,
            'ETH-PERP-INTX': 0.1,
            'DOGE-PERP-INTX': 0.0001,
            'SOL-PERP-INTX': 0.01,
            'XRP-PERP-INTX': 0.001,
            '1000SHIB-PERP-INTX': 0.000001,
            'NEAR-PERP-INTX': 0.001,
            'SUI-PERP-INTX': 0.0001,
            'ATOM-PERP-INTX': 0.001,
        }.get(product_id, 0.01)

    def round_to_precision(value: float, precision: float) -> float:
        return round(value / precision) * precision if precision > 0 else value

    def calculate_base_size(product_id: str, size_usd: float, current_price: float) -> float:
        return max(size_usd / max(current_price, 1e-9), 0.0)


def _decimals_for_tick(tick: float) -> int:
    if tick <= 0:
        return 0
    s = f"{tick:.10f}".rstrip("0").rstrip(".")
    if "." in s:
        return len(s.split(".")[1])
    return 0


@dataclass
class ParsedFinder:
    symbol: str
    side: str  # LONG | SHORT
    entry: float
    stop: float
    take_profit: float
    pos_size_pct: float
    score: float


def normalize_perp(symbol: str, prefer: str = "PERP-INTX") -> str:
    s = (symbol or "").upper().strip()
    if not s:
        return ""
    if prefer == "INTX-PERP":
        return f"{s}-INTX-PERP"
    return f"{s}-PERP-INTX"


def split_blocks(text: str) -> List[str]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    marker = "Short-Line Summaries"
    idx = t.find(marker)
    if idx != -1:
        t = t[:idx]
    heads = [m.start() for m in re.finditer(r"(?m)^\s*\d+\.\s+\S+\s*\(", t)]
    if not heads:
        return [text]
    heads.append(len(t))
    blocks: List[str] = []
    for i in range(len(heads) - 1):
        blocks.append(t[heads[i]:heads[i+1]].strip())
    return blocks


def _num_after(label: str, text: str) -> Optional[float]:
    pat = rf"{label}\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)"
    m = re.search(pat, text, re.I)
    return float(m.group(1)) if m else None


def _parse_score(text: str) -> float:
    """Best-effort extraction of a numeric score from a block.

    Looks for common phrases like "Overall Score", "Score", "Confidence Score".
    Returns 0.0 if no score is found.
    """
    patterns = [
        r"Overall\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Signal\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Confidence\s*Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        r"Score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return 0.0


def parse_block(text: str) -> ParsedFinder:
    m_sym = re.search(r"The Ticker Is\s+([A-Z0-9]{2,20})", text)
    if not m_sym:
        m_sym = re.search(r"^\s*\d+\.\s*([A-Z0-9]{2,20})\b", text, re.M)
    if not m_sym:
        raise ValueError("Could not find symbol in text")
    symbol = m_sym.group(1).upper()

    m_side = re.search(r"—\s*(LONG|SHORT)", text)
    if not m_side:
        m_side = re.search(r"TRADING LEVELS\s*\((LONG|SHORT)\)", text)
    side = (m_side.group(1) if m_side else "LONG").upper()

    entry = _num_after("Entry Price", text) or _num_after("Price", text)
    stop = _num_after("Stop Loss", text)
    take_profit = _num_after("Take Profit", text)
    if entry is None or stop is None or take_profit is None:
        raise ValueError("Missing entry/stop/take-profit values in text")

    m_sz = re.search(r"Recommended Position Size\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
    pos_pct = float(m_sz.group(1)) if m_sz else 0.0

    score = _parse_score(text)
    return ParsedFinder(symbol=symbol, side=side, entry=entry, stop=stop, take_profit=take_profit, pos_size_pct=pos_pct, score=score)


def setup_cb() -> CoinbaseService:
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise RuntimeError("Missing API_KEY_PERPS/API_SECRET_PERPS in config.py or env")
    return CoinbaseService(api_key, api_secret)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create top-5 perp positions from finder text output (2L/2S + next best)")
    ap.add_argument("--file", type=str, help="Path to finder output text; omit to read stdin")
    ap.add_argument("--portfolio-usd", type=float, required=True, help="Total portfolio value in USD")
    ap.add_argument("--leverage", type=float, default=5.0, help="Leverage 1-20 (default 5)")
    ap.add_argument("--product-form", type=str, choices=["PERP-INTX", "INTX-PERP"], default="PERP-INTX", help="Perp suffix format to display")
    ap.add_argument("--order", type=str, choices=["market", "limit"], default="market", help="Order type")
    ap.add_argument("--execute", action="store_true", help="Actually place the order (otherwise dry-run)")
    ap.add_argument("--expiry", type=str, choices=["12h", "24h", "30d"], default="30d", help="GTD expiry for bracket orders")

    args = ap.parse_args()

    # Read text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    blocks = split_blocks(text)
    parsed_items: List[ParsedFinder] = []
    for b in blocks:
        try:
            parsed_items.append(parse_block(b))
        except Exception as e:
            print(f"Skipping block due to parse error: {e}")
            continue

    if not parsed_items:
        print("No valid trades found in input.")
        return

    # Rank: top 2 longs, top 2 shorts, then next best remaining by score
    longs = [p for p in parsed_items if p.side == "LONG"]
    shorts = [p for p in parsed_items if p.side == "SHORT"]
    longs.sort(key=lambda x: x.score, reverse=True)
    shorts.sort(key=lambda x: x.score, reverse=True)

    selected: List[ParsedFinder] = []
    selected.extend(longs[:2])
    selected.extend(shorts[:2])
    remaining = [p for p in parsed_items if p not in selected]
    remaining.sort(key=lambda x: x.score, reverse=True)
    if remaining:
        selected.append(remaining[0])

    # Limit to at most 5 (in case of duplicates or small sets)
    selected = selected[:5]

    # If fewer than required trades exist, proceed with what we have
    # Prepare commands and summaries
    leverage_str = f"{args.leverage:g}"
    commands: List[List[str]] = []
    summaries: List[str] = []
    api_pids: List[str] = []
    side_perps: List[str] = []
    tps: List[float] = []
    sls: List[float] = []
    limits: List[Optional[float]] = []
    sizes_usd: List[float] = []

    for p in selected:
        display_pid = normalize_perp(p.symbol, prefer=args.product_form)
        api_pid = normalize_perp(p.symbol, prefer="PERP-INTX")
        side_perp = "SELL" if p.side == "SHORT" else "BUY"
        size_usd = (args.portfolio_usd * (p.pos_size_pct / 100.0)) if p.pos_size_pct > 0 else (args.portfolio_usd * 0.05)
        tick = get_price_precision(api_pid)
        tp = round_to_precision(p.take_profit, tick)
        sl = round_to_precision(p.stop, tick)
        limit_price = round_to_precision(p.entry, tick) if args.order == "limit" else None

        decimals = _decimals_for_tick(tick)
        tp_str = f"{tp:.{decimals}f}"
        sl_str = f"{sl:.{decimals}f}"
        limit_str = f"{limit_price:.{decimals}f}" if limit_price is not None else None

        cmd = [
            "python", "trade_btc_perp.py",
            "--product", api_pid,
            "--side", side_perp,
            "--size", f"{size_usd:.2f}",
            "--leverage", leverage_str,
            "--tp", tp_str,
            "--sl", sl_str,
        ]
        if limit_str is not None:
            cmd += ["--limit", limit_str]
        cmd += ["--expiry", args.expiry]

        commands.append(cmd)
        api_pids.append(api_pid)
        side_perps.append(side_perp)
        tps.append(tp)
        sls.append(sl)
        limits.append(limit_price)
        sizes_usd.append(size_usd)

        entry_disp = f"{p.entry:.{decimals}f}"
        summaries.append(
            f"Symbol: {p.symbol} Side: {p.side}  Score: {p.score:.2f}  Entry: ${entry_disp}  TP: ${tp_str}  SL: ${sl_str}\n"
            f"Product: {display_pid} (API {api_pid})  Size: {p.pos_size_pct or 5.0:.2f}% of ${args.portfolio_usd:.2f} ≈ ${size_usd:.2f}  Expiry: {args.expiry}"
        )

    print("\n=== Selected Top-5 Signals (2L/2S + next best) ===")
    for s in summaries:
        print("\n" + s)

    print("\nCommands:")
    for cmd in commands:
        print(" ".join(cmd))

    if not args.execute:
        return

    # Execute all sequentially
    cb = setup_cb()
    for i, api_pid in enumerate(api_pids):
        try:
            trades = cb.client.get_market_trades(product_id=api_pid, limit=1)
            current_price = float(trades['trades'][0]['price'])
            base_size = calculate_base_size(api_pid, sizes_usd[i], current_price)

            if limits[i] is not None:
                res = cb.place_limit_order_with_targets(
                    product_id=api_pid,
                    side=side_perps[i],
                    size=base_size,
                    entry_price=limits[i],
                    take_profit_price=tps[i],
                    stop_loss_price=sls[i],
                    leverage=leverage_str,
                    expiry=args.expiry,
                )
                if isinstance(res, dict) and "error" in res:
                    print(f"\n[{api_pid}] Error placing limit order: {res['error']}")
                else:
                    print(f"\n[{api_pid}] Limit order submitted.")
            else:
                res = cb.place_market_order_with_targets(
                    product_id=api_pid,
                    side=side_perps[i],
                    size=base_size,
                    take_profit_price=tps[i],
                    stop_loss_price=sls[i],
                    leverage=leverage_str,
                    expiry=args.expiry,
                )
                if isinstance(res, dict) and "error" in res:
                    print(f"\n[{api_pid}] Error placing market order: {res['error']}")
                else:
                    print(f"\n[{api_pid}] Market order submitted.")
        except Exception as e:
            print(f"\n[{api_pid}] Execution error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass


