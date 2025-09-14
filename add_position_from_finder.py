#!/usr/bin/env python3
"""
Add Perp Position From Finder Output

Parses a single-asset text block produced by long_term_crypto_finder.py and
prepares a perpetual order using the conventions in trade_btc_perp.py.

Default behavior is dry-run: prints a ready-to-run trade_btc_perp.py command
and a summarized order plan. Pass --execute to actually place the order using
CoinbaseService (market or limit with brackets). API keys must be configured.

Assumptions
- Side comes from lines like "— LONG/SHORT" or "TRADING LEVELS (LONG/SHORT)".
- Uses TRADING LEVELS values for entry/TP/SL.
- Product id is constructed from the symbol as SYMBOL-PERP-INTX. If input or
  expectation is SYMBOL-INTX-PERP, we normalize to SYMBOL-PERP-INTX for API.

Examples
  python add_position_from_finder.py --file finder.txt \
    --portfolio-usd 25000 --leverage 5 --order market

  python add_position_from_finder.py --file finder.txt \
    --portfolio-usd 25000 --leverage 5 --order limit --execute
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

# Reuse tick/size helpers from trade_btc_perp.py
try:
    from trade_btc_perp import get_price_precision, round_to_precision, calculate_base_size
except Exception:
    # Minimal fallbacks if import path changes
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


@dataclass
class ParsedFinder:
    symbol: str
    side: str  # LONG | SHORT
    entry: float
    stop: float
    take_profit: float
    pos_size_pct: float  # percentage


def normalize_perp(symbol: str, prefer: str = "PERP-INTX") -> str:
    s = (symbol or "").upper().strip()
    if not s:
        return ""
    if prefer == "INTX-PERP":
        return f"{s}-INTX-PERP"
    return f"{s}-PERP-INTX"


def parse_finder_text(text: str) -> ParsedFinder:
    # Symbol: from first line or explicit "The Ticker Is XXX"
    m_sym = re.search(r"The Ticker Is\s+([A-Z0-9]{2,20})", text)
    if not m_sym:
        m_sym = re.search(r"^\s*\d+\.\s*([A-Z0-9]{2,20})\b", text, re.M)
    if not m_sym:
        raise ValueError("Could not find symbol in text")
    symbol = m_sym.group(1).upper()

    # Side: from header line or TRADING LEVELS block
    m_side = re.search(r"—\s*(LONG|SHORT)", text)
    if not m_side:
        m_side = re.search(r"TRADING LEVELS\s*\((LONG|SHORT)\)", text)
    side = (m_side.group(1) if m_side else "LONG").upper()

    # Trading levels
    def _num_after(label: str) -> Optional[float]:
        pat = rf"{label}\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)"
        m = re.search(pat, text, re.I)
        return float(m.group(1)) if m else None

    entry = _num_after("Entry Price") or _num_after("Price")
    stop = _num_after("Stop Loss")
    take_profit = _num_after("Take Profit")
    if entry is None or stop is None or take_profit is None:
        raise ValueError("Missing entry/stop/take-profit values in text")

    # Position size percent
    m_sz = re.search(r"Recommended Position Size\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
    pos_pct = float(m_sz.group(1)) if m_sz else 0.0

    return ParsedFinder(symbol=symbol, side=side, entry=entry, stop=stop, take_profit=take_profit, pos_size_pct=pos_pct)


def setup_cb() -> CoinbaseService:
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise RuntimeError("Missing API_KEY_PERPS/API_SECRET_PERPS in config.py or env")
    return CoinbaseService(api_key, api_secret)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create perp position from long_term_crypto_finder text output")
    ap.add_argument("--file", type=str, help="Path to finder output text; omit to read stdin")
    ap.add_argument("--portfolio-usd", type=float, required=True, help="Total portfolio value in USD")
    ap.add_argument("--leverage", type=float, default=5.0, help="Leverage 1-20 (default 5)")
    ap.add_argument("--product-form", type=str, choices=["PERP-INTX", "INTX-PERP"], default="PERP-INTX", help="Perp suffix format to display")
    ap.add_argument("--order", type=str, choices=["market", "limit"], default="market", help="Order type")
    ap.add_argument("--execute", action="store_true", help="Actually place the order (otherwise dry-run)")

    args = ap.parse_args()

    # Read text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    parsed = parse_finder_text(text)

    # Build product id
    display_pid = normalize_perp(parsed.symbol, prefer=args.product_form)
    api_pid = normalize_perp(parsed.symbol, prefer="PERP-INTX")

    side_perp = "SELL" if parsed.side == "SHORT" else "BUY"

    # Compute USD size
    size_usd = (args.portfolio_usd * (parsed.pos_size_pct / 100.0)) if parsed.pos_size_pct > 0 else (args.portfolio_usd * 0.05)

    # Tick rounding for targets and optional limit
    tick = get_price_precision(api_pid)
    tp = round_to_precision(parsed.take_profit, tick)
    sl = round_to_precision(parsed.stop, tick)
    limit_price = round_to_precision(parsed.entry, tick) if args.order == "limit" else None

    # Dry-run summary and suggested command
    cmd = [
        "python", "trade_btc_perp.py",
        "--product", api_pid,
        "--side", side_perp,
        "--size", f"{size_usd:.2f}",
        "--leverage", f"{args.leverage}",
        "--tp", f"{tp}",
        "--sl", f"{sl}",
    ]
    if limit_price is not None:
        cmd += ["--limit", f"{limit_price}"]

    print("\n=== Parsed Finder Signal ===")
    print(f"Symbol: {parsed.symbol}  Side: {parsed.side}")
    print(f"Entry: ${parsed.entry:.6f}  TP: ${tp:.6f}  SL: ${sl:.6f}")
    print(f"Position Size: {parsed.pos_size_pct or 5.0:.2f}% of ${args.portfolio_usd:.2f} ≈ ${size_usd:.2f}")
    print(f"Product: {display_pid} (API uses {api_pid})  Order: {args.order.upper()}")

    print("\nDry‑run command:")
    print(" ".join(cmd))

    if not args.execute:
        return

    # Execute via CoinbaseService similar to trade_btc_perp flow
    cb = setup_cb()

    # Derive current price for base size calc
    trades = cb.client.get_market_trades(product_id=api_pid, limit=1)
    current_price = float(trades['trades'][0]['price'])
    base_size = calculate_base_size(api_pid, size_usd, current_price)

    if limit_price is not None:
        res = cb.place_limit_order_with_targets(
            product_id=api_pid,
            side=side_perp,
            size=base_size,
            entry_price=limit_price,
            take_profit_price=tp,
            stop_loss_price=sl,
            leverage=str(args.leverage),
        )
        if isinstance(res, dict) and "error" in res:
            print(f"\nError placing limit order: {res['error']}")
            return
        print("\nLimit order submitted. Monitor/follow-up bracket flow as needed.")
    else:
        res = cb.place_market_order_with_targets(
            product_id=api_pid,
            side=side_perp,
            size=base_size,
            take_profit_price=tp,
            stop_loss_price=sl,
            leverage=str(args.leverage),
        )
        if isinstance(res, dict) and "error" in res:
            print(f"\nError placing market order: {res['error']}")
            return
        print("\nMarket order submitted with brackets.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

