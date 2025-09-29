#!/usr/bin/env python3
"""
Add Perp Position From Finder Output

Parses a single-asset text block produced by ``long_term_crypto_finder.py`` or
``short_term_crypto_finder.py`` and prepares a perpetual order using the
conventions in ``trade_btc_perp.py``.

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
from typing import Optional, Tuple, List

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


def _decimals_for_tick(tick: float) -> int:
    """Return the number of decimal places implied by a tick size.

    Examples: 1.0 -> 0, 0.1 -> 1, 0.01 -> 2, 0.0001 -> 4
    """
    if tick <= 0:
        return 0
    # Convert to string safely and count fractional digits after trimming zeros
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


def split_blocks(text: str) -> List[str]:
    """Split a multi-asset finder text into blocks starting with "<n>. SYMBOL" lines.

    Falls back to a single block when no numbering is detected.
    """
    # Normalize line endings and strip tailing summary sections
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    marker = "Short-Line Summaries"
    idx = t.find(marker)
    if idx != -1:
        t = t[:idx]
    # Find all header indices
    heads = [m.start() for m in re.finditer(r"(?m)^\s*\d+\.\s+\S+\s*\(", t)]
    if not heads:
        return [text]
    heads.append(len(t))
    blocks = []
    for i in range(len(heads) - 1):
        blocks.append(t[heads[i]:heads[i+1]].strip())
    return blocks


def setup_cb() -> CoinbaseService:
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise RuntimeError("Missing API_KEY_PERPS/API_SECRET_PERPS in config.py or env")
    return CoinbaseService(api_key, api_secret)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create perp position from long- or short-term finder text output")
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
    parsed_list: List[ParsedFinder] = []
    for b in blocks:
        try:
            parsed_list.append(parse_finder_text(b))
        except Exception as e:
            print(f"Skipping block due to parse error: {e}")
            continue

    # Format leverage without unnecessary decimals (e.g., 50.0 -> 50)
    leverage_str = f"{args.leverage:g}"

    commands: List[List[str]] = []
    summaries: List[str] = []
    api_pids: List[str] = []
    side_perps: List[str] = []
    tps: List[float] = []
    sls: List[float] = []
    limits: List[Optional[float]] = []
    sizes_usd: List[float] = []

    for parsed in parsed_list:
        display_pid = normalize_perp(parsed.symbol, prefer=args.product_form)
        api_pid = normalize_perp(parsed.symbol, prefer="PERP-INTX")
        side_perp = "SELL" if parsed.side == "SHORT" else "BUY"
        size_usd = (args.portfolio_usd * (parsed.pos_size_pct / 100.0)) if parsed.pos_size_pct > 0 else (args.portfolio_usd * 0.05)
        tick = get_price_precision(api_pid)
        tp = round_to_precision(parsed.take_profit, tick)
        sl = round_to_precision(parsed.stop, tick)
        limit_price = round_to_precision(parsed.entry, tick) if args.order == "limit" else None

        # Format numbers according to tick precision to avoid float noise
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
        entry_disp = f"{parsed.entry:.{decimals}f}"
        summaries.append(
            f"Symbol: {parsed.symbol} Side: {parsed.side}  Entry: ${entry_disp}  TP: ${tp_str}  SL: ${sl_str}\n"
            f"Product: {display_pid} (API {api_pid})  Size: {parsed.pos_size_pct or 5.0:.2f}% of ${args.portfolio_usd:.2f} ≈ ${size_usd:.2f}  Expiry: {args.expiry}"
        )

    print("\n=== Parsed Finder Signals ===")
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
            # Current price for base size calc
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
