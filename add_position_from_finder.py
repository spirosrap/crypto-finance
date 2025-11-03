#!/usr/bin/env python3
"""
Add Perp Position From Finder Output

Parses a single-asset text block produced by ``long_term_crypto_finder.py`` or
``short_term_crypto_finder.py`` and prepares a perpetual order using the
conventions in ``ccxt_trade_perp.py``.

Default behavior is dry-run: prints a ready-to-run ccxt_trade_perp.py command
and a summarized order plan. Pass --execute to actually place the order using
the CCXT Coinbase Advanced client (market or limit with brackets). API keys
must be configured for execution.

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
import re
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

from ccxt_trade_perp import (
    MarketMeta,
    calculate_base_size,
    ensure_margin_balance,
    fetch_reference_price,
    get_market_meta,
    load_exchange,
    place_entry_order,
    place_trigger_bracket_order,
    quantize_price,
    wait_for_fill,
)
from perp_support import (
    canonical_perp_symbol,
    perp_price_multiplier,
)

def round_to_step(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step``."""
    if step <= 0:
        return value
    return round(value / step) * step


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


def _decimals_for_value(value: float, max_places: int = 8) -> int:
    """Infer the number of decimal places from a numeric value."""
    s = f"{value:.{max_places}f}".rstrip("0").rstrip(".")
    if "." in s:
        return len(s.split(".")[1])
    return 0


def normalize_perp(symbol: str, prefer: str = "PERP-INTX") -> str:
    base = canonical_perp_symbol(symbol)
    if not base:
        return ""
    suffix = "INTX-PERP" if prefer == "INTX-PERP" else "PERP-INTX"
    return f"{base}-{suffix}"


@dataclass
class ParsedFinder:
    symbol: str
    base_symbol: str
    side: str  # LONG | SHORT
    entry: float
    stop: float
    take_profit: float
    pos_size_pct: float  # percentage
    entry_decimals: int
    stop_decimals: int
    take_profit_decimals: int

    def max_price_decimals(self) -> int:
        """Return the max precision implied by parsed price levels (cap at 8)."""
        return min(8, max(self.entry_decimals, self.stop_decimals, self.take_profit_decimals))

    def min_price_value(self) -> float:
        """Return the smallest positive price among entry/TP/SL."""
        positives = [v for v in (self.entry, self.stop, self.take_profit) if v > 0]
        return min(positives) if positives else 0.0


def parse_finder_text(text: str) -> ParsedFinder:
    # Symbol: from first line or explicit "The Ticker Is XXX"
    m_sym = re.search(r"The Ticker Is\s+([A-Z0-9]{2,20})", text)
    if not m_sym:
        m_sym = re.search(r"^\s*\d+\.\s*([A-Z0-9]{2,20})\b", text, re.M)
    if not m_sym:
        raise ValueError("Could not find symbol in text")
    symbol = m_sym.group(1).upper()
    m_base = re.search(r"\(([A-Z0-9]{2,20})-[A-Z0-9]{2,10}\)", text)
    base_symbol = m_base.group(1).upper() if m_base else symbol

    # Side: from header line or TRADING LEVELS block
    m_side = re.search(r"—\s*(LONG|SHORT)", text)
    if not m_side:
        m_side = re.search(r"TRADING LEVELS\s*\((LONG|SHORT)\)", text)
    side = (m_side.group(1) if m_side else "LONG").upper()

    # Trading levels
    def _extract_value(label: str) -> Tuple[Optional[float], int]:
        pat = rf"{label}\s*:\s*\$?\s*([0-9]+(?:\.[0-9]+)?)"
        m = re.search(pat, text, re.I)
        if not m:
            return None, 0
        raw = m.group(1)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        return float(raw), decimals

    entry, entry_dec = _extract_value("Entry Price")
    if entry is None:
        entry, entry_dec = _extract_value("Price")
    stop, stop_dec = _extract_value("Stop Loss")
    take_profit, tp_dec = _extract_value("Take Profit")
    if entry is None or stop is None or take_profit is None:
        raise ValueError("Missing entry/stop/take-profit values in text")

    # Position size percent
    m_sz = re.search(r"Recommended Position Size\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text, re.I)
    pos_pct = float(m_sz.group(1)) if m_sz else 0.0

    return ParsedFinder(
        symbol=symbol,
        base_symbol=base_symbol,
        side=side,
        entry=entry,
        stop=stop,
        take_profit=take_profit,
        pos_size_pct=pos_pct,
        entry_decimals=entry_dec,
        stop_decimals=stop_dec,
        take_profit_decimals=tp_dec,
    )


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Create perp position from long- or short-term finder text output")
    ap.add_argument("--file", type=str, help="Path to finder output text; omit to read stdin")
    ap.add_argument("--portfolio-usd", type=float, required=True, help="Total portfolio value in USD")
    ap.add_argument("--leverage", type=float, default=5.0, help="Leverage 1-20 (default 5)")
    ap.add_argument("--product-form", type=str, choices=["PERP-INTX", "INTX-PERP"], default="PERP-INTX", help="Perp suffix format to display")
    ap.add_argument("--order", type=str, choices=["market", "limit"], default="market", help="Order type")
    ap.add_argument("--execute", action="store_true", help="Actually place the order (otherwise dry-run)")
    ap.add_argument("--expiry", type=str, choices=["GTC", "12h", "24h", "30d"], default="30d", help="GTD expiry for bracket orders")

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

    exchange_obj: Optional[object] = None
    exchange_error: Optional[Exception] = None
    exchange_warning_shown = False

    def ensure_exchange(require: bool) -> Optional[object]:
        nonlocal exchange_obj, exchange_error, exchange_warning_shown
        if exchange_obj is not None:
            return exchange_obj
        if exchange_error is not None:
            if require:
                raise RuntimeError("Unable to initialise CCXT Coinbase Advanced client.") from exchange_error
            return None
        try:
            exchange_obj = load_exchange()
            return exchange_obj
        except Exception as exc:
            exchange_error = exc
            if require:
                raise RuntimeError("Unable to initialise CCXT Coinbase Advanced client.") from exc
            if not exchange_warning_shown:
                print(f"Warning: could not load Coinbase Advanced markets via CCXT ({exc}). "
                      "Falling back to heuristic precision estimates.")
                exchange_warning_shown = True
            return None

    commands: List[List[str]] = []
    summaries: List[str] = []
    api_pids: List[str] = []
    side_perps: List[str] = []
    tps: List[float] = []
    sls: List[float] = []
    limits: List[Optional[float]] = []
    sizes_usd: List[float] = []
    metas: List[Optional[MarketMeta]] = []
    price_steps: List[float] = []

    unsupported: List[Tuple[str, str, Optional[str]]] = []

    for parsed in parsed_list:
        display_symbol = canonical_perp_symbol(parsed.base_symbol or parsed.symbol)
        display_pid = normalize_perp(parsed.base_symbol or parsed.symbol, prefer=args.product_form)
        api_pid = normalize_perp(parsed.base_symbol or parsed.symbol, prefer="PERP-INTX")
        exchange_candidate = ensure_exchange(require=False)
        meta: Optional[MarketMeta] = None
        if exchange_candidate is not None:
            try:
                meta = get_market_meta(exchange_candidate, api_pid)
            except Exception as exc:
                unsupported.append((display_symbol or parsed.symbol, api_pid, str(exc)))
                continue
        price_multiplier = perp_price_multiplier(parsed.base_symbol or parsed.symbol)
        scaled_entry = parsed.entry * price_multiplier
        scaled_tp_val = parsed.take_profit * price_multiplier
        scaled_sl_val = parsed.stop * price_multiplier
        side_perp = "SELL" if parsed.side == "SHORT" else "BUY"
        size_usd = (args.portfolio_usd * (parsed.pos_size_pct / 100.0)) if parsed.pos_size_pct > 0 else (args.portfolio_usd * 0.05)
        tick = meta.price_precision if meta and getattr(meta, "price_precision", 0.0) else 0.0
        price_candidates = [v for v in (scaled_entry, scaled_tp_val, scaled_sl_val) if v > 0]
        min_price_value = min(price_candidates) if price_candidates else 0.0
        fallback_decimals = parsed.max_price_decimals()
        if price_multiplier != 1.0:
            fallback_decimals = max(
                fallback_decimals,
                _decimals_for_value(scaled_entry),
                _decimals_for_value(scaled_tp_val),
                _decimals_for_value(scaled_sl_val),
            )
        fallback_tick = 10 ** (-fallback_decimals) if fallback_decimals > 0 else (tick if tick > 0 else 1.0)
        effective_tick = tick if tick and tick > 0 else fallback_tick
        if min_price_value > 0 and min_price_value < effective_tick:
            effective_tick = fallback_tick
        if effective_tick <= 0:
            effective_tick = fallback_tick if fallback_tick > 0 else 0.01

        if meta and getattr(meta, "price_precision", 0.0):
            tp = quantize_price(scaled_tp_val, meta.price_precision)
            sl = quantize_price(scaled_sl_val, meta.price_precision)
            entry_value = quantize_price(scaled_entry, meta.price_precision)
            rounding_step = meta.price_precision
        else:
            tp = round_to_step(scaled_tp_val, effective_tick)
            sl = round_to_step(scaled_sl_val, effective_tick)
            entry_value = round_to_step(scaled_entry, effective_tick)
            rounding_step = effective_tick
        limit_price = entry_value if args.order == "limit" else None

        # Format numbers according to combined precision (tick + finder text)
        decimals = max(_decimals_for_tick(effective_tick), fallback_decimals)
        if exchange_candidate is not None and meta is not None:
            tp_str = exchange_candidate.price_to_precision(meta.ccxt_symbol, tp)
            sl_str = exchange_candidate.price_to_precision(meta.ccxt_symbol, sl)
            limit_str = (
                exchange_candidate.price_to_precision(meta.ccxt_symbol, limit_price)
                if limit_price is not None
                else None
            )
        else:
            tp_str = f"{tp:.{decimals}f}"
            sl_str = f"{sl:.{decimals}f}"
            limit_str = f"{limit_price:.{decimals}f}" if limit_price is not None else None

        cmd = [
            "python", "ccxt_trade_perp.py",
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
        metas.append(meta)
        price_steps.append(rounding_step if rounding_step > 0 else effective_tick)
        entry_basis = limit_price if limit_price is not None else entry_value
        if entry_basis <= 0:
            entry_basis = max(scaled_entry, 1e-9)
        if exchange_candidate is not None and meta is not None:
            entry_disp = exchange_candidate.price_to_precision(meta.ccxt_symbol, entry_basis)
        else:
            entry_disp = f"{entry_basis:.{decimals}f}"
        # Estimate percentage move between entry and TP/SL; clamp to avoid negatives
        reward_denominator = max(entry_basis, 1e-9)
        if parsed.side == "SHORT":
            reward_pct = max((entry_basis - tp) / reward_denominator, 0.0)
            risk_pct = max((sl - entry_basis) / reward_denominator, 0.0)
        else:
            reward_pct = max((tp - entry_basis) / reward_denominator, 0.0)
            risk_pct = max((entry_basis - sl) / reward_denominator, 0.0)
        margin_usd = size_usd / max(args.leverage, 1e-9)
        reward_usd = reward_pct * size_usd
        risk_usd = risk_pct * size_usd
        summaries.append(
            f"Symbol: {display_symbol} Side: {parsed.side}  Entry: ${entry_disp}  TP: ${tp_str}  SL: ${sl_str}\n"
            f"Product: {display_pid} (API {api_pid})  Size: {parsed.pos_size_pct or 5.0:.2f}% of ${args.portfolio_usd:.2f} ≈ ${size_usd:.2f} (Margin ≈ ${margin_usd:.2f})  Expiry: {args.expiry}\n"
            f"PnL vs position: TP +${reward_usd:.2f} ({reward_pct * 100:.2f}%) | SL -${risk_usd:.2f} ({risk_pct * 100:.2f}%)"
        )

    print("\n=== Parsed Finder Signals ===")
    for s in summaries:
        print("\n" + s)

    print("\nCommands:")
    for cmd in commands:
        print(" ".join(cmd))

    if not args.execute:
        if unsupported:
            print("\nSkipped unsupported Coinbase perps:")
            for sym, pid, reason in unsupported:
                if reason:
                    print(f"- {sym}: {pid} ({reason})")
                else:
                    print(f"- {sym}: {pid}")
        return

    # Execute all sequentially
    exchange_live = ensure_exchange(require=True)
    if exchange_live is None:
        print("Execution aborted: unable to initialise CCXT exchange client.")
        return
    for i, api_pid in enumerate(api_pids):
        try:
            meta = metas[i]
            if meta is None:
                meta = get_market_meta(exchange_live, api_pid)
                metas[i] = meta
            current_price = fetch_reference_price(exchange_live, meta.ccxt_symbol)
            entry_price = limits[i] if limits[i] is not None else current_price
            base_size = calculate_base_size(sizes_usd[i], entry_price, meta)
            ensure_margin_balance(exchange_live, sizes_usd[i] / max(args.leverage, 1e-9))

            precision_step = meta.price_precision if meta.price_precision and meta.price_precision > 0 else price_steps[i]
            if precision_step and precision_step > 0:
                tp_price = quantize_price(tps[i], precision_step)
                sl_price = quantize_price(sls[i], precision_step)
                limit_price = quantize_price(limits[i], precision_step) if limits[i] is not None else None
            else:
                tp_price = tps[i]
                sl_price = sls[i]
                limit_price = limits[i]

            entry_order = place_entry_order(
                exchange_live,
                meta,
                side_perps[i],
                base_size,
                limit_price,
                args.leverage,
                args.expiry,
                dry_run=False,
            )

            bracket_response = {}
            if tp_price > 0 and sl_price > 0:
                entry_order_id = entry_order.get("id")
                entry_status = entry_order.get("status")
                if limit_price is not None and entry_order_id and entry_status != "closed":
                    final_state = wait_for_fill(exchange_live, meta.ccxt_symbol, entry_order_id)
                    if final_state and final_state.get("status") != "closed":
                        print(f"\n[{api_pid}] Entry order not filled (status={final_state.get('status')}); bracket submission skipped.")
                        continue
                bracket_response = place_trigger_bracket_order(
                    exchange_live,
                    meta,
                    side_perps[i],
                    base_size,
                    tp_price,
                    sl_price,
                    args.leverage,
                    args.expiry,
                    dry_run=False,
                )
            else:
                print(f"\n[{api_pid}] Skipping bracket submission (tp/sl <= 0).")

            print(f"\n[{api_pid}] Entry order submitted (id={entry_order.get('id')}). "
                  f"Bracket response: {bracket_response if bracket_response else 'n/a'}")
        except Exception as e:
            print(f"\n[{api_pid}] Execution error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
