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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Set, TextIO

import datetime as dt

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

import pandas as pd

DEFAULT_EXCLUSION_FILE = Path("config/excluded_perps.txt")

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
    predicted_return: Optional[float] = None

    def max_price_decimals(self) -> int:
        """Return the max precision implied by parsed price levels (cap at 8)."""
        return min(8, max(self.entry_decimals, self.stop_decimals, self.take_profit_decimals))

    def min_price_value(self) -> float:
        """Return the smallest positive price among entry/TP/SL."""
        positives = [v for v in (self.entry, self.stop, self.take_profit) if v > 0]
        return min(positives) if positives else 0.0


@dataclass
class OrderSettings:
    """Execution settings shared between CLI and programmatic integrations."""

    portfolio_usd: Optional[float]
    leverage: float
    position_usd: Optional[float] = None
    product_form: str = "PERP-INTX"
    order_type: str = "market"
    execute: bool = False
    expiry: str = "30d"
    exclude_file: Optional[Path] = DEFAULT_EXCLUSION_FILE
    excluded_products: Optional[Set[str]] = None
    confidence_scale: float = 0.0
    confidence_threshold: float = 0.0
    max_confidence_multiplier: float = 2.0
    risk_log_path: Path = Path("trade_logs/watchdog_closed_positions.csv")
    expectancy_window: int = 0
    min_expectancy: Optional[float] = None
    max_daily_loss: Optional[float] = None
    max_consecutive_losses: Optional[int] = None


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

    predicted_return = None
    m_pred = re.search(r"Predicted Return .*?:\s*([-+]?[0-9]*\.?[0-9]+)%", text, re.I)
    if m_pred:
        predicted_return = float(m_pred.group(1)) / 100.0
    else:
        m_compact = re.search(r"pred=([-+]?[0-9]*\.?[0-9]+)", text, re.I)
        if m_compact:
            predicted_return = float(m_compact.group(1))

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
        predicted_return=predicted_return,
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


def load_exclusion_list(path: Optional[Path]) -> Set[str]:
    """Return uppercase product ids parsed from ``path`` (ignoring comments)."""

    exclusions: Set[str] = set()
    if path is None:
        return exclusions
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return exclusions
    except Exception as exc:  # pragma: no cover - filesystem I/O
        print(f"Warning: unable to read exclusion file {path}: {exc}", file=sys.stderr)
        return exclusions

    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        exclusions.add(line.upper())
    return exclusions


def evaluate_recent_performance(
    path: Path,
    window: int,
    min_expectancy: Optional[float],
    max_daily_loss: Optional[float],
    max_consecutive_losses: Optional[int],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not path.exists():
        return True, reasons
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        reasons.append(f"Failed to read performance log {path}: {exc}")
        return False, reasons

    if "closed_at" not in df.columns or "profit_loss" not in df.columns:
        reasons.append("Performance log missing required columns.")
        return False, reasons

    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True, errors="coerce")
    df["profit_loss"] = pd.to_numeric(df["profit_loss"], errors="coerce")
    df = df.dropna(subset=["closed_at", "profit_loss"])
    if df.empty:
        return True, reasons

    df = df.sort_values("closed_at")

    if window and window > 0 and len(df) >= window:
        window_slice = df.tail(window)
    else:
        window_slice = df

    if min_expectancy is not None and not window_slice.empty:
        expectancy = window_slice["profit_loss"].mean()
        if expectancy < min_expectancy:
            reasons.append(
                f"Rolling expectancy {expectancy:.2f} < minimum {min_expectancy:.2f} over last {len(window_slice)} trades."
            )

    if max_consecutive_losses is not None and max_consecutive_losses > 0:
        losses = 0
        for pl in reversed(df["profit_loss"]):
            if pl < 0:
                losses += 1
            else:
                break
        if losses >= max_consecutive_losses:
            reasons.append(
                f"{losses} consecutive losses ≥ limit ({max_consecutive_losses})."
            )

    if max_daily_loss is not None:
        today = dt.datetime.utcnow().date()
        day_slice = df[df["closed_at"].dt.date == today]
        if not day_slice.empty:
            daily_loss = day_slice["profit_loss"].sum()
            if daily_loss <= -abs(max_daily_loss):
                reasons.append(
                    f"Today's net P/L {daily_loss:.2f} ≤ -{abs(max_daily_loss):.2f} limit."
                )

    return len(reasons) == 0, reasons


def process_parsed_signals(
    parsed_list: List[ParsedFinder],
    settings: OrderSettings,
    *,
    exchange: Optional[object] = None,
    stream: Optional[TextIO] = None,
) -> Dict[str, Any]:
    """Prepare command strings and optionally execute orders for parsed finder signals."""

    output = stream or sys.stdout
    if not parsed_list:
        print("No finder entries to process.", file=output)
        return {"commands": [], "summaries": [], "unsupported": [], "executions": []}

    order_type = settings.order_type.lower()
    if order_type not in {"market", "limit"}:
        raise ValueError(f"Unsupported order type: {settings.order_type}")

    product_form = settings.product_form.upper()
    if product_form not in {"PERP-INTX", "INTX-PERP"}:
        raise ValueError(f"Unsupported product form: {settings.product_form}")

    expiry = settings.expiry
    leverage_str = f"{settings.leverage:g}"

    exchange_obj = exchange
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
                print(
                    f"Warning: could not load Coinbase Advanced markets via CCXT ({exc}). "
                    "Falling back to heuristic precision estimates.",
                    file=output,
                )
                exchange_warning_shown = True
            return None

    commands: List[str] = []
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
    manually_excluded: List[Tuple[str, str]] = []

    exclusion_set = (
        settings.excluded_products
        if settings.excluded_products is not None
        else load_exclusion_list(settings.exclude_file)
    )

    for parsed in parsed_list:
        display_symbol = canonical_perp_symbol(parsed.base_symbol or parsed.symbol)
        display_pid = normalize_perp(parsed.base_symbol or parsed.symbol, prefer=product_form)
        api_pid = normalize_perp(parsed.base_symbol or parsed.symbol, prefer="PERP-INTX")
        if exclusion_set and api_pid.upper() in exclusion_set:
            manually_excluded.append((display_symbol or parsed.symbol, api_pid))
            continue
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
        size_multiplier = 1.0
        if settings.position_usd is not None and settings.position_usd > 0:
            size_usd = float(settings.position_usd)
            applied_pct = None
        else:
            portfolio_base = settings.portfolio_usd if settings.portfolio_usd is not None else 0.0
            base_pct = parsed.pos_size_pct if parsed.pos_size_pct > 0 else 5.0
            size_usd = portfolio_base * (base_pct / 100.0)
            applied_pct = base_pct
        if (
            settings.confidence_scale > 0
            and parsed.predicted_return is not None
            and parsed.predicted_return > 0
        ):
            threshold = settings.confidence_threshold if settings.confidence_threshold > 0 else 1.0
            ratio = parsed.predicted_return / threshold
            if ratio > 0:
                size_multiplier = 1.0 + settings.confidence_scale * ratio
                size_multiplier = min(size_multiplier, settings.max_confidence_multiplier)
                size_usd *= size_multiplier
                if applied_pct is not None and settings.portfolio_usd is not None and settings.portfolio_usd > 0:
                    applied_pct = (size_usd / settings.portfolio_usd) * 100.0
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
        limit_price = entry_value if order_type == "limit" else None

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

        cmd_parts = [
            "python",
            "ccxt_trade_perp.py",
            "--product",
            api_pid,
            "--side",
            side_perp,
            "--size",
            f"{size_usd:.2f}",
            "--leverage",
            leverage_str,
            "--tp",
            tp_str,
            "--sl",
            sl_str,
        ]
        if limit_str is not None:
            cmd_parts += ["--limit", limit_str]
        cmd_parts += ["--expiry", expiry]

        commands.append(" ".join(cmd_parts))
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
        reward_denominator = max(entry_basis, 1e-9)
        if parsed.side == "SHORT":
            reward_pct = max((entry_basis - tp) / reward_denominator, 0.0)
            risk_pct = max((sl - entry_basis) / reward_denominator, 0.0)
        else:
            reward_pct = max((tp - entry_basis) / reward_denominator, 0.0)
            risk_pct = max((entry_basis - sl) / reward_denominator, 0.0)
        margin_usd = size_usd / max(settings.leverage, 1e-9)
        reward_usd = reward_pct * size_usd
        risk_usd = risk_pct * size_usd
        if applied_pct is None:
            size_blurb = f"Fixed size ≈ ${size_usd:.2f}"
        elif settings.portfolio_usd is None:
            size_blurb = f"{applied_pct:.2f}% sizing (portfolio unspecified) ≈ ${size_usd:.2f}"
        else:
            size_blurb = (
                f"{applied_pct:.2f}% of ${settings.portfolio_usd:.2f} ≈ ${size_usd:.2f}"
            )
        if size_multiplier != 1.0:
            size_blurb += f" (multiplier {size_multiplier:.2f}x)"
        summaries.append(
            f"Symbol: {display_symbol} Side: {parsed.side}  Entry: ${entry_disp}  TP: {tp_str}  SL: {sl_str}\n"
            f"Product: {display_pid} (API {api_pid})  Size: {size_blurb} (Margin ≈ ${margin_usd:.2f})  Expiry: {expiry}\n"
            f"PnL vs position: TP +${reward_usd:.2f} ({reward_pct * 100:.2f}%) | SL -${risk_usd:.2f} ({risk_pct * 100:.2f}%)"
            + (
                f"\nPredicted return: {parsed.predicted_return:.4f}"
                if parsed.predicted_return is not None
                else ""
            )
        )

    print("\n=== Parsed Finder Signals ===", file=output)
    for s in summaries:
        print("\n" + s, file=output)

    print("\nCommands:", file=output)
    for cmd in commands:
        print(cmd, file=output)

    if not settings.execute:
        if unsupported:
            print("\nSkipped unsupported Coinbase perps:", file=output)
            for sym, pid, reason in unsupported:
                if reason:
                    print(f"- {sym}: {pid} ({reason})", file=output)
                else:
                    print(f"- {sym}: {pid}", file=output)
        if manually_excluded:
            print("\nSkipped excluded perps:", file=output)
            for sym, pid in manually_excluded:
                print(f"- {sym}: {pid}", file=output)
        return {
            "commands": commands,
            "summaries": summaries,
            "unsupported": unsupported,
            "excluded": manually_excluded,
            "executions": [],
        }

    execution_logs: List[Dict[str, Any]] = []
    exchange_live = ensure_exchange(require=True)
    if exchange_live is None:
        print("Execution aborted: unable to initialise CCXT exchange client.", file=output)
        return {
            "commands": commands,
            "summaries": summaries,
            "unsupported": unsupported,
            "excluded": manually_excluded,
            "executions": execution_logs,
        }

    for i, api_pid in enumerate(api_pids):
        try:
            meta = metas[i]
            if meta is None:
                meta = get_market_meta(exchange_live, api_pid)
                metas[i] = meta
            current_price = fetch_reference_price(exchange_live, meta.ccxt_symbol)
            entry_price = limits[i] if limits[i] is not None else current_price
            base_size = calculate_base_size(sizes_usd[i], entry_price, meta)
            ensure_margin_balance(exchange_live, sizes_usd[i] / max(settings.leverage, 1e-9))

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
                settings.leverage,
                expiry,
                dry_run=False,
            )

            bracket_response: Dict[str, Any] = {}
            if tp_price > 0 and sl_price > 0:
                entry_order_id = entry_order.get("id")
                entry_status = entry_order.get("status")
                if limit_price is not None and entry_order_id and entry_status != "closed":
                    final_state = wait_for_fill(exchange_live, meta.ccxt_symbol, entry_order_id)
                    if final_state and final_state.get("status") != "closed":
                        message = f"[{api_pid}] Entry order not filled (status={final_state.get('status')}); bracket submission skipped."
                        print(f"\n{message}", file=output)
                        execution_logs.append(
                            {
                                "product": api_pid,
                                "status": "entry_not_filled",
                                "entry_order": entry_order,
                                "final_state": final_state,
                            }
                        )
                        continue
                bracket_response = place_trigger_bracket_order(
                    exchange_live,
                    meta,
                    side_perps[i],
                    base_size,
                    tp_price,
                    sl_price,
                    settings.leverage,
                    expiry,
                    dry_run=False,
                )
            else:
                print(f"\n[{api_pid}] Skipping bracket submission (tp/sl <= 0).", file=output)

            print(
                f"\n[{api_pid}] Entry order submitted (id={entry_order.get('id')}). "
                f"Bracket response: {bracket_response if bracket_response else 'n/a'}",
                file=output,
            )
            execution_logs.append(
                {
                    "product": api_pid,
                    "status": "submitted",
                    "entry_order": entry_order,
                    "bracket": bracket_response,
                }
            )
        except Exception as exc:
            print(f"\n[{api_pid}] Execution error: {exc}", file=output)
            execution_logs.append(
                {
                    "product": api_pid,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return {
        "commands": commands,
        "summaries": summaries,
        "unsupported": unsupported,
        "excluded": manually_excluded,
        "executions": execution_logs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create perp position from long- or short-term finder text output")
    ap.add_argument("--file", type=str, help="Path to finder output text; omit to read stdin")
    ap.add_argument("--portfolio-usd", type=float, help="Total portfolio value in USD (optional when --position-usd is provided)")
    ap.add_argument(
        "--position-usd",
        type=float,
        help="Fixed USD notional per trade (overrides percentage sizing from finder).",
    )
    ap.add_argument("--leverage", type=float, default=5.0, help="Leverage 1-20 (default 5)")
    ap.add_argument("--product-form", type=str, choices=["PERP-INTX", "INTX-PERP"], default="PERP-INTX", help="Perp suffix format to display")
    ap.add_argument("--order", type=str, choices=["market", "limit"], default="market", help="Order type")
    ap.add_argument("--execute", action="store_true", help="Actually place the order (otherwise dry-run)")
    ap.add_argument("--expiry", type=str, choices=["GTC", "12h", "24h", "30d"], default="30d", help="GTD expiry for bracket orders")
    ap.add_argument(
        "--exclude-file",
        type=Path,
        default=DEFAULT_EXCLUSION_FILE,
        help=f"Path to newline-separated perp ids to skip (default: {DEFAULT_EXCLUSION_FILE})",
    )
    ap.add_argument(
        "--confidence-scale",
        type=float,
        default=0.0,
        help="Scale position size by predicted-return confidence (0 disables).",
    )
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Reference predicted-return threshold for scaling (set to reservoir threshold).",
    )
    ap.add_argument(
        "--max-confidence-multiplier",
        type=float,
        default=2.0,
        help="Maximum multiplier applied via confidence scaling.",
    )
    ap.add_argument(
        "--expectancy-window",
        type=int,
        default=0,
        help="Number of recent trades to evaluate expectancy (0 disables).",
    )
    ap.add_argument(
        "--min-expectancy",
        type=float,
        help="Minimum average PnL (USD) over expectancy window to allow live execution.",
    )
    ap.add_argument(
        "--max-daily-loss",
        type=float,
        help="Abort execution when today's cumulative PnL ≤ -value (USD).",
    )
    ap.add_argument(
        "--max-consecutive-losses",
        type=int,
        help="Abort execution after this many consecutive losing trades.",
    )
    ap.add_argument(
        "--performance-log",
        type=Path,
        default=Path("trade_logs/watchdog_closed_positions.csv"),
        help="Path to closed-trade log for expectancy/killswitch checks.",
    )

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

    ok_to_trade, rail_reasons = evaluate_recent_performance(
        settings.risk_log_path,
        settings.expectancy_window,
        settings.min_expectancy,
        settings.max_daily_loss,
        settings.max_consecutive_losses,
    )
    if not ok_to_trade:
        print("Risk rails triggered; skipping execution:")
        for reason in rail_reasons:
            print(f"- {reason}")
        if not parsed_list:
            return
        settings = replace(settings, execute=False)

    settings = OrderSettings(
        portfolio_usd=args.portfolio_usd,
        leverage=args.leverage,
        position_usd=args.position_usd,
        product_form=args.product_form,
        order_type=args.order,
        execute=args.execute,
        expiry=args.expiry,
        exclude_file=args.exclude_file,
        confidence_scale=args.confidence_scale,
        confidence_threshold=args.confidence_threshold,
        max_confidence_multiplier=args.max_confidence_multiplier,
        risk_log_path=args.performance_log,
        expectancy_window=args.expectancy_window,
        min_expectancy=args.min_expectancy,
        max_daily_loss=args.max_daily_loss,
        max_consecutive_losses=args.max_consecutive_losses,
    )
    process_parsed_signals(parsed_list, settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
