#!/usr/bin/env python3
"""
Export finder-style plain-text signals to a structured JSON payload for Freqtrade.

Usage:
  python export_finder_signals.py --file finder_short.txt --out signals/freqtrade_signals.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

from add_position_from_finder import (
    ParsedFinder,
    parse_finder_text,
    split_blocks,
    normalize_perp,
)
from perp_support import perp_price_multiplier


def _derive_pair(product_id: str) -> str:
    # Normalize product (e.g., 1000SHIB-PERP-INTX -> 1000SHIB/USDC)
    base = product_id.replace("-PERP-INTX", "")
    return f"{base}/USDC"


def _format_price(value: float) -> float:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return float(text) if text else value


def export_signals(
    parsed_signals: List[ParsedFinder],
    portfolio_usd: float,
    leverage: float,
    expiry_hours: float,
    include_zero: bool,
) -> dict:
    generated_at = datetime.now(timezone.utc)
    payload = {
        "generated_at": generated_at.isoformat(),
        "expiry_hours": expiry_hours,
        "portfolio_usd": portfolio_usd,
        "leverage": leverage,
        "signals": [],
    }

    for parsed in parsed_signals:
        if not include_zero and parsed.take_profit == parsed.entry:
            continue

        product_id = normalize_perp(parsed.base_symbol or parsed.symbol, prefer="PERP-INTX")
        pair = _derive_pair(product_id)
        multiplier = perp_price_multiplier(parsed.base_symbol or parsed.symbol)
        entry = _format_price(parsed.entry * multiplier)
        tp = _format_price(parsed.take_profit * multiplier)
        sl = _format_price(parsed.stop * multiplier)

        payload["signals"].append(
            {
                "pair": pair,
                "product_id": product_id,
                "side": "LONG" if parsed.side.upper() == "LONG" else "SHORT",
                "entry": entry,
                "take_profit": tp,
                "stop_loss": sl,
                "leverage": leverage,
                "position_pct": parsed.pos_size_pct or 5.0,
                "confidence": parsed.pos_size_pct / 100.0 if parsed.pos_size_pct else None,
                "expires_at": (generated_at + timedelta(hours=expiry_hours)).isoformat(),
            }
        )

    return payload


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export finder signals for Freqtrade bridge strategy.")
    ap.add_argument("--file", type=Path, default=Path("finder_short.txt"), help="Finder plain-text file to parse.")
    ap.add_argument("--out", type=Path, default=Path("signals") / "freqtrade_signals.json", help="Output JSON path.")
    ap.add_argument("--portfolio-usd", type=float, default=13000.0, help="Portfolio value for sizing reference.")
    ap.add_argument("--leverage", type=float, default=50.0, help="Finder leverage assumption.")
    ap.add_argument("--expiry-hours", type=float, default=24.0, help="Signal expiry horizon in hours.")
    ap.add_argument("--include-zero", action="store_true", help="Include entries where TP equals entry.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not args.file.exists():
        raise FileNotFoundError(f"Finder file not found: {args.file}")

    raw_text = args.file.read_text(encoding="utf-8")
    blocks = split_blocks(raw_text)
    parsed_records: List[ParsedFinder] = []
    for block in blocks:
        try:
            parsed_records.append(parse_finder_text(block))
        except Exception as exc:
            print(f"Skipping block due to parse error: {exc}")

    output_payload = export_signals(
        parsed_records,
        portfolio_usd=args.portfolio_usd,
        leverage=args.leverage,
        expiry_hours=args.expiry_hours,
        include_zero=args.include_zero,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Exported {len(output_payload['signals'])} signal(s) to {args.out}")


if __name__ == "__main__":
    main()
