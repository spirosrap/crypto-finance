#!/usr/bin/env python3
"""
List all USDC-settled perpetual swap pairs available on Coinbase Advanced via CCXT.

Usage:
    python scripts/list_coinbase_perps.py
"""

from __future__ import annotations

import sys
from typing import Dict, Any, List

try:
    import ccxt  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard
    sys.stderr.write("ccxt is required for this script. Install with 'pip install ccxt'.\n")
    raise SystemExit(1) from exc


EXCHANGE_ID = "coinbaseadvanced"


def load_exchange() -> ccxt.Exchange:
    """Initialise the CCXT exchange client with rate limiting enabled."""
    exchange_cls = getattr(ccxt, EXCHANGE_ID, None)
    if exchange_cls is None:
        raise RuntimeError(f"CCXT exchange '{EXCHANGE_ID}' is not available in this ccxt build.")
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def filter_perp_markets(markets: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return symbols for USDC-settled perpetual swaps.

    CCXT uses the notation BASE/QUOTE:SETTLEMENT for derivatives, so BTC/USDC:USDC
    indicates a BTC perpetual margined and settled in USDC.
    """
    results: List[str] = []
    for symbol, meta in markets.items():
        if not isinstance(meta, dict):
            continue
        if not meta.get("swap"):
            continue
        if meta.get("quote") != "USDC":
            continue
        if not meta.get("linear", True):
            continue
        if not meta.get("active", True):
            continue
        results.append(symbol)
    return sorted(results)


def main() -> None:
    exchange = load_exchange()
    perps = filter_perp_markets(exchange.markets)

    print(f"Found {len(perps)} USDC-settled perpetual pairs on Coinbase Advanced:\n")
    for symbol in perps:
        print(symbol)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # pragma: no cover - CLI error surfacing
        sys.stderr.write(f"Error listing perpetual markets: {err}\n")
        raise SystemExit(1) from err

