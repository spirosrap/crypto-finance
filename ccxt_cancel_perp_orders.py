#!/usr/bin/env python3
"""
Cancel open Coinbase Advanced perpetual orders using CCXT.

Examples:
    python ccxt_cancel_perp_orders.py --product BTC-PERP-INTX
    python ccxt_cancel_perp_orders.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Iterable, Tuple

import ccxt  # type: ignore

try:
    from credentials import get_perps_credentials
except ImportError:  # pragma: no cover
    def get_perps_credentials() -> Tuple[str, str]:
        return os.getenv("API_KEY_PERPS", ""), os.getenv("API_SECRET_PERPS", "")


LOG = logging.getLogger("ccxt_cancel_perp_orders")
if not LOG.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOG.addHandler(handler)
LOG.setLevel(logging.INFO)


def load_exchange() -> ccxt.Exchange:
    key_env = os.getenv("COINBASE_PERP_API_KEY")
    secret_env = os.getenv("COINBASE_PERP_API_SECRET")
    key_cfg, secret_cfg = get_perps_credentials()
    api_key = key_env or key_cfg
    api_secret = secret_env or secret_cfg
    if not api_key or not api_secret:
        raise RuntimeError("Missing API credentials (set API_KEY_PERPS / API_SECRET_PERPS or matching env vars).")
    exchange = ccxt.coinbaseadvanced(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        }
    )
    exchange.load_markets()
    return exchange


def normalize_symbol(product: str | None) -> str | None:
    if product is None:
        return None
    product = product.strip()
    if "/" in product:
        return product
    product = product.upper()
    if product.endswith("-PERP-INTX"):
        base = product.split("-")[0]
        return f"{base}/USDC:USDC"
    raise ValueError(f"Unrecognised product id '{product}'. Provide CCXT symbol or e.g. BTC-PERP-INTX.")


def fetch_open_orders(exchange: ccxt.Exchange, symbol: str | None) -> Iterable[dict]:
    params = {}
    if symbol is None:
        LOG.info("Fetching open orders for all perpetual products…")
    else:
        LOG.info("Fetching open orders for %s…", symbol)
    return exchange.fetch_open_orders(symbol, params)


def cancel_orders(exchange: ccxt.Exchange, symbol: str | None, dry_run: bool) -> None:
    orders = list(fetch_open_orders(exchange, symbol))
    if not orders:
        LOG.info("No open orders found.")
        return

    LOG.info("Found %d open orders.", len(orders))
    for order in orders:
        oid = order.get("id")
        osymbol = order.get("symbol")
        status = order.get("status")
        LOG.info("Canceling %s (%s, %s)", oid, osymbol, status)
        if dry_run:
            continue
        try:
            exchange.cancel_order(oid, osymbol)
        except Exception as exc:
            LOG.error("Failed to cancel %s: %s", oid, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cancel Coinbase perpetual orders via CCXT.")
    parser.add_argument("--product", help="Specific perpetual product (e.g., BTC-PERP-INTX). If omitted, cancels all.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    parser.add_argument("--dry-run", action="store_true", help="List orders without cancelling.")
    args = parser.parse_args()

    symbol = normalize_symbol(args.product)
    exchange = load_exchange()

    if not args.dry_run and not args.yes:
        prompt = "Cancel"
        prompt += f" orders for {symbol}" if symbol else " all open orders"
        confirm = input(f"{prompt}? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted by user.")
            return

    cancel_orders(exchange, symbol, args.dry_run)
    if args.dry_run:
        LOG.info("Dry run complete; no cancellations submitted.")
    else:
        LOG.info("Cancellation routine finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI entry point
        LOG.error("Error: %s", exc)
        sys.exit(1)

