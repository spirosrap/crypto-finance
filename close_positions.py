#!/usr/bin/env python3
"""
Manually flatten open INTX perp positions with retry + verification.

This script is intentionally strict:
- It does not treat order submission as success.
- It keeps polling open positions and retries closes until flat (or attempts exhausted).
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from typing import Any, List, Tuple

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
from ccxt_trade_perp import get_market_meta, load_exchange
from watchdog_close_old_positions import (
    _extract_symbol_and_size,
    _get_portfolio_uuid,
)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _list_open_positions(cb: CoinbaseService) -> List[Tuple[str, float, str, str]]:
    """Return open positions as (symbol, net_size, position_side, leverage)."""
    logger = logging.getLogger(__name__)

    portfolio_uuid = _get_portfolio_uuid(cb)
    if not portfolio_uuid:
        logger.error("Could not find INTX portfolio UUID.")
        return []

    try:
        response = cb.client.list_perps_positions(portfolio_uuid=portfolio_uuid)
    except Exception as exc:
        logger.error("Failed to list perp positions: %s", exc)
        return []

    if isinstance(response, dict):
        positions_raw = response.get("positions", []) or []
    else:
        positions_raw = getattr(response, "positions", []) or []

    open_positions: List[Tuple[str, float, str, str]] = []
    for pos in positions_raw:
        symbol, net_size, position_side, leverage = _extract_symbol_and_size(pos)
        if not symbol:
            continue
        if abs(float(net_size)) <= 0:
            continue
        open_positions.append((symbol, float(net_size), str(position_side or ""), str(leverage or "1")))

    return open_positions


def _close_with_retries(
    cb: CoinbaseService,
    *,
    product_filter: str | None,
    attempts: int,
    wait_seconds: float,
) -> bool:
    logger = logging.getLogger(__name__)
    try:
        exchange = load_exchange()
    except Exception as exc:
        logger.error("Failed to initialize CCXT exchange: %s", exc)
        return False

    def _close_one(symbol: str, net_size: float, position_side: str) -> bool:
        side = "BUY" if ("SHORT" in str(position_side).upper() or net_size < 0) else "SELL"
        close_qty_abs = abs(float(net_size))
        if close_qty_abs <= 0:
            return True
        try:
            meta = get_market_meta(exchange, symbol)
        except Exception as exc:
            logger.error("Failed market metadata lookup for %s: %s", symbol, exc)
            return False

        # Try full size first, then slightly smaller clips to avoid occasional preview/fill edge cases.
        for size_pct in (1.0, 0.99, 0.95):
            candidate = close_qty_abs * size_pct
            if candidate <= 0:
                continue
            base_size = exchange.amount_to_precision(meta.ccxt_symbol, candidate)
            payload = {
                "client_order_id": f"manual-close-{uuid.uuid4().hex[:16]}",
                "product_id": meta.market["id"],
                "side": side,
                "order_configuration": {
                    "market_market_ioc": {
                        "base_size": base_size,
                        "reduce_only": True,
                    }
                },
            }
            try:
                response = exchange.v3PrivatePostBrokerageOrders(payload)
            except Exception as exc:
                logger.warning(
                    "Raw v3 close failed for %s size_pct=%.2f base_size=%s: %s",
                    symbol,
                    size_pct,
                    base_size,
                    exc,
                )
                continue

            logger.info(
                "Submitted close for %s side=%s size_pct=%.2f base_size=%s response=%s",
                symbol,
                side,
                size_pct,
                base_size,
                response,
            )

            success = bool(response.get("success", True)) if isinstance(response, dict) else True
            if not success:
                continue

            # Verify reduction actually happened before claiming success.
            reduced = False
            for _ in range(8):
                time.sleep(0.5)
                current_positions = _list_open_positions(cb)
                current = next((abs(size) for sym, size, _, _ in current_positions if sym == symbol), 0.0)
                if current <= 1e-9 or current < close_qty_abs * 0.01:
                    reduced = True
                    break
                if current + 1e-9 < close_qty_abs:
                    reduced = True
                    break
            if reduced:
                return True

        return False

    for attempt in range(1, max(1, attempts) + 1):
        open_positions = _list_open_positions(cb)
        if product_filter:
            open_positions = [p for p in open_positions if p[0] == product_filter]

        if not open_positions:
            logger.info("No open perpetual positions remain.")
            return True

        logger.info(
            "Close attempt %d/%d: %d open position(s): %s",
            attempt,
            attempts,
            len(open_positions),
            ", ".join(f"{sym}:{size:+.8f}" for sym, size, _, _ in open_positions),
        )

        # Cancel open orders first to avoid bracket interference.
        try:
            if product_filter:
                cb.cancel_all_orders(product_id=product_filter)
            else:
                cb.cancel_all_orders()
        except Exception as exc:
            logger.warning("cancel_all_orders failed (continuing): %s", exc)

        for symbol, net_size, position_side, leverage in open_positions:
            _ = leverage  # Leverage is not required for reduce-only market close payload.
            closed = _close_one(symbol, net_size, position_side)
            logger.info(
                "Close submit %s size=%+.8f side=%s -> closed=%s",
                symbol,
                net_size,
                position_side,
                closed,
            )

        if attempt < attempts:
            time.sleep(max(0.0, float(wait_seconds)))

    # Final verification after all attempts.
    remaining = _list_open_positions(cb)
    if product_filter:
        remaining = [p for p in remaining if p[0] == product_filter]
    if not remaining:
        logger.info("All target positions flattened.")
        return True

    logger.error(
        "Still open after %d attempts: %s",
        attempts,
        ", ".join(f"{sym}:{size:+.8f}" for sym, size, _, _ in remaining),
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Close open INTX perp positions with retries and verification.")
    parser.add_argument(
        "--product",
        type=str,
        default=None,
        help="Optional product filter (e.g., XTZ-PERP-INTX). If omitted, closes all open perps.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Maximum close attempts with re-check between attempts (default 5).",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=2.0,
        help="Seconds to wait between attempts (default 2).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    setup_logging(verbose=bool(args.verbose))
    logger = logging.getLogger(__name__)

    try:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    except Exception as exc:
        logger.error("Failed to initialize CoinbaseService: %s", exc)
        raise SystemExit(1)

    ok = _close_with_retries(
        cb,
        product_filter=(args.product.strip().upper() if args.product else None),
        attempts=max(1, int(args.attempts)),
        wait_seconds=max(0.0, float(args.wait_seconds)),
    )
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
