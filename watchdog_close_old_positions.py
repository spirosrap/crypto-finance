#!/usr/bin/env python3
"""
Watchdog: Close Perp Positions Older Than N Hours (default 24h)

Runs once (or on an interval) to:
  - Query INTX perpetual positions
  - Inspect each position's open/entry timestamp
  - Market-close any position older than the configured age threshold

Usage examples:
  python watchdog_close_old_positions.py --max-age-hours 24
  python watchdog_close_old_positions.py --max-age-hours 24 --interval-seconds 300
  python watchdog_close_old_positions.py --product BTC-PERP-INTX

Notes:
  - Cancels open orders for a product before attempting to close its position
  - Uses market IOC orders to close positions similar to close_all_positions()
  - Timestamps are parsed from multiple common keys to be robust across payloads
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def _get_portfolio_uuid(cb: CoinbaseService) -> Optional[str]:
    ports = cb.client.get_portfolios()
    # Normalize to iterable of portfolio entries
    portfolios_list = None
    if isinstance(ports, dict):
        portfolios_list = ports.get('portfolios', [])
    else:
        # Try attribute access
        plist = getattr(ports, 'portfolios', None)
        if plist is not None:
            portfolios_list = plist
        else:
            # Fall back to __dict__ if present
            try:
                ports_dict = vars(ports)
                portfolios_list = ports_dict.get('portfolios', [])
            except Exception:
                portfolios_list = []

    for p in portfolios_list or []:
        if isinstance(p, dict):
            p_type = p.get('type')
            p_uuid = p.get('uuid')
        else:
            p_type = getattr(p, 'type', None)
            p_uuid = getattr(p, 'uuid', None)
        if p_type == 'INTX' and p_uuid:
            return p_uuid
    return None


def _parse_iso8601(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    s = ts if isinstance(ts, str) else str(ts)
    fmts = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for f in fmts:
        try:
            return datetime.strptime(s, f)
        except Exception:
            continue
    return None


def _extract_position_open_time(pos: Any) -> Optional[datetime]:
    # Handle dict and object-like
    def g(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    candidate_keys = [
        'created_time', 'open_time', 'opened_at', 'entry_time', 'position_created_time'
    ]
    for key in candidate_keys:
        dt = _parse_iso8601(g(pos, key))
        if dt:
            return dt

    # Sometimes nested under 'position_pnl' or similar metadata
    for parent in ['position_pnl', 'metadata', 'details']:
        dt = _parse_iso8601(g(g(pos, parent), 'open_time'))
        if dt:
            return dt

    return None


def _extract_symbol_and_size(pos: Any) -> tuple[Optional[str], float, str, str]:
    symbol = None
    size = 0.0
    side_field = ''
    leverage = '1'

    if isinstance(pos, dict):
        symbol = pos.get('symbol') or pos.get('product_id')
        try:
            size = float(pos.get('net_size', 0) or 0)
        except Exception:
            size = 0.0
        side_field = pos.get('position_side', '')
        leverage = str(pos.get('leverage', '1'))
    else:
        symbol = getattr(pos, 'symbol', None) or getattr(pos, 'product_id', None)
        try:
            size = float(getattr(pos, 'net_size', 0) or 0)
        except Exception:
            size = 0.0
        side_field = getattr(pos, 'position_side', '')
        leverage = str(getattr(pos, 'leverage', '1'))

    return symbol, size, side_field, leverage


def _close_position(cb: CoinbaseService, product_id: str, net_size: float, position_side: str, leverage: str) -> bool:
    logger = logging.getLogger(__name__)
    # Determine closing side
    side = 'BUY' if position_side == 'FUTURES_POSITION_SIDE_SHORT' else 'SELL'
    close_size = abs(net_size)

    # Cancel open orders for this product first
    try:
        cb.cancel_all_orders(product_id=product_id)
    except Exception as e:
        logger.warning(f"Failed to cancel existing orders for {product_id}: {e}")

    # Market IOC close
    try:
        client_order_id = f"close_{int(time.time())}"
        order_config = {"market_market_ioc": {"base_size": str(close_size)}}
        result = cb.client.create_order(
            client_order_id=client_order_id,
            product_id=product_id,
            side=side,
            order_configuration=order_config,
            leverage=leverage,
            margin_type="CROSS"
        )
        if isinstance(result, dict) and result.get('success', True):
            logger.info(f"Closed {product_id} position via {side} {close_size}")
            return True
        logger.error(f"Close order result for {product_id}: {result}")
        return False
    except Exception as e:
        logger.error(f"Error closing position for {product_id}: {e}")
        return False


def run_once(max_age_hours: int, product_filter: Optional[str]) -> None:
    logger = logging.getLogger(__name__)
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)

    portfolio_uuid = _get_portfolio_uuid(cb)
    if not portfolio_uuid:
        logger.error("Could not find INTX portfolio UUID")
        return

    portfolio = cb.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
    positions = []
    if isinstance(portfolio, dict):
        breakdown = portfolio.get('breakdown', {})
        # breakdown can be dict or object
        if isinstance(breakdown, dict):
            positions = breakdown.get('perp_positions', [])
        else:
            positions = getattr(breakdown, 'perp_positions', []) or []
    else:
        breakdown = getattr(portfolio, 'breakdown', None)
        if breakdown is not None:
            if isinstance(breakdown, dict):
                positions = breakdown.get('perp_positions', [])
            else:
                positions = getattr(breakdown, 'perp_positions', []) or []

    if not positions:
        logger.info("No perpetual positions found")
        return

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
    logger.info(f"Closing positions opened before {cutoff.isoformat()}Z")

    for pos in positions:
        symbol, net_size, position_side, leverage = _extract_symbol_and_size(pos)
        if not symbol or abs(net_size) <= 0:
            continue
        if product_filter and symbol != product_filter:
            continue

        opened_at = _extract_position_open_time(pos)
        if not opened_at:
            logger.warning(f"No open/entry timestamp found for {symbol}; skipping")
            continue

        if opened_at <= cutoff:
            logger.info(f"Position {symbol} opened at {opened_at.isoformat()}Z exceeds {max_age_hours}h; closing...")
            _close_position(cb, symbol, net_size, position_side, leverage)
        else:
            logger.debug(f"Position {symbol} age OK (opened {opened_at.isoformat()}Z)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Watchdog to close perp positions older than N hours")
    ap.add_argument("--max-age-hours", type=int, default=24, help="Age threshold in hours (default 24)")
    ap.add_argument("--product", type=str, help="Only check/close for a specific product id (e.g., BTC-PERP-INTX)")
    ap.add_argument("--interval-seconds", type=int, default=0, help="If >0, run continuously with this interval")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = ap.parse_args()
    setup_logging(verbose=args.verbose)

    if args.interval_seconds and args.interval_seconds > 0:
        while True:
            try:
                run_once(args.max_age_hours, args.product)
            except Exception as e:
                logging.getLogger(__name__).error(f"Watchdog iteration error: {e}")
            time.sleep(args.interval_seconds)
    else:
        run_once(args.max_age_hours, args.product)


if __name__ == "__main__":
    main()


