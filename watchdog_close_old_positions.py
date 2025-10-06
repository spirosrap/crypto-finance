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
import csv
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS


LOG_HEADERS = [
    'closed_at',
    'product_id',
    'position_side',
    'net_size',
    'leverage',
    'opened_at',
    'closure_reason',
    'entry_price',
    'exit_price',
    'profit_loss',
    'profit_loss_pct',
    'mae',
    'mfe',
    'duration_seconds',
]


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def _log_file_path() -> Path:
    base_dir = os.environ.get('WATCHDOG_LOG_DIR', 'trade_logs')
    return Path(base_dir).expanduser() / 'watchdog_closed_positions.csv'


def _ensure_log_file() -> Path:
    path = _log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_HEADERS)
            writer.writeheader()
    return path


def _breakeven_threshold() -> float:
    raw = os.environ.get('WATCHDOG_BREAKEVEN_ABS', '1.0')
    try:
        threshold = abs(float(raw))
        return threshold
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Invalid WATCHDOG_BREAKEVEN_ABS=%r; defaulting to 1.0", raw
        )
        return 1.0


def _get_value(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ('rawCurrency', 'userNativeCurrency', 'value', 'amount'):
            nested = value.get(key)
            result = _coerce_numeric(nested)
            if result is not None:
                return result
        return None
    try:
        attrs = vars(value)
    except TypeError:
        attrs = None
    if attrs:
        return _coerce_numeric(attrs)
    return None


def _gather_containers(pos: Any) -> list[Any]:
    containers = [pos]
    for key in ('position_pnl', 'metadata', 'details', 'stats', 'metrics', 'extras'):
        value = _get_value(pos, key)
        if value is not None:
            containers.append(value)
    return containers


def _extract_entry_price(pos: Any) -> Optional[float]:
    for key in ('vwap', 'entry_price', 'average_entry', 'avg_entry_price'):
        value = _get_value(pos, key)
        numeric = _coerce_numeric(value)
        if numeric is not None:
            return numeric
    return None


def _extract_mark_price(pos: Any) -> Optional[float]:
    for key in ('mark_price', 'current_price', 'price', 'last_price'):
        value = _get_value(pos, key)
        numeric = _coerce_numeric(value)
        if numeric is not None:
            return numeric
    return None


def _extract_unrealized_pnl(pos: Any, net_size: float, entry_price: Optional[float], mark_price: Optional[float]) -> Optional[float]:
    value = _get_value(pos, 'unrealized_pnl')
    pnl = _coerce_numeric(value)
    if pnl is not None:
        return pnl
    if entry_price is not None and mark_price is not None:
        return net_size * (mark_price - entry_price)
    return None


def _extract_excursions(pos: Any) -> tuple[Optional[float], Optional[float]]:
    containers = _gather_containers(pos)
    mae: Optional[float] = None
    mfe: Optional[float] = None

    mae_keys = (
        'max_unrealized_loss',
        'max_adverse_excursion',
        'mae',
        'max_drawdown',
        'worst_unrealized_pnl',
    )
    mfe_keys = (
        'max_unrealized_pnl',
        'max_favorable_excursion',
        'mfe',
        'best_unrealized_pnl',
        'peak_unrealized_pnl',
    )

    for container in containers:
        for key in mae_keys:
            value = _coerce_numeric(_get_value(container, key))
            if value is not None:
                mae = value if mae is None else min(mae, value)
        for key in mfe_keys:
            value = _coerce_numeric(_get_value(container, key))
            if value is not None:
                mfe = value if mfe is None else max(mfe, value)

    return mae, mfe


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_mae_mfe_from_history(
    cb: CoinbaseService,
    product_id: str,
    net_size: float,
    entry_price: Optional[float],
    open_time: Optional[datetime],
    close_time: datetime,
    exit_price: Optional[float] = None,
    granularity: str = 'ONE_MINUTE',
) -> tuple[Optional[float], Optional[float]]:
    """Derive MAE/MFE PnL excursions using historical candles.

    Returns tuple of (mae, mfe) quoted in the same units as trade PnL.
    """

    logger = logging.getLogger(__name__)

    if net_size == 0 or entry_price is None:
        return None, None

    start = _as_utc(open_time)
    end = _as_utc(close_time)
    if start is None or end is None:
        return None, None

    # Add small buffer to capture immediate pre/post trade ticks
    start -= timedelta(minutes=1)
    end += timedelta(minutes=1)
    if end <= start:
        end = start + timedelta(minutes=1)

    mae: Optional[float] = None
    mfe: Optional[float] = None

    try:
        candles = cb.historical_data.get_historical_data(
            product_id,
            start,
            end,
            granularity,
        )
    except Exception as exc:
        logger.warning("Failed to fetch candles for %s: %s", product_id, exc)
        candles = []

    for candle in candles or []:
        if isinstance(candle, dict):
            low = _coerce_numeric(candle.get('low'))
            high = _coerce_numeric(candle.get('high'))
        else:
            low = _coerce_numeric(getattr(candle, 'low', None))
            high = _coerce_numeric(getattr(candle, 'high', None))
        for price in (low, high):
            if price is None:
                continue
            pnl = net_size * (price - entry_price)
            mae = pnl if mae is None or pnl < mae else mae
            mfe = pnl if mfe is None or pnl > mfe else mfe

    if exit_price is not None:
        pnl = net_size * (exit_price - entry_price)
        mae = pnl if mae is None or pnl < mae else mae
        mfe = pnl if mfe is None or pnl > mfe else mfe

    return mae, mfe


def _calculate_pnl(net_size: float, entry_price: Optional[float], exit_price: Optional[float]) -> Optional[float]:
    if entry_price is None or exit_price is None or net_size == 0:
        return None
    return net_size * (exit_price - entry_price)


def _calculate_pnl_pct(net_size: float, entry_price: Optional[float], exit_price: Optional[float]) -> Optional[float]:
    if entry_price is None or entry_price == 0 or exit_price is None or net_size == 0:
        return None
    direction = 1.0 if net_size > 0 else -1.0
    return direction * ((exit_price - entry_price) / entry_price) * 100.0


def _normalize_side(position_side: str, net_size: float) -> str:
    if position_side:
        upper = position_side.upper()
        if 'SHORT' in upper:
            return 'SHORT'
        if 'LONG' in upper:
            return 'LONG'
    return 'LONG' if net_size >= 0 else 'SHORT'


def _format_float(value: Optional[float], precision: int) -> str:
    if value is None:
        return ''
    formatted = f"{value:.{precision}f}"
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    if formatted in ('-0', '-0.0', '0.0'):
        return '0'
    return formatted


def _determine_closure_reason(pos: Any, fallback: str = 'expired') -> str:
    candidates = []
    for key in ('exit_reason', 'close_reason', 'closure_reason'):
        candidates.append(_get_value(pos, key))
    for parent in ('position_pnl', 'metadata', 'details'):
        container = _get_value(pos, parent)
        if container:
            candidates.append(_get_value(container, 'exit_reason'))
            candidates.append(_get_value(container, 'close_reason'))
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate).lower()
        if 'take' in text or 'tp' in text:
            return 'take_profit'
        if 'stop' in text or 'sl' in text:
            return 'stop_loss'
    return fallback


def _apply_breakeven_adjustment(
    closure_reason: str,
    pnl: Optional[float],
    entry_price: Optional[float],
    exit_price: Optional[float],
    net_size: float,
) -> tuple[Optional[float], Optional[float], str]:
    if pnl is None:
        return pnl, exit_price, closure_reason

    reason_normalized = (closure_reason or '').lower()
    if 'expired' not in reason_normalized:
        return pnl, exit_price, closure_reason

    threshold = _breakeven_threshold()
    if abs(pnl) > threshold:
        return pnl, exit_price, closure_reason

    adjusted_reason = 'expired_breakeven'
    adjusted_exit = entry_price if entry_price is not None else exit_price
    adjusted_pnl = 0.0
    return adjusted_pnl, adjusted_exit, adjusted_reason


def _format_datetime(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0).isoformat() + 'Z'


def _create_closure_record(
    product_id: str,
    position_side: str,
    net_size: float,
    leverage: str,
    opened_at: Optional[datetime],
    close_time: datetime,
    entry_price: Optional[float],
    exit_price: Optional[float],
    pnl: Optional[float],
    closure_reason: str,
    mae: Optional[float],
    mfe: Optional[float],
) -> Dict[str, str]:
    if pnl is None:
        pnl = _calculate_pnl(net_size, entry_price, exit_price)
    if exit_price is None and pnl is not None and entry_price is not None and net_size != 0:
        exit_price = entry_price + (pnl / net_size)
    pnl_pct = _calculate_pnl_pct(net_size, entry_price, exit_price)

    opened_str = ''
    if opened_at is not None:
        opened_str = _format_datetime(opened_at)

    closed_str = _format_datetime(close_time)
    duration_seconds: Optional[int] = None
    if opened_at is not None:
        duration_seconds = int((close_time - opened_at).total_seconds())

    record: Dict[str, str] = {
        'closed_at': closed_str,
        'product_id': product_id,
        'position_side': _normalize_side(position_side, net_size),
        'net_size': _format_float(net_size, 8),
        'leverage': leverage or '',
        'opened_at': opened_str,
        'closure_reason': closure_reason,
        'entry_price': _format_float(entry_price, 6),
        'exit_price': _format_float(exit_price, 6),
        'profit_loss': _format_float(pnl, 2),
        'profit_loss_pct': _format_float(pnl_pct, 4),
        'mae': _format_float(mae, 2),
        'mfe': _format_float(mfe, 2),
        'duration_seconds': str(duration_seconds) if duration_seconds is not None else '',
    }
    return record


def _record_position_close(record: Dict[str, str]) -> None:
    path = _ensure_log_file()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_HEADERS)
        writer.writerow(record)


def _order_close_success(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, dict):
        if 'success' in result:
            return bool(result['success'])
        if result.get('failure_reason'):
            return False
        if result.get('order_id') or result.get('order_configuration'):
            return True
        return True

    success_attr = _get_value(result, 'success')
    if success_attr is not None:
        try:
            return bool(success_attr)
        except Exception:
            return True

    failure_reason = _get_value(result, 'failure_reason')
    if failure_reason:
        return False

    status = _get_value(result, 'status')
    if isinstance(status, str) and status.upper() in {'FILLED', 'OPEN', 'PENDING'}:
        return True

    # Default to success if API didn't provide an explicit failure flag
    return True


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


def _to_datetime(order: Any) -> Optional[datetime]:
    # Prefer completion_time, fallback to created_time
    def g(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)
    for key in ('completion_time', 'created_time'):
        dt = _parse_iso8601(g(order, key))
        if dt:
            return dt
    return None


def _format_duration_hms(td: timedelta) -> str:
    """Return a human-readable string like '5 hours, 3 minutes, 10 seconds'.

    Always includes hours, minutes, and seconds (with pluralization), even if zero.
    """
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours == 1:
        hours_str = "1 hour"
    else:
        hours_str = f"{hours} hours"

    if minutes == 1:
        minutes_str = "1 minute"
    else:
        minutes_str = f"{minutes} minutes"

    if seconds == 1:
        seconds_str = "1 second"
    else:
        seconds_str = f"{seconds} seconds"

    return ", ".join([hours_str, minutes_str, seconds_str])


def _orders_for_product(cb: CoinbaseService, portfolio_uuid: str, product_id: str, limit: int = 200) -> list[Any]:
    logger = logging.getLogger(__name__)
    try:
        orders = cb.client.list_orders(
            portfolio_uuid=portfolio_uuid,
            product_id=product_id,
            order_status="FILLED",
            limit=limit,
        )
        if isinstance(orders, dict):
            return orders.get('orders', []) or []
        if hasattr(orders, 'orders'):
            return getattr(orders, 'orders') or []
        if hasattr(orders, '__dict__'):
            return getattr(orders, '__dict__', {}).get('orders', []) or []
    except Exception as e:
        logger.warning(f"Failed to fetch orders for {product_id}: {e}")
    return []


def _infer_open_time_from_orders(cb: CoinbaseService, portfolio_uuid: str, product_id: str, expected_net: float, position_side: str) -> Optional[datetime]:
    """Infer current position open time by replaying filled orders chronologically.

    Maintains a running net base size; returns the timestamp when the position last
    crossed from 0 to non-zero (start of current holding). If inference fails,
    returns None.
    """
    orders = _orders_for_product(cb, portfolio_uuid, product_id, limit=500)
    if not orders:
        return None

    # Sort ascending by time
    def order_time(o: Any) -> float:
        dt = _to_datetime(o)
        return dt.timestamp() if dt else 0.0

    orders_sorted = sorted(orders, key=order_time)

    def g(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    running = 0.0
    open_start: Optional[datetime] = None

    for o in orders_sorted:
        side = (g(o, 'side') or '').upper()
        # base_size may appear as filled_size or base_size
        try:
            base_size = float(g(o, 'filled_size') or g(o, 'base_size') or 0.0)
        except Exception:
            base_size = 0.0
        if base_size <= 0:
            continue
        delta = base_size if side == 'BUY' else -base_size

        prev_running = running
        running = running + delta
        # Detect zero -> non-zero transition as start of current holding window
        if prev_running == 0.0 and running != 0.0:
            open_start = _to_datetime(o)
        # Detect non-zero -> zero transition resets window
        if running == 0.0:
            open_start = None

    # Validate expected direction and magnitude loosely; tolerate rounding
    try:
        if abs(abs(running) - abs(expected_net)) <= max(0.0001, 0.02 * abs(expected_net)):
            return open_start
    except Exception:
        pass

    # Fallback heuristic: accumulate orders of the current position side from newest backward
    want_side = 'SELL' if position_side == 'FUTURES_POSITION_SIDE_SHORT' else 'BUY'
    acc = 0.0
    for o in sorted(orders_sorted, key=order_time, reverse=True):
        side = (g(o, 'side') or '').upper()
        try:
            base_size = float(g(o, 'filled_size') or g(o, 'base_size') or 0.0)
        except Exception:
            base_size = 0.0
        if side != want_side or base_size <= 0:
            continue
        acc += base_size
        ts = _to_datetime(o)
        if acc >= abs(expected_net):
            return ts
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
        if _order_close_success(result):
            logger.info(f"Closed {product_id} position via {side} {close_size}")
            return True
        logger.error(f"Close order did not report success for {product_id}: {result}")
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

    now_utc = datetime.utcnow()
    cutoff = now_utc - timedelta(hours=max_age_hours)
    logger.info(f"Closing positions opened before {cutoff.isoformat()}Z")

    for pos in positions:
        symbol, net_size, position_side, leverage = _extract_symbol_and_size(pos)
        if not symbol or abs(net_size) <= 0:
            continue
        if product_filter and symbol != product_filter:
            continue

        opened_at = _extract_position_open_time(pos)
        if not opened_at:
            # Try inference from order history
            opened_at = _infer_open_time_from_orders(cb, portfolio_uuid, symbol, net_size, position_side)
            if not opened_at:
                logger.warning(f"No open/entry timestamp found for {symbol}; skipping")
                continue

        if opened_at <= cutoff:
            logger.info(f"Position {symbol} opened at {_format_datetime(opened_at)} exceeds {max_age_hours}h; closing...")
            entry_price = _extract_entry_price(pos)
            mark_price = _extract_mark_price(pos)
            unrealized_pnl = _extract_unrealized_pnl(pos, net_size, entry_price, mark_price)
            mae, mfe = _extract_excursions(pos)
            closure_reason = _determine_closure_reason(pos, fallback='expired')
            pnl_for_record = unrealized_pnl
            if pnl_for_record is None:
                pnl_for_record = _calculate_pnl(net_size, entry_price, mark_price)
            pnl_for_record, mark_price, closure_reason = _apply_breakeven_adjustment(
                closure_reason,
                pnl_for_record,
                entry_price,
                mark_price,
                net_size,
            )
            closed = _close_position(cb, symbol, net_size, position_side, leverage)
            if closed:
                close_time = datetime.utcnow()
                hist_mae, hist_mfe = compute_mae_mfe_from_history(
                    cb=cb,
                    product_id=symbol,
                    net_size=net_size,
                    entry_price=entry_price,
                    open_time=opened_at,
                    close_time=close_time,
                    exit_price=mark_price,
                )
                if mae is None:
                    mae = hist_mae
                if mfe is None:
                    mfe = hist_mfe
                record = _create_closure_record(
                    product_id=symbol,
                    position_side=position_side,
                    net_size=net_size,
                    leverage=leverage,
                    opened_at=opened_at,
                    close_time=close_time,
                    entry_price=entry_price,
                    exit_price=mark_price,
                    pnl=pnl_for_record,
                    closure_reason=closure_reason,
                    mae=mae,
                    mfe=mfe,
                )
                _record_position_close(record)
                logger.info(f"Recorded closure for {symbol} to {_log_file_path()}")
        else:
            # Report time remaining until threshold
            deadline = opened_at + timedelta(hours=max_age_hours)
            remaining = deadline - now_utc
            # Clamp negative to zero
            if remaining.total_seconds() < 0:
                remaining = timedelta(seconds=0)
            # Format as human-readable H/M/S
            remaining_str = _format_duration_hms(remaining)
            logger.info(
                f"Position {symbol} time remaining to {max_age_hours}h threshold: {remaining_str} (opened {_format_datetime(opened_at)})"
            )


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
