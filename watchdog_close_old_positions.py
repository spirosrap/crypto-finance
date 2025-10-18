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
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
from fills_pnl import fetch_fills


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


CHECKPOINT_PATH = Path('trade_logs') / 'watchdog_tp_sl_checkpoint.json'
UTC = timezone.utc


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


def _extract_avg_filled_price(container: Any) -> Optional[float]:
    """Best-effort extraction of an average fill price from order/fill payloads."""

    price_keys = (
        'average_filled_price',
        'avg_filled_price',
        'average_price',
        'avg_price',
        'price',
        'fill_price',
        'execution_price',
    )
    for key in price_keys:
        price = _coerce_numeric(_get_value(container, key))
        if price is not None:
            return price

    nested_order = _get_value(container, 'order')
    if nested_order is not None:
        price = _extract_avg_filled_price(nested_order)
        if price is not None:
            return price

    filled_value = _coerce_numeric(_get_value(container, 'filled_value'))
    filled_size = _coerce_numeric(_get_value(container, 'filled_size'))
    if filled_value is not None and filled_size not in (None, 0):
        try:
            return float(filled_value) / float(filled_size)
        except ZeroDivisionError:
            return None

    return None


def _extract_order_id(container: Any) -> Optional[str]:
    """Pull a string order identifier from API responses if available."""

    candidates = (
        'order_id',
        'orderId',
        'orderID',
        'id',
    )
    for key in candidates:
        value = _get_value(container, key)
        if value:
            return str(value)

    nested = _get_value(container, 'order')
    if nested:
        return _extract_order_id(nested)
    return None


def _extract_fills_from_response(resp: Any) -> List[Dict[str, Any]]:
    """Normalize fills response structures into a list of dicts."""

    if resp is None:
        return []

    raw: List[Any]
    if isinstance(resp, dict):
        if 'fills' in resp:
            raw = resp.get('fills') or []
        elif 'data' in resp and isinstance(resp['data'], list):
            raw = resp['data']
        else:
            raw = []
    elif hasattr(resp, 'fills'):
        raw = getattr(resp, 'fills') or []
    elif isinstance(resp, list):
        raw = resp
    else:
        raw = []

    fills: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            fills.append(item)
        else:
            try:
                fills.append(vars(item))
            except TypeError:
                fills.append({})
    return fills


def _extract_fill_size(fill: Dict[str, Any]) -> Optional[float]:
    size_keys = (
        'filled_size',
        'size',
        'base_size',
        'quantity',
        'base_quantity',
    )
    for key in size_keys:
        value = _coerce_numeric(fill.get(key))
        if value is not None:
            return float(value)
    return None


def _fill_time(fill: Dict[str, Any]) -> Optional[datetime]:
    time_keys = (
        'trade_time',
        'time',
        'created_time',
        'ts',
        'completion_time',
    )
    for key in time_keys:
        value = fill.get(key)
        if not value:
            continue
        dt = _parse_iso8601(value)
        if dt is not None:
            return dt
        # Accept epoch seconds
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except Exception:
            continue
    return None


def _average_fill_price(fills: List[Dict[str, Any]], target_size: Optional[float] = None) -> Optional[float]:
    if not fills:
        return None

    total_value = 0.0
    total_size = 0.0

    for fill in fills:
        price = _extract_avg_filled_price(fill)
        size = _extract_fill_size(fill)
        if price is None or size is None or size == 0:
            continue
        abs_size = abs(float(size))
        total_value += float(price) * abs_size
        total_size += abs_size
        if target_size is not None and total_size >= abs(target_size) - 1e-9:
            break

    if total_size > 0:
        return total_value / total_size

    # Fallback: return first parsable price
    for fill in fills:
        price = _extract_avg_filled_price(fill)
        if price is not None:
            return float(price)

    return None


def _lookup_order_fill_price(cb: CoinbaseService, order_id: Optional[str], product_id: str) -> Optional[float]:
    """Attempt to recover the executed price for a close order via follow-up API calls."""

    logger = logging.getLogger(__name__)

    # Try fetching the order details directly first
    if order_id:
        try:
            order_details = cb.client.get_order(order_id=order_id)
            price = _extract_avg_filled_price(order_details)
            if price is not None:
                return price
        except TypeError:
            # Some SDKs require keyword variations; ignore and fall back
            pass
        except Exception as exc:
            logger.debug("get_order failed for %s: %s", order_id, exc)

    fills_fn = getattr(cb.client, 'list_fills', None) or getattr(cb.client, 'get_fills', None)
    if fills_fn is None:
        return None

    fills_resp = None
    try:
        if order_id is not None:
            fills_resp = fills_fn(order_id=order_id, product_id=product_id, limit=50)
        else:
            fills_resp = fills_fn(product_id=product_id, limit=50)
    except TypeError:
        try:
            if order_id is not None:
                fills_resp = fills_fn(order_id=order_id)
            else:
                fills_resp = fills_fn(limit=50)
        except Exception as exc:
            logger.debug("list_fills fallback failed: %s", exc)
            fills_resp = None
    except Exception as exc:
        logger.debug("list_fills failed: %s", exc)

    fills = _extract_fills_from_response(fills_resp)
    if not fills:
        return None

    def _match_fill(fill: Dict[str, Any]) -> bool:
        if order_id:
            fid = str(fill.get('order_id') or fill.get('orderId') or fill.get('orderID') or '')
            if fid == order_id:
                return True
        pid = fill.get('product_id') or fill.get('productId') or fill.get('symbol')
        return pid == product_id

    matched_fills = [fill for fill in fills if _match_fill(fill)]
    price = _average_fill_price(matched_fills, target_size=None)
    if price is not None:
        return price

    return _average_fill_price(fills, target_size=None)


def _lookup_recent_fill_price(
    cb: CoinbaseService,
    product_id: str,
    close_time: datetime,
    net_size: float,
    target_size: Optional[float],
    lookback_seconds: int = 600,
) -> Optional[float]:
    fills_fn = getattr(cb.client, 'list_fills', None) or getattr(cb.client, 'get_fills', None)
    if fills_fn is None:
        return None

    try:
        fills_resp = fills_fn(product_id=product_id, limit=200)
    except TypeError:
        try:
            fills_resp = fills_fn(limit=200)
        except Exception:
            return None
    except Exception:
        return None

    fills = _extract_fills_from_response(fills_resp)
    if not fills:
        return None

    window_start = close_time - timedelta(seconds=lookback_seconds)
    window_end = close_time + timedelta(seconds=lookback_seconds)

    matched: List[Dict[str, Any]] = []
    for fill in fills:
        pid = fill.get('product_id') or fill.get('productId') or fill.get('symbol')
        if pid != product_id:
            continue
        dt = _fill_time(fill)
        if dt is None:
            continue
        if window_start <= dt <= window_end:
            matched.append(fill)

    if not matched:
        return None

    close_side = 'BUY' if net_size < 0 else 'SELL'
    matched.sort(key=lambda fill: (_fill_time(fill) or close_time))
    closing_fills: List[Dict[str, Any]] = []
    accumulated = 0.0
    for fill in matched:
        side = str(fill.get('side') or '').upper()
        if side != close_side:
            continue
        size = _extract_fill_size(fill)
        if size is None or size <= 0:
            continue
        closing_fills.append(fill)
        accumulated += size
        if target_size and accumulated >= abs(target_size) - 1e-9:
            break

    if closing_fills:
        return _average_fill_price(closing_fills, target_size=target_size)

    return _average_fill_price(matched, target_size=target_size)


def _lookup_cycle_details(
    cb: CoinbaseService,
    product_id: str,
    open_time: Optional[datetime],
    close_time: datetime,
    net_size: float,
    tolerance_seconds: int = 120,
) -> Optional[Cycle]:
    fills_raw = fetch_fills(cb, limit=500)
    if not fills_raw:
        return None

    def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)

    close_utc = _ensure_utc(close_time)
    open_utc = _ensure_utc(open_time)
    pre_buffer = timedelta(seconds=tolerance_seconds)
    post_buffer = timedelta(seconds=tolerance_seconds)
    if open_utc is not None:
        window_start = open_utc - pre_buffer
    else:
        window_start = close_utc - timedelta(hours=6)
    window_end = close_utc + post_buffer

    filtered: List[Fill] = []
    for raw in fills_raw:
        if raw.get('product_id') != product_id:
            continue
        ts = raw.get('time')
        if ts is None:
            continue
        ts_utc = _ensure_utc(ts)
        if ts_utc is None or ts_utc < window_start or ts_utc > window_end:
            continue
        filtered.append(
            Fill(
                product_id=raw.get('product_id', ''),
                side=str(raw.get('side') or '').upper(),
                size=float(raw.get('size') or 0.0),
                price=float(raw.get('price') or 0.0),
                fee=float(raw.get('fee') or 0.0),
                time=ts_utc,
                order_id=str(raw.get('order_id') or raw.get('trade_id') or ''),
            )
        )

    if not filtered:
        return None

    cycles = _detect_cycles(filtered)
    if not cycles:
        return None

    target_qty = abs(float(net_size))
    target_sign = 1 if net_size >= 0 else -1
    tolerance_time = timedelta(seconds=tolerance_seconds)
    qty_tolerance = max(1e-6, target_qty * 0.001 + 1e-3)

    for cycle in cycles:
        end_time = cycle.end_time
        if abs((end_time - close_utc).total_seconds()) > tolerance_seconds:
            continue
        cycle_qty = cycle.entry_qty
        cycle_sign = 1 if cycle.side == 'LONG' else -1
        if target_sign != cycle_sign:
            continue
        if abs(cycle_qty - target_qty) > qty_tolerance:
            continue
        return cycle

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


def _rewrite_log_rows(rows: List[Dict[str, str]]) -> None:
    path = _ensure_log_file()
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def _reason_priority(reason: str) -> int:
    order = {
        'take_profit': 3,
        'stop_loss': 3,
        'manual': 2,
        'manual_close': 2,
        'expired': 0,
        '': 0,
    }
    normalized = (reason or '').strip().lower()
    return order.get(normalized, 1)


def _parse_log_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_log_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    return _parse_iso8601(value)


def _float_close(a: Optional[float], b: Optional[float], rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1.0))


def _time_close(a: Optional[datetime], b: Optional[datetime], seconds: int) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs((a - b).total_seconds()) <= seconds


def _record_position_close_if_new(record: Dict[str, str], tolerance_seconds: int = 60) -> bool:
    """Append closure record only if a similar entry does not already exist."""

    path = _ensure_log_file()
    closed_at = _parse_log_datetime(record.get('closed_at', ''))
    product_id = record.get('product_id', '')
    if closed_at is None or not product_id:
        _record_position_close(record)
        return True

    tolerance = abs(int(tolerance_seconds))
    rows: List[Dict[str, str]] = []
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    for idx, row in enumerate(rows):
        if (row.get('product_id') or '') != product_id:
            continue
        row_closed = _parse_log_datetime(row.get('closed_at', ''))
        if row_closed is None:
            continue
        if abs((row_closed - closed_at).total_seconds()) <= tolerance:
            existing_net = _parse_log_float(row.get('net_size', ''))
            current_net = _parse_log_float(record.get('net_size', ''))
            same_net = _float_close(existing_net, current_net)

            existing_entry = _parse_log_float(row.get('entry_price', ''))
            current_entry = _parse_log_float(record.get('entry_price', ''))
            same_entry = _float_close(existing_entry, current_entry)

            existing_exit = _parse_log_float(row.get('exit_price', ''))
            current_exit = _parse_log_float(record.get('exit_price', ''))
            same_exit = _float_close(existing_exit, current_exit)

            existing_open = _parse_log_datetime(row.get('opened_at', ''))
            current_open = _parse_log_datetime(record.get('opened_at', ''))
            same_open = _time_close(existing_open, current_open, tolerance)

            if same_net and same_entry and same_exit and same_open:
                existing_reason = (row.get('closure_reason') or '').strip().lower()
                current_reason = (record.get('closure_reason') or '').strip().lower()
                if _reason_priority(current_reason) > _reason_priority(existing_reason):
                    updated = {field: record.get(field, row.get(field, '')) or '' for field in LOG_HEADERS}
                    rows[idx] = updated
                    _rewrite_log_rows(rows)
                return False

    _record_position_close(record)
    return True


@dataclass(frozen=True)
class Fill:
    product_id: str
    side: str
    size: float
    price: float
    fee: float
    time: datetime
    order_id: str


@dataclass(frozen=True)
class Cycle:
    product_id: str
    side: str  # 'LONG' or 'SHORT'
    start_time: datetime
    end_time: datetime
    entry_qty: float
    entry_value: float
    exit_qty: float
    exit_value: float
    realized_pnl: float
    fees: float
    closing_order_id: str


def _checkpoint_path() -> Path:
    path = CHECKPOINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_checkpoint() -> Dict[str, Any]:
    path = _checkpoint_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        logging.getLogger(__name__).warning("Could not parse checkpoint; starting fresh")
        return {}


def _store_checkpoint(last_time: datetime, last_order_id: str) -> None:
    data = {
        'last_time': last_time.isoformat(),
        'last_order_id': last_order_id,
    }
    _checkpoint_path().write_text(json.dumps(data, indent=2))


def _is_new_cycle(cycle: Cycle, checkpoint: Dict[str, Any], bootstrap_existing: bool) -> bool:
    if not checkpoint:
        return bootstrap_existing

    last_time_raw = checkpoint.get('last_time')
    last_order_id = checkpoint.get('last_order_id')
    if not last_time_raw:
        return True

    try:
        last_time = datetime.fromisoformat(last_time_raw)
    except ValueError:
        return True

    if cycle.end_time > last_time:
        return True
    if cycle.end_time == last_time:
        if not last_order_id:
            return True
        # Coinbase order IDs are UUIDs; lexical ordering does not guarantee chronology.
        # Treat any differing order_id at the same timestamp as a new cycle to avoid misses.
        return cycle.closing_order_id != last_order_id
    return False


def _convert_fill(raw: Dict[str, Any]) -> Optional[Fill]:
    try:
        product_id = raw['product_id']
        side = str(raw['side']).upper()
        size = float(raw['size'])
        price = float(raw['price'])
        fee = float(raw.get('fee', 0.0))
        when = raw['time']
        if isinstance(when, str):
            when_dt = datetime.fromisoformat(when.replace('Z', '+00:00'))
        else:
            when_dt = when
        if when_dt.tzinfo is None:
            when_dt = when_dt.replace(tzinfo=UTC)
        order_id = str(raw.get('order_id') or raw.get('trade_id') or '')
        return Fill(
            product_id=product_id,
            side=side,
            size=size,
            price=price,
            fee=fee,
            time=when_dt,
            order_id=order_id,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning("Skipping fill due to parse error: %s; data=%s", exc, raw)
        return None


def _finalize_cycle(
    product_id: str,
    side: str,
    start_time: datetime,
    end_time: datetime,
    entry_qty: float,
    entry_value: float,
    exit_qty: float,
    exit_value: float,
    realized_pnl: float,
    total_fees: float,
    closing_order_id: str,
) -> Optional[Cycle]:
    qty = max(entry_qty, exit_qty)
    if qty <= 1e-12:
        return None
    return Cycle(
        product_id=product_id,
        side=side,
        start_time=start_time,
        end_time=end_time,
        entry_qty=entry_qty,
        entry_value=entry_value,
        exit_qty=exit_qty,
        exit_value=exit_value,
        realized_pnl=realized_pnl - total_fees,
        fees=total_fees,
        closing_order_id=closing_order_id,
    )


def _process_product_fills(fills: Iterable[Fill]) -> List[Cycle]:
    fills_sorted = sorted(fills, key=lambda f: f.time)
    long_inventory: deque[Dict[str, float]] = deque()
    short_inventory: deque[Dict[str, float]] = deque()
    long_qty = 0.0
    short_qty = 0.0

    cycles: List[Cycle] = []
    cycle_side: Optional[str] = None
    cycle_start: Optional[datetime] = None
    entry_qty = 0.0
    entry_value = 0.0
    exit_qty = 0.0
    exit_value = 0.0
    realized = 0.0
    total_fees = 0.0
    last_fill_id = ''

    def close_cycle(end_time: datetime) -> None:
        nonlocal cycle_side, cycle_start, entry_qty, entry_value, exit_qty, exit_value, realized, total_fees, last_fill_id
        if cycle_side and cycle_start:
            cycle = _finalize_cycle(
                product_id=fills_sorted[0].product_id if fills_sorted else '',
                side=cycle_side,
                start_time=cycle_start,
                end_time=end_time,
                entry_qty=entry_qty,
                entry_value=entry_value,
                exit_qty=exit_qty,
                exit_value=exit_value,
                realized_pnl=realized,
                total_fees=total_fees,
                closing_order_id=last_fill_id,
            )
            if cycle:
                cycles.append(cycle)
        cycle_side = None
        cycle_start = None
        entry_qty = 0.0
        entry_value = 0.0
        exit_qty = 0.0
        exit_value = 0.0
        realized = 0.0
        total_fees = 0.0
        last_fill_id = ''

    for fill in fills_sorted:
        last_fill_id = fill.order_id or ''
        total_fees += fill.fee

        if long_qty == 0.0 and short_qty == 0.0:
            cycle_side = 'LONG' if fill.side == 'BUY' else 'SHORT'
            cycle_start = fill.time

        if fill.side == 'BUY':
            remaining = fill.size
            while remaining > 1e-12 and short_inventory:
                lot = short_inventory[0]
                match_qty = min(remaining, lot['qty'])
                realized += (lot['price'] - fill.price) * match_qty
                exit_qty += match_qty
                exit_value += fill.price * match_qty
                lot['qty'] -= match_qty
                remaining -= match_qty
                short_qty -= match_qty
                if lot['qty'] <= 1e-12:
                    short_inventory.popleft()
            if short_qty <= 1e-12:
                short_qty = 0.0
            if remaining > 1e-12:
                long_inventory.append({'qty': remaining, 'price': fill.price})
                long_qty += remaining
                entry_qty += remaining
                entry_value += fill.price * remaining
        else:
            remaining = fill.size
            while remaining > 1e-12 and long_inventory:
                lot = long_inventory[0]
                match_qty = min(remaining, lot['qty'])
                realized += (fill.price - lot['price']) * match_qty
                exit_qty += match_qty
                exit_value += fill.price * match_qty
                lot['qty'] -= match_qty
                remaining -= match_qty
                long_qty -= match_qty
                if lot['qty'] <= 1e-12:
                    long_inventory.popleft()
            if long_qty <= 1e-12:
                long_qty = 0.0
            if remaining > 1e-12:
                short_inventory.append({'qty': remaining, 'price': fill.price})
                short_qty += remaining
                entry_qty += remaining
                entry_value += fill.price * remaining

        if long_qty == 0.0 and short_qty == 0.0:
            close_cycle(fill.time)

    return cycles


def _detect_cycles(fills: Iterable[Fill]) -> List[Cycle]:
    grouped: Dict[str, List[Fill]] = defaultdict(list)
    for fill in fills:
        grouped[fill.product_id].append(fill)

    cycles: List[Cycle] = []
    for pfills in grouped.values():
        cycles.extend(_process_product_fills(pfills))
    cycles.sort(key=lambda c: c.end_time)
    return cycles


def _classify_reason(side: str, pnl: float, threshold: float) -> str:
    if abs(pnl) <= threshold:
        return 'expired_breakeven'
    if pnl > 0:
        return 'take_profit'
    return 'stop_loss'


def _cycle_to_record(
    cycle: Cycle,
    pn_threshold: float,
    mae_mfe_fetcher: Optional[Callable[..., tuple[Optional[float], Optional[float]]]] = None,
) -> Dict[str, str]:
    entry_price = cycle.entry_value / cycle.entry_qty if cycle.entry_qty else None
    exit_price = cycle.exit_value / cycle.exit_qty if cycle.exit_qty else None
    net_size = cycle.entry_qty if cycle.side == 'LONG' else -cycle.entry_qty
    reason = _classify_reason(cycle.side, cycle.realized_pnl, pn_threshold)
    mae = None
    mfe = None

    if mae_mfe_fetcher:
        try:
            mae, mfe = mae_mfe_fetcher(
                product_id=cycle.product_id,
                net_size=net_size,
                entry_price=entry_price,
                open_time=cycle.start_time,
                close_time=cycle.end_time,
                exit_price=exit_price,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Failed to derive MAE/MFE for %s: %s", cycle.product_id, exc
            )

    record = _create_closure_record(
        product_id=cycle.product_id,
        position_side=cycle.side,
        net_size=net_size,
        leverage='',
        opened_at=cycle.start_time,
        close_time=cycle.end_time,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=cycle.realized_pnl,
        closure_reason=reason,
        mae=mae,
        mfe=mfe,
    )
    return record


def _log_tp_sl_once(cb: CoinbaseService, limit: int, bootstrap_existing: bool) -> None:
    logger = logging.getLogger(__name__)

    raw_fills = fetch_fills(cb, limit=limit)
    if not raw_fills:
        logger.info("No fills returned")
        return

    fills: List[Fill] = []
    for raw in raw_fills:
        converted = _convert_fill(raw)
        if converted:
            fills.append(converted)

    if not fills:
        logger.info("No fills parsed successfully")
        return

    cycles = _detect_cycles(fills)
    if not cycles:
        logger.debug("No closed cycles detected in recent fills")
        return

    checkpoint = _load_checkpoint()
    threshold = _breakeven_threshold()
    new_cycles = [c for c in cycles if _is_new_cycle(c, checkpoint, bootstrap_existing)]

    if not new_cycles:
        if not checkpoint and not bootstrap_existing:
            latest = cycles[-1]
            _store_checkpoint(latest.end_time, latest.closing_order_id)
            logger.info("Initial checkpoint stored; rerun to log new TP/SL closures")
        else:
            logger.debug("No new cycles beyond checkpoint")
        return

    def _mae_mfe_fetcher(**kwargs: Any) -> tuple[Optional[float], Optional[float]]:
        return compute_mae_mfe_from_history(cb=cb, **kwargs)

    appended = 0
    for cycle in new_cycles:
        record = _cycle_to_record(cycle, threshold, mae_mfe_fetcher=_mae_mfe_fetcher)
        if _record_position_close_if_new(record):
            appended += 1
            logger.info(
                "Recorded TP/SL closure for %s at %s (reason=%s, pnl=%s)",
                cycle.product_id,
                cycle.end_time.isoformat(),
                record['closure_reason'],
                record['profit_loss'],
            )
        else:
            logger.debug(
                "Skipped duplicate TP/SL closure for %s at %s",
                cycle.product_id,
                cycle.end_time.isoformat(),
            )

    latest_logged = new_cycles[-1]
    _store_checkpoint(latest_logged.end_time, latest_logged.closing_order_id)
    if appended:
        logger.info("TP/SL logging complete; appended %d new records", appended)
    else:
        logger.info("TP/SL logging found no new rows after deduplication")


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

    if isinstance(ts, datetime):
        return ts.astimezone(UTC) if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    s = str(ts).strip()
    if not s:
        return None

    normalized = s.replace('Z', '+00:00').replace('z', '+00:00')
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        fmts = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in fmts:
            try:
                parsed = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


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


def _backfill_last_entries(cb: CoinbaseService, count: int) -> None:
    logger = logging.getLogger(__name__)
    csv_path = _ensure_log_file()
    if not csv_path.exists():
        logger.info("No log file found to backfill")
        return

    with csv_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        logger.info("Log file is empty; nothing to backfill")
        return

    start_index = max(len(rows) - count, 0)
    updated = False

    for idx in range(start_index, len(rows)):
        row = rows[idx]
        product_id = row.get('product_id') or ''
        if not product_id:
            continue

        net_size = _parse_log_float(row.get('net_size', ''))
        entry_price = _parse_log_float(row.get('entry_price', ''))
        if net_size is None or entry_price is None:
            continue

        close_time = _parse_log_datetime(row.get('closed_at', ''))
        open_time = _parse_log_datetime(row.get('opened_at', ''))
        if close_time is None:
            continue

        current_pnl = _parse_log_float(row.get('profit_loss', ''))
        reason = row.get('closure_reason') or 'expired'

        entry_candidate = entry_price
        exit_candidate: Optional[float]
        pnl_candidate: Optional[float]
        reason_candidate = reason

        cycle = _lookup_cycle_details(cb, product_id, open_time, close_time, net_size)
        if cycle is not None and cycle.entry_qty > 0 and cycle.exit_qty > 0:
            entry_candidate = cycle.entry_value / cycle.entry_qty
            exit_candidate = cycle.exit_value / cycle.exit_qty
            pnl_candidate = cycle.realized_pnl
            open_time = cycle.start_time
            close_time = cycle.end_time
            reason_candidate = _classify_reason(cycle.side, pnl_candidate, _breakeven_threshold())
        else:
            target_size = abs(net_size)
            exit_candidate = _lookup_recent_fill_price(cb, product_id, close_time, net_size, target_size)
            if exit_candidate is None:
                logger.info(
                    "Backfill skipped for %s at %s (no fills found)",
                    product_id,
                    row.get('closed_at', ''),
                )
                continue
            pnl_candidate = _calculate_pnl(net_size, entry_candidate, exit_candidate)
            reason_candidate = reason

        if pnl_candidate is None:
            continue

        existing_exit = _parse_log_float(row.get('exit_price', ''))
        existing_entry = entry_price
        if (
            current_pnl is not None
            and abs(pnl_candidate - current_pnl) <= 1e-6
            and (
                (existing_exit is None and exit_candidate is None)
                or (existing_exit is not None and exit_candidate is not None and abs(existing_exit - exit_candidate) <= 1e-6)
            )
            and (
                existing_entry is None or abs(entry_candidate - existing_entry) <= 1e-6
            )
        ):
            continue

        pnl_adjusted, exit_adjusted, adjusted_reason = _apply_breakeven_adjustment(
            reason_candidate,
            pnl_candidate,
            entry_candidate,
            exit_candidate,
            net_size,
        )
        pnl_pct = _calculate_pnl_pct(net_size, entry_candidate, exit_adjusted)

        mae = _parse_log_float(row.get('mae', ''))
        mfe = _parse_log_float(row.get('mfe', ''))
        hist_mae, hist_mfe = compute_mae_mfe_from_history(
            cb=cb,
            product_id=product_id,
            net_size=net_size,
            entry_price=entry_candidate,
            open_time=open_time,
            close_time=close_time,
            exit_price=exit_adjusted,
        )
        if hist_mae is not None:
            mae = hist_mae
        if hist_mfe is not None:
            mfe = hist_mfe
        leverage = row.get('leverage', '')

        record = _create_closure_record(
            product_id=product_id,
            position_side=row.get('position_side', ''),
            net_size=net_size,
            leverage=leverage,
            opened_at=open_time,
            close_time=close_time,
            entry_price=entry_candidate,
            exit_price=exit_adjusted,
            pnl=pnl_adjusted,
            closure_reason=adjusted_reason,
            mae=mae,
            mfe=mfe,
        )

        # Preserve original closure timestamp format and duration if available
        record['closed_at'] = row.get('closed_at', record['closed_at'])
        record['duration_seconds'] = row.get('duration_seconds', record['duration_seconds'])
        record['profit_loss_pct'] = _format_float(pnl_pct, 4)

        rows[idx] = record
        updated = True
        logger.info(
            "Backfilled %s at %s -> pnl=%s",
            product_id,
            record['closed_at'],
            record['profit_loss'],
        )

    if not updated:
        logger.info("No rows required backfill adjustments")
        return

    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Backfill complete; updated %s", csv_path)


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

    normalized_side = (side_field or '').upper()
    if 'SHORT' in normalized_side:
        size = -abs(size)
    elif 'LONG' in normalized_side:
        size = abs(size)

    return symbol, size, normalized_side or side_field, leverage


def _close_position(
    cb: CoinbaseService,
    product_id: str,
    net_size: float,
    position_side: str,
    leverage: str,
) -> tuple[bool, Optional[float], Optional[str]]:
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
        order_id = _extract_order_id(result)
        fill_price = _extract_avg_filled_price(result)
        if _order_close_success(result):
            if fill_price is None:
                fill_price = _lookup_order_fill_price(cb, order_id, product_id)
            logger.info(
                "Closed %s position via %s %s at %s",
                product_id,
                side,
                close_size,
                f"price~{fill_price:.6f}" if fill_price is not None else "unknown price",
            )
            return True, fill_price, order_id
        logger.error(f"Close order did not report success for {product_id}: {result}")
        return False, None, order_id
    except Exception as e:
        logger.error(f"Error closing position for {product_id}: {e}")
        return False, None, None


def run_once(
    max_age_hours: int,
    product_filter: Optional[str],
    log_closures: bool = True,
) -> None:
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

    now_utc = datetime.now(UTC)
    cutoff = now_utc - timedelta(hours=max_age_hours)
    logger.info(f"Closing positions opened before {_format_datetime(cutoff)}")

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
            initial_pnl_estimate = unrealized_pnl
            if initial_pnl_estimate is None:
                initial_pnl_estimate = _calculate_pnl(net_size, entry_price, mark_price)

            closed, execution_price, close_order_id = _close_position(
                cb, symbol, net_size, position_side, leverage
            )
            if closed:
                close_time = datetime.now(UTC)
                exit_price = execution_price if execution_price is not None else mark_price
                if exit_price is None:
                    exit_price = _lookup_order_fill_price(cb, close_order_id, symbol)
                if exit_price is None and mark_price is not None:
                    exit_price = mark_price

                pnl_for_record = _calculate_pnl(net_size, entry_price, exit_price)
                if pnl_for_record is None:
                    pnl_for_record = initial_pnl_estimate

                pnl_for_record, exit_price, closure_reason = _apply_breakeven_adjustment(
                    closure_reason,
                    pnl_for_record,
                    entry_price,
                    exit_price,
                    net_size,
                )
                hist_mae, hist_mfe = compute_mae_mfe_from_history(
                    cb=cb,
                    product_id=symbol,
                    net_size=net_size,
                    entry_price=entry_price,
                    open_time=opened_at,
                    close_time=close_time,
                    exit_price=exit_price,
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
                    exit_price=exit_price,
                    pnl=pnl_for_record,
                    closure_reason=closure_reason,
                    mae=mae,
                    mfe=mfe,
                )
                if log_closures:
                    _record_position_close(record)
                    logger.info(f"Recorded closure for {symbol} to {_log_file_path()}")
                else:
                    logger.info(
                        "Closure logging disabled; skipping CSV append for %s (reason=%s, pnl=%s)",
                        symbol,
                        closure_reason,
                        pnl_for_record,
                    )
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
    ap.add_argument("--backfill-last", type=int, default=0,
                    help="Recompute exit price/PnL for the most recent N logged closures and exit")
    ap.add_argument("--skip-close", action="store_true", help="Skip the age-based closing step")
    ap.add_argument("--log-fills", action="store_true", help="Log take-profit/stop-loss closures from recent fills")
    ap.add_argument("--fills-limit", type=int, default=500, help="Number of recent fills to fetch when logging TP/SL closures")
    ap.add_argument("--fills-interval", type=int, default=0, help="If >0, poll fills continuously every N seconds")
    ap.add_argument("--fills-bootstrap-existing", action="store_true",
                    help="On first run, log existing fill cycles instead of only new ones")
    ap.add_argument("--no-log-closures", action="store_true",
                    help="Skip writing age-based closure rows to watchdog_closed_positions.csv")
    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = ap.parse_args()
    setup_logging(verbose=args.verbose)

    log_closures = not args.no_log_closures

    if args.log_fills and args.fills_interval > 0 and args.interval_seconds > 0 and not args.skip_close:
        ap.error("Cannot run fill logging loop and closing loop simultaneously; run separate processes or use --skip-close.")

    if int(args.backfill_last or 0) > 0:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        _backfill_last_entries(cb, int(args.backfill_last))
        return

    if args.log_fills:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)

        def _log_once() -> None:
            _log_tp_sl_once(cb, limit=int(args.fills_limit), bootstrap_existing=bool(args.fills_bootstrap_existing))

        if args.fills_interval and args.fills_interval > 0:
            while True:
                try:
                    _log_once()
                except Exception as exc:
                    logging.getLogger(__name__).error(f"TP/SL logging iteration error: {exc}")
                time.sleep(args.fills_interval)
        else:
            _log_once()

        if args.skip_close:
            return

    if args.skip_close:
        return

    if args.interval_seconds and args.interval_seconds > 0:
        while True:
            try:
                run_once(args.max_age_hours, args.product, log_closures=log_closures)
            except Exception as e:
                logging.getLogger(__name__).error(f"Watchdog iteration error: {e}")
            time.sleep(args.interval_seconds)
    else:
        run_once(args.max_age_hours, args.product, log_closures=log_closures)


if __name__ == "__main__":
    main()
