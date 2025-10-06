#!/usr/bin/env python3
"""Watchdog: Log TP/SL Closures From Coinbase Fills.

Polls INTX fills, detects when a round-trip position closes (returning net size
to zero), and appends structured records to `trade_logs/watchdog_closed_positions.csv`.

Designed to complement `watchdog_close_old_positions.py`, which handles time
stop exits. This script captures take-profit and stop-loss completions driven by
exchange brackets or manual closures reflected in the fills feed.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

from fills_pnl import fetch_fills
from watchdog_close_old_positions import (
    _breakeven_threshold,
    _create_closure_record,
    compute_mae_mfe_from_history,
    _record_position_close,
    setup_logging,
)


UTC = datetime.utcnow().astimezone().tzinfo  # best effort; fetch_fills returns tz-aware
CHECKPOINT_PATH = Path('trade_logs') / 'watchdog_tp_sl_checkpoint.json'


@dataclass
class Fill:
    product_id: str
    side: str
    size: float
    price: float
    fee: float
    time: datetime
    order_id: str


@dataclass
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
        return cycle.closing_order_id > last_order_id
    return False


def _convert_fill(raw: Dict[str, Any]) -> Optional[Fill]:
    try:
        product_id = raw['product_id']
        side = raw['side'].upper()
        size = float(raw['size'])
        price = float(raw['price'])
        fee = float(raw.get('fee', 0.0))
        when = raw['time']
        if isinstance(when, str):
            when_dt = datetime.fromisoformat(when.replace('Z', '+00:00'))
        else:
            when_dt = when
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
        logging.getLogger(__name__).warning(f"Skipping fill due to parse error: {exc}; data={raw}")
        return None


def _classify_reason(side: str, pnl: float, threshold: float) -> str:
    if abs(pnl) <= threshold:
        return 'expired_breakeven'
    if pnl > 0:
        return 'take_profit'
    return 'stop_loss'


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
    long_inventory: Deque[Dict[str, float]] = deque()
    short_inventory: Deque[Dict[str, float]] = deque()
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
            # First close any outstanding short inventory
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
        else:  # SELL
            remaining = fill.size
            # Close long inventory first
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

        # If both inventories empty, cycle finished
        if long_qty == 0.0 and short_qty == 0.0:
            close_cycle(fill.time)

    return cycles


def _detect_cycles(fills: List[Fill]) -> List[Cycle]:
    grouped: Dict[str, List[Fill]] = defaultdict(list)
    for fill in fills:
        grouped[fill.product_id].append(fill)

    cycles: List[Cycle] = []
    for pfills in grouped.values():
        cycles.extend(_process_product_fills(pfills))
    cycles.sort(key=lambda c: c.end_time)
    return cycles


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


def _run_once(limit: int, bootstrap_existing: bool) -> None:
    logger = logging.getLogger(__name__)
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
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
        if not checkpoint:
            # first run without bootstrap: set checkpoint to latest to avoid duplicates later
            latest = cycles[-1]
            _store_checkpoint(latest.end_time, latest.closing_order_id)
            logger.info("Initial checkpoint stored; rerun to log new TP/SL closures")
        else:
            logger.debug("No new cycles beyond checkpoint")
        return

    def _mae_mfe_fetcher(**kwargs: Any) -> tuple[Optional[float], Optional[float]]:
        return compute_mae_mfe_from_history(cb=cb, **kwargs)

    for cycle in new_cycles:
        record = _cycle_to_record(cycle, threshold, mae_mfe_fetcher=_mae_mfe_fetcher)
        _record_position_close(record)
        logger.info(
            "Recorded TP/SL closure for %s at %s (reason=%s, pnl=%s)",
            cycle.product_id,
            cycle.end_time.isoformat(),
            record['closure_reason'],
            record['profit_loss'],
        )

    latest_logged = new_cycles[-1]
    _store_checkpoint(latest_logged.end_time, latest_logged.closing_order_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log TP/SL closures from Coinbase fills")
    parser.add_argument('--limit', type=int, default=500, help='Number of recent fills to fetch (default 500)')
    parser.add_argument('--interval-seconds', type=int, default=0, help='If >0, poll continuously every N seconds')
    parser.add_argument(
        '--bootstrap-existing',
        action='store_true',
        help='On first run, log existing cycles instead of only new ones',
    )
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.interval_seconds and args.interval_seconds > 0:
        while True:
            try:
                _run_once(limit=args.limit, bootstrap_existing=args.bootstrap_existing)
            except Exception as exc:
                logging.getLogger(__name__).error(f"TP/SL watchdog iteration error: {exc}")
            time.sleep(args.interval_seconds)
    else:
        _run_once(limit=args.limit, bootstrap_existing=args.bootstrap_existing)


if __name__ == '__main__':
    main()
