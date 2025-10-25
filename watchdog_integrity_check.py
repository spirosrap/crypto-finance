#!/usr/bin/env python3
"""Watchdog data integrity check.

This utility compares the most recent closed trade cycles detected from
Coinbase fills with the contents of ``watchdog_closed_positions.csv``.
If a closed cycle is missing from the CSV it raises a warning (and exits
with status code 1 when any gaps are found).

Example:

    python watchdog_integrity_check.py --lookback-hours 24 \
        --product EURC-PERP-INTX --limit 400

The script re-uses the cycle detection logic from
``watchdog_close_old_positions`` so the validation exactly mirrors the
production logging pipeline.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
from fills_pnl import fetch_fills
from watchdog_close_old_positions import (
    _apply_breakeven_adjustment,
    _breakeven_threshold,
    _convert_fill,
    _cycle_to_record,
    _detect_cycles,
    _ensure_log_file,
    _float_close,
    _format_float,
    _parse_log_datetime,
    _parse_log_float,
    _time_close,
)


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate watchdog CSV against recent Coinbase fills.")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to watchdog_closed_positions.csv (defaults to trade_logs/watchdog_closed_positions.csv).",
    )
    parser.add_argument(
        "--product",
        type=str,
        help="Limit validation to a specific product (e.g. EURC-PERP-INTX).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of recent fills to fetch from Coinbase (default: 500).",
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=24.0,
        help="Only validate cycles closed within the past N hours (default: 24).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print diagnostic details for matching cycles as well.",
    )
    return parser.parse_args()


def load_recent_cycles(limit: int) -> List[object]:
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    raw_fills = fetch_fills(cb, limit=limit)
    fills = []
    for raw in raw_fills:
        converted = _convert_fill(raw)
        if converted:
            fills.append(converted)

    return _detect_cycles(fills)


def filter_cycles(
    cycles: Iterable[object],
    *,
    product: str | None,
    cutoff: datetime | None,
) -> List[object]:
    results: List[object] = []
    for cycle in cycles:
        if product and cycle.product_id != product:
            continue
        if cutoff and cycle.end_time < cutoff:
            continue
        results.append(cycle)
    return results


def record_exists(record: dict, *, tolerance_seconds: int = 120, pnl_tolerance: float = 0.5) -> bool:
    path = _ensure_log_file()
    target_close = _parse_log_datetime(record.get("closed_at", ""))
    target_open = _parse_log_datetime(record.get("opened_at", ""))
    target_net = _parse_log_float(record.get("net_size", ""))
    target_entry = _parse_log_float(record.get("entry_price", ""))
    target_exit = _parse_log_float(record.get("exit_price", ""))
    target_pnl = _parse_log_float(record.get("profit_loss", ""))

    tolerance_seconds = max(1, int(abs(tolerance_seconds)))
    pnl_tolerance = max(0.1, float(pnl_tolerance))
    adaptive_pnl_tol = max(pnl_tolerance, abs(target_pnl) * 0.1)

    import csv

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("product_id") or "") != (record.get("product_id") or ""):
                continue
            row_close = _parse_log_datetime(row.get("closed_at", ""))
            if row_close and target_close:
                if abs((row_close - target_close).total_seconds()) > tolerance_seconds:
                    continue
            row_open = _parse_log_datetime(row.get("opened_at", ""))
            if row_open and target_open:
                if not _time_close(row_open, target_open, tolerance_seconds):
                    continue

            row_net = _parse_log_float(row.get("net_size", ""))
            row_entry = _parse_log_float(row.get("entry_price", ""))
            row_exit = _parse_log_float(row.get("exit_price", ""))
            row_pnl = _parse_log_float(row.get("profit_loss", ""))

            if not _float_close(row_net, target_net, rel_tol=1e-4, abs_tol=1e-4):
                continue
            if not _float_close(row_entry, target_entry, rel_tol=1e-4, abs_tol=1e-4):
                continue
            if not _float_close(row_exit, target_exit, rel_tol=1e-4, abs_tol=1e-4):
                continue
            if abs(row_pnl - target_pnl) > adaptive_pnl_tol:
                continue
            return True
    return False


def check_cycles(cycles: Iterable[object], verbose: bool = False) -> List[dict]:
    missing: List[dict] = []
    threshold = _breakeven_threshold()

    for cycle in cycles:
        record = _cycle_to_record(cycle, threshold)
        pnl = _parse_log_float(record.get("profit_loss", ""))
        entry = _parse_log_float(record.get("entry_price", ""))
        exit_price = _parse_log_float(record.get("exit_price", ""))
        net_size = _parse_log_float(record.get("net_size", "")) or 0.0
        adjusted_pnl, adjusted_exit, adjusted_reason = _apply_breakeven_adjustment(
            record.get("closure_reason", ""),
            pnl,
            entry,
            exit_price,
            net_size,
        )
        if adjusted_pnl is not None:
            record["profit_loss"] = _format_float(adjusted_pnl, 2)
        if adjusted_exit is not None:
            record["exit_price"] = _format_float(adjusted_exit, 6)
        record["closure_reason"] = adjusted_reason

        if record_exists(record):
            if verbose:
                print(
                    f"OK  {record['product_id']:>15} | {record['closed_at']} | "
                    f"{float(record['profit_loss']):>8.2f}"
                )
            continue

        missing.append(record)

    return missing


def main() -> None:
    args = parse_args()

    if args.csv:
        csv_path = os.path.abspath(args.csv)
        os.environ["WATCHDOG_LOG_DIR"] = os.path.dirname(csv_path)

    csv_file = _ensure_log_file()

    cycles = load_recent_cycles(limit=args.limit)

    cutoff = None
    if args.lookback_hours and args.lookback_hours > 0:
        cutoff = datetime.now(UTC) - timedelta(hours=args.lookback_hours)

    filtered_cycles = filter_cycles(cycles, product=args.product, cutoff=cutoff)

    if not filtered_cycles:
        print("No recent cycles found for the requested filters; nothing to validate.")
        return

    missing = check_cycles(filtered_cycles, verbose=args.verbose)

    if not missing:
        print(
            f"Integrity check passed for {len(filtered_cycles)} cycles (CSV: {csv_file})."
        )
        return

    print(f"Found {len(missing)} cycle(s) missing from {csv_file}:")
    for record in missing:
        profit = float(record.get("profit_loss") or 0.0)
        net_size = float(record.get("net_size") or 0.0)
        print(
            f" - {record['product_id']} closed {record['closed_at']} | "
            f"net_size={net_size:+.4f} | pnl={profit:+.2f} | reason={record['closure_reason']}"
        )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
