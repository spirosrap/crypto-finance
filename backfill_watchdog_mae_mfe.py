#!/usr/bin/env python3
"""Backfill missing MAE/MFE values in watchdog closure logs."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS
from watchdog_close_old_positions import (
    _format_float,
    _parse_iso8601,
    compute_mae_mfe_from_history,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    """Aggregate counters describing a backfill run."""

    total_rows: int = 0
    rows_updated: int = 0
    rows_skipped: int = 0
    failures: int = 0

    def __add__(self, other: "BackfillStats") -> "BackfillStats":
        return BackfillStats(
            total_rows=self.total_rows + other.total_rows,
            rows_updated=self.rows_updated + other.rows_updated,
            rows_skipped=self.rows_skipped + other.rows_skipped,
            failures=self.failures + other.failures,
        )


def _normalize_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _needs_backfill(row: dict[str, str]) -> bool:
    return not (row.get('mae') and row.get('mfe'))


def _update_row(
    row: dict[str, str],
    cb: CoinbaseService,
    granularity: str,
) -> tuple[bool, bool]:
    """Return tuple(updated: bool, failure: bool) for a single row."""

    net_size = _normalize_float(row.get('net_size'))
    entry_price = _normalize_float(row.get('entry_price'))
    exit_price = _normalize_float(row.get('exit_price'))
    opened_at = _parse_iso8601(row.get('opened_at'))
    closed_at = _parse_iso8601(row.get('closed_at'))

    if not row.get('product_id'):
        LOGGER.debug("Skipping row without product_id")
        return False, False
    if net_size in (None, 0.0):
        LOGGER.debug("Skipping row %s due to zero/unknown net_size", row.get('product_id'))
        return False, False
    if entry_price is None or opened_at is None or closed_at is None:
        LOGGER.debug(
            "Skipping %s due to missing entry_price/opened_at/closed_at",
            row.get('product_id'),
        )
        return False, False

    try:
        mae, mfe = compute_mae_mfe_from_history(
            cb=cb,
            product_id=row['product_id'],
            net_size=net_size,
            entry_price=entry_price,
            open_time=opened_at,
            close_time=closed_at,
            exit_price=exit_price,
            granularity=granularity,
        )
    except Exception as exc:  # pragma: no cover - safety net
        LOGGER.error(
            "Failed to derive MAE/MFE for %s: %s",
            row.get('product_id'),
            exc,
        )
        return False, True

    if mae is None and mfe is None:
        LOGGER.debug("No excursions derived for %s", row.get('product_id'))
        return False, False

    updated = False
    if mae is not None and not row.get('mae'):
        row['mae'] = _format_float(mae, 2)
        updated = True
    if mfe is not None and not row.get('mfe'):
        row['mfe'] = _format_float(mfe, 2)
        updated = True

    return updated, False


def backfill_file(
    path: Path,
    *,
    cb: Optional[CoinbaseService] = None,
    dry_run: bool = False,
    limit: Optional[int] = None,
    granularity: str = 'ONE_MINUTE',
) -> BackfillStats:
    """Backfill MAE/MFE fields in-place for the given CSV file."""

    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open('r', newline='') as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    stats = BackfillStats(total_rows=len(rows))

    if not rows:
        LOGGER.info("No rows found in %s", path)
        return stats

    if cb is None:
        cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)

    updated_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        if _needs_backfill(row):
            updated, failed = _update_row(row, cb, granularity)
            if updated:
                stats.rows_updated += 1
                LOGGER.info(
                    "Updated MAE/MFE for %s closed at %s",
                    row.get('product_id'),
                    row.get('closed_at'),
                )
            elif not failed:
                stats.rows_skipped += 1
            if failed:
                stats.failures += 1
            updated_rows.append(row)
            if limit is not None and stats.rows_updated >= limit:
                LOGGER.info("Update limit %s reached; stopping early", limit)
                updated_rows.extend(rows[idx + 1 :])
                break
        else:
            stats.rows_skipped += 1
            updated_rows.append(row)
    else:
        # For loop exhausted without hitting limit; ensure counts align (rows already appended)
        pass

    if not dry_run and stats.rows_updated > 0:
        temp_path = path.with_suffix(path.suffix + '.tmp')
        with temp_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)
        temp_path.replace(path)
    elif dry_run:
        LOGGER.info("Dry run complete; no file written")

    return stats


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill MAE/MFE values in watchdog logs")
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('trade_logs/watchdog_closed_positions.csv'),
        help='Path to watchdog CSV (default: trade_logs/watchdog_closed_positions.csv)',
    )
    parser.add_argument('--limit', type=int, help='Maximum number of rows to update (default: unlimited)')
    parser.add_argument('--granularity', default='ONE_MINUTE', help='Candle granularity for history fetches')
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing to disk')
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    try:
        stats = backfill_file(
            args.input,
            dry_run=args.dry_run,
            limit=args.limit,
            granularity=args.granularity,
        )
    except FileNotFoundError as exc:
        LOGGER.error(str(exc))
        raise SystemExit(1) from exc

    LOGGER.info(
        "Backfill complete: total=%s updated=%s skipped=%s failures=%s",
        stats.total_rows,
        stats.rows_updated,
        stats.rows_skipped,
        stats.failures,
    )


if __name__ == '__main__':
    main()
