#!/usr/bin/env python3
"""Backfill MAE/MFE and expiry labels for paper-trade closed positions."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from watchdog_close_old_positions import (
    _format_float,
    _parse_iso8601,
    compute_mae_mfe_from_history,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    total_rows: int = 0
    rows_updated: int = 0
    mae_mfe_updated: int = 0
    reason_updated: int = 0
    rows_skipped: int = 0
    failures: int = 0


def _get_perps_credentials() -> tuple[str, str]:
    try:
        from credentials import get_perps_credentials

        return get_perps_credentials()
    except Exception:
        try:  # pragma: no cover - fallback for older setups
            from config import API_KEY_PERPS, API_SECRET_PERPS  # type: ignore
        except Exception:
            return ("", "")
        return (API_KEY_PERPS or "", API_SECRET_PERPS or "")  # type: ignore[arg-type]


def _get_coinbase_service():
    api_key, api_secret = _get_perps_credentials()
    if not api_key or not api_secret:
        LOGGER.warning("MAE/MFE backfill disabled: Coinbase perps credentials not found.")
        return None
    try:
        from coinbaseservice import CoinbaseService
    except Exception as exc:
        LOGGER.warning("Unable to import CoinbaseService (%s).", exc)
        return None
    try:
        return CoinbaseService(api_key, api_secret)
    except Exception as exc:
        LOGGER.warning("Unable to init CoinbaseService (%s).", exc)
        return None


def _normalize_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _expiry_label(pnl_pct: Optional[float], breakeven_pct: float) -> Optional[str]:
    if pnl_pct is None:
        return None
    if abs(pnl_pct) <= breakeven_pct:
        return "expired_breakeven"
    return "expired_profit" if pnl_pct > 0 else "expired_loss"


def _derive_pnl_pct(row: dict[str, str]) -> Optional[float]:
    pct = _normalize_float(row.get("profit_loss_pct"))
    if pct is not None:
        return pct
    entry = _normalize_float(row.get("entry_price"))
    exit_price = _normalize_float(row.get("exit_price"))
    if entry is None or exit_price is None or entry == 0:
        return None
    side = (row.get("position_side") or "").upper()
    if "SHORT" in side:
        return (entry - exit_price) / entry * 100.0
    return (exit_price - entry) / entry * 100.0


def _update_expiry_reason(row: dict[str, str], breakeven_pct: float) -> bool:
    reason = (row.get("closure_reason") or "").lower()
    if not reason.startswith("expired"):
        return False
    pnl_pct = _derive_pnl_pct(row)
    new_reason = _expiry_label(pnl_pct, breakeven_pct)
    if new_reason and new_reason != row.get("closure_reason"):
        row["closure_reason"] = new_reason
        return True
    return False


def _update_mae_mfe(
    row: dict[str, str],
    cb,
    granularity: str,
) -> tuple[bool, bool]:
    """Return tuple(updated: bool, failed: bool)."""
    if cb is None:
        return False, False
    if row.get("mae") and row.get("mfe"):
        return False, False
    product_id = row.get("product_id")
    if not product_id:
        return False, False
    net_size = _normalize_float(row.get("net_size"))
    entry_price = _normalize_float(row.get("entry_price"))
    exit_price = _normalize_float(row.get("exit_price"))
    opened_at = _parse_iso8601(row.get("opened_at"))
    closed_at = _parse_iso8601(row.get("closed_at"))
    if net_size in (None, 0.0) or entry_price is None or opened_at is None or closed_at is None:
        return False, False
    try:
        mae, mfe = compute_mae_mfe_from_history(
            cb=cb,
            product_id=product_id,
            net_size=net_size,
            entry_price=entry_price,
            open_time=opened_at,
            close_time=closed_at,
            exit_price=exit_price,
            granularity=granularity,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error("Failed MAE/MFE for %s: %s", product_id, exc)
        return False, True
    updated = False
    if mae is not None and not row.get("mae"):
        row["mae"] = _format_float(mae, 2)
        updated = True
    if mfe is not None and not row.get("mfe"):
        row["mfe"] = _format_float(mfe, 2)
        updated = True
    return updated, False


def backfill_file(
    path: Path,
    *,
    breakeven_pct: float,
    granularity: str,
    dry_run: bool,
    limit: Optional[int],
) -> BackfillStats:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    stats = BackfillStats(total_rows=len(rows))
    if not rows:
        LOGGER.info("No rows found in %s", path)
        return stats

    cb = _get_coinbase_service()

    updated_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        updated = False
        reason_updated = _update_expiry_reason(row, breakeven_pct)
        if reason_updated:
            stats.reason_updated += 1
            updated = True

        mae_updated, failed = _update_mae_mfe(row, cb, granularity)
        if mae_updated:
            stats.mae_mfe_updated += 1
            updated = True
        if failed:
            stats.failures += 1

        if updated:
            stats.rows_updated += 1
        else:
            stats.rows_skipped += 1

        updated_rows.append(row)
        if limit is not None and stats.rows_updated >= limit:
            LOGGER.info("Update limit %s reached; stopping early", limit)
            updated_rows.extend(rows[idx + 1 :])
            break

    if not dry_run and stats.rows_updated > 0:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(updated_rows)
        temp_path.replace(path)
    elif dry_run:
        LOGGER.info("Dry run complete; no file written")

    return stats


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill paper trade MAE/MFE and expiry labels")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("trade_logs/paper_finder_closed_positions.csv"),
        help="Paper closed CSV (default: trade_logs/paper_finder_closed_positions.csv)",
    )
    parser.add_argument(
        "--expiry-breakeven-pct",
        type=float,
        default=0.10,
        help="Breakeven band in percent for expiry relabel (default: 0.10).",
    )
    parser.add_argument(
        "--granularity",
        default="ONE_MINUTE",
        help="Candle granularity for MAE/MFE (default: ONE_MINUTE).",
    )
    parser.add_argument("--limit", type=int, help="Max rows to update (default: unlimited).")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)
    stats = backfill_file(
        args.input,
        breakeven_pct=args.expiry_breakeven_pct,
        granularity=args.granularity,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    LOGGER.info(
        "Backfill complete: updated=%s (mae/mfe=%s, reasons=%s) skipped=%s failures=%s",
        stats.rows_updated,
        stats.mae_mfe_updated,
        stats.reason_updated,
        stats.rows_skipped,
        stats.failures,
    )


if __name__ == "__main__":
    main()
