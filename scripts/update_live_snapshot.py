#!/usr/bin/env python3
"""Refresh live positions + balance and persist a dashboard snapshot.

Writes `logs/live_snapshot.json` so the dashboard can render live data
without hitting Coinbase on every refresh.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime, timezone


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _load_dotenv_if_available()

    try:
        from watchdog_dashboard import (
            load_open_positions,
            load_perp_usdc_balance,
            _prepare_open_positions_df,
            _save_live_snapshot,
        )
    except Exception as exc:
        logger.error("Failed to import dashboard helpers: %s", exc)
        raise SystemExit(1)

    positions_df, total_unrealized = load_open_positions()
    positions_df = _prepare_open_positions_df(positions_df)
    usdc_balance = load_perp_usdc_balance()
    _save_live_snapshot(positions_df, total_unrealized, usdc_balance)

    logger.info(
        "Saved live snapshot (%d positions, unrealized=%+.2f, usdc=%s) at %s",
        len(positions_df),
        total_unrealized,
        f"{usdc_balance:.2f}" if usdc_balance is not None else "n/a",
        datetime.now(timezone.utc).isoformat(),
    )


if __name__ == "__main__":
    main()
