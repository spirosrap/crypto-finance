#!/usr/bin/env python3
"""CLI entry point for the intraday zero-fee INTX trading bot."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

from trading.intraday_zero_fee_trader import IntradayTraderConfig, ZeroFeePerpTrader


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-fee INTX intraday trader")
    parser.add_argument(
        "--products",
        help="Comma-separated product ids (e.g. BTC-PERP-INTX,ETH-PERP-INTX)",
    )
    parser.add_argument(
        "--granularity",
        choices=["ONE_MINUTE", "TWO_MINUTE", "THREE_MINUTE", "FIVE_MINUTE"],
        help="Candle granularity; defaults to env or ONE_MINUTE",
    )
    parser.add_argument(
        "--poll",
        type=int,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Run the loop a fixed number of times (useful for smoke tests)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute real orders (requires API credentials)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    config = IntradayTraderConfig.from_env()

    if args.products:
        config.product_ids = [item.strip() for item in args.products.split(",") if item.strip()]
    if args.granularity:
        config.granularity = args.granularity
    if args.poll:
        config.poll_seconds = max(5, args.poll)
    if args.live:
        config.dry_run = False

    trader = ZeroFeePerpTrader(config)

    logging.getLogger(__name__).info(
        "Starting trader products=%s granularity=%s dry_run=%s",
        config.product_ids,
        config.granularity,
        config.dry_run,
    )

    if args.max_iterations is not None:
        iterations = max(1, args.max_iterations)
        for _ in range(iterations):
            trader.run_once()
            time.sleep(config.poll_seconds)
        return 0

    trader.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
