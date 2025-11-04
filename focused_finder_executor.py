#!/usr/bin/env python3
"""
Focused Finder Executor

Runs the short-term crypto finder with the trusted ``focused_llm_100`` profile,
selects the top N opportunities, and forwards them directly to the CCXT perp
trading bridge (`add_position_from_finder -> ccxt_trade_perp`).

This collapses the manual workflow:
 1. python short_term_crypto_finder.py --profile focused_llm_100 --plain-output finder_short.txt --force-refresh
 2. Edit finder_short.txt down to five trades
 3. python add_position_from_finder.py --file finder_short.txt ...

into a single command.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import List

from add_position_from_finder import (
    DEFAULT_EXCLUSION_FILE,
    OrderSettings,
    ParsedFinder,
    parse_finder_text,
    process_parsed_signals,
    split_blocks,
)
from short_term_crypto_finder import (
    PROFILE_PRESETS,
    ShortTermCryptoFinder,
    build_short_term_config,
)


def _apply_profile_overrides(config, overrides: dict) -> None:
    """Apply focused_llm_100 profile tweaks directly to the finder config."""

    if not overrides:
        return

    if "max_results" in overrides:
        config.max_results = int(overrides["max_results"])
    if "top_per_side" in overrides:
        config.top_per_side = int(overrides["top_per_side"])
    if "use_openai_scoring" in overrides:
        config.use_openai_scoring = bool(overrides["use_openai_scoring"])
    if "min_volume_24h" in overrides:
        config.min_volume_24h = float(overrides["min_volume_24h"])
    if "min_volume_market_cap_ratio" in overrides:
        config.min_volume_market_cap_ratio = float(overrides["min_volume_market_cap_ratio"])
    if "intraday_lookback_days" in overrides:
        config.intraday_lookback_days = int(overrides["intraday_lookback_days"])
    if "unique_by_symbol" in overrides:
        config.unique_by_symbol = bool(overrides["unique_by_symbol"])
    if "max_risk_level" in overrides:
        config.max_risk_level = str(overrides["max_risk_level"]).upper()


def _select_top(results, top_n: int) -> List:
    """Return the best ``top_n`` results sorted by overall score."""
    if not results:
        return []
    sorted_results = sorted(results, key=lambda row: getattr(row, "overall_score", 0.0), reverse=True)
    return sorted_results[:top_n]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally execute focused short-term perp trades via CCXT."
    )
    parser.add_argument(
        "--portfolio-usd",
        type=float,
        required=True,
        help="Total portfolio value in USD.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        default="focused_llm_100",
        help="Finder profile preset to apply (default focused_llm_100).",
    )
    parser.add_argument("--leverage", type=float, default=50.0, help="Target leverage (default 50).")
    parser.add_argument("--top", type=int, default=5, help="Number of trades to forward (default 5).")
    parser.add_argument(
        "--finder-limit",
        type=int,
        default=None,
        help="Override the finder symbol scan limit (defaults to profile limit).",
    )
    parser.add_argument(
        "--order",
        choices=["market", "limit"],
        default="market",
        help="Entry order type for ccxt_trade_perp.py (default market).",
    )
    parser.add_argument(
        "--product-form",
        choices=["PERP-INTX", "INTX-PERP"],
        default="PERP-INTX",
        help="Display format for perp ids when reporting (default PERP-INTX).",
    )
    parser.add_argument(
        "--expiry",
        choices=["GTC", "12h", "24h", "30d"],
        default="30d",
        help="Bracket expiry horizon (default 30d).",
    )
    parser.add_argument(
        "--save-text",
        type=Path,
        help="Optional path to write the trimmed finder report (plain text).",
    )
    parser.add_argument(
        "--no-force-refresh",
        action="store_true",
        help="Disable candle force refresh (enabled by default to mirror manual workflow).",
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=DEFAULT_EXCLUSION_FILE,
        help=f"Perp exclusion list to honour (default: {DEFAULT_EXCLUSION_FILE})",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually submit CCXT orders (default: dry run / command preview).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print command/execution output (skip selection summary banner).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    profile_name = args.profile
    profile_overrides = PROFILE_PRESETS.get(profile_name, {})
    config = build_short_term_config()
    _apply_profile_overrides(config, profile_overrides)
    config.force_refresh_candles = not args.no_force_refresh
    # Ensure CCXT backend remains the default for execution consistency.
    config.exchange_backend = "ccxt"
    config.ccxt_exchange_id = getattr(config, "ccxt_exchange_id", "coinbaseadvanced") or "coinbaseadvanced"

    finder = ShortTermCryptoFinder(config=config)
    finder_limit = args.finder_limit or int(profile_overrides.get("limit", 100))
    try:
        results = finder.find_best_opportunities(limit=finder_limit)
    except Exception as exc:
        print(f"Failed to run short-term finder ({profile_name}): {exc}", file=sys.stderr)
        sys.exit(2)

    if not results:
        print("Short-term finder produced no candidates. Adjust filters or retry later.")
        sys.exit(1)

    selected = _select_top(results, max(1, args.top))
    if not selected:
        print("Unable to select any trades from finder output.")
        sys.exit(1)

    if not args.quiet:
        print(f"Profile '{profile_name}' produced {len(results)} candidates (limit {finder_limit}).")
        print(f"Forwarding top {len(selected)} opportunities:")
        for idx, crypto in enumerate(selected, 1):
            side = getattr(crypto, "position_side", "LONG")
            score = getattr(crypto, "overall_score", 0.0)
            entry = getattr(crypto, "entry_price", getattr(crypto, "current_price", 0.0))
            tp = getattr(crypto, "take_profit_price", 0.0)
            sl = getattr(crypto, "stop_loss_price", 0.0)
            print(
                f" {idx}. {crypto.symbol} {side:>5} | score {score:6.2f} | entry {entry:.6f} | TP {tp:.6f} | SL {sl:.6f}"
            )

    buffer = io.StringIO()
    finder.print_results(selected, stream=buffer)
    report_text = buffer.getvalue()

    if args.save_text:
        args.save_text.parent.mkdir(parents=True, exist_ok=True)
        args.save_text.write_text(report_text, encoding="utf-8")
        if not args.quiet:
            print(f"Saved focused finder report to {args.save_text}")

    parsed_blocks = split_blocks(report_text)
    parsed_signals: List[ParsedFinder] = []
    for block in parsed_blocks:
        try:
            parsed_signals.append(parse_finder_text(block))
        except Exception as exc:
            print(f"Skipping block due to parse error: {exc}", file=sys.stderr)

    if not parsed_signals:
        print("No parsable finder entries were produced; aborting.", file=sys.stderr)
        sys.exit(1)

    settings = OrderSettings(
        portfolio_usd=args.portfolio_usd,
        leverage=args.leverage,
        product_form=args.product_form,
        order_type=args.order,
        execute=args.execute,
        expiry=args.expiry,
        exclude_file=args.exclude_file,
    )
    process_parsed_signals(parsed_signals, settings)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
