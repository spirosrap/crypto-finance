#!/usr/bin/env python3
"""Generate a shareable equity report for paper trades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.equity_report import build_daily_equity, build_equity_figure, load_closed_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper equity report (HTML + PNG).")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("trade_logs/paper_finder_closed_positions.csv"),
        help="Closed trades CSV (default: trade_logs/paper_finder_closed_positions.csv).",
    )
    parser.add_argument(
        "--starting-equity",
        type=float,
        default=1000.0,
        help="Starting equity for the curve (default: 1000.0).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory (default: reports).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="paper_equity_report",
        help="Output file prefix (default: paper_equity_report).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Paper Equity Report",
        help="Report title (default: Paper Equity Report).",
    )
    args = parser.parse_args()

    trades = load_closed_positions(args.csv)
    daily, metrics = build_daily_equity(trades, args.starting_equity)
    fig = build_equity_figure(daily, metrics, args.title)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.out_dir / f"{args.prefix}.html"
    png_path = args.out_dir / f"{args.prefix}.png"

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Wrote HTML report to {html_path}")

    try:
        fig.write_image(png_path, scale=2)
        print(f"Wrote PNG report to {png_path}")
    except Exception as exc:  # pragma: no cover - depends on kaleido runtime
        print(f"PNG export failed ({exc}). Install kaleido to enable PNG export.")


if __name__ == "__main__":
    main()
