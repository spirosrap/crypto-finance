#!/usr/bin/env python3
"""Convert finder outputs, trade logs, and backtest CSVs to parquet."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from pandas.errors import ParserError

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from add_position_from_finder import ParsedFinder, parse_finder_text, split_blocks
from trading.parquet_utils import write_parquet


def _parse_generated_at(text: str) -> Optional[str]:
    match = re.search(r"Generated on \\(UTC\\):\\s*([0-9\\-: ]+Z)", text)
    return match.group(1) if match else None


def _finder_records(path: Path) -> List[Dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    blocks = split_blocks(raw)
    generated_at = _parse_generated_at(raw)
    records: List[Dict[str, object]] = []
    for block in blocks:
        try:
            parsed = parse_finder_text(block)
        except Exception:
            continue
        rr = 0.0
        if parsed.stop != parsed.entry:
            rr = abs(parsed.take_profit - parsed.entry) / abs(parsed.entry - parsed.stop)
        records.append(
            {
                "symbol": parsed.symbol,
                "base_symbol": parsed.base_symbol,
                "side": parsed.side,
                "entry_price": parsed.entry,
                "stop_loss": parsed.stop,
                "take_profit": parsed.take_profit,
                "risk_reward": rr,
                "position_pct": parsed.pos_size_pct,
                "generated_at": generated_at,
                "source_file": path.name,
            }
        )
    return records


def _convert_csv(path: Path, out_path: Path, overwrite: bool, skip_bad_lines: bool) -> Optional[Path]:
    if out_path.exists() and not overwrite:
        return None
    try:
        df = pd.read_csv(path)
    except ParserError as exc:
        if not skip_bad_lines:
            raise exc
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
    return write_parquet(df, out_path)


def _convert_finder(path: Path, out_dir: Path, overwrite: bool) -> Optional[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.parquet"
    if out_path.exists() and not overwrite:
        return None
    records = _finder_records(path)
    if not records:
        return None
    df = pd.DataFrame(records)
    return write_parquet(df, out_path)


def _convert_trade_logs(overwrite: bool, skip_bad_lines: bool) -> List[Path]:
    outputs: List[Path] = []
    for path in Path("trade_logs").glob("*.csv"):
        out_path = path.with_suffix(".parquet")
        converted = _convert_csv(path, out_path, overwrite, skip_bad_lines)
        if converted:
            outputs.append(converted)
    return outputs


def _convert_backtests(overwrite: bool, skip_bad_lines: bool) -> List[Path]:
    outputs: List[Path] = []
    root = Path("backtest_results")
    if not root.exists():
        return outputs
    for path in root.rglob("*.csv"):
        out_path = path.with_suffix(".parquet")
        converted = _convert_csv(path, out_path, overwrite, skip_bad_lines)
        if converted:
            outputs.append(converted)
    return outputs


def _convert_finders(files: Iterable[Path], overwrite: bool) -> List[Path]:
    outputs: List[Path] = []
    out_dir = Path("finder_logs")
    for path in files:
        if not path.exists():
            continue
        converted = _convert_finder(path, out_dir, overwrite)
        if converted:
            outputs.append(converted)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert finder outputs, trade logs, and backtest CSVs to parquet.")
    parser.add_argument("--finder", action="store_true", help="Convert finder_short.txt / finder_long.txt to parquet.")
    parser.add_argument("--finder-file", action="append", help="Specific finder text file(s) to convert.")
    parser.add_argument("--trade-logs", action="store_true", help="Convert trade_logs/*.csv to parquet.")
    parser.add_argument("--backtests", action="store_true", help="Convert backtest_results/**/*.csv to parquet.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet files.")
    parser.add_argument(
        "--skip-bad-lines",
        action="store_true",
        help="Skip malformed CSV rows when parsing (best-effort conversion).",
    )
    args = parser.parse_args()

    if not (args.finder or args.finder_file or args.trade_logs or args.backtests):
        raise SystemExit("Select at least one conversion target (--finder, --trade-logs, --backtests).")

    converted: List[Path] = []

    if args.finder or args.finder_file:
        finder_files: List[Path] = []
        if args.finder:
            finder_files.extend([Path("finder_short.txt"), Path("finder_long.txt")])
        if args.finder_file:
            for item in args.finder_file:
                if item:
                    finder_files.append(Path(item))
        converted.extend(_convert_finders(finder_files, args.overwrite))

    if args.trade_logs:
        converted.extend(_convert_trade_logs(args.overwrite, args.skip_bad_lines))

    if args.backtests:
        converted.extend(_convert_backtests(args.overwrite, args.skip_bad_lines))

    if converted:
        for path in converted:
            print(f"wrote {path}")
    else:
        print("No parquet files written (missing inputs or already up to date).")


if __name__ == "__main__":
    main()
