#!/usr/bin/env python3
"""Run ad-hoc DuckDB SQL against parquet/CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trading.parquet_utils import query_sql


def _parse_paths(raw_paths: List[str]) -> List[Path]:
    paths: List[Path] = []
    for chunk in raw_paths:
        for part in chunk.split(","):
            part = part.strip()
            if not part:
                continue
            paths.append(Path(part))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DuckDB SQL against parquet/CSV files.")
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help="Path(s) to parquet/CSV files (repeatable, comma-separated allowed).",
    )
    parser.add_argument("--sql", help="SQL query to run (use view name 'data' by default).")
    parser.add_argument("--sql-file", type=Path, help="Path to a .sql file to execute.")
    parser.add_argument("--view", default="data", help="View name to expose in SQL (default: data).")
    args = parser.parse_args()

    sql = args.sql
    if args.sql_file:
        sql = args.sql_file.read_text(encoding="utf-8")
    if not sql:
        raise SystemExit("Provide --sql or --sql-file.")

    paths = _parse_paths(args.path or [])
    if not paths:
        raise SystemExit("No paths provided.")

    df = query_sql(sql, paths, view=args.view)
    if df.empty:
        print("No rows returned.")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
