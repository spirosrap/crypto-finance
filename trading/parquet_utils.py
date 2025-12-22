"""Parquet IO + DuckDB query helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a pandas DataFrame."""
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path, *, index: bool = False) -> Path:
    """Write a DataFrame to parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    return path


def _quote_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _relation_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    quoted = _quote_path(path)
    if suffix == ".parquet":
        return f"read_parquet('{quoted}')"
    if suffix == ".csv":
        return f"read_csv_auto('{quoted}', header=True)"
    raise ValueError(f"Unsupported file type for DuckDB query: {path}")


def query_sql(sql: str, paths: Sequence[Path], *, view: str = "data") -> pd.DataFrame:
    """Run a SQL query against one or more parquet/CSV files."""
    if not paths:
        raise ValueError("No paths provided for DuckDB query.")

    relations = [f"SELECT * FROM {_relation_for_path(path)}" for path in paths]
    union_sql = " UNION ALL ".join(relations)

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW {view} AS {union_sql}")
    return con.execute(sql).df()
