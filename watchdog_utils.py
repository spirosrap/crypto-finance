#!/usr/bin/env python3
"""
Shared helpers for watchdog analytics scripts.

Centralises common DataFrame filters so the various watchdog tools stay
consistent when slicing by date windows, positional count windows, or
tail selections.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _prepare_sort_frame(df: pd.DataFrame, ordering_col: Optional[str]) -> pd.DataFrame:
    """
    Return a DataFrame sorted by the requested column when available.

    Falls back to the original order if the column is missing or cannot be
    coerced into a usable datetime sequence.
    """
    if ordering_col is None or ordering_col not in df.columns:
        return df

    try:
        sort_values = pd.to_datetime(df[ordering_col], errors="coerce", utc=True)
    except Exception:
        sort_values = None

    if sort_values is not None and sort_values.notna().any():
        ordered = df.assign(_sort_key=sort_values)
        ordered = ordered.sort_values("_sort_key", kind="stable")
        return ordered.drop(columns="_sort_key")

    try:
        return df.sort_values(ordering_col, kind="stable")
    except Exception:
        return df


def filter_by_date(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    *,
    timestamp_col: str = "closed_at",
) -> pd.DataFrame:
    """
    Filter rows by an inclusive/exclusive UTC date window.

    Args:
        df: Source DataFrame.
        start_date: Inclusive lower bound (YYYY-MM-DD or ISO8601). None to disable.
        end_date: Exclusive upper bound. None to disable.
        timestamp_col: Column containing timestamps.
    """
    if df.empty or timestamp_col not in df.columns:
        return df

    result = df.copy()
    timestamps = pd.to_datetime(result[timestamp_col], errors="coerce", utc=True)
    mask = timestamps.notna()
    if not mask.all():
        result = result.loc[mask].copy()
        timestamps = timestamps.loc[mask]
    result[timestamp_col] = timestamps

    if start_date:
        start = pd.to_datetime(start_date, utc=True)
        result = result[result[timestamp_col] >= start]
    if end_date:
        end = pd.to_datetime(end_date, utc=True)
        result = result[result[timestamp_col] < end]

    return result


def select_count_window(
    df: pd.DataFrame,
    start_count: int = 0,
    end_count: int = 0,
    *,
    ordering_col: str = "closed_at",
) -> pd.DataFrame:
    """
    Slice by 1-based positional indices after sorting by the ordering column.

    Args:
        df: Source DataFrame.
        start_count: 1-based inclusive start index (0 disables lower bound).
        end_count: 1-based inclusive end index (0 disables upper bound).
        ordering_col: Column used to sort before slicing.
    """
    if df.empty:
        return df

    start = start_count if start_count and start_count > 0 else None
    end = end_count if end_count and end_count > 0 else None
    if start is None and end is None:
        return df

    ordered = _prepare_sort_frame(df, ordering_col)
    start_idx = (start - 1) if start is not None else 0
    if start_idx >= len(ordered):
        return ordered.iloc[0:0].copy()

    if end is not None and end < (start or 1):
        return ordered.iloc[0:0].copy()

    if end is not None:
        subset = ordered.iloc[start_idx:end]
    else:
        subset = ordered.iloc[start_idx:]
    return subset.copy()


def select_last(
    df: pd.DataFrame,
    last: int,
    *,
    ordering_col: str = "closed_at",
) -> pd.DataFrame:
    """
    Return the most recent N rows after sorting by the ordering column.

    Args:
        df: Source DataFrame.
        last: Number of tail rows to keep (<=0 returns original DataFrame).
        ordering_col: Column used to sort before taking the tail.
    """
    if last <= 0 or df.empty:
        return df

    ordered = _prepare_sort_frame(df, ordering_col)
    return ordered.tail(last).copy()


__all__ = [
    "filter_by_date",
    "select_count_window",
    "select_last",
]

