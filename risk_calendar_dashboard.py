#!/usr/bin/env python3
"""
Streamlit dashboard for the risk calendar.

Launch with:
    streamlit run risk_calendar_dashboard.py
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from risk_calendar import DEFAULT_STORE, RiskCalendar


UTC = timezone.utc


def load_events(store_path: Path) -> List[dict]:
    calendar = RiskCalendar(store_path=store_path)
    events = calendar.all_events()
    payload = []
    for event in events:
        payload.append(
            {
                "id": event.id,
                "name": event.name,
                "category": event.category,
                "impact": event.impact,
                "start_utc": event.start_utc,
                "end_utc": event.end_utc,
                "symbols": ", ".join(event.symbols) if event.symbols else "",
                "notes": event.notes or "",
                "source": event.source or "",
                "completed": event.completed,
                "created_utc": event.created_utc,
                "updated_utc": event.updated_utc,
            }
        )
    return payload


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    st.sidebar.header("Filters")
    min_date = df["start_utc"].min().date()
    max_date = df["start_utc"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    start_dt, end_dt = None, None
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_dt = datetime.combine(date_range[0], datetime.min.time(), tzinfo=UTC)
        end_dt = datetime.combine(date_range[1], datetime.max.time(), tzinfo=UTC)
    categories = sorted(df["category"].dropna().unique())
    impacts = sorted(df["impact"].dropna().unique())
    selected_categories = st.sidebar.multiselect("Categories", categories, default=categories)
    selected_impacts = st.sidebar.multiselect("Impact levels", impacts, default=impacts)
    hide_completed = st.sidebar.checkbox("Hide completed events", value=False)

    filtered = df.copy()
    if start_dt and end_dt:
        filtered = filtered[(filtered["start_utc"] >= start_dt) & (filtered["start_utc"] <= end_dt)]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if selected_impacts:
        filtered = filtered[filtered["impact"].isin(selected_impacts)]
    if hide_completed:
        filtered = filtered[~filtered["completed"]]
    return filtered.sort_values("start_utc")


def main() -> None:
    st.set_page_config(page_title="Risk Calendar", layout="wide")
    st.title("Risk Calendar Dashboard")
    st.caption("Track upcoming macro, exchange, and crypto-specific risk events.")

    store_input = st.sidebar.text_input("Store path", str(DEFAULT_STORE))
    store_path = Path(store_input).expanduser()

    try:
        events_payload = load_events(store_path)
    except Exception as exc:
        st.error(f"Failed to load events: {exc}")
        return

    df = pd.DataFrame(events_payload)
    if df.empty:
        st.info("No risk events recorded yet. Use `python risk_calendar.py add ...` to populate the calendar.")
        return

    df["start_utc"] = pd.to_datetime(df["start_utc"], utc=True)
    if "end_utc" in df.columns:
        df["end_utc"] = pd.to_datetime(df["end_utc"], utc=True)
    filtered = apply_filters(df)

    st.subheader("Summary")
    total_events = len(filtered)
    high_impact = int((filtered["impact"].str.lower() == "high").sum())
    upcoming_week = int(
        filtered[
            (filtered["start_utc"] >= datetime.now(UTC))
            & (filtered["start_utc"] <= datetime.now(UTC) + pd.Timedelta(days=7))
        ].shape[0]
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Events", total_events)
    c2.metric("High impact", high_impact)
    c3.metric("Next 7 days", upcoming_week)

    st.markdown("---")
    st.subheader("Event Table")
    display_df = filtered.copy()
    display_df["start_utc"] = display_df["start_utc"].dt.strftime("%Y-%m-%d %H:%M")
    if "end_utc" in display_df.columns:
        display_df["end_utc"] = display_df["end_utc"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered events (CSV)",
        data=csv_bytes,
        file_name="risk_calendar_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
