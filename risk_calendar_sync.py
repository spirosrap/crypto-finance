#!/usr/bin/env python3
"""
Risk calendar synchronisation from external data sources.

Currently supports:
  • TradingEconomics' public economic calendar feed (guest key or personal).
  • The Federal Reserve FOMC meeting schedule (scraped directly from the Fed).
  • A bundled set of high-impact BLS/BEA releases (CPI, NFP, GDP, PCE) stored
    in manual_data/us_macro_surprises.json, plus any custom JSON files you add.

Persisted events are stored via the RiskCalendar utility in risk_calendar.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import requests
import time

from bs4 import BeautifulSoup

from risk_calendar import CalendarEvent, RiskCalendar, UTC, parse_datetime


TE_BASE_URL = "https://api.tradingeconomics.com/calendar"
FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
MANUAL_DATA_DIR = Path(__file__).with_name("manual_data")
DEFAULT_MANUAL_FILE = MANUAL_DATA_DIR / "us_macro_surprises.json"
DEFAULT_COUNTRY = "United States"
DEFAULT_LOOKAHEAD_DAYS = 21
IMPACT_MAP = {1: "low", 2: "medium", 3: "high"}
SUPPORTED_SOURCES = {"tradingeconomics", "fomc", "manual"}


def _parse_date(value: str) -> datetime:
    """TradingEconomics returns ISO timestamps in UTC without timezone suffix."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(UTC)


def _parse_cli_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _format_note(entry: dict) -> str:
    fields = []
    category = entry.get("Category")
    if category:
        fields.append(category)
    previous = entry.get("Previous")
    forecast = entry.get("Forecast")
    te_forecast = entry.get("TEForecast")
    actual = entry.get("Actual")
    details = []
    if actual:
        details.append(f"Actual {actual}")
    if forecast:
        details.append(f"Forecast {forecast}")
    elif te_forecast:
        details.append(f"TE forecast {te_forecast}")
    if previous:
        details.append(f"Previous {previous}")
    if details:
        fields.append(" | ".join(details))
    source = entry.get("Source")
    if source:
        fields.append(f"Source: {source}")
    return "; ".join(fields)


def convert_te_entry(entry: dict, *, symbols: Sequence[str]) -> CalendarEvent:
    start = _parse_date(entry["Date"])
    end = start + timedelta(minutes=45)
    impact = IMPACT_MAP.get(int(entry.get("Importance", 2) or 2), "medium")
    category = entry.get("Category", "macro").lower().replace(" ", "_")
    name = entry.get("Event") or entry.get("Category") or "Economic Event"
    source_url = entry.get("SourceURL") or entry.get("URL")
    notes = _format_note(entry)
    return CalendarEvent(
        id=f"te-{entry['CalendarId']}",
        name=name,
        start_utc=start,
        end_utc=end,
        category=category,
        impact=impact,
        symbols=list(symbols),
        source=source_url,
        notes=notes if notes else None,
        completed=start < datetime.now(UTC),
    )


def fetch_trading_economics(
    *,
    countries: Sequence[str],
    start: datetime,
    end: datetime,
    min_importance: int,
    api_key: str,
    chunk_days: int = 7,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> List[dict]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    params = {
        "format": "json",
        "c": api_key,
    }
    if countries:
        params["country"] = ";".join(countries)
    url = TE_BASE_URL

    start_date = start.date()
    end_date = end.date()
    current_date = start_date
    results: List[dict] = []
    target_countries: Set[str] = {c.strip().lower() for c in countries if c.strip()}

    while current_date <= end_date:
        chunk_end_date = min(current_date + timedelta(days=chunk_days - 1), end_date)
        query_params = params.copy()
        query_params["start"] = current_date.isoformat()
        query_params["end"] = chunk_end_date.isoformat()
        if min_importance:
            query_params["importance"] = min_importance

        for attempt in range(max_retries):
            response = requests.get(url, params=query_params, timeout=20)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                status = response.status_code
                if status >= 500 and attempt + 1 < max_retries:
                    time.sleep(retry_delay_seconds)
                    continue
                raise exc

            payload = response.json()
            if isinstance(payload, dict) and payload.get("Message"):
                raise ValueError(f"TradingEconomics error: {payload['Message']}")
            if not isinstance(payload, list):
                raise ValueError(f"Unexpected TradingEconomics payload: {payload!r}")
            if target_countries:
                payload = [
                    item
                    for item in payload
                    if str(item.get("Country", "")).strip().lower() in target_countries
                ]
            results.extend(payload)
            break
        else:
            raise RuntimeError(
                f"Failed to fetch TradingEconomics data for {current_date} to {chunk_end_date} after {max_retries} attempts"
            )

        current_date = chunk_end_date + timedelta(days=1)

    return results


def _month_to_number(name: str) -> int:
    return datetime.strptime(name, "%B").month


def parse_fomc_html(
    html: str,
    *,
    symbols: Sequence[str],
    years: Sequence[int],
    press_conf_note: str = "* indicates press conference",
) -> List[CalendarEvent]:
    soup = BeautifulSoup(html, "html.parser")
    events: List[CalendarEvent] = []
    target_years = set(years)
    now = datetime.now(UTC)

    panels = soup.select("div.panel")
    for panel in panels:
        heading_el = panel.select_one("h4")
        if not heading_el:
            continue
        heading_text = heading_el.get_text(" ", strip=True)
        match = re.search(r"(\d{4})", heading_text)
        if not match:
            continue
        year = int(match.group(1))
        if target_years and year not in target_years:
            continue

        for row in panel.select("div.fomc-meeting"):
            month_el = row.select_one(".fomc-meeting__month")
            date_el = row.select_one(".fomc-meeting__date")
            info_el = row.select_one(".fomc-meeting__info")
            if not month_el or not date_el:
                continue

            month_name = month_el.get_text(strip=True)
            raw_date = date_el.get_text(" ", strip=True)
            info_text = info_el.get_text(" ", strip=True) if info_el else ""

            note_parts: List[str] = []
            if info_text:
                note_parts.append(info_text)

            cleaned = raw_date.strip()
            press_conf = cleaned.endswith("*")
            if press_conf:
                cleaned = cleaned[:-1].strip()
                note_parts.append("Press conference")
            # Remove parenthetical notes from the date string but capture them in notes.
            if "(" in cleaned:
                cleaned, rest = cleaned.split("(", 1)
                rest = rest.rstrip(") ")
                note_parts.append(rest.strip())
            cleaned = cleaned.strip()

            if "-" in cleaned:
                start_day_str, end_day_str = [part.strip() for part in cleaned.split("-", 1)]
            else:
                start_day_str = cleaned
                end_day_str = cleaned

            try:
                start_day = int(re.sub(r"\D", "", start_day_str))
                end_day = int(re.sub(r"\D", "", end_day_str))
            except ValueError:
                continue

            try:
                month_num = _month_to_number(month_name)
            except ValueError:
                continue

            start_dt = datetime(year, month_num, start_day, 18, 0, tzinfo=UTC)
            end_dt = datetime(year, month_num, end_day, 20, 0, tzinfo=UTC)

            event_name = f"FOMC Meeting ({month_name} {start_day_str})"
            notes = ", ".join(note_parts) if note_parts else None

            event = CalendarEvent(
                id=f"fomc-{year}-{month_num:02d}-{start_day:02d}",
                name=event_name,
                start_utc=start_dt,
                end_utc=end_dt if end_day != start_day or press_conf else None,
                category="fomc",
                impact="high",
                symbols=list(symbols),
                source=FOMC_URL,
                notes=notes,
                completed=start_dt < now,
            )
            events.append(event)

    return events


def fetch_fomc_events(
    *,
    symbols: Sequence[str],
    years: Sequence[int],
) -> List[dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(FOMC_URL, headers=headers, timeout=20)
    response.raise_for_status()
    events = parse_fomc_html(response.text, symbols=symbols, years=years)
    return [event for event in events]


def load_manual_events(
    files: Sequence[Path],
    *,
    default_symbols: Sequence[str],
    default_category: str = "macro",
    default_impact: str = "high",
) -> List[CalendarEvent]:
    events: List[CalendarEvent] = []
    now = datetime.now(UTC)

    for file_path in files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            print(f"Manual events file not found: {file_path}", file=sys.stderr)
            continue
        except json.JSONDecodeError as exc:
            print(f"Failed to parse JSON in {file_path}: {exc}", file=sys.stderr)
            continue

        if isinstance(payload, dict):
            items = payload.get("events") or payload.get("data") or []
        else:
            items = payload

        if not isinstance(items, list):
            print(f"Manual events payload in {file_path} must be a list.", file=sys.stderr)
            continue

        for item in items:
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            start_raw = item.get("start") or item.get("start_utc")
            if not name or not start_raw:
                continue

            try:
                start_dt = parse_datetime(str(start_raw))
            except Exception as exc:
                print(f"Skipping {name!r} in {file_path}: invalid start ({exc})", file=sys.stderr)
                continue

            end_raw = item.get("end") or item.get("end_utc")
            end_dt: Optional[datetime] = None
            if end_raw:
                try:
                    end_dt = parse_datetime(str(end_raw))
                except Exception:
                    end_dt = None

            raw_symbols = item.get("symbols") or default_symbols
            if isinstance(raw_symbols, str):
                symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
            else:
                symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]

            category = str(item.get("category") or default_category)
            impact = str(item.get("impact") or default_impact).lower()
            notes = item.get("notes")
            source = item.get("source") or str(file_path)
            event_id = item.get("id") or f"manual-{uuid.uuid4().hex[:10]}"

            events.append(
                CalendarEvent(
                    id=event_id,
                    name=str(name),
                    start_utc=start_dt,
                    end_utc=end_dt,
                    category=category,
                    impact=impact,
                    symbols=symbols,
                    source=source,
                    notes=notes,
                    completed=start_dt < now,
                )
            )

    return events


def filter_entries(entries: Iterable[dict], *, min_importance: int) -> List[dict]:
    filtered = []
    for entry in entries:
        try:
            importance = int(entry.get("Importance", 0) or 0)
        except (TypeError, ValueError):
            importance = 0
        if importance < min_importance:
            continue
        filtered.append(entry)
    return filtered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync external macro risk events into the local risk calendar."
    )
    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="Override risk calendar store path (default risk_calendar/risk_events.json).",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="tradingeconomics",
        help="Comma-separated list of sources: tradingeconomics, fomc, manual.",
    )
    parser.add_argument(
        "--countries",
        type=str,
        default=DEFAULT_COUNTRY,
        help="Comma-separated list of countries (TradingEconomics only).",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=DEFAULT_LOOKAHEAD_DAYS,
        help="Days ahead for TradingEconomics data when --end absent.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (TradingEconomics, ISO).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (TradingEconomics, ISO).",
    )
    parser.add_argument(
        "--min-importance",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Minimum TradingEconomics importance (1=low, 3=high).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC,ETH",
        help="Comma-separated symbols affected by these macro events.",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=7,
        help="Chunk size in days when fetching the calendar (default 7).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="TradingEconomics API key (default reads TE_API_KEY env or guest:guest).",
    )
    parser.add_argument(
        "--fomc-years",
        type=str,
        default=None,
        help="Comma-separated years to include for FOMC source (defaults to current and next year).",
    )
    parser.add_argument(
        "--manual-files",
        type=str,
        default=None,
        help="Comma-separated JSON files containing manual events.",
    )
    parser.add_argument(
        "--manual-category",
        type=str,
        default="macro",
        help="Default category for manual events when omitted.",
    )
    parser.add_argument(
        "--manual-impact",
        type=str,
        default="high",
        help="Default impact for manual events when omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and display events without writing to the store.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SUPPORTED_SOURCES]
    if unknown:
        print(f"Unsupported sources: {', '.join(unknown)}", file=sys.stderr)
        sys.exit(1)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    events: List[CalendarEvent] = []

    if "tradingeconomics" in sources:
        now = datetime.now(UTC)
        start = _parse_cli_datetime(args.start) if args.start else now
        end = _parse_cli_datetime(args.end) if args.end else start + timedelta(days=args.lookahead_days)
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
        api_key = (
            args.api_key
            or os.getenv("TE_API_KEY")
            or os.getenv("TRADING_ECONOMICS_KEY")
            or "guest:guest"
        )

        try:
            entries = fetch_trading_economics(
                countries=countries,
                start=start,
                end=end,
                min_importance=args.min_importance,
                api_key=api_key,
                chunk_days=args.chunk_days,
            )
        except requests.HTTPError as exc:
            status = exc.response.status_code if getattr(exc, "response", None) is not None else None
            if status and status >= 500:
                print(
                    "TradingEconomics returned a server error (likely guest API limits). "
                    "No TradingEconomics events imported; retry later or provide --api-key.",
                    file=sys.stderr,
                )
            else:
                print(f"Failed to fetch TradingEconomics calendar: {exc}", file=sys.stderr)
            entries = []
        except Exception as exc:
            print(f"Failed to fetch TradingEconomics calendar: {exc}", file=sys.stderr)
            entries = []

        if entries:
            entries = filter_entries(entries, min_importance=args.min_importance)
            upcoming_entries = [entry for entry in entries if _parse_date(entry["Date"]) >= now]
            events.extend(
                convert_te_entry(entry, symbols=symbols) for entry in upcoming_entries
            )

    if "fomc" in sources:
        if args.fomc_years:
            fomc_years = [int(y.strip()) for y in args.fomc_years.split(",") if y.strip()]
        else:
            current_year = datetime.now(UTC).year
            fomc_years = [current_year, current_year + 1]
        try:
            fomc_events = fetch_fomc_events(symbols=symbols, years=fomc_years)
            events.extend(fomc_events)
        except Exception as exc:
            print(f"Failed to fetch FOMC schedule: {exc}", file=sys.stderr)

    manual_requested = "manual" in sources or bool(args.manual_files)
    if manual_requested and "manual" not in sources:
        sources.append("manual")

    if "manual" in sources:
        manual_paths: List[Path] = []
        if args.manual_files:
            manual_paths = [Path(p.strip()).expanduser() for p in args.manual_files.split(",") if p.strip()]
        elif DEFAULT_MANUAL_FILE.exists():
            manual_paths = [DEFAULT_MANUAL_FILE]
        else:
            print("No --manual-files supplied and default dataset missing; manual source skipped.", file=sys.stderr)

        if manual_paths:
            manual_events = load_manual_events(
                manual_paths,
                default_symbols=symbols,
                default_category=args.manual_category,
                default_impact=args.manual_impact,
            )
            events.extend(manual_events)

    if not events:
        print("No matching events collected.")
        return

    events.sort(key=lambda ev: ev.start_utc)

    if args.dry_run:
        for event in events:
            print(
                f"{event.start_utc.strftime('%Y-%m-%d %H:%M')} UTC | "
                f"{event.impact.upper():<6} | {event.name} | {event.notes or ''}"
            )
        return

    if args.store:
        calendar = RiskCalendar(store_path=Path(args.store).expanduser())
    else:
        calendar = RiskCalendar()
    written = calendar.upsert_events(events)
    print(f"Upserted {len(written)} events into {calendar.store_path}")


if __name__ == "__main__":
    main()
