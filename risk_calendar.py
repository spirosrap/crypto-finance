#!/usr/bin/env python3
"""
Risk calendar management tool.

Maintains a structured list of upcoming macro/crypto risk events so
trading playbooks can quickly decide whether to run, pause, or scale
positions around the current pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


UTC = timezone.utc
DEFAULT_STORE = Path("risk_calendar") / "risk_events.json"
DEFAULT_CATEGORY = "macro"
DEFAULT_IMPACT = "medium"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _normalise_dt(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_datetime(value: str) -> datetime:
    """Parse ISO-8601 or YYYY-MM-DD into an aware UTC datetime."""
    if not value:
        raise argparse.ArgumentTypeError("Datetime value cannot be empty")
    text = value.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        base = datetime.fromisoformat(text)
        return base.replace(tzinfo=UTC)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid datetime: {value}") from exc
    return _normalise_dt(parsed)


def parse_symbols(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


@dataclass
class CalendarEvent:
    id: str
    name: str
    start_utc: datetime
    end_utc: Optional[datetime] = None
    category: str = DEFAULT_CATEGORY
    impact: str = DEFAULT_IMPACT
    symbols: Sequence[str] = field(default_factory=list)
    source: Optional[str] = None
    notes: Optional[str] = None
    completed: bool = False
    created_utc: datetime = field(default_factory=_utcnow)
    updated_utc: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "start_utc": self.start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": self.end_utc.strftime("%Y-%m-%dT%H:%M:%SZ") if self.end_utc else None,
            "category": self.category,
            "impact": self.impact,
            "symbols": list(self.symbols),
            "source": self.source,
            "notes": self.notes,
            "completed": self.completed,
            "created_utc": self.created_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "updated_utc": self.updated_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalendarEvent":
        return cls(
            id=data["id"],
            name=data["name"],
            start_utc=parse_datetime(data["start_utc"]),
            end_utc=parse_datetime(data["end_utc"]) if data.get("end_utc") else None,
            category=data.get("category", DEFAULT_CATEGORY),
            impact=data.get("impact", DEFAULT_IMPACT),
            symbols=list(data.get("symbols") or []),
            source=data.get("source"),
            notes=data.get("notes"),
            completed=bool(data.get("completed", False)),
            created_utc=parse_datetime(data["created_utc"]) if data.get("created_utc") else _utcnow(),
            updated_utc=parse_datetime(data["updated_utc"]) if data.get("updated_utc") else _utcnow(),
        )


class RiskCalendar:
    def __init__(self, store_path: Path = DEFAULT_STORE) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[CalendarEvent] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self.store_path.exists():
            self._events = []
            self._loaded = True
            return
        with self.store_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        events = payload.get("events", [])
        self._events = [CalendarEvent.from_dict(item) for item in events]
        self._loaded = True

    def save(self) -> None:
        payload = {
            "version": 1,
            "generated_utc": _utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "events": [event.to_dict() for event in sorted(self._events, key=lambda e: e.start_utc)],
        }
        tmp_path = self.store_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(self.store_path)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def all_events(self) -> List[CalendarEvent]:
        self._ensure_loaded()
        return list(self._events)

    def add_event(self, event: CalendarEvent) -> CalendarEvent:
        self._ensure_loaded()
        self._events.append(event)
        self.save()
        return event

    def upsert_events(self, events: Iterable[CalendarEvent], *, replace: bool = True) -> List[CalendarEvent]:
        """Insert or update multiple events by identifier.

        Args:
            events: Iterable of CalendarEvent objects to persist.
            replace: When True, replace matching ids with the incoming payload.

        Returns the list of events written to disk.
        """

        self._ensure_loaded()
        index = {event.id: idx for idx, event in enumerate(self._events)}
        persisted: List[CalendarEvent] = []
        for incoming in events:
            if incoming.id in index:
                if replace:
                    self._events[index[incoming.id]] = incoming
                    persisted.append(incoming)
            else:
                self._events.append(incoming)
                index[incoming.id] = len(self._events) - 1
                persisted.append(incoming)
        if persisted:
            self.save()
        return persisted

    def remove_event(self, event_id: str) -> bool:
        self._ensure_loaded()
        initial = len(self._events)
        self._events = [item for item in self._events if item.id != event_id]
        removed = len(self._events) != initial
        if removed:
            self.save()
        return removed

    def mark_completed(self, event_id: str, completed: bool = True) -> bool:
        self._ensure_loaded()
        for event in self._events:
            if event.id == event_id:
                event.completed = completed
                event.updated_utc = _utcnow()
                self.save()
                return True
        return False

    def list_events(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        categories: Optional[Sequence[str]] = None,
        impacts: Optional[Sequence[str]] = None,
        include_completed: bool = True,
    ) -> List[CalendarEvent]:
        self._ensure_loaded()
        result: List[CalendarEvent] = []
        start_utc = _normalise_dt(start) if start else None
        end_utc = _normalise_dt(end) if end else None
        categories_set = {c.lower() for c in categories} if categories else None
        impacts_set = {i.lower() for i in impacts} if impacts else None
        for event in self._events:
            if not include_completed and event.completed:
                continue
            if start_utc and event.start_utc < start_utc:
                continue
            if end_utc and event.start_utc >= end_utc:
                continue
            if categories_set and event.category.lower() not in categories_set:
                continue
            if impacts_set and event.impact.lower() not in impacts_set:
                continue
            result.append(event)
        return sorted(result, key=lambda ev: ev.start_utc)

    def upcoming(self, *, within: timedelta, include_completed: bool = False) -> List[CalendarEvent]:
        now = _utcnow()
        horizon = now + within
        return self.list_events(start=now, end=horizon, include_completed=include_completed)


def _print_events(events: Sequence[CalendarEvent]) -> None:
    if not events:
        print("No matching events.")
        return
    columns = [
        ("ID", 8),
        ("Start (UTC)", 20),
        ("Impact", 8),
        ("Category", 12),
        ("Name", 28),
        ("Symbols", 18),
        ("Completed", 9),
    ]
    header = "  ".join(f"{name:<{width}}" for name, width in columns)
    print(header)
    print("-" * len(header))
    for event in events:
        row = [
            event.id[:8],
            event.start_utc.strftime("%Y-%m-%d %H:%M"),
            event.impact,
            event.category,
            event.name,
            ",".join(event.symbols) if event.symbols else "-",
            "yes" if event.completed else "no",
        ]
        print("  ".join(f"{value:<{width}}" for value, (_, width) in zip(row, columns)))
        if event.notes:
            print(f"    notes: {event.notes}")
        if event.source:
            print(f"    source: {event.source}")
        if event.end_utc:
            print(f"    ends:  {event.end_utc.strftime('%Y-%m-%d %H:%M')} UTC")


def _add_event_from_args(calendar: RiskCalendar, args: argparse.Namespace) -> None:
    event = CalendarEvent(
        id=str(uuid.uuid4()),
        name=args.name,
        start_utc=parse_datetime(args.start),
        end_utc=parse_datetime(args.end) if args.end else None,
        category=args.category,
        impact=args.impact,
        symbols=parse_symbols(args.symbols),
        source=args.source,
        notes=args.notes,
    )
    calendar.add_event(event)
    print(f"Added event {event.name} ({event.id}) starting {event.start_utc.isoformat()}")


def _list_events_from_args(calendar: RiskCalendar, args: argparse.Namespace) -> None:
    categories = parse_symbols(args.categories) if args.categories else None
    impacts = parse_symbols(args.impacts) if args.impacts else None
    events = calendar.list_events(
        start=parse_datetime(args.start) if args.start else None,
        end=parse_datetime(args.end) if args.end else None,
        categories=categories,
        impacts=impacts,
        include_completed=not args.hide_completed,
    )
    if args.json:
        json.dump([event.to_dict() for event in events], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_events(events)


def _upcoming_from_args(calendar: RiskCalendar, args: argparse.Namespace) -> None:
    window = timedelta(days=args.days)
    events = calendar.upcoming(within=window, include_completed=False)
    _print_events(events)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage risk calendar events for the trading pipeline.")
    parser.add_argument("--store", type=str, default=None, help="Override the default JSON store path.")
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="Add a new calendar event.")
    add_parser.add_argument("--name", required=True, help="Short event name.")
    add_parser.add_argument("--start", required=True, help="ISO-8601 start datetime (UTC or local).")
    add_parser.add_argument("--end", help="Optional end datetime.")
    add_parser.add_argument("--category", default=DEFAULT_CATEGORY, help="Event category (macro, crypto, exchange, regulatory, ...).")
    add_parser.add_argument("--impact", default=DEFAULT_IMPACT, help="Impact level (low, medium, high).")
    add_parser.add_argument("--symbols", help="Comma-separated symbols affected.")
    add_parser.add_argument("--source", help="Optional link or description of the information source.")
    add_parser.add_argument("--notes", help="Free-form notes.")

    list_parser = subparsers.add_parser("list", help="List events by time window.")
    list_parser.add_argument("--start", help="Start datetime (inclusive).")
    list_parser.add_argument("--end", help="End datetime (exclusive).")
    list_parser.add_argument("--categories", help="Comma-separated categories filter.")
    list_parser.add_argument("--impacts", help="Comma-separated impact levels filter.")
    list_parser.add_argument("--hide-completed", action="store_true", help="Hide completed events.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON instead of a table.")

    upcoming_parser = subparsers.add_parser("upcoming", help="Show upcoming events within N days.")
    upcoming_parser.add_argument("--days", type=int, default=7, help="Lookahead window in days (default 7).")

    remove_parser = subparsers.add_parser("remove", help="Remove an event by id.")
    remove_parser.add_argument("event_id", help="Event identifier (UUID).")

    complete_parser = subparsers.add_parser("complete", help="Mark an event as (in)complete.")
    complete_parser.add_argument("event_id", help="Event identifier (UUID).")
    complete_parser.add_argument("--undo", action="store_true", help="Clear completion flag instead of setting it.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return
    store_path = Path(args.store).expanduser() if args.store else DEFAULT_STORE
    calendar = RiskCalendar(store_path=store_path)
    if args.command == "add":
        _add_event_from_args(calendar, args)
    elif args.command == "list":
        _list_events_from_args(calendar, args)
    elif args.command == "upcoming":
        _upcoming_from_args(calendar, args)
    elif args.command == "remove":
        removed = calendar.remove_event(args.event_id)
        if removed:
            print(f"Removed event {args.event_id}")
        else:
            print(f"No event found for id {args.event_id}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "complete":
        updated = calendar.mark_completed(args.event_id, completed=not args.undo)
        if updated:
            state = "completed" if not args.undo else "pending"
            print(f"Event {args.event_id} marked {state}.")
        else:
            print(f"No event found for id {args.event_id}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
