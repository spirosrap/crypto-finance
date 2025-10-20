import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from risk_calendar import CalendarEvent, RiskCalendar, UTC


class RiskCalendarTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store_path = Path(self.temp_dir.name) / "risk_events.json"
        self.calendar = RiskCalendar(store_path=self.store_path)

    def test_add_and_list_event(self) -> None:
        start_dt = datetime(2025, 10, 22, 18, 0, tzinfo=UTC)
        event = CalendarEvent(
            id="abc123",
            name="FOMC decision",
            start_utc=start_dt,
            category="macro",
            impact="high",
            symbols=["BTC", "ETH"],
            notes="Watch for volatility",
        )
        self.calendar.add_event(event)

        results = self.calendar.list_events()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].name, "FOMC decision")
        self.assertEqual(results[0].symbols, ["BTC", "ETH"])

    def test_remove_event(self) -> None:
        event = CalendarEvent(
            id="to-remove",
            name="Test Removal",
            start_utc=datetime(2025, 11, 1, 12, 0, tzinfo=UTC),
        )
        self.calendar.add_event(event)
        removed = self.calendar.remove_event("to-remove")
        self.assertTrue(removed)
        self.assertEqual(self.calendar.list_events(), [])

    def test_upcoming_window(self) -> None:
        now = datetime.now(UTC)
        event_inside = CalendarEvent(
            id="inside",
            name="Event Inside",
            start_utc=now + timedelta(days=2),
        )
        event_outside = CalendarEvent(
            id="outside",
            name="Event Outside",
            start_utc=now + timedelta(days=10),
        )
        self.calendar.add_event(event_inside)
        self.calendar.add_event(event_outside)

        upcoming = self.calendar.upcoming(within=timedelta(days=7))
        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming[0].id, "inside")

    def test_upsert_events_replaces_existing(self) -> None:
        start_dt = datetime(2025, 12, 1, 12, 0, tzinfo=UTC)
        original = CalendarEvent(
            id="dup",
            name="Original Event",
            start_utc=start_dt,
            category="macro",
        )
        updated = CalendarEvent(
            id="dup",
            name="Updated Event",
            start_utc=start_dt + timedelta(hours=1),
            category="macro",
        )
        self.calendar.add_event(original)
        written = self.calendar.upsert_events([updated])
        self.assertEqual(len(written), 1)
        events = self.calendar.list_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "Updated Event")
        self.assertEqual(events[0].start_utc, updated.start_utc)


if __name__ == "__main__":
    unittest.main()
