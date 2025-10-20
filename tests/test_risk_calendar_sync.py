import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest import mock

import requests

from risk_calendar_sync import (
    DEFAULT_MANUAL_FILE,
    convert_te_entry,
    filter_entries,
    fetch_trading_economics,
    load_manual_events,
    parse_fomc_html,
    _parse_date,
)


class RiskCalendarSyncTests(unittest.TestCase):
    def test_parse_date_assumes_utc(self) -> None:
        dt = _parse_date("2025-05-01T12:30:00")
        self.assertEqual(dt.tzinfo, timezone.utc)
        self.assertEqual(dt.isoformat(), "2025-05-01T12:30:00+00:00")

    def test_convert_te_entry(self) -> None:
        entry = {
            "CalendarId": "999",
            "Date": "2025-05-01T12:30:00",
            "Importance": 3,
            "Category": "Nonfarm Payrolls",
            "Event": "Nonfarm Payrolls",
            "Previous": "210K",
            "Forecast": "180K",
            "Source": "Bureau of Labor Statistics",
            "SourceURL": "https://www.bls.gov/",
        }
        event = convert_te_entry(entry, symbols=["BTC"])
        self.assertEqual(event.id, "te-999")
        self.assertEqual(event.impact, "high")
        self.assertTrue(event.notes)
        self.assertEqual(event.symbols, ["BTC"])
        self.assertEqual(event.category, "nonfarm_payrolls")

    def test_filter_entries(self) -> None:
        entries = [
            {"Importance": 3},
            {"Importance": 2},
            {"Importance": 1},
        ]
        filtered = filter_entries(entries, min_importance=2)
        self.assertEqual(len(filtered), 2)

    def test_parse_fomc_html(self) -> None:
        sample_html = """
        <div class="panel">
            <div class="panel-heading"><h4>2025 FOMC Meetings</h4></div>
            <div class="row fomc-meeting">
                <div class="fomc-meeting__month">January</div>
                <div class="fomc-meeting__date">28-29*</div>
                <div class="fomc-meeting__info"></div>
            </div>
            <div class="row fomc-meeting">
                <div class="fomc-meeting__month">August</div>
                <div class="fomc-meeting__date">22 (notation vote)</div>
                <div class="fomc-meeting__info"></div>
            </div>
        </div>
        """
        events = parse_fomc_html(sample_html, symbols=["BTC"], years=[2025])
        self.assertEqual(len(events), 2)
        self.assertIn("Press conference", events[0].notes)
        self.assertIn("notation vote", events[1].notes.lower())

    @mock.patch("risk_calendar_sync.time.sleep", return_value=None)
    @mock.patch("risk_calendar_sync.requests.get")
    def test_fetch_trading_economics_retries(self, mock_get, _mock_sleep) -> None:
        class FakeResponse:
            def __init__(self, status_code: int, payload):
                self.status_code = status_code
                self._payload = payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise requests.HTTPError(response=self)

            def json(self):
                return self._payload

        mock_get.side_effect = [
            FakeResponse(500, []),
            FakeResponse(200, [{"CalendarId": "1", "Date": "2025-05-01T12:30:00", "Country": "United States"}]),
        ]

        start = datetime(2025, 5, 1, tzinfo=timezone.utc)
        end = start
        result = fetch_trading_economics(
            countries=["United States"],
            start=start,
            end=end,
            min_importance=3,
            api_key="guest:guest",
            chunk_days=1,
            max_retries=2,
            retry_delay_seconds=0,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["CalendarId"], "1")

    def test_load_manual_events(self) -> None:
        manual_events = load_manual_events(
            [DEFAULT_MANUAL_FILE],
            default_symbols=["BTC", "ETH"],
        )
        self.assertGreaterEqual(len(manual_events), 1)
        self.assertTrue(all(event.start_utc.tzinfo is timezone.utc for event in manual_events))


if __name__ == "__main__":
    unittest.main()
