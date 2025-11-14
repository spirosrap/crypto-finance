import unittest
from datetime import datetime, timedelta, timezone

from paper_finder_simulator import (
    UTC,
    _close_and_update_rows,
    _compute_unrealized_pct,
    _isoformat,
    _maybe_close_reason,
    gather_candidates,
)


SAMPLE_TEXT = """
1. ALPHA (Alpha Coin) — LONG
Entry Price: 10.00
Take Profit: 12.00
Stop Loss: 9.00
Recommended Position Size: 5%
Overall Score: 78

2. BETA (Beta Asset) — SHORT
Entry Price: 100.0
Take Profit: 90.0
Stop Loss: 110.0
Recommended Position Size: 4%
Score: 81.5
"""


class PaperFinderSimulatorTests(unittest.TestCase):
    def test_gather_candidates_parses_blocks(self) -> None:
        candidates = gather_candidates(SAMPLE_TEXT)
        self.assertEqual(2, len(candidates))
        first = candidates[0]
        self.assertEqual("ALPHA", first.parsed.symbol)
        self.assertEqual("LONG", first.parsed.side)
        self.assertAlmostEqual(10.0, first.parsed.entry)
        self.assertAlmostEqual(78.0, first.score)
        second = candidates[1]
        self.assertEqual("BETA", second.parsed.symbol)
        self.assertEqual("SHORT", second.parsed.side)
        self.assertAlmostEqual(81.5, second.score)

    def test_compute_unrealized_pct_handles_long_and_short(self) -> None:
        self.assertAlmostEqual(10.0, _compute_unrealized_pct("LONG", 100.0, 110.0))
        self.assertAlmostEqual(-10.0, _compute_unrealized_pct("LONG", 100.0, 90.0))
        self.assertAlmostEqual(10.0, _compute_unrealized_pct("SHORT", 100.0, 90.0))
        self.assertAlmostEqual(-5.0, _compute_unrealized_pct("SHORT", 100.0, 105.0))

    def test_maybe_close_reason_checks_tp_sl_and_expiry(self) -> None:
        now = datetime.now(tz=UTC)
        expires = now + timedelta(hours=2)
        self.assertEqual(
            "take_profit",
            _maybe_close_reason("LONG", 12.5, 12.0, 9.0, expires, now),
        )
        self.assertEqual(
            "stop_loss",
            _maybe_close_reason("LONG", 8.5, 12.0, 9.0, expires, now),
        )
        self.assertEqual(
            "take_profit",
            _maybe_close_reason("SHORT", 88.0, 90.0, 110.0, expires, now),
        )
        expired = now - timedelta(minutes=1)
        self.assertEqual(
            "expired_breakeven",
            _maybe_close_reason("SHORT", 100.0, 90.0, 110.0, expired, now),
        )

    def test_close_and_update_rows_marks_closed_and_updates_remaining(self) -> None:
        now = datetime.now(tz=UTC)
        open_rows = [
            {
                "trade_id": "t1",
                "product_id": "ALPHA-PERP-INTX",
                "position_side": "LONG",
                "entry_price": 10.0,
                "stop_loss": 9.0,
                "take_profit": 12.0,
                "position_usd": 1000.0,
                "opened_at": _isoformat(now - timedelta(hours=1)),
                "expires_at": _isoformat(now + timedelta(hours=23)),
            },
            {
                "trade_id": "t2",
                "product_id": "BETA-PERP-INTX",
                "position_side": "SHORT",
                "entry_price": 100.0,
                "stop_loss": 110.0,
                "take_profit": 90.0,
                "position_usd": 2000.0,
                "opened_at": _isoformat(now - timedelta(hours=2)),
                "expires_at": _isoformat(now + timedelta(hours=20)),
            },
        ]

        price_map = {
            "ALPHA-PERP-INTX": 12.2,  # hits long TP
            "BETA-PERP-INTX": 112.0,  # hits short stop
        }

        def lookup(product_id: str) -> float:
            return price_map[product_id]

        updated, closed = _close_and_update_rows(open_rows, lookup, now)
        self.assertEqual(0, len(updated))
        self.assertEqual(2, len(closed))
        reasons = {row["product_id"]: row["closure_reason"] for row in closed}
        self.assertEqual("take_profit", reasons["ALPHA-PERP-INTX"])
        self.assertEqual("stop_loss", reasons["BETA-PERP-INTX"])
        pnl_lookup = {row["product_id"]: row["profit_loss"] for row in closed}
        self.assertGreater(pnl_lookup["ALPHA-PERP-INTX"], 0)
        self.assertLess(pnl_lookup["BETA-PERP-INTX"], 0)


if __name__ == "__main__":
    unittest.main()
