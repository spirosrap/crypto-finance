import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import paper_finder_simulator as sim
from paper_finder_simulator import (
    UTC,
    _close_and_update_rows,
    _compute_unrealized_pct,
    _filter_supported_candidates,
    _format_time_left,
    _isoformat,
    _maybe_close_reason,
    gather_candidates,
)
from add_position_from_finder import ParsedFinder


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
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        base = Path(self.tempdir.name)
        sim.OPEN_CSV = base / "open.csv"
        sim.CLOSED_CSV = base / "closed.csv"
        sim._SUPPORTED_PERPS = set()
        sim._CCXT_PRODUCTS = set()
        sim._EXCLUDED_PERPS = set()

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

    def test_filter_supported_candidates_skips_unknown_products(self) -> None:
        sim._SUPPORTED_PERPS = {"ALPHA-PERP-INTX"}
        candidates = gather_candidates(SAMPLE_TEXT)
        filtered = _filter_supported_candidates(candidates)
        self.assertEqual(1, len(filtered))
        self.assertEqual("ALPHA", filtered[0].parsed.symbol)

    def test_filter_supported_candidates_skips_excluded(self) -> None:
        sim._SUPPORTED_PERPS = {"ALPHA-PERP-INTX", "BETA-PERP-INTX"}
        sim._EXCLUDED_PERPS = {"BETA-PERP-INTX"}
        candidates = gather_candidates(SAMPLE_TEXT)
        filtered = _filter_supported_candidates(candidates)
        symbols = [cand.parsed.symbol for cand in filtered]
        self.assertEqual(["ALPHA"], symbols)
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

    def test_format_time_left_outputs_human_readable(self) -> None:
        now = datetime.now(tz=UTC)
        expires_hours = now + timedelta(hours=5, minutes=30)
        expires_days = now + timedelta(days=1, hours=3)
        self.assertEqual("5h 30m", _format_time_left(_isoformat(expires_hours), now))
        self.assertEqual("1d 3h", _format_time_left(_isoformat(expires_days), now))

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

    def test_open_selected_trades_appends_rows(self) -> None:
        parsed = ParsedFinder(
            symbol="ALPHA",
            base_symbol="ALPHA",
            side="LONG",
            entry=10.0,
            stop=9.0,
            take_profit=12.0,
            pos_size_pct=5.0,
            entry_decimals=2,
            stop_decimals=2,
            take_profit_decimals=2,
            predicted_return=None,
        )
        candidate = sim.FinderCandidate(rank=1, block="ALPHA", parsed=parsed, score=70.0)

        sim._open_selected_trades(
            [candidate],
            portfolio_usd=10000.0,
            leverage=3.0,
            expiry_hours=24,
            default_pct=5.0,
            tag="unit",
            note="single",
            fixed_position_usd=None,
            dry_run=False,
        )

        df = pd.read_csv(sim.OPEN_CSV)
        self.assertEqual(1, len(df))
        self.assertEqual("ALPHA-PERP-INTX", df.loc[0, "product_id"])
        self.assertAlmostEqual(500.0, df.loc[0, "position_usd"])

    def test_balanced_top_selection_prioritises_sides(self) -> None:
        def _candidate(symbol: str, side: str, score: float, rank: int) -> sim.FinderCandidate:
            parsed = ParsedFinder(
                symbol=symbol,
                base_symbol=symbol,
                side=side,
                entry=10.0,
                stop=9.0,
                take_profit=12.0,
                pos_size_pct=5.0,
                entry_decimals=2,
                stop_decimals=2,
                take_profit_decimals=2,
                predicted_return=None,
            )
            return sim.FinderCandidate(rank=rank, block=symbol, parsed=parsed, score=score)

        candidates = [
            _candidate("L1", "LONG", 90, 1),
            _candidate("S1", "SHORT", 85, 2),
            _candidate("L2", "LONG", 80, 3),
            _candidate("S2", "SHORT", 75, 4),
            _candidate("L3", "LONG", 70, 5),
            _candidate("S3", "SHORT", 65, 6),
        ]

        picks = sim._select_balanced_top(candidates, total=5)
        self.assertEqual(5, len(picks))
        long_count = sum(1 for c in picks if c.parsed.side == "LONG")
        short_count = sum(1 for c in picks if c.parsed.side == "SHORT")
        self.assertGreaterEqual(long_count, 2)
        self.assertGreaterEqual(short_count, 2)
        symbols = [c.parsed.symbol for c in picks]
        self.assertIn("L3", symbols)  # remaining best overall after 2+2

    def test_handle_update_tracks_unrealized_pnl(self) -> None:
        now = datetime.now(tz=UTC)
        open_row = {
            "trade_id": "t-open",
            "symbol": "ALPHA",
            "product_id": "ALPHA-PERP-INTX",
            "position_side": "LONG",
            "entry_price": 10.0,
            "stop_loss": 9.0,
            "take_profit": 13.0,
            "position_usd": 1000.0,
            "leverage": 3.0,
            "opened_at": _isoformat(now - timedelta(hours=1)),
            "expires_at": _isoformat(now + timedelta(hours=23)),
            "status": "OPEN",
            "last_price": 10.0,
            "last_price_at": _isoformat(now - timedelta(hours=1)),
            "unrealized_pnl": 0.0,
            "unrealized_pct": 0.0,
            "finder_score": 70.0,
            "finder_rank": 1,
            "recommended_position_pct": 5.0,
            "tag": "",
            "notes": "",
        }
        pd.DataFrame([open_row]).to_csv(sim.OPEN_CSV, index=False)
        pd.DataFrame(columns=sim.CLOSED_COLUMNS).to_csv(sim.CLOSED_CSV, index=False)

        args = SimpleNamespace(override=["ALPHA=11"])
        sim.handle_update(args)

        df = pd.read_csv(sim.OPEN_CSV)
        self.assertGreater(float(df.loc[0, "unrealized_pnl"]), 0.0)


if __name__ == "__main__":
    unittest.main()
