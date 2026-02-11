import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock


class RegimeDetectorTests(unittest.TestCase):
    def test_calculate_ema(self) -> None:
        from research_agent.regime_detector import calculate_ema

        # period=3, multiplier=0.5, seed=1
        # ema1=1
        # ema2=(2-1)*0.5+1=1.5
        # ema3=(3-1.5)*0.5+1.5=2.25
        self.assertAlmostEqual(calculate_ema([1, 2, 3], 3), 2.25, places=10)

    def test_get_regime_status_bullish(self) -> None:
        from research_agent.regime_detector import get_regime_status

        candles = []
        for i in range(20):
            # [time, low, high, open, close, volume]
            candles.append([1000 + i, 100, 100, 100, 100, 1])

        stats = {"last": "103", "open": "100", "high": "110", "low": "90", "volume": "10"}

        with mock.patch("research_agent.regime_detector.get_coinbase_candles", return_value=candles), mock.patch(
            "research_agent.regime_detector.get_coinbase_stats", return_value=stats
        ):
            status = get_regime_status("BTC-USD", ema_period=20, neutral_band_pct=2.0)
        self.assertEqual(status["regime"], "BULLISH")
        self.assertIn("70% long", status["recommendation"])
        self.assertGreater(status["distance_pct"], 2.0)
        self.assertIn("atr_raw", status)
        self.assertIn("atr_used", status)

    def test_get_regime_status_bearish(self) -> None:
        from research_agent.regime_detector import get_regime_status

        candles = []
        for i in range(20):
            candles.append([1000 + i, 100, 100, 100, 100, 1])

        stats = {"last": "97", "open": "100", "high": "110", "low": "90", "volume": "10"}

        with mock.patch("research_agent.regime_detector.get_coinbase_candles", return_value=candles), mock.patch(
            "research_agent.regime_detector.get_coinbase_stats", return_value=stats
        ):
            status = get_regime_status("BTC-USD", ema_period=20, neutral_band_pct=2.0)
        self.assertEqual(status["regime"], "BEARISH")
        self.assertIn("70% short", status["recommendation"])
        self.assertLess(status["distance_pct"], -2.0)
        self.assertIn("atr_raw", status)
        self.assertIn("atr_used", status)

    def test_calculate_atr_wilder(self) -> None:
        from research_agent.coinbase_api import Candle
        from research_agent.regime_detector import calculate_atr_wilder

        candles = [
            Candle(time=1, low=8.0, high=10.0, open=9.0, close=9.0, volume=1.0),
            Candle(time=2, low=9.0, high=11.0, open=10.0, close=10.0, volume=1.0),
            Candle(time=3, low=10.0, high=12.0, open=11.0, close=11.0, volume=1.0),
            Candle(time=4, low=11.0, high=14.0, open=12.0, close=13.0, volume=1.0),
        ]

        # TRs after first candle: 2, 2, 3; period=2 => ATR = ((2+2)/2 then Wilder update with 3) = 2.5
        self.assertAlmostEqual(calculate_atr_wilder(candles, period=2), 2.5, places=10)

    def test_scan_opportunities_includes_reclaim_rejection_and_levels(self) -> None:
        from research_agent.regime_detector import scan_opportunities

        statuses = {
            "BTC-USD": {
                "current_price": 75000.0,
                "ema_20": 75946.21,
                "atr_period": 7,
                "atr_used": 700.0,
                "24h_high": 77000.0,
            }
        }

        lines = scan_opportunities(
            statuses,
            sl_atr_mult=0.8,
            tp1_rr=0.8,
            tp2_rr=1.5,
            entry_buffer_pct=0.3,
        )
        self.assertGreaterEqual(len(lines), 1)
        self.assertIn("reclaim close above EMA", lines[0])
        self.assertIn("Entries: LONG reclaim close >=", lines[0])
        self.assertIn("SHORT rejection close <=", lines[0])
        self.assertIn("SHORT SL", lines[0])
        self.assertIn("TP1", lines[0])
        self.assertIn("TP2", lines[0])

    def test_upsert_regime_history_csv(self) -> None:
        from research_agent.regime_detector import upsert_regime_history_csv

        run_date = date(2026, 2, 6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regime_history_BTC.csv"

            upsert_regime_history_csv(path, run_date=run_date, price=100.0, ema=95.0, regime="BULLISH")
            upsert_regime_history_csv(path, run_date=run_date, price=101.0, ema=96.0, regime="BULLISH")

            rows = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(rows), 2)
            # header + 1 row
            self.assertEqual(len(rows), 2)
            self.assertIn("2026-02-06", rows[1])
            self.assertIn("101.00000000", rows[1])
            self.assertIn("96.00000000", rows[1])


if __name__ == "__main__":
    unittest.main()
