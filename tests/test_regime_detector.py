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

