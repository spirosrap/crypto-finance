import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

from long_term_crypto_finder import LongTermCryptoFinder, CryptoFinderConfig


class TestIndicators(unittest.TestCase):
    def setUp(self):
        cfg = CryptoFinderConfig(offline=True)
        self.finder = LongTermCryptoFinder(cfg)

    def test_rsi_wilder_monotonic(self):
        # Strictly increasing should yield RSI ~ 100
        prices_up = np.arange(1, 1 + 20, dtype=float)
        rsi_up = self.finder._rsi_wilder(prices_up, period=14)
        self.assertGreaterEqual(rsi_up, 99.0)

        # Strictly decreasing should yield RSI ~ 0
        prices_dn = np.arange(20, 0, -1, dtype=float)
        rsi_dn = self.finder._rsi_wilder(prices_dn, period=14)
        self.assertLessEqual(rsi_dn, 1.0)

    def test_macd_label(self):
        # Uptrend should be bullish histogram
        prices = np.linspace(100, 200, 60)
        macd, sig, hist, label = self.finder._macd_signal(prices, fast=12, slow=26, signal=9)
        self.assertEqual(label, "BULLISH")
        # Downtrend should be bearish
        prices2 = np.linspace(200, 100, 60)
        macd2, sig2, hist2, label2 = self.finder._macd_signal(prices2, fast=12, slow=26, signal=9)
        self.assertEqual(label2, "BEARISH")

    def test_atr_units(self):
        # Build a small OHLC set to compute TRs
        base = datetime(2024, 1, 1, tzinfo=UTC)
        rows = [
            {"timestamp": base + timedelta(days=i), "open": o, "high": h, "low": l, "price": c, "volume": 0.0}
            for i, (o, h, l, c) in enumerate([
                (100, 110, 90, 105),  # seed
                (105, 115, 95, 110),
                (110, 120, 100, 115),
                (115, 125, 105, 120),
            ])
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        atr = self.finder._calculate_atr(df, period=3)
        # Compute expected TRs for last 3 periods
        # TR1 (idx 1): max(h-l=20, |h-prev_c|=10, |l-prev_c|=10) = 20
        # TR2 (idx 2): max(20, 10, 10) = 20
        # TR3 (idx 3): max(20, 10, 10) = 20
        self.assertAlmostEqual(atr, 20.0, places=6)

    def test_max_drawdown(self):
        base = datetime(2024, 1, 1, tzinfo=UTC)
        prices = [100, 120, 80, 90, 70]
        highs = [100, 120, 100, 95, 90]
        lows = [95, 110, 75, 80, 65]
        rows = [
            {"timestamp": base + timedelta(days=i), "price": p, "high": h, "low": l, "open": p, "volume": 0.0}
            for i, (p, h, l) in enumerate(zip(prices, highs, lows))
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        metrics = self.finder.calculate_technical_indicators(df)
        mdd = metrics.get("max_drawdown", 0)
        # Peak at 120 to trough 70 => (70-120)/120 = -0.416666...
        self.assertAlmostEqual(mdd, -0.4166666667, places=6)


if __name__ == "__main__":
    unittest.main()

