import unittest
from datetime import UTC, datetime, timedelta

import numpy as np

from scripts.watchdog_baseline_backtest import (
    _effective_atr_cap_usd,
    calculate_atr_wilder,
    simulate_bracket_exit,
)


class WatchdogBaselineBacktestTests(unittest.TestCase):
    def test_calculate_atr_wilder_constant_tr(self) -> None:
        # Constant TR=2 -> ATR should converge to 2.
        n = 30
        high = np.full(n, 11.0)
        low = np.full(n, 9.0)
        close = np.full(n, 10.0)
        atr = calculate_atr_wilder(high, low, close, period=7)
        self.assertAlmostEqual(atr, 2.0, places=6)

    def test_effective_atr_cap_usd_tiered_bps_for_btc(self) -> None:
        cap = _effective_atr_cap_usd(90_000.0, max_atr_usd=3000.0, max_atr_bps=400.0)
        # 325 bps tier dominates: 90000 * 0.0325 = 2925
        self.assertIsNotNone(cap)
        self.assertAlmostEqual(cap or 0.0, 2925.0, places=6)

    def test_simulate_bracket_exit_stop_first_on_same_candle(self) -> None:
        opened = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
        expiry = opened + timedelta(hours=24)
        candles = [
            {
                "timestamp": opened,
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
            }
        ]
        exit_price, reason, closed_at, mae, mfe = simulate_bracket_exit(
            candles=candles,
            opened_at=opened,
            expiry_at=expiry,
            timeframe_seconds=3600,
            side="LONG",
            entry=100.0,
            stop=95.0,
            take_profit=105.0,
        )
        self.assertEqual(reason, "stop_loss")
        self.assertEqual(exit_price, 95.0)
        self.assertEqual(closed_at, opened)
        self.assertLessEqual(mae or 0.0, 0.0)
        self.assertGreaterEqual(mfe or 0.0, 0.0)

    def test_simulate_bracket_exit_take_profit(self) -> None:
        opened = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
        expiry = opened + timedelta(hours=24)
        candles = [
            {"timestamp": opened, "open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0},
            {"timestamp": opened + timedelta(hours=1), "open": 103.0, "high": 106.0, "low": 102.0, "close": 105.0},
        ]
        exit_price, reason, closed_at, _, _ = simulate_bracket_exit(
            candles=candles,
            opened_at=opened,
            expiry_at=expiry,
            timeframe_seconds=3600,
            side="LONG",
            entry=100.0,
            stop=95.0,
            take_profit=105.0,
        )
        self.assertEqual(reason, "take_profit")
        self.assertEqual(exit_price, 105.0)
        self.assertEqual(closed_at, opened + timedelta(hours=1))

    def test_simulate_bracket_exit_expired_uses_last_close(self) -> None:
        opened = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
        expiry = opened + timedelta(hours=2)
        candles = [
            {"timestamp": opened, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"timestamp": opened + timedelta(hours=1), "open": 100.5, "high": 101.0, "low": 100.0, "close": 100.9},
            {"timestamp": opened + timedelta(hours=2), "open": 100.9, "high": 101.0, "low": 100.0, "close": 100.2},
        ]
        exit_price, reason, closed_at, _, _ = simulate_bracket_exit(
            candles=candles,
            opened_at=opened,
            expiry_at=expiry,
            timeframe_seconds=3600,
            side="SHORT",
            entry=100.0,
            stop=110.0,
            take_profit=90.0,
        )
        self.assertEqual(reason, "expired")
        self.assertEqual(exit_price, 100.9)
        self.assertEqual(closed_at, opened + timedelta(hours=1))


if __name__ == "__main__":
    unittest.main()

