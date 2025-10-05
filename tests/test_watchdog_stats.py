import unittest
from datetime import datetime

import pandas as pd

from watchdog_stats import compute_metrics, _equity_curve_and_max_dd_pct, _derive_r_denominator


class WatchdogStatsTests(unittest.TestCase):
    def test_compute_basic_metrics(self) -> None:
        df = pd.DataFrame({
            'profit_loss': [100, -50, 0, 150, -25],
            'closure_reason': ['take_profit', 'stop_loss', 'expired_breakeven', 'take_profit', 'stop_loss'],
        })

        res = compute_metrics(df, starting_equity=1000.0)

        self.assertEqual(res.total_trades, 5)
        self.assertEqual(res.wins, 2)
        self.assertEqual(res.losses, 2)
        self.assertEqual(res.breakevens, 1)
        self.assertAlmostEqual(res.expectancy_currency, 35.0)
        self.assertAlmostEqual(res.win_rate_pct, 40.0)
        self.assertIsNotNone(res.average_r)

    def test_average_r_median_basis(self) -> None:
        df = pd.DataFrame({
            'profit_loss': [100, -20, -40, 60],
            'closure_reason': ['take_profit', 'stop_loss', 'stop_loss', 'take_profit'],
        })
        res = compute_metrics(df, r_basis='median_loss')
        self.assertAlmostEqual(res.average_r or 0.0, (100 - 20 - 40 + 60) / 4 / 30.0, places=6)

    def test_average_r_fixed_basis(self) -> None:
        df = pd.DataFrame({
            'profit_loss': [50, -25, 75],
            'closure_reason': ['take_profit', 'stop_loss', 'take_profit'],
        })
        res = compute_metrics(df, r_basis='fixed', fixed_risk=25.0)
        self.assertAlmostEqual(res.average_r or 0.0, ((50 - 25 + 75) / 3) / 25.0, places=6)

    def test_equity_curve_drawdown(self) -> None:
        _, dd = _equity_curve_and_max_dd_pct(pd.Series([100, -200, 50]).to_numpy(), 1000)
        self.assertLessEqual(dd, 0.0)

    def test_r_denominator_handles_no_losses(self) -> None:
        denom = _derive_r_denominator(pd.Series([], dtype=float).to_numpy(), 'avg_loss', None)
        self.assertIsNone(denom)


if __name__ == '__main__':
    unittest.main()
