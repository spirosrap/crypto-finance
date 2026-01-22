import unittest
from datetime import datetime, timezone

import pandas as pd

from watchdog_dashboard import filter_paper_trade_batches


class TestWatchdogDashboardPaperBatches(unittest.TestCase):
    def test_filter_keeps_all_batches(self):
        trades = pd.DataFrame(
            [
                {
                    "product_id": "BTC-PERP-INTX",
                    "position_side": "LONG",
                    "opened_at": datetime(2026, 1, 20, 0, 1, tzinfo=timezone.utc),
                    "closed_at": datetime(2026, 1, 20, 5, 0, tzinfo=timezone.utc),
                    "profit_loss": 1.0,
                },
                {
                    "product_id": "ETH-PERP-INTX",
                    "position_side": "SHORT",
                    "opened_at": datetime(2026, 1, 20, 0, 20, tzinfo=timezone.utc),
                    "closed_at": datetime(2026, 1, 20, 6, 0, tzinfo=timezone.utc),
                    "profit_loss": -0.5,
                },
            ]
        )

        filtered = filter_paper_trade_batches(trades)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(set(filtered["product_id"]), {"BTC-PERP-INTX", "ETH-PERP-INTX"})

    def test_filter_dedupes_exact_rows(self):
        base_trade = {
            "product_id": "SOL-PERP-INTX",
            "position_side": "LONG",
            "opened_at": datetime(2026, 1, 21, 1, 0, tzinfo=timezone.utc),
            "closed_at": datetime(2026, 1, 21, 2, 0, tzinfo=timezone.utc),
            "profit_loss": 0.75,
        }
        trades = pd.DataFrame([base_trade, dict(base_trade)])

        filtered = filter_paper_trade_batches(trades)
        self.assertEqual(len(filtered), 1)


if __name__ == "__main__":
    unittest.main()
