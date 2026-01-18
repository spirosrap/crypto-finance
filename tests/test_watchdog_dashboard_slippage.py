import unittest
from datetime import datetime, timezone

import pandas as pd

from watchdog_dashboard import (
    _compute_exit_slippage,
    _extract_target_price,
    build_exit_slippage_table,
)


class TestWatchdogDashboardSlippage(unittest.TestCase):
    def test_extract_target_price_bracket(self):
        order = {
            "order_configuration": {
                "trigger_bracket_gtd": {
                    "limit_price": "100",
                    "stop_trigger_price": "90",
                }
            }
        }
        self.assertEqual(_extract_target_price(order, "take_profit"), 100.0)
        self.assertEqual(_extract_target_price(order, "partial_take"), 100.0)
        self.assertEqual(_extract_target_price(order, "stop_loss"), 90.0)

    def test_extract_target_price_stop_limit(self):
        order = {
            "order_configuration": {
                "stop_limit_stop_limit_gtd": {
                    "stop_price": "95",
                    "limit_price": "94",
                }
            }
        }
        self.assertEqual(_extract_target_price(order, "stop_loss"), 95.0)
        self.assertEqual(_extract_target_price(order, "take_profit"), 94.0)

    def test_compute_exit_slippage(self):
        slippage_price, slippage_bps = _compute_exit_slippage(95.0, 100.0, "LONG")
        self.assertAlmostEqual(slippage_price, 5.0)
        self.assertAlmostEqual(slippage_bps, 500.0)

        slippage_price, slippage_bps = _compute_exit_slippage(105.0, 100.0, "SHORT")
        self.assertAlmostEqual(slippage_price, 5.0)
        self.assertAlmostEqual(slippage_bps, 500.0)

    def test_build_exit_slippage_table(self):
        now = datetime.now(timezone.utc)
        trades = pd.DataFrame(
            [
                {
                    "order_id": "order-a",
                    "exit_price": 95.0,
                    "net_size": 2.0,
                    "position_side": "LONG",
                    "closure_reason": "take_profit",
                    "product_id": "TEST-PERP-INTX",
                    "closed_at": now,
                },
                {
                    "order_id": "order-b",
                    "exit_price": 105.0,
                    "net_size": -1.0,
                    "position_side": "SHORT",
                    "closure_reason": "take_profit",
                    "product_id": "TEST-PERP-INTX",
                    "closed_at": now,
                },
            ]
        )

        def fetch_order(order_id: str):
            if order_id == "order-a":
                return {
                    "order_configuration": {
                        "trigger_bracket_gtd": {
                            "limit_price": "100",
                            "stop_trigger_price": "90",
                        }
                    }
                }
            return {
                "order_configuration": {
                    "trigger_bracket_gtd": {
                        "limit_price": "100",
                        "stop_trigger_price": "110",
                    }
                }
            }

        result = build_exit_slippage_table(trades, fetch_order, max_orders=0)
        self.assertEqual(len(result), 2)
        self.assertIn("slippage_usd", result.columns)
        slippage_values = sorted(result["slippage_usd"].tolist())
        self.assertEqual(slippage_values, [5.0, 10.0])


if __name__ == "__main__":
    unittest.main()
