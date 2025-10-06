import os
import unittest
from datetime import datetime, timezone, timedelta

from watchdog_log_tp_sl import (
    Fill,
    _cycle_to_record,
    _is_new_cycle,
    _process_product_fills,
)


UTC = timezone.utc


class WatchdogLogTpSlTests(unittest.TestCase):
    def test_long_cycle_detection(self) -> None:
        fills = [
            Fill('BTC-PERP-INTX', 'BUY', 1.0, 100.0, 0.0, datetime(2025, 10, 5, 0, 0, tzinfo=UTC), '1'),
            Fill('BTC-PERP-INTX', 'BUY', 1.0, 105.0, 0.0, datetime(2025, 10, 5, 0, 5, tzinfo=UTC), '2'),
            Fill('BTC-PERP-INTX', 'SELL', 2.0, 110.0, 0.0, datetime(2025, 10, 5, 0, 10, tzinfo=UTC), '3'),
        ]
        cycles = _process_product_fills(fills)
        self.assertEqual(len(cycles), 1)
        cycle = cycles[0]
        self.assertEqual(cycle.side, 'LONG')
        self.assertAlmostEqual(cycle.entry_qty, 2.0)
        self.assertAlmostEqual(cycle.entry_value, 205.0)
        self.assertAlmostEqual(cycle.exit_value, 220.0)
        self.assertAlmostEqual(cycle.realized_pnl, 15.0)

    def test_short_cycle_detection(self) -> None:
        fills = [
            Fill('ETH-PERP-INTX', 'SELL', 1.5, 200.0, 0.0, datetime(2025, 10, 5, 1, 0, tzinfo=UTC), '10'),
            Fill('ETH-PERP-INTX', 'BUY', 1.5, 180.0, 0.0, datetime(2025, 10, 5, 1, 30, tzinfo=UTC), '11'),
        ]
        cycles = _process_product_fills(fills)
        self.assertEqual(len(cycles), 1)
        cycle = cycles[0]
        self.assertEqual(cycle.side, 'SHORT')
        self.assertAlmostEqual(cycle.entry_qty, 1.5)
        self.assertAlmostEqual(cycle.realized_pnl, 30.0)

    def test_cycle_to_record_breakeven(self) -> None:
        cycle = _process_product_fills([
            Fill('SOL-PERP-INTX', 'SELL', 1.0, 50.0, 0.0, datetime(2025, 10, 5, 2, 0, tzinfo=UTC), '21'),
            Fill('SOL-PERP-INTX', 'BUY', 1.0, 45.0, 0.0, datetime(2025, 10, 5, 2, 15, tzinfo=UTC), '22'),
        ])[0]

        os.environ['WATCHDOG_BREAKEVEN_ABS'] = '1.0'
        self.addCleanup(lambda: os.environ.pop('WATCHDOG_BREAKEVEN_ABS', None))

        record = _cycle_to_record(cycle, 1.0)
        self.assertEqual(record['closure_reason'], 'take_profit')

        # Adjust threshold higher to force breakeven classification
        record_breakeven = _cycle_to_record(cycle, 40.0)
        self.assertEqual(record_breakeven['closure_reason'], 'expired_breakeven')

    def test_cycle_to_record_injects_mae_mfe(self) -> None:
        cycle = _process_product_fills([
            Fill('DOGE-PERP-INTX', 'BUY', 1000.0, 0.1, 0.0, datetime(2025, 10, 5, 4, 0, tzinfo=UTC), '51'),
            Fill('DOGE-PERP-INTX', 'SELL', 1000.0, 0.105, 0.0, datetime(2025, 10, 5, 4, 30, tzinfo=UTC), '52'),
        ])[0]

        captured: Dict[str, Any] = {}

        def fake_fetcher(**kwargs):
            captured.update(kwargs)
            return -12.34, 45.67

        record = _cycle_to_record(cycle, 1.0, mae_mfe_fetcher=fake_fetcher)
        self.assertEqual(captured['product_id'], 'DOGE-PERP-INTX')
        self.assertAlmostEqual(captured['net_size'], 1000.0)
        self.assertAlmostEqual(captured['entry_price'], 0.1)
        self.assertEqual(record['mae'], '-12.34')
        self.assertEqual(record['mfe'], '45.67')

    def test_checkpoint_filter(self) -> None:
        cycle = _process_product_fills([
            Fill('ADA-PERP-INTX', 'BUY', 1.0, 0.3, 0.0, datetime(2025, 10, 5, 3, 0, tzinfo=UTC), '31'),
            Fill('ADA-PERP-INTX', 'SELL', 1.0, 0.33, 0.0, datetime(2025, 10, 5, 3, 15, tzinfo=UTC), '32'),
        ])[0]

        checkpoint = {
            'last_time': (cycle.end_time - timedelta(minutes=1)).isoformat(),
            'last_order_id': '1',
        }
        self.assertTrue(_is_new_cycle(cycle, checkpoint, bootstrap_existing=False))

        checkpoint_same = {
            'last_time': cycle.end_time.isoformat(),
            'last_order_id': '42',
        }
        self.assertFalse(_is_new_cycle(cycle, checkpoint_same, bootstrap_existing=False))


if __name__ == '__main__':
    unittest.main()
