import csv
import importlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace


class WatchdogCloseOldPositionsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(lambda: os.environ.pop('WATCHDOG_LOG_DIR', None))
        os.environ['WATCHDOG_LOG_DIR'] = self.temp_dir.name
        import watchdog_close_old_positions

        self.module = importlib.reload(watchdog_close_old_positions)

    def test_create_closure_record_long(self) -> None:
        opened_at = datetime(2025, 10, 4, 12, 0, 0)
        close_time = opened_at + timedelta(hours=24)
        record = self.module._create_closure_record(
            product_id='BTC-PERP-INTX',
            position_side='FUTURES_POSITION_SIDE_LONG',
            net_size=0.5,
            leverage='5',
            opened_at=opened_at,
            close_time=close_time,
            entry_price=100.0,
            exit_price=110.0,
            pnl=None,
            closure_reason='expired',
            mae=-7.5,
            mfe=15.25,
        )

        self.assertEqual(record['position_side'], 'LONG')
        self.assertEqual(record['profit_loss'], '5')
        self.assertEqual(record['profit_loss_pct'], '10')
        self.assertEqual(record['duration_seconds'], str(int(timedelta(hours=24).total_seconds())))
        self.assertEqual(record['entry_price'], '100')
        self.assertEqual(record['exit_price'], '110')
        self.assertEqual(record['mae'], '-7.5')
        self.assertEqual(record['mfe'], '15.25')

    def test_create_closure_record_short(self) -> None:
        opened_at = datetime(2025, 10, 4, 12, 0, 0)
        close_time = opened_at + timedelta(hours=6)
        record = self.module._create_closure_record(
            product_id='ETH-PERP-INTX',
            position_side='FUTURES_POSITION_SIDE_SHORT',
            net_size=-2.0,
            leverage='3',
            opened_at=opened_at,
            close_time=close_time,
            entry_price=100.0,
            exit_price=90.0,
            pnl=None,
            closure_reason='stop_loss',
            mae=-25.0,
            mfe=12.0,
        )

        self.assertEqual(record['position_side'], 'SHORT')
        self.assertEqual(record['profit_loss'], '20')
        self.assertEqual(record['profit_loss_pct'], '10')
        self.assertEqual(record['leverage'], '3')
        self.assertEqual(record['mae'], '-25')
        self.assertEqual(record['mfe'], '12')

    def test_record_position_close_appends_csv(self) -> None:
        opened_at = datetime(2025, 10, 5, 0, 0, 0)
        close_time = opened_at + timedelta(hours=1)
        record = self.module._create_closure_record(
            product_id='SOL-PERP-INTX',
            position_side='LONG',
            net_size=1.25,
            leverage='2',
            opened_at=opened_at,
            close_time=close_time,
            entry_price=50.0,
            exit_price=48.0,
            pnl=None,
            closure_reason='expired',
            mae=-8.75,
            mfe=5.0,
        )

        self.module._record_position_close(record)

        log_path = self.module._log_file_path()
        self.assertTrue(log_path.exists())

        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['product_id'], 'SOL-PERP-INTX')
        self.assertEqual(rows[0]['profit_loss'], record['profit_loss'])
        self.assertEqual(rows[0]['mae'], record['mae'])
        self.assertEqual(rows[0]['mfe'], record['mfe'])

    def test_breakeven_adjustment_for_expired_position(self) -> None:
        os.environ['WATCHDOG_BREAKEVEN_ABS'] = '0.75'
        self.addCleanup(lambda: os.environ.pop('WATCHDOG_BREAKEVEN_ABS', None))

        pnl, exit_price, reason = self.module._apply_breakeven_adjustment(
            closure_reason='expired',
            pnl=0.5,
            entry_price=100.0,
            exit_price=99.0,
            net_size=1.0,
        )

        self.assertEqual(pnl, 0.0)
        self.assertEqual(exit_price, 100.0)
        self.assertEqual(reason, 'expired_breakeven')

    def test_breakeven_adjustment_ignores_take_profit(self) -> None:
        os.environ['WATCHDOG_BREAKEVEN_ABS'] = '0.75'
        self.addCleanup(lambda: os.environ.pop('WATCHDOG_BREAKEVEN_ABS', None))

        pnl, exit_price, reason = self.module._apply_breakeven_adjustment(
            closure_reason='take_profit',
            pnl=0.5,
            entry_price=100.0,
            exit_price=101.0,
            net_size=1.0,
        )

        self.assertEqual(pnl, 0.5)
        self.assertEqual(exit_price, 101.0)
        self.assertEqual(reason, 'take_profit')

    def test_order_close_success_handles_dict(self) -> None:
        result = {'success': True, 'order_id': 'abc'}
        self.assertTrue(self.module._order_close_success(result))

    def test_order_close_success_handles_object(self) -> None:
        result = SimpleNamespace(success=True, order_id='abc')
        self.assertTrue(self.module._order_close_success(result))

    def test_order_close_success_detects_failure(self) -> None:
        result = {'success': False, 'failure_reason': 'margin'}
        self.assertFalse(self.module._order_close_success(result))


if __name__ == '__main__':
    unittest.main()
