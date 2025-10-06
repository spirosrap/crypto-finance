import csv
import importlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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

    def test_extract_symbol_and_size_normalizes_short_sign(self) -> None:
        pos = {
            'product_id': 'NEAR-PERP-INTX',
            'net_size': '167.4',
            'position_side': 'SHORT',
            'leverage': '50',
        }
        symbol, size, side, lev = self.module._extract_symbol_and_size(pos)
        self.assertEqual(symbol, 'NEAR-PERP-INTX')
        self.assertLess(size, 0.0)
        self.assertEqual(side, 'SHORT')
        self.assertEqual(lev, '50')

    def test_extract_symbol_and_size_normalizes_long_sign(self) -> None:
        pos = SimpleNamespace(
            product_id='ENA-PERP-INTX',
            net_size=-250.0,
            position_side='FUTURES_POSITION_SIDE_LONG',
            leverage='30',
        )
        symbol, size, side, lev = self.module._extract_symbol_and_size(pos)
        self.assertEqual(symbol, 'ENA-PERP-INTX')
        self.assertGreater(size, 0.0)
        self.assertEqual(side, 'FUTURES_POSITION_SIDE_LONG')
        self.assertEqual(lev, '30')

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

    def test_compute_mae_mfe_from_history_long(self) -> None:
        opened_at = datetime(2025, 10, 4, 0, 0, 0, tzinfo=timezone.utc)
        close_time = opened_at + timedelta(hours=6)

        candles = [
            {'low': '95', 'high': '105'},
            {'low': 98, 'high': 120},
        ]

        cb = SimpleNamespace(
            historical_data=SimpleNamespace(
                get_historical_data=lambda *args, **kwargs: candles
            )
        )

        mae, mfe = self.module.compute_mae_mfe_from_history(
            cb=cb,
            product_id='BTC-PERP-INTX',
            net_size=2.0,
            entry_price=100.0,
            open_time=opened_at,
            close_time=close_time,
        )

        self.assertAlmostEqual(mae, -10.0)
        self.assertAlmostEqual(mfe, 40.0)

    def test_compute_mae_mfe_from_history_short(self) -> None:
        opened_at = datetime(2025, 10, 4, 0, 0, 0)
        close_time = opened_at + timedelta(hours=2)

        candles = [
            {'low': 45, 'high': 52},
            {'low': 47, 'high': 55},
        ]

        cb = SimpleNamespace(
            historical_data=SimpleNamespace(
                get_historical_data=lambda *args, **kwargs: candles
            )
        )

        mae, mfe = self.module.compute_mae_mfe_from_history(
            cb=cb,
            product_id='ETH-PERP-INTX',
            net_size=-3.0,
            entry_price=50.0,
            open_time=opened_at,
            close_time=close_time,
        )

        self.assertAlmostEqual(mae, -15.0)
        self.assertAlmostEqual(mfe, 15.0)

    def test_close_position_returns_fill_price_from_result(self) -> None:
        def cancel_all_orders(**kwargs):
            return None

        def create_order(**kwargs):
            return {
                'success': True,
                'order_id': 'abc123',
                'average_filled_price': '3.5',
            }

        client = SimpleNamespace(
            create_order=create_order,
            get_order=lambda **kwargs: {},
            list_fills=lambda **kwargs: {'fills': []},
        )

        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        closed, fill_price, order_id = self.module._close_position(
            cb,
            product_id='BTC-PERP-INTX',
            net_size=-1.0,
            position_side='FUTURES_POSITION_SIDE_SHORT',
            leverage='5',
        )

        self.assertTrue(closed)
        self.assertAlmostEqual(fill_price or 0.0, 3.5)
        self.assertEqual(order_id, 'abc123')

    def test_close_position_fetches_fill_price_from_fills(self) -> None:
        recorded_kwargs: dict = {}

        def cancel_all_orders(**kwargs):
            return None

        def create_order(**kwargs):
            return {
                'success': True,
                'order_id': 'fill123',
            }

        def list_fills(**kwargs):
            recorded_kwargs.update(kwargs)
            return {
                'fills': [
                    {
                        'order_id': 'fill123',
                        'price': '12.34',
                    }
                ]
            }

        client = SimpleNamespace(
            create_order=create_order,
            get_order=lambda **kwargs: {},
            list_fills=list_fills,
        )

        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        closed, fill_price, order_id = self.module._close_position(
            cb,
            product_id='ADA-PERP-INTX',
            net_size=2.0,
            position_side='FUTURES_POSITION_SIDE_LONG',
            leverage='3',
        )

        self.assertTrue(closed)
        self.assertAlmostEqual(fill_price or 0.0, 12.34)
        self.assertEqual(order_id, 'fill123')
        self.assertIn('order_id', recorded_kwargs)

    def test_compute_mae_mfe_handles_empty_candles(self) -> None:
        cb = SimpleNamespace(
            historical_data=SimpleNamespace(
                get_historical_data=lambda *args, **kwargs: []
            )
        )

        mae, mfe = self.module.compute_mae_mfe_from_history(
            cb=cb,
            product_id='SOL-PERP-INTX',
            net_size=1.0,
            entry_price=25.0,
            open_time=datetime(2025, 10, 4, 0, 0, 0),
            close_time=datetime(2025, 10, 4, 1, 0, 0),
        )

        self.assertIsNone(mae)
        self.assertIsNone(mfe)

    def test_backfill_last_entries_updates_row(self) -> None:
        log_path = self.module._ensure_log_file()
        with log_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.module.LOG_HEADERS)
            writer.writeheader()
            writer.writerow({
                'closed_at': '2025-10-06T18:57:01Z',
                'product_id': 'NEAR-PERP-INTX',
                'position_side': 'SHORT',
                'net_size': ' -167.4 ',
                'leverage': '50',
                'opened_at': '2025-10-05T18:52:44Z',
                'closure_reason': 'expired_breakeven',
                'entry_price': '3.0791',
                'exit_price': '3.0791',
                'profit_loss': '0',
                'profit_loss_pct': '0',
                'mae': '',
                'mfe': '',
                'duration_seconds': '86656',
            })

        fills = {
            'fills': [
                {
                    'product_id': 'NEAR-PERP-INTX',
                    'price': '3.05',
                    'filled_size': '167.4',
                    'trade_time': '2025-10-06T18:57:02Z',
                }
            ]
        }

        client = SimpleNamespace(list_fills=lambda **kwargs: fills)
        cb = SimpleNamespace(client=client)

        self.module._backfill_last_entries(cb, 1)

        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['product_id'], 'NEAR-PERP-INTX')
        self.assertEqual(row['closure_reason'], 'expired')
        self.assertNotEqual(row['profit_loss'], '0')
        self.assertNotEqual(row['exit_price'], '3.0791')


if __name__ == '__main__':
    unittest.main()
