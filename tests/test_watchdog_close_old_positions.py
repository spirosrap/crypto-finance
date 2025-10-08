import csv
import importlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict


UTC = timezone.utc


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
        cb = SimpleNamespace(
            client=client,
            historical_data=SimpleNamespace(get_historical_data=lambda *args, **kwargs: []),
        )

        original_compute = self.module.compute_mae_mfe_from_history
        self.module.compute_mae_mfe_from_history = lambda **kwargs: (-2.5, 4.5)
        self.addCleanup(lambda: setattr(self.module, 'compute_mae_mfe_from_history', original_compute))

        self.module._backfill_last_entries(cb, 1)

        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['product_id'], 'NEAR-PERP-INTX')
        self.assertEqual(row['closure_reason'], 'expired_breakeven')
        self.assertNotEqual(row['profit_loss'], '0')
        self.assertNotEqual(row['exit_price'], '3.0791')

    def test_long_cycle_detection(self) -> None:
        fills = [
            self.module.Fill('BTC-PERP-INTX', 'BUY', 1.0, 100.0, 0.0, datetime(2025, 10, 5, 0, 0, tzinfo=UTC), '1'),
            self.module.Fill('BTC-PERP-INTX', 'BUY', 1.0, 105.0, 0.0, datetime(2025, 10, 5, 0, 5, tzinfo=UTC), '2'),
            self.module.Fill('BTC-PERP-INTX', 'SELL', 2.0, 110.0, 0.0, datetime(2025, 10, 5, 0, 10, tzinfo=UTC), '3'),
        ]
        cycles = self.module._process_product_fills(fills)
        self.assertEqual(len(cycles), 1)
        cycle = cycles[0]
        self.assertEqual(cycle.side, 'LONG')
        self.assertAlmostEqual(cycle.entry_qty, 2.0)
        self.assertAlmostEqual(cycle.entry_value, 205.0)
        self.assertAlmostEqual(cycle.exit_value, 220.0)
        self.assertAlmostEqual(cycle.realized_pnl, 15.0)

    def test_short_cycle_detection(self) -> None:
        fills = [
            self.module.Fill('ETH-PERP-INTX', 'SELL', 1.5, 200.0, 0.0, datetime(2025, 10, 5, 1, 0, tzinfo=UTC), '10'),
            self.module.Fill('ETH-PERP-INTX', 'BUY', 1.5, 180.0, 0.0, datetime(2025, 10, 5, 1, 30, tzinfo=UTC), '11'),
        ]
        cycles = self.module._process_product_fills(fills)
        self.assertEqual(len(cycles), 1)
        cycle = cycles[0]
        self.assertEqual(cycle.side, 'SHORT')
        self.assertAlmostEqual(cycle.entry_qty, 1.5)
        self.assertAlmostEqual(cycle.realized_pnl, 30.0)

    def test_cycle_to_record_breakeven(self) -> None:
        cycle = self.module._process_product_fills([
            self.module.Fill('SOL-PERP-INTX', 'SELL', 1.0, 50.0, 0.0, datetime(2025, 10, 5, 2, 0, tzinfo=UTC), '21'),
            self.module.Fill('SOL-PERP-INTX', 'BUY', 1.0, 45.0, 0.0, datetime(2025, 10, 5, 2, 15, tzinfo=UTC), '22'),
        ])[0]

        os.environ['WATCHDOG_BREAKEVEN_ABS'] = '1.0'
        self.addCleanup(lambda: os.environ.pop('WATCHDOG_BREAKEVEN_ABS', None))

        record = self.module._cycle_to_record(cycle, 1.0)
        self.assertEqual(record['closure_reason'], 'take_profit')

        record_breakeven = self.module._cycle_to_record(cycle, 40.0)
        self.assertEqual(record_breakeven['closure_reason'], 'expired_breakeven')

    def test_cycle_to_record_injects_mae_mfe(self) -> None:
        cycle = self.module._process_product_fills([
            self.module.Fill('DOGE-PERP-INTX', 'BUY', 1000.0, 0.1, 0.0, datetime(2025, 10, 5, 4, 0, tzinfo=UTC), '51'),
            self.module.Fill('DOGE-PERP-INTX', 'SELL', 1000.0, 0.105, 0.0, datetime(2025, 10, 5, 4, 30, tzinfo=UTC), '52'),
        ])[0]

        captured: Dict[str, Any] = {}

        def fake_fetcher(**kwargs: Any):
            captured.update(kwargs)
            return -12.34, 45.67

        record = self.module._cycle_to_record(cycle, 1.0, mae_mfe_fetcher=fake_fetcher)
        self.assertEqual(captured['product_id'], 'DOGE-PERP-INTX')
        self.assertAlmostEqual(captured['net_size'], 1000.0)
        self.assertAlmostEqual(captured['entry_price'], 0.1)
        self.assertEqual(record['mae'], '-12.34')
        self.assertEqual(record['mfe'], '45.67')

    def test_extract_avg_filled_price_handles_nested_order(self) -> None:
        price = self.module._extract_avg_filled_price({
            'order': {
                'average_filled_price': '5.325',
                'filled_value': '484.575',
                'filled_size': '91',
            }
        })
        self.assertAlmostEqual(price or 0.0, 5.325)

    def test_checkpoint_filter(self) -> None:
        cycle = self.module._process_product_fills([
            self.module.Fill('ADA-PERP-INTX', 'BUY', 1.0, 0.3, 0.0, datetime(2025, 10, 5, 3, 0, tzinfo=UTC), '31'),
            self.module.Fill('ADA-PERP-INTX', 'SELL', 1.0, 0.33, 0.0, datetime(2025, 10, 5, 3, 15, tzinfo=UTC), '32'),
        ])[0]

        checkpoint = {
            'last_time': (cycle.end_time - timedelta(minutes=1)).isoformat(),
            'last_order_id': '1',
        }
        self.assertTrue(self.module._is_new_cycle(cycle, checkpoint, bootstrap_existing=False))

        checkpoint_same = {
            'last_time': cycle.end_time.isoformat(),
            'last_order_id': '42',
        }
        self.assertFalse(self.module._is_new_cycle(cycle, checkpoint_same, bootstrap_existing=False))

    def test_cycle_details_backfill_updates_entry_exit(self) -> None:
        log_path = self.module._log_file_path()
        with log_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.module.LOG_HEADERS)
            writer.writeheader()
            writer.writerow({
                'closed_at': '2025-10-02T00:00:00Z',
                'product_id': 'APT-PERP-INTX',
                'position_side': 'SHORT',
                'net_size': '-10',
                'leverage': '50',
                'opened_at': '2025-10-01T00:00:00Z',
                'closure_reason': 'expired',
                'entry_price': '5.10',
                'exit_price': '5.10',
                'profit_loss': '0',
                'profit_loss_pct': '0',
                'mae': '',
                'mfe': '',
                'duration_seconds': '86400',
            })

        fills = {
            'fills': [
                {
                    'product_id': 'APT-PERP-INTX',
                    'side': 'SELL',
                    'size': '10',
                    'price': '5.49',
                    'fee': '0',
                    'order_id': 'open1',
                    'trade_time': '2025-10-01T00:00:05Z',
                },
                {
                    'product_id': 'APT-PERP-INTX',
                    'side': 'BUY',
                    'size': '10',
                    'price': '5.35',
                    'fee': '0',
                    'order_id': 'close1',
                    'trade_time': '2025-10-02T00:00:03Z',
                },
            ]
        }

        client = SimpleNamespace(list_fills=lambda **kwargs: fills)
        cb = SimpleNamespace(
            client=client,
            historical_data=SimpleNamespace(get_historical_data=lambda *args, **kwargs: []),
        )

        original_compute = self.module.compute_mae_mfe_from_history
        self.module.compute_mae_mfe_from_history = lambda **kwargs: (-2.5, 4.5)
        self.addCleanup(lambda: setattr(self.module, 'compute_mae_mfe_from_history', original_compute))

        self.module._backfill_last_entries(cb, 1)

        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['entry_price'], '5.49')
        self.assertEqual(row['exit_price'], '5.35')
        self.assertEqual(row['profit_loss'], '1.4')
        self.assertEqual(row['closure_reason'], 'take_profit')
        self.assertEqual(row['mae'], '-2.5')
        self.assertEqual(row['mfe'], '4.5')


if __name__ == '__main__':
    unittest.main()
