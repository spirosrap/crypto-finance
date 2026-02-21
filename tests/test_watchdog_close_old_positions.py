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

    def test_infer_open_time_from_orders_short_alias_uses_sell_entries(self) -> None:
        opened_at = datetime(2025, 10, 4, 12, 0, 0, tzinfo=UTC)
        partial_close_at = opened_at + timedelta(minutes=5)
        orders = [
            {
                'side': 'SELL',
                'filled_size': '1.0',
                'created_time': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
            },
            {
                'side': 'BUY',
                'filled_size': '0.2',
                'created_time': partial_close_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
            },
        ]

        original_orders = self.module._orders_for_product
        self.module._orders_for_product = lambda *args, **kwargs: orders
        self.addCleanup(lambda: setattr(self.module, '_orders_for_product', original_orders))

        inferred = self.module._infer_open_time_from_orders(
            cb=SimpleNamespace(),
            portfolio_uuid='uuid',
            product_id='BTC-PERP-INTX',
            expected_net=-1.0,
            position_side='SHORT',
        )

        self.assertEqual(inferred, opened_at)

    def test_classify_partial_reason_prefers_realized_pnl(self) -> None:
        reason = self.module._classify_partial_reason(
            'SHORT',
            100.0,
            95.0,
            realized_pnl=-0.25,
        )
        self.assertEqual(reason, 'partial_sl')

        reason = self.module._classify_partial_reason(
            'LONG',
            100.0,
            105.0,
            realized_pnl=-0.1,
        )
        self.assertEqual(reason, 'partial_sl')

        reason = self.module._classify_partial_reason(
            'SHORT',
            100.0,
            105.0,
            realized_pnl=0.4,
        )
        self.assertEqual(reason, 'partial_tp')

    def test_classify_partial_reason_falls_back_to_prices(self) -> None:
        reason = self.module._classify_partial_reason(
            'SHORT',
            100.0,
            95.0,
            realized_pnl=0.0,
        )
        self.assertEqual(reason, 'partial_tp')

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

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                return create_order()

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, fill_price, order_id, close_path = self.module._close_position(
            cb,
            product_id='BTC-PERP-INTX',
            net_size=-1.0,
            position_side='FUTURES_POSITION_SIDE_SHORT',
            leverage='5',
        )

        self.assertTrue(closed)
        self.assertAlmostEqual(fill_price or 0.0, 3.5)
        self.assertEqual(order_id, 'abc123')
        self.assertEqual(close_path, 'ccxt_close_position')

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

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                return create_order()

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, fill_price, order_id, close_path = self.module._close_position(
            cb,
            product_id='ADA-PERP-INTX',
            net_size=2.0,
            position_side='FUTURES_POSITION_SIDE_LONG',
            leverage='3',
        )

        self.assertTrue(closed)
        self.assertAlmostEqual(fill_price or 0.0, 12.34)
        self.assertEqual(order_id, 'fill123')
        self.assertEqual(close_path, 'ccxt_close_position')
        self.assertIn('order_id', recorded_kwargs)

    def test_close_position_short_alias_submits_buy(self) -> None:
        submitted_sides = []

        def cancel_all_orders(**kwargs):
            return None

        client = SimpleNamespace(
            get_order=lambda **kwargs: {},
            list_fills=lambda **kwargs: {'fills': []},
        )
        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                submitted_sides.append(args[2] if len(args) > 2 else kwargs.get('side'))
                return {
                    'success': True,
                    'order_id': 'short-close',
                    'average_filled_price': '10.0',
                }

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, _, order_id, close_path = self.module._close_position(
            cb,
            product_id='SOL-PERP-INTX',
            net_size=-2.0,
            position_side='SHORT',
            leverage='2',
        )

        self.assertTrue(closed)
        self.assertEqual(order_id, 'short-close')
        self.assertEqual(close_path, 'ccxt_close_position')
        self.assertEqual(submitted_sides, ['buy'])

    def test_close_position_client_order_id_error_falls_back_to_rest(self) -> None:
        rest_kwargs: Dict[str, Any] = {}

        def cancel_all_orders(**kwargs):
            return None

        def rest_create_order(**kwargs):
            rest_kwargs.update(kwargs)
            return {
                'success': True,
                'order_id': 'rest-close-1',
            }

        client = SimpleNamespace(
            create_order=rest_create_order,
            get_order=lambda **kwargs: {},
            list_fills=lambda **kwargs: {'fills': []},
        )
        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                raise Exception("coinbaseadvanced closePosition() requires a clientOrderId parameter")

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, fill_price, order_id, close_path = self.module._close_position(
            cb,
            product_id='LTC-PERP-INTX',
            net_size=-1.0,
            position_side='SHORT',
            leverage='5',
        )

        self.assertTrue(closed)
        self.assertIsNone(fill_price)
        self.assertEqual(order_id, 'rest-close-1')
        self.assertEqual(close_path, 'rest_fallback_close')
        self.assertEqual(rest_kwargs.get('product_id'), 'LTC-PERP-INTX')
        self.assertEqual(rest_kwargs.get('side'), 'BUY')
        self.assertTrue(rest_kwargs.get('client_order_id'))

    def test_close_position_margin_mode_error_falls_back_to_rest(self) -> None:
        rest_kwargs: Dict[str, Any] = {}

        def cancel_all_orders(**kwargs):
            return None

        def rest_create_order(**kwargs):
            rest_kwargs.update(kwargs)
            return {
                'success': True,
                'order_id': 'rest-close-2',
            }

        client = SimpleNamespace(
            create_order=rest_create_order,
            get_order=lambda **kwargs: {},
            list_fills=lambda **kwargs: {'fills': []},
        )
        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                raise Exception(
                    'coinbaseadvanced {"error":"unknown","error_details":"proto: (line 1:85): unknown field \\"marginMode\\""}'
                )

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, fill_price, order_id, close_path = self.module._close_position(
            cb,
            product_id='SEI-PERP-INTX',
            net_size=10.0,
            position_side='LONG',
            leverage='5',
        )

        self.assertTrue(closed)
        self.assertIsNone(fill_price)
        self.assertEqual(order_id, 'rest-close-2')
        self.assertEqual(close_path, 'rest_fallback_close')
        self.assertEqual(rest_kwargs.get('product_id'), 'SEI-PERP-INTX')
        self.assertEqual(rest_kwargs.get('side'), 'SELL')

    def test_close_position_amount_field_error_falls_back_to_rest(self) -> None:
        rest_kwargs: Dict[str, Any] = {}

        def cancel_all_orders(**kwargs):
            return None

        def rest_create_order(**kwargs):
            rest_kwargs.update(kwargs)
            return {
                'success': True,
                'order_id': 'rest-close-amount',
            }

        client = SimpleNamespace(
            create_order=rest_create_order,
            get_order=lambda **kwargs: {},
            list_fills=lambda **kwargs: {'fills': []},
        )
        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                raise Exception(
                    'coinbaseadvanced {"error":"unknown","error_details":"proto: (line 1:85): unknown field \\"amount\\""}'
                )

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, fill_price, order_id, close_path = self.module._close_position(
            cb,
            product_id='XTZ-PERP-INTX',
            net_size=25.0,
            position_side='LONG',
            leverage='5',
        )

        self.assertTrue(closed)
        self.assertIsNone(fill_price)
        self.assertEqual(order_id, 'rest-close-amount')
        self.assertEqual(close_path, 'rest_fallback_close')
        self.assertEqual(rest_kwargs.get('product_id'), 'XTZ-PERP-INTX')
        self.assertEqual(rest_kwargs.get('side'), 'SELL')

    def test_close_position_rest_object_response_treated_as_success(self) -> None:
        rest_kwargs: Dict[str, Any] = {}

        def cancel_all_orders(**kwargs):
            return None

        def rest_create_order(**kwargs):
            rest_kwargs.update(kwargs)
            return SimpleNamespace(success=True, order_id='rest-close-object')

        client = SimpleNamespace(
            create_order=rest_create_order,
            get_order=lambda **kwargs: {},
            list_fills=lambda **kwargs: {'fills': []},
        )
        cb = SimpleNamespace(cancel_all_orders=cancel_all_orders, client=client)

        class DummyExchange:
            def cancel_all_orders(self, symbol=None):
                return None

            def create_order(self, *args, **kwargs):
                raise Exception("coinbaseadvanced closePosition() requires a clientOrderId parameter")

        original_exchange = self.module._ensure_ccxt_exchange
        self.module._ensure_ccxt_exchange = lambda: DummyExchange()
        self.addCleanup(lambda: setattr(self.module, '_ensure_ccxt_exchange', original_exchange))

        closed, fill_price, order_id, close_path = self.module._close_position(
            cb,
            product_id='BTC-PERP-INTX',
            net_size=1.0,
            position_side='LONG',
            leverage='5',
        )

        self.assertTrue(closed)
        self.assertIsNone(fill_price)
        self.assertEqual(order_id, 'rest-close-object')
        self.assertEqual(close_path, 'rest_fallback_close')
        self.assertEqual(rest_kwargs.get('product_id'), 'BTC-PERP-INTX')
        self.assertEqual(rest_kwargs.get('side'), 'SELL')

    def test_dust_notional_helper(self) -> None:
        notional = self.module._dust_notional_usd(
            net_size=2.0,
            entry_price=1.5,
            mark_price=None,
            threshold=5.0,
        )
        self.assertAlmostEqual(notional or 0.0, 3.0)

        notional = self.module._dust_notional_usd(
            net_size=2.0,
            entry_price=3.0,
            mark_price=3.0,
            threshold=5.0,
        )
        self.assertIsNone(notional)

    def test_run_once_closes_dust_positions(self) -> None:
        pos = {
            'product_id': 'SUI-PERP-INTX',
            'net_size': '1.0',
            'position_side': 'LONG',
            'leverage': '5',
            'mark_price': '2.5',
        }

        class DummyClient:
            def get_portfolios(self):
                return {'portfolios': [{'type': 'INTX', 'uuid': 'uuid'}]}

            def get_portfolio_breakdown(self, portfolio_uuid=None):
                return {'breakdown': {'perp_positions': [pos]}}

        class DummyCB:
            def __init__(self, *args, **kwargs):
                self.client = DummyClient()

        closed = {}

        def dummy_close(*args, **kwargs):
            closed['called'] = True
            return True, 2.6, 'dust-order', 'ccxt_close_position'

        original_cb = self.module.CoinbaseService
        original_close = self.module._close_position
        self.module.CoinbaseService = DummyCB
        self.module._close_position = dummy_close
        self.addCleanup(lambda: setattr(self.module, 'CoinbaseService', original_cb))
        self.addCleanup(lambda: setattr(self.module, '_close_position', original_close))

        self.module.run_once(
            max_age_hours=24,
            product_filter=None,
            log_closures=True,
            recent_order_grace_minutes=0,
            dust_notional_usd=5.0,
        )

        self.assertTrue(closed.get('called'))
        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]['closure_reason'], 'dust')

    def test_run_once_closes_dust_positions_when_logging_disabled(self) -> None:
        pos = {
            'product_id': 'SUI-PERP-INTX',
            'net_size': '1.0',
            'position_side': 'LONG',
            'leverage': '5',
            'mark_price': '2.5',
        }

        class DummyClient:
            def get_portfolios(self):
                return {'portfolios': [{'type': 'INTX', 'uuid': 'uuid'}]}

            def get_portfolio_breakdown(self, portfolio_uuid=None):
                return {'breakdown': {'perp_positions': [pos]}}

        class DummyCB:
            def __init__(self, *args, **kwargs):
                self.client = DummyClient()

        closed = {}

        def dummy_close(*args, **kwargs):
            closed['called'] = True
            return True, 2.6, 'dust-order', 'ccxt_close_position'

        original_cb = self.module.CoinbaseService
        original_close = self.module._close_position
        self.module.CoinbaseService = DummyCB
        self.module._close_position = dummy_close
        self.addCleanup(lambda: setattr(self.module, 'CoinbaseService', original_cb))
        self.addCleanup(lambda: setattr(self.module, '_close_position', original_close))

        self.module.run_once(
            max_age_hours=24,
            product_filter=None,
            log_closures=False,
            recent_order_grace_minutes=0,
            dust_notional_usd=5.0,
        )

        self.assertTrue(closed.get('called'))
        self.assertFalse(self.module._log_file_path().exists())

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

    def test_partial_fill_long_detection(self) -> None:
        fills = [
            self.module.Fill('BTC-PERP-INTX', 'BUY', 2.0, 100.0, 0.0, datetime(2025, 10, 5, 0, 0, tzinfo=UTC), '1'),
            self.module.Fill('BTC-PERP-INTX', 'SELL', 1.0, 105.0, 0.0, datetime(2025, 10, 5, 0, 5, tzinfo=UTC), '2'),
            self.module.Fill('BTC-PERP-INTX', 'SELL', 1.0, 110.0, 0.0, datetime(2025, 10, 5, 0, 10, tzinfo=UTC), '3'),
        ]
        cycles, partials = self.module._process_product_fills_with_partials(fills)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(partials), 1)
        event = partials[0]
        self.assertEqual(event.side, 'LONG')
        self.assertAlmostEqual(event.qty, 1.0)
        self.assertAlmostEqual(event.entry_price, 100.0)
        self.assertAlmostEqual(event.exit_price, 105.0)
        self.assertAlmostEqual(event.realized_pnl, 5.0)

    def test_partial_fill_short_detection(self) -> None:
        fills = [
            self.module.Fill('ETH-PERP-INTX', 'SELL', 2.0, 200.0, 0.0, datetime(2025, 10, 5, 1, 0, tzinfo=UTC), '10'),
            self.module.Fill('ETH-PERP-INTX', 'BUY', 1.0, 195.0, 0.0, datetime(2025, 10, 5, 1, 5, tzinfo=UTC), '11'),
            self.module.Fill('ETH-PERP-INTX', 'BUY', 1.0, 190.0, 0.0, datetime(2025, 10, 5, 1, 10, tzinfo=UTC), '12'),
        ]
        cycles, partials = self.module._process_product_fills_with_partials(fills)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(len(partials), 1)
        event = partials[0]
        self.assertEqual(event.side, 'SHORT')
        self.assertAlmostEqual(event.qty, 1.0)
        self.assertAlmostEqual(event.entry_price, 200.0)
        self.assertAlmostEqual(event.exit_price, 195.0)
        self.assertAlmostEqual(event.realized_pnl, 5.0)

    def test_remaining_cycle_after_partials(self) -> None:
        fills = [
            self.module.Fill('BTC-PERP-INTX', 'BUY', 2.0, 100.0, 0.0, datetime(2025, 10, 5, 0, 0, tzinfo=UTC), '1'),
            self.module.Fill('BTC-PERP-INTX', 'SELL', 1.0, 105.0, 0.0, datetime(2025, 10, 5, 0, 5, tzinfo=UTC), '2'),
            self.module.Fill('BTC-PERP-INTX', 'SELL', 1.0, 95.0, 0.0, datetime(2025, 10, 5, 0, 10, tzinfo=UTC), '3'),
        ]
        cycles, partials = self.module._process_product_fills_with_partials(fills)
        cycle = cycles[0]
        self.assertEqual(len(partials), 1)
        remaining_qty, remaining_pnl = self.module._remaining_cycle_after_partials(cycle, partials)
        self.assertAlmostEqual(remaining_qty, 1.0)
        self.assertAlmostEqual(remaining_pnl, -5.0)

    def test_logged_partial_totals_for_cycle(self) -> None:
        log_path = self.module._log_file_path()
        opened_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=10)
        later_closed = opened_at + timedelta(minutes=20)

        with log_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.module.LOG_HEADERS)
            writer.writeheader()
            writer.writerow({
                'closed_at': closed_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'ARB-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '2',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'partial_take',
                'entry_price': '0.21',
                'exit_price': '0.22',
                'profit_loss': '1.0',
                'profit_loss_pct': '0.5',
                'mae': '',
                'mfe': '',
                'duration_seconds': '600',
                'order_id': 'p1',
            })
            writer.writerow({
                'closed_at': later_closed.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'ARB-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '3',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'partial_take',
                'entry_price': '0.21',
                'exit_price': '0.20',
                'profit_loss': '-0.5',
                'profit_loss_pct': '-0.25',
                'mae': '',
                'mfe': '',
                'duration_seconds': '1200',
                'order_id': 'p2',
            })
            writer.writerow({
                'closed_at': later_closed.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'ARB-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '3',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'stop_loss',
                'entry_price': '0.21',
                'exit_price': '0.20',
                'profit_loss': '-0.5',
                'profit_loss_pct': '-0.25',
                'mae': '',
                'mfe': '',
                'duration_seconds': '1200',
                'order_id': 'skip',
            })

        cycle = self.module.Cycle(
            product_id='ARB-PERP-INTX',
            side='LONG',
            start_time=opened_at,
            end_time=opened_at + timedelta(hours=1),
            entry_qty=5.0,
            entry_value=1.05,
            exit_qty=5.0,
            exit_value=1.0,
            realized_pnl=0.5,
            fees=0.0,
            closing_order_id='close',
        )

        qty, pnl = self.module._logged_partial_totals_for_cycle(cycle, exclude_order_ids={'p1'})
        self.assertAlmostEqual(qty, 3.0)
        self.assertAlmostEqual(pnl, -0.5)

    def test_logged_partial_order_ids_for_cycle(self) -> None:
        log_path = self.module._log_file_path()
        opened_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=10)
        later_closed = opened_at + timedelta(minutes=20)

        with log_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=self.module.LOG_HEADERS)
            writer.writeheader()
            writer.writerow({
                'closed_at': closed_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'ARB-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '2',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'partial_take',
                'entry_price': '0.21',
                'exit_price': '0.22',
                'profit_loss': '1.0',
                'profit_loss_pct': '0.5',
                'mae': '',
                'mfe': '',
                'duration_seconds': '600',
                'order_id': 'p1',
            })
            writer.writerow({
                'closed_at': later_closed.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'ARB-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '3',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'partial_take',
                'entry_price': '0.21',
                'exit_price': '0.20',
                'profit_loss': '-0.5',
                'profit_loss_pct': '-0.25',
                'mae': '',
                'mfe': '',
                'duration_seconds': '1200',
                'order_id': 'p2',
            })
            writer.writerow({
                'closed_at': later_closed.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'ARB-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '3',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'stop_loss',
                'entry_price': '0.21',
                'exit_price': '0.20',
                'profit_loss': '-0.5',
                'profit_loss_pct': '-0.25',
                'mae': '',
                'mfe': '',
                'duration_seconds': '1200',
                'order_id': 'skip',
            })
            writer.writerow({
                'closed_at': later_closed.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'product_id': 'BTC-PERP-INTX',
                'position_side': 'LONG',
                'net_size': '1',
                'leverage': '',
                'opened_at': opened_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'closure_reason': 'partial_take',
                'entry_price': '50000',
                'exit_price': '50500',
                'profit_loss': '5',
                'profit_loss_pct': '0.01',
                'mae': '',
                'mfe': '',
                'duration_seconds': '1200',
                'order_id': 'other',
            })

        cycle = self.module.Cycle(
            product_id='ARB-PERP-INTX',
            side='LONG',
            start_time=opened_at,
            end_time=opened_at + timedelta(hours=1),
            entry_qty=5.0,
            entry_value=1.05,
            exit_qty=5.0,
            exit_value=1.0,
            realized_pnl=0.5,
            fees=0.0,
            closing_order_id='close',
        )

        order_ids = self.module._logged_partial_order_ids_for_cycle(cycle, exclude_order_ids={'p1'})
        self.assertEqual(order_ids, {'p2'})

    def test_log_tp_sl_promotes_single_partial_order(self) -> None:
        opened_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=5)
        cycle = self.module.Cycle(
            product_id='BTC-PERP-INTX',
            side='LONG',
            start_time=opened_at,
            end_time=closed_at,
            entry_qty=1.0,
            entry_value=100.0,
            exit_qty=1.0,
            exit_value=105.0,
            realized_pnl=5.0,
            fees=0.0,
            closing_order_id='close',
        )
        partial = self.module.PartialFillEvent(
            product_id='BTC-PERP-INTX',
            side='LONG',
            time=closed_at,
            qty=1.0,
            entry_price=100.0,
            exit_price=105.0,
            realized_pnl=5.0,
            fees=0.0,
            order_id='partial1',
            open_time=opened_at,
        )
        fill = self.module.Fill(
            'BTC-PERP-INTX',
            'BUY',
            1.0,
            100.0,
            0.0,
            opened_at,
            'fill1',
        )

        def patch(name: str, value: Any) -> None:
            original = getattr(self.module, name)
            setattr(self.module, name, value)
            self.addCleanup(lambda name=name, original=original: setattr(self.module, name, original))

        patch('fetch_fills', lambda cb, limit=0: [{'dummy': True}])
        patch('_convert_fill', lambda raw: fill)
        patch('_detect_cycles_with_partials', lambda fills: ([cycle], [partial]))
        patch('_load_checkpoint', lambda: None)
        patch('_is_new_cycle', lambda cycle, checkpoint, bootstrap_existing: True)
        patch('_is_new_partial', lambda event, checkpoint, bootstrap_existing: True)
        patch('_active_positions', lambda cb: {})
        patch('_breakeven_threshold', lambda: 0.0)
        patch('compute_mae_mfe_from_history', lambda **kwargs: (None, None))
        patch('_store_checkpoint', lambda *args, **kwargs: None)
        patch('_store_fill_checkpoint', lambda *args, **kwargs: None)

        self.module._log_tp_sl_once(SimpleNamespace(), limit=1, bootstrap_existing=True)

        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get('order_id'), 'partial1')
        self.assertEqual(rows[0].get('closure_reason'), 'take_profit')

    def test_log_tp_sl_collapses_same_order_multifill_close(self) -> None:
        opened_at = datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
        closed_at = opened_at + timedelta(minutes=5)
        cycle = self.module.Cycle(
            product_id='BTC-PERP-INTX',
            side='LONG',
            start_time=opened_at,
            end_time=closed_at,
            entry_qty=1.0,
            entry_value=100.0,
            exit_qty=1.0,
            exit_value=105.0,
            realized_pnl=5.0,
            fees=0.0,
            closing_order_id='ord_same',
        )
        partial = self.module.PartialFillEvent(
            product_id='BTC-PERP-INTX',
            side='LONG',
            time=closed_at,
            qty=0.8,
            entry_price=100.0,
            exit_price=105.0,
            realized_pnl=4.0,
            fees=0.0,
            order_id='ord_same',
            open_time=opened_at,
        )
        fill = self.module.Fill(
            'BTC-PERP-INTX',
            'BUY',
            1.0,
            100.0,
            0.0,
            opened_at,
            'fill1',
        )

        def patch(name: str, value: Any) -> None:
            original = getattr(self.module, name)
            setattr(self.module, name, value)
            self.addCleanup(lambda name=name, original=original: setattr(self.module, name, original))

        patch('fetch_fills', lambda cb, limit=0: [{'dummy': True}])
        patch('_convert_fill', lambda raw: fill)
        patch('_detect_cycles_with_partials', lambda fills: ([cycle], [partial]))
        patch('_load_checkpoint', lambda: None)
        patch('_is_new_cycle', lambda cycle, checkpoint, bootstrap_existing: True)
        patch('_is_new_partial', lambda event, checkpoint, bootstrap_existing: True)
        patch('_active_positions', lambda cb: {})
        patch('_breakeven_threshold', lambda: 0.0)
        patch('compute_mae_mfe_from_history', lambda **kwargs: (None, None))
        patch('_store_checkpoint', lambda *args, **kwargs: None)
        patch('_store_fill_checkpoint', lambda *args, **kwargs: None)

        self.module._log_tp_sl_once(SimpleNamespace(), limit=1, bootstrap_existing=True)

        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].get('order_id'), 'ord_same')
        self.assertEqual(rows[0].get('closure_reason'), 'take_profit')
        self.assertEqual(rows[0].get('profit_loss'), '5')

    def test_log_tp_sl_skips_boundary_truncated_cycle(self) -> None:
        boundary_start = datetime(2025, 8, 27, 19, 7, 51, tzinfo=UTC)
        closed_at = datetime(2026, 2, 18, 10, 33, 46, tzinfo=UTC)
        cycle = self.module.Cycle(
            product_id='BTC-PERP-INTX',
            side='LONG',
            start_time=boundary_start,
            end_time=closed_at,
            entry_qty=0.0037,
            entry_value=302.468080001,
            exit_qty=0.0037,
            exit_value=251.50047,
            realized_pnl=-51.04,
            fees=0.0,
            closing_order_id='27a538d0-547c-4209-8748-8180c69d8641',
        )
        fill = self.module.Fill(
            'BTC-PERP-INTX',
            'BUY',
            0.0037,
            81748.12973,
            0.0,
            boundary_start,
            'historic-buy',
        )

        def patch(name: str, value: Any) -> None:
            original = getattr(self.module, name)
            setattr(self.module, name, value)
            self.addCleanup(lambda name=name, original=original: setattr(self.module, name, original))

        stored_checkpoint: Dict[str, Any] = {}

        patch('fetch_fills', lambda cb, limit=0: [{'dummy': True}])
        patch('_convert_fill', lambda raw: fill)
        patch('_detect_cycles_with_partials', lambda fills: ([cycle], []))
        patch('_load_checkpoint', lambda: None)
        patch('_is_new_cycle', lambda cycle, checkpoint, bootstrap_existing: True)
        patch('_is_new_partial', lambda event, checkpoint, bootstrap_existing: False)
        patch('_active_positions', lambda cb: {'BTC-PERP-INTX': (-0.0037, closed_at)})
        patch('_breakeven_threshold', lambda: 0.0)
        patch('compute_mae_mfe_from_history', lambda **kwargs: (None, None))
        patch(
            '_store_checkpoint',
            lambda *args, **kwargs: stored_checkpoint.update(
                {'time': args[0], 'order_id': args[1]}
            ),
        )
        patch('_store_fill_checkpoint', lambda *args, **kwargs: None)

        self.module._ensure_log_file()
        self.module._log_tp_sl_once(SimpleNamespace(), limit=1, bootstrap_existing=True)

        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [])
        self.assertEqual(stored_checkpoint.get('time'), closed_at)
        self.assertEqual(stored_checkpoint.get('order_id'), cycle.closing_order_id)

    def test_log_tp_sl_skips_boundary_truncated_partial(self) -> None:
        old_open = datetime(2025, 8, 27, 19, 7, 51, tzinfo=UTC)
        event_time = datetime(2026, 2, 18, 10, 33, 46, tzinfo=UTC)
        partial = self.module.PartialFillEvent(
            product_id='BTC-PERP-INTX',
            side='LONG',
            time=event_time,
            qty=0.0037,
            entry_price=81748.12973,
            exit_price=67973.1,
            realized_pnl=-51.04,
            fees=0.0,
            order_id='27a538d0-547c-4209-8748-8180c69d8641',
            open_time=old_open,
        )
        fill = self.module.Fill(
            'BTC-PERP-INTX',
            'SELL',
            0.0037,
            67973.1,
            0.0,
            event_time,
            '27a538d0-547c-4209-8748-8180c69d8641',
        )

        def patch(name: str, value: Any) -> None:
            original = getattr(self.module, name)
            setattr(self.module, name, value)
            self.addCleanup(lambda name=name, original=original: setattr(self.module, name, original))

        stored_fill_checkpoint: Dict[str, Any] = {}

        patch('fetch_fills', lambda cb, limit=0: [{'dummy': True}])
        patch('_convert_fill', lambda raw: fill)
        patch('_detect_cycles_with_partials', lambda fills: ([], [partial]))
        patch('_load_checkpoint', lambda: {'last_time': (event_time - timedelta(minutes=1)).isoformat()})
        patch('_is_new_cycle', lambda cycle, checkpoint, bootstrap_existing: False)
        patch('_is_new_partial', lambda event, checkpoint, bootstrap_existing: True)
        patch('_active_positions', lambda cb: {'BTC-PERP-INTX': (-0.0037, event_time)})
        patch('_breakeven_threshold', lambda: 0.0)
        patch('compute_mae_mfe_from_history', lambda **kwargs: (None, None))
        patch('_store_checkpoint', lambda *args, **kwargs: None)
        patch(
            '_store_fill_checkpoint',
            lambda *args, **kwargs: stored_fill_checkpoint.update(
                {'time': args[0], 'order_id': args[1]}
            ),
        )

        self.module._ensure_log_file()
        self.module._log_tp_sl_once(SimpleNamespace(), limit=1, bootstrap_existing=True)

        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [])
        self.assertEqual(stored_fill_checkpoint.get('time'), event_time)
        self.assertEqual(stored_fill_checkpoint.get('order_id'), partial.order_id)

    def test_log_tp_sl_skips_boundary_anchored_stale_partial_without_active_position(self) -> None:
        old_open = datetime(2025, 8, 27, 19, 7, 51, tzinfo=UTC)
        event_time = datetime(2026, 2, 18, 14, 32, 22, tzinfo=UTC)
        partial = self.module.PartialFillEvent(
            product_id='BTC-PERP-INTX',
            side='LONG',
            time=event_time,
            qty=0.0037,
            entry_price=78961.964865,
            exit_price=67241.1,
            realized_pnl=-43.44,
            fees=0.0,
            order_id='166eba0f-a05b-4000-a844-78c2dc996d0c',
            open_time=old_open,
        )
        boundary_fill = self.module.Fill(
            'BTC-PERP-INTX',
            'BUY',
            0.0037,
            81748.12973,
            0.0,
            old_open,
            'historic-boundary',
        )

        def patch(name: str, value: Any) -> None:
            original = getattr(self.module, name)
            setattr(self.module, name, value)
            self.addCleanup(lambda name=name, original=original: setattr(self.module, name, original))

        stored_fill_checkpoint: Dict[str, Any] = {}

        patch('fetch_fills', lambda cb, limit=0: [{'dummy': True}])
        patch('_convert_fill', lambda raw: boundary_fill)
        patch('_detect_cycles_with_partials', lambda fills: ([], [partial]))
        patch('_load_checkpoint', lambda: {'last_time': (event_time - timedelta(minutes=1)).isoformat()})
        patch('_is_new_cycle', lambda cycle, checkpoint, bootstrap_existing: False)
        patch('_is_new_partial', lambda event, checkpoint, bootstrap_existing: True)
        patch('_active_positions', lambda cb: {})
        patch('_breakeven_threshold', lambda: 0.0)
        patch('compute_mae_mfe_from_history', lambda **kwargs: (None, None))
        patch('_store_checkpoint', lambda *args, **kwargs: None)
        patch(
            '_store_fill_checkpoint',
            lambda *args, **kwargs: stored_fill_checkpoint.update(
                {'time': args[0], 'order_id': args[1]}
            ),
        )

        self.module._ensure_log_file()
        self.module._log_tp_sl_once(SimpleNamespace(), limit=1, bootstrap_existing=True)

        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [])
        self.assertEqual(stored_fill_checkpoint.get('time'), event_time)
        self.assertEqual(stored_fill_checkpoint.get('order_id'), partial.order_id)

    def test_log_tp_sl_skips_partial_outside_visible_fill_window(self) -> None:
        # Open time predates the earliest fetched fill for this product.
        old_open = datetime(2025, 8, 27, 19, 7, 51, tzinfo=UTC)
        earliest_visible = datetime(2026, 1, 9, 2, 8, 43, tzinfo=UTC)
        event_time = datetime(2026, 2, 19, 2, 33, 49, tzinfo=UTC)
        partial = self.module.PartialFillEvent(
            product_id='BTC-PERP-INTX',
            side='LONG',
            time=event_time,
            qty=0.0037,
            entry_price=78065.248649,
            exit_price=66825.9,
            realized_pnl=-41.66,
            fees=0.0,
            order_id='443a3b05-a257-40f7-a3bc-807898120c37',
            open_time=old_open,
        )
        earliest_fill = self.module.Fill(
            'BTC-PERP-INTX',
            'SELL',
            0.0018,
            68000.0,
            0.0,
            earliest_visible,
            'earliest-visible-fill',
        )

        def patch(name: str, value: Any) -> None:
            original = getattr(self.module, name)
            setattr(self.module, name, value)
            self.addCleanup(lambda name=name, original=original: setattr(self.module, name, original))

        stored_fill_checkpoint: Dict[str, Any] = {}

        patch('fetch_fills', lambda cb, limit=0: [{'dummy': True}])
        patch('_convert_fill', lambda raw: earliest_fill)
        patch('_detect_cycles_with_partials', lambda fills: ([], [partial]))
        patch('_load_checkpoint', lambda: {'last_time': (event_time - timedelta(minutes=1)).isoformat()})
        patch('_is_new_cycle', lambda cycle, checkpoint, bootstrap_existing: False)
        patch('_is_new_partial', lambda event, checkpoint, bootstrap_existing: True)
        patch('_active_positions', lambda cb: {})
        patch('_breakeven_threshold', lambda: 0.0)
        patch('compute_mae_mfe_from_history', lambda **kwargs: (None, None))
        patch('_store_checkpoint', lambda *args, **kwargs: None)
        patch(
            '_store_fill_checkpoint',
            lambda *args, **kwargs: stored_fill_checkpoint.update(
                {'time': args[0], 'order_id': args[1]}
            ),
        )

        self.module._ensure_log_file()
        self.module._log_tp_sl_once(SimpleNamespace(), limit=1, bootstrap_existing=True)

        log_path = self.module._log_file_path()
        with log_path.open(newline='') as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(rows, [])
        self.assertEqual(stored_fill_checkpoint.get('time'), event_time)
        self.assertEqual(stored_fill_checkpoint.get('order_id'), partial.order_id)

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
            'last_order_id': cycle.closing_order_id,
        }
        self.assertFalse(self.module._is_new_cycle(cycle, checkpoint_same, bootstrap_existing=False))

        checkpoint_diff = {
            'last_time': cycle.end_time.isoformat(),
            'last_order_id': '42',
        }
        self.assertTrue(self.module._is_new_cycle(cycle, checkpoint_diff, bootstrap_existing=False))

    def test_trim_fills_for_checkpoint_drops_stale_inventory(self) -> None:
        stale_open = self.module.Fill(
            'BTC-PERP-INTX',
            'SELL',
            0.01,
            70000.0,
            0.0,
            datetime(2025, 8, 27, 19, 7, 51, tzinfo=UTC),
            'stale-open',
        )
        recent_open = self.module.Fill(
            'BTC-PERP-INTX',
            'SELL',
            0.0037,
            66825.9,
            0.0,
            datetime(2026, 2, 19, 2, 33, 49, tzinfo=UTC),
            'recent-open',
        )
        recent_close = self.module.Fill(
            'BTC-PERP-INTX',
            'BUY',
            0.0037,
            67075.2,
            0.0,
            datetime(2026, 2, 19, 6, 50, 23, tzinfo=UTC),
            'recent-close',
        )
        fills = [stale_open, recent_open, recent_close]
        checkpoint = {'last_time': '2026-02-19T06:50:23.847063+00:00'}

        trimmed = self.module._trim_fills_for_checkpoint(
            fills,
            checkpoint,
            lookback_hours=24.0,
        )

        self.assertEqual([fill.order_id for fill in trimmed], ['recent-open', 'recent-close'])
        cycles, _ = self.module._detect_cycles_with_partials(trimmed)
        btc_cycles = [cycle for cycle in cycles if cycle.product_id == 'BTC-PERP-INTX']
        self.assertEqual(len(btc_cycles), 1)
        self.assertEqual(btc_cycles[0].closing_order_id, 'recent-close')

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
