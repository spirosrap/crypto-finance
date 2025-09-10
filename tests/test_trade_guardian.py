import os
import types
import unittest
from unittest import mock
from datetime import datetime, timedelta

import pandas as pd

import trade_guardian as tg


def _make_df(days=120, start_price=900.0, step=3.0):
    now = datetime.utcnow()
    idx = [now - timedelta(days=(days - i)) for i in range(days)]
    prices = [start_price + i * step for i in range(days)]
    opens = [p * 0.999 for p in prices]
    highs = [p * 1.005 for p in prices]
    lows = [p * 0.995 for p in prices]
    vols = [1000.0 for _ in prices]
    df = pd.DataFrame({
        'price': prices,
        'open': opens,
        'high': highs,
        'low': lows,
        'volume': vols,
    }, index=pd.to_datetime(idx))
    return df


class _StubFinder:
    def __init__(self, tech):
        self._tech = dict(tech)
        # historical_data is referenced by _latest_price_from_hourly but we patch that in tests
        self.historical_data = types.SimpleNamespace()

    def get_historical_data(self, product_id: str, days: int = 365):
        return _make_df(days=min(days, 120), start_price=950.0, step=3.0)

    def calculate_technical_indicators(self, df):
        return dict(self._tech)


class TradeGuardianTests(unittest.TestCase):
    def setUp(self):
        # Keep sensible defaults
        os.environ.pop('TRADE_GUARD_MIN_CONF', None)
        os.environ.pop('TRADE_GUARD_MIN_ATR_CUSH', None)

    def test_raise_sl_with_high_confidence(self):
        tech = {
            'atr': 50.0,
            'adx': 30.0,
            'trend_strength': 20.0,
            'macd_hist': 0.5,
            'rsi_14': 60.0,
        }
        finder = _StubFinder(tech)

        trade = tg.Trade(
            product_id='BTC-USDC', side='LONG', entry_price=1000.0,
            stop_loss=900.0, initial_stop=900.0, take_profit=None
        )

        # Price well above entry
        cur_price = 1300.0
        def _lp(_h, _pid):
            return (cur_price, datetime.utcnow())

        with mock.patch.object(tg, '_latest_price_from_hourly', side_effect=_lp):
            rec = tg._recommend_for_trade(finder, trade, analysis_days=120, extend_tp=False, rr_target=3.0, quotes=['USDC'])
        self.assertEqual(rec.action, 'RAISE_SL')
        # Should raise stop meaningfully above old SL and below current price
        self.assertGreater(rec.new_stop_loss, 1200.0)
        self.assertLess(rec.new_stop_loss, cur_price)
        self.assertGreaterEqual(rec.confidence, 0.55)

    def test_low_confidence_gates_update(self):
        tech = {
            'atr': 50.0,
            'adx': 12.0,  # weak trend
            'trend_strength': 0.0,
            'macd_hist': -0.4,  # counter-trend for long
            'rsi_14': 80.0,  # extreme
        }
        finder = _StubFinder(tech)
        trade = tg.Trade(
            product_id='ETH-USDC', side='LONG', entry_price=1000.0,
            stop_loss=900.0, initial_stop=900.0, take_profit=None
        )
        cur_price = 1300.0
        def _lp(_h, _pid):
            return (cur_price, datetime.utcnow())
        with mock.patch.object(tg, '_latest_price_from_hourly', side_effect=_lp):
            rec = tg._recommend_for_trade(finder, trade, analysis_days=120, extend_tp=False, rr_target=3.0, quotes=['USDC'])
        # Even though candidates exist, low confidence should keep stop
        self.assertEqual(rec.action, 'KEEP')
        self.assertAlmostEqual(rec.new_stop_loss, 900.0, places=6)
        self.assertLess(rec.confidence, 0.45)

    def test_exit_hit_when_price_below_sl(self):
        tech = {
            'atr': 30.0,
            'adx': 20.0,
            'trend_strength': -5.0,
            'macd_hist': -0.2,
            'rsi_14': 40.0,
        }
        finder = _StubFinder(tech)
        trade = tg.Trade(
            product_id='BTC-USDC', side='LONG', entry_price=1000.0,
            stop_loss=900.0, initial_stop=900.0, take_profit=None
        )
        cur_price = 880.0
        def _lp(_h, _pid):
            return (cur_price, datetime.utcnow())
        with mock.patch.object(tg, '_latest_price_from_hourly', side_effect=_lp):
            rec = tg._recommend_for_trade(finder, trade, analysis_days=60, extend_tp=False, rr_target=3.0, quotes=['USDC'])
        self.assertEqual(rec.status, 'EXIT_HIT')
        self.assertEqual(rec.action, 'EXIT_HIT')


if __name__ == '__main__':
    unittest.main()
