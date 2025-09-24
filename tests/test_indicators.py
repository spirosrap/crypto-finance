import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, UTC

import os
import logging
import time
import json
from unittest import mock
from long_term_crypto_finder import LongTermCryptoFinder, CryptoFinderConfig, _finite


class TestIndicators(unittest.TestCase):
    def setUp(self):
        os.environ['CRYPTO_FINDER_LOG_TO_CONSOLE'] = '0'
        logging.getLogger('long_term_crypto_finder').setLevel(logging.WARNING)
        cfg = CryptoFinderConfig(offline=True)
        self.finder = LongTermCryptoFinder(cfg)

    def test_rsi_wilder_monotonic(self):
        # Strictly increasing should yield RSI ~ 100
        prices_up = np.arange(1, 1 + 20, dtype=float)
        rsi_up = self.finder._rsi_wilder(prices_up, period=14)
        self.assertGreaterEqual(rsi_up, 99.0)

        # Strictly decreasing should yield RSI ~ 0
        prices_dn = np.arange(20, 0, -1, dtype=float)
        rsi_dn = self.finder._rsi_wilder(prices_dn, period=14)
        self.assertLessEqual(rsi_dn, 1.0)

    def test_macd_label(self):
        # Uptrend should be bullish histogram
        prices = np.linspace(100, 200, 60)
        macd, sig, hist, label = self.finder._macd_signal(prices, fast=12, slow=26, signal=9)
        self.assertEqual(label, "BULLISH")
        # Downtrend should be bearish
        prices2 = np.linspace(200, 100, 60)
        macd2, sig2, hist2, label2 = self.finder._macd_signal(prices2, fast=12, slow=26, signal=9)
        self.assertEqual(label2, "BEARISH")

    def test_macd_cross_flag(self):
        # Build a series that flips histogram sign at the end by a sharp drop
        arr = np.concatenate([
            np.linspace(100, 150, 40),
            np.linspace(150, 120, 10),
            np.linspace(120, 80, 15)
        ])
        df = pd.DataFrame({
            'price': arr,
            'high': arr * 1.01,
            'low': arr * 0.99,
            'open': arr,
            'volume': np.ones_like(arr)
        })
        metrics = self.finder.calculate_technical_indicators(df)
        self.assertIn('macd_cross', metrics)
        self.assertIsInstance(metrics['macd_cross'], bool)

    def test_atr_units(self):
        # Build a small OHLC set to compute TRs
        base = datetime(2024, 1, 1, tzinfo=UTC)
        rows = [
            {"timestamp": base + timedelta(days=i), "open": o, "high": h, "low": l, "price": c, "volume": 0.0}
            for i, (o, h, l, c) in enumerate([
                (100, 110, 90, 105),  # seed
                (105, 115, 95, 110),
                (110, 120, 100, 115),
                (115, 125, 105, 120),
            ])
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        atr = self.finder._calculate_atr(df, period=3)
        # Compute expected TRs for last 3 periods
        # TR1 (idx 1): max(h-l=20, |h-prev_c|=10, |l-prev_c|=10) = 20
        # TR2 (idx 2): max(20, 10, 10) = 20
        # TR3 (idx 3): max(20, 10, 10) = 20
        self.assertAlmostEqual(atr, 20.0, places=6)

    def test_max_drawdown(self):
        base = datetime(2024, 1, 1, tzinfo=UTC)
        prices = [100, 120, 80, 90, 70]
        highs = [100, 120, 100, 95, 90]
        lows = [95, 110, 75, 80, 65]
        rows = [
            {"timestamp": base + timedelta(days=i), "price": p, "high": h, "low": l, "open": p, "volume": 0.0}
            for i, (p, h, l) in enumerate(zip(prices, highs, lows))
        ]
        df = pd.DataFrame(rows).set_index("timestamp")
        metrics = self.finder.calculate_technical_indicators(df)
        mdd = metrics.get("max_drawdown", 0)
        # Peak at 120 to trough 70 => (70-120)/120 = -0.416666...
        self.assertAlmostEqual(mdd, -0.4166666667, places=6)

    def test_adx_wilder_bounds(self):
        base = datetime(2024, 1, 1, tzinfo=UTC)
        # Increasing series -> ADX positive finite
        p = np.linspace(100, 200, 40)
        df = pd.DataFrame({
            'timestamp': [base + timedelta(days=i) for i in range(len(p))],
            'price': p, 'high': p * 1.01, 'low': p * 0.99, 'open': p, 'volume': 1.0
        }).set_index('timestamp')
        adx = self.finder._calculate_adx(df, period=14)
        self.assertTrue(np.isfinite(adx))
        self.assertGreaterEqual(adx, 0.0)
        self.assertLessEqual(adx, 100.0)

    def test_edge_cases_indicators(self):
        # Constant series, short length, NaNs at head/tail, alternating up/down
        base = datetime(2024, 1, 1, tzinfo=UTC)
        const = np.full(20, 100.0)
        alt = np.array([100 + ((-1)**i) for i in range(40)], dtype=float)
        # NaNs at head/tail
        with_nan = np.concatenate([[np.nan, np.nan], const, [np.nan]])
        dfc = pd.DataFrame({
            'timestamp': [base + timedelta(days=i) for i in range(len(with_nan))],
            'price': with_nan, 'high': with_nan, 'low': with_nan, 'open': with_nan, 'volume': 1.0
        }).set_index('timestamp')

        metrics = self.finder.calculate_technical_indicators(dfc)
        # Ranges and finiteness
        self.assertTrue(0.0 <= metrics['rsi_14'] <= 100.0)
        self.assertTrue(np.isfinite(metrics['sharpe_ratio']))
        self.assertTrue(np.isfinite(metrics['sortino_ratio']))
        self.assertTrue(np.isfinite(metrics['max_drawdown']))

        # Alternating series
        dfa = pd.DataFrame({
            'timestamp': [base + timedelta(days=i) for i in range(len(alt))],
            'price': alt, 'high': alt * 1.01, 'low': alt * 0.99, 'open': alt, 'volume': 1.0
        }).set_index('timestamp')
        m2 = self.finder.calculate_technical_indicators(dfa)
        self.assertTrue(0.0 <= m2['rsi_14'] <= 100.0)

    def test_trading_levels_invariants(self):
        base = datetime(2024, 1, 1, tzinfo=UTC)
        p = np.linspace(100, 110, 40)
        df = pd.DataFrame({
            'timestamp': [base + timedelta(days=i) for i in range(len(p))],
            'price': p, 'high': p * 1.01, 'low': p * 0.99, 'open': p, 'volume': 1.0
        }).set_index('timestamp')
        tm = self.finder.calculate_technical_indicators(df)
        lv_long = self.finder.calculate_trading_levels(df, float(p[-1]), tm)
        self.assertLess(lv_long['stop_loss_price'], lv_long['entry_price'])
        self.assertLess(lv_long['entry_price'], lv_long['take_profit_price'])
        self.assertGreaterEqual(lv_long['risk_reward_ratio'], 0.0)

        lv_short = self.finder.calculate_short_trading_levels(df, float(p[-1]), tm)
        self.assertLess(lv_short['take_profit_price'], lv_short['entry_price'])
        self.assertLess(lv_short['entry_price'], lv_short['stop_loss_price'])
        self.assertGreaterEqual(lv_short['risk_reward_ratio'], 0.0)

    def test_finite_helper(self):
        self.assertEqual(_finite(float('nan')), 0.0)
        self.assertEqual(_finite(float('inf')), 0.0)
        self.assertEqual(_finite(-float('inf')), 0.0)
        self.assertEqual(_finite(1.23), 1.23)

    def test_json_cache_ttl(self):
        # write cache then expire by editing timestamp
        key = 'unit_test_key'
        self.finder._set_cached_data(key, {'hello': 'world'})
        path = self.finder.cache_dir / f"{self.finder._get_cache_key(f'https://x:{key}')}"  # wrong path; use direct
        # Use direct path used by _set_cached_data
        direct_path = self.finder.cache_dir / f"{key}.json"
        # Ensure we used direct write; if not exist, construct envelope
        if not direct_path.exists():
            self.finder._atomic_write_json(direct_path, {'v': 1, 'ts': time.time(), 'data': {'hello': 'world'}})
        # Confirm read works (bypass via method)
        data = self.finder._get_cached_data(key)
        self.assertIsInstance(data, dict)
        # Expire by setting old ts
        with open(direct_path, 'r', encoding='utf-8') as f:
            env = json.load(f)
        env['ts'] = 0
        with open(direct_path, 'w', encoding='utf-8') as f:
            json.dump(env, f)
        expired = self.finder._get_cached_data(key)
        self.assertIsNone(expired)

    def test_lru_candles(self):
        calls = {'n': 0}
        # Patch underlying historical_data.get_historical_data
        def fake_get(product_id, start, end, gran, **kwargs):
            calls['n'] += 1
            return [{'start': int(start.timestamp()), 'open': 1, 'high': 2, 'low': 0.5, 'close': 1.5, 'volume': 10}]
        self.finder.historical_data.get_historical_data = fake_get  # type: ignore
        key = ('BTC-USDC', 'ONE_DAY', '2024-01-01T00:00:00+00:00', '2024-02-01T00:00:00+00:00')
        out1 = self.finder._cached_candles(*key)
        out2 = self.finder._cached_candles(*key)
        self.assertEqual(calls['n'], 1)
        self.assertEqual(len(out1), len(out2))

    def test_product_fallback(self):
        with mock.patch.object(self.finder, '_make_cached_request', return_value={}):
            res = self.finder._fetch_usdc_products()
        # Expect static set present
        for sym in ["BTC", "ETH", "SOL", "ADA", "MATIC", "AVAX", "DOT", "LINK"]:
            self.assertIn(sym, res)
            self.assertTrue(res[sym]['product_id'].endswith('-USDC'))

    def test_public_only_offline(self):
        os.environ['CRYPTO_PUBLIC_ONLY'] = '1'
        cfg = CryptoFinderConfig(offline=True)
        # Force any session.get to raise if called
        with mock.patch('requests.Session.get', side_effect=AssertionError('network call attempted')):
            LongTermCryptoFinder(cfg)  # should not raise


if __name__ == "__main__":
    unittest.main()
