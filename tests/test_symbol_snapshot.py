import importlib.util
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd


def _load_symbol_snapshot_module():
    os.environ.setdefault("CRYPTO_FINDER_LOG_TO_CONSOLE", "0")
    os.environ.setdefault("CRYPTO_FINDER_LOG_LEVEL", "WARNING")
    module_path = Path(__file__).resolve().parent.parent / "scripts" / "symbol_snapshot.py"
    spec = importlib.util.spec_from_file_location("symbol_snapshot", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load symbol_snapshot module spec")
    spec.loader.exec_module(module)
    return module


class TestSymbolSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbol_snapshot = _load_symbol_snapshot_module()

    def test_true_range_last_basic(self):
        df = pd.DataFrame(
            [
                {"high": 110.0, "low": 90.0, "price": 100.0},
                {"high": 110.0, "low": 90.0, "price": 105.0},
            ]
        )
        tr1 = self.symbol_snapshot._true_range_last(df)
        self.assertAlmostEqual(tr1, 20.0, places=6)

    def test_true_range_last_gap(self):
        df = pd.DataFrame(
            [
                {"high": 100.0, "low": 100.0, "price": 100.0},
                {"high": 102.0, "low": 101.0, "price": 101.5},
            ]
        )
        # TR = max(high-low=1, |high-prev_close|=2, |low-prev_close|=1) = 2
        tr1 = self.symbol_snapshot._true_range_last(df)
        self.assertAlmostEqual(tr1, 2.0, places=6)

    def test_true_range_last_insufficient_rows(self):
        df = pd.DataFrame([{"high": 110.0, "low": 90.0, "price": 100.0}])
        self.assertIsNone(self.symbol_snapshot._true_range_last(df))

    def test_true_range_last_missing_columns(self):
        df = pd.DataFrame([{"high": 110.0, "low": 90.0}, {"high": 111.0, "low": 91.0}])
        self.assertIsNone(self.symbol_snapshot._true_range_last(df))

    def test_vol_regime_ratios(self):
        atr7_to_21, tr1_to_atr7 = self.symbol_snapshot._vol_regime_ratios(20.0, 20.0, 20.0)
        self.assertAlmostEqual(atr7_to_21, 1.0, places=6)
        self.assertAlmostEqual(tr1_to_atr7, 1.0, places=6)

        atr7_to_21, tr1_to_atr7 = self.symbol_snapshot._vol_regime_ratios(20.0, 0.0, 20.0)
        self.assertIsNone(atr7_to_21)
        self.assertAlmostEqual(tr1_to_atr7, 1.0, places=6)

    def test_compute_perf_exclusions_drops_worst_fraction(self):
        df = pd.DataFrame(
            {
                "product_id": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
                "profit_loss": [1.0, 1.0, -2.0, -1.5, 0.5, 0.7],
            }
        )
        excluded, meta = self.symbol_snapshot._compute_perf_exclusions(
            df,
            min_trades=2,
            drop_worst_frac=0.34,
            min_expectancy=None,
        )
        self.assertEqual({"BBB"}, excluded)
        self.assertEqual(3, meta["eligible"])

    def test_compute_perf_exclusions_min_expectancy(self):
        df = pd.DataFrame(
            {
                "product_id": ["AAA", "AAA", "BBB", "BBB"],
                "profit_loss": [1.0, -0.5, -1.0, -0.5],
            }
        )
        excluded, _ = self.symbol_snapshot._compute_perf_exclusions(
            df,
            min_trades=2,
            drop_worst_frac=0.0,
            min_expectancy=0.0,
        )
        self.assertEqual({"BBB"}, excluded)

    def test_select_balanced_rows_even_top(self):
        rows = [
            {"symbol": "AAA", "best_side": "LONG", "rr_gap": 0.2, "headroom_bps": 10.0},
            {"symbol": "BBB", "best_side": "LONG", "rr_gap": 0.1, "headroom_bps": 5.0},
            {"symbol": "CCC", "best_side": "SHORT", "rr_gap": 0.05, "headroom_bps": 3.0},
            {"symbol": "DDD", "best_side": "SHORT", "rr_gap": 0.2, "headroom_bps": 4.0},
            {"symbol": "EEE", "best_side": "LONG", "rr_gap": 0.3, "headroom_bps": 2.0},
        ]
        selected, meta = self.symbol_snapshot._select_balanced_rows(rows, 4)
        self.assertEqual(meta["min_per_side"], 2)
        self.assertEqual(meta["longs"], 3)
        self.assertEqual(meta["shorts"], 2)
        selected_symbols = {row["symbol"] for row in selected}
        self.assertSetEqual(selected_symbols, {"AAA", "BBB", "CCC", "DDD"})
        sides = [row["best_side"] for row in selected]
        self.assertEqual(sides.count("LONG"), 2)
        self.assertEqual(sides.count("SHORT"), 2)

    def test_select_balanced_rows_odd_top(self):
        rows = [
            {"symbol": "AAA", "best_side": "LONG", "rr_gap": 0.2, "headroom_bps": 10.0},
            {"symbol": "BBB", "best_side": "LONG", "rr_gap": 0.1, "headroom_bps": 5.0},
            {"symbol": "CCC", "best_side": "SHORT", "rr_gap": 0.05, "headroom_bps": 3.0},
            {"symbol": "DDD", "best_side": "SHORT", "rr_gap": 0.2, "headroom_bps": 4.0},
            {"symbol": "EEE", "best_side": "LONG", "rr_gap": 0.3, "headroom_bps": 2.0},
        ]
        selected, meta = self.symbol_snapshot._select_balanced_rows(rows, 5)
        self.assertEqual(meta["min_per_side"], 2)
        self.assertEqual(len(selected), 5)
        sides = [row["best_side"] for row in selected]
        self.assertEqual(sides.count("LONG"), 3)
        self.assertEqual(sides.count("SHORT"), 2)

    def test_select_balanced_rows_insufficient_side(self):
        rows = [
            {"symbol": "AAA", "best_side": "LONG", "rr_gap": 0.1, "headroom_bps": 10.0},
        ]
        selected, meta = self.symbol_snapshot._select_balanced_rows(rows, 2)
        self.assertEqual(selected, [])
        self.assertEqual(meta["min_per_side"], 1)
        self.assertEqual(meta["longs"], 1)
        self.assertEqual(meta["shorts"], 0)

    def test_load_supported_perps_parses_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "perps.txt"
            path.write_text("# comment\nBTC-PERP-INTX\n\nETH-PERP-INTX\n", encoding="utf-8")
            perps = self.symbol_snapshot._load_supported_perps(path)
            self.assertSetEqual(perps, {"BTC-PERP-INTX", "ETH-PERP-INTX"})

    def test_is_supported_perp_product_handles_thousand_aliases(self):
        supported = {"SHIB-PERP-INTX"}
        self.assertTrue(
            self.symbol_snapshot._is_supported_perp_product("1000SHIB-PERP-INTX", supported)
        )
        supported = {"1000SHIB-PERP-INTX"}
        self.assertTrue(
            self.symbol_snapshot._is_supported_perp_product("SHIB-PERP-INTX", supported)
        )
        self.assertFalse(
            self.symbol_snapshot._is_supported_perp_product("AWE-PERP-INTX", {"BTC-PERP-INTX"})
        )

    def test_is_supported_perp_product_allows_missing_list(self):
        self.assertTrue(
            self.symbol_snapshot._is_supported_perp_product("AWE-PERP-INTX", set())
        )

    # ------------------------------------------------------------------
    # _detect_btc_regime tests
    # ------------------------------------------------------------------

    def _make_btc_df(self, closes, tz_aware=True):
        """Build a daily OHLC DataFrame from a list of close prices."""
        now = datetime.now(timezone.utc)
        # Start far enough back so we have room; dates are all in the past
        base = now - timedelta(days=len(closes))
        dates = [base + timedelta(days=i) for i in range(len(closes))]
        index = pd.DatetimeIndex(dates, tz="UTC" if tz_aware else None)
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "price": closes,
                "volume": [1000.0] * len(closes),
            },
            index=index,
        )
        return df

    def _mock_finder_with_df(self, df):
        finder = MagicMock()
        finder.get_historical_data.return_value = df
        return finder

    def test_detect_btc_regime_bullish(self):
        # EMA20 with 2-day confirm: put 25 candles at 100, then 2 at 120 (well above EMA)
        closes = [100.0] * 25 + [120.0, 120.0]
        df = self._make_btc_df(closes)
        finder = self._mock_finder_with_df(df)
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2, buffer_bps=0.0
        )
        self.assertEqual(result, "bullish")

    def test_detect_btc_regime_bearish(self):
        closes = [100.0] * 25 + [80.0, 80.0]
        df = self._make_btc_df(closes)
        finder = self._mock_finder_with_df(df)
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2, buffer_bps=0.0
        )
        self.assertEqual(result, "bearish")

    def test_detect_btc_regime_neutral_mixed(self):
        # One day above, one day below -> neutral
        closes = [100.0] * 25 + [120.0, 80.0]
        df = self._make_btc_df(closes)
        finder = self._mock_finder_with_df(df)
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2, buffer_bps=0.0
        )
        self.assertEqual(result, "neutral")

    def test_detect_btc_regime_buffer_prevents_whipsaw(self):
        # Closes slightly above EMA but within the buffer -> neutral
        # EMA20 of 25x100 ~ 100; close at 100.1 is within 50bps buffer
        closes = [100.0] * 25 + [100.1, 100.1]
        df = self._make_btc_df(closes)
        finder = self._mock_finder_with_df(df)
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2, buffer_bps=50.0
        )
        self.assertEqual(result, "neutral")

    def test_detect_btc_regime_no_data_returns_neutral(self):
        finder = MagicMock()
        finder.get_historical_data.return_value = None
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2
        )
        self.assertEqual(result, "neutral")

    def test_detect_btc_regime_insufficient_rows_returns_neutral(self):
        closes = [100.0] * 5  # Way too few for EMA20+2
        df = self._make_btc_df(closes)
        finder = self._mock_finder_with_df(df)
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2
        )
        self.assertEqual(result, "neutral")

    def test_detect_btc_regime_invalid_params_returns_neutral(self):
        finder = MagicMock()
        self.assertEqual(
            self.symbol_snapshot._detect_btc_regime(finder, ema_period=0, confirm_days=2),
            "neutral",
        )
        self.assertEqual(
            self.symbol_snapshot._detect_btc_regime(finder, ema_period=20, confirm_days=0),
            "neutral",
        )

    def test_detect_btc_regime_drops_incomplete_candle(self):
        # Last candle is "today" -> should be dropped; regime from the 2 before it
        now = datetime.now(timezone.utc)
        base = now - timedelta(days=27)
        dates = [base + timedelta(days=i) for i in range(28)]
        # 25 candles at 100, then 2 at 120 (bullish), then today's incomplete at 50
        closes = [100.0] * 25 + [120.0, 120.0, 50.0]
        index = pd.DatetimeIndex(dates, tz="UTC")
        df = pd.DataFrame(
            {
                "open": closes, "high": closes, "low": closes,
                "close": closes, "price": closes, "volume": [1000.0] * 28,
            },
            index=index,
        )
        finder = self._mock_finder_with_df(df)
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2
        )
        # Today's 50 is dropped; the 2 confirmed at 120 should give bullish
        self.assertEqual(result, "bullish")

    def test_detect_btc_regime_falls_back_across_btc_symbols(self):
        """First BTC symbol fails, second succeeds."""
        closes = [100.0] * 25 + [120.0, 120.0]
        df = self._make_btc_df(closes)
        finder = MagicMock()
        call_count = {"n": 0}
        def side_effect(symbol, days=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("timeout")
            return df
        finder.get_historical_data.side_effect = side_effect
        result = self.symbol_snapshot._detect_btc_regime(
            finder, ema_period=20, confirm_days=2
        )
        self.assertEqual(result, "bullish")
        self.assertGreaterEqual(call_count["n"], 2)

    def test_regime_tilt_split_bearish_is_30_70(self):
        long_pct, short_pct = self.symbol_snapshot._regime_tilt_split("bearish", 70.0)
        self.assertEqual(long_pct, 30.0)
        self.assertEqual(short_pct, 70.0)

    def test_regime_slot_targets_bearish_top15(self):
        long_slots, short_slots = self.symbol_snapshot._regime_slot_targets(15, "bearish", 70.0)
        self.assertEqual(long_slots, 5)
        self.assertEqual(short_slots, 10)

    # ------------------------------------------------------------------
    # _select_balanced_rows with regime tilt tests
    # ------------------------------------------------------------------

    def _make_rows(self, n_longs, n_shorts):
        """Generate gate-scan rows with ascending rr_gap."""
        rows = []
        for i in range(n_longs):
            rows.append({
                "symbol": f"L{i}", "best_side": "LONG",
                "rr_gap": 0.1 * (i + 1), "headroom_bps": 10.0 - i,
            })
        for i in range(n_shorts):
            rows.append({
                "symbol": f"S{i}", "best_side": "SHORT",
                "rr_gap": 0.1 * (i + 1), "headroom_bps": 10.0 - i,
            })
        return rows

    def test_balanced_tilt_bullish_allocates_70_30(self):
        rows = self._make_rows(10, 10)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=10, regime="bullish", tilt_pct=70.0
        )
        self.assertFalse(meta["suppressed"])
        self.assertEqual(meta["long_slots"], 7)
        self.assertEqual(meta["short_slots"], 3)
        self.assertEqual(meta["selected_longs"], 7)
        self.assertEqual(meta["selected_shorts"], 3)
        self.assertEqual(len(selected), 10)

    def test_balanced_tilt_bearish_allocates_30_70(self):
        rows = self._make_rows(10, 10)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=10, regime="bearish", tilt_pct=70.0
        )
        self.assertFalse(meta["suppressed"])
        self.assertEqual(meta["long_slots"], 3)
        self.assertEqual(meta["short_slots"], 7)
        self.assertEqual(meta["selected_longs"], 3)
        self.assertEqual(meta["selected_shorts"], 7)

    def test_balanced_tilt_neutral_is_even(self):
        rows = self._make_rows(10, 10)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=10, regime="neutral", tilt_pct=70.0
        )
        self.assertFalse(meta["suppressed"])
        self.assertEqual(meta["long_slots"], 5)
        self.assertEqual(meta["short_slots"], 5)

    def test_balanced_tilt_bullish_suppressed_insufficient_shorts(self):
        """Bullish 70/30 on top=10 requires min(7,3)=3 per side; only 2 shorts -> suppressed."""
        rows = self._make_rows(10, 2)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=10, regime="bullish", tilt_pct=70.0
        )
        self.assertTrue(meta["suppressed"])
        self.assertEqual(selected, [])
        self.assertEqual(meta["min_required"], 3)
        self.assertEqual(meta["regime"], "bullish")
        self.assertEqual(meta["suppression_reason"], "insufficient_balance")

    def test_balanced_tilt_bearish_suppressed_insufficient_longs(self):
        """Bearish 70/30 on top=10 requires min(3,7)=3 per side; only 2 longs -> suppressed."""
        rows = self._make_rows(2, 10)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=10, regime="bearish", tilt_pct=70.0
        )
        self.assertTrue(meta["suppressed"])
        self.assertEqual(selected, [])
        self.assertEqual(meta["min_required"], 3)

    def test_balanced_tilt_bullish_top15_slot_math(self):
        """Verify actual production params: top=15, bullish, 70% -> 10L/5S."""
        rows = self._make_rows(12, 8)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=15, regime="bullish", tilt_pct=70.0
        )
        self.assertFalse(meta["suppressed"])
        self.assertEqual(meta["long_slots"], 10)
        self.assertEqual(meta["short_slots"], 5)
        self.assertEqual(meta["min_required"], 5)
        self.assertEqual(len(selected), 15)

    def test_balanced_tilt_remainder_fills_from_pool(self):
        """When one side has fewer than its slots, remainder pool fills the gap."""
        # Bullish top=10: 7 long slots, 3 short slots; but only 5 longs available
        rows = self._make_rows(5, 8)
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            rows, top=10, regime="bullish", tilt_pct=70.0
        )
        # min_required = min(7,3) = 3; we have 5 longs >= 3 and 8 shorts >= 3 -> not suppressed
        self.assertFalse(meta["suppressed"])
        self.assertEqual(len(selected), 10)
        # 5 longs used (all available) + 3 shorts from slots + 2 more shorts from remainder
        self.assertEqual(meta["selected_longs"], 5)
        self.assertEqual(meta["selected_shorts"], 5)

    def test_balanced_tilt_empty_rows(self):
        selected, meta = self.symbol_snapshot._select_balanced_rows(
            [], top=10, regime="bullish", tilt_pct=70.0
        )
        self.assertEqual(selected, [])
        self.assertFalse(meta["suppressed"])
        self.assertEqual(meta["regime"], "bullish")
