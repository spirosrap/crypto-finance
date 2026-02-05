import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

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
