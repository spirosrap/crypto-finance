import unittest

import pandas as pd

import multi_coin_reservoir_daytrader as mcrd
from perp_support import perp_price_multiplier


class MultiCoinReservoirHelperTests(unittest.TestCase):
    def test_canonical_perp_symbol_applies_prefix(self) -> None:
        self.assertEqual(mcrd.canonical_perp_symbol("shib"), "1000SHIB")
        self.assertEqual(mcrd.canonical_perp_symbol("BTC"), "BTC")

    def test_spot_to_perp_id_handles_thousand_suffix(self) -> None:
        self.assertEqual(mcrd.spot_to_perp_id("SHIB-USDC"), "1000SHIB-PERP-INTX")
        self.assertEqual(mcrd.spot_to_perp_id("BTC-USD"), "BTC-PERP-INTX")

    def test_perp_price_multiplier(self) -> None:
        self.assertEqual(perp_price_multiplier("shib"), 1000.0)
        self.assertEqual(perp_price_multiplier("BTC"), 1.0)

    def test_prefer_usdc_products_selects_usdc_quotes(self) -> None:
        df = pd.DataFrame(
            [
                {"coin": "BONK-USD", "signal": 1, "predicted_return": 0.01},
                {"coin": "BONK-USDC", "signal": 1, "predicted_return": 0.02},
                {"coin": "DOGE-USD", "signal": -1, "predicted_return": -0.03},
            ]
        )
        filtered = mcrd.prefer_usdc_products(df)
        self.assertEqual(filtered["coin"].tolist(), ["BONK-USDC", "DOGE-USD"])


if __name__ == "__main__":
    unittest.main()
