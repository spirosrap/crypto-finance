import unittest

import short_term_crypto_finder


class SpreadMarginGateTest(unittest.TestCase):
    def test_spread_margin_pct_returns_none_for_unknown_spread(self) -> None:
        self.assertIsNone(short_term_crypto_finder._spread_margin_pct(None, 50))

    def test_spread_margin_pct_calculates_cost_on_margin(self) -> None:
        # 75 bps at 50x leverage => 37.5% of margin
        self.assertAlmostEqual(short_term_crypto_finder._spread_margin_pct(75, 50), 37.5)

    def test_spread_margin_pct_zero_or_negative_inputs(self) -> None:
        self.assertEqual(short_term_crypto_finder._spread_margin_pct(0, 50), 0.0)
        self.assertEqual(short_term_crypto_finder._spread_margin_pct(-1, 50), 0.0)
        self.assertEqual(short_term_crypto_finder._spread_margin_pct(10, 0), 0.0)

    def test_spread_margin_gate_passes_when_unknown(self) -> None:
        self.assertTrue(short_term_crypto_finder._passes_spread_margin_gate(None, 50, 20))

    def test_spread_margin_gate_passes_when_disabled(self) -> None:
        self.assertTrue(short_term_crypto_finder._passes_spread_margin_gate(75, 50, None))
        self.assertTrue(short_term_crypto_finder._passes_spread_margin_gate(75, 50, 0))
        self.assertTrue(short_term_crypto_finder._passes_spread_margin_gate(75, 50, -1))

    def test_spread_margin_gate_blocks_wide_spreads(self) -> None:
        self.assertFalse(short_term_crypto_finder._passes_spread_margin_gate(75, 50, 20))
        self.assertTrue(short_term_crypto_finder._passes_spread_margin_gate(30, 50, 20))


if __name__ == "__main__":
    unittest.main()

