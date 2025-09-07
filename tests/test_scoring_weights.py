import unittest

from long_term_crypto_finder import LongTermCryptoFinder, CryptoFinderConfig


class TestScoringWeights(unittest.TestCase):
    def setUp(self):
        cfg = CryptoFinderConfig()
        cfg.offline = True
        self.finder = LongTermCryptoFinder(config=cfg)

    def test_long_technical_score_weights(self):
        # Choose stable bucketed values
        metrics = {
            'rsi_14': 50.0,                # RSI neutral -> 70
            'macd_signal': 'NEUTRAL',      # MACD neutral -> 60
            'bb_position': 'NEUTRAL',      # BB neutral -> 60
            'trend_strength': 0.2,         # Between 0.1 and 0.5 -> 70
            'macd_cross': False,
            'macd_hist': 0.0,
        }
        momentum = 80.0                    # Momentum contribution

        got = self.finder._calculate_technical_score(metrics, momentum)

        # Expected from weights: RSI .25, MACD .05, BB .20, Trend .30, Momentum .20
        exp = (70 * 0.25) + (60 * 0.05) + (60 * 0.20) + (70 * 0.30) + (momentum * 0.20)
        self.assertAlmostEqual(got, exp, places=5)

    def test_short_technical_score_weights_and_momentum_map(self):
        # Stable bucketed values mirroring long
        metrics = {
            'rsi_14': 80.0,                # Overbought -> 90 for shorts
            'macd_signal': 'BEARISH',      # Bearish preferred -> 85
            'bb_position': 'OVERBOUGHT',   # Overbought preferred -> 85
            'trend_strength': -0.2,        # Negative between -0.1 and -0.5 -> 70
            'macd_cross': False,
            'macd_hist': 0.0,
        }
        long_mom1 = 20.0
        long_mom2 = 45.0

        # Score at two different long momenta
        s1 = self.finder._calculate_technical_score_short(metrics, long_mom1)
        s2 = self.finder._calculate_technical_score_short(metrics, long_mom2)

        # Base (without momentum) part for reference
        base = (90 * 0.25) + (85 * 0.05) + (85 * 0.20) + (70 * 0.30)
        # Short momentum contribution should be 0.20 * (100 - long_momentum)
        exp1 = base + 0.20 * (100 - long_mom1)
        exp2 = base + 0.20 * (100 - long_mom2)

        self.assertAlmostEqual(s1, exp1, places=5)
        self.assertAlmostEqual(s2, exp2, places=5)

        # Delta should match 0.20 * (long_mom2 - long_mom1)
        self.assertAlmostEqual(s1 - s2, 0.20 * (long_mom2 - long_mom1), places=5)


if __name__ == '__main__':
    unittest.main()
