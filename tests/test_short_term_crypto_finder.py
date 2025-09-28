import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from long_term_crypto_finder import CryptoFinderConfig, RiskLevel
from short_term_crypto_finder import (
    ShortTermCryptoFinder,
    build_cli_parser,
    make_risk_level_validator,
)


class ShortTermCryptoFinderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finder = ShortTermCryptoFinder.__new__(ShortTermCryptoFinder)
        self.finder.config = CryptoFinderConfig(
            analysis_days=120,
            rsi_period=7,
            atr_period=7,
            macd_fast=8,
            macd_slow=21,
            macd_signal=5,
        )

        dates = pd.date_range('2025-01-01', periods=40, freq='D')
        base = np.linspace(100, 140, 40)
        noise = np.sin(np.linspace(0, np.pi, 40)) * 2
        prices = base + noise
        highs = prices + 1.5
        lows = prices - 1.5
        volumes = np.linspace(10_000, 25_000, 40)

        self.df = pd.DataFrame({
            'price': prices,
            'open': prices - 0.5,
            'high': highs,
            'low': lows,
            'volume': volumes,
        }, index=dates)

    def test_indicator_enrichment_adds_impulse_metrics(self) -> None:
        metrics = self.finder.calculate_technical_indicators(self.df)

        self.assertIn('volume_thrust_3_15', metrics)
        self.assertIn('impulse_3v10', metrics)
        self.assertIn('continuation_10v21', metrics)
        self.assertGreater(metrics['volume_thrust_3_15'], 1.0)
        self.assertNotEqual(metrics['impulse_3v10'], 0.0)

    def test_long_score_rewards_positive_impulse(self) -> None:
        metrics = self.finder.calculate_technical_indicators(self.df)

        bullish_score = self.finder._calculate_technical_score(metrics, momentum_score=70.0)

        bearish_metrics = dict(metrics)
        bearish_metrics['impulse_3v10'] = -abs(metrics.get('impulse_3v10', 0.05)) - 0.05
        bearish_metrics['continuation_10v21'] = -abs(metrics.get('continuation_10v21', 0.03))
        bearish_metrics['breakout_distance_5d'] = -0.02
        bearish_score = self.finder._calculate_technical_score(bearish_metrics, momentum_score=70.0)

        self.assertGreater(bullish_score, bearish_score)

    def test_short_score_prefers_breakdown_alignment(self) -> None:
        metrics = self.finder.calculate_technical_indicators(self.df)

        bearish_metrics = dict(metrics)
        bearish_metrics['impulse_3v10'] = -abs(metrics.get('impulse_3v10', 0.04)) - 0.04
        bearish_metrics['continuation_10v21'] = -abs(metrics.get('continuation_10v21', 0.02)) - 0.01
        bearish_metrics['breakdown_distance_5d'] = 0.04

        bullish_metrics = dict(metrics)
        bullish_metrics['impulse_3v10'] = abs(metrics.get('impulse_3v10', 0.04)) + 0.04
        bullish_metrics['continuation_10v21'] = abs(metrics.get('continuation_10v21', 0.02))
        bullish_metrics['breakdown_distance_5d'] = 0.0

        bearish_score = self.finder._calculate_technical_score_short(bearish_metrics, momentum_score_long=40.0)
        bullish_score = self.finder._calculate_technical_score_short(bullish_metrics, momentum_score_long=40.0)

        self.assertGreater(bearish_score, bullish_score)


class ShortTermFinderCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_defaults = CryptoFinderConfig()
        self.env_defaults.unique_by_symbol = True
        self.env_defaults.offline = False
        valid_levels = {level.name for level in RiskLevel}
        risk_level_type = make_risk_level_validator(valid_levels)
        self.parser = build_cli_parser(
            env_defaults=self.env_defaults,
            default_limit=30,
            profile_default='default',
            default_max_risk='MEDIUM',
            risk_level_type=risk_level_type,
        )

    def test_boolean_optional_unique_by_symbol_toggle(self) -> None:
        args = self.parser.parse_args(['--no-unique-by-symbol'])
        self.assertFalse(args.unique_by_symbol)

        args_default = self.parser.parse_args([])
        self.assertTrue(args_default.unique_by_symbol)

    def test_boolean_optional_offline_toggle(self) -> None:
        args_enable = self.parser.parse_args(['--offline'])
        self.assertTrue(args_enable.offline)

        args_disable = self.parser.parse_args(['--no-offline'])
        self.assertFalse(args_disable.offline)

    def test_positive_int_validation_for_limit(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(['--limit', '0'])

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(['--limit', '-5'])

    def test_min_volume_arguments_parse(self) -> None:
        args = self.parser.parse_args(['--min-volume', '7500000', '--min-vmc-ratio', '0.04'])
        self.assertEqual(args.min_volume, 7_500_000.0)
        self.assertAlmostEqual(args.min_vmc_ratio, 0.04)

    def test_positive_float_validation_for_min_volume(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(['--min-volume', '0'])

    def test_max_risk_level_default_applied(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.max_risk_level, 'MEDIUM')


if __name__ == '__main__':
    unittest.main()
