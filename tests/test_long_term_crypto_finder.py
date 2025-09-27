import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch

from long_term_crypto_finder import (
    CryptoFinderConfig,
    CryptoMetrics,
    LongTermCryptoFinder,
    RiskLevel,
    build_cli_parser,
    make_risk_level_validator,
)


class LongTermCryptoFinderRiskFilterTests(unittest.TestCase):
    def setUp(self):
        coinbase_patcher = patch('long_term_crypto_finder.CoinbaseService')
        historical_patcher = patch('long_term_crypto_finder.HistoricalData')

        self.mock_coinbase = coinbase_patcher.start()
        self.mock_historical = historical_patcher.start()

        self.addCleanup(coinbase_patcher.stop)
        self.addCleanup(historical_patcher.stop)

        coinbase_instance = MagicMock()
        coinbase_instance.client = MagicMock()
        self.mock_coinbase.return_value = coinbase_instance
        self.mock_historical.return_value = MagicMock()

    def _make_metrics(self, symbol: str, risk_level: RiskLevel, overall_score: float = 60.0) -> CryptoMetrics:
        return CryptoMetrics(
            symbol=symbol,
            name=f"{symbol} Coin",
            position_side='LONG',
            current_price=100.0,
            market_cap=1_000_000_000.0,
            market_cap_rank=10,
            volume_24h=50_000_000.0,
            price_change_24h=2.5,
            price_change_7d=5.0,
            price_change_30d=12.0,
            ath_price=150.0,
            ath_date='2024-01-01',
            atl_price=10.0,
            atl_date='2022-01-01',
            volatility_30d=0.4,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.35,
            rsi_14=55.0,
            macd_signal='NEUTRAL',
            bb_position='NEUTRAL',
            trend_strength=0.1,
            momentum_score=65.0,
            fundamental_score=70.0,
            technical_score=68.0,
            risk_score=30.0,
            overall_score=overall_score,
            risk_level=risk_level,
            entry_price=95.0,
            stop_loss_price=85.0,
            take_profit_price=130.0,
            risk_reward_ratio=3.0,
            position_size_percentage=5.0,
            data_timestamp_utc='2024-01-01T00:00:00Z'
        )

    def test_filter_by_max_risk_level_respects_threshold(self):
        config = CryptoFinderConfig(max_risk_level='MEDIUM', offline=True)
        finder = LongTermCryptoFinder(config=config)

        low = self._make_metrics('LOW', RiskLevel.LOW)
        medium = self._make_metrics('MED', RiskLevel.MEDIUM)
        high = self._make_metrics('HIGH', RiskLevel.HIGH)

        filtered = finder._filter_by_max_risk_level([low, medium, high])

        self.assertEqual({'LOW', 'MED'}, {metric.symbol for metric in filtered})

    def test_filter_by_max_risk_level_no_limit_returns_all(self):
        config = CryptoFinderConfig(max_risk_level=None, offline=True)
        finder = LongTermCryptoFinder(config=config)

        metrics = [
            self._make_metrics('LOW', RiskLevel.LOW),
            self._make_metrics('MED', RiskLevel.MEDIUM),
            self._make_metrics('HIGH', RiskLevel.HIGH),
        ]

        filtered = finder._filter_by_max_risk_level(metrics)

        self.assertEqual(metrics, filtered)

    def test_invalid_risk_level_configuration_is_ignored(self):
        config = CryptoFinderConfig(max_risk_level='not_a_level', offline=True)
        finder = LongTermCryptoFinder(config=config)

        self.assertIsNone(finder._max_risk_level)


class LongTermFinderCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.env_defaults = CryptoFinderConfig()
        self.env_defaults.unique_by_symbol = True
        self.env_defaults.offline = False
        valid_levels = {level.name for level in RiskLevel}
        risk_level_type = make_risk_level_validator(valid_levels)
        self.parser = build_cli_parser(
            env_defaults=self.env_defaults,
            default_limit=50,
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

    def test_force_refresh_toggle(self) -> None:
        args_force = self.parser.parse_args(['--force-refresh'])
        self.assertTrue(args_force.force_refresh)

        args_default = self.parser.parse_args([])
        self.assertFalse(args_default.force_refresh)

    def test_positive_int_validation_for_limit(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(['--limit', '0'])

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parser.parse_args(['--limit', '-10'])

    def test_max_risk_level_default_applied(self) -> None:
        args = self.parser.parse_args([])
        self.assertEqual(args.max_risk_level, 'MEDIUM')


if __name__ == '__main__':
    unittest.main()
