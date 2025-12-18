import unittest
from types import SimpleNamespace

from scripts.long_term_snapshot import _fmt_usd_compact, _format_side, _parse_symbols, _price_precision


class LongTermSnapshotHelpersTest(unittest.TestCase):
    def test_parse_symbols_uppercases(self) -> None:
        self.assertEqual(_parse_symbols("btc, eth ,SOL"), ["BTC", "ETH", "SOL"])

    def test_price_precision_tiers(self) -> None:
        self.assertEqual(_price_precision(0.5), 6)
        self.assertEqual(_price_precision(5.0), 4)
        self.assertEqual(_price_precision(500.0), 3)
        self.assertEqual(_price_precision(5000.0), 2)

    def test_fmt_usd_compact(self) -> None:
        self.assertEqual(_fmt_usd_compact(123), "123")
        self.assertEqual(_fmt_usd_compact(1_234), "1.23K")
        self.assertEqual(_fmt_usd_compact(5_000_000), "5.00M")
        self.assertEqual(_fmt_usd_compact(7_000_000_000), "7.00B")

    def test_format_side_includes_rr_and_risk(self) -> None:
        metric = SimpleNamespace(
            risk_reward_ratio=2.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            rsi_14=55.5,
            trend_strength=0.123,
            momentum_score=66.6,
            risk_level="MEDIUM",
        )
        text = _format_side("LONG", metric)
        self.assertIn("RR=2.00", text)
        self.assertIn("risk=MEDIUM", text)


if __name__ == "__main__":
    unittest.main()

