import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import add_position_from_finder


FINDER_SAMPLE = """1. NOICE (NOICE-USDC) — LONG
--------------------------------------------------
Data Timestamp (UTC): 2025-10-30 07:00:00Z
Price: $0.000391
Predicted Return (next period): 1.55%
RSI: 39.90
ATR %: 5.51%
Relative Volume: 0.27

💼 TRADING LEVELS (LONG):
Entry Price: $0.000391
Stop Loss: $0.000370
Take Profit: $0.000434
Risk:Reward Ratio: 2.00:1
Recommended Position Size: 5.0% of portfolio
Take-Profit Distance: 11.01% | Stop-Loss Distance: 5.51%
Signal Expires In: 24 hours
"""

FINDER_SHIB = """1. SHIB (SHIB-USDC) — SHORT
--------------------------------------------------
Data Timestamp (UTC): 2025-10-30 07:00:00Z
Price: $0.000010
Predicted Return (next period): -0.50%
RSI: 48.00
ATR %: 1.50%
Relative Volume: 0.30

💼 TRADING LEVELS (SHORT):
Entry Price: $0.000010
Stop Loss: $0.000011
Take Profit: $0.000009
Risk:Reward Ratio: 2.00:1
Recommended Position Size: 5.0% of portfolio
Take-Profit Distance: 10.00% | Stop-Loss Distance: 5.00%
Signal Expires In: 24 hours
"""


class AddPositionFromFinderTest(unittest.TestCase):
    def test_parse_preserves_decimal_precision(self) -> None:
        parsed = add_position_from_finder.parse_finder_text(FINDER_SAMPLE)
        self.assertEqual(parsed.base_symbol, "NOICE")
        self.assertEqual(parsed.entry_decimals, 6)
        self.assertEqual(parsed.stop_decimals, 6)
        self.assertEqual(parsed.take_profit_decimals, 6)
        self.assertEqual(parsed.max_price_decimals(), 6)

    def test_normalize_perp_applies_thousand_prefix(self) -> None:
        self.assertEqual(
            add_position_from_finder.normalize_perp("SHIB"),
            "1000SHIB-PERP-INTX",
        )
        self.assertEqual(
            add_position_from_finder.normalize_perp("shib", prefer="INTX-PERP"),
            "1000SHIB-INTX-PERP",
        )

    @mock.patch("add_position_from_finder.is_perp_supported", return_value=True)
    def test_micro_price_levels_survive_rounding(self, mock_support: mock.Mock) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(FINDER_SAMPLE)
            tmp_path = tmp.name
        try:
            with mock.patch("add_position_from_finder.get_price_precision", return_value=0.01):
                argv = [
                    "add_position_from_finder.py",
                    "--file",
                    tmp_path,
                    "--portfolio-usd",
                    "13000",
                    "--leverage",
                    "50",
                    "--order",
                    "market",
                ]
                with mock.patch("sys.argv", argv):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        add_position_from_finder.main()
            output = buf.getvalue()
            self.assertIn("--tp 0.000434", output)
            self.assertIn("--sl 0.000370", output)
            self.assertIn("TP +$71.48", output)
        finally:
            os.unlink(tmp_path)

    @mock.patch("add_position_from_finder.is_perp_supported", return_value=False)
    def test_skips_unsupported_products(self, mock_support: mock.Mock) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(FINDER_SAMPLE)
            tmp_path = tmp.name
        try:
            argv = [
                "add_position_from_finder.py",
                "--file",
                tmp_path,
                "--portfolio-usd",
                "13000",
                "--leverage",
                "50",
                "--order",
                "market",
            ]
            with mock.patch("sys.argv", argv):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    add_position_from_finder.main()
            output = buf.getvalue()
            self.assertIn("Skipped unsupported Coinbase perps", output)
            self.assertNotIn("--product", output)
        finally:
            os.unlink(tmp_path)

    @mock.patch("add_position_from_finder.is_perp_supported", return_value=True)
    def test_thousand_unit_multiplier(self, mock_support: mock.Mock) -> None:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.write(FINDER_SHIB)
            tmp_path = tmp.name
        try:
            argv = [
                "add_position_from_finder.py",
                "--file",
                tmp_path,
                "--portfolio-usd",
                "13000",
                "--leverage",
                "50",
                "--order",
                "market",
            ]
            with mock.patch("sys.argv", argv):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    add_position_from_finder.main()
            output = buf.getvalue()
            self.assertIn("--product 1000SHIB-PERP-INTX", output)
            self.assertIn("--tp 0.009000", output)
            self.assertIn("--sl 0.011000", output)
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
