import unittest
from unittest.mock import patch, MagicMock
from base import CryptoTrader
from config import API_KEY, API_SECRET, NEWS_API_KEY
class TestCryptoTrader(unittest.TestCase):

    @patch('base.RESTClient')
    def setUp(self, MockRESTClient):
        # Set up the mock REST client
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.trader = CryptoTrader(self.api_key, self.api_secret)
        self.trader.client = MockRESTClient

    @patch('coinbaseservice.CoinbaseService.get_portfolio_info')
    def test_get_portfolio_info(self, mock_get_portfolio_info):
        # Mock the return value
        mock_get_portfolio_info.return_value = (1000.0, 0.01)
        
        fiat_usd, btc = self.trader.get_portfolio_info()
        self.assertEqual(fiat_usd, 1000.0)
        self.assertEqual(btc, 0.01)

    @patch('coinbaseservice.CoinbaseService.get_btc_prices')
    def test_get_btc_prices(self, mock_get_btc_prices):
        # Mock the return value
        mock_get_btc_prices.return_value = {
            "BTC-USD": {"bid": 60000.0, "ask": 60001.0}
        }
        
        prices = self.trader.get_btc_prices()
        self.assertEqual(prices["BTC-USDC"]["bid"], 60000.0)
        self.assertEqual(prices["BTC-USDC"]["ask"], 60001.0)

    def test_calculate_trade_amount_and_fee_buy(self):
        balance = 1000.0
        price = 60000.0
        is_buy = True
        
        trade_amount, fee = self.trader.calculate_trade_amount_and_fee(balance, price, is_buy)
        self.assertAlmostEqual(trade_amount, 0.0166, places=4)  # Adjust based on expected value
        self.assertAlmostEqual(fee, 0.0, places=4)  # Adjust based on expected value

    def test_calculate_trade_amount_and_fee_sell(self):
        balance = 0.01 * 60000.0  # Equivalent to 600.0
        price = 60000.0
        is_buy = False
        
        trade_amount, fee = self.trader.calculate_trade_amount_and_fee(balance, price, is_buy)
        self.assertAlmostEqual(trade_amount, 0.0095, places=4)  # Adjust based on expected value
        self.assertAlmostEqual(fee, 3.0, places=4)  # Adjust based on expected value

    @patch('base.CryptoTrader.get_historical_data')
    def test_backtest(self, mock_get_historical_data):
        # Mock historical data
        mock_get_historical_data.return_value = [
            {'close': 60000, 'start': 1610000000},
            {'close': 61000, 'start': 1610003600},
            {'close': 62000, 'start': 1610007200},
        ]

        final_value, trades = self.trader.backtest("BTC-USDC", "2021-01-01", "2021-01-03", 1000.0)
        self.assertGreater(final_value, 1000.0)  # Expecting a profit
        self.assertIsInstance(trades, list)  # Ensure trades is a list

if __name__ == '__main__':
    unittest.main()