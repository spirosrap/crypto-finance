import unittest
from technicalanalysis import TechnicalAnalysis, TechnicalAnalysisConfig
from coinbaseservice import CoinbaseService

class TestTechnicalAnalysis(unittest.TestCase):
    def setUp(self):
        self.coinbase_service = CoinbaseService()  # Mock this if necessary
        self.ta = TechnicalAnalysis(self.coinbase_service)

    def test_calculate_rsi(self):
        prices = [10, 12, 11, 13, 15, 14, 16]
        rsi = self.ta.calculate_rsi(tuple(prices), period=5)
        self.assertIsInstance(rsi, float)
        self.assertTrue(0 <= rsi <= 100)

    # Add more tests for other methods

if __name__ == '__main__':
    unittest.main()