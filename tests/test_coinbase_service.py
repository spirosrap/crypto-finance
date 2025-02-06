import unittest
import logging
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET

class TestCoinbaseService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.coinbase_service = CoinbaseService(API_KEY, API_SECRET)

    def test_get_trading_pairs(self):
        """Test getting available trading pairs."""
        try:
            # Get trading pairs
            pairs = self.coinbase_service.get_trading_pairs()
            
            # Print raw pairs for debugging
            print("\nRetrieved pairs:", pairs)
            
            # Basic validation
            self.assertIsNotNone(pairs, "Trading pairs response is None")
            
            if len(pairs) == 0:
                print("Warning: No trading pairs returned. This might indicate an API issue.")
            else:
                print(f"\n✓ Successfully retrieved {len(pairs)} trading pairs")
                print("Sample pairs:", ', '.join(pairs[:5]))
                print("\nAll Available Trading Pairs:")
                for pair in pairs:
                    print(f"  - {pair}")
                
                # Check if common pairs are present
                common_pairs = ['BTC-USDC', 'ETH-USDC', 'SOL-USDC']
                for pair in common_pairs:
                    if pair in pairs:
                        print(f"✓ Found common pair: {pair}")
                    else:
                        print(f"✗ Missing common pair: {pair}")
                
                # Check pair format
                for pair in pairs:
                    self.assertRegex(pair, r'^[A-Z0-9]+-USDC$', f"Invalid pair format: {pair}")
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            self.fail(f"Getting trading pairs failed with error: {str(e)}")

def main():
    # Configure the test runner
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoinbaseService)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main() 