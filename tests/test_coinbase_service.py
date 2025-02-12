import unittest
import logging
import sys
import os
import timeout_decorator  # You may need to: pip install timeout-decorator

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    @timeout_decorator.timeout(30)  # Set 30 second timeout
    def test_get_trading_pairs(self):
        """Test getting available trading pairs."""
        try:
            print("\nStarting trading pairs test...")
            
            # Get trading pairs with progress logging
            print("Requesting trading pairs from Coinbase...")
            pairs = self.coinbase_service.get_trading_pairs()
            print("Finished requesting trading pairs")
            
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
                    self.assertRegex(pair, r'^[A-Z0-9]+-[A-Z0-9]+$', f"Invalid pair format: {pair}")
            
        except timeout_decorator.TimeoutError:
            self.fail("Test timed out after 30 seconds - the API request may be hanging")
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.fail(f"Getting trading pairs failed with error: {str(e)}")

    @timeout_decorator.timeout(30)  # Set 30 second timeout
    def test_get_portfolio_info(self):
        """Test getting portfolio information."""
        try:
            print("\nStarting portfolio info test...")
            
            # Get portfolio info
            print("Requesting portfolio info from Coinbase...")
            fiat_balance, crypto_balance = self.coinbase_service.get_portfolio_info()
            print("Finished requesting portfolio info")
            
            # Print results for debugging
            print(f"\nPortfolio Information:")
            print(f"Fiat Balance: {fiat_balance}")
            print(f"Crypto Balance (BTC): {crypto_balance}")
            
            # Basic validation
            self.assertIsInstance(fiat_balance, float, "Fiat balance should be a float")
            self.assertIsInstance(crypto_balance, float, "Crypto balance should be a float")
            self.assertGreaterEqual(fiat_balance, 0.0, "Fiat balance should not be negative")
            self.assertGreaterEqual(crypto_balance, 0.0, "Crypto balance should not be negative")
            
            print("\n✓ Successfully retrieved portfolio information")
            
        except timeout_decorator.TimeoutError:
            self.fail("Test timed out after 30 seconds - the API request may be hanging")
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.fail(f"Getting portfolio info failed with error: {str(e)}")

def main():
    # Configure the test runner
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoinbaseService)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main() 