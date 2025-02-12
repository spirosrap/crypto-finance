import unittest
import logging
import sys
import os
import timeout_decorator  # You may need to: pip install timeout-decorator

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS  

class TestCoinbaseService(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.coinbase_service = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)

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
            # print("\nRetrieved pairs:", pairs)
            
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

    @timeout_decorator.timeout(30)
    def test_get_portfolio_info(self):
        """Test getting portfolio information for both DEFAULT and PERPETUALS portfolios."""
        try:
            print("\nStarting portfolio info test...")
            
            # Test both portfolio types
            portfolio_types = ["DEFAULT", "INTX"]
            
            for portfolio_type in portfolio_types:
                print(f"\nTesting {portfolio_type} portfolio:")
                print(f"Requesting {portfolio_type} portfolio info from Coinbase...")
                
                fiat_balance, crypto_balance = self.coinbase_service.get_portfolio_info(portfolio_type)
                print("Finished requesting portfolio info")
                
                # Print results for debugging
                print(f"\n{portfolio_type} Portfolio Information:")
                print(f"Fiat Balance: {fiat_balance}")
                print(f"Crypto Balance (BTC): {crypto_balance}")
                
                # Basic validation
                self.assertIsInstance(fiat_balance, float, 
                                    f"{portfolio_type} fiat balance should be a float")
                self.assertIsInstance(crypto_balance, float, 
                                    f"{portfolio_type} crypto balance should be a float")
                self.assertGreaterEqual(fiat_balance, 0.0, 
                                      f"{portfolio_type} fiat balance should not be negative")
                self.assertGreaterEqual(crypto_balance, 0.0, 
                                      f"{portfolio_type} crypto balance should not be negative")
                
                print(f"\n✓ Successfully retrieved {portfolio_type} portfolio information")
                
        except timeout_decorator.TimeoutError:
            self.fail("Test timed out after 30 seconds - the API request may be hanging")
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.fail(f"Getting portfolio info failed with error: {str(e)}")

    @timeout_decorator.timeout(30)
    def test_place_market_order_with_targets(self):
        """Test placing a market order with take profit and stop loss price targets."""
        try:
            print("\nStarting market order with targets test...")
            
            # First check INTX portfolio balance
            fiat_balance, crypto_balance = self.coinbase_service.get_portfolio_info("INTX")
            print("\nINTX Portfolio Balance:")
            print(f"Fiat Balance: ${fiat_balance}")
            print(f"Crypto Balance: {crypto_balance} BTC")
            
            # Test parameters
            product_id = "BTC-PERP-INTX" 
            side = "SELL"
            leverage = "5"  # Lower leverage for testing
            
            # Get current price from recent trades
            trades = self.coinbase_service.client.get_market_trades(product_id=product_id, limit=1)
            current_price = float(trades['trades'][0]['price'])
            
            # Calculate size in BTC based on current price
            # Try with $1000 order size
            usd_order_size = 100  # Increased to $100
            size = round(usd_order_size / current_price, 4)  # Round to 4 decimal places
            
            # Calculate required margin
            required_margin = (size * current_price) / float(leverage)  # Full position value / leverage
            
            print(f"\nMargin Calculations:")
            print(f"Current BTC Price: ${current_price}")
            print(f"Order Size in BTC: {size} BTC")
            print(f"Order Size in USD: ${usd_order_size}")
            print(f"Position Value: ${size * current_price}")
            print(f"Required Margin: ${required_margin}")
            print(f"Available Balance: ${fiat_balance}")
            
            if fiat_balance < required_margin:
                self.fail(f"Insufficient margin. Have: ${fiat_balance}, Need: ${required_margin}")
            
            # Validate minimum size
            min_btc_size = 0.001  # Increased minimum BTC size
            if size < min_btc_size:
                self.fail(f"Order size too small. Have: {size} BTC, Need at least: {min_btc_size} BTC")
            
            take_profit_price = current_price * 1.01  # 1% above current price
            stop_loss_price = current_price * 0.99   # 1% below current price
            
            print(f"\nPlacing {side} order:")
            print(f"Product ID: {product_id}")
            print(f"Size in BTC: {size}")
            print(f"Size in USD: ${usd_order_size}")
            print(f"Leverage: {leverage}x")
            print(f"Take Profit Price: ${take_profit_price}")
            print(f"Stop Loss Price: ${stop_loss_price}")
            
            # Place the order with correct parameter names
            result = self.coinbase_service.place_market_order_with_targets(
                product_id=product_id,
                side=side,
                size=size,  # Now passing BTC amount
                leverage=leverage,
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price
            )
            
            # Basic validation
            self.assertIsNotNone(result, "Order result should not be None")
            self.assertIsInstance(result, dict, "Order result should be a dictionary")
            
            # Check if there was an error
            if 'error' in result:
                print("\nOrder failed:")
                print(f"Error: {result['error']}")
                self.fail(f"Order failed: {result['error']}")
            
            # If no error, verify the order details
            print("\nOrder Result:")
            print(f"Market Order: {result.get('market_order', 'N/A')}")
            print(f"Take Profit Order: {result.get('take_profit_order', 'N/A')}")
            print(f"Stop Loss Order: {result.get('stop_loss_order', 'N/A')}")
            
            # Only verify required fields if no error
            required_fields = ['market_order', 'take_profit_order', 'stop_loss_order']
            for field in required_fields:
                self.assertIn(field, result, f"Result missing required field: {field}")
            
            print("\n✓ Successfully placed market order with targets")
            
        except timeout_decorator.TimeoutError:
            self.fail("Test timed out after 30 seconds - the API request may be hanging")
        except Exception as e:
            print(f"Error details: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            self.fail(f"Placing market order with targets failed with error: {str(e)}")

def main():
    # Configure the test runner
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoinbaseService)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main() 