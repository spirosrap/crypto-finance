import os
from coinbaseservice import CoinbaseService
import logging
from config import API_KEY_PERPS, API_SECRET_PERPS
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_coinbase():
    """Initialize CoinbaseService with API credentials."""
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    
    if not api_key or not api_secret:
        raise ValueError("API credentials not found")
    
    return CoinbaseService(api_key, api_secret)

def check_sufficient_funds(cb_service, size_usd: float, leverage: float) -> bool:
    """
    Check if there are sufficient funds for the trade.
    Required margin = Position Size / Leverage
    """
    try:
        # Get portfolio balance
        balance, _ = cb_service.get_portfolio_info(portfolio_type="INTX")
        required_margin = size_usd / leverage
        
        logger.info(f"Available balance: ${balance}")
        logger.info(f"Required margin: ${required_margin}")
        
        return balance >= required_margin
    except Exception as e:
        logger.error(f"Error checking funds: {e}")
        return False

def validate_params(side: str, size_usd: float, leverage: float, tp_price: float, sl_price: float):
    """Validate input parameters."""
    if side not in ['BUY', 'SELL']:
        raise ValueError("Side must be either 'BUY' or 'SELL'")
    
    if size_usd <= 0:
        raise ValueError("Position size must be positive")
    
    if not 1 <= leverage <= 20:
        raise ValueError("Leverage must be between 1 and 20")
    
    if tp_price <= 0:
        raise ValueError("Take profit price must be positive")
    
    if sl_price <= 0:
        raise ValueError("Stop loss price must be positive")
    
    # Validate TP and SL based on side
    if side == 'BUY':
        if tp_price <= sl_price:
            raise ValueError("For BUY orders, take profit price must be higher than stop loss price")
    else:  # SELL
        if tp_price >= sl_price:
            raise ValueError("For SELL orders, take profit price must be lower than stop loss price")

def main():
    parser = argparse.ArgumentParser(description='Place a leveraged market order for BTC-PERP-INTX')
    parser.add_argument('--side', type=str, required=True, choices=['BUY', 'SELL'],
                      help='Trade direction (BUY/SELL)')
    parser.add_argument('--size', type=float, required=True,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=float, required=True,
                      help='Leverage (1-20)')
    parser.add_argument('--tp', type=float, required=True,
                      help='Take profit price in USD')
    parser.add_argument('--sl', type=float, required=True,
                      help='Stop loss price in USD')
    parser.add_argument('--no-confirm', action='store_true',
                      help='Skip order confirmation')

    args = parser.parse_args()

    try:
        # Validate parameters
        validate_params(args.side, args.size, args.leverage, args.tp, args.sl)
        
        # Initialize CoinbaseService
        cb_service = setup_coinbase()
        
        # Check for sufficient funds
        if not check_sufficient_funds(cb_service, args.size, args.leverage):
            raise ValueError("Insufficient funds for this trade")
        
        # Show order summary
        print("\n=== Order Summary ===")
        print(f"Product: BTC-PERP-INTX")
        print(f"Side: {args.side}")
        print(f"Position Size: ${args.size}")
        print(f"Leverage: {args.leverage}x")
        print(f"Required Margin: ${args.size / args.leverage}")
        print(f"Take Profit Price: ${args.tp}")
        print(f"Stop Loss Price: ${args.sl}")
        
        # Check for confirmation unless --no-confirm is specified
        if not args.no_confirm:
            confirm = input("\nConfirm order? (yes/no): ").lower()
            if confirm != 'yes':
                print("Order cancelled.")
                return
        
        # Place the order
        trades = cb_service.client.get_market_trades(product_id="BTC-PERP-INTX", limit=1)
        current_price = float(trades['trades'][0]['price'])
        
        # Calculate size in BTC based on current price
        # Try with $1000 order size
        usd_order_size = 100  # Increased to $100
        size = round(usd_order_size / current_price, 4)  # Round to 4 decimal places


        result = cb_service.place_market_order_with_targets(
            product_id="BTC-PERP-INTX",
            side=args.side,
            size=size,  # Pass USD size directly
            take_profit_price=args.tp,
            stop_loss_price=args.sl,
            leverage=str(args.leverage)
        )
        
        if "error" in result:
            print(f"\nError placing order: {result['error']}")
        else:
            print("\nOrder placed successfully!")
            print(f"Order ID: {result['order_id']}")
            print(f"Take Profit Price: ${result['tp_price']}")
            print(f"Stop Loss Price: ${result['sl_price']}")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main() 