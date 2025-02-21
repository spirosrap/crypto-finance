import os
from coinbaseservice import CoinbaseService
import logging
from config import API_KEY_PERPS, API_SECRET_PERPS
import argparse
import time

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

def preview_order(cb_service, product_id: str, side: str, size: float, leverage: str, is_limit: bool = False, limit_price: float = None):
    """
    Preview the order before placing it to check for potential issues.
    """
    try:
        if is_limit:
            preview = cb_service.client.preview_limit_order_gtc(
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                limit_price=str(limit_price),
                leverage=leverage,
                margin_type="CROSS"
            )
        else:
            preview = cb_service.client.preview_market_order(
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                leverage=leverage,
                margin_type="CROSS"
            )
        
        logger.info(f"Order preview response: {preview}")
        
        # Check for preview errors
        if hasattr(preview, 'error_response'):
            error_msg = preview.error_response
            logger.error(f"Order preview failed: {error_msg}")
            return False, error_msg
            
        return True, None
        
    except Exception as e:
        logger.error(f"Error previewing order: {e}")
        return False, str(e)

def validate_params(product_id: str, side: str, size_usd: float, leverage: float, tp_price: float, sl_price: float, limit_price: float, cb_service):
    """Validate input parameters."""
    valid_products = ['BTC-PERP-INTX', 'DOGE-PERP-INTX', 'SOL-PERP-INTX', 'ETH-PERP-INTX', 'XRP-PERP-INTX', "1000SHIB-PERP-INTX"]
    if product_id not in valid_products:
        raise ValueError(f"Invalid product. Must be one of: {', '.join(valid_products)}")

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
    
    if limit_price is not None and limit_price <= 0:
        raise ValueError("Limit price must be positive")
    
    # Get current price
    trades = cb_service.client.get_market_trades(product_id=product_id, limit=1)
    current_price = float(trades['trades'][0]['price'])
    logger.info(f"Current market price: ${current_price}")
    
    # Define maximum allowed price deviation (as percentage)
    max_deviation = 0.80  # 80% deviation
    
    # Calculate price bounds
    upper_bound = current_price * (1 + max_deviation)
    lower_bound = current_price * (1 - max_deviation)
    
    # Validate TP and SL based on side and price bounds
    if side == 'BUY':
        if tp_price <= sl_price:
            raise ValueError("For BUY orders, take profit price must be higher than stop loss price")
        if tp_price > upper_bound:
            raise ValueError(f"Take profit price (${tp_price}) is too high. Maximum allowed is ${upper_bound:.2f} (80% above current price ${current_price:.2f})")
        if sl_price < lower_bound:
            raise ValueError(f"Stop loss price (${sl_price}) is too low. Minimum allowed is ${lower_bound:.2f} (80% below current price ${current_price:.2f})")
        if limit_price is not None:
            if limit_price > current_price:
                raise ValueError(f"For BUY limit orders, limit price (${limit_price}) should be below current price (${current_price})")
            if limit_price <= sl_price:
                raise ValueError(f"For BUY limit orders, limit price (${limit_price}) should be above stop loss (${sl_price})")
            if limit_price >= tp_price:
                raise ValueError(f"For BUY limit orders, limit price (${limit_price}) should be below take profit (${tp_price})")
    else:  # SELL
        if tp_price >= sl_price:
            raise ValueError("For SELL orders, take profit price must be lower than stop loss price")
        if tp_price < lower_bound:
            raise ValueError(f"Take profit price (${tp_price}) is too low. Minimum allowed is ${lower_bound:.2f} (80% below current price ${current_price:.2f})")
        if sl_price > upper_bound:
            raise ValueError(f"Stop loss price (${sl_price}) is too high. Maximum allowed is ${upper_bound:.2f} (80% above current price ${current_price:.2f})")
        if limit_price is not None:
            if limit_price < current_price:
                raise ValueError(f"For SELL limit orders, limit price (${limit_price}) should be above current price (${current_price})")
            if limit_price >= sl_price:
                raise ValueError(f"For SELL limit orders, limit price (${limit_price}) should be below stop loss (${sl_price})")
            if limit_price <= tp_price:
                raise ValueError(f"For SELL limit orders, limit price (${limit_price}) should be above take profit (${tp_price})")

def get_min_base_size(product_id: str) -> float:
    """Get minimum base size for the given product."""
    min_sizes = {
        'BTC-PERP-INTX': 0.0001,  # 0.0001 BTC
        'ETH-PERP-INTX': 0.001,   # 0.001 ETH
        'DOGE-PERP-INTX': 100,    # 100 DOGE
        'SOL-PERP-INTX': 0.1,     # 0.1 SOL
        'XRP-PERP-INTX': 10,      # 10 XRP
        '1000SHIB-PERP-INTX': 100 # 100 units of 1000SHIB
    }
    return min_sizes.get(product_id, 0.0001)

def calculate_base_size(product_id: str, size_usd: float, current_price: float) -> float:
    """Calculate the base size respecting minimum size requirements."""
    min_base_size = get_min_base_size(product_id)
    calculated_size = size_usd / current_price
    
    # Round to appropriate decimal places based on min size
    if min_base_size >= 1:
        # For assets like DOGE, round to whole numbers
        base_size = max(round(calculated_size), min_base_size)
    else:
        # For assets like BTC, maintain precision
        decimal_places = len(str(min_base_size).split('.')[-1])
        base_size = max(round(calculated_size, decimal_places), min_base_size)
    
    logger.info(f"Calculated base size: {base_size} (min: {min_base_size})")
    return base_size

def main():
    parser = argparse.ArgumentParser(description='Place a leveraged market or limit order for perpetual futures')
    parser.add_argument('--product', type=str, default='BTC-PERP-INTX',
                      choices=['BTC-PERP-INTX', 'DOGE-PERP-INTX', 'SOL-PERP-INTX', 'ETH-PERP-INTX', 'XRP-PERP-INTX', "1000SHIB-PERP-INTX"],
                      help='Trading product (default: BTC-PERP-INTX)')
    parser.add_argument('--side', type=str, choices=['BUY', 'SELL'],
                      help='Trade direction (BUY/SELL)')
    parser.add_argument('--size', type=float,
                      help='Position size in USD')
    parser.add_argument('--leverage', type=float,
                      help='Leverage (1-20)')
    parser.add_argument('--tp', type=float,
                      help='Take profit price in USD')
    parser.add_argument('--sl', type=float,
                      help='Stop loss price in USD')
    parser.add_argument('--limit', type=float,
                      help='Limit price in USD (if not provided, a market order will be placed)')
    parser.add_argument('--no-confirm', action='store_true',
                      help='Skip order confirmation')
    # New arguments for placing bracket orders after fill
    parser.add_argument('--place-bracket', action='store_true',
                      help='Place bracket orders for an already filled limit order')
    parser.add_argument('--order-id', type=str,
                      help='Order ID of the filled limit order')

    args = parser.parse_args()

    try:
        # Initialize CoinbaseService
        cb_service = setup_coinbase()
        
        # Handle placing bracket orders after fill
        if args.place_bracket:
            if not all([args.order_id, args.product, args.size, args.tp, args.sl]):
                raise ValueError("For placing bracket orders, --order-id, --product, --size, --tp, and --sl are required")
            
            result = cb_service.place_bracket_after_fill(
                product_id=args.product,
                order_id=args.order_id,
                size=args.size,
                take_profit_price=args.tp,
                stop_loss_price=args.sl,
                leverage=str(args.leverage) if args.leverage else None
            )
            
            if "error" in result:
                if result.get("status") == "pending_fill":
                    print("\nLimit order not filled yet. Please try again once the order is filled.")
                    return
                print(f"\nError placing bracket orders: {result['error']}")
                return
                
            print("\nBracket orders placed successfully!")
            print(f"Take Profit Price: ${result['tp_price']}")
            print(f"Stop Loss Price: ${result['sl_price']}")
            return
        
        # Regular order placement flow
        if not all([args.side, args.size, args.leverage, args.tp, args.sl]):
            raise ValueError("For new orders, --side, --size, --leverage, --tp, and --sl are required")
        
        # Validate parameters
        validate_params(args.product, args.side, args.size, args.leverage, args.tp, args.sl, args.limit, cb_service)
        
        # Check for sufficient funds
        if not check_sufficient_funds(cb_service, args.size, args.leverage):
            raise ValueError("Insufficient funds for this trade")
        
        # Replace the current size calculation with the new one
        trades = cb_service.client.get_market_trades(product_id=args.product, limit=1)
        current_price = float(trades['trades'][0]['price'])
        size = calculate_base_size(args.product, args.size, current_price)
        
        # Preview the order
        is_valid, error_msg = preview_order(
            cb_service=cb_service,
            product_id=args.product,
            side=args.side,
            size=size,
            leverage=str(args.leverage),
            is_limit=args.limit is not None,
            limit_price=args.limit
        )
        
        if not is_valid:
            raise ValueError(f"Order preview failed: {error_msg}")
        
        # Show order summary
        print("\n=== Order Summary ===")
        print(f"Product: {args.product}")
        print(f"Side: {args.side}")
        print(f"Position Size: ${args.size} (≈{size} {args.product.split('-')[0]})")
        print(f"Leverage: {args.leverage}x")
        print(f"Required Margin: ${args.size / args.leverage}")
        print(f"Current Price: ${current_price}")
        print(f"Take Profit Price: ${args.tp}")
        print(f"Stop Loss Price: ${args.sl}")
        if args.limit:
            print(f"Limit Price: ${args.limit}")
        else:
            print("Order Type: Market")
        
        # Place the order based on order type
        if args.limit:
            result = cb_service.place_limit_order_with_targets(
                product_id=args.product,
                side=args.side,
                size=size,
                entry_price=args.limit,
                take_profit_price=args.tp,
                stop_loss_price=args.sl,
                leverage=str(args.leverage)
            )
            
            if "error" in result:
                print(f"\nError placing limit order: {result['error']}")
                return
                
            print("\nLimit order placed successfully!")
            print(f"Order ID: {result['order_id']}")
            print(f"Entry Price: ${result['entry_price']}")
            print(f"Status: {result['status']}")
            
            # Ask if user wants to monitor for fill
            if not args.no_confirm:
                monitor = input("\nWould you like to monitor the order until filled? (yes/no): ").lower()
                if monitor != 'yes':
                    print(f"\n{result['message']}")
                    print("\nTo place take profit and stop loss orders after fill, run:")
                    print(f"python trade_btc_perp.py --place-bracket --order-id {result['order_id']} --product {args.product} --size {size} --tp {args.tp} --sl {args.sl} --leverage {args.leverage}")
                    return
            
            print("\nMonitoring limit order for fill...")
            monitor_result = cb_service.monitor_limit_order_and_place_bracket(
                product_id=args.product,
                order_id=result['order_id'],
                size=size,
                take_profit_price=args.tp,
                stop_loss_price=args.sl,
                leverage=str(args.leverage)
            )
            
            if monitor_result['status'] == 'success':
                print(f"\n{monitor_result['message']}")
                print(f"Take Profit Price: ${monitor_result['tp_price']}")
                print(f"Stop Loss Price: ${monitor_result['sl_price']}")
            else:
                print(f"\n{monitor_result['message']}")
                if monitor_result['status'] == 'timeout':
                    print("\nTo place take profit and stop loss orders after fill, run:")
                    print(f"python trade_btc_perp.py --place-bracket --order-id {result['order_id']} --product {args.product} --size {size} --tp {args.tp} --sl {args.sl} --leverage {args.leverage}")
                elif monitor_result['status'] == 'error':
                    print(f"Error: {monitor_result.get('error', 'Unknown error')}")
            
        else:
            result = cb_service.place_market_order_with_targets(
                product_id=args.product,
                side=args.side,
                size=size,
                take_profit_price=args.tp,
                stop_loss_price=args.sl,
                leverage=str(args.leverage)
            )
            
            if "error" in result:
                print(f"\nError placing order: {result['error']}")
                if isinstance(result['error'], dict):
                    print(f"Error details: {result['error'].get('message', 'No message')}")
                    print(f"Preview failure reason: {result['error'].get('preview_failure_reason', 'Unknown')}")
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