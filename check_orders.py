from coinbaseservice import CoinbaseService
import os
import logging
from typing import Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_api_keys() -> tuple[Optional[str], Optional[str]]:
    """Load API keys from environment variables or config file."""
    try:
        # First try to import from config.py
        from config import API_KEY_PERPS, API_SECRET_PERPS
        return API_KEY_PERPS, API_SECRET_PERPS
    except ImportError:
        # If config.py doesn't exist, try environment variables
        api_key = os.getenv('API_KEY_PERPS')
        api_secret = os.getenv('API_SECRET_PERPS')
        
        if not (api_key and api_secret):
            logging.error("API keys not found. Please set COINBASE_API_KEY and COINBASE_API_SECRET in config.py or as environment variables.")
            return None, None
            
        return api_key, api_secret

def format_order(order: dict) -> str:
    """Format order details into a readable string."""
    try:
        # Extract basic order information
        order_id = order.get('order_id', 'N/A')
        product_id = order.get('product_id', 'N/A')
        side = order.get('side', 'N/A')
        order_type = order.get('order_type', 'N/A')
        status = order.get('status', 'N/A')
        
        # Extract size and price information from order configuration
        order_config = order.get('order_configuration', {})
        size = 'N/A'
        price = 'N/A'
        
        for config_type in order_config:
            config = order_config[config_type]
            size = config.get('base_size', config.get('quote_size', 'N/A'))
            price = config.get('limit_price', 'MARKET')
        
        return (f"Order ID: {order_id}\n"
                f"Product: {product_id}\n"
                f"Type: {order_type}\n"
                f"Side: {side}\n"
                f"Size: {size}\n"
                f"Price: {price}\n"
                f"Status: {status}\n")
    except Exception as e:
        logging.error(f"Error formatting order: {e}")
        return str(order)

def main():
    # Load API keys
    api_key, api_secret = load_api_keys()
    if not (api_key and api_secret):
        return
    
    try:
        # Initialize CoinbaseService
        service = CoinbaseService(api_key, api_secret)
        
        # Get portfolio UUID for INTX (perpetuals)
        ports = service.client.get_portfolios()
        portfolio_uuid = None
        
        for p in ports['portfolios']:
            if p['type'] == "INTX":
                portfolio_uuid = p['uuid']
                break
        
        if not portfolio_uuid:
            print("No perpetuals portfolio found.")
            return
        
        # Get open orders
        open_orders = service.client.list_orders(
            order_status="OPEN",
            portfolio_uuid=portfolio_uuid
        )
        
        # Convert response to dictionary if needed
        if not isinstance(open_orders, dict):
            if hasattr(open_orders, '__dict__'):
                open_orders = vars(open_orders)
            else:
                open_orders = {'orders': []}
        
        orders = open_orders.get('orders', [])
        
        if not orders:
            print("\n🔍 No open orders found.")
            return
        
        print(f"\n📋 Found {len(orders)} open orders:\n")
        for order in orders:
            if isinstance(order, dict):
                print(format_order(order))
                print("-" * 50)
            else:
                # If order is not a dict, convert it to one
                order_dict = json.loads(json.dumps(order, default=lambda x: vars(x) if hasattr(x, '__dict__') else str(x)))
                print(format_order(order_dict))
                print("-" * 50)
        
    except Exception as e:
        logging.error(f"Error checking orders: {e}")
        print(f"❌ Error checking orders: {str(e)}")

if __name__ == "__main__":
    main() 