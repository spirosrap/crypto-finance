import logging
from coinbaseservice import CoinbaseService
from config import API_KEY_PERPS, API_SECRET_PERPS

def setup_logging():
    """Set up basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize CoinbaseService with your API credentials
        coinbase = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
        
        # Cancel all orders (passing None as product_id will cancel orders for all products)
        logger.info("Cancelling all open orders...")
        coinbase.cancel_all_orders()
        logger.info("Order cancellation complete")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 