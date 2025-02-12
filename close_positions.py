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
        
        # First cancel all open orders
        logger.info("Cancelling all open orders first...")
        coinbase.cancel_all_orders()
        
        # Then close all positions
        logger.info("Now closing all positions...")
        coinbase.close_all_positions()
        logger.info("Position closing complete")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 