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
        
        # Close all positions (or specify a product_id to close specific position)
        logger.info("Closing all positions...")
        coinbase.close_all_positions()
        logger.info("Position closing complete")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 