from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET
import matplotlib.pyplot as plt
from ml_model import MLSignal
import logging

# Add this near the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Starting main function")
    # Initialize necessary classes
    coinbase_service = CoinbaseService(API_KEY, API_SECRET)  # Create CoinbaseService instance
    historical_data = HistoricalData(coinbase_service.client)
    logger = logging.getLogger(__name__)

    # Use this logger when creating the MLSignal instance
    ml_signal = MLSignal(logger, historical_data, product_id='BTC-USDC', granularity='ONE_HOUR')

    # Create and train the model
    try:
        ml_signal.train_model()
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")