import sys
import os
# Add this at the very top of the file to adjust Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET
import matplotlib.pyplot as plt
from ml_model import MLSignal
import logging
import argparse

# Add this near the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(product_id, granularity):
    print(f"\nStarting training for {product_id} with {granularity} granularity")
    # Initialize necessary classes
    print("Initializing services...")
    coinbase_service = CoinbaseService(API_KEY, API_SECRET)
    historical_data = HistoricalData(coinbase_service.client)
    logger = logging.getLogger(__name__)

    # Use this logger when creating the MLSignal instance
    print("Creating ML model...")
    ml_signal = MLSignal(logger, historical_data, product_id=product_id, granularity=granularity)

    # Create and train the model
    try:
        print("\nStarting model training process...")
        ml_signal.train_model()
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred during model training: {e}")
        return

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train ML model for Bitcoin prediction")
    parser.add_argument("--product_id", type=str, default="BTC-USDC", help="Product ID (e.g., BTC-USDC)")
    parser.add_argument("--granularity", type=str, default="ONE_HOUR", help="Granularity (e.g., ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE)")

    # Parse arguments
    args = parser.parse_args()

    try:
        main(args.product_id, args.granularity)
    except Exception as e:
        print(f"An error occurred: {e}")
