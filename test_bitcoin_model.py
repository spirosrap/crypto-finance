import pandas as pd
import numpy as np
from bitcoinpredictionmodel import BitcoinPredictionModel
from datetime import datetime, timedelta
from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY, API_SECRET
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from external_data import ExternalDataFetcher
from sklearn.model_selection import train_test_split
import schedule
import time
import argparse
import logging


def main(granularity="ONE_MINUTE", product_id="BTC-USDC"):
    print(f"Starting main function with granularity: {granularity}, product_id: {product_id}")
    
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize necessary classes
    coinbase_service = CoinbaseService(API_KEY, API_SECRET)
    logger.debug("CoinbaseService initialized")

    # Create and train the model with custom granularity and product_id
    model = BitcoinPredictionModel(coinbase_service, granularity=granularity, product_id=product_id)
    try:
        logger.debug("Creating model instance...")
        model.train()
    except Exception as e:
        logger.error(f"An error occurred during model training: {str(e)}")
        logger.error("Error details:", exc_info=True)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bitcoin prediction model with custom parameters")
    parser.add_argument("--granularity", type=str, default="ONE_HOUR", 
                        help="Granularity for the data (e.g., ONE_MINUTE, FIVE_MINUTE, ONE_HOUR)")
    parser.add_argument("--product_id", type=str, default="BTC-USDC", 
                        help="Product ID for the cryptocurrency pair (e.g., BTC-USDC, ETH-USDC)")
    
    args = parser.parse_args()
    main(granularity=args.granularity, product_id=args.product_id)
