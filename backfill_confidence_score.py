#!/usr/bin/env python3
"""
Script to backfill the Confidence Score column for all trades in automated_trades.csv.
This adds confidence scores for trades that were executed before the column was added.
"""

import pandas as pd
import logging
from datetime import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backfill_confidence_score.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def backfill_confidence_score():
    """Backfill Confidence Score column for all trades"""
    try:
        # Read the CSV file
        csv_file_path = 'automated_trades.csv'
        
        # Check if file exists
        if not os.path.isfile(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return
            
        # Read the trades
        trades_df = pd.read_csv(csv_file_path)
        
        # Check if Confidence Score column already exists
        if 'Confidence Score' in trades_df.columns:
            logger.info("Confidence Score column already exists")
            return
            
        # Add Confidence Score column with default value 0.0
        trades_df['Confidence Score'] = 0.0
        
        # Save the updated DataFrame back to CSV
        trades_df.to_csv(csv_file_path, index=False)
        logger.info(f"Successfully added Confidence Score column to {csv_file_path}")
        
    except Exception as e:
        logger.error(f"Error backfilling Confidence Score: {str(e)}")

if __name__ == "__main__":
    backfill_confidence_score() 