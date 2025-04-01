#!/usr/bin/env python3
"""
Script to backfill the Exit Trade column for completed trades in automated_trades.csv.
This adds exit times for trades that were closed before the column was added.
"""

import csv
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backfill_exit_trade.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def backfill_exit_trade():
    """Backfill Exit Trade column for completed trades"""
    try:
        # Read the CSV file
        trades = []
        csv_file_path = 'automated_trades.csv'
        
        # Check if file exists
        if not os.path.isfile(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return
            
        # Read the trades
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        # Track if we made any updates
        updated = False
        
        # Process trades
        for trade in trades:
            # Skip pending trades
            if trade['Outcome'] == 'PENDING':
                continue
                
            # For completed trades, use the trade's timestamp as the exit time
            if trade['Outcome'] in ['SUCCESS', 'STOP LOSS'] and trade['Timestamp']:
                try:
                    exit_time = trade['Timestamp']
                    logger.info(f"Trade {trade['No.']}: Setting exit time to {exit_time}")
                    trade['Exit Trade'] = exit_time
                    updated = True
                except (ValueError, KeyError) as e:
                    logger.error(f"Error setting exit time for trade {trade['No.']}: {str(e)}")
                    continue
        
        # Write updated trades back to CSV if any changes were made
        if updated:
            with open(csv_file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
            logger.info(f"Successfully backfilled Exit Trade values in {csv_file_path}")
        else:
            logger.info("No trades needed to be backfilled")
            
    except Exception as e:
        logger.error(f"Error backfilling exit trade data: {str(e)}")
        raise

if __name__ == "__main__":
    backfill_exit_trade() 