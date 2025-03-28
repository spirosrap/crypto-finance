"""
Script to update pending trades in automated_trades.csv based on current price.
This script checks if any pending trades have hit their take profit or stop loss levels.
"""

import csv
import os
from datetime import datetime
import logging
from coinbaseservice import CoinbaseService
from dotenv import load_dotenv
from config import API_KEY_PERPS, API_SECRET_PERPS
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pending_trades_updates.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_api_keys():
    """Load API keys from .env file"""
    load_dotenv()
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API keys not found in .env file")
    return api_key, api_secret

def get_current_btc_price(client):
    """Get current BTC price from Coinbase"""
    try:
        prices = client.get_btc_prices()
        if 'BTC-USDC' in prices:
            return float(prices['BTC-USDC']['ask'])
        elif 'BTC-EUR' in prices:
            return float(prices['BTC-EUR']['ask'])
        else:
            raise ValueError("No BTC price found in available pairs")
    except Exception as e:
        logger.error(f"Error getting current BTC price: {str(e)}")
        raise

def update_pending_trades():
    """Update pending trades based on current price"""
    try:
        # Load API keys and initialize Coinbase client
        api_key, api_secret = load_api_keys()
        client = CoinbaseService(api_key, api_secret)
        
        # Get current BTC price
        current_price = get_current_btc_price(client)
        logger.info(f"Current BTC price: {current_price}")
        
        # Read the CSV file
        trades = []
        with open('automated_trades.csv', 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        # Update pending trades
        updated = False
        for trade in trades:
            if trade['Outcome'] == 'PENDING':
                entry_price = float(trade['ENTRY'])
                take_profit = float(trade['Take Profit'])
                stop_loss = float(trade['Stop Loss'])
                
                # Check if price has hit take profit or stop loss
                if trade['SIDE'] == 'LONG':
                    if current_price >= take_profit:
                        trade['Outcome'] = 'SUCCESS'
                        trade['Outcome %'] = str(round(((take_profit - entry_price) / entry_price) * 100, 2))
                        updated = True
                        logger.info(f"Trade {trade['No.']} hit take profit at {current_price}")
                    elif current_price <= stop_loss:
                        trade['Outcome'] = 'STOP LOSS'
                        trade['Outcome %'] = str(round(((stop_loss - entry_price) / entry_price) * 100, 2))
                        updated = True
                        logger.info(f"Trade {trade['No.']} hit stop loss at {current_price}")
                else:  # SHORT
                    if current_price <= take_profit:
                        trade['Outcome'] = 'SUCCESS'
                        trade['Outcome %'] = str(round(((entry_price - take_profit) / entry_price) * 100, 2))
                        updated = True
                        logger.info(f"Trade {trade['No.']} hit take profit at {current_price}")
                    elif current_price >= stop_loss:
                        trade['Outcome'] = 'STOP LOSS'
                        trade['Outcome %'] = str(round(((entry_price - stop_loss) / entry_price) * 100, 2))
                        updated = True
                        logger.info(f"Trade {trade['No.']} hit stop loss at {current_price}")
        
        # Write updated trades back to CSV if any changes were made
        if updated:
            with open('automated_trades.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
            logger.info("Updated trades saved to automated_trades.csv")
        else:
            logger.info("No pending trades needed updating")
            
    except Exception as e:
        logger.error(f"Error updating pending trades: {str(e)}")
        raise

if __name__ == "__main__":
    update_pending_trades() 