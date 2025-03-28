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
        
        # Update all trades
        updated = False
        for trade in trades:
            entry_price = float(trade['ENTRY'])
            margin = float(trade['Margin'])
            leverage = float(trade['Leverage'].replace('x', ''))
            
            # Calculate position size in BTC
            position_size_usd = margin * leverage
            position_size_btc = position_size_usd / entry_price
            
            # Calculate current value
            current_value = position_size_btc * current_price
            
            # Calculate profit/loss in USD
            if trade['SIDE'] == 'LONG':
                profit_loss_usd = current_value - position_size_usd
            else:  # SHORT
                profit_loss_usd = position_size_usd - current_value
            
            # Calculate profit/loss percentage based on margin
            profit_loss_percentage = (profit_loss_usd / margin) * 100
            
            # Update outcome if pending
            if trade['Outcome'] == 'PENDING':
                if profit_loss_percentage > 0:
                    trade['Outcome'] = 'SUCCESS'
                    logger.info(f"Trade {trade['No.']} marked as SUCCESS with {profit_loss_percentage:.2f}% profit on margin")
                else:
                    trade['Outcome'] = 'STOP LOSS'
                    logger.info(f"Trade {trade['No.']} marked as STOP LOSS with {profit_loss_percentage:.2f}% loss on margin")
            
            # Always update the outcome percentage
            trade['Outcome %'] = str(round(profit_loss_percentage, 2))
            updated = True
        
        # Write updated trades back to CSV if any changes were made
        if updated:
            with open('automated_trades.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
            logger.info("Updated trades saved to automated_trades.csv")
        else:
            logger.info("No trades needed updating")
            
    except Exception as e:
        logger.error(f"Error updating trades: {str(e)}")
        raise

if __name__ == "__main__":
    update_pending_trades() 