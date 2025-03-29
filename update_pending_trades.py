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
        
        # Update only pending trades
        updated = False
        for trade in trades:
            # Only process pending trades
            if trade['Outcome'] == 'PENDING':
                try:
                    entry_price = float(trade['ENTRY'])
                    take_profit = trade['Take Profit']
                    stop_loss = trade['Stop Loss']
                    
                    # Validate take profit and stop loss values
                    if not take_profit or not stop_loss:
                        logger.warning(f"Trade {trade['No.']} has missing Take Profit or Stop Loss values. Skipping update.")
                        continue
                        
                    take_profit = float(take_profit)
                    stop_loss = float(stop_loss)
                    
                    # Log trade details for debugging
                    logger.info(f"Processing Trade {trade['No.']}:")
                    logger.info(f"Entry: {entry_price}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")
                    logger.info(f"Current Price: {current_price}")
                    
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
                        # Check if price has hit take profit or stop loss
                        if current_price >= take_profit:
                            trade['Outcome'] = 'SUCCESS'
                            logger.info(f"Trade {trade['No.']} marked as SUCCESS - Take Profit hit at {current_price}")
                        elif current_price <= stop_loss:
                            trade['Outcome'] = 'STOP LOSS'
                            logger.info(f"Trade {trade['No.']} marked as STOP LOSS - Stop Loss hit at {current_price}")
                        else:
                            logger.info(f"Trade {trade['No.']} still pending - Price {current_price} between TP {take_profit} and SL {stop_loss}")
                    else:  # SHORT
                        profit_loss_usd = position_size_usd - current_value
                        # Check if price has hit take profit or stop loss
                        if current_price <= take_profit:
                            trade['Outcome'] = 'SUCCESS'
                            logger.info(f"Trade {trade['No.']} marked as SUCCESS - Take Profit hit at {current_price}")
                        elif current_price >= stop_loss:
                            trade['Outcome'] = 'STOP LOSS'
                            logger.info(f"Trade {trade['No.']} marked as STOP LOSS - Stop Loss hit at {current_price}")
                        else:
                            logger.info(f"Trade {trade['No.']} still pending - Price {current_price} between TP {take_profit} and SL {stop_loss}")
                    
                    # Only update outcome percentage if trade is closed
                    if trade['Outcome'] != 'PENDING':
                        profit_loss_percentage = (profit_loss_usd / margin) * 100
                        trade['Outcome %'] = str(round(profit_loss_percentage, 2))
                        updated = True
                except ValueError as e:
                    logger.error(f"Error processing trade {trade['No.']}: {str(e)}")
                    continue
        
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
        logger.error(f"Error updating trades: {str(e)}")
        raise

if __name__ == "__main__":
    update_pending_trades() 