"""
Script to update pending trades in automated_trades.csv based on current price.
This script checks if any pending trades have hit their take profit or stop loss levels.
It also tracks Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
"""

import csv
import os
import json
import hashlib
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

# Path for state tracking file
STATE_FILE_PATH = 'data/trade_state_tracker.json'

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

def get_historical_candles(client, start_timestamp, end_timestamp, granularity="FIVE_MINUTE"):
    """Get historical candles for calculating MAE and MFE"""
    try:
        product_id = "BTC-USDC"
        # Convert timestamps to datetime objects
        start_date = datetime.fromtimestamp(start_timestamp)
        end_date = datetime.fromtimestamp(end_timestamp)
        
        logger.info(f"Fetching historical candles from {start_date} to {end_date}")
        
        # Use the historical_data service to get candles
        candles = client.historical_data.get_historical_data(
            product_id, 
            start_date, 
            end_date, 
            granularity
        )
        
        logger.info(f"Retrieved {len(candles)} candles")
        return candles
    except Exception as e:
        logger.error(f"Error getting historical candles: {str(e)}")
        return []

def generate_unique_trade_id(trade):
    """Generate a unique ID for each trade to avoid duplicates in state tracker"""
    try:
        # Use trade number if available as primary key
        if 'No.' in trade and trade['No.']:
            return str(trade['No.'])
        
        # As fallback, generate a hash from timestamp + entry + side
        timestamp = trade.get('Timestamp', '')
        entry = trade.get('ENTRY', '0')
        side = trade.get('SIDE', '')
        leverage = trade.get('Leverage', '1x')
        
        # Create a unique string combining key trade elements
        unique_str = f"{timestamp}_{entry}_{side}_{leverage}"
        
        # Generate a hash to use as ID
        hash_obj = hashlib.md5(unique_str.encode())
        return hash_obj.hexdigest()[:12]  # Use first 12 chars of hash as ID
    except Exception as e:
        logger.error(f"Error generating unique trade ID: {str(e)}")
        # Last resort fallback
        return datetime.now().strftime("%Y%m%d%H%M%S")

def load_state_tracker():
    """Load the persistent state tracker for MAE/MFE"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        
        if os.path.exists(STATE_FILE_PATH):
            with open(STATE_FILE_PATH, 'r') as f:
                state_data = json.load(f)
                logger.info(f"Loaded state tracker with {len(state_data)} trades")
                return state_data
        else:
            logger.info("No existing state tracker found, creating new one")
            return {}
    except Exception as e:
        logger.error(f"Error loading state tracker: {str(e)}")
        return {}

def save_state_tracker(state_data):
    """Save the persistent state tracker for MAE/MFE"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state_data, f, indent=2)
        logger.info(f"Saved state tracker with {len(state_data)} trades")
    except Exception as e:
        logger.error(f"Error saving state tracker: {str(e)}")

def reset_state_for_trade(unique_id, trade_state, entry_price, side, leverage):
    """Reset the state for a trade to recalculate MAE/MFE from scratch"""
    trade_state['entry_price'] = entry_price
    trade_state['side'] = side
    trade_state['min_price'] = entry_price
    trade_state['max_price'] = entry_price
    trade_state['mae_pct'] = 0.0
    trade_state['mfe_pct'] = 0.0
    trade_state['leverage'] = leverage
    trade_state['initialized'] = True
    logger.info(f"Reset state for trade {unique_id} to recalculate from entry")
    return trade_state

def update_pending_trades():
    """Update pending trades based on current price"""
    try:
        # Load API keys and initialize Coinbase client
        api_key, api_secret = load_api_keys()
        client = CoinbaseService(api_key, api_secret)
        
        # Get current BTC price
        current_price = get_current_btc_price(client)
        logger.info(f"Current BTC price: {current_price}")
        
        # Get current time
        current_time = int(datetime.now().timestamp())
        
        # Load persistent state tracker
        state_tracker = load_state_tracker()
        
        # Read the CSV file
        trades = []
        csv_file_path = 'automated_trades.csv'
        
        # Check if file exists and read headers
        file_exists = os.path.isfile(csv_file_path)
        headers = []
        
        if file_exists:
            with open(csv_file_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
        
        # Check if MAE and MFE columns exist, if not, need to add them
        mae_index = -1
        mfe_index = -1
        exit_trade_index = -1
        
        if 'MAE' in headers:
            mae_index = headers.index('MAE')
        if 'MFE' in headers:
            mfe_index = headers.index('MFE')
        if 'Exit Trade' in headers:
            exit_trade_index = headers.index('Exit Trade')
        
        # Read the trades
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        # Add MAE and MFE fields if they don't exist
        if mae_index == -1 or mfe_index == -1 or exit_trade_index == -1:
            for trade in trades:
                if 'MAE' not in trade:
                    trade['MAE'] = '0.0'
                if 'MFE' not in trade:
                    trade['MFE'] = '0.0'
                if 'Exit Trade' not in trade:
                    trade['Exit Trade'] = ''
        
        # Process closed trades first
        updated = False
        for trade in trades:
            if trade['Outcome'] in ['SUCCESS', 'STOP LOSS'] and trade['Exit Trade']:
                try:
                    # Skip if MAE and MFE are already calculated (not zero or missing)
                    if trade['MAE'] and trade['MAE'] != '0.0' and trade['MFE'] and trade['MFE'] != '0.0':
                        continue

                    trade_no = trade['No.']
                    entry_price = float(trade['ENTRY'])
                    side = trade['SIDE']
                    leverage = float(trade['Leverage'].replace('x', ''))
                    
                    # Get entry and exit timestamps
                    entry_timestamp = int(datetime.strptime(trade['Timestamp'], "%Y-%m-%d %H:%M:%S").timestamp())
                    exit_timestamp = int(datetime.strptime(trade['Exit Trade'], "%Y-%m-%d %H:%M:%S").timestamp())
                    
                    logger.info(f"Processing closed trade {trade_no} from {trade['Timestamp']} to {trade['Exit Trade']}")
                    
                    # Get historical candles for the trade period
                    candles = get_historical_candles(client, entry_timestamp, exit_timestamp)
                    
                    if not candles:
                        logger.warning(f"No candle data available for closed trade {trade_no}. Skipping.")
                        continue
                    
                    # Initialize min and max prices with entry price
                    min_price = entry_price
                    max_price = entry_price
                    
                    # Process candles to find min and max prices
                    for candle in candles:
                        candle_low = float(candle['low'])
                        candle_high = float(candle['high'])
                        
                        if side == 'LONG':
                            if candle_low < min_price:
                                min_price = candle_low
                            if candle_high > max_price:
                                max_price = candle_high
                        else:  # SHORT
                            if candle_high > max_price:
                                max_price = candle_high
                            if candle_low < min_price:
                                min_price = candle_low
                    
                    # Calculate MAE and MFE based on trade direction
                    if side == 'LONG':
                        mae_price_diff = entry_price - min_price
                        mae_pct = (mae_price_diff / entry_price) * 100
                        mfe_price_diff = max_price - entry_price
                        mfe_pct = (mfe_price_diff / entry_price) * 100
                    else:  # SHORT
                        mae_price_diff = max_price - entry_price
                        mae_pct = (mae_price_diff / entry_price) * 100 * -1
                        mfe_price_diff = entry_price - min_price
                        mfe_pct = (mfe_price_diff / entry_price) * 100
                    
                    # Update trade with calculated values
                    trade['MAE'] = str(round(abs(mae_pct), 2))
                    trade['MFE'] = str(round(mfe_pct, 2))
                    
                    logger.info(f"Updated closed trade {trade_no} with MAE={trade['MAE']}%, MFE={trade['MFE']}%")
                    updated = True
                    
                except Exception as e:
                    logger.error(f"Error processing closed trade {trade.get('No.', 'unknown')}: {str(e)}")
                    continue

        # Generate unique IDs for all trades
        trade_to_unique_id = {}
        for trade in trades:
            unique_id = generate_unique_trade_id(trade)
            trade_to_unique_id[trade['No.']] = unique_id
            
        # Clean up state tracker - remove closed trades
        active_trade_ids = set(trade_to_unique_id[trade['No.']] for trade in trades if trade['Outcome'] == 'PENDING')
        closed_trade_ids = [trade_id for trade_id in state_tracker.keys() if trade_id not in active_trade_ids]
        for trade_id in closed_trade_ids:
            del state_tracker[trade_id]
            logger.info(f"Removed closed trade {trade_id} from state tracker")
        
        # Update only pending trades
        for trade in trades:
            # Only process pending trades
            if trade['Outcome'] == 'PENDING':
                try:
                    trade_no = trade['No.']
                    unique_id = trade_to_unique_id[trade_no]
                    entry_price = float(trade['ENTRY'])
                    take_profit = trade['Take Profit']
                    stop_loss = trade['Stop Loss']
                    
                    # Validate take profit and stop loss values
                    if not take_profit or not stop_loss:
                        logger.warning(f"Trade {trade_no} (ID: {unique_id}) has missing Take Profit or Stop Loss values. Skipping update.")
                        continue
                        
                    take_profit = float(take_profit)
                    stop_loss = float(stop_loss)
                    
                    # Log trade details for debugging
                    logger.info(f"Processing Trade {trade_no} (ID: {unique_id}):")
                    logger.info(f"Entry: {entry_price}, Take Profit: {take_profit}, Stop Loss: {stop_loss}")
                    logger.info(f"Current Price: {current_price}")
                    
                    margin = float(trade['Margin'])
                    leverage = float(trade['Leverage'].replace('x', ''))
                    side = trade['SIDE']
                    
                    # Get timestamp from the trade (entry time)
                    trade_timestamp_str = trade['Timestamp']
                    trade_timestamp = int(datetime.strptime(trade_timestamp_str, "%Y-%m-%d %H:%M:%S").timestamp())
                    
                    # Initialize or get state tracking for this trade
                    if unique_id not in state_tracker:
                        state_tracker[unique_id] = {
                            'entry_price': entry_price,
                            'side': side,
                            'min_price': entry_price,  # Track actual min price
                            'max_price': entry_price,  # Track actual max price
                            'mae_pct': 0.0,  # Percentage
                            'mfe_pct': 0.0,  # Percentage
                            'leverage': leverage,
                            'initialized': True  # Flag to indicate we've fully initialized the trade
                        }
                    
                    trade_state = state_tracker[unique_id]
                    
                    # Check if entry price changed, if so reset state
                    if abs(trade_state['entry_price'] - entry_price) > 0.01 or trade_state['side'] != side:
                        trade_state = reset_state_for_trade(unique_id, trade_state, entry_price, side, leverage)
                    
                    # Always get historical candles from entry time to present
                    # This ensures we capture all price movement since trade began
                    candles = get_historical_candles(client, trade_timestamp, current_time)
                    
                    # Calculate position size in BTC
                    position_size_usd = margin * leverage
                    position_size_btc = position_size_usd / entry_price
                    
                    # Calculate current value
                    current_value = position_size_btc * current_price
                    
                    # Get current min and max prices from state
                    min_price = trade_state['min_price']
                    max_price = trade_state['max_price']
                    
                    # Initialize MAE and MFE variables
                    mae_pct = 0.0
                    mfe_pct = 0.0
                    
                    # Check if we have meaningful candle data
                    if not candles:
                        logger.warning(f"No candle data available for trade {trade_no}. Using current price only.")
                        # If no candles, just use current price to update extremes
                        if current_price < min_price:
                            min_price = current_price
                        if current_price > max_price:
                            max_price = current_price
                    else:
                        logger.info(f"Processing {len(candles)} candles for trade {trade_no}")
                        
                        # Reset min/max only if we're refetching all data
                        if not trade_state.get('initialized', False):
                            min_price = entry_price
                            max_price = entry_price
                        
                        # Process based on trade direction
                        if side == 'LONG':
                            # Update min and max based on candles
                            for candle in candles:
                                candle_low = float(candle['low'])
                                candle_high = float(candle['high'])
                                
                                # Log candle data for debugging
                                logger.debug(f"Candle: Low={candle_low}, High={candle_high}")
                                
                                # Check for new minimum price (worst case for LONG)
                                if candle_low < min_price:
                                    min_price = candle_low
                                    logger.debug(f"New min price: {min_price}")
                                
                                # Check for new maximum price (best case for LONG)
                                if candle_high > max_price:
                                    max_price = candle_high
                                    logger.debug(f"New max price: {max_price}")
                            
                            # Check if current price creates new min/max
                            if current_price < min_price:
                                min_price = current_price
                                logger.debug(f"Current price creates new min: {min_price}")
                            if current_price > max_price:
                                max_price = current_price
                                logger.debug(f"Current price creates new max: {max_price}")
                            
                            # Calculate MAE and MFE based on price differences first, then convert to percentages
                            # MAE - Maximum Adverse Excursion (worst drawdown)
                            mae_price_diff = entry_price - min_price  # Positive value means price went below entry
                            mae_pct = (mae_price_diff / entry_price) * 100
                            
                            # MFE - Maximum Favorable Excursion (highest point reached)
                            mfe_price_diff = max_price - entry_price  # Positive value means price went above entry
                            mfe_pct = (mfe_price_diff / entry_price) * 100
                                
                            # Check if price has hit take profit or stop loss
                            if current_price >= take_profit:
                                # Calculate final MAE and MFE before marking as closed
                                mae_price_diff = entry_price - min_price
                                mae_pct = (mae_price_diff / entry_price) * 100
                                mfe_price_diff = max_price - entry_price
                                mfe_pct = (mfe_price_diff / entry_price) * 100
                                
                                # Update final values in trade dict
                                trade['MAE'] = str(round(abs(mae_pct), 2))
                                trade['MFE'] = str(round(mfe_pct, 2))
                                
                                trade['Outcome'] = 'SUCCESS'
                                trade['Exit Trade'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                logger.info(f"Trade {trade_no} (ID: {unique_id}) marked as SUCCESS - Take Profit hit at {current_price}")
                            elif current_price <= stop_loss:
                                # Calculate final MAE and MFE before marking as closed
                                mae_price_diff = entry_price - min_price
                                mae_pct = (mae_price_diff / entry_price) * 100
                                mfe_price_diff = max_price - entry_price
                                mfe_pct = (mfe_price_diff / entry_price) * 100
                                
                                # Update final values in trade dict
                                trade['MAE'] = str(round(abs(mae_pct), 2))
                                trade['MFE'] = str(round(mfe_pct, 2))
                                
                                trade['Outcome'] = 'STOP LOSS'
                                trade['Exit Trade'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                logger.info(f"Trade {trade_no} (ID: {unique_id}) marked as STOP LOSS - Stop Loss hit at {current_price}")
                            else:
                                logger.info(f"Trade {trade_no} (ID: {unique_id}) still pending - Price {current_price} between TP {take_profit} and SL {stop_loss}")
                        else:  # SHORT
                            # Update min and max based on candles
                            for candle in candles:
                                candle_low = float(candle['low'])
                                candle_high = float(candle['high'])
                                
                                # Log candle data for debugging
                                logger.debug(f"Candle: Low={candle_low}, High={candle_high}")
                                
                                # For shorts, high prices are bad (worst case)
                                if candle_high > max_price:
                                    max_price = candle_high
                                    logger.debug(f"New max price: {max_price}")
                                
                                # For shorts, low prices are good (best case)
                                if candle_low < min_price:
                                    min_price = candle_low
                                    logger.debug(f"New min price: {min_price}")
                            
                            # Check if current price creates new min/max
                            if current_price < min_price:
                                min_price = current_price
                                logger.debug(f"Current price creates new min: {min_price}")
                            if current_price > max_price:
                                max_price = current_price
                                logger.debug(f"Current price creates new max: {max_price}")
                            
                            # Calculate MAE and MFE based on price differences first, then convert to percentages
                            # For shorts, MAE is when price goes above entry
                            mae_price_diff = max_price - entry_price  # Positive value means price went above entry (bad for shorts)
                            mae_pct = (mae_price_diff / entry_price) * 100 * -1  # Negative percentage for shorts
                            
                            # For shorts, MFE is when price goes below entry
                            mfe_price_diff = entry_price - min_price  # Positive value means price went below entry (good for shorts)
                            mfe_pct = (mfe_price_diff / entry_price) * 100
                            
                            # Check if price has hit take profit or stop loss
                            if current_price <= take_profit:
                                # Calculate final MAE and MFE before marking as closed
                                mae_price_diff = max_price - entry_price
                                mae_pct = (mae_price_diff / entry_price) * 100 * -1
                                mfe_price_diff = entry_price - min_price
                                mfe_pct = (mfe_price_diff / entry_price) * 100
                                
                                # Update final values in trade dict
                                trade['MAE'] = str(round(abs(mae_pct), 2))
                                trade['MFE'] = str(round(mfe_pct, 2))
                                
                                trade['Outcome'] = 'SUCCESS'
                                trade['Exit Trade'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                logger.info(f"Trade {trade_no} (ID: {unique_id}) marked as SUCCESS - Take Profit hit at {current_price}")
                            elif current_price >= stop_loss:
                                # Calculate final MAE and MFE before marking as closed
                                mae_price_diff = max_price - entry_price
                                mae_pct = (mae_price_diff / entry_price) * 100 * -1
                                mfe_price_diff = entry_price - min_price
                                mfe_pct = (mfe_price_diff / entry_price) * 100
                                
                                # Update final values in trade dict
                                trade['MAE'] = str(round(abs(mae_pct), 2))
                                trade['MFE'] = str(round(mfe_pct, 2))
                                
                                trade['Outcome'] = 'STOP LOSS'
                                trade['Exit Trade'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                logger.info(f"Trade {trade_no} (ID: {unique_id}) marked as STOP LOSS - Stop Loss hit at {current_price}")
                            else:
                                logger.info(f"Trade {trade_no} (ID: {unique_id}) still pending - Price {current_price} between TP {take_profit} and SL {stop_loss}")
                    
                    # Log the MAE/MFE values being calculated
                    logger.info(f"Trade {trade_no}: Min Price={min_price}, Max Price={max_price}")
                    logger.info(f"Trade {trade_no}: MAE={mae_pct:.2f}%, MFE={mfe_pct:.2f}%")
                    
                    # Update state tracker with new values
                    trade_state['min_price'] = min_price
                    trade_state['max_price'] = max_price
                    trade_state['mae_pct'] = mae_pct
                    trade_state['mfe_pct'] = mfe_pct
                    
                    # Update MAE and MFE values in the trade dict (rounded for display)
                    # Convert to absolute value for display clarity
                    trade['MAE'] = str(round(abs(mae_pct), 2))
                    trade['MFE'] = str(round(mfe_pct, 2))
                    
                    # Only update outcome percentage if trade is closed
                    if trade['Outcome'] != 'PENDING':
                        # Calculate Outcome % based on the new formula
                        outcome_percentage = ((take_profit - entry_price) / entry_price) * leverage * 100 if trade['Outcome'] == 'SUCCESS' else ((entry_price - stop_loss) / entry_price) * leverage * -100
                        trade['Outcome %'] = str(round(outcome_percentage, 2))
                   
                except ValueError as e:
                    logger.error(f"Error processing trade {trade.get('No.', 'unknown')}: {str(e)}")
                    continue
        
        # Write updated trades back to CSV with new headers if needed
        if updated:
            # Update headers if MAE or MFE columns were added
            if mae_index == -1 or mfe_index == -1 or exit_trade_index == -1:
                all_fields = list(trades[0].keys())
                with open(csv_file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=all_fields)
                    writer.writeheader()
                    writer.writerows(trades)
            else:
                with open(csv_file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                    writer.writeheader()
                    writer.writerows(trades)
            logger.info("Updated trades saved to automated_trades.csv")
        else:
            logger.info("No pending trades needed updating")
        
        # Save the state tracker
        save_state_tracker(state_tracker)
            
    except Exception as e:
        logger.error(f"Error updating trades: {str(e)}")
        raise

if __name__ == "__main__":
    update_pending_trades() 