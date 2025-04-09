"""
Script to update pending trades in automated_trades.csv based on current price.
This script checks if any pending trades have hit their take profit or stop loss levels.
It also tracks Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE).
"""

import csv
import os
import json
import hashlib
from datetime import datetime, UTC
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from coinbaseservice import CoinbaseService
from dotenv import load_dotenv
from config import API_KEY_PERPS, API_SECRET_PERPS
import time
import numpy as np
import talib
# Constants
STATE_FILE_PATH = 'data/trade_state_tracker.json'
CSV_FILE_PATH = 'automated_trades.csv'
LOG_FILE_PATH = 'logs/pending_trades_updates.log'
DEFAULT_GRANULARITY = "FIVE_MINUTE"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradeState:
    """Data class to hold trade state information"""
    entry_price: float
    side: str
    min_price: float
    max_price: float
    mae_pct: float
    mfe_pct: float
    leverage: float
    initialized: bool = True

def load_api_keys() -> Tuple[str, str]:
    """Load API keys from .env file"""
    load_dotenv()
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    if not api_key or not api_secret:
        raise ValueError("API keys not found in .env file")
    return api_key, api_secret

def get_current_btc_price_from_candles(client: CoinbaseService) -> float:
    """Get current BTC price from historical candles with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            # Get current timestamp and timestamp from 5 minutes ago
            current_time = int(datetime.now(UTC).timestamp())
            five_minutes_ago = current_time - 300  # 5 minutes in seconds
            
            # Get historical candles for the last 5 minutes
            candles = get_historical_candles(client, five_minutes_ago, current_time)
            
            if not candles:
                raise ValueError("No candle data available")
            
            # Get the most recent candle's closing price
            latest_candle = candles[-1]
            return float(latest_candle['close'])
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Error getting current BTC price from candles after {MAX_RETRIES} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed to get BTC price from candles: {str(e)}")
            time.sleep(RETRY_DELAY)

def get_historical_candles(
    client: CoinbaseService,
    start_timestamp: int,
    end_timestamp: int,
    granularity: str = DEFAULT_GRANULARITY
) -> List[Dict[str, Any]]:
    """Get historical candles for calculating MAE and MFE with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            product_id = "BTC-PERP-INTX"
            start_date = datetime.fromtimestamp(start_timestamp)
            end_date = datetime.fromtimestamp(end_timestamp)
            
            logger.info(f"Fetching historical candles from {start_date} to {end_date}")
            
            candles = client.historical_data.get_historical_data(
                product_id, 
                start_date, 
                end_date, 
                granularity
            )
            
            logger.info(f"Retrieved {len(candles)} candles")
            return candles
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Error getting historical candles after {MAX_RETRIES} attempts: {str(e)}")
                return []
            logger.warning(f"Attempt {attempt + 1} failed to get historical candles: {str(e)}")
            time.sleep(RETRY_DELAY)

def calculate_ema200(candles: List[Dict[str, Any]]) -> float:
    """Calculate EMA200 from candle data"""
    try:
        prices = [float(candle['close']) for candle in candles]
        ema200 = talib.EMA(np.array(prices), timeperiod=200)
        return ema200[-1]
    except Exception as e:
        logger.error(f"Error calculating EMA200: {str(e)}")
        return 0.0

def determine_trend_regime(current_price: float, ema200: float) -> str:
    """Determine trend regime based on price vs EMA200"""
    if current_price > ema200:
        return "Bullish"
    else:
        return "Bearish"

def generate_unique_trade_id(trade: Dict[str, Any]) -> str:
    """Generate a unique ID for each trade to avoid duplicates in state tracker"""
    try:
        if 'No.' in trade and trade['No.']:
            return str(trade['No.'])
        
        timestamp = trade.get('Timestamp', '')
        entry = trade.get('ENTRY', '0')
        side = trade.get('SIDE', '')
        leverage = trade.get('Leverage', '1x')
        
        unique_str = f"{timestamp}_{entry}_{side}_{leverage}"
        hash_obj = hashlib.md5(unique_str.encode())
        return hash_obj.hexdigest()[:12]
    except Exception as e:
        logger.error(f"Error generating unique trade ID: {str(e)}")
        return datetime.now(UTC).strftime("%Y%m%d%H%M%S")

def load_state_tracker() -> Dict[str, Dict[str, Any]]:
    """Load the persistent state tracker for MAE/MFE"""
    try:
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

def save_state_tracker(state_data: Dict[str, Dict[str, Any]]) -> None:
    """Save the persistent state tracker for MAE/MFE"""
    try:
        os.makedirs(os.path.dirname(STATE_FILE_PATH), exist_ok=True)
        
        with open(STATE_FILE_PATH, 'w') as f:
            json.dump(state_data, f, indent=2)
        logger.info(f"Saved state tracker with {len(state_data)} trades")
    except Exception as e:
        logger.error(f"Error saving state tracker: {str(e)}")

def reset_state_for_trade(
    unique_id: str,
    trade_state: Dict[str, Any],
    entry_price: float,
    side: str,
    leverage: float
) -> Dict[str, Any]:
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

def update_mae_mfe(
    trade: Dict[str, Any],
    current_price: float,
    trade_state: Dict[str, Any]
) -> None:
    """Update MAE and MFE values for a trade"""
    entry = float(trade["ENTRY"])
    prev_mae = float(trade_state.get("mae_pct", 0.0))
    prev_mfe = float(trade_state.get("mfe_pct", 0.0))
    trade_id = trade.get("ID", "UNKNOWN")

    price_delta = (current_price - entry) / entry * 100

    if price_delta < 0:  # Adverse movement
        current_mae = abs(price_delta)
        if current_mae > prev_mae:
            trade_state["mae_pct"] = current_mae
            logger.info(f"[LIVE MAE SPIKE] Trade {trade_id}: MAE updated from {prev_mae:.2f}% to {current_mae:.2f}%")
            assert current_mae >= prev_mae, f"MAE regressed in trade {trade_id}"
    else:  # Favorable movement
        current_mfe = price_delta
        if current_mfe > prev_mfe:
            trade_state["mfe_pct"] = current_mfe
            logger.info(f"[LIVE MFE SPIKE] Trade {trade_id}: MFE updated from {prev_mfe:.2f}% to {current_mfe:.2f}%")
            assert current_mfe >= prev_mfe, f"MFE regressed in trade {trade_id}"

def process_closed_trade(
    trade: Dict[str, Any],
    client: CoinbaseService,
    entry_timestamp: int,
    exit_timestamp: int
) -> Dict[str, Any]:
    """Process a closed trade to calculate final MAE and MFE"""
    try:
        trade_no = trade['No.']
        entry_price = float(trade['ENTRY'])
        side = trade['SIDE']
        leverage = float(trade['Leverage'].replace('x', ''))
        
        logger.info(f"Processing closed trade {trade_no} from {trade['Timestamp']} to {trade['Exit Trade']}")
        
        # Parse timestamps and ensure they're in UTC
        try:
            # Parse entry timestamp
            entry_str = trade['Timestamp']
            if 'UTC' not in entry_str:
                entry_dt = datetime.strptime(entry_str, "%Y-%m-%d %H:%M:%S")
                entry_dt = entry_dt.astimezone(UTC)
            else:
                entry_dt = datetime.strptime(entry_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)
            
            # Parse exit timestamp
            exit_str = trade['Exit Trade']
            if 'UTC' not in exit_str:
                exit_dt = datetime.strptime(exit_str, "%Y-%m-%d %H:%M:%S")
                exit_dt = exit_dt.astimezone(UTC)
            else:
                exit_dt = datetime.strptime(exit_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)
            
            # Calculate duration in hours
            duration = exit_dt - entry_dt
            duration_hours = duration.total_seconds() / 3600
            trade['Duration'] = str(round(duration_hours, 2))
            
            logger.info(f"Closed trade {trade_no} duration: {trade['Duration']} hours (from {entry_dt} to {exit_dt})")
        except Exception as e:
            logger.error(f"Error calculating duration for trade {trade_no}: {str(e)}")
            # Fallback to simple calculation if parsing fails
            duration_hours = (exit_timestamp - entry_timestamp) / 3600
            trade['Duration'] = str(round(duration_hours, 2))
        
        candles = get_historical_candles(client, entry_timestamp, exit_timestamp)
        
        if not candles:
            logger.warning(f"No candle data available for closed trade {trade_no}. Skipping.")
            return trade
        
        min_price = entry_price
        max_price = entry_price
        
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
        
        trade['MAE'] = str(round(abs(mae_pct), 2))
        trade['MFE'] = str(round(mfe_pct, 2))
        
        logger.info(f"Updated closed trade {trade_no} with MAE={trade['MAE']}%, MFE={trade['MFE']}%")
        return trade
        
    except Exception as e:
        logger.error(f"Error processing closed trade {trade.get('No.', 'unknown')}: {str(e)}")
        return trade

def process_pending_trade(
    trade: Dict[str, Any],
    current_price: float,
    trade_state: Dict[str, Any],
    client: CoinbaseService,
    take_profit: float,
    stop_loss: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process a pending trade to update its state and check for exit conditions"""
    try:
        trade_no = trade['No.']
        entry_price = float(trade['ENTRY'])
        side = trade['SIDE']
        leverage = float(trade['Leverage'].replace('x', ''))
        
        # Get historical candles for EMA200 calculation
        try:
            # Parse the timestamp and ensure it's in UTC
            timestamp_str = trade['Timestamp']
            if 'UTC' not in timestamp_str:
                # If no timezone is specified, assume it's in local time
                entry_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                # Convert to UTC
                entry_dt = entry_dt.astimezone(UTC)
            else:
                # If UTC is specified, parse directly
                entry_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=UTC)
            
            trade_timestamp = int(entry_dt.timestamp())
        except ValueError as e:
            logger.error(f"Error parsing timestamp for trade {trade_no}: {str(e)}")
            # Fallback to current time if parsing fails
            trade_timestamp = int(datetime.now(UTC).timestamp())
        
        current_time = int(datetime.now(UTC).timestamp())
        current_dt = datetime.now(UTC)
        
        # Calculate duration in hours
        duration = current_dt - entry_dt
        duration_hours = duration.total_seconds() / 3600
        trade['Duration'] = str(round(duration_hours, 2))
        
        logger.info(f"Trade {trade_no} duration: {trade['Duration']} hours (from {entry_dt} to {current_dt})")
        
        # FIRST: Check current price against take profit and stop loss levels
        # This ensures we prioritize current price conditions over historical wicks
        if side == 'LONG':
            if current_price >= take_profit:
                trade['Outcome'] = 'SUCCESS'
                trade['Exit Trade'] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                trade['Exit Reason'] = 'TP HIT'
                trade['Outcome %'] = str(round(((take_profit - entry_price) / entry_price) * leverage * 100, 2))
                logger.info(f"Trade {trade_no} marked as SUCCESS - Take Profit hit at {current_price}")
                return trade, trade_state
            elif current_price <= stop_loss:
                trade['Outcome'] = 'STOP LOSS'
                trade['Exit Trade'] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                trade['Exit Reason'] = 'SL HIT'
                # For long positions, loss percentage is (stop_loss - entry) / entry * leverage
                loss_pct = ((stop_loss - entry_price) / entry_price) * 100 * leverage
                trade['Outcome %'] = str(round(loss_pct, 2))
                logger.info(f"Trade {trade_no} marked as STOP LOSS - Stop Loss hit at {current_price}")
                return trade, trade_state
        else:  # SHORT
            if current_price <= take_profit:
                trade['Outcome'] = 'SUCCESS'
                trade['Exit Trade'] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                trade['Exit Reason'] = 'TP HIT'
                # For short positions, profit percentage is (entry - take_profit) / entry * leverage
                profit_pct = ((entry_price - take_profit) / entry_price) * 100 * leverage
                trade['Outcome %'] = str(round(profit_pct, 2))
                logger.info(f"Trade {trade_no} marked as SUCCESS - Take Profit hit at {current_price}")
                return trade, trade_state
            elif current_price >= stop_loss:
                trade['Outcome'] = 'STOP LOSS'
                trade['Exit Trade'] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                trade['Exit Reason'] = 'SL HIT'
                # For short positions, loss percentage is (stop_loss - entry) / entry * leverage
                loss_pct = ((stop_loss - entry_price) / entry_price) * 100 * leverage * -1
                trade['Outcome %'] = str(round(loss_pct, 2))
                logger.info(f"Trade {trade_no} marked as STOP LOSS - Stop Loss hit at {current_price}")
                return trade, trade_state
        
        candles = get_historical_candles(client, trade_timestamp, current_time)
        
        # Calculate EMA200 and determine trend regime
        ema200 = calculate_ema200(candles)
        market_trend = determine_trend_regime(current_price, ema200)
        trade['Market Trend'] = market_trend
        
        # Check if entry price changed, if so reset state
        if abs(trade_state['entry_price'] - entry_price) > 0.01 or trade_state['side'] != side:
            trade_state = reset_state_for_trade(trade_no, trade_state, entry_price, side, leverage)
        
        # Update MAE/MFE based on current price
        update_mae_mfe(trade, current_price, trade_state)
        
        # Initialize MAE and MFE variables
        mae_pct = trade_state['mae_pct']
        mfe_pct = trade_state['mfe_pct']
        
        # SECOND: Process candles if available - only if current price didn't trigger an exit
        if candles:
            logger.info(f"Processing {len(candles)} candles for trade {trade_no}")
            
            if not trade_state.get('initialized', False):
                trade_state['min_price'] = entry_price
                trade_state['max_price'] = entry_price

            for candle in candles:
                high = float(candle['high'])
                low = float(candle['low'])

                if side == 'LONG':
                    if high >= take_profit:
                        trade['Exit Reason'] = 'TP HIT (wick)'
                        trade['Exit Trade'] = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
                        trade['Outcome'] = 'SUCCESS'
                        trade['Outcome %'] = str(round(((take_profit - entry_price) / entry_price) * leverage * 100, 2))
                        return trade, trade_state
                    elif low <= stop_loss:
                        trade['Exit Reason'] = 'SL HIT (wick)'
                        trade['Exit Trade'] = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
                        trade['Outcome'] = 'STOP LOSS'
                        # For long positions, loss percentage is (stop_loss - entry) / entry * leverage
                        loss_pct = ((stop_loss - entry_price) / entry_price) * 100 * leverage
                        trade['Outcome %'] = str(round(loss_pct, 2))
                        return trade, trade_state
                else:  # SHORT
                    if low <= take_profit:
                        trade['Exit Reason'] = 'TP HIT (wick)'
                        trade['Exit Trade'] = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
                        trade['Outcome'] = 'SUCCESS'
                        # For short positions, profit percentage is (entry - take_profit) / entry * leverage
                        profit_pct = ((entry_price - take_profit) / entry_price) * 100 * leverage
                        trade['Outcome %'] = str(round(profit_pct, 2))
                        return trade, trade_state
                    elif high >= stop_loss:
                        trade['Exit Reason'] = 'SL HIT (wick)'
                        trade['Exit Trade'] = datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
                        trade['Outcome'] = 'STOP LOSS'
                        # For short positions, loss percentage is (stop_loss - entry) / entry * leverage
                        loss_pct = ((stop_loss - entry_price) / entry_price) * 100 * leverage * -1
                        trade['Outcome %'] = str(round(loss_pct, 2))
                        return trade, trade_state                
            
            for candle in candles:
                candle_low = float(candle['low'])
                candle_high = float(candle['high'])
                
                if side == 'LONG':
                    if candle_low < trade_state['min_price']:
                        trade_state['min_price'] = candle_low
                    if candle_high > trade_state['max_price']:
                        trade_state['max_price'] = candle_high
                else:  # SHORT
                    if candle_high > trade_state['max_price']:
                        trade_state['max_price'] = candle_high
                    if candle_low < trade_state['min_price']:
                        trade_state['min_price'] = candle_low
        
        # Check if current price creates new min/max
        if current_price < trade_state['min_price']:
            trade_state['min_price'] = current_price
        if current_price > trade_state['max_price']:
            trade_state['max_price'] = current_price
        
        # Calculate final MAE and MFE
        if side == 'LONG':
            mae_price_diff = entry_price - trade_state['min_price']
            mae_pct = (mae_price_diff / entry_price) * 100
            mfe_price_diff = trade_state['max_price'] - entry_price
            mfe_pct = (mfe_price_diff / entry_price) * 100
        else:  # SHORT
            mae_price_diff = trade_state['max_price'] - entry_price
            mae_pct = (mae_price_diff / entry_price) * 100 * -1
            mfe_price_diff = entry_price - trade_state['min_price']
            mfe_pct = (mfe_price_diff / entry_price) * 100
        
        # Update trade with calculated values
        trade['MAE'] = str(round(abs(mae_pct), 2))
        trade['MFE'] = str(round(mfe_pct, 2))
        
        return trade, trade_state
        
    except Exception as e:
        logger.error(f"Error processing pending trade {trade.get('No.', 'unknown')}: {str(e)}")
        return trade, trade_state

def update_pending_trades() -> None:
    """Update pending trades based on current price"""
    try:
        # Load API keys and initialize Coinbase client
        api_key, api_secret = load_api_keys()
        client = CoinbaseService(api_key, api_secret)
        
        # Get current BTC price from historical candles
        current_price = get_current_btc_price_from_candles(client)
        logger.info(f"Current BTC price: {current_price}")
        
        # Load persistent state tracker
        state_tracker = load_state_tracker()
        
        # Read the CSV file
        trades = []
        file_exists = os.path.isfile(CSV_FILE_PATH)
        headers = []
        
        if file_exists:
            with open(CSV_FILE_PATH, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
        
        # Check if MAE, MFE, and Market Trend columns exist
        mae_index = headers.index('MAE') if 'MAE' in headers else -1
        mfe_index = headers.index('MFE') if 'MFE' in headers else -1
        exit_trade_index = headers.index('Exit Trade') if 'Exit Trade' in headers else -1
        trend_regime_index = headers.index('Market Trend') if 'Market Trend' in headers else -1
        duration_index = headers.index('Duration') if 'Duration' in headers else -1
        
        # Read the trades
        with open(CSV_FILE_PATH, 'r') as f:
            reader = csv.DictReader(f)
            trades = list(reader)
        
        # Add MAE, MFE, and Market Trend fields if they don't exist
        if mae_index == -1 or mfe_index == -1 or exit_trade_index == -1 or trend_regime_index == -1 or duration_index == -1:
            for trade in trades:
                if 'MAE' not in trade:
                    trade['MAE'] = '0.0'
                if 'MFE' not in trade:
                    trade['MFE'] = '0.0'
                if 'Exit Trade' not in trade:
                    trade['Exit Trade'] = ''
                if 'Market Trend' not in trade:
                    trade['Market Trend'] = ''
                if 'Duration' not in trade:
                    trade['Duration'] = ''
        
        # Process closed trades first
        updated = False
        for trade in trades:
            if trade['Outcome'] in ['SUCCESS', 'STOP LOSS'] and trade['Exit Trade']:
                try:
                    # Check if we need to update this trade
                    needs_update = False
                    
                    # Skip if MAE and MFE are already calculated
                    if trade['MAE'] and trade['MAE'] != '0.0' and trade['MFE'] and trade['MFE'] != '0.0':
                        # Only skip if Duration is also already calculated
                        if trade['Duration'] and trade['Duration'] != '':
                            continue
                        else:
                            needs_update = True
                    else:
                        needs_update = True
                    
                    if needs_update:
                        try:
                            entry_timestamp = int(datetime.strptime(trade['Timestamp'], "%Y-%m-%d %H:%M:%S").timestamp())
                            exit_timestamp = int(datetime.strptime(trade['Exit Trade'], "%Y-%m-%d %H:%M:%S UTC").timestamp())
                        except ValueError:
                            # Try with UTC suffix if the first attempt fails
                            entry_timestamp = int(datetime.strptime(trade['Timestamp'], "%Y-%m-%d %H:%M:%S UTC").timestamp())
                            exit_timestamp = int(datetime.strptime(trade['Exit Trade'], "%Y-%m-%d %H:%M:%S UTC").timestamp())
                        
                        trade = process_closed_trade(trade, client, entry_timestamp, exit_timestamp)
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
            if trade['Outcome'] == 'PENDING':
                try:
                    trade_no = trade['No.']
                    unique_id = trade_to_unique_id[trade_no]
                    take_profit = float(trade['Take Profit'])
                    stop_loss = float(trade['Stop Loss'])
                    
                    if not take_profit or not stop_loss:
                        logger.warning(f"Trade {trade_no} (ID: {unique_id}) has missing Take Profit or Stop Loss values. Skipping update.")
                        continue
                    
                    # Initialize or get state tracking for this trade
                    if unique_id not in state_tracker:
                        state_tracker[unique_id] = {
                            'entry_price': float(trade['ENTRY']),
                            'side': trade['SIDE'],
                            'min_price': float(trade['ENTRY']),
                            'max_price': float(trade['ENTRY']),
                            'mae_pct': 0.0,
                            'mfe_pct': 0.0,
                            'leverage': float(trade['Leverage'].replace('x', '')),
                            'initialized': True
                        }
                    
                    trade, state_tracker[unique_id] = process_pending_trade(
                        trade,
                        current_price,
                        state_tracker[unique_id],
                        client,
                        take_profit,
                        stop_loss
                    )
                    updated = True
                    
                except Exception as e:
                    logger.error(f"Error processing pending trade {trade.get('No.', 'unknown')}: {str(e)}")
                    continue
        
        # Write updated trades back to CSV
        if updated:
            if mae_index == -1 or mfe_index == -1 or exit_trade_index == -1 or trend_regime_index == -1 or duration_index == -1:
                all_fields = list(trades[0].keys())
                with open(CSV_FILE_PATH, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=all_fields)
                    writer.writeheader()
                    writer.writerows(trades)
            else:
                with open(CSV_FILE_PATH, 'w', newline='') as f:
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