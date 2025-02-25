#!/usr/bin/env python3
"""
Order Monitor - Continuously monitors open orders until all are closed.

This script checks for open orders on Coinbase and provides status updates
until all orders are closed. It can be configured to check at specific intervals
and provides detailed information about each open order.
"""

import os
import time
import json
import argparse
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from coinbaseservice import CoinbaseService
import uuid
import dateutil.parser as parser

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see all log messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitor_orders.log"),
        logging.StreamHandler()
    ]
)

# ANSI color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def extract_value(obj, default=0):
    """
    Extract value from nested dictionaries.
    
    Args:
        obj: The object to extract value from
        default: Default value to return if extraction fails
        
    Returns:
        Extracted value or default
    """
    if isinstance(obj, dict):
        # Try to get value from rawCurrency or userNativeCurrency
        if 'rawCurrency' in obj and isinstance(obj['rawCurrency'], dict) and 'value' in obj['rawCurrency']:
            return obj['rawCurrency']['value']
        elif 'userNativeCurrency' in obj and isinstance(obj['userNativeCurrency'], dict) and 'value' in obj['userNativeCurrency']:
            return obj['userNativeCurrency']['value']
        elif 'value' in obj:
            return obj['value']
    return default

def ensure_dict(obj):
    """
    Convert an object to a dictionary if it's not already.
    
    Args:
        obj: The object to convert
        
    Returns:
        Dictionary representation of the object
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, '__dict__'):
        return vars(obj)
    elif hasattr(obj, '_asdict'):  # For namedtuples
        return obj._asdict()
    else:
        # Try json serialization as a fallback
        try:
            return json.loads(json.dumps(obj, default=lambda x: str(x)))
        except:
            # If all else fails, create a minimal dict
            result = {}
            for attr in dir(obj):
                if not attr.startswith('_') and not callable(getattr(obj, attr)):
                    result[attr] = getattr(obj, attr)
            return result

def safe_get(obj, key, default=None):
    """
    Safely get an attribute from an object or dictionary.
    
    Args:
        obj: The object or dictionary to get the attribute from
        key: The attribute name
        default: Default value to return if attribute is not found
        
    Returns:
        The attribute value or default
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    elif hasattr(obj, key):
        return getattr(obj, key)
    return default

def load_api_keys() -> tuple[Optional[str], Optional[str]]:
    """Load API keys from environment variables or config file."""
    try:
        # First try to import from config.py
        from config import API_KEY_PERPS, API_SECRET_PERPS
        return API_KEY_PERPS, API_SECRET_PERPS
    except ImportError:
        # If config.py doesn't exist, try environment variables
        api_key = os.getenv('API_KEY_PERPS')
        api_secret = os.getenv('API_SECRET_PERPS')
        
        if not (api_key and api_secret):
            logging.error("API keys not found. Please set API_KEY_PERPS and API_SECRET_PERPS in config.py or as environment variables.")
            return None, None
            
        return api_key, api_secret

def format_order(order: Dict) -> str:
    """
    Format an order for display.
    
    Args:
        order: Order dictionary
        
    Returns:
        Formatted string representation of the order
    """
    try:
        # Ensure order is a dictionary
        order_dict = ensure_dict(order)
        
        # Extract order details
        order_id = safe_get(order_dict, 'order_id', 'Unknown')
        product_id = safe_get(order_dict, 'product_id', 'Unknown')
        side = safe_get(order_dict, 'side', '')
        order_type = safe_get(order_dict, 'order_type', safe_get(order_dict, 'type', ''))
        status = safe_get(order_dict, 'status', 'Unknown')
        
        # Get size and price
        size = safe_get(order_dict, 'size', safe_get(order_dict, 'base_size', 0))
        price = safe_get(order_dict, 'price', 0)
        stop_price = safe_get(order_dict, 'stop_price', 0)
        
        # Try to convert size to float
        try:
            size = float(size)
        except (ValueError, TypeError):
            size = 0
        
        # Try to convert price to float
        try:
            price = float(price)
        except (ValueError, TypeError):
            price = 0
        
        # Try to convert stop_price to float
        try:
            stop_price = float(stop_price)
        except (ValueError, TypeError):
            stop_price = 0
        
        # Get created time
        created_time = safe_get(order_dict, 'created_time', safe_get(order_dict, 'created_at', ''))
        if created_time:
            try:
                if isinstance(created_time, (int, float)):
                    created_time = datetime.fromtimestamp(created_time)
                else:
                    created_time = parser.parse(created_time)
                time_str = created_time.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = str(created_time)
        else:
            time_str = "unknown"
        
        # Format side with color
        side_color = GREEN if side.upper() == 'BUY' else RED
        side_display = f"{side_color}{side.upper()}{RESET}"
        
        # Format order type with color
        if 'STOP' in order_type.upper():
            type_color = RED
        elif 'LIMIT' in order_type.upper():
            type_color = GREEN
        else:
            type_color = YELLOW
        type_display = f"{type_color}{order_type.upper()}{RESET}"
        
        # Format the order string
        order_str = f"{order_id[:8]}... | "
        order_str += f"{product_id} | "
        order_str += f"{type_display} | "
        order_str += f"{side_display} | "
        order_str += f"Size: {size:.8f} | "
        
        if price > 0:
            order_str += f"Price: ${price:.2f} | "
        
        if stop_price > 0:
            order_str += f"Stop: ${stop_price:.2f} | "
        
        order_str += f"Status: {status} | "
        order_str += f"Created: {time_str}"
        
        return order_str
    
    except Exception as e:
        logging.error(f"Error formatting order: {e}")
        return f"Error formatting order: {e}"

def get_open_orders(service: CoinbaseService) -> List[Dict]:
    """
    Get all open orders.
    
    Args:
        service: CoinbaseService instance
        
    Returns:
        List of order dictionaries
    """
    try:
        # Get open orders
        orders_response = service.client.list_orders(status='OPEN')
        
        # Log the raw response for debugging
        logging.info(f"Raw orders response: {json.dumps(ensure_dict(orders_response), default=str)[:500]}...")
        
        # Extract orders from response
        orders = []
        
        if isinstance(orders_response, dict):
            if 'orders' in orders_response:
                orders = orders_response['orders']
        else:
            # Try to handle if orders_response is an object
            orders = safe_get(orders_response, 'orders', [])
        
        # Convert orders to list of dictionaries
        if orders:
            if not isinstance(orders, list):
                orders = [orders]
            
            orders_dicts = []
            for order in orders:
                order_dict = ensure_dict(order)
                orders_dicts.append(order_dict)
            
            orders = orders_dicts
        
        logging.info(f"Found {len(orders)} open orders")
        return orders
    
    except Exception as e:
        logging.error(f"Error getting open orders: {e}")
        return []

def get_positions(service):
    """
    Get all open positions from Coinbase.
    
    Args:
        service: CoinbaseService instance
        
    Returns:
        List of position dictionaries
    """
    positions = []
    
    try:
        logging.info("Starting position detection...")
        
        # Method 1: Try to get positions from portfolio info
        try:
            logging.info("Attempting to get positions from portfolio info...")
            # Get portfolio info for INTX (perpetuals)
            usd_balance, perp_position_size = service.get_portfolio_info(portfolio_type="INTX")
            logging.info(f"Portfolio info - USD Balance: {usd_balance}, Position Size: {perp_position_size}")
            
            if abs(perp_position_size) > 0:
                # We have a position
                position = {
                    'product_id': 'BTC-PERP-INTX',  # Assuming BTC perpetual
                    'side': 'LONG' if perp_position_size > 0 else 'SHORT',
                    'net_size': perp_position_size,
                    'source': 'portfolio_info'
                }
                positions.append(position)
                logging.info(f"Found position from portfolio info: {position}")
        except Exception as e:
            logging.error(f"Error getting positions from portfolio info: {e}")
        
        # Method 2: Try to infer positions from bracket orders
        if not positions:
            try:
                logging.info("Attempting to infer positions from bracket orders...")
                # Get open orders
                # Note: CoinbaseService doesn't have a direct get_open_orders method
                # We'll need to use the client directly
                open_orders = []
                try:
                    # Try to get open orders using the client
                    orders_response = service.client.list_orders(status="OPEN")
                    logging.info(f"Orders response type: {type(orders_response)}")
                    
                    if isinstance(orders_response, dict) and 'orders' in orders_response:
                        open_orders = orders_response['orders']
                    elif hasattr(orders_response, 'orders'):
                        open_orders = orders_response.orders
                    
                    logging.info(f"Found {len(open_orders)} open orders")
                except Exception as e:
                    logging.error(f"Error getting open orders: {e}")
                
                if open_orders:
                    # Group orders by product
                    product_orders = {}
                    for order in open_orders:
                        order_dict = ensure_dict(order)
                        product_id = safe_get(order_dict, 'product_id')
                        
                        if product_id:
                            if product_id not in product_orders:
                                product_orders[product_id] = []
                            
                            product_orders[product_id].append(order_dict)
                    
                    logging.info(f"Grouped orders by {len(product_orders)} products")
                    
                    # Look for bracket orders
                    for product_id, orders in product_orders.items():
                        if len(orders) >= 2:
                            logging.info(f"Checking for bracket orders for {product_id} with {len(orders)} orders")
                            
                            # Check for take profit and stop loss orders
                            has_tp = any(safe_get(o, 'order_type', '').upper() == 'LIMIT' for o in orders)
                            has_sl = any(safe_get(o, 'order_type', '').upper() == 'STOP' for o in orders)
                            
                            if has_tp and has_sl:
                                logging.info(f"Found potential bracket order for {product_id}")
                                
                                # Get the orders
                                tp_order = next((o for o in orders if safe_get(o, 'order_type', '').upper() == 'LIMIT'), None)
                                sl_order = next((o for o in orders if safe_get(o, 'order_type', '').upper() == 'STOP'), None)
                                
                                if tp_order and sl_order:
                                    tp_side = safe_get(tp_order, 'side', '').upper()
                                    sl_side = safe_get(sl_order, 'side', '').upper()
                                    
                                    logging.info(f"TP order side: {tp_side}, SL order side: {sl_side}")
                                    
                                    # Determine position side
                                    position_side = None
                                    if tp_side == 'SELL' and sl_side == 'SELL':
                                        position_side = 'LONG'
                                    elif tp_side == 'BUY' and sl_side == 'BUY':
                                        position_side = 'SHORT'
                                    
                                    if position_side:
                                        logging.info(f"Inferred {position_side} position from bracket orders")
                                        
                                        # Get position size
                                        position_size = float(safe_get(tp_order, 'size', 0))
                                        if position_side == 'SHORT':
                                            position_size = -position_size
                                        
                                        # Create position
                                        position = {
                                            'product_id': product_id,
                                            'side': position_side,
                                            'net_size': position_size,
                                            'entry_price': float(safe_get(sl_order, 'stop_price', 0)),
                                            'source': 'inferred_from_bracket_orders'
                                        }
                                        
                                        logging.info(f"Inferred position: {position}")
                                        positions.append(position)
            except Exception as e:
                logging.error(f"Error inferring positions from bracket orders: {e}")
        
        # Method 3: Try to derive positions from recent fills
        if not positions:
            try:
                logging.info("Attempting to derive positions from recent fills...")
                # Get recent fills
                # Note: CoinbaseService doesn't have a direct get_fills method
                # We'll need to use the client directly
                fills = []
                try:
                    # Try to get fills using the client
                    fills_response = service.client.list_fills(limit=50)
                    logging.info(f"Fills response type: {type(fills_response)}")
                    
                    if isinstance(fills_response, dict) and 'fills' in fills_response:
                        fills = fills_response['fills']
                    elif hasattr(fills_response, 'fills'):
                        fills = fills_response.fills
                    
                    logging.info(f"Found {len(fills)} fills")
                except Exception as e:
                    logging.error(f"Error getting fills: {e}")
                
                if fills:
                    # Group fills by product
                    product_fills = {}
                    for fill in fills:
                        fill_dict = ensure_dict(fill)
                        product_id = safe_get(fill_dict, 'product_id')
                        
                        if product_id:
                            if product_id not in product_fills:
                                product_fills[product_id] = []
                            
                            product_fills[product_id].append(fill_dict)
                    
                    logging.info(f"Grouped fills by {len(product_fills)} products")
                    
                    # Calculate net position for each product
                    for product_id, product_fills_list in product_fills.items():
                        logging.info(f"Calculating position for {product_id} from {len(product_fills_list)} fills")
                        net_size = 0
                        total_value = 0
                        
                        for fill in product_fills_list:
                            size = float(safe_get(fill, 'size', 0))
                            price = float(safe_get(fill, 'price', 0))
                            side = safe_get(fill, 'side', '').upper()
                            
                            logging.debug(f"Fill: Side={side}, Size={size}, Price={price}")
                            
                            if side == 'BUY':
                                net_size += size
                                total_value += size * price
                            elif side == 'SELL':
                                net_size -= size
                                total_value -= size * price
                        
                        logging.info(f"Calculated net size for {product_id}: {net_size}")
                        
                        # If we have a non-zero position, add it
                        if abs(net_size) > 0.00001:
                            avg_price = abs(total_value / net_size) if net_size != 0 else 0
                            position_side = 'LONG' if net_size > 0 else 'SHORT'
                            
                            position = {
                                'product_id': product_id,
                                'net_size': net_size,
                                'side': position_side,
                                'entry_price': avg_price,
                                'source': 'derived_from_fills'
                            }
                            
                            logging.info(f"Derived position for {product_id}: {position}")
                            positions.append(position)
                        else:
                            logging.info(f"No significant net position for {product_id}")
            except Exception as e:
                logging.error(f"Error deriving positions from fills: {e}")
        
        # Method 4: Try to get positions from historical data
        if not positions:
            try:
                logging.info("Attempting to get positions from historical data...")
                # Check if the service has historical_data
                if hasattr(service, 'historical_data'):
                    # Get recent trades
                    trades = service.historical_data.get_recent_trades('BTC-PERP-INTX', limit=50)
                    logging.info(f"Historical trades response type: {type(trades)}")
                    
                    if trades:
                        logging.info(f"Found {len(trades)} historical trades")
                        
                        # Calculate net position
                        net_size = 0
                        total_value = 0
                        
                        for trade in trades:
                            trade_dict = ensure_dict(trade)
                            size = float(safe_get(trade_dict, 'size', 0))
                            price = float(safe_get(trade_dict, 'price', 0))
                            side = safe_get(trade_dict, 'side', '').upper()
                            
                            if side == 'BUY':
                                net_size += size
                                total_value += size * price
                            elif side == 'SELL':
                                net_size -= size
                                total_value -= size * price
                        
                        logging.info(f"Calculated net size from historical trades: {net_size}")
                        
                        # If we have a non-zero position, add it
                        if abs(net_size) > 0.00001:
                            avg_price = abs(total_value / net_size) if net_size != 0 else 0
                            position_side = 'LONG' if net_size > 0 else 'SHORT'
                            
                            position = {
                                'product_id': 'BTC-PERP-INTX',
                                'net_size': net_size,
                                'side': position_side,
                                'entry_price': avg_price,
                                'source': 'derived_from_historical_trades'
                            }
                            
                            logging.info(f"Derived position from historical trades: {position}")
                            positions.append(position)
                else:
                    logging.warning("Service does not have historical_data attribute")
            except Exception as e:
                logging.error(f"Error getting positions from historical data: {e}")
        
        # Method 5: Create a dummy position for testing if no positions found
        if not positions and os.environ.get('DEBUG_DUMMY_POSITION') == '1':
            logging.info("Creating dummy position for testing")
            position = {
                'product_id': 'BTC-PERP-INTX',
                'side': 'SHORT',
                'net_size': -0.001,
                'entry_price': 50000.0,
                'source': 'dummy_for_testing'
            }
            positions.append(position)
        
        # Log the final result
        if positions:
            logging.info(f"Found {len(positions)} total positions")
            for i, pos in enumerate(positions):
                product_id = safe_get(pos, 'product_id', safe_get(pos, 'symbol', 'unknown'))
                side = safe_get(pos, 'side', 'unknown')
                size = safe_get(pos, 'net_size', safe_get(pos, 'size', safe_get(pos, 'position_size', 0)))
                logging.info(f"Final position {i+1}: Product={product_id}, Side={side}, Size={size}")
        else:
            logging.warning("No positions found using any method")
        
        return positions
    
    except Exception as e:
        logging.error(f"Error in get_positions: {e}")
        return []

def format_position(position, current_prices=None):
    """
    Format a position for display.
    
    Args:
        position: Position dictionary
        current_prices: Dictionary of current prices by product_id
        
    Returns:
        Formatted position string
    """
    try:
        # Ensure position is a dictionary
        position_dict = ensure_dict(position)
        
        # Extract basic position information
        product_id = safe_get(position_dict, 'product_id', safe_get(position_dict, 'symbol', 'unknown'))
        side = safe_get(position_dict, 'side', 'unknown').upper()
        
        # Get position size
        size = 0
        for size_field in ['net_size', 'size', 'position_size', 'quantity']:
            size_value = safe_get(position_dict, size_field)
            if size_value is not None:
                try:
                    size = float(size_value)
                    break
                except (ValueError, TypeError):
                    pass
        
        # Skip positions with zero size
        if abs(size) <= 0.00001:
            return None
        
        # Determine position side from size if not specified
        if side == 'unknown':
            side = 'LONG' if size > 0 else 'SHORT'
        
        # Get current price
        current_price = 0
        
        # First try to get mark price from position
        mark_price = safe_get(position_dict, 'mark_price')
        if mark_price:
            try:
                # Handle nested mark_price structure
                if isinstance(mark_price, dict):
                    # Try to extract from rawCurrency or userNativeCurrency
                    if 'rawCurrency' in mark_price and isinstance(mark_price['rawCurrency'], dict) and 'value' in mark_price['rawCurrency']:
                        current_price = float(mark_price['rawCurrency']['value'])
                    elif 'userNativeCurrency' in mark_price and isinstance(mark_price['userNativeCurrency'], dict) and 'value' in mark_price['userNativeCurrency']:
                        current_price = float(mark_price['userNativeCurrency']['value'])
                    elif 'value' in mark_price:
                        current_price = float(mark_price['value'])
                else:
                    current_price = float(mark_price)
            except (ValueError, TypeError) as e:
                logging.debug(f"Error converting mark_price to float: {e}")
        
        # If mark_price didn't work, try other price fields
        if current_price == 0:
            for price_field in ['price', 'current_price', 'last_price']:
                price_value = safe_get(position_dict, price_field)
                if price_value:
                    try:
                        if isinstance(price_value, dict) and 'value' in price_value:
                            current_price = float(price_value['value'])
                        else:
                            current_price = float(price_value)
                        break
                    except (ValueError, TypeError):
                        pass
        
        # If we still don't have a price, try the current_prices dictionary
        if current_price == 0 and current_prices and product_id in current_prices:
            current_price = current_prices[product_id]
        
        # Get entry price
        entry_price = 0
        
        # Try to get entry price from vwap
        vwap = safe_get(position_dict, 'vwap')
        if vwap:
            try:
                if isinstance(vwap, dict) and 'value' in vwap:
                    entry_price = float(vwap['value'])
                else:
                    entry_price = float(vwap)
            except (ValueError, TypeError):
                pass
        
        # If vwap didn't work, try other entry price fields
        if entry_price == 0:
            for entry_field in ['entry_price', 'average_entry', 'avg_entry_price']:
                entry_value = safe_get(position_dict, entry_field)
                if entry_value:
                    try:
                        if isinstance(entry_value, dict) and 'value' in entry_value:
                            entry_price = float(entry_value['value'])
                        else:
                            entry_price = float(entry_value)
                        break
                    except (ValueError, TypeError):
                        pass
        
        # If we still don't have an entry price, use current price
        if entry_price == 0 and current_price > 0:
            entry_price = current_price
        
        # Get unrealized PnL
        unrealized_pnl = 0
        unrealized_pnl_value = safe_get(position_dict, 'unrealized_pnl')
        if unrealized_pnl_value:
            try:
                if isinstance(unrealized_pnl_value, dict):
                    # Try to extract from rawCurrency or userNativeCurrency
                    if 'rawCurrency' in unrealized_pnl_value and isinstance(unrealized_pnl_value['rawCurrency'], dict) and 'value' in unrealized_pnl_value['rawCurrency']:
                        unrealized_pnl = float(unrealized_pnl_value['rawCurrency']['value'])
                    elif 'userNativeCurrency' in unrealized_pnl_value and isinstance(unrealized_pnl_value['userNativeCurrency'], dict) and 'value' in unrealized_pnl_value['userNativeCurrency']:
                        unrealized_pnl = float(unrealized_pnl_value['userNativeCurrency']['value'])
                    elif 'value' in unrealized_pnl_value:
                        unrealized_pnl = float(unrealized_pnl_value['value'])
                else:
                    unrealized_pnl = float(unrealized_pnl_value)
            except (ValueError, TypeError):
                pass
        
        # If we don't have unrealized PnL but have prices, calculate it
        if unrealized_pnl == 0 and current_price > 0 and entry_price > 0:
            if side == 'LONG':
                unrealized_pnl = abs(size) * (current_price - entry_price)
            else:  # SHORT
                unrealized_pnl = abs(size) * (entry_price - current_price)
        
        # Calculate position value
        position_value = abs(size) * current_price if current_price > 0 else 0
        
        # Calculate PnL percentage
        pnl_percentage = 0
        if unrealized_pnl != 0 and position_value > 0:
            pnl_percentage = (unrealized_pnl / position_value) * 100
        elif entry_price > 0 and current_price > 0:
            if side == 'LONG':
                pnl_percentage = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_percentage = ((entry_price - current_price) / entry_price) * 100
        
        # Get liquidation price
        liquidation_price = 0
        liquidation_price_value = safe_get(position_dict, 'liquidation_price')
        if liquidation_price_value:
            try:
                if isinstance(liquidation_price_value, dict) and 'value' in liquidation_price_value:
                    liquidation_price = float(liquidation_price_value['value'])
                else:
                    liquidation_price = float(liquidation_price_value)
            except (ValueError, TypeError):
                pass
        
        # Get leverage
        leverage = safe_get(position_dict, 'leverage', 1)
        try:
            leverage = float(leverage)
        except (ValueError, TypeError):
            leverage = 1
        
        # Get position creation time
        creation_time = safe_get(position_dict, 'created_at', safe_get(position_dict, 'creation_time'))
        time_str = ""
        if creation_time:
            try:
                # Convert to datetime if it's a string
                if isinstance(creation_time, str):
                    creation_time = datetime.fromisoformat(creation_time.replace('Z', '+00:00'))
                
                # Calculate time since creation
                time_since = datetime.now(timezone.utc) - creation_time.astimezone(timezone.utc)
                hours = time_since.total_seconds() / 3600
                
                if hours < 1:
                    time_str = f"{int(hours * 60)}m"
                elif hours < 24:
                    time_str = f"{int(hours)}h"
                else:
                    time_str = f"{int(hours / 24)}d"
            except Exception as e:
                logging.debug(f"Error calculating time since creation: {e}")
        
        # Format position string with color based on side
        color = GREEN if side == 'LONG' else RED
        
        # Get position source for debugging
        source = safe_get(position_dict, 'source', 'api')
        
        # Format the position string
        position_str = f"{color}{product_id} {side} "
        
        # Add size
        position_str += f"Size: {abs(size):.8f} "
        
        # Add entry price if available
        if entry_price > 0:
            position_str += f"Entry: ${entry_price:.2f} "
        
        # Add current price if available
        if current_price > 0:
            position_str += f"Current: ${current_price:.2f} "
        
        # Add position value if available
        if position_value > 0:
            position_str += f"Value: ${position_value:.2f} "
        
        # Add leverage if available and not 1
        if leverage > 1:
            position_str += f"Leverage: {leverage}x "
        
        # Add liquidation price if available
        if liquidation_price > 0:
            position_str += f"Liq: ${liquidation_price:.2f} "
        
        # Add PnL if available
        if pnl_percentage != 0:
            pnl_color = GREEN if pnl_percentage > 0 else RED
            position_str += f"PnL: {pnl_color}{pnl_percentage:.2f}%{color} "
        
        # Add unrealized PnL if available
        if unrealized_pnl != 0:
            pnl_color = GREEN if unrealized_pnl > 0 else RED
            position_str += f"Unrealized: {pnl_color}${abs(unrealized_pnl):.2f}{color} "
        
        # Add time if available
        if time_str:
            position_str += f"Time: {time_str} "
        
        # Add source for debugging
        position_str += f"[{source}]"
        
        # Reset color at the end
        position_str += RESET
        
        return position_str
    
    except Exception as e:
        logging.error(f"Error formatting position: {e}")
        return None

def monitor_orders(service: CoinbaseService, check_interval: int = 60, max_duration: int = 0, show_positions: bool = True):
    """
    Continuously monitor open orders on Coinbase and provide status updates until all orders are closed.
    
    Args:
        service: The CoinbaseService instance to use for API calls
        check_interval: How often to check for updates (in seconds)
        max_duration: Maximum duration to monitor (in seconds), 0 for unlimited
        show_positions: Whether to show open positions
    """
    start_time = time.time()
    
    while True:
        # Clear screen for readability
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Check if we've exceeded the maximum duration
        elapsed_time = time.time() - start_time
        if max_duration > 0 and elapsed_time > max_duration:
            logging.info(f"Maximum monitoring duration of {max_duration} seconds reached. Exiting.")
            break
        
        # Get current prices from positions and open orders
        current_prices = {}
        
        # Always show positions regardless of the parameter
        show_positions = True
        
        # Get positions from API
        api_positions = []
        if show_positions:
            try:
                api_positions = get_positions(service)
                logging.info(f"Found {len(api_positions)} positions from API")
                for pos in api_positions:
                    logging.debug(f"API Position: {pos}")
            except Exception as e:
                logging.error(f"Error getting positions: {e}")
                api_positions = []
        
        # Get open orders
        try:
            open_orders = get_open_orders(service)
            logging.info(f"Found {len(open_orders)} open orders")
        except Exception as e:
            logging.error(f"Error getting open orders: {e}")
            open_orders = []
        
        # Display open positions and orders
        if show_positions:
            print("\n" + "="*80)
            print("POSITIONS".center(80))
            print("="*80 + "\n")
            
            if api_positions:
                for position in api_positions:
                    try:
                        position_str = format_position(position)
                        if position_str:
                            print(position_str)
                            logging.debug(f"Displayed position: {position_str}")
                        else:
                            logging.warning(f"Position formatting returned None: {position}")
                    except Exception as e:
                        logging.error(f"Error formatting position: {e}")
                        print(f"Error displaying position: {e}")
            else:
                print("No positions found.")
                logging.info("No positions found to display")
            
            print("\n" + "-"*80 + "\n")
        
        # Filter for open bracket orders
        bracket_orders = []
        if open_orders:
            # Group orders by product
            product_orders = {}
            for order in open_orders:
                order_dict = ensure_dict(order)
                product_id = safe_get(order_dict, 'product_id')
                
                if product_id:
                    if product_id not in product_orders:
                        product_orders[product_id] = []
                    
                    product_orders[product_id].append(order_dict)
            
            # Identify bracket orders
            for product_id, orders in product_orders.items():
                if len(orders) >= 2:
                    # Check for take profit and stop loss orders
                    has_tp = any(safe_get(o, 'order_type', '').upper() == 'LIMIT' for o in orders)
                    has_sl = any(safe_get(o, 'order_type', '').upper() == 'STOP' for o in orders)
                    
                    if has_tp and has_sl:
                        logging.info(f"Found bracket order for {product_id}")
                        # Add these orders to bracket_orders
                        bracket_orders.extend(orders)
        
        # Display open bracket orders
        print("\n" + "="*80)
        print("OPEN BRACKET ORDERS".center(80))
        print("="*80 + "\n")
        
        if bracket_orders:
            # Filter for only orders with status "OPEN"
            open_bracket_orders = [order for order in bracket_orders if safe_get(order, 'status', '').upper() == 'OPEN']
            if open_bracket_orders:
                for order in open_bracket_orders:
                    try:
                        order_str = format_order(order)
                        print(order_str)
                    except Exception as e:
                        logging.error(f"Error formatting order: {e}")
            else:
                print("No open bracket orders found.")
        else:
            print("No open bracket orders found.")
        
        # Display all open orders (including standalone limit orders)
        print("\n" + "="*80)
        print("ALL OPEN ORDERS".center(80))
        print("="*80 + "\n")
        
        if open_orders:
            # Filter out orders that are already displayed as bracket orders
            bracket_order_ids = [safe_get(order, 'order_id') for order in bracket_orders]
            # Filter for only orders with status "OPEN"
            standalone_orders = [order for order in open_orders 
                               if safe_get(order, 'order_id') not in bracket_order_ids
                               and safe_get(order, 'status', '').upper() == 'OPEN']
            
            if standalone_orders:
                for order in standalone_orders:
                    try:
                        order_str = format_order(order)
                        print(order_str)
                    except Exception as e:
                        logging.error(f"Error formatting order: {e}")
            else:
                print("No standalone open orders found.")
        else:
            print("No open orders found.")
        
        print("-"*80 + "\n")
        print(f"Next check in {check_interval} seconds. Press Ctrl+C to exit.")
        
        try:
            time.sleep(check_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
            break

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Monitor open orders until all are closed',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--interval', type=int, default=60,
                        help='Check interval in seconds')
    parser.add_argument('-d', '--duration', type=int, default=0,
                        help='Maximum monitoring duration in minutes (0 for unlimited)')
    parser.add_argument('-p', '--positions', action='store_true', default=True,
                        help='Show open positions in addition to orders')
    
    args = parser.parse_args()
    
    # Load API keys
    api_key, api_secret = load_api_keys()
    if not api_key or not api_secret:
        logging.error("Failed to load API keys")
        return
    
    # Initialize Coinbase service
    service = CoinbaseService(api_key, api_secret)
    
    # Start monitoring
    monitor_orders(
        service=service,
        check_interval=args.interval,
        max_duration=args.duration * 60 if args.duration > 0 else 0,  # Convert minutes to seconds
        show_positions=True  # Always show positions
    )

if __name__ == "__main__":
    main() 