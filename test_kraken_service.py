#!/usr/bin/env python3
"""
Test script for KrakenService class.
This script demonstrates how to use the KrakenService class to interact with Kraken Pro,
including placing orders with stop loss and take profit values.
"""

import os
import logging
import time
from datetime import datetime
from krakenservice import KrakenService
import config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to test KrakenService functionality."""
    # Get API credentials from config.py
    api_key = config.KRAKEN_API_KEY
    api_secret = config.KRAKEN_API_SECRET
    
    if not api_key or not api_secret:
        logger.error("API credentials not found in config.py. Please set KRAKEN_API_KEY and KRAKEN_API_SECRET in config.py.")
        return
    
    # Initialize KrakenService
    kraken = KrakenService(api_key=api_key, api_secret=api_secret)
    
    # Test portfolio information
    test_portfolio_info(kraken)
    
    # # Test getting BTC prices
    test_get_btc_prices(kraken)
    
    # Test getting OHLC data
    test_get_ohlc_data(kraken)
    
    # Test getting recent trades
    test_get_recent_trades(kraken)
    
    # Test checking for open orders and positions
    test_check_open_orders_and_positions(kraken)

    # # Test placing orders with stop loss and take profit
    # test_place_orders_with_sl_tp(kraken)
    
    # # Test closing positions
    # test_close_positions(kraken)

def test_portfolio_info(kraken):
    """Test getting portfolio information."""
    logger.info("Testing portfolio information...")
    
    # Test raw balance endpoint
    logger.info("Testing raw balance endpoint...")
    response = kraken._api_request("POST", "/private/Balance", {})
    logger.info(f"Raw balance response: {response}")
    
    # Get spot portfolio info
    fiat_balance, crypto_balance = kraken.get_portfolio_info(portfolio_type="SPOT")
    logger.info(f"Spot Portfolio - Fiat: {fiat_balance}, Crypto: {crypto_balance}")
    
    # Get futures portfolio info
    usd_balance, position_size = kraken.get_portfolio_info(portfolio_type="FUTURES")
    logger.info(f"Futures Portfolio - USD: {usd_balance}, Position Size: {position_size}")

def test_get_btc_prices(kraken):
    """Test getting BTC prices."""
    logger.info("Testing BTC prices...")
    
    # Get spot prices
    prices = kraken.get_btc_prices(include_futures=False)
    logger.info(f"Spot Prices: {prices}")
    
    # Get spot and futures prices
    prices_with_futures = kraken.get_btc_prices(include_futures=True)
    logger.info(f"Prices with Futures: {prices_with_futures}")

def test_place_orders_with_sl_tp(kraken):
    """Test placing orders with stop loss and take profit."""
    logger.info("Testing placing orders with stop loss and take profit...")
    
    # Get current BTC price
    prices = kraken.get_btc_prices(include_futures=True)
    current_price = float(prices.get('XBTUSD', {}).get('ask', 0))
    
    if current_price <= 0:
        logger.error("Could not get current BTC price. Skipping order tests.")
        return
    
    # Calculate stop loss and take profit prices
    stop_loss_percentage = 0.02  # 2% below entry
    take_profit_percentage = 0.04  # 4% above entry
    
    stop_loss_price = current_price * (1 - stop_loss_percentage)
    take_profit_price = current_price * (1 + take_profit_percentage)
    
    logger.info(f"Current BTC Price: {current_price}")
    logger.info(f"Stop Loss Price: {stop_loss_price} ({stop_loss_percentage*100}% below)")
    logger.info(f"Take Profit Price: {take_profit_price} ({take_profit_percentage*100}% above)")
    
    # Test futures orders with leverage
    logger.info("Testing futures orders with leverage...")
    
    # Place a futures market order with leverage
    leverage = 5  # 5x leverage
    futures_volume = 0.001  # Small amount for testing
    
    # First set leverage
    leverage_result = kraken._set_leverage("XBTUSD", leverage)
    if 'error' in leverage_result:
        logger.error(f"Error setting leverage: {leverage_result['error']}")
        return
    
    logger.info(f"Leverage set to {leverage}x: {leverage_result}")
    
    # Place a futures market order
    logger.info("Placing futures market order...")
    futures_order_result = kraken.place_order(
        pair="XBTUSD",
        side="buy",
        volume=futures_volume,
        order_type="market",
        leverage=leverage,
        is_futures=True
    )
    
    if 'error' in futures_order_result:
        logger.error(f"Error placing futures market order: {futures_order_result['error']}")
        return
    
    logger.info(f"Futures market order placed: {futures_order_result}")
    
    # Place futures stop loss order
    logger.info("Placing futures stop loss order...")
    futures_sl_order_result = kraken.place_order(
        pair="XBTUSD",
        side="sell",
        volume=futures_volume,
        order_type="stop-loss",
        price=stop_loss_price,
        leverage=leverage,
        is_futures=True
    )
    
    if 'error' in futures_sl_order_result:
        logger.error(f"Error placing futures stop loss order: {futures_sl_order_result['error']}")
    else:
        logger.info(f"Futures stop loss order placed: {futures_sl_order_result}")
    
    # Place futures take profit order
    logger.info("Placing futures take profit order...")
    futures_tp_order_result = kraken.place_order(
        pair="XBTUSD",
        side="sell",
        volume=futures_volume,
        order_type="take-profit",
        price=take_profit_price,
        leverage=leverage,
        is_futures=True
    )
    
    if 'error' in futures_tp_order_result:
        logger.error(f"Error placing futures take profit order: {futures_tp_order_result['error']}")
    else:
        logger.info(f"Futures take profit order placed: {futures_tp_order_result}")

def test_get_ohlc_data(kraken):
    """Test getting OHLC data."""
    logger.info("Testing OHLC data...")
    
    # Get spot OHLC data
    spot_ohlc = kraken.get_ohlc_data(pair="XBTUSD", interval=15, is_futures=False)
    logger.info(f"Spot OHLC data (first 5 entries): {spot_ohlc[:5]}")
    
    # Get futures OHLC data
    futures_ohlc = kraken.get_ohlc_data(pair="XBTUSD", interval=15, is_futures=True)
    logger.info(f"Futures OHLC data (first 5 entries): {futures_ohlc[:5]}")

def test_get_recent_trades(kraken):
    """Test getting recent trades."""
    logger.info("Testing recent trades...")
    
    # Get spot recent trades
    spot_trades = kraken.get_recent_trades(pair="XBTUSD", is_futures=False)
    logger.info(f"Spot recent trades (first 5 entries): {spot_trades[:5]}")
    
    # Get futures recent trades
    futures_trades = kraken.get_recent_trades(pair="XBTUSD", is_futures=True)
    logger.info(f"Futures recent trades (first 5 entries): {futures_trades[:5]}")

def test_close_positions(kraken):
    """Test closing positions."""
    logger.info("Testing closing positions...")
    
    # Close all futures positions
    close_result = kraken.close_all_positions(pair="XBTUSD")
    logger.info(f"Close positions result: {close_result}")
    
    # Cancel all orders
    cancel_result = kraken.cancel_all_orders(pair="XBTUSD", is_futures=True)
    logger.info(f"Cancel orders result: {cancel_result}")

def test_check_open_orders_and_positions(kraken):
    """Test checking for open orders and positions."""
    logger.info("Testing check_open_orders_and_positions...")
    
    # Check for open orders and positions
    has_open_orders, has_positions = kraken.check_open_orders_and_positions()
    
    logger.info(f"Open Orders: {'Yes' if has_open_orders else 'No'}")
    logger.info(f"Open Positions: {'Yes' if has_positions else 'No'}")
    
    if has_open_orders:
        # Get details of open orders
        response = kraken._api_request("POST", "/private/OpenOrders", {})
        if 'error' not in response or not response['error']:
            result = response.get('result', {})
            open_orders = result.get('open', {})
            logger.info(f"Open Orders Details: {open_orders}")
    
    if has_positions:
        # Get details of positions
        usd_balance, position_size = kraken.get_portfolio_info(portfolio_type="FUTURES")
        logger.info(f"Position Details - USD Balance: {usd_balance}, Position Size: {position_size}")

if __name__ == "__main__":
    main() 