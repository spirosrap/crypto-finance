import krakenex
from pykrakenapi import KrakenAPI
import logging
from datetime import datetime, timedelta
import time
import uuid
import requests
import hmac
import hashlib
import base64
import json
from typing import Tuple, List, Dict, Optional, Union
import urllib.parse

class KrakenService:
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the Kraken service with API credentials.
        
        Args:
            api_key (str): Kraken API key
            api_secret (str): Kraken API secret
        """
        # Validate API credentials
        if not api_key or not api_secret:
            raise ValueError("API key and secret are required")
        
        # Remove any whitespace
        api_key = api_key.strip()
        api_secret = api_secret.strip()
        
        # Validate API key format
        if not all(c in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/+=' for c in api_key):
            raise ValueError("API key contains invalid characters")
        
        # Validate API secret format (should be base64 encoded)
        try:
            base64.b64decode(api_secret)
        except Exception:
            raise ValueError("API secret is not valid base64")
        
        # Store API credentials directly
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Initialize Kraken API for spot trading
        self.kraken = krakenex.API(api_key, api_secret)
        self.api = KrakenAPI(self.kraken)
        
        # Settings
        self.DEFAULT_FEE_RATE = 0.0026  # 0.26% for maker orders
        self.MAX_RETRIES = 3
        self.RETRY_DELAY_SECONDS = 5
        self.logger = logging.getLogger(__name__)
        
        # Kraken API URL - using main API for both spot and futures
        self.api_url = "https://api.kraken.com/0"
        
        # Check if API is configured
        self.logger.info("Kraken API initialized")

    def get_portfolio_info(self, portfolio_type: str = "SPOT") -> Tuple[float, float]:
        """
        Get portfolio information including fiat and crypto balances.
        
        Args:
            portfolio_type (str): Type of portfolio to query ("SPOT" or "FUTURES")
            
        Returns:
            Tuple[float, float]: (fiat_balance, crypto_balance) for spot positions
                                or (usd_balance, position_size) for futures
        """
        if portfolio_type.upper() == "FUTURES":
            return self._get_futures_portfolio_info()
        else:
            return self._get_spot_portfolio_info()
    
    def _get_spot_portfolio_info(self) -> Tuple[float, float]:
        """Get spot portfolio information."""
        try:
            balance = self.api.get_account_balance()
            
            # Initialize balances
            fiat_balance = 0.0
            crypto_balance = 0.0
            
            # Kraken returns balances in a dictionary with currency as key
            for currency, amount in balance.items():
                if currency in ['USD', 'EUR', 'GBP']:
                    fiat_balance += float(amount)
                elif currency in ['XBT', 'BTC']:  # Kraken uses XBT for Bitcoin
                    crypto_balance += float(amount)
            
            self.logger.info(f"Retrieved spot portfolio - Fiat: {fiat_balance}, Crypto: {crypto_balance}")
            return fiat_balance, crypto_balance
            
        except Exception as e:
            self.logger.error(f"Error getting spot portfolio info: {str(e)}")
            return 0.0, 0.0
    
    def _get_futures_portfolio_info(self) -> Tuple[float, float]:
        """
        Get futures portfolio information from Kraken Pro API.
        
        Returns:
            Tuple[float, float]: (usd_balance, position_size)
        """
        try:
            # Get account balance
            response = self._api_request("POST", "/private/Balance", {})
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting futures portfolio: {response['error']}")
                return 0.0, 0.0
                
            # Extract account information
            result = response.get('result', {})
            
            # Get USD balance
            usd_balance = float(result.get('ZUSD', 0))
            
            # Get open positions
            positions_response = self._api_request("POST", "/private/OpenPositions", {})
            
            if 'error' in positions_response and positions_response['error']:
                self.logger.error(f"Error getting open positions: {positions_response['error']}")
                return usd_balance, 0.0
            
            positions = positions_response.get('result', {})
            
            # Calculate total position size for futures
            position_size = 0.0
            for pos_id, pos in positions.items():
                pair = pos.get('pair', '')
                if pair.startswith('PI_'):  # Perpetual futures pairs start with PI_
                    position_size += abs(float(pos.get('vol', 0)))  # Use abs() to handle both long and short positions
            
            self.logger.info(f"Retrieved futures portfolio - USD: {usd_balance}, Position Size: {position_size}")
            return usd_balance, position_size
            
        except Exception as e:
            self.logger.error(f"Error getting futures portfolio info: {str(e)}")
            return 0.0, 0.0

    def get_btc_prices(self, include_futures: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Get current BTC prices from Kraken.
        
        Args:
            include_futures (bool): Whether to include futures prices
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of trading pairs and their bid/ask prices
        """
        prices = {}
        
        # Get spot prices
        try:
            response = self._api_request("GET", "/public/Ticker", {'pair': 'XBTUSD'})
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting spot BTC prices: {response['error']}")
            else:
                result = response.get('result', {})
                for pair, data in result.items():
                    prices[pair] = {
                        'bid': float(data['b'][0]),  # Best bid price
                        'ask': float(data['a'][0])   # Best ask price
                    }
        except Exception as e:
            self.logger.error(f"Error getting spot BTC prices: {str(e)}")
        
        # Get futures prices if requested
        if include_futures:
            try:
                # Get futures ticker using the correct pair format for perpetual futures
                response = self._api_request("GET", "/public/Ticker", {'pair': 'XBTUSD'})
                
                if 'error' in response and response['error']:
                    self.logger.error(f"Error getting futures BTC prices: {response['error']}")
                else:
                    result = response.get('result', {})
                    if 'XBTUSD' in result:
                        data = result['XBTUSD']
                        prices['XBTUSD'] = {
                            'bid': float(data['b'][0]),  # Best bid price
                            'ask': float(data['a'][0]),  # Best ask price
                            'last': float(data['c'][0]), # Last trade closed price
                            'volume': float(data['v'][1]) # Volume today
                        }
            except Exception as e:
                self.logger.error(f"Error getting futures BTC prices: {str(e)}")
        
        return prices

    def place_order(self, pair: str, side: str, volume: float, 
                   order_type: str = "market", price: Optional[float] = None,
                   leverage: Optional[int] = None, is_futures: bool = False) -> Dict:
        """
        Place an order on Kraken.
        
        Args:
            pair (str): Trading pair (e.g., 'XBTUSD')
            side (str): 'buy' or 'sell'
            volume (float): Amount to trade
            order_type (str): 'market' or 'limit'
            price (float, optional): Required for limit orders
            leverage (int, optional): Leverage for futures orders
            is_futures (bool): Whether this is a futures order
            
        Returns:
            Dict: Order details
        """
        if is_futures:
            return self._place_futures_order(pair, side, volume, order_type, price, leverage)
        else:
            return self._place_spot_order(pair, side, volume, order_type, price)
    
    def _place_spot_order(self, pair: str, side: str, volume: float, 
                         order_type: str = "market", price: Optional[float] = None) -> Dict:
        """Place a spot order."""
        try:
            # Generate a unique userref
            userref = int(uuid.uuid4().int & (1<<32)-1)
            
            # Prepare order parameters
            params = {
                'pair': pair,
                'type': side.lower(),
                'volume': str(volume),
                'userref': userref
            }
            
            if order_type.lower() == 'limit':
                if price is None:
                    raise ValueError("Price must be specified for limit orders")
                params['price'] = str(price)
                params['ordertype'] = 'limit'
            else:
                params['ordertype'] = 'market'
            
            # Place the order
            result = self.api.add_standard_order(**params)
            
            if 'error' in result and result['error']:
                self.logger.error(f"Error placing spot order: {result['error']}")
                return {'error': result['error']}
            
            self.logger.info(f"Spot order placed successfully: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing spot order: {str(e)}")
            return {'error': str(e)}
    
    def _place_futures_order(self, pair: str, side: str, volume: float, 
                           order_type: str = "market", price: Optional[float] = None,
                           leverage: Optional[int] = None) -> Dict:
        """
        Place a futures order on Kraken.
        
        Args:
            pair (str): Trading pair (e.g., 'XBTUSD')
            side (str): 'buy' or 'sell'
            volume (float): Order volume in BTC
            order_type (str): Type of order (market, limit)
            price (float, optional): Limit price for limit orders
            leverage (int, optional): Leverage for futures orders
            
        Returns:
            Dict: Order response from Kraken
        """
        try:
            # Prepare order parameters
            params = {
                'pair': 'XBTUSD',  # Perpetual futures pair
                'type': side.lower(),
                'ordertype': order_type.lower(),
                'volume': str(abs(volume))
            }
            
            if price is not None:
                params['price'] = str(price)
            
            if leverage is not None:
                params['leverage'] = str(leverage)
            
            # Place the order using the main API
            response = self._api_request("POST", "/private/AddOrder", params)
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error placing futures order: {response['error']}")
                return {'error': response['error']}
            
            return response.get('result', {})
            
        except Exception as e:
            self.logger.error(f"Error placing futures order: {str(e)}")
            return {'error': str(e)}
    
    def _set_leverage(self, pair: str, leverage: int) -> Dict:
        """Set leverage for a trading pair."""
        try:
            params = {
                'pair': pair,
                'leverage': str(leverage)
            }
            
            response = self._api_request("POST", "/private/SetLeverage", params)
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error setting leverage: {response['error']}")
                return {'error': response['error']}
            
            self.logger.info(f"Leverage set successfully: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error setting leverage: {str(e)}")
            return {'error': str(e)}

    def cancel_all_orders(self, pair: Optional[str] = None, is_futures: bool = False) -> Dict:
        """
        Cancel all open orders, optionally filtered by trading pair.
        
        Args:
            pair (str, optional): Trading pair to cancel orders for
            is_futures (bool): Whether to cancel futures orders
            
        Returns:
            Dict: Cancellation results
        """
        if is_futures:
            return self._cancel_all_futures_orders(pair)
        else:
            return self._cancel_all_spot_orders(pair)
    
    def _cancel_all_spot_orders(self, pair: Optional[str] = None) -> Dict:
        """Cancel all spot orders."""
        try:
            # Get open orders
            open_orders = self.api.get_open_orders()
            
            if not open_orders:
                self.logger.info("No open spot orders found")
                return {'status': 'success', 'message': 'No open orders'}
            
            cancelled_orders = []
            for order_id in open_orders.index:
                # If pair is specified, only cancel orders for that pair
                if pair and open_orders.loc[order_id, 'descr']['pair'] != pair:
                    continue
                
                try:
                    result = self.api.cancel_open_order(order_id)
                    if result['error']:
                        self.logger.error(f"Error cancelling order {order_id}: {result['error']}")
                    else:
                        cancelled_orders.append(order_id)
                except Exception as e:
                    self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
            
            return {
                'status': 'success',
                'cancelled_orders': cancelled_orders,
                'total_cancelled': len(cancelled_orders)
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling spot orders: {str(e)}")
            return {'error': str(e)}
    
    def _cancel_all_futures_orders(self, pair: Optional[str] = None) -> Dict:
        """Cancel all futures orders on Kraken Pro."""
        try:
            # Get open orders
            response = self._api_request("GET", "/private/OpenOrders")
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting open futures orders: {response['error']}")
                return {'error': response['error']}
            
            open_orders = response.get('result', {}).get('open', [])
            
            if not open_orders:
                self.logger.info("No open futures orders found")
                return {'status': 'success', 'message': 'No open orders'}
            
            cancelled_orders = []
            for order_id, order in open_orders.items():
                # If pair is specified, only cancel orders for that pair
                if pair and order.get('pair') != pair:
                    continue
                
                try:
                    cancel_params = {
                        'txid': order_id
                    }
                    
                    cancel_response = self._api_request("POST", "/private/CancelOrder", cancel_params)
                    
                    if 'error' in cancel_response and cancel_response['error']:
                        self.logger.error(f"Error cancelling futures order {order_id}: {cancel_response['error']}")
                    else:
                        cancelled_orders.append(order_id)
                except Exception as e:
                    self.logger.error(f"Error cancelling futures order {order_id}: {str(e)}")
            
            return {
                'status': 'success',
                'cancelled_orders': cancelled_orders,
                'total_cancelled': len(cancelled_orders)
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling futures orders: {str(e)}")
            return {'error': str(e)}

    def close_all_positions(self, pair: Optional[str] = None) -> Dict:
        """
        Close all open positions, optionally filtered by trading pair.
        
        Args:
            pair (str, optional): Trading pair to close positions for
            
        Returns:
            Dict: Results of closing positions
        """
        try:
            # Get open positions
            response = self._api_request("GET", "/private/OpenPositions")
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting open futures positions: {response['error']}")
                return {'error': response['error']}
            
            open_positions = response.get('result', {}).get('open', [])
            
            if not open_positions:
                self.logger.info("No open futures positions found")
                return {'status': 'success', 'message': 'No open positions'}
            
            closed_positions = []
            for position_id, position in open_positions.items():
                # If pair is specified, only close positions for that pair
                if pair and position.get('pair') != pair:
                    continue
                
                try:
                    # Determine the side to close the position
                    side = "sell" if position.get('type') == "buy" else "buy"
                    volume = position.get('vol')
                    
                    # Place a market order to close the position
                    close_params = {
                        'pair': position.get('pair'),
                        'type': side,
                        'ordertype': 'market',
                        'volume': volume,
                        'leverage': position.get('leverage', '1')
                    }
                    
                    close_response = self._api_request("POST", "/private/AddOrder", close_params)
                    
                    if 'error' in close_response and close_response['error']:
                        self.logger.error(f"Error closing position {position.get('pair')}: {close_response['error']}")
                    else:
                        closed_positions.append({
                            'pair': position.get('pair'),
                            'volume': volume,
                            'side': position.get('type')
                        })
                except Exception as e:
                    self.logger.error(f"Error closing position {position.get('pair')}: {str(e)}")
            
            return {
                'status': 'success',
                'closed_positions': closed_positions,
                'total_closed': len(closed_positions)
            }
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {str(e)}")
            return {'error': str(e)}

    def get_recent_trades(self, pair: str = 'XBTUSD', since: Optional[str] = None, is_futures: bool = False) -> List[Dict]:
        """
        Get recent trades for a trading pair.
        
        Args:
            pair (str): Trading pair (default: 'XBTUSD')
            since (str, optional): Return trades since this timestamp
            is_futures (bool): Whether to get futures trades
            
        Returns:
            List[Dict]: List of recent trades
        """
        if is_futures:
            return self._get_recent_futures_trades(pair, since)
        else:
            return self._get_recent_spot_trades(pair, since)
    
    def _get_recent_spot_trades(self, pair: str, since: Optional[str] = None) -> List[Dict]:
        """Get recent spot trades."""
        try:
            params = {'pair': 'XXBTZUSD'}  # Use XXBTZUSD for spot trading
            if since:
                params['since'] = since
            
            response = self._api_request("GET", "/public/Trades", params)
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting recent spot trades: {response['error']}")
                return []
            
            result = response.get('result', {})
            trades = result.get('XXBTZUSD', [])  # Use XXBTZUSD as the key
            
            formatted_trades = []
            for trade in trades:
                formatted_trade = {
                    'trade_time': int(trade[2]),
                    'side': 'buy' if trade[3] == 'b' else 'sell',
                    'price': float(trade[0]),
                    'volume': float(trade[1]),
                    'pair': 'XXBTZUSD'
                }
                formatted_trades.append(formatted_trade)
            
            return formatted_trades
            
        except Exception as e:
            self.logger.error(f"Error getting recent spot trades: {str(e)}")
            return []
    
    def _get_recent_futures_trades(self, pair: str = 'XBTUSD', since: Optional[str] = None) -> List[Dict]:
        """
        Get recent futures trades from Kraken.
        
        Args:
            pair (str): Trading pair (e.g., 'XBTUSD')
            since (str, optional): Return trade data since given timestamp
            
        Returns:
            List[Dict]: List of recent trades
        """
        try:
            params = {
                'pair': 'XBTUSD'  # Use XBTUSD directly for perpetual futures
            }
            
            if since:
                params['since'] = since
            
            response = self._api_request("GET", "/public/Trades", params)
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting recent futures trades: {response['error']}")
                return []
            
            result = response.get('result', {})
            trades = result.get('XBTUSD', [])  # Use XBTUSD as the key
            
            formatted_trades = []
            for trade in trades:
                formatted_trade = {
                    'trade_time': int(trade[2]),
                    'side': 'buy' if trade[3] == 'b' else 'sell',
                    'price': float(trade[0]),
                    'volume': float(trade[1]),
                    'pair': 'XBTUSD'
                }
                formatted_trades.append(formatted_trade)
            
            return formatted_trades
            
        except Exception as e:
            self.logger.error(f"Error getting recent futures trades: {str(e)}")
            return []

    def get_ohlc_data(self, pair: str, interval: int = 1, is_futures: bool = False) -> List[Dict]:
        """
        Get OHLC (Open, High, Low, Close) data for a trading pair.
        
        Args:
            pair (str): Trading pair (e.g., 'XBTUSD')
            interval (int): Time interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            is_futures (bool): Whether to get futures OHLC data
            
        Returns:
            List[Dict]: List of OHLC data points
        """
        if is_futures:
            return self._get_futures_ohlc_data(pair, interval)
        else:
            return self._get_spot_ohlc_data(pair, interval)
    
    def _get_spot_ohlc_data(self, pair: str, interval: int = 1) -> List[Dict]:
        """Get spot OHLC data."""
        try:
            response = self._api_request("GET", "/public/OHLC", {
                'pair': 'XXBTZUSD',  # Use XXBTZUSD for spot trading
                'interval': interval
            })
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting spot OHLC data: {response['error']}")
                return []
            
            result = response.get('result', {})
            ohlc_data = result.get('XXBTZUSD', [])  # Kraken returns data with this key for XBT/USD
            
            formatted_data = []
            for candle in ohlc_data:
                data_point = {
                    'time': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[6])
                }
                formatted_data.append(data_point)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error getting spot OHLC data: {str(e)}")
            return []
    
    def _get_futures_ohlc_data(self, pair: str, interval: int = 1) -> List[Dict]:
        """Get futures OHLC data."""
        try:
            response = self._api_request("GET", "/public/OHLC", {
                'pair': 'XXBTZUSD',  # Use XXBTZUSD for both spot and futures
                'interval': interval
            })
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting futures OHLC data: {response['error']}")
                return []
            
            result = response.get('result', {})
            ohlc_data = result.get('XXBTZUSD', [])  # Use XXBTZUSD as the key
            
            formatted_data = []
            for candle in ohlc_data:
                data_point = {
                    'time': int(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'vwap': float(candle[5]),
                    'volume': float(candle[6]),
                    'count': int(candle[7])
                }
                formatted_data.append(data_point)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error getting futures OHLC data: {str(e)}")
            return []

    def _api_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the Kraken API.
        
        Args:
            method (str): HTTP method ('GET' or 'POST')
            endpoint (str): API endpoint
            params (Dict, optional): Request parameters
            
        Returns:
            Dict: API response
        """
        try:
            # For private endpoints, use krakenex's query method
            if endpoint.startswith('/private/'):
                # Remove the leading slash as krakenex adds it
                endpoint = endpoint[1:]
                return self.kraken.query_private(endpoint.split('/')[-1], params or {})
            else:
                # For public endpoints, use krakenex's query method
                endpoint = endpoint[1:]
                return self.kraken.query_public(endpoint.split('/')[-1], params or {})
            
        except Exception as e:
            self.logger.error(f"Error making Kraken API request: {str(e)}")
            return {'error': str(e)} 