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

class KrakenService:
    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize the Kraken service with API credentials.
        
        Args:
            api_key (str): Kraken API key
            api_secret (str): Kraken API secret
        """
        self.kraken = krakenex.API(api_key, api_secret)
        self.api = KrakenAPI(self.kraken)
        self.DEFAULT_FEE_RATE = 0.0026  # 0.26% for maker orders
        self.MAX_RETRIES = 3
        self.RETRY_DELAY_SECONDS = 5
        self.logger = logging.getLogger(__name__)
        
        # Kraken Pro API URL
        self.pro_api_url = "https://api.kraken.com/0"
        
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
        """Get futures portfolio information from Kraken Pro."""
        try:
            # Get account information from Kraken Pro API
            response = self._pro_api_request("GET", "/private/Balance")
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting futures portfolio: {response['error']}")
                return 0.0, 0.0
                
            # Extract account information
            result = response.get('result', {})
            
            # Get USD balance
            usd_balance = float(result.get('ZUSD', 0))
            
            # Get open positions
            positions_response = self._pro_api_request("GET", "/private/OpenPositions")
            positions = positions_response.get('result', {}).get('open', [])
            
            # Calculate total position size
            position_size = 0.0
            for position in positions:
                if position.get('pair') == 'XBTUSD':  # Bitcoin perpetual
                    position_size += float(position.get('vol', 0))
            
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
            ticker = self.api.get_ticker_information(['XBTUSD', 'XBTEUR'])
            
            for pair, data in ticker.items():
                prices[pair] = {
                    'bid': float(data['b'][0]),  # Best bid price
                    'ask': float(data['a'][0])   # Best ask price
                }
        except Exception as e:
            self.logger.error(f"Error getting spot BTC prices: {str(e)}")
        
        # Get futures prices if requested
        if include_futures:
            try:
                # Get perpetual futures ticker from Kraken Pro
                ticker_response = self._pro_api_request("GET", "/public/Ticker", {'pair': 'XBTUSD'})
                
                if 'error' in ticker_response and ticker_response['error']:
                    self.logger.error(f"Error getting futures BTC prices: {ticker_response['error']}")
                else:
                    result = ticker_response.get('result', {})
                    if 'XXBTZUSD' in result:  # Kraken Pro uses XXBTZUSD for BTC/USD
                        ticker_data = result['XXBTZUSD']
                        prices['XBTUSD-PERP'] = {
                            'bid': float(ticker_data.get('b', [0])[0]),
                            'ask': float(ticker_data.get('a', [0])[0]),
                            'last': float(ticker_data.get('c', [0])[0]),
                            'volume': float(ticker_data.get('v', [0])[0])
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
        """Place a futures order on Kraken Pro."""
        try:
            # Set leverage if provided
            if leverage is not None:
                self._set_leverage(pair, leverage)
            
            # Prepare order parameters
            params = {
                'pair': pair,
                'type': side.lower(),
                'ordertype': order_type.lower(),
                'volume': str(volume),
                'leverage': str(leverage) if leverage else '1'
            }
            
            if order_type.lower() == 'limit':
                if price is None:
                    raise ValueError("Price must be specified for limit orders")
                params['price'] = str(price)
            
            # Place the order
            response = self._pro_api_request("POST", "/private/AddOrder", params)
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error placing futures order: {response['error']}")
                return {'error': response['error']}
            
            self.logger.info(f"Futures order placed successfully: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error placing futures order: {str(e)}")
            return {'error': str(e)}
    
    def _set_leverage(self, pair: str, leverage: int) -> Dict:
        """Set leverage for a futures trading pair on Kraken Pro."""
        try:
            params = {
                'pair': pair,
                'leverage': str(leverage)
            }
            
            response = self._pro_api_request("POST", "/private/Leverage", params)
            
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
            response = self._pro_api_request("GET", "/private/OpenOrders")
            
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
                    
                    cancel_response = self._pro_api_request("POST", "/private/CancelOrder", cancel_params)
                    
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
            response = self._pro_api_request("GET", "/private/OpenPositions")
            
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
                    
                    close_response = self._pro_api_request("POST", "/private/AddOrder", close_params)
                    
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
            return self._get_recent_futures_trades(pair)
        else:
            return self._get_recent_spot_trades(pair, since)
    
    def _get_recent_spot_trades(self, pair: str, since: Optional[str] = None) -> List[Dict]:
        """Get recent spot trades."""
        try:
            trades = self.api.get_recent_trades(pair, since)
            
            formatted_trades = []
            for trade in trades.itertuples():
                formatted_trade = {
                    'trade_time': int(trade.time.timestamp()),
                    'side': 'buy' if trade.type == 'b' else 'sell',
                    'price': float(trade.price),
                    'volume': float(trade.volume),
                    'pair': pair
                }
                formatted_trades.append(formatted_trade)
            
            return formatted_trades
            
        except Exception as e:
            self.logger.error(f"Error getting recent spot trades: {str(e)}")
            return []
    
    def _get_recent_futures_trades(self, pair: str) -> List[Dict]:
        """Get recent futures trades from Kraken Pro."""
        try:
            # Get recent trades from Kraken Pro API
            response = self._pro_api_request("GET", "/private/TradesHistory", {'pair': pair})
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting recent futures trades: {response['error']}")
                return []
            
            trades = response.get('result', {}).get('trades', [])
            
            formatted_trades = []
            for trade_id, trade in trades.items():
                formatted_trade = {
                    'trade_time': int(trade.get('time', 0)),
                    'side': 'buy' if trade.get('type') == 'buy' else 'sell',
                    'price': float(trade.get('price', 0)),
                    'volume': float(trade.get('vol', 0)),
                    'pair': trade.get('pair', pair)
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
            ohlc, last = self.api.get_ohlc_data(pair, interval=interval)
            
            formatted_data = []
            for index, row in ohlc.iterrows():
                data_point = {
                    'time': int(index.timestamp()),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                formatted_data.append(data_point)
            
            return formatted_data
            
        except Exception as e:
            self.logger.error(f"Error getting spot OHLC data: {str(e)}")
            return []
    
    def _get_futures_ohlc_data(self, pair: str, interval: int = 1) -> List[Dict]:
        """Get futures OHLC data from Kraken Pro."""
        try:
            # Convert interval to seconds
            interval_seconds = interval * 60
            
            # Get OHLC data from Kraken Pro API
            params = {
                'pair': pair,
                'interval': interval_seconds,
                'since': int(time.time() - 86400)  # Last 24 hours
            }
            
            response = self._pro_api_request("GET", "/public/OHLC", params)
            
            if 'error' in response and response['error']:
                self.logger.error(f"Error getting futures OHLC data: {response['error']}")
                return []
            
            ohlc_data = response.get('result', {}).get(pair, [])
            
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
    
    def _pro_api_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to the Kraken Pro API.
        
        Args:
            method (str): HTTP method ('GET' or 'POST')
            endpoint (str): API endpoint
            params (Dict, optional): Request parameters
            
        Returns:
            Dict: API response
        """
        try:
            url = f"{self.pro_api_url}{endpoint}"
            
            # Add authentication headers
            headers = self._get_pro_auth_headers(method, endpoint, params)
            
            # Make the request
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            else:
                response = requests.post(url, headers=headers, data=params)
            
            # Parse the response
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error making Kraken Pro API request: {str(e)}")
            return {'error': str(e)}
    
    def _get_pro_auth_headers(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Generate authentication headers for the Kraken Pro API.
        
        Args:
            method (str): HTTP method ('GET' or 'POST')
            endpoint (str): API endpoint
            params (Dict, optional): Request parameters
            
        Returns:
            Dict: Authentication headers
        """
        # Create nonce
        nonce = str(int(time.time() * 1000))
        
        # Create post data
        post_data = ""
        if params:
            if method.upper() == 'GET':
                # For GET requests, convert params to query string
                post_data = "&".join([f"{k}={v}" for k, v in params.items()])
            else:
                # For POST requests, convert params to form data
                post_data = "&".join([f"{k}={v}" for k, v in params.items()])
        
        # Create signature
        signature_payload = nonce + endpoint + post_data
        signature = hmac.new(
            base64.b64decode(self.kraken._secret),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Create headers
        headers = {
            'API-Key': self.kraken._key,
            'API-Sign': signature
        }
        
        return headers 