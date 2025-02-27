from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, orders
from datetime import datetime, timedelta
import time
import uuid
import logging
from typing import Tuple, List
from historicaldata import HistoricalData
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from concurrent.futures import wait

class CoinbaseService:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.DEFAULT_FEE_RATE = 0.005  # 0.5%
        self.MAX_RETRIES = 1
        self.RETRY_DELAY_SECONDS = 60
        self.BRACKET_ORDER_TAKE_PROFIT_MULTIPLIER = 1.02
        self.BRACKET_ORDER_STOP_LOSS_MULTIPLIER = 0.98
        self.historical_data = HistoricalData(self.client)  # Initialize HistoricalData
        self.logger = logging.getLogger(__name__)

    def get_portfolio_info(self, portfolio_type="DEFAULT"):
        """
        Get portfolio information for either DEFAULT or PERPETUALS portfolio.
        
        Args:
            portfolio_type (str): Type of portfolio to query ("DEFAULT" or "PERPETUALS")
            
        Returns:
            Tuple[float, float]: (fiat_balance, crypto_balance) for spot positions
                                or (usd_balance, perp_position_size) for perpetuals
        """
        try:
            ports = portfolios.get_portfolios(self.client)["portfolios"]
            # Log available portfolio types for debugging
            available_types = [p["type"] for p in ports]
            self.logger.debug(f"Available portfolio types: {available_types}")
            
            for p in ports:
                if p["type"] == portfolio_type:
                    uuid = p["uuid"]
                    breakdown = portfolios.get_portfolio_breakdown(self.client, portfolio_uuid=uuid)
                    if portfolio_type == "DEFAULT":
                        spot = breakdown["breakdown"]["spot_positions"]
                        
                        # Initialize balances
                        fiat_balance = 0.0
                        crypto_balance = 0.0
                        
                        for position in spot:
                            if position["asset"] == "BTC":
                                fiat_balance = float(position["total_balance_fiat"])
                                crypto_balance = float(position["total_balance_crypto"])
                                break
                        
                        self.logger.info(f"Retrieved {portfolio_type} portfolio - "
                                       f"Fiat: {fiat_balance}, Crypto: {crypto_balance}")
                        return fiat_balance, crypto_balance
                    
                    elif portfolio_type == "INTX":
                        perps = breakdown["breakdown"]["portfolio_balances"]
                        
                        # Initialize perpetual values
                        usd_balance = float(perps["total_balance"]["value"])
                        perp_position_size = 0.0
                        
                        
                        self.logger.info(f"Retrieved {portfolio_type} portfolio - "
                                       f"USD Balance: {usd_balance}, Position Size: {perp_position_size}")
                        return usd_balance, perp_position_size
            
            self.logger.warning(f"Portfolio type {portfolio_type} not found")
            return 0.0, 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio info: {str(e)}")
            return 0.0, 0.0

    def get_btc_prices(self):
        prices = {}
    
        for p in products.get_best_bid_ask(self.client)["pricebooks"]:
            if p["product_id"] in ["BTC-EUR", "BTC-USDC"]:
                prices[p["product_id"]] = {
                    "bid": float(p["bids"][0]["price"]),
                    "ask": float(p["asks"][0]["price"])
                }
        return prices

    def place_order(self, product_id: str, side: str, size: float, order_type: str = "MARKET", price: float = None, time_in_force: str = "IOC"):
        """
        Place an order with the specified parameters.
        
        Args:
            product_id (str): The trading pair
            side (str): "BUY" or "SELL"
            size (float): The amount to trade
            order_type (str): "MARKET" or "LIMIT"
            price (float, optional): Required for LIMIT orders
            time_in_force (str): Time in force policy (default "IOC")
        """
        try:
            # Generate a unique client_order_id
            client_order_id = f"order_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # Check if this is a perpetual product
            is_perpetual = "-PERP-" in product_id
            
            if is_perpetual:
                # For perpetual futures, use create_market_order_perp
                if order_type.upper() == "MARKET":
                    order_config = {
                        "market_market_ioc": {
                            "base_size": str(size)
                        }
                    }
                    
                    self.logger.info(f"Placing perpetual {side} market order for {size} {product_id}")
                    market_order = self.client.create_order(
                        client_order_id=client_order_id,
                        product_id=product_id,
                        side=side.upper(),
                        order_configuration=order_config
                    )
                    self.logger.info(f"Perpetual market order response: {market_order}")
                    return market_order
                    
                else:
                    raise ValueError("Only MARKET orders supported for perpetual futures currently")
            else:
                # Original order logic for spot trading
                order_params = {
                    "client_order_id": client_order_id,
                    "product_id": product_id,
                    "side": side.upper(),
                    "order_configuration": {
                        order_type.lower(): {
                            "quote_size" if side.upper() == "BUY" else "base_size": str(size)
                        }
                    }
                }

                if order_type.upper() == "LIMIT":
                    if price is None:
                        raise ValueError("Price must be specified for LIMIT orders")
                    order_params["order_configuration"]["limit"]["limit_price"] = str(price)
                    order_params["order_configuration"]["limit"]["post_only"] = False
                    order_params["order_configuration"]["limit"]["time_in_force"] = time_in_force

                # Place the order
                self.logger.info(f"Placing order with params: {order_params}")
                return self.client.create_order(**order_params)
                
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None

    def place_bracket_order(self, product_id, side, size, entry_price, take_profit_price, stop_loss_price):
        try:
            # Generate a unique client_order_id
            client_order_id = f"bracket_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # Set end time to 30 days from now
            end_time = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"

            if side.upper() == "BUY":
                order = orders.trigger_bracket_order_gtd_buy(
                    self.client,
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(size),
                    limit_price=str(entry_price),
                    stop_trigger_price=str(stop_loss_price),
                    take_profit_price=str(take_profit_price),
                    end_time=end_time
                )
            elif side.upper() == "SELL":
                order = orders.trigger_bracket_order_gtd_sell(
                    self.client,
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size=str(size),
                    limit_price=str(entry_price),
                    stop_trigger_price=str(stop_loss_price),
                    take_profit_price=str(take_profit_price),
                    end_time=end_time
                )
            else:
                raise ValueError("Invalid side. Must be 'BUY' or 'SELL'.")

            return order
        except Exception as e:
            print(f"Error placing bracket order: {e}")
            return None            
 
    def calculate_trade_amount_and_fee(self, balance: float, price: float, is_buy: bool) -> Tuple[float, float]:
        """
        Calculate the trade amount and fee for a given balance and price.
        
        :param balance: The available balance for the trade
        :param price: The current price of the asset
        :param is_buy: True if it's a buy order, False if it's a sell order
        :return: A tuple of (trade_amount, fee)
        """
        # Return zeros if balance is too low
        if balance < 5:
            return 0.0, 0.0
        
        try:
            # Get the transaction summary which includes fee rates
            summary = self.client.get_transaction_summary()
            
            if hasattr(summary, 'fee_tier'):
                fee_tier = summary.fee_tier
                
                # Try to get the fee rate from the fee_tier object
                try:
                    if isinstance(fee_tier, dict):
                        if 'taker_fee_rate' in fee_tier:
                            fee_rate = float(fee_tier['taker_fee_rate'])
                            # self.logger.info(f"Using taker fee rate: {fee_rate}")
                        elif 'maker_fee_rate' in fee_tier:
                            fee_rate = float(fee_tier['maker_fee_rate'])
                            self.logger.info(f"Using maker fee rate: {fee_rate}")
                        else:
                            fee_rate = self.DEFAULT_FEE_RATE
                            self.logger.warning("No fee rate found in fee_tier dictionary")
                    else:
                        fee_rate = self.DEFAULT_FEE_RATE
                        self.logger.warning("Fee tier is not a dictionary")
                except Exception as e:
                    self.logger.warning(f"Error accessing fee rate: {str(e)}")
                    fee_rate = self.DEFAULT_FEE_RATE
            else:
                fee_rate = self.DEFAULT_FEE_RATE
                self.logger.warning(f"No fee_tier attribute found in summary")
            
            # self.logger.info(f"Using fee rate: {fee_rate}")
                
        except Exception as e:
            self.logger.warning(f"Could not get fee rates, using default fee rate. Error: {str(e)}")
            fee_rate = self.DEFAULT_FEE_RATE
        
        if is_buy:
            trade_amount = (balance / price) / (1 + fee_rate)
            fee = balance - (trade_amount * price)
        else:
            fee = balance * fee_rate
            trade_amount = balance - fee
        
        return trade_amount, fee

    def monitor_price_and_place_bracket_order(self, product_id, target_price, size):
        logger = logging.getLogger(__name__)
        logger.info(f"Placing bracket order with target price {target_price}.")
        
        for attempt in range(self.MAX_RETRIES):
            # Assuming we want to place a buy order when monitoring price
            order = self.place_bracket_order(
                product_id=product_id,
                side="BUY",  # Added missing side parameter
                size=size,
                entry_price=target_price,  # Use target_price as entry_price
                take_profit_price=target_price * self.BRACKET_ORDER_TAKE_PROFIT_MULTIPLIER,
                stop_loss_price=target_price * self.BRACKET_ORDER_STOP_LOSS_MULTIPLIER
            )
            
            # Check if order is not None before accessing success key
            if order and order.get("success", False):
                logger.info(f"Bracket order placed successfully: {order}")
                return
            else:
                logger.error(f"Failed to place order: {order}")
                if attempt < self.MAX_RETRIES - 1:  # Only sleep if we're going to retry
                    time.sleep(self.RETRY_DELAY_SECONDS)
                continue

        logger.info("Max retries reached. Unable to place bracket order.")

    def get_trading_pairs(self) -> List[str]:
        """
        Get list of available trading pairs from Coinbase.
        
        Returns:
            List[str]: List of available trading pairs (e.g., ['BTC-USDC', 'ETH-USDC', ...])
        """
        try:
            # Get all products using the public endpoint
            response = self.client.get_public_products()
            
            # Filter for active USDC pairs
            usdc_pairs = []
            for product in response['products']:
                # Check if product is active and is a USDC pair
                usdc_pairs.append(product['product_id'])
            
            # Sort pairs alphabetically
            usdc_pairs.sort()
            
            self.logger.info(f"Found {len(usdc_pairs)} active USDC trading pairs")
            
            # Print first few pairs for verification
            if usdc_pairs:
                self.logger.debug(f"Sample pairs: {', '.join(usdc_pairs[:5])}")
            
            return usdc_pairs
            
        except Exception as e:
            self.logger.error(f"Error getting trading pairs: {str(e)}")
            return []

    def place_market_order_with_targets(self, product_id: str, side: str, size: float, 
                                      take_profit_price: float, stop_loss_price: float,
                                      leverage: str = None) -> dict:
        """
        Place a market order followed by a bracket order for take profit and stop loss.
        """
        try:
            # First preview the market order
            preview = self.client.preview_market_order(
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                leverage=leverage,
                margin_type="CROSS" if leverage else None
            )
            
            self.logger.info(f"Order preview response: {preview}")
            
            # Check preview response
            if hasattr(preview, 'error_response'):
                self.logger.error(f"Order preview failed: {preview.error_response}")
                return {"error": preview.error_response}
                
            # Generate a unique client_order_id
            client_order_id = f"market_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # Place the initial market order
            market_order = self.client.market_order(
                client_order_id=client_order_id,
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                leverage=leverage,
                margin_type="CROSS" if leverage else None
            )
            
            if hasattr(market_order, 'error_response'):
                self.logger.error(f"Failed to place market order: {market_order.error_response}")
                return {"error": market_order.error_response}
            
            self.logger.info(f"Market order placed: {market_order}")
            
            # Extract order ID - Updated this section
            order_id = None
            if (isinstance(market_order, dict) and 
                'success_response' in market_order and 
                'order_id' in market_order['success_response']):
                order_id = market_order['success_response']['order_id']
            elif hasattr(market_order, 'success_response'):
                success_response = getattr(market_order, 'success_response')
                if isinstance(success_response, dict) and 'order_id' in success_response:
                    order_id = success_response['order_id']
            
            if not order_id:
                self.logger.error(f"Could not find order ID in response: {market_order}")
                return {"error": "Could not find order ID", "market_order": str(market_order)}
            
            self.logger.info(f"Extracted order ID: {order_id}")
            
            # Wait briefly for market order to fill
            time.sleep(2)
            
            # Get the order status
            order_status = self.client.get_order(order_id=order_id)
            self.logger.info(f"Order status response: {order_status}")
            
            # Check if order is filled - Updated this section
            is_filled = False
            if isinstance(order_status, dict) and 'order' in order_status:
                is_filled = order_status['order'].get('status') == 'FILLED'
            elif hasattr(order_status, 'order'):
                order = getattr(order_status, 'order')
                is_filled = getattr(order, 'status', None) == 'FILLED'
            
            if not is_filled:
                self.logger.error(f"Market order not filled: {order_status}")
                return {"error": "Market order not filled", "market_order": str(market_order)}

            # Generate client_order_id for bracket order
            bracket_client_order_id = f"bracket_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # Set end time to 30 days from now for GTD orders
            end_time = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
            
            # Place bracket order - use opposite side of market order
            bracket_side = "SELL" if side.upper() == "BUY" else "BUY"
            
            # Place the bracket order
            try:
                if bracket_side == "SELL":
                    bracket_order = self.client.trigger_bracket_order_gtd_sell(
                        client_order_id=bracket_client_order_id,
                        product_id=product_id,
                        base_size=str(size),
                        limit_price=str(take_profit_price),
                        stop_trigger_price=str(stop_loss_price),
                        end_time=end_time,
                        leverage=leverage,
                        margin_type="CROSS" if leverage else None
                    )
                else:
                    bracket_order = self.client.trigger_bracket_order_gtd_buy(
                        client_order_id=bracket_client_order_id,
                        product_id=product_id,
                        base_size=str(size),
                        limit_price=str(take_profit_price),
                        stop_trigger_price=str(stop_loss_price),
                        end_time=end_time,
                        leverage=leverage,
                        margin_type="CROSS" if leverage else None
                    )
                    
                if hasattr(bracket_order, 'error_response'):
                    self.logger.error(f"Failed to place bracket order: {bracket_order.error_response}")
                    return {
                        "error": "Failed to place bracket order",
                        "market_order": str(market_order),
                        "bracket_error": bracket_order.error_response
                    }
                    
                self.logger.info(f"Bracket order placed: {bracket_order}")
                
                # Return all order details
                return {
                    "market_order": str(market_order),
                    "bracket_order": str(bracket_order),
                    "status": "success",
                    "order_id": order_id,
                    "tp_price": take_profit_price,
                    "sl_price": stop_loss_price
                }
                
            except Exception as e:
                self.logger.error(f"Error placing bracket order: {str(e)}")
                return {
                    "error": f"Failed to place bracket order: {str(e)}",
                    "market_order": str(market_order)
                }
                
        except Exception as e:
            self.logger.error(f"Error placing market order with targets: {str(e)}")
            return {"error": str(e)}

    def place_limit_order_with_targets(self, product_id: str, side: str, size: float, 
                                     entry_price: float, take_profit_price: float, 
                                     stop_loss_price: float, leverage: str = None) -> dict:
        """
        Place a limit order first, then monitor for fill before placing take profit and stop loss orders.
        
        Args:
            product_id (str): The trading pair (e.g., 'BTC-USD')
            side (str): Order side ('BUY' or 'SELL')
            size (float): Size of the order in base currency
            entry_price (float): Limit price for the entry order
            take_profit_price (float): Price to take profit
            stop_loss_price (float): Price to stop loss
            leverage (str, optional): Leverage value for margin trading
            
        Returns:
            dict: Order details including entry order information
        """
        try:
            # First preview the limit order
            preview = self.client.preview_limit_order_gtc(
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                limit_price=str(entry_price),
                leverage=leverage,
                margin_type="CROSS" if leverage else None
            )
            
            self.logger.info(f"Order preview response: {preview}")
            
            # Check preview response
            if hasattr(preview, 'error_response'):
                self.logger.error(f"Order preview failed: {preview.error_response}")
                return {"error": preview.error_response}
                
            # Generate a unique client_order_id
            client_order_id = f"limit_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # Place the initial limit order
            limit_order = self.client.limit_order_gtc(
                client_order_id=client_order_id,
                product_id=product_id,
                side=side.upper(),
                base_size=str(size),
                limit_price=str(entry_price),
                leverage=leverage,
                margin_type="CROSS" if leverage else None
            )
            
            if hasattr(limit_order, 'error_response'):
                self.logger.error(f"Failed to place limit order: {limit_order.error_response}")
                return {"error": limit_order.error_response}
            
            self.logger.info(f"Limit order placed: {limit_order}")
            
            # Extract order ID
            order_id = None
            if (isinstance(limit_order, dict) and 
                'success_response' in limit_order and 
                'order_id' in limit_order['success_response']):
                order_id = limit_order['success_response']['order_id']
            elif hasattr(limit_order, 'success_response'):
                success_response = getattr(limit_order, 'success_response')
                if isinstance(success_response, dict) and 'order_id' in success_response:
                    order_id = success_response['order_id']
            
            if not order_id:
                self.logger.error(f"Could not find order ID in response: {limit_order}")
                return {"error": "Could not find order ID", "limit_order": str(limit_order)}
            
            self.logger.info(f"Extracted order ID: {order_id}")
            
            # Return the limit order details immediately
            return {
                "limit_order": str(limit_order),
                "status": "pending_fill",
                "order_id": order_id,
                "entry_price": entry_price,
                "tp_price": take_profit_price,
                "sl_price": stop_loss_price,
                "message": "Limit order placed. Once filled, you can place take profit and stop loss orders using place_bracket_after_fill method."
            }
                
        except Exception as e:
            self.logger.error(f"Error placing limit order with targets: {str(e)}")
            return {"error": str(e)}

    def place_bracket_after_fill(self, product_id: str, order_id: str, size: float,
                               take_profit_price: float, stop_loss_price: float,
                               leverage: str = None) -> dict:
        """
        Place bracket orders (take profit and stop loss) after a limit order has been filled.
        
        Args:
            product_id (str): The trading pair
            order_id (str): The original limit order ID
            size (float): Size of the position
            take_profit_price (float): Price to take profit
            stop_loss_price (float): Price to stop loss
            leverage (str, optional): Leverage value for margin trading
            
        Returns:
            dict: Bracket order details
        """
        try:
            # Check if the original order is filled
            order_status = self.client.get_order(order_id=order_id)
            self.logger.info(f"Order status response: {order_status}")
            
            # Check if order is filled
            is_filled = False
            if isinstance(order_status, dict) and 'order' in order_status:
                is_filled = order_status['order'].get('status') == 'FILLED'
                side = order_status['order'].get('side')
            elif hasattr(order_status, 'order'):
                order = getattr(order_status, 'order')
                is_filled = getattr(order, 'status', None) == 'FILLED'
                side = getattr(order, 'side', None)
            
            if not is_filled:
                return {
                    "error": "Original limit order not filled yet",
                    "status": "pending_fill",
                    "order_id": order_id
                }
            
            # Generate client_order_id for bracket order
            bracket_client_order_id = f"bracket_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # Set end time to 30 days from now for GTD orders
            end_time = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
            
            # Place bracket order - use opposite side of the filled limit order
            bracket_side = "SELL" if side == "BUY" else "BUY"
            
            # Place the bracket order
            try:
                if bracket_side == "SELL":
                    bracket_order = self.client.trigger_bracket_order_gtd_sell(
                        client_order_id=bracket_client_order_id,
                        product_id=product_id,
                        base_size=str(size),
                        limit_price=str(take_profit_price),
                        stop_trigger_price=str(stop_loss_price),
                        end_time=end_time,
                        leverage=leverage,
                        margin_type="CROSS" if leverage else None
                    )
                else:
                    bracket_order = self.client.trigger_bracket_order_gtd_buy(
                        client_order_id=bracket_client_order_id,
                        product_id=product_id,
                        base_size=str(size),
                        limit_price=str(take_profit_price),
                        stop_trigger_price=str(stop_loss_price),
                        end_time=end_time,
                        leverage=leverage,
                        margin_type="CROSS" if leverage else None
                    )
                    
                if hasattr(bracket_order, 'error_response'):
                    self.logger.error(f"Failed to place bracket order: {bracket_order.error_response}")
                    return {
                        "error": "Failed to place bracket order",
                        "bracket_error": bracket_order.error_response
                    }
                    
                self.logger.info(f"Bracket order placed: {bracket_order}")
                
                # Return bracket order details
                return {
                    "bracket_order": str(bracket_order),
                    "status": "success",
                    "tp_price": take_profit_price,
                    "sl_price": stop_loss_price
                }
                
            except Exception as e:
                self.logger.error(f"Error placing bracket order: {str(e)}")
                return {"error": f"Failed to place bracket order: {str(e)}"}
                
        except Exception as e:
            self.logger.error(f"Error checking order status: {str(e)}")
            return {"error": str(e)}

    def cancel_all_orders(self, product_id: str = None):
        """
        Cancel all open orders for a given product_id or all products if none specified.
        
        Args:
            product_id (str, optional): The trading pair to cancel orders for
        """
        try:
            self.logger.info("Fetching open orders...")
            
            # First get the INTX portfolio UUID
            ports = self.client.get_portfolios()
            portfolio_uuid = None
            
            self.logger.info("Available portfolios:")
            for p in ports['portfolios']:
                self.logger.info(f"Portfolio type: {p['type']}, UUID: {p['uuid']}")
                if p['type'] == "INTX":
                    portfolio_uuid = p['uuid']
            
            if not portfolio_uuid:
                self.logger.error("Could not find INTX portfolio")
                return
            
            self.logger.info(f"Using portfolio UUID: {portfolio_uuid}")
            
            # Get open orders
            open_orders = self.client.list_orders(
                order_status="OPEN",
                product_id=product_id,
                portfolio_uuid=portfolio_uuid
            )
            
            # Convert response to dictionary if needed
            if not isinstance(open_orders, dict):
                if hasattr(open_orders, '__dict__'):
                    open_orders = vars(open_orders)
                else:
                    open_orders = {'orders': []}
            
            self.logger.info(f"Raw response: {open_orders}")
            
            if 'orders' not in open_orders or not open_orders['orders']:
                self.logger.info("No open orders found")
                return
            
            # Extract order IDs and cancel each order individually
            for order in open_orders['orders']:
                # Handle both dictionary and object types
                if isinstance(order, dict):
                    order_id = order.get('order_id')
                    order_type = order.get('order_type')
                    attached_order_id = order.get('attached_order_id')
                    originating_order_id = order.get('originating_order_id')
                else:
                    order_id = getattr(order, 'order_id', None)
                    order_type = getattr(order, 'order_type', None)
                    attached_order_id = getattr(order, 'attached_order_id', None)
                    originating_order_id = getattr(order, 'originating_order_id', None)
                
                if order_id:
                    try:
                        # First try to cancel any bracket orders
                        if order_type == 'BRACKET':
                            self.logger.info(f"Cancelling bracket order {order_id}")
                            # Cancel the bracket order and any attached orders
                            if attached_order_id:
                                self.logger.info(f"Cancelling attached order {attached_order_id}")
                                self.client.cancel_orders(order_ids=[attached_order_id])
                            if originating_order_id:
                                self.logger.info(f"Cancelling originating order {originating_order_id}")
                                self.client.cancel_orders(order_ids=[originating_order_id])
                        
                        # Cancel the main order
                        self.logger.info(f"Cancelling order {order_id}")
                        result = self.client.cancel_orders(order_ids=[order_id])
                        self.logger.info(f"Cancel result for order {order_id}: {result}")
                        
                        # Wait briefly to ensure the cancel is processed
                        time.sleep(0.5)
                        
                    except Exception as e:
                        self.logger.error(f"Error cancelling order {order_id}: {str(e)}")
                        continue
            
            # Verify all orders are cancelled
            time.sleep(1)  # Wait for cancellations to process
            verify_orders = self.client.list_orders(
                order_status="OPEN",
                product_id=product_id,
                portfolio_uuid=portfolio_uuid
            )
            
            # Convert verification response to dictionary if needed
            if not isinstance(verify_orders, dict):
                if hasattr(verify_orders, '__dict__'):
                    verify_orders = vars(verify_orders)
                else:
                    verify_orders = {'orders': []}
            
            remaining_orders = verify_orders.get('orders', [])
            if remaining_orders:
                self.logger.warning(f"Some orders remain uncancelled: {len(remaining_orders)} orders")
                for order in remaining_orders:
                    if isinstance(order, dict):
                        self.logger.warning(f"Uncancelled order: {order.get('order_id')} - {order.get('order_type')}")
                    else:
                        self.logger.warning(f"Uncancelled order: {getattr(order, 'order_id', None)} - {getattr(order, 'order_type', None)}")
            else:
                self.logger.info("All orders successfully cancelled")
            
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {str(e)}")
            self.logger.exception("Full error details:")

    def close_all_positions(self, product_id: str = None, timeout: int = 30):
        """
        Close all open positions for a given product_id or all products if none specified.
        Uses parallel execution and aggressive retry logic for faster closing.
        
        Args:
            product_id (str, optional): The trading pair to close positions for
            timeout (int, optional): Maximum time in seconds to wait for all positions to close
        """
        try:
            # First cancel all open orders - don't wait between cancellations
            self.logger.info("Cancelling all open orders first...")
            self.cancel_all_orders(product_id)
            
            self.logger.info("Fetching open positions...")
            
            # Get the INTX portfolio UUID
            ports = self.client.get_portfolios()
            portfolio_uuid = None
            
            for p in ports['portfolios']:
                if p['type'] == "INTX":
                    portfolio_uuid = p['uuid']
                    break
            
            if not portfolio_uuid:
                self.logger.error("Could not find INTX portfolio")
                return
            
            # Get portfolio positions
            portfolio = self.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
            
            # Initialize positions as empty list
            positions = []
            
            # Convert portfolio to dictionary if needed
            if not isinstance(portfolio, dict):
                if hasattr(portfolio, '__dict__'):
                    portfolio = vars(portfolio)
                else:
                    self.logger.error("Invalid portfolio response format")
                    return
            
            # Handle breakdown object
            breakdown = portfolio.get('breakdown')
            if breakdown is None:
                self.logger.error("No breakdown found in portfolio")
                return
            
            if not isinstance(breakdown, dict):
                if hasattr(breakdown, '__dict__'):
                    breakdown = vars(breakdown)
                    positions = breakdown.get('perp_positions', [])
                elif hasattr(breakdown, 'perp_positions'):
                    positions = breakdown.perp_positions
                else:
                    self.logger.error("Invalid breakdown format")
                    return
            else:
                positions = breakdown.get('perp_positions', [])
            
            # Convert positions to list if it's not already
            if not isinstance(positions, list):
                if hasattr(positions, '__iter__'):
                    positions = list(positions)
                else:
                    self.logger.error("Invalid positions format")
                    return
            
            if not positions:
                self.logger.info("No perpetual positions found")
                return
            
            self.logger.info(f"Found {len(positions)} perpetual positions to close")
            
            def close_single_position(position):
                """Helper function to close a single position with retries"""
                # Handle both dictionary and object types
                if isinstance(position, dict):
                    position_symbol = position.get('symbol')
                    position_size = float(position.get('net_size', '0'))
                    position_side = position.get('position_side', '')
                    leverage = position.get('leverage', '1')
                else:
                    position_symbol = getattr(position, 'symbol', None)
                    position_size = float(getattr(position, 'net_size', '0'))
                    position_side = getattr(position, 'position_side', '')
                    leverage = getattr(position, 'leverage', '1')
                
                if not position_symbol or abs(position_size) <= 0:
                    return
                    
                if product_id and position_symbol != product_id:
                    return
                
                # Try closing with different size percentages if needed
                size_percentages = [1.0, 0.99, 0.98, 0.95]  # Try 100%, 99%, 98%, 95% of position
                
                for size_pct in size_percentages:
                    try:
                        client_order_id = f"close_{uuid.uuid4().hex[:16]}_{int(time.time())}"
                        side = "BUY" if position_side == "FUTURES_POSITION_SIDE_SHORT" else "SELL"
                        close_size = abs(position_size) * size_pct
                        
                        self.logger.info(f"Attempting to close {position_side} position for {position_symbol}: {close_size} ({size_pct*100}%) using {side} order")
                        
                        order_config = {
                            "market_market_ioc": {
                                "base_size": str(close_size)
                            }
                        }
                        
                        result = self.client.create_order(
                            client_order_id=client_order_id,
                            product_id=position_symbol,
                            side=side,
                            order_configuration=order_config,
                            leverage=leverage,
                            margin_type="CROSS"
                        )
                        
                        if isinstance(result, dict) and result.get('success', True):
                            self.logger.info(f"Successfully closed position for {position_symbol} with {size_pct*100}% size")
                            return True
                        
                        error_response = result.get('error_response', {}) if isinstance(result, dict) else str(result)
                        self.logger.warning(f"Failed to close position with {size_pct*100}% size: {error_response}")
                        
                        # If error is not related to insufficient funds, break the loop
                        if 'PREVIEW_INSUFFICIENT_FUNDS' not in str(error_response):
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Error closing position for {position_symbol} with {size_pct*100}% size: {str(e)}")
                        continue
                
                return False
            
            # Use ThreadPoolExecutor for parallel execution
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all positions for closing
                future_to_position = {executor.submit(close_single_position, pos): pos for pos in positions}
                
                # Wait for all positions to close or timeout
                remaining_time = max(0, timeout - (time.time() - start_time))
                done, not_done = wait(future_to_position.keys(), timeout=remaining_time)
                
                # Cancel any remaining futures
                for future in not_done:
                    future.cancel()
                
                # Log results
                success_count = sum(1 for future in done if future.result())
                self.logger.info(f"Successfully closed {success_count} out of {len(positions)} positions")
                
                if not_done:
                    self.logger.warning(f"{len(not_done)} positions did not close within the timeout period")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {str(e)}")
            self.logger.exception("Full error details:")

    def monitor_limit_order_and_place_bracket(self, product_id: str, order_id: str, size: float,
                                          take_profit_price: float, stop_loss_price: float,
                                          leverage: str = None, max_wait_time: int = 3600) -> dict:
        """
        Monitor a limit order until it's filled, then place bracket orders.
        
        Args:
            product_id (str): The trading pair
            order_id (str): The original limit order ID
            size (float): Size of the position
            take_profit_price (float): Price to take profit
            stop_loss_price (float): Price to stop loss
            leverage (str, optional): Leverage value for margin trading
            max_wait_time (int): Maximum time to wait for fill in seconds (default 1 hour)
            
        Returns:
            dict: Order status and bracket order details if filled
        """
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        self.logger.info(f"Starting to monitor limit order {order_id}")
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check order status
                order_status = self.client.get_order(order_id=order_id)
                
                # Extract status and side
                status = None
                side = None
                
                if isinstance(order_status, dict) and 'order' in order_status:
                    status = order_status['order'].get('status')
                    side = order_status['order'].get('side')
                elif hasattr(order_status, 'order'):
                    order = getattr(order_status, 'order')
                    status = getattr(order, 'status', None)
                    side = getattr(order, 'side', None)
                
                self.logger.info(f"Order status: {status}")
                
                if status == 'FILLED':
                    self.logger.info("Limit order filled! Placing bracket orders...")
                    
                    # Generate client_order_id for bracket order
                    bracket_client_order_id = f"bracket_{uuid.uuid4().hex[:16]}_{int(time.time())}"
                    
                    # Set end time to 30 days from now for GTD orders
                    end_time = (datetime.utcnow() + timedelta(days=30)).isoformat() + "Z"
                    
                    # Place bracket order - use opposite side of the filled limit order
                    bracket_side = "SELL" if side == "BUY" else "BUY"
                    
                    try:
                        if bracket_side == "SELL":
                            bracket_order = self.client.trigger_bracket_order_gtd_sell(
                                client_order_id=bracket_client_order_id,
                                product_id=product_id,
                                base_size=str(size),
                                limit_price=str(take_profit_price),
                                stop_trigger_price=str(stop_loss_price),
                                end_time=end_time,
                                leverage=leverage,
                                margin_type="CROSS" if leverage else None
                            )
                        else:
                            bracket_order = self.client.trigger_bracket_order_gtd_buy(
                                client_order_id=bracket_client_order_id,
                                product_id=product_id,
                                base_size=str(size),
                                limit_price=str(take_profit_price),
                                stop_trigger_price=str(stop_loss_price),
                                end_time=end_time,
                                leverage=leverage,
                                margin_type="CROSS" if leverage else None
                            )
                            
                        if hasattr(bracket_order, 'error_response'):
                            self.logger.error(f"Failed to place bracket order: {bracket_order.error_response}")
                            return {
                                "status": "error",
                                "message": "Limit order filled but failed to place bracket orders",
                                "error": bracket_order.error_response
                            }
                            
                        self.logger.info(f"Bracket order placed successfully: {bracket_order}")
                        
                        return {
                            "status": "success",
                            "message": "Limit order filled and bracket orders placed",
                            "bracket_order": str(bracket_order),
                            "tp_price": take_profit_price,
                            "sl_price": stop_loss_price
                        }
                        
                    except Exception as e:
                        self.logger.error(f"Error placing bracket order: {str(e)}")
                        return {
                            "status": "error",
                            "message": "Limit order filled but failed to place bracket orders",
                            "error": str(e)
                        }
                
                elif status == 'CANCELLED' or status == 'EXPIRED' or status == 'FAILED':
                    return {
                        "status": "cancelled",
                        "message": f"Limit order {status.lower()} before fill"
                    }
                
                # Wait before next check
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error monitoring order: {str(e)}")
                return {
                    "status": "error",
                    "message": "Error monitoring limit order",
                    "error": str(e)
                }
        
        return {
            "status": "timeout",
            "message": f"Limit order not filled within {max_wait_time} seconds"
        }

    def get_recent_trades(self):
        """
        Fetch recent trades from the account.
        
        Returns:
            list: List of recent trades with their details
        """
        try:
            # Get the INTX portfolio UUID
            ports = self.client.get_portfolios()
            portfolio_uuid = None
            
            for p in ports['portfolios']:
                if p['type'] == "INTX":
                    portfolio_uuid = p['uuid']
                    break
            
            if not portfolio_uuid:
                self.logger.error("Could not find INTX portfolio")
                return []
            
            # Get recent trades using the orders endpoint
            orders = self.client.list_orders(
                portfolio_uuid=portfolio_uuid,
                order_status="FILLED",  # Only get filled orders
                limit=10  # Get last 10 trades
            )
            
            # Convert orders to list of trade dictionaries
            trades = []
            
            if isinstance(orders, dict) and 'orders' in orders:
                for order in orders['orders']:
                    # Extract the trade details
                    trade_time = order.get('created_time') or order.get('completion_time')
                    if trade_time:
                        trade = {
                            'trade_time': int(datetime.strptime(trade_time, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()),
                            'side': order.get('side', ''),
                            'price': order.get('average_filled_price') or order.get('limit_price'),
                            'size': order.get('filled_size') or order.get('base_size'),
                            'product_id': order.get('product_id', '')
                        }
                        trades.append(trade)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error fetching recent trades: {str(e)}")
            return []