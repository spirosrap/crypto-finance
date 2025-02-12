from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, orders
from datetime import datetime, timedelta
import time
import uuid
import logging
from typing import Tuple, List
from historicaldata import HistoricalData

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

    def place_order(self, product_id, side, size, order_type="MARKET", price=None, time_in_force="IOC"):
        try:
            # Generate a unique client_order_id
            client_order_id = f"order_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
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

            if order_type.upper() == "MARKET":
                order_func = orders.market_order
            elif order_type.upper() == "LIMIT":
                order_func = orders.limit_order_gtc if time_in_force == "GTC" else orders.limit_order_ioc

            order = order_func(self.client, **order_params)
            return order
        except Exception as e:
            print(f"Error placing order: {e}")
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
        Place a market order with take profit and stop loss targets.
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
            
            # Wait for the market order to fill
            time.sleep(2)
            
            # Get the order details from the response
            # Debug the response type and structure
            self.logger.info(f"Market order type: {type(market_order)}")
            
            # Convert response to dictionary if needed
            market_order_dict = vars(market_order)
            self.logger.info(f"Market order dict: {market_order_dict}")
            
            # Extract order ID from success_response
            if (hasattr(market_order, 'success_response') and 
                isinstance(market_order.success_response, dict) and 
                'order_id' in market_order.success_response):
                order_id = market_order.success_response['order_id']
                self.logger.info(f"Found order ID: {order_id}")
            else:
                self.logger.error(f"Could not find order ID in response: {market_order}")
                return {"error": "Could not find order ID", "market_order": str(market_order)}
            
            # Get the order status
            order_status = self.client.get_order(order_id=order_id)
            self.logger.info(f"Order status response: {order_status}")
            
            # Check if order is filled
            if hasattr(order_status, 'order') and hasattr(order_status.order, 'status') and order_status.order.status == 'FILLED':
                self.logger.info("Market order is filled")
                # Get the fill price
                entry_price = float(order_status.order['average_filled_price'])
                self.logger.info(f"Entry price: {entry_price}")
            else:
                self.logger.error(f"Market order not filled: {order_status}")
                return {"error": "Market order not filled", "market_order": str(market_order)}
            
            # Place take profit limit order
            tp_client_order_id = f"tp_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            tp_side = "BUY" if side.upper() == "SELL" else "SELL"  # Opposite of entry order
            
            # Calculate and round prices based on entry price
            if side.upper() == "SELL":
                # For a short position:
                # - Take profit should be BELOW entry price
                # - Stop loss should be ABOVE entry price
                take_profit_price = round(entry_price * 0.99, 1)  # 1% below entry
                stop_loss_price = round(entry_price * 1.01, 1)   # 1% above entry
            else:
                # For a long position:
                # - Take profit should be ABOVE entry price
                # - Stop loss should be BELOW entry price
                take_profit_price = round(entry_price * 1.01, 1)  # 1% above entry
                stop_loss_price = round(entry_price * 0.99, 1)   # 1% below entry
            
            self.logger.info(f"Entry price: {entry_price}")
            self.logger.info(f"Take profit price: {take_profit_price}")
            self.logger.info(f"Stop loss price: {stop_loss_price}")
            
            # Place take profit order
            self.logger.info(f"Placing take profit order: side={tp_side}, price={take_profit_price}")
            take_profit_order = self.client.limit_order_gtc(
                client_order_id=tp_client_order_id,
                product_id=product_id,
                side=tp_side,
                base_size=str(size),
                limit_price=str(take_profit_price),
                post_only=False,
                leverage=leverage,
                margin_type="CROSS" if leverage else None
            )
            
            if hasattr(take_profit_order, 'error_response'):
                self.logger.error(f"Failed to place take profit order: {take_profit_order.error_response}")
                return {
                    "error": "Failed to place take profit order",
                    "market_order": str(market_order),
                    "tp_error": take_profit_order.error_response
                }
            
            self.logger.info(f"Take profit order placed: {take_profit_order}")
            
            # Place stop loss order
            sl_client_order_id = f"sl_{uuid.uuid4().hex[:16]}_{int(time.time())}"
            
            # For a SELL position:
            # - Stop loss should be ABOVE entry (stop when price rises)
            # - Direction should be STOP_DIRECTION_STOP_UP
            # - Side should be BUY to close the position
            #
            # For a BUY position:
            # - Stop loss should be BELOW entry (stop when price falls)
            # - Direction should be STOP_DIRECTION_STOP_DOWN
            # - Side should be SELL to close the position
            
            sl_side = "BUY" if side.upper() == "SELL" else "SELL"  # Opposite of entry order
            stop_direction = "STOP_DIRECTION_STOP_DOWN" if side.upper() == "BUY" else "STOP_DIRECTION_STOP_UP"
            
            self.logger.info(f"Placing stop loss order: side={sl_side}, price={stop_loss_price}, direction={stop_direction}")
            stop_loss_order = self.client.stop_limit_order_gtc(
                client_order_id=sl_client_order_id,
                product_id=product_id,
                side=sl_side,
                base_size=str(size),
                limit_price=str(stop_loss_price),
                stop_price=str(stop_loss_price),
                stop_direction=stop_direction,
                leverage=leverage,
                margin_type="CROSS" if leverage else None
            )
            
            if hasattr(stop_loss_order, 'error_response'):
                self.logger.error(f"Failed to place stop loss order: {stop_loss_order.error_response}")
                return {
                    "error": "Failed to place stop loss order",
                    "market_order": str(market_order),
                    "take_profit_order": str(take_profit_order),
                    "sl_error": stop_loss_order.error_response
                }
            
            self.logger.info(f"Stop loss order placed: {stop_loss_order}")
            
            # Return all order details
            return {
                "market_order": str(market_order),
                "take_profit_order": str(take_profit_order),
                "stop_loss_order": str(stop_loss_order),
                "status": "success",
                "order_id": order_id,
                "tp_price": take_profit_price,
                "sl_price": stop_loss_price
            }
            
        except Exception as e:
            self.logger.error(f"Error placing market order with targets: {str(e)}")
            return {"error": str(e)}

    def cancel_all_orders(self, product_id: str = None):
        """
        Cancel all open orders for a given product_id or all products if none specified.
        
        Args:
            product_id (str, optional): The trading pair to cancel orders for
        """
        try:
            if product_id:
                self.client.cancel_orders(product_ids=[product_id])
            else:
                self.client.cancel_orders()
            self.logger.info(f"Cancelled all orders for {product_id or 'all products'}")
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {str(e)}")