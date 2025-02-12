from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, orders
from datetime import datetime, timedelta
import time
import uuid
import logging
from typing import Tuple, List
from historicaldata import HistoricalData
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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
            
            # Wait briefly for market order to fill
            time.sleep(2)
            
            # Get the order details and verify it's filled
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
            if not (hasattr(order_status, 'order') and 
                    hasattr(order_status.order, 'status') and 
                    order_status.order.status == 'FILLED'):
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
            for p in ports['portfolios']:  # Access as dictionary
                self.logger.info(f"Portfolio type: {p['type']}, UUID: {p['uuid']}")
                if p['type'] == "INTX":
                    portfolio_uuid = p['uuid']
            
            if not portfolio_uuid:
                self.logger.error("Could not find INTX portfolio")
                return
            
            self.logger.info(f"Using portfolio UUID: {portfolio_uuid}")
            
            # Use list_orders with OPEN status and portfolio UUID
            open_orders = self.client.list_orders(
                order_status="OPEN",
                product_id=product_id,
                portfolio_uuid=portfolio_uuid  # Use UUID instead of portfolio_id
            )
            
            # Debug logging to see what we got back
            self.logger.info(f"Raw response: {open_orders}")
            
            if hasattr(open_orders, 'orders'):
                for order in open_orders.orders:
                    self.logger.info(f"Order found: ID={getattr(order, 'order_id', 'N/A')}, "
                                   f"Status={getattr(order, 'status', 'N/A')}, "
                                   f"Product={getattr(order, 'product_id', 'N/A')}")
            
            if not open_orders or not hasattr(open_orders, 'orders'):
                self.logger.info("No open orders found")
                return
            
            # Log the number of orders found
            order_count = len(open_orders.orders) if hasattr(open_orders, 'orders') else 0
            self.logger.info(f"Found {order_count} open orders")
            
            if order_count == 0:
                return
            
            # Extract order IDs from open orders
            order_ids = [
                order.order_id 
                for order in open_orders.orders 
                if hasattr(order, 'order_id')
            ]
            
            if not order_ids:
                self.logger.info("No active orders to cancel")
                return
            
            self.logger.info(f"Attempting to cancel {len(order_ids)} orders")
            
            # Process orders in batches of 100 (Coinbase API limit)
            batch_size = 100
            total_cancelled = 0
            
            for i in range(0, len(order_ids), batch_size):
                batch = order_ids[i:i + batch_size]
                try:
                    response = self.client.cancel_orders(order_ids=batch)
                    
                    # Count successful cancellations
                    if hasattr(response, 'results'):
                        # Handle response.results being a list of CancelOrderObject
                        successful = sum(1 for r in response.results if hasattr(r, 'success') and r.success)
                        total_cancelled += successful
                        self.logger.info(f"Successfully cancelled {successful} orders from batch")
                    
                except Exception as e:
                    self.logger.error(f"Error cancelling batch: {str(e)}")
                    continue
            
            self.logger.info(f"Cancelled total of {total_cancelled} orders for {product_id or 'all products'}")
            
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {str(e)}")
            self.logger.exception("Full error details:")

    def close_all_positions(self, product_id: str = None):
        """
        Close all open positions for a given product_id or all products if none specified.
        
        Args:
            product_id (str, optional): The trading pair to close positions for
        """
        try:
            self.logger.info("Fetching open positions...")
            
            # First get the INTX portfolio UUID
            self.logger.info("Getting portfolios...")
            ports = self.client.get_portfolios()
            portfolio_uuid = None
            
            for p in ports['portfolios']:
                if p['type'] == "INTX":
                    portfolio_uuid = p['uuid']
                    self.logger.info(f"Found INTX portfolio with UUID: {portfolio_uuid}")
                    break
            
            if not portfolio_uuid:
                self.logger.error("Could not find INTX portfolio")
                return
            
            # Get portfolio positions
            self.logger.info(f"Getting breakdown for portfolio {portfolio_uuid}...")
            portfolio = self.client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid)
            
            self.logger.info("Got portfolio breakdown")
            
            # Access the perp_positions from the response
            if not hasattr(portfolio, 'breakdown'):
                self.logger.error("No breakdown in portfolio response")
                return
            
            if not hasattr(portfolio.breakdown, 'perp_positions'):
                self.logger.error("No perp_positions in portfolio breakdown")
                return
            
            positions = portfolio.breakdown.perp_positions
            self.logger.info(f"Found {len(positions)} perpetual positions")
            
            for position in positions:
                self.logger.info(f"Processing position: {position}")
                
                # Get symbol instead of product_id for perpetual futures
                position_symbol = getattr(position, 'symbol', None)
                if product_id and position_symbol != product_id:
                    self.logger.info(f"Skipping position for {position_symbol}")
                    continue
                
                position_size = float(getattr(position, 'net_size', '0'))
                self.logger.info(f"Position size: {position_size}")
                
                if position_size != 0:
                    side = "BUY" if getattr(position, 'position_side', '') == 'FUTURES_POSITION_SIDE_SHORT' else "SELL"
                    size = abs(position_size)
                    
                    self.logger.info(f"Closing {side} position of size {size} for {position_symbol}")
                    
                    # Place market order to close position
                    result = self.place_order(
                        product_id=position_symbol,
                        side=side,
                        size=size,
                        order_type="MARKET"
                    )
                    
                    self.logger.info(f"Close position result: {result}")
                else:
                    self.logger.info(f"Position size is 0, skipping")
                
        except Exception as e:
            self.logger.error(f"Error closing positions: {str(e)}")
            self.logger.exception("Full error details:")