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

    def get_portfolio_info(self):
        ports = portfolios.get_portfolios(self.client)["portfolios"]
        
        for p in ports:
            if p["type"] == "DEFAULT":
                uuid = p["uuid"]
                breakdown = portfolios.get_portfolio_breakdown(self.client, portfolio_uuid=uuid)
                spot = breakdown["breakdown"]["spot_positions"]
                for s in spot:
                    if s["asset"] == "BTC":
                        return float(s["total_balance_fiat"]), float(s["total_balance_crypto"])
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
            
            if 'products' in response:
                for product in response['products']:
                    # Check if product is active and is a USDC pair
                    if (product['quote_currency_id'] == 'USDC' and 
                        product['status'] == 'online' and 
                        not product.get('is_disabled', False) and
                        not product.get('trading_disabled', False)):
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