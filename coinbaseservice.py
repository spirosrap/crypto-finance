from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, market_data, orders
from datetime import datetime, timedelta
import time
import uuid
import logging
from typing import Tuple

class CoinbaseService:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.DEFAULT_FEE_RATE = 0.005  # 0.5%
        self.MAX_RETRIES = 1
        self.RETRY_DELAY_SECONDS = 60
        self.BRACKET_ORDER_TAKE_PROFIT_MULTIPLIER = 1.02
        self.BRACKET_ORDER_STOP_LOSS_MULTIPLIER = 0.98

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
            if p["product_id"] in ["BTC-EUR", "BTC-USD"]:
                prices[p["product_id"]] = {
                    "bid": float(p["bids"][0]["price"]),
                    "ask": float(p["asks"][0]["price"])
                }
        return prices

    def get_hourly_data(self, product_id, days=60):
        end = int(datetime.utcnow().timestamp())
        all_candles = []
        
        for i in range(0, days, 14):
            start = end - 86400 * min(14, days - i)
            try:
                candles = market_data.get_candles(
                    self.client,
                    product_id=product_id,
                    start=start,
                    end=end,
                    granularity="ONE_HOUR"
                )
                all_candles = candles['candles'] + all_candles
                end = start - 1  # Set end to 1 second before start for next iteration
            except requests.exceptions.HTTPError as e:
                print(f"Error fetching candle data: {e}")
                break
        print(f"Fetched {len(all_candles)} candles")
        return [float(candle['close']) for candle in all_candles]
        
    def get_6h_data(self, product_id):
        end = int(datetime.utcnow().timestamp())
        start = end - 86400*30  # 30 days in seconds
        try:
            candles = market_data.get_candles(
                self.client,
                product_id=product_id,
                start=start,
                end=end,
                granularity="SIX_HOUR"
            )
            return [float(candle['close']) for candle in candles['candles']]
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching 6-hour candle data: {e}")
            return []

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
    # ... (other Coinbase-related methods)

    def calculate_trade_amount_and_fee(self, balance: float, price: float, is_buy: bool) -> Tuple[float, float]:
        """
        Calculate the trade amount and fee for a given balance and price.
        
        :param balance: The available balance for the trade
        :param price: The current price of the asset
        :param is_buy: True if it's a buy order, False if it's a sell order
        :return: A tuple of (trade_amount, fee)
        """
        transaction_summary = self.client.get_transaction_summary()
        fee_tier = transaction_summary.get('fee_tier', {})
        fee_rate = float(fee_tier.get('taker_fee_rate', self.DEFAULT_FEE_RATE))
        
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
            order = self.place_bracket_order(
                product_id, 
                size, 
                target_price * self.BRACKET_ORDER_TAKE_PROFIT_MULTIPLIER, 
                target_price * self.BRACKET_ORDER_STOP_LOSS_MULTIPLIER
            )
            if order["success"] == True:
                logger.info(f"{order}, Bracket order placed successfully.")
                return
            else:
                logger.error(f"{order}, Failed to place order.")
                return
            logger.info(f"Failed to place order. Retrying in {self.RETRY_DELAY_SECONDS} seconds...")
            time.sleep(self.RETRY_DELAY_SECONDS)
        logger.info("Max retries reached. Unable to place bracket order.")

    # ... other existing methods ...