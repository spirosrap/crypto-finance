from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, market_data, orders
from datetime import datetime, timedelta
import time
import uuid

class CoinbaseService:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)

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

    def place_order(self, product_id, side, size):
        return orders.market_order(
            self.client,
            client_order_id=f"spiros_{int(time.time())}",
            product_id=product_id,
            side=side,
            base_size=str(size)
        )
        
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