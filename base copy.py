import time
from datetime import datetime, timedelta
from coinbase.rest import RESTClient
from coinbase import jwt_generator
from coinbase.rest import perpetuals
from coinbase.rest import portfolios
from coinbase.rest import products
from coinbase.rest import orders
from coinbase.rest import market_data

api_key = "organizations/036970ec-9cee-41a6-b035-bc41973e71e5/apiKeys/f7761b3a-2099-4655-aae9-e7a9dc30c7c4"
api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIHgSbP3bEY556865KkU0JGo+VYBM4uxjo26EN9UYkcVloAoGCCqGSM49\nAwEHoUQDQgAExUrvb17nj9l8Hgl8xanJav+WFEzxandjRQ6XJYVA7XN1J0EvyRfM\njSH1KedrYRNyH9yhfRzlx8jjUnMZJ/VZBA==\n-----END EC PRIVATE KEY-----\n"

client = RESTClient(api_key=api_key, api_secret=api_secret)

def main():
    # jwt_token = jwt_generator.build_ws_jwt(api_key, api_secret)
    # print(f"export JWT={jwt_token}")
    # print(client.get_accounts())
    # print(portfolios.get_portfolios(client))
    ports = portfolios.get_portfolios(client)["portfolios"]
    # print(ports)
    uuid = ""
    fiat_usd = 0.0
    btc = 0.0
    for p in ports:
        if p["type"] == "DEFAULT":
            uuid = p["uuid"]

            breakdown = portfolios.get_portfolio_breakdown(client,portfolio_uuid = uuid)
            # print(breakdown["breakdown"]["spot_positions"])
            spot = breakdown["breakdown"]["spot_positions"]
            for s in spot:
                if s["asset"] == "BTC":
                    print("BTC:")
                    print("$" + str("{:.2f}".format(s["total_balance_fiat"])), str(s["total_balance_crypto"]) + " btc")
                    fiat_usd = float(s["total_balance_fiat"])
                    btc = float(s["total_balance_crypto"])                                        
                    break            
    # print(perpetuals.list_perps_positions(client,"018b9290-3810-7910-aa8b-24b5f14e980b"))
    # print(products.get_best_bid_ask(client)["pricebooks"])
    for p in products.get_best_bid_ask(client)["pricebooks"]:
        if p["product_id"] == "BTC-EUR":
            print("€""{:.2f}".format(btc*float(p["bids"][0]["price"])) )
            btc_euro = float(p["bids"][0]["price"])
                      
            btc_euro_ask = float(p["asks"][0]["price"])
            # print("\033[32mThis text is green!\033[0m")
            print("€"+"\033[1;32m{:.2f}\033[0m".format(0.00187597*btc_euro_ask) + "  0.00187597 btc" )
            # print("{:.2f}".format(btc_euro) + " euro" )

        if p["product_id"] == "BTC-USD":
            btc_usd = float(p["bids"][0]["price"])
            btc_usd_ask = float(p["asks"][0]["price"])
            print("$"+"{:.2f}".format(btc_usd))

    def get_hourly_data(client, product_id):
        # Get the current time
        end_time = datetime.now()
        # Get the time 300 hours ago (as we need 300 data points for RSI)
        start_time = end_time - timedelta(hours=300)
        
        # Format times for the API call
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # Get candle data
        candles = market_data.get_candles(
            client,
            product_id=product_id,
            start=start_timestamp,
            end=end_timestamp,
            granularity="ONE_HOUR"
        )
        
        # Extract closing prices
        closing_prices = [float(candle['close']) for candle in candles['candles']]
        
        return closing_prices

    def compute_rsi(client, product_id, period=14):
        closing_prices = get_hourly_data(client, product_id)
        
        if len(closing_prices) < period + 1:
            raise ValueError("Not enough data points to calculate RSI")
        
        # Calculate price changes
        deltas = [closing_prices[i] - closing_prices[i-1] for i in range(1, len(closing_prices))]
        
        # Separate gains and losses
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        # Calculate average gains and losses
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Calculate RSI
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    # Example usage:
    rsi = compute_rsi(client, "BTC-USD")
    print(f"Current RSI for BTC-USD: {rsi:.2f}")

    def generate_signal(rsi):
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        else:
            return "HOLD"

    # Generate signal based on computed RSI
    signal = generate_signal(rsi)
    print(f"Signal for BTC-USD: {signal}")

    # print(products.get_products(client))

    # print(orders.market_order(client, client_order_id = f"spiros_{int(time.time())}", product_id = "MATIC-USDC", side = "SELL", 
    #     base_size = "1"))

    

    # base_size: The amount of the first Asset in the Trading Pair
    # quote_size: The amount of the second Asset in the Trading Pair. BTC/USD Order Book, USD is the Quote Asset.
    # Market sells must be parameterized in base currency

    # {'success': False, 'failure_reason': 'UNKNOWN_FAILURE_REASON', 'order_id': '', 
    # 'error_response': {'error': 'INSUFFICIENT_FUND', 'message': 'Insufficient balance in source account',
    #  'error_details': '', 'preview_failure_reason': 'PREVIEW_INSUFFICIENT_FUND'}, 'order_configuration': 
    #  {'market_market_ioc': {'quote_size': '1'}}}

    # 2024-08-30 12:48:19 - coinbase.RESTClient - ERROR - HTTP Error: 400 Client Error: Bad Request 
    # {"error":"INVALID_ARGUMENT","error_details":"account is not available","message":"account is not available"}


if __name__ == "__main__":
    main()