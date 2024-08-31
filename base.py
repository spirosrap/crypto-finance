import time
from datetime import datetime, timedelta
from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, market_data, orders
import numpy as np
import pandas as pd
# ... existing imports ...

class CryptoTrader:
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

    def get_hourly_data(self, product_id):
        end = int(datetime.utcnow().timestamp())
        start = end - 86400  # 24 hours in seconds
        try:
            candles = market_data.get_candles(
                self.client,
                product_id=product_id,
                start=start,
                end=end,
                granularity="ONE_HOUR"
            )
            return [float(candle['close']) for candle in candles['candles']]
        except requests.exceptions.HTTPError as e:
            print(f"Error fetching candle data: {e}")
            return []

    def compute_rsi(self, product_id, period=14):
        prices = self.get_hourly_data(product_id)
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi[-1]
        
    def compute_macd(self, product_id):
        prices = np.array(self.get_hourly_data(product_id))
        ema12 = self.exponential_moving_average(prices, 12)
        ema26 = self.exponential_moving_average(prices, 26)
        macd = ema12 - ema26
        signal = self.exponential_moving_average(macd, 9)
        histogram = macd - signal
        return macd[-1], signal[-1], histogram[-1]

    def exponential_moving_average(self, data, span):
        return pd.Series(data).ewm(span=span, adjust=False).mean().values

    def generate_combined_signal(self, rsi, macd, signal, histogram):
        rsi_signal = self.generate_signal(rsi)
        
        if macd > signal and histogram > 0:
            macd_signal = "BUY"
        elif macd < signal and histogram < 0:
            macd_signal = "SELL"
        else:
            macd_signal = "HOLD"
        
        if rsi_signal == macd_signal:
            return rsi_signal
        elif rsi_signal == "HOLD" or macd_signal == "HOLD":
            return "HOLD"
        else:
            return "CONFLICTING"



    def generate_signal(self, rsi):
        if rsi > 70:
            return "SELL"
        elif rsi < 30:
            return "BUY"
        else:
            return "HOLD"

    def place_order(self, product_id, side, size):
        return orders.market_order(
            self.client,
            client_order_id=f"spiros_{int(time.time())}",
            product_id=product_id,
            side=side,
            base_size=str(size)
        )

def main():
    api_key = "organizations/036970ec-9cee-41a6-b035-bc41973e71e5/apiKeys/f7761b3a-2099-4655-aae9-e7a9dc30c7c4"
    api_secret = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIHgSbP3bEY556865KkU0JGo+VYBM4uxjo26EN9UYkcVloAoGCCqGSM49\nAwEHoUQDQgAExUrvb17nj9l8Hgl8xanJav+WFEzxandjRQ6XJYVA7XN1J0EvyRfM\njSH1KedrYRNyH9yhfRzlx8jjUnMZJ/VZBA==\n-----END EC PRIVATE KEY-----\n"
    

    trader = CryptoTrader(api_key, api_secret)

    fiat_usd, btc = trader.get_portfolio_info()
    print(f"BTC: ${fiat_usd:.2f}, {btc} btc")

    prices = trader.get_btc_prices()
    for currency, price in prices.items():
        print(f"{currency}: Bid: {price['bid']:.2f}, Ask: {price['ask']:.2f}")

    rsi = trader.compute_rsi("BTC-USD")
    print(f"Current RSI for BTC-USD: {rsi:.2f}")

    signal = trader.generate_signal(rsi)
    print(f"Signal for BTC-USD: {signal}")
    macd, signal, histogram = trader.compute_macd("BTC-USD")
    print(f"Current MACD for BTC-USD: MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {histogram:.2f}")
    
    # Calculate the value of 0.00187597 BTC in EUR including fees
    btc_amount = 0.00187597
    btc_eur_price = trader.get_btc_prices().get('BTC-EUR', {}).get('bid', 0)
    
    # Get the actual fee percentage based on transaction summary
    transaction_summary = trader.client.get_transaction_summary()
    fee_tier = transaction_summary.get('fee_tier', {})
    fee_percentage = fee_tier.get('taker_fee_rate', 0)
    
    eur_value_before_fees = btc_amount * btc_eur_price
    fees = float(eur_value_before_fees) * float(fee_percentage)
    eur_value_after_fees = eur_value_before_fees - fees
    
    print(f"0.00187597 BTC is worth approximately {eur_value_after_fees:.2f} EUR (including {float(fee_percentage)*100}% fees)")
    print(f"Fees: {fees:.2f} EUR")
    print(f"Value before fees: {eur_value_before_fees:.2f} EUR")

    combined_signal = trader.generate_combined_signal(rsi, macd, signal, histogram)
    print(f"Combined signal for BTC-USD: {combined_signal}")

    # Uncomment to place an order
    # if signal == "BUY":
    #     trader.place_order("BTC-USD", "BUY", 0.001)
    # elif signal == "SELL":
    #     trader.place_order("BTC-USD", "SELL", 0.001)

if __name__ == "__main__":
    main()