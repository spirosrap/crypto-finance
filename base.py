import time
from datetime import datetime, timedelta
from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, market_data, orders
import numpy as np
import pandas as pd
import requests
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from typing import List, Tuple
from config import API_KEY, API_SECRET, NEWS_API_KEY
#nltk.download('vader_lexicon')

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
        start = end - 86400*12  # 12*24 hours in seconds
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
    
    def identify_trend(self, product_id, window=20):
        prices = self.get_hourly_data(product_id)
        if len(prices) < window:
            return "Not enough data"
        
        # Calculate Simple Moving Average
        sma = np.convolve(prices, np.ones(window), 'valid') / window
        
        # Calculate the slope of the SMA
        slope = np.gradient(sma)
        
        # Determine the trend based on the recent slope
        recent_slope = slope[-5:].mean()  # Use the average of the last 5 slope values
        
        if recent_slope > 0.01:  # Threshold can be adjusted
            return "Uptrend"
        elif recent_slope < -0.01:  # Threshold can be adjusted
            return "Downtrend"
        else:
            return "Sideways"


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
        # print(macd_signal, rsi_signal)
        if rsi_signal == macd_signal:
            return rsi_signal
        elif rsi_signal == "HOLD" or macd_signal == "HOLD":
            return "HOLD"
        else:
            return "CONFLICTING"

    def analyze_sentiment(self, keyword):
        try:
            # Using NewsAPI to fetch recent news articles
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            articles = newsapi.get_everything(q=keyword, language='en', sort_by='publishedAt', page_size=100)
            
            # Initialize NLTK's VADER sentiment analyzer
            sia = SentimentIntensityAnalyzer()
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles['articles']:
                sentiment = sia.polarity_scores(article['title'] + ' ' + article['description'])
                sentiments.append(sentiment['compound'])
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiments) / len(sentiments)
            
            # Interpret the sentiment
            if avg_sentiment > 0.05:
                return "Positive"
            elif avg_sentiment < -0.05:
                return "Negative"
            else:
                return "Neutral"
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "Unable to analyze"


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

        
    def place_bracket_order(self, product_id, size, limit_price, stop_trigger_price):
        try:
            end_time = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            return orders.trigger_bracket_order_gtd_sell(
                self.client,
                client_order_id=f"spiros_bracket_{int(time.time())}",
                product_id=product_id,
                base_size=str(size),
                limit_price=str(limit_price),
                stop_trigger_price=str(stop_trigger_price),
                end_time=end_time
            )
        except Exception as e:
            print(f"Error placing bracket order: {e}")
            return None

    def monitor_price_and_place_bracket_order(self, product_id, target_price, size):
        max_retries = 1
        retry_delay = 60  # seconds

        print(f"Placing bracket order with target price {target_price}.")
        for attempt in range(max_retries):
            order = self.place_bracket_order(product_id, size, target_price * 1.02, target_price * 0.98)
            if order["success"] == True:
                print(order, "Bracket order placed successfully.")
                return
            else:
                print(order, "Failed to place order.")
                return
            print(f"Failed to place order. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        print("Max retries reached. Unable to place bracket order.")

    def backtest(self, product_id: str, start_date: datetime, end_date: datetime, initial_balance: float) -> Tuple[float, List[dict]]:
        try:
            # Fetch historical data
            print(f"Fetching historical data from {start_date} to {end_date}...")
            candles = self.get_historical_data(product_id, start_date, end_date)
            print(f"Fetched {len(candles)} candles.")

            if not candles:
                print("No historical data available for backtesting.")
                return initial_balance, []

            balance = initial_balance
            btc_balance = 0
            trades = []

            for i, candle in enumerate(candles):
                close_price = float(candle['close'])
                
                # Calculate indicators
                rsi = self.compute_rsi_for_backtest(candles[:i+1])
                macd, signal, histogram = self.compute_macd_for_backtest(candles[:i+1])
                
                # Generate signal
                combined_signal = self.generate_combined_signal(rsi, macd, signal, histogram)
                
                # Execute trade based on signal
                if combined_signal == "BUY" and balance > 0:
                    btc_to_buy = balance / close_price
                    balance = 0
                    btc_balance += btc_to_buy
                    trades.append({
                        'date': candle['start'],
                        'action': 'BUY',
                        'price': close_price,
                        'amount': btc_to_buy
                    })
                elif combined_signal == "SELL" and btc_balance > 0:
                    balance += btc_balance * close_price
                    btc_balance = 0
                    trades.append({
                        'date': candle['start'],
                        'action': 'SELL',
                        'price': close_price,
                        'amount': btc_balance
                    })

            # Calculate final portfolio value
            final_value = balance + (btc_balance * float(candles[-1]['close']))
            return final_value, trades
        except Exception as e:
            print(f"An error occurred during backtesting: {e}")
            return initial_balance, []

    def get_historical_data(self, product_id: str, start_date: datetime, end_date: datetime) -> List[dict]:
        all_candles = []
        current_start = start_date
        chunk_size = timedelta(hours=300)  # Fetch 300 hours (less than 350 candles) at a time

        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            start = int(current_start.timestamp())
            end = int(current_end.timestamp())

            try:
                candles = market_data.get_candles(
                    self.client,
                    product_id=product_id,
                    start=start,
                    end=end,
                    granularity="ONE_HOUR"
                )
                all_candles.extend(candles['candles'])
                current_start = current_end

                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except requests.exceptions.HTTPError as e:
                print(f"Error fetching candle data: {e}")
                # You might want to add more sophisticated error handling here

        return all_candles

    def compute_rsi_for_backtest(self, candles: List[dict], period: int = 14) -> float:
        prices = [float(candle['close']) for candle in candles]
        return self.compute_rsi_from_prices(prices, period)

    def compute_macd_for_backtest(self, candles: List[dict]) -> Tuple[float, float, float]:
        prices = [float(candle['close']) for candle in candles]
        return self.compute_macd_from_prices(prices)

    def compute_rsi_from_prices(self, prices: List[float], period: int = 14) -> float:
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        
        # Avoid division by zero
        if down == 0:
            rs = float('inf')
        else:
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
            
            # Avoid division by zero
            if down == 0:
                rs = float('inf')
            else:
                rs = up/down
            
            rsi[i] = 100. - 100./(1. + rs)

        return rsi[-1]

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        ema12 = self.exponential_moving_average(prices, 12)
        ema26 = self.exponential_moving_average(prices, 26)
        macd = ema12 - ema26
        signal = self.exponential_moving_average(macd, 9)
        histogram = macd - signal
        return macd[-1], signal[-1], histogram[-1]


def main():
    api_key = API_KEY
    api_secret = API_SECRET
    

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
    # print(f"Value before fees: {eur_value_before_fees:.2f} EUR")

    combined_signal = trader.generate_combined_signal(rsi, macd, signal, histogram)
    print(f"Combined signal for BTC-USD: {combined_signal}")

    # Uncomment to place an order
    # if signal == "BUY":
    #     trader.place_order("BTC-USD", "BUY", 0.001)
    # elif signal == "SELL":
    #     trader.place_order("BTC-USD", "SELL", 0.001)

    # trader.monitor_price_and_place_bracket_order("BTC-EUR", 60000, btc_amount)
    
    trend = trader.identify_trend("BTC-USD") 
    print(f"Current 1h (12 days) trend for BTC-USD: {trend}")

    bitcoin_sentiment = trader.analyze_sentiment("Bitcoin")
    print(f"Current sentiment for Bitcoin: {bitcoin_sentiment}")
    
    # Add backtesting
    # Bear market 2021: from $69000 to $15000
    start_date = datetime(2021, 11, 1)
    end_date = datetime(2022, 11, 1)
    initial_balance = 10000  # USD

    final_value, trades = trader.backtest("BTC-USD", start_date, end_date, initial_balance)
    
    print(f"Backtesting results:")
    print(f"Initial balance: ${initial_balance}")
    print(f"Final portfolio value: ${final_value:.2f}")
    print(f"Total return: {(final_value - initial_balance) / initial_balance * 100:.2f}%")
    print(f"Number of trades: {len(trades)}")
    print("Trades executed during backtesting:")
    
    for trade in trades:
        print(f"Date: {trade['date']}, Action: {trade['action']}, Price: {trade['price']}, Amount: {trade['amount']}")

    # You can further analyze the trades list for more insights


if __name__ == "__main__":
    main()
    