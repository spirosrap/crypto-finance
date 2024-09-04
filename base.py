import time
from datetime import datetime, timedelta
from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, market_data, orders
import numpy as np
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from typing import List, Tuple
from config import API_KEY, API_SECRET, NEWS_API_KEY
#nltk.download('vader_lexicon')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from sentimentanalysis import SentimentAnalysis
import os
import json
import logging
from tqdm import tqdm

# Constants
DEFAULT_FEE_RATE = 0.005  # 0.5%
CHUNK_SIZE_HOURS = 300
RETRY_DELAY_SECONDS = 60
MAX_RETRIES = 1
BRACKET_ORDER_TAKE_PROFIT_MULTIPLIER = 1.02
BRACKET_ORDER_STOP_LOSS_MULTIPLIER = 0.98
RSI_PERIOD = 14
BOLLINGER_WINDOW = 20
BOLLINGER_NUM_STD = 2
TREND_WINDOW = 20

# Configure logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CryptoTrader:
    def __init__(self, api_key, api_secret):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.coinbase_service = CoinbaseService(api_key, api_secret)
        self.technical_analysis = TechnicalAnalysis(self.coinbase_service)
        self.sentiment_analysis = SentimentAnalysis()
        logger.info("CryptoTrader initialized with API key.")

    def get_portfolio_info(self):
        logger.debug("Fetching portfolio info.")
        return self.coinbase_service.get_portfolio_info()

    def get_btc_prices(self):
        logger.debug("Fetching BTC prices.")
        return self.coinbase_service.get_btc_prices()

    def get_hourly_data(self, product_id):
        return self.coinbase_service.get_hourly_data(product_id)
 
    def get_6h_data(self, product_id):
        return self.coinbase_service.get_6h_data(product_id)
 
    def compute_rsi(self, product_id, candles, period=RSI_PERIOD):
        return self.technical_analysis.compute_rsi(product_id, candles, period)

    def compute_macd(self, product_id, candles):
        return self.technical_analysis.compute_macd(product_id, candles)

    def exponential_moving_average(self, data, span):
        return self.technical_analysis.exponential_moving_average(data, span)

    def compute_bollinger_bands(self, candles: List[dict], window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_NUM_STD) -> Tuple[float, float, float]:
        return self.technical_analysis.compute_bollinger_bands(candles, window, num_std)

    def generate_bollinger_bands_signal(self, candles: List[dict]) -> str:
        return self.technical_analysis.generate_bollinger_bands_signal(candles)

    def compute_rsi_from_prices(self, prices: List[float], period: int = RSI_PERIOD) -> float:
        return self.technical_analysis.compute_rsi_from_prices(prices, period)

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        return self.technical_analysis.compute_macd_from_prices(prices)

    def analyze_sentiment(self, keyword):
        return self.sentiment_analysis.analyze_sentiment(keyword)

    def compute_rsi_for_backtest(self, candles: List[dict], period: int = RSI_PERIOD) -> float:
        prices = [float(candle['close']) for candle in candles]
        return self.compute_rsi_from_prices(prices, period)

    def compute_macd_for_backtest(self, candles: List[dict]) -> Tuple[float, float, float]:
        prices = [float(candle['close']) for candle in candles]
        return self.compute_macd_from_prices(prices)

    def compute_bollinger_bands_for_backtest(self, candles: List[dict], window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_NUM_STD) -> Tuple[float, float, float]:
        prices = [float(candle['close']) for candle in candles]
        return self.compute_bollinger_bands(prices, window, num_std)

    def generate_bollinger_bands_signal_for_backtest(self, candles: List[dict]) -> str:
        prices = [float(candle['close']) for candle in candles]
        return self.generate_bollinger_bands_signal(prices)


    def identify_trend(self, product_id, candles, window=TREND_WINDOW):
        return self.technical_analysis.identify_trend(product_id, candles, window)

    def generate_combined_signal(self, rsi, macd, signal, histogram, candles):
        return self.technical_analysis.generate_combined_signal(rsi, macd, signal, histogram, candles) 

    def generate_signal(self, rsi):
        return self.technical_analysis.generate_signal(rsi)

    def place_order(self, product_id, side, size):
        return self.coinbase_service.place_order(product_id, side, size)
        
    def place_bracket_order(self, product_id, size, limit_price, stop_trigger_price):
        return self.coinbase_service.place_bracket_order(product_id, size, limit_price, stop_trigger_price)

    def monitor_price_and_place_bracket_order(self, product_id, target_price, size):
        logger.info(f"Placing bracket order with target price {target_price}.")
        for attempt in range(MAX_RETRIES):
            order = self.place_bracket_order(
                product_id, 
                size, 
                target_price * BRACKET_ORDER_TAKE_PROFIT_MULTIPLIER, 
                target_price * BRACKET_ORDER_STOP_LOSS_MULTIPLIER
            )
            if order["success"] == True:
                logger.info(f"{order}, Bracket order placed successfully.")
                return
            else:
                logger.error(f"{order}, Failed to place order.")
                return
            logger.info(f"Failed to place order. Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)
        logger.info("Max retries reached. Unable to place bracket order.")

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
        fee_rate = float(fee_tier.get('taker_fee_rate', DEFAULT_FEE_RATE))
        
        if is_buy:
            trade_amount = (balance / price) / (1 + fee_rate)
            fee = balance - (trade_amount * price)
        else:
            fee = balance * fee_rate
            trade_amount = balance - fee
        
        return trade_amount, fee

    def backtest(self, product_id: str, start_date, end_date, initial_balance: float) -> Tuple[float, List[dict]]:
        try:
            # Convert start_date and end_date to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

            logger.info(f"Starting backtest for {product_id} from {start_date} to {end_date}.")
            # Create a filename based on the product_id and date range
            filename = f"{product_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            
            if os.path.exists(filename):
                # Load historical data from file if it exists
                logger.info(f"Loading historical data from {filename}...")
                with open(filename, 'r') as f:
                    candles = json.load(f)
            else:
                # Fetch historical data if file doesn't exist
                logger.info(f"Fetching historical data from {start_date} to {end_date}...")
                candles = self.get_historical_data(product_id, start_date, end_date)
                
                # Save the fetched data to a file
                with open(filename, 'w') as f:
                    json.dump(candles, f)
                logger.info(f"Saved historical data to {filename}")
            logger.info(f"Fetched {len(candles)} candles.")

            if not candles:
                logger.warning("No historical data available for backtesting.")
                return initial_balance, []

            balance = initial_balance
            btc_balance = 0
            trades = []
            last_trade_time = None

            # Initialize tqdm progress bar for the entire loop
            with tqdm(total=len(candles), desc="Processing candles") as pbar:
                for i, candle in enumerate(candles):
                    close_price = float(candle['close'])
                    current_time = int(candle['start'])
                    
                    # Calculate indicators without caching
                    rsi = self.compute_rsi_for_backtest(candles[:i+1])
                    macd, signal, histogram = self.compute_macd_for_backtest(candles[:i+1])
                    
                    # Generate signal
                    combined_signal = self.generate_combined_signal(rsi, macd, signal, histogram, candles)

                    # Execute trade based on signal
                    if combined_signal == "BUY" and balance > 0:
                        btc_to_buy, fee = self.calculate_trade_amount_and_fee(balance, close_price, is_buy=True)
                        balance = 0
                        btc_balance += btc_to_buy
                        trades.append(self.create_trade_record(current_time, 'BUY', close_price, btc_to_buy, fee))
                        last_trade_time = current_time

                    elif combined_signal == "SELL" and btc_balance > 0:
                        balance_to_add, fee = self.calculate_trade_amount_and_fee(btc_balance * close_price, close_price, is_buy=False)
                        balance += balance_to_add
                        trades.append(self.create_trade_record(current_time, 'SELL', close_price, btc_balance, fee))
                        btc_balance = 0
                        last_trade_time = current_time

                    # Update the progress bar
                    pbar.update(1)
            # Calculate final portfolio value
            final_value = balance + (btc_balance * float(candles[-1]['close']))
            logger.info(f"Backtest completed. Final value: {final_value:.2f}")
            logger.info("Trades:")
            for trade in trades:
                usd_value = trade['amount'] * trade['price']
                logger.info(f"Date: {datetime.utcfromtimestamp(trade['date']).strftime('%Y-%m-%d %H:%M:%S')}, "
                            f"Action: {trade['action']}, Price: {trade['price']:.2f}, "
                            f"Amount: {trade['amount']:.8f}, Fee: {trade['fee']:.2f}, "
                            f"USD Value: {(usd_value - trade['fee']):.2f}")
            return final_value, trades
        except Exception as e:
            logger.error(f"An error occurred during backtesting: {e}", exc_info=True)
            return initial_balance, []

    def create_trade_record(self, time: int, action: str, price: float, amount: float, fee: float) -> dict:
        return {
            'date': time,
            'action': action,
            'price': price,
            'amount': amount,
            'fee': fee
        }

    def get_historical_data(self, product_id: str, start_date: datetime, end_date: datetime) -> List[dict]:
        all_candles = []
        current_start = start_date
        chunk_size = timedelta(hours=CHUNK_SIZE_HOURS)

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
                logger.error(f"Error fetching candle data: {e}", exc_info=True)
                # You might want to add more sophisticated error handling here

        # with open('historical_data.txt', 'w') as f:
        #     f.write("# Historical Data\n")
        #     for candle in all_candles:
        #         f.write(f"Date: {datetime.utcfromtimestamp(int(candle['start'])).strftime('%Y-%m-%d %H:%M:%S')}, "
        #                 f"Open: {candle['open']}, High: {candle['high']}, Low: {candle['low']}, "
        #                 f"Close: {candle['close']}, Volume: {candle['volume']}\n")

        # Sort the candles by their start time to ensure they are in chronological order
        all_candles.sort(key=lambda x: x['start'])
        logger.info(f"Fetched {len(all_candles)} candles for {product_id}.")
        return all_candles

def main():
    api_key = API_KEY
    api_secret = API_SECRET
    
    logger.info("Starting the trading bot.")

    trader = CryptoTrader(api_key, api_secret)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    candles = trader.get_historical_data("BTC-USD", start_date, end_date)

    fiat_usd, btc = trader.get_portfolio_info()
    logger.info(f"Portfolio: ${fiat_usd:.2f}, {btc} BTC")

    prices = trader.get_btc_prices()
    for currency, price in prices.items():
        logger.info(f"{currency}: Bid: {price['bid']:.2f}, Ask: {price['ask']:.2f}")
    rsi = trader.compute_rsi("BTC-USD", candles, period=RSI_PERIOD)
    logger.info(f"Current RSI for BTC-USD: {rsi:.2f}")

    signal = trader.generate_signal(rsi)
    logger.info(f"Signal for BTC-USD: {signal}")
    macd, signal, histogram = trader.compute_macd("BTC-USD", candles)
    logger.info(f"Current MACD for BTC-USD: MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {histogram:.2f}")
    
    # Calculate the value of 0.00187597 BTC in EUR including fees
    btc_amount = 0.00187597
    btc_eur_price = trader.get_btc_prices().get('BTC-EUR', {}).get('bid', 0)
    
    # Get the actual fee percentage based on transaction summary
    transaction_summary = trader.client.get_transaction_summary()
    fee_tier = transaction_summary.get('fee_tier', {})
    fee_percentage = fee_tier.get('taker_fee_rate', DEFAULT_FEE_RATE)
    
    eur_value_before_fees = btc_amount * btc_eur_price
    fees = float(eur_value_before_fees) * float(fee_percentage)
    eur_value_after_fees = eur_value_before_fees - fees
    
    logger.info(f"0.00187597 BTC is worth approximately {eur_value_after_fees:.2f} EUR (including {float(fee_percentage)*100}% fees)")
    logger.info(f"Fees: {fees:.2f} EUR")
    # logger.info(f"Value before fees: {eur_value_before_fees:.2f} EUR")
    
    combined_signal = trader.generate_combined_signal(rsi, macd, signal, histogram, candles)
    logger.info(f"Combined signal for BTC-USD: {combined_signal}")

    # Uncomment to place an order
    # if signal == "BUY":
    #     trader.place_order("BTC-USD", "BUY", 0.001)
    # elif signal == "SELL":
    #     trader.place_order("BTC-USD", "SELL", 0.001)

    # trader.monitor_price_and_place_bracket_order("BTC-EUR", 60000, btc_amount)
    
    trend = trader.identify_trend("BTC-USD", candles, window=TREND_WINDOW) 
    logger.info(f"Current 1h (12 days) trend for BTC-USD: {trend}")

    bitcoin_sentiment = trader.analyze_sentiment("Bitcoin")
    logger.info(f"Current sentiment for Bitcoin: {bitcoin_sentiment}")

    # ml_signal = trader.generate_ml_signal("BTC-USD")
    # logger.info(f"ML Signal for BTC-USD: {ml_signal}")

    backtest = True
    if backtest:
        logger.info("Starting backtesting.")
        # Add backtesting
 
        # # Bear market 2021: from $69000 to $15000
        # initial_balance = 10000  # USD        
        # start_date = "2021-11-01"
        # end_date = "2022-11-01"       
        #     
        # More recent backtesting 
        start_date = "2024-01-01"
        end_date = "2024-10-01"
        initial_balance = 10000  # USD

        final_value, trades = trader.backtest("BTC-USD", start_date, end_date, initial_balance)
        
        logger.info(f"Backtesting results: Initial balance: ${initial_balance}, Final portfolio value: ${final_value:.2f}")
        logger.info(f"Total return: {(final_value - initial_balance) / initial_balance * 100:.2f}%")
        logger.info(f"Number of trades: {len(trades)}")
        logger.debug("Trades executed during backtesting:")
        
        sorted_trades = sorted(trades, key=lambda x: x['date'])
        for trade in sorted_trades:
            # f.write(f"Date: {datetime.utcfromtimestamp(int(candle['start'])).strftime('%Y-%m-%d %H:%M:%S')}, "
            human_readable_date = datetime.utcfromtimestamp(int(trade['date'])).strftime('%Y-%m-%d %H:%M:%S')
            logger.debug(f"Date: {human_readable_date}, Action: {trade['action']}, Price: {trade['price']}, Amount: {trade['amount']}")

    # # You can further analyze the trades list for more insights
    # # Test ML signal for the past 200 days
    # logger.info("\nTesting ML signal for the past 200 days:")
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=200)
    
    # for i in range(200):
    #     current_date = end_date - timedelta(days=i)
    #     timestamp = int(current_date.timestamp())
    #     ml_signal = trader.generate_ml_signal("BTC-USD", end=timestamp)
    #     human_readable_date = current_date.strftime('%Y-%m-%d')
    #     logger.info(f"Date: {human_readable_date}, ML Signal: {ml_signal}")

if __name__ == "__main__":
    main()
    