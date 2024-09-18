import time
from datetime import datetime, timedelta, timezone
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
from historicaldata import HistoricalData
import os
import json
import logging
from tqdm import tqdm
from backtester import Backtester
import argparse

# Constants
DEFAULT_FEE_RATE = 0.005  # 0.5%
RETRY_DELAY_SECONDS = 60
MAX_RETRIES = 1
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
        self.historical_data = HistoricalData(self.client)
        self.backtester = Backtester(self)
        logger.info("CryptoTrader initialized with API key.")

    def get_portfolio_info(self):
        logger.debug("Fetching portfolio info.")
        return self.coinbase_service.get_portfolio_info()

    def get_btc_prices(self):
        logger.debug("Fetching BTC prices.")
        return self.coinbase_service.get_btc_prices()

    def get_hourly_data(self, product_id):
        return self.historical_data.get_hourly_data(product_id)

    def get_6h_data(self, product_id):
        return self.historical_data.get_6h_data(product_id)

    def get_historical_data(self, product_id, start_date, end_date):
        return self.historical_data.get_historical_data(product_id, start_date, end_date)

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

    def generate_combined_signal(self, rsi, macd, signal, histogram, candles, market_conditions=None):
        return self.technical_analysis.generate_combined_signal(rsi, macd, signal, histogram, candles, market_conditions) 

    def generate_signal(self, rsi, volatility):
        return self.technical_analysis.generate_signal(rsi, volatility)

    def place_order(self, product_id, side, size):
        return self.coinbase_service.place_order(product_id, side, size)
        
    def place_bracket_order(self, product_id, size, limit_price, stop_trigger_price):
        return self.coinbase_service.place_bracket_order(product_id, size, limit_price, stop_trigger_price)

    def monitor_price_and_place_bracket_order(self, product_id, target_price, size):
        return self.coinbase_service.monitor_price_and_place_bracket_order(product_id, target_price, size)

    def calculate_trade_amount_and_fee(self, balance: float, price: float, is_buy: bool) -> Tuple[float, float]:
        return self.coinbase_service.calculate_trade_amount_and_fee(balance, price, is_buy)

    def run_backtest(self, product_id: str, start_date, end_date, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float):
        return self.backtester.backtest(product_id, start_date, end_date, initial_balance, risk_per_trade, trailing_stop_percent)

    def create_trade_record(self, time: int, action: str, price: float, amount: float, fee: float) -> dict:
        return {
            'date': time,
            'action': action,
            'price': price,
            'amount': amount,
            'fee': fee
        }


def main():
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--start_date", help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end_date", help="End date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--bearmarket", action="store_true", help="Use bear market period (2021-11-01 to 2022-11-01)")
    parser.add_argument("--bullmarket", action="store_true", help="Use bull market period (2020-10-01 to 2021-04-01)")
    parser.add_argument("--ytd", action="store_true", help="Use year-to-date period (2024-01-01 to current date)")
    parser.add_argument("--skip_backtest", action="store_true", help="Skip backtesting")
    parser.add_argument("--live", action="store_true", help="Run live trading simulation")
    parser.add_argument("--product_id", default="BTC-USD", help="Product ID for trading (default: BTC-USD)")
    args = parser.parse_args()

    api_key = API_KEY
    api_secret = API_SECRET
    
    trader = CryptoTrader(api_key, api_secret)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    candles = trader.get_historical_data(args.product_id, start_date, end_date)

    fiat_usd, btc = trader.get_portfolio_info()
    logger.info(f"Portfolio: ${fiat_usd:.2f}, {btc} BTC")

    prices = trader.get_btc_prices()
    for currency, price in prices.items():
        logger.info(f"{currency}: Bid: {price['bid']:.2f}, Ask: {price['ask']:.2f}")
    rsi = trader.compute_rsi(args.product_id, candles, period=RSI_PERIOD)

    # Calculate volatility
    prices = [float(candle['close']) for candle in candles[-20:]]  # Use last 20 candles
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility

    signal = trader.generate_signal(rsi, volatility)
    macd, signal, histogram = trader.compute_macd(args.product_id, candles)
    logger.info(f"Current MACD for {args.product_id}: MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {histogram:.2f}, RSI: {rsi:.2f}")
    
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
    
    logger.info(f"0.00187597 BTC is worth approximately {eur_value_after_fees:.2f} EUR (including {float(fee_percentage)*100}% fees), Fees: {fees:.2f} EUR")
    # logger.info(f"Value before fees: {eur_value_before_fees:.2f} EUR")
    
    combined_signal = trader.generate_combined_signal(rsi, macd, signal, histogram, candles)
    logger.info(f"Combined signal for {args.product_id}: {combined_signal}")

    # Add this block to print market conditions for the most current signal
    current_market_conditions = trader.technical_analysis.analyze_market_conditions(candles)
    logger.info(f"Current market conditions: {current_market_conditions}")

    # Uncomment to place an order
    # if signal == "BUY":
    #     trader.place_order("BTC-USD", "BUY", 0.001)
    # elif signal == "SELL":
    #     trader.place_order("BTC-USD", "SELL", 0.001)

    # trader.monitor_price_and_place_bracket_order("BTC-EUR", 60000, btc_amount)
    
    trend = trader.identify_trend(args.product_id, candles, window=TREND_WINDOW) 
    logger.info(f"Current 1h (12 days) trend for {args.product_id}: {trend}")

    bitcoin_sentiment = trader.analyze_sentiment("Bitcoin")
    logger.info(f"Current sentiment for {args.product_id}: {bitcoin_sentiment}")

    # ml_signal = trader.generate_ml_signal("BTC-USD")
    # logger.info(f"ML Signal for BTC-USD: {ml_signal}")

    initial_balance = 10000  # USD
    risk_per_trade = 0.02  # 2% risk per trade
    trailing_stop_percent = 0.08  # 8% trailing stop

    if args.live:
        logger.info("Starting live trading simulation.")
        trader.backtester.run_live(args.product_id, initial_balance, risk_per_trade, trailing_stop_percent)
    elif not args.skip_backtest:
        logger.info("Starting backtesting.")

        # Use command-line arguments if provided, otherwise use default values
        if args.bearmarket:
            start_date = "2021-11-01 00:00:00"
            end_date = "2022-11-01 23:59:59"
        elif args.bullmarket:
            start_date = "2020-10-01 00:00:00"
            end_date = "2021-04-01 23:59:59"
        elif args.ytd:
            start_date = "2024-01-01 00:00:00"
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif args.start_date:
            start_date = f"{args.start_date} 00:00:00"
            end_date = f"{args.end_date} 23:59:59" if args.end_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d 00:00:00")
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        final_value, trades = trader.run_backtest(args.product_id, start_date, end_date, initial_balance, risk_per_trade, trailing_stop_percent)
        
        logger.info(f"Backtesting results: Initial balance: ${initial_balance}, Final portfolio value: ${final_value:.2f}")
        logger.info(f"Total return: {(final_value - initial_balance) / initial_balance * 100:.2f}%")
        logger.info(f"Number of trades: {len(trades)}")
        logger.debug("Trades executed during backtesting:")
        
        sorted_trades = sorted(trades, key=lambda x: x['date'])
        for trade in sorted_trades:
            human_readable_date = datetime.fromtimestamp(int(trade['date']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
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
