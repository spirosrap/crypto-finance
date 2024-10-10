import time
from datetime import datetime, timedelta, timezone
from coinbase.rest import RESTClient
from coinbase.rest import portfolios, products, market_data, orders
import numpy as np
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from typing import List, Tuple, Dict, Any, Optional
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
from dataclasses import dataclass

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

@dataclass
class TradeRecord:
    date: int
    action: str
    price: float
    amount: float
    fee: float

class CryptoTrader:
    def __init__(self, api_key: str, api_secret: str, product_id: str = 'BTC-USDC', granularity: str = 'ONE_HOUR'):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.coinbase_service = CoinbaseService(api_key, api_secret)
        self.technical_analysis = TechnicalAnalysis(self.coinbase_service, candle_interval=granularity, product_id=product_id)
        self.sentiment_analysis = SentimentAnalysis()
        self.historical_data = HistoricalData(self.client)
        self.backtester = Backtester(self)
        logger.info(f"CryptoTrader initialized with API key for {product_id} with {granularity} granularity.")

    def get_portfolio_info(self) -> Tuple[float, float]:
        logger.debug("Fetching portfolio info.")
        return self.coinbase_service.get_portfolio_info()

    def get_btc_prices(self) -> Dict[str, Dict[str, float]]:
        logger.debug("Fetching BTC prices.")
        return self.coinbase_service.get_btc_prices()

    def get_historical_data(self, product_id: str, start_date: datetime, end_date: datetime, granularity: str = "ONE_HOUR") -> List[Dict[str, str]]:
        return self.historical_data.get_historical_data(product_id, start_date, end_date, granularity)

    def compute_rsi(self, product_id: str, candles: List[Dict[str, str]], period: int = RSI_PERIOD) -> float:
        return self.technical_analysis.compute_rsi(product_id, candles, period)

    def compute_macd(self, product_id: str, candles: List[Dict[str, str]]) -> Tuple[float, float, float]:
        return self.technical_analysis.compute_macd(product_id, candles)

    def exponential_moving_average(self, data: List[float], span: int) -> List[float]:
        return self.technical_analysis.exponential_moving_average(data, span)

    def compute_bollinger_bands(self, candles: List[dict], window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_NUM_STD) -> Tuple[float, float, float]:
        return self.technical_analysis.compute_bollinger_bands(candles, window, num_std)

    def generate_bollinger_bands_signal(self, candles: List[dict]) -> str:
        return self.technical_analysis.generate_bollinger_bands_signal(candles)

    def compute_rsi_from_prices(self, prices: List[float], period: int = RSI_PERIOD) -> float:
        return self.technical_analysis.compute_rsi_from_prices(prices, period)

    def compute_macd_from_prices(self, prices: List[float]) -> Tuple[float, float, float]:
        return self.technical_analysis.compute_macd_from_prices(prices)

    def analyze_sentiment(self, keyword: str) -> Dict[str, float]:
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


    def identify_trend(self, product_id: str, candles: List[dict], window: int = TREND_WINDOW) -> str:
        return self.technical_analysis.identify_trend(product_id, candles, window)

    def generate_combined_signal(self, rsi: float, macd: float, signal: float, histogram: float, candles: List[dict], market_conditions: Optional[Dict[str, str]] = None) -> str:
        return self.technical_analysis.generate_combined_signal(rsi, macd, signal, histogram, candles, market_conditions) 

    def generate_signal(self, rsi: float, volatility: float) -> str:
        return self.technical_analysis.generate_signal(rsi, volatility)

    def place_order(self, product_id: str, side: str, size: float) -> Dict[str, Any]:
        return self.coinbase_service.place_order(product_id, side, size)
        
    def place_bracket_order(self, product_id: str, size: float, limit_price: float, stop_trigger_price: float) -> Dict[str, Any]:
        return self.coinbase_service.place_bracket_order(product_id, size, limit_price, stop_trigger_price)

    def monitor_price_and_place_bracket_order(self, product_id: str, target_price: float, size: float) -> None:
        return self.coinbase_service.monitor_price_and_place_bracket_order(product_id, target_price, size)

    def calculate_trade_amount_and_fee(self, balance: float, price: float, is_buy: bool) -> Tuple[float, float]:
        return self.coinbase_service.calculate_trade_amount_and_fee(balance, price, is_buy)

    def run_backtest(self, product_id: str, start_date: str, end_date: str, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float, granularity: str = "ONE_HOUR") -> Tuple[float, List[TradeRecord]]:
        return self.backtester.backtest(product_id, start_date, end_date, initial_balance, risk_per_trade, trailing_stop_percent, granularity)

    def create_trade_record(self, time: int, action: str, price: float, amount: float, fee: float) -> TradeRecord:
        return TradeRecord(date=time, action=action, price=price, amount=amount, fee=fee)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Crypto Trading Bot")
    parser.add_argument("--start_date", help="Start date and time for backtesting (YYYY-MM-DD [HH:MM])")
    parser.add_argument("--end_date", help="End date and time for backtesting (YYYY-MM-DD [HH:MM])")
    parser.add_argument("--bearmarket", action="store_true", help="Use bear market period (2021-11-01 to 2022-11-01)")
    parser.add_argument("--bullmarket", action="store_true", help="Use bull market period (2020-10-01 to 2021-04-01)")
    parser.add_argument("--ytd", action="store_true", help="Use year-to-date period (2024-01-01 to current date)")
    parser.add_argument("--skip_backtest", action="store_true", help="Skip backtesting")
    parser.add_argument("--live", action="store_true", help="Run live trading simulation")
    parser.add_argument("--product_id", default="BTC-USDC", help="Product ID for trading (default: BTC-USDC)")
    parser.add_argument("--granularity", default="ONE_HOUR", choices=[
        "ONE_MINUTE", "FIVE_MINUTE", "TEN_MINUTE", "FIFTEEN_MINUTE", 
        "THIRTY_MINUTE", "ONE_HOUR", "SIX_HOUR", "ONE_DAY"
    ], help="Granularity for candle data (default: ONE_HOUR)")
    parser.add_argument("--continuous", action="store_true", help="Run continuous backtesting simulation")
    parser.add_argument("--update_interval", type=int, default=3600, help="Update interval for continuous backtesting (in seconds, default: 3600)")
    parser.add_argument("--month", action="store_true", help="Use last month period")
    parser.add_argument("--week", action="store_true", help="Use last week period")
    parser.add_argument("--start_hour", type=int, choices=range(24), help="Start hour of the day (0-23)")
    parser.add_argument("--start_minute", type=int, choices=range(60), help="Start minute of the hour (0-59)")
    parser.add_argument("--end_hour", type=int, choices=range(24), help="End hour of the day (0-23)")
    parser.add_argument("--end_minute", type=int, choices=range(60), help="End minute of the hour (0-59)")
    parser.add_argument("--oldbear", action="store_true", help="Use old bear market period (2018-01-01 to 2020-06-01)")
    parser.add_argument("--oldbull", action="store_true", help="Use old bull market period (2018-01-01 to 2019-12-01)")
    return parser.parse_args()

def parse_datetime(date_string):
    formats = [
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    raise ValueError(f"Invalid date format: {date_string}. Use YYYY-MM-DD [HH:MM]")

def display_portfolio_info(trader, product_id):
    fiat_usd, btc = trader.get_portfolio_info()
    logger.info(f"Portfolio: ${fiat_usd:.2f}, {btc} BTC")

    prices = trader.get_btc_prices()
    for currency, price in prices.items():
        logger.info(f"{currency}: Bid: {price['bid']:.2f}, Ask: {price['ask']:.2f}")

def display_technical_indicators(trader, product_id, candles):
    if len(candles) < 20:  # Adjust this number based on your minimum required data points
        print(f"Not enough data for analysis. Only {len(candles)} candles available.")
        return

    rsi = trader.compute_rsi(product_id, candles, period=RSI_PERIOD)
    prices = [float(candle['close']) for candle in candles[-20:]]
    returns = np.diff(np.log(prices))
    volatility = np.std(returns) * np.sqrt(252)

    signal = trader.generate_signal(rsi, volatility)
    macd, signal, histogram = trader.compute_macd(product_id, candles)
    logger.debug(f"Current MACD for {product_id}: MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {histogram:.2f}, RSI: {rsi:.2f}")

    combined_signal = trader.generate_combined_signal(rsi, macd, signal, histogram, candles)
    logger.debug(f"Combined signal for {product_id}: {combined_signal}")

    current_market_conditions = trader.technical_analysis.analyze_market_conditions(candles)
    logger.info(f"Current market conditions: {current_market_conditions}")

    trend = trader.identify_trend(product_id, candles, window=TREND_WINDOW) 
    logger.info(f"Current 1h (12 days) trend for {product_id}: {trend}")

def display_sentiment_analysis(trader, product_id):
    bitcoin_sentiment = trader.analyze_sentiment("Bitcoin")
    logger.debug(f"Current sentiment for {product_id}: {bitcoin_sentiment}")

def calculate_btc_eur_value(trader):
    btc_amount = 0.00187597
    btc_eur_price = trader.get_btc_prices().get('BTC-EUR', {}).get('bid', 0)
    
    transaction_summary = trader.client.get_transaction_summary()
    fee_tier = transaction_summary.get('fee_tier', {})
    fee_percentage = fee_tier.get('taker_fee_rate', DEFAULT_FEE_RATE)
    
    eur_value_before_fees = btc_amount * btc_eur_price
    fees = float(eur_value_before_fees) * float(fee_percentage)
    eur_value_after_fees = eur_value_before_fees - fees
    
    logger.info(f"0.00187597 BTC is worth approximately {eur_value_after_fees:.2f} EUR (including {float(fee_percentage)*100}% fees), Fees: {fees:.2f} EUR")

def run_backtest(trader, args, initial_balance, risk_per_trade, trailing_stop_percent, granularity):
    end_date = datetime.now()
    
    if args.bearmarket:
        start_date = datetime(2021, 11, 1)
        end_date = datetime(2022, 11, 1, 23, 59, 59)
    elif args.bullmarket:
        start_date = datetime(2020, 10, 1)
        end_date = datetime(2021, 4, 1, 23, 59, 59)
    elif args.ytd:
        start_date = datetime(2024, 1, 1)
    elif args.month:
        start_date = end_date - timedelta(days=30)
    elif args.week:
        start_date = end_date - timedelta(days=7)
    elif args.oldbear:
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2020, 6, 1, 23, 59, 59)
    elif args.oldbull:
        start_date = datetime(2016, 12, 1)
        end_date = datetime(2017, 12, 1, 23, 59, 59)
    elif args.start_date:
        start_date = parse_datetime(args.start_date)
        if args.end_date:
            end_date = parse_datetime(args.end_date)
    else:
        start_date = end_date - timedelta(days=365)

    # If only date was provided, set default time
    if start_date.hour == 0 and start_date.minute == 0:
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if end_date.hour == 0 and end_date.minute == 0:
        end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Running backtest from {start_date_str} to {end_date_str}")

    final_value, trades = trader.run_backtest(args.product_id, start_date_str, end_date_str, initial_balance, risk_per_trade, trailing_stop_percent, granularity)
    
    logger.info(f"Initial: ${initial_balance}, Final value: ${final_value:.2f}")
    logger.info(f"Number of trades: {len(trades)}")
    logger.debug("Trades executed during backtesting:")
    logger.info(f"Total return: {(final_value - initial_balance) / initial_balance * 100:.2f}%")

    sorted_trades = sorted(trades, key=lambda x: x.date)
    for trade in sorted_trades:
        human_readable_date = datetime.fromtimestamp(int(trade.date), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        logger.debug(f"Date: {human_readable_date}, Action: {trade.action}, Price: {trade.price}, Amount: {trade.amount}")

def main():
    args = parse_arguments()
    trader = CryptoTrader(API_KEY, API_SECRET, product_id=args.product_id, granularity=args.granularity)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    candles = trader.get_historical_data(args.product_id, start_date, end_date, args.granularity)

    display_portfolio_info(trader, args.product_id)
    display_technical_indicators(trader, args.product_id, candles)
    calculate_btc_eur_value(trader)
    display_sentiment_analysis(trader, args.product_id)

    initial_balance = 10000  # USD
    risk_per_trade = 0.02  # 2% risk per trade
    trailing_stop_percent = 0.08  # 8% trailing stop

    if args.live:
        logger.info("Starting live trading simulation.")
        trader.backtester.run_live(args.product_id, initial_balance, risk_per_trade, trailing_stop_percent, granularity=args.granularity)
    elif args.continuous:
        logger.info("Starting continuous backtesting simulation.")
        trader.backtester.run_continuous_backtest(args.product_id, initial_balance, risk_per_trade, trailing_stop_percent, update_interval=args.update_interval)
    elif not args.skip_backtest:
        logger.info("Starting backtesting.")
        run_backtest(trader, args, initial_balance, risk_per_trade, trailing_stop_percent, granularity=args.granularity)

if __name__ == "__main__":
    main()