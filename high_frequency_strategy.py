import logging
from datetime import datetime, timedelta
from typing import List, Dict
from coinbase.rest import RESTClient
from historicaldata import HistoricalData
from technicalanalysis import TechnicalAnalysis
from sentimentanalysis import SentimentAnalysis
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET, NEWS_API_KEY
import numpy as np

class HighFrequencyStrategy:
    def __init__(self, api_key: str, api_secret: str, product_id: str):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.product_id = product_id
        self.coinbase_service = CoinbaseService(api_key, api_secret)
        self.historical_data = HistoricalData(self.client)
        self.technical_analysis = TechnicalAnalysis(self.coinbase_service)  # Pass coinbase_service here
        self.sentiment_analysis = SentimentAnalysis()
        self.logger = logging.getLogger(__name__)

    def extract_prices(self, candles: List[Dict], key: str = 'close') -> np.ndarray:
        """
        Extract prices from candle data.

        :param candles: List of historical candle data.
        :param key: Key to extract from each candle (default is 'close').
        :return: NumPy array of prices.
        """
        return np.array([float(candle[key]) for candle in candles])        

    def get_signals(self, candles: List[Dict]) -> Dict[str, float]:
        # Calculate technical indicators
        close_prices = tuple(self.extract_prices(candles, key='close'))
        volumes = tuple(self.extract_prices(candles, key='volume'))

        rsi = self.technical_analysis.calculate_rsi(close_prices, period=14)
        macd, signal, _ = self.technical_analysis.compute_macd(self.product_id, candles)
        bollinger_upper, bollinger_lower, _ = self.technical_analysis.compute_bollinger_bands(candles)
        obv = self.technical_analysis.compute_on_balance_volume(close_prices, volumes)  # Changed method name here

        # Get the most recent values
        current_price = close_prices[-1]
        current_rsi = rsi
        current_macd = macd
        current_signal = signal
        current_bollinger_upper = bollinger_upper
        current_bollinger_lower = bollinger_lower
        current_obv = obv[-1]

        # Calculate signals
        rsi_signal = 1 if current_rsi < 30 else (-1 if current_rsi > 70 else 0)
        macd_signal = 1 if current_macd > current_signal else -1
        bollinger_signal = 1 if current_price < current_bollinger_lower else (-1 if current_price > current_bollinger_upper else 0)
        obv_signal = 1 if current_obv > obv[-2] else -1

        # Combine signals
        combined_signal = (rsi_signal + macd_signal + bollinger_signal + obv_signal) / 4

        return {
            "rsi": rsi_signal,
            "macd": macd_signal,
            "bollinger": bollinger_signal,
            "obv": obv_signal,
            "combined": combined_signal
        }

    def run_strategy(self, lookback_hours: int = 24):
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)

        candles = self.historical_data.get_historical_data(
            self.product_id,
            start_date,
            end_date,
            granularity="FIVE_MINUTE"
        )

        if not candles:
            self.logger.error("No historical data available.")
            return

        signals = self.get_signals(candles)

        # Get sentiment data (you may need to adjust this based on your SentimentAnalysis implementation)
        sentiment_score = self.sentiment_analysis.get_sentiment(self.product_id)

        # Combine technical signals with sentiment
        final_signal = (signals['combined'] + sentiment_score) / 2

        self.logger.info(f"Technical Signals: {signals}")
        self.logger.info(f"Sentiment Score: {sentiment_score}")
        self.logger.info(f"Final Signal: {final_signal}")

        # Implement your trading logic here based on the final_signal
        if final_signal > 0.5:
            self.logger.info("Strong buy signal")
        elif final_signal < -0.5:
            self.logger.info("Strong sell signal")
        else:
            self.logger.info("Hold or no clear signal")

        return final_signal

    def backtest(self, start_date: datetime, end_date: datetime, initial_balance: float = 10000):
        current_date = start_date
        balance = initial_balance
        btc_balance = 0
        trades = []

        while current_date <= end_date:
            lookback_start = current_date - timedelta(hours=24)
            candles = self.historical_data.get_historical_data(
                self.product_id,
                lookback_start,
                current_date,
                granularity="FIVE_MINUTE"
            )

            if not candles:
                self.logger.warning(f"No data available for {current_date}")
                current_date += timedelta(hours=1)
                continue

            signals = self.get_signals(candles)
            # For backtesting, we'll use a constant sentiment score or implement historical sentiment
            sentiment_score = 0  # Replace with historical sentiment if available
            final_signal = (signals['combined'] + sentiment_score) / 2

            current_price = float(candles[-1]['close'])

            self.logger.info(f"Date: {current_date}, Price: {current_price}, Signal: {final_signal}")
            self.logger.info(f"Current balance: ${balance}, BTC balance: {btc_balance}")

            # Implement a more flexible trading strategy
            if final_signal > 0.3 and balance > 0:  # Lowered threshold for buying
                btc_to_buy = (balance * 0.1) / current_price  # Buy with 10% of available balance
                balance -= btc_to_buy * current_price
                btc_balance += btc_to_buy
                trades.append(('buy', current_date, current_price, btc_to_buy))
                self.logger.info(f"Bought {btc_to_buy} BTC at {current_price} on {current_date}")
            elif final_signal < -0.3 and btc_balance > 0:  # Lowered threshold for selling
                btc_to_sell = btc_balance * 0.1  # Sell 10% of BTC holdings
                balance += btc_to_sell * current_price
                btc_balance -= btc_to_sell
                trades.append(('sell', current_date, current_price, btc_to_sell))
                self.logger.info(f"Sold {btc_to_sell} BTC at {current_price} on {current_date}")

            current_date += timedelta(hours=1)

        final_balance = balance + btc_balance * current_price
        roi = (final_balance - initial_balance) / initial_balance * 100

        self.logger.info(f"Backtesting completed")
        self.logger.info(f"Initial balance: ${initial_balance}")
        self.logger.info(f"Final balance: ${final_balance}")
        self.logger.info(f"ROI: {roi:.2f}%")
        self.logger.info(f"Number of trades: {len(trades)}")

        return final_balance, roi, trades

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the strategy with your API key and secret
    strategy = HighFrequencyStrategy(API_KEY, API_SECRET, "BTC-USDC")
    
    # Run backtest
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 7)  # Reduced to one week for faster testing
    final_balance, roi, trades = strategy.backtest(start_date, end_date)