import logging
from datetime import datetime, timedelta
from typing import List, Dict, Union
from coinbase.rest import RESTClient
from historicaldata import HistoricalData
from technicalanalysis import TechnicalAnalysis
from sentimentanalysis import SentimentAnalysis
from coinbaseservice import CoinbaseService
from config import API_KEY, API_SECRET, NEWS_API_KEY
import numpy as np
from scipy import stats
from ml_model import MLSignal

class HighFrequencyStrategy:
    def __init__(self, api_key: str, api_secret: str, product_id: str, granularity: str = "ONE_MINUTE"):
        self.client = RESTClient(api_key=api_key, api_secret=api_secret)
        self.product_id = product_id
        self.coinbase_service = CoinbaseService(api_key, api_secret)
        self.historical_data = HistoricalData(self.client)
        self.granularity = granularity
        self.technical_analysis = TechnicalAnalysis(self.coinbase_service, candle_interval=granularity, product_id=self.product_id)  # Pass coinbase_service here
        self.sentiment_analysis = SentimentAnalysis()
        self.logger = logging.getLogger(__name__)
        self.ml_signal = MLSignal(self.logger, self.historical_data, product_id=self.product_id, granularity=granularity)
    def extract_prices(self, candles: List[Dict], key: str = 'close') -> np.ndarray:
        """
        Extract prices from candle data.

        :param candles: List of historical candle data.
        :param key: Key to extract from each candle (default is 'close').
        :return: NumPy array of prices.
        """
        return np.array([float(candle[key]) for candle in candles])        

    def get_signals(self, candles: List[Dict]) -> Dict[str, float]:
        close_prices = self.extract_prices(candles, key='close')
        volumes = self.extract_prices(candles, key='volume')

        # Calculate technical indicators
        rsi = self.technical_analysis.calculate_rsi(tuple(close_prices), period=14)
        macd, signal, _ = self.technical_analysis.compute_macd(self.product_id, candles)
        bollinger_upper, bollinger_lower, bollinger_mid = self.technical_analysis.compute_bollinger_bands(candles)
        obv = self.technical_analysis.compute_on_balance_volume(tuple(close_prices), tuple(volumes))

        # Get the most recent values
        current_price = close_prices[-1]
        current_rsi = rsi if isinstance(rsi, (int, float)) else rsi[-1]
        current_macd = macd[-1] if isinstance(macd, np.ndarray) else macd
        current_signal = signal[-1] if isinstance(signal, np.ndarray) else signal
        current_bollinger_upper = bollinger_upper[-1] if isinstance(bollinger_upper, np.ndarray) else bollinger_upper
        current_bollinger_lower = bollinger_lower[-1] if isinstance(bollinger_lower, np.ndarray) else bollinger_lower
        current_bollinger_mid = bollinger_mid[-1] if isinstance(bollinger_mid, np.ndarray) else bollinger_mid
        current_obv = obv[-1] if isinstance(obv, np.ndarray) else obv

        # Calculate improved signals
        rsi_signal = self.get_rsi_signal(current_rsi)
        macd_signal = self.get_macd_signal(macd, signal)
        bollinger_signal = self.get_bollinger_signal(current_price, current_bollinger_upper, current_bollinger_lower, current_bollinger_mid)
        obv_signal = self.get_obv_signal(obv)
        
        # Add new signals
        price_momentum = self.get_price_momentum(close_prices)
        volume_trend = self.get_volume_trend(volumes)
        # Get ML signal
        ml_signal = self.get_ml_signal(candles)

        weights = {
            'rsi': 0.15,
            'macd': 0.15,
            'bollinger': 0.15,
            'obv': 0.1,
            'price_momentum': 0.15,
            'volume_trend': 0.1,
            'ml_signal': 0.5
        }

        combined_signal = (
            rsi_signal * weights['rsi'] +
            macd_signal * weights['macd'] +
            bollinger_signal * weights['bollinger'] +
            obv_signal * weights['obv'] +
            price_momentum * weights['price_momentum'] +
            volume_trend * weights['volume_trend'] +
            ml_signal * weights['ml_signal']  # Include ML signal in combined signal
        )

        return {
            "rsi": rsi_signal,
            "macd": macd_signal,
            "bollinger": bollinger_signal,
            "obv": obv_signal,
            "price_momentum": price_momentum,
            "volume_trend": volume_trend,
            "ml_signal": ml_signal,  # Add ML signal to the returned dictionary
            "combined": combined_signal
        }

    def get_rsi_signal(self, rsi: float) -> float:
        if rsi < 30:
            return 1 - (rsi / 30)  # Stronger buy signal as RSI approaches 0
        elif rsi > 70:
            return -((rsi - 70) / 30)  # Stronger sell signal as RSI approaches 100
        else:
            return 0

    def get_macd_signal(self, macd: Union[np.ndarray, float], signal: Union[np.ndarray, float]) -> float:
        if isinstance(macd, np.ndarray) and isinstance(signal, np.ndarray):
            diff = macd[-1] - signal[-1]
        else:
            diff = macd - signal
        return np.tanh(diff)  # Use hyperbolic tangent to normalize the signal

    def get_bollinger_signal(self, price: float, upper: float, lower: float, mid: float) -> float:
        if price < lower:
            return (lower - price) / (mid - lower)  # Stronger buy signal as price drops below lower band
        elif price > upper:
            return -((price - upper) / (upper - mid))  # Stronger sell signal as price rises above upper band
        else:
            return 0

    def get_obv_signal(self, obv: Union[np.ndarray, float]) -> float:
        if isinstance(obv, np.ndarray):
            obv_sma = np.mean(obv[-20:])  # 20-period SMA of OBV
            return np.tanh((obv[-1] - obv_sma) / obv_sma)  # Normalize using hyperbolic tangent
        else:
            return 0  # Return 0 if OBV is a scalar, as we can't calculate a trend

    def get_price_momentum(self, prices: np.ndarray) -> float:
        returns = np.diff(prices) / prices[:-1]
        z_score = stats.zscore(returns)[-1]
        return np.tanh(z_score)  # Normalize using hyperbolic tangent

    def get_volume_trend(self, volumes: np.ndarray) -> float:
        short_ma = np.mean(volumes[-10:])
        long_ma = np.mean(volumes[-30:])
        return np.tanh((short_ma - long_ma) / long_ma)  # Normalize using hyperbolic tangent

    def get_ml_signal(self, candles: List[Dict]) -> float:
        """
        Get the ML model's prediction signal.
        
        :param candles: List of historical candle data.
        :return: ML model's prediction signal (-1 to 1).
        """
        # Extract features from candles (you may need to adjust this based on your ML model's input requirements)
        features = self.extract_features(candles)
        
        # Get prediction from ML model
        prediction = self.ml_signal.predict_signal(candles)
        
        # Normalize prediction to be between -1 and 1
        return np.tanh(prediction)

    def extract_features(self, candles: List[Dict]) -> np.ndarray:
        """
        Extract features from candles for ML model input.
        Adjust this method based on your ML model's requirements.
        """
        # This is a placeholder implementation. Adjust according to your ML model's needs.
        close_prices = self.extract_prices(candles, key='close')
        returns = np.diff(close_prices) / close_prices[:-1]
        features = np.array([
            np.mean(returns),
            np.std(returns),
            np.percentile(returns, 25),
            np.percentile(returns, 75)
        ])
        return features.reshape(1, -1)  # Reshape for single sample prediction

    def run_strategy(self, lookback_hours: int = 24):
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)

        candles = self.historical_data.get_historical_data(
            self.product_id,
            start_date,
            end_date,
            granularity= self.granularity
        )

        if not candles:
            self.logger.error("No historical data available.")
            return

        signals = self.get_signals(candles)

        # Get sentiment data (you may need to adjust this based on your SentimentAnalysis implementation)
        sentiment_score = self.sentiment_analysis.get_sentiment(self.product_id)

        # Combine technical signals with sentiment
        #final_signal = (signals['combined'] + sentiment_score) / 2
        final_signal = signals['combined']
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
        fee_rate = 0.0075  # 0.75% fee

        while current_date <= end_date:
            lookback_start = current_date - timedelta(hours=24)
            candles = self.historical_data.get_historical_data(
                self.product_id,
                lookback_start,
                current_date,
                granularity= self.granularity
            )

            if not candles:
                self.logger.warning(f"No data available for {current_date}")
                current_date += timedelta(hours=1)
                continue

            signals = self.get_signals(candles)
            final_signal = signals['combined']

            current_price = float(candles[-1]['close'])

            self.logger.info(f"Date: {current_date}, Price: {current_price}, Signal: {final_signal}")
            total_usd = balance + (btc_balance * current_price)
            self.logger.info(f"Current balance: ${balance:.2f}, BTC balance: {btc_balance:.8f}, Total in USD: ${total_usd:.2f}")

            if final_signal > 0.4 and balance > 0:  # Buy signal
                btc_to_buy = (balance * 0.1) / current_price  # Buy with 10% of available balance
                fee = btc_to_buy * current_price * fee_rate
                if balance >= (btc_to_buy * current_price + fee):
                    balance -= btc_to_buy * current_price + fee
                    btc_balance += btc_to_buy
                    trades.append(('buy', current_date, current_price, btc_to_buy))
                    self.logger.info(f"Bought {btc_to_buy:.8f} BTC at {current_price} on {current_date}")
                    self.logger.info(f"Fee paid: ${fee:.2f}")
                else:
                    self.logger.info("Insufficient balance to buy including fee")

            elif final_signal < -0.2 and btc_balance > 0:  # Sell signal
                btc_to_sell = btc_balance * 0.1  # Sell 10% of BTC holdings
                gross_sale = btc_to_sell * current_price
                fee = gross_sale * fee_rate
                net_sale = gross_sale - fee
                balance += net_sale
                btc_balance -= btc_to_sell
                trades.append(('sell', current_date, current_price, btc_to_sell))
                self.logger.info(f"Sold {btc_to_sell:.8f} BTC at {current_price} on {current_date}")
                self.logger.info(f"Fee paid: ${fee:.2f}")

            current_date += timedelta(hours=1)

        final_balance = balance + btc_balance * current_price
        roi = (final_balance - initial_balance) / initial_balance * 100

        self.logger.info(f"Backtesting completed")
        self.logger.info(f"Initial balance: ${initial_balance:.2f}")
        self.logger.info(f"Final balance: ${final_balance:.2f}")
        self.logger.info(f"ROI: {roi:.2f}%")
        self.logger.info(f"Number of trades: {len(trades)}")

        return final_balance, roi, trades

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the strategy with your API key and secret
    strategy = HighFrequencyStrategy(API_KEY, API_SECRET, product_id="BTC-USDC", granularity="ONE_MINUTE")
    
    # Run backtest
    start_date = datetime(2024, 9, 13)  # Changed to a past date
    end_date = datetime(2024, 10, 13)
    final_balance, roi, trades = strategy.backtest(start_date, end_date)