from datetime import datetime, timezone, timedelta
import os
import json
import logging
from tqdm import tqdm
import numpy as np
import time
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

from trading.models import TradeRecord, PerformanceMetrics
from trading.granularity_settings import GRANULARITY_SETTINGS

class Backtester:
    def __init__(self, trader: Any):
        self.trader = trader
        self.logger = logging.getLogger(__name__)
        self.granularity_settings = GRANULARITY_SETTINGS
        
        # Default settings
        self.set_granularity_settings("ONE_HOUR")
        self.atr_period = 14
        self.atr_multiplier = 2
        self.drawdown_threshold = 0.1

    def create_trade_record(self, date: int, action: str, price: float, amount: float, fee: float) -> TradeRecord:
        return TradeRecord(date, action, price, amount, fee)

    def plot_trades(self, candles: List[Dict[str, Any]], trades: List[TradeRecord], balance_history: List[float], btc_balance_history: List[float]) -> None:
        # Convert timestamps to datetime
        dates = [datetime.fromtimestamp(float(candle['start']), tz=timezone.utc) for candle in candles]
        prices = [float(candle['close']) for candle in candles]

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot price
        ax1.plot(dates, prices, label='Price')

        # Plot trades
        for trade in trades:
            date = datetime.fromtimestamp(int(trade.date), tz=timezone.utc)
            if trade.action in ['BUY', 'STRONG BUY']:
                ax1.plot(date, trade.price, '^', color='g', markersize=10)
            elif trade.action in ['SELL', 'STRONG SELL', 'STOP LOSS', 'TRAILING STOP']:
                ax1.plot(date, trade.price, 'v', color='r', markersize=10)

        # Plot balances
        ax2.plot(dates, balance_history, label='USD Balance')
        ax2.plot(dates, btc_balance_history, label='BTC Balance (in USD)')

        # Format the plot
        ax1.set_title('Price and Trades')
        ax1.legend()
        ax2.set_title('Account Balance')
        ax2.legend()

        plt.xlabel('Date')
        fig.autofmt_xdate()  # Rotate and align the tick labels

        plt.tight_layout()
        plt.savefig('trades_and_balance.png')
        plt.close()

    def calculate_dynamic_stop_loss(self, candles: List[Dict], entry_price: float) -> float:
        if entry_price is None:
            return None
        atr = self.trader.technical_analysis.compute_atr(candles, self.atr_period)
        return entry_price - (atr * self.atr_multiplier)

    def calculate_dynamic_take_profit(self, candles: List[Dict], entry_price: float) -> float:
        if entry_price is None:
            return None
        atr = self.trader.technical_analysis.compute_atr(candles, self.atr_period)
        return entry_price + (atr * self.atr_multiplier * 1.5)  # 1.5x the stop-loss distance

    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float, risk_per_trade: float) -> float:
        risk_amount = balance * risk_per_trade
        risk_per_coin = entry_price - stop_loss
        return risk_amount / risk_per_coin

    def backtest(self, product_id: str, start_date: str, end_date: str, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float, granularity: str = "ONE_HOUR") -> Tuple[float, List[TradeRecord]]:
        try:
            # Apply the granularity settings at the start of backtest
            self.set_granularity_settings(granularity)
            
            self.logger.info(f"Starting backtest for {product_id} from {start_date} to {end_date} with granularity {granularity}.")

            # Convert start_date and end_date to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

            # Create a directory for candle files if it doesn't exist
            candle_dir = "candle_data"
            os.makedirs(candle_dir, exist_ok=True)

            # Create a filename based on the product_id, date range, and granularity
            filename = f"{product_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{granularity}.json"
            filepath = os.path.join(candle_dir, filename)
            
            # Check if we need the most recent data
            need_recent_data = end_date.date() == datetime.now().date()

            # Fetch historical data
            self.logger.info(f"Fetching all historical data from {start_date} to {end_date} with granularity {granularity}...")
            candles = self.trader.get_historical_data(product_id, start_date, end_date, granularity)
            
            # Save the fetched data to a file
            with open(filepath, 'w') as f:
                json.dump(candles, f)
            self.logger.info(f"Saved {len(candles)} candles to {filepath}")

            if not candles:
                self.logger.warning("No historical data available for backtesting.")
                return initial_balance, []

            balance = initial_balance
            btc_balance = 0
            trades: List[TradeRecord] = []
            balance_history: List[float] = []
            btc_balance_history: List[float] = []
            portfolio_values: List[float] = []  # Store daily portfolio values for trading strategy
            buy_and_hold_values: List[float] = []  # Store daily portfolio values for buy and hold strategy
            last_trade_time: Optional[int] = None
            last_trade_price: Optional[float] = None
            last_portfolio_date: Optional[datetime.date] = None

            # Calculate initial buy and hold position
            initial_btc_balance = initial_balance / float(candles[0]['close'])

            # Define a minimum number of candles required for analysis
            min_candles = 50  # Adjust this value based on your longest indicator period

            # Define constraints
            trades_today = 0
            last_trade_date: Optional[datetime.date] = None
            last_buy_price: Optional[float] = None

            # Initialize tqdm progress bar for the entire loop
            with tqdm(total=len(candles), desc="Processing candles") as pbar:
                highest_price_since_buy = 0  # Track the highest price since the last buy
                take_profit = None  # Initialize the variable to store the take profit price
                for i, candle in enumerate(candles):
                    close_price = float(candle['close'])
                    current_time = int(candle['start'])
                    current_date = datetime.fromtimestamp(current_time, tz=timezone.utc).date()
                    
                    # Ensure the current candle's timestamp is greater than the last trade time
                    if last_trade_time is not None and current_time <= last_trade_time:
                        continue

                    # Reset trades_today if it's a new day
                    if last_trade_date != current_date:
                        trades_today = 0
                        last_trade_date = current_date


                    if i == len(candles) - 1:
                        trend = self.trader.identify_trend(product_id, candles[:i+1])
                        volume_signal = self.trader.technical_analysis.analyze_volume(candles[:i+1])
                        market_conditions = self.trader.technical_analysis.analyze_market_conditions(candles[:i+1])
                        rsi = self.trader.compute_rsi_for_backtest(candles[:i+1])
                        macd, signal, histogram = self.trader.compute_macd_for_backtest(candles[:i+1])
                        combined_signal = self.trader.generate_combined_signal(rsi, macd, signal, histogram, candles[:i+1], market_conditions=market_conditions)

                        current_datetime = datetime.fromtimestamp(current_time, tz=timezone.utc)  # Convert to datetime
                        self.logger.debug(f"(Current date: {current_datetime.strftime('%Y-%m-%d %H:%M')})")  # Print date to the hour
                        self.logger.debug(f"Combined signal for today: {combined_signal}")
                        self.logger.debug(f"Market conditions for today: {market_conditions}")
                        self.logger.debug(f"Trend for today: {trend}")
                        self.logger.debug(f"Volume signal for today: {volume_signal}")
                        self.logger.debug(f"Current {product_id} value: {close_price:.2f} USD")

                    # Only generate signals if we have enough historical data and haven't exceeded max trades for the day
                    if i >= min_candles and trades_today < self.max_trades_per_day:
                        # Check if enough time has passed since the last trade
                        if last_trade_time is None or (current_time - last_trade_time) >= self.cooldown_period:
                            # Check if the price has changed enough since the last trade
                            if last_trade_price is None or abs(close_price - last_trade_price) / last_trade_price >= self.min_price_change:
                                # Calculate indicators and generate signal
                                rsi = self.trader.compute_rsi_for_backtest(candles[:i+1])
                                macd, signal, histogram = self.trader.compute_macd_for_backtest(candles[:i+1])
                                market_conditions = self.trader.technical_analysis.analyze_market_conditions(candles[:i+1])
                                combined_signal = self.trader.generate_combined_signal(rsi, macd, signal, histogram, candles[:i+1], market_conditions=market_conditions)

                                # Adjust trade size based on market conditions
                                if market_conditions == "Bullish":
                                    trade_size_multiplier = 1.2  # Increase trade size in bullish conditions
                                elif market_conditions == "Bull Market":
                                    trade_size_multiplier = 1.5  # Increase trade size in bull market conditions
                                elif market_conditions == "Bearish":
                                    trade_size_multiplier = 0.8  # Decrease trade size in bearish conditions
                                elif market_conditions == "Bear Market":
                                    trade_size_multiplier = 0.5  # Decrease trade size in bear market conditions
                                else:
                                    trade_size_multiplier = 1.0  # No change in neutral conditions

                                        
                                # Execute trade based on signal
                                if combined_signal in ["BUY", "STRONG BUY"] and balance > 0:
                                    # Only buy if the current price is lower than the last buy price or if it's the first buy

                                    if last_buy_price is None or close_price < last_buy_price:

                                        # Check for additional entry conditions
                                        trend = self.trader.identify_trend(product_id, candles[:i+1])
                                        volume_signal = self.trader.technical_analysis.analyze_volume(candles[:i+1])

                                        if trend == "Uptrend" and volume_signal == "High":  # Ensure trend and volume conditions are met                                            
                                            if combined_signal == "STRONG BUY":
                                                btc_to_buy, fee = self.trader.calculate_trade_amount_and_fee(balance * self.strong_buy_percentage * trade_size_multiplier, close_price, is_buy=True)
                                                balance -= (btc_to_buy * close_price + fee)
                                            else:  # Regular BUY
                                                btc_to_buy, fee = self.trader.calculate_trade_amount_and_fee(balance * self.buy_percentage * trade_size_multiplier, close_price, is_buy=True)
                                                balance -= (btc_to_buy * close_price + fee)

                                            btc_balance += btc_to_buy
                                            trades.append(self.create_trade_record(current_time, combined_signal, close_price, btc_to_buy, fee))
                                            last_trade_time = current_time
                                            last_trade_price = close_price
                                            highest_price_since_buy = close_price  # Reset highest price after buying
                                            trades_today += 1
                                            stop_loss = self.calculate_dynamic_stop_loss(candles[:i+1], close_price)
                                            take_profit = self.calculate_dynamic_take_profit(candles[:i+1], close_price)
                                            position_size = self.calculate_position_size(balance, close_price, stop_loss, risk_per_trade)
                                            btc_to_buy = min(position_size, balance / close_price)  # Ensure we don't exceed available balance

                                elif combined_signal in ["SELL", "STRONG SELL"] and btc_balance > 0:
                                    # Implement a trailing stop
                                    if last_buy_price and (close_price < last_buy_price * (1 - self.drawdown_threshold) or combined_signal == "STRONG SELL"):
                                        amount_to_sell = btc_balance
                                        balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(amount_to_sell * close_price, close_price, is_buy=False)
                                        balance += balance_to_add
                                        btc_balance = 0
                                        trades.append(self.create_trade_record(current_time, "STOP LOSS" if close_price < last_buy_price else combined_signal, close_price, amount_to_sell, fee))
                                        last_trade_time = current_time
                                        last_trade_price = close_price
                                        highest_price_since_buy = 0  # Reset after selling
                                        trades_today += 1
                                    if close_price <= stop_loss or close_price >= take_profit:
                                        # Execute the sell
                                        amount_to_sell = btc_balance
                                        balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(amount_to_sell * close_price, close_price, is_buy=False)
                                        balance += balance_to_add
                                        btc_balance = 0
                                        trades.append(self.create_trade_record(current_time, "TAKE PROFIT" if close_price >= take_profit else "STOP LOSS", close_price, amount_to_sell, fee))
                                        last_trade_time = current_time
                                        last_trade_price = close_price
                                        highest_price_since_buy = 0  # Reset after selling
                                        trades_today += 1

                    # Update highest price for trailing stop
                    if btc_balance > 0:
                        highest_price_since_buy = max(highest_price_since_buy, close_price)  # Update highest price
                        if close_price <= highest_price_since_buy * (1 - trailing_stop_percent):  # Check for trailing stop
                            # Trigger trailing stop
                            amount_to_sell = btc_balance
                            balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(amount_to_sell * close_price, close_price, is_buy=False)
                            balance += balance_to_add
                            btc_balance = 0
                            trades.append(self.create_trade_record(current_time, "TRAILING STOP", close_price, amount_to_sell, fee))
                            last_trade_time = current_time
                            last_trade_price = close_price
                            highest_price_since_buy = 0  # Reset after selling
                            trades_today += 1

                    # Update balance histories and portfolio value
                    balance_history.append(balance)
                    btc_balance_history.append(btc_balance * close_price)
                    
                    # Only add to portfolio_values and buy_and_hold_values if it's a new day
                    if last_portfolio_date is None or current_date > last_portfolio_date:
                        portfolio_values.append(balance + btc_balance * close_price)
                        buy_and_hold_values.append(initial_btc_balance * close_price)
                        last_portfolio_date = current_date

                    # Update the progress bar
                    pbar.update(1)

            # Calculate final portfolio value
            final_value = balance + (btc_balance * float(candles[-1]['close']))
            # self.logger.info(f"Backtest completed. Final value: {final_value:.2f}")
            self.logger.info(f"Trades: ({len(trades)})")

            for trade in trades:
                usd_value = trade.amount * trade.price
                self.logger.info(f"Date: {datetime.fromtimestamp(trade.date, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}, "
                            f"Action: {trade.action}, Price: {trade.price:.2f}, "
                            f"Amount: {trade.amount:.8f}, Fee: {trade.fee:.2f}, "
                            f"USD Value: {(usd_value - trade.fee):.2f}")

            # Log the final take profit price if there was a buy
            if take_profit is not None:
                self.logger.info(f"Final take profit price: {take_profit:.2f} USD | Combined Signal: {combined_signal}" )
                print(f"Min price change: {abs(close_price - last_trade_price) / last_trade_price} | Min price change threshold: {self.min_price_change}")
            else:
                self.logger.info("No take profit price set (no buy trades executed)")

            # Print the current trailing stop value in Bitcoin terms
            last_btc_price = float(candles[-1]['close'])
            if highest_price_since_buy > 0:
                trailing_stop_btc_value = highest_price_since_buy * (1 - trailing_stop_percent)
                self.logger.info(f"Current trailing stop: {trailing_stop_btc_value:.2f} USD in Bitcoin value")
                self.logger.info(f"(Based on peak price of {highest_price_since_buy:.2f} USD since last buy)")
                self.logger.info(f"(Triggers when Bitcoin price drops by {trailing_stop_percent * 100:.2f}% from its peak after buying)")
            else:
                self.logger.info("No buy trades executed, so no trailing stop was set.")

            # Calculate and print Sharpe ratio and Sortino ratio
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            buy_and_hold_returns = np.diff(buy_and_hold_values) / buy_and_hold_values[:-1]
            if daily_returns.std() != 0:
                sharpe_ratio = np.sqrt(365) * daily_returns.mean() / daily_returns.std()  # Adjusted for daily returns in a 24/7/365 market
                sharpe_ratio_buy_and_hold = np.sqrt(365) * buy_and_hold_returns.mean() / buy_and_hold_returns.std()
                # Calculate Sortino ratio
                negative_returns = daily_returns[daily_returns < 0]
                negative_returns_buy_and_hold = buy_and_hold_returns[buy_and_hold_returns < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.sqrt(np.mean(negative_returns**2))
                    sortino_ratio = np.sqrt(365) * daily_returns.mean() / downside_deviation
                    downside_deviation_buy_and_hold = np.sqrt(np.mean(negative_returns_buy_and_hold**2))
                    sortino_ratio_buy_and_hold = np.sqrt(365) * buy_and_hold_returns.mean() / downside_deviation_buy_and_hold
                else:
                    sortino_ratio = float('inf')  # If there are no negative returns, set Sortino ratio to infinity
                    sortino_ratio_buy_and_hold = float('inf')  # If there are no negative returns, set Sortino ratio to infinity
            else:
                sharpe_ratio = 0
                sortino_ratio = 0  # Set to 0 if standard deviation is 0 to avoid division by zero


            # # Calculate and print total return
            # total_return = (final_value - initial_balance) / initial_balance * 100
            # self.logger.info(f"Total Return: {total_return:.2f}%")

            # Calculate and print maximum drawdown
            cumulative_max = np.maximum.accumulate(portfolio_values)
            drawdown = (cumulative_max - portfolio_values) / cumulative_max
            max_drawdown = drawdown.max() * 100
            
            metrics = PerformanceMetrics.calculate_metrics(portfolio_values, buy_and_hold_values)
            self.logger.info(
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f} | "
                f"Sortino Ratio: {metrics['sortino_ratio']:.4f} | "
                f"Max Drawdown: {metrics['max_drawdown']:.2f}% | "
                f"Sharpe Ratio Buy and Hold: {metrics['sharpe_ratio_buy_and_hold']:.4f} | "
                f"Sortino Ratio Buy and Hold: {metrics['sortino_ratio_buy_and_hold']:.4f}"
            )


            return final_value, trades
        except Exception as e:
            self.logger.error(f"An error occurred during backtesting: {e}", exc_info=True)
            return initial_balance, []

    def run_continuous_backtest(self, product_id: str, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float, update_interval: int = 3600) -> None:
        while True:
            try:
                self.logger.info(f"Starting continuous backtest for {product_id}")
                
                balance = initial_balance
                btc_balance = 0
                trades = []

                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=365)  # One year of data

                # Fetch historical data
                candles = self.trader.get_historical_data(product_id, start_date, end_date, "ONE_HOUR")

                if not candles:
                    self.logger.warning("No historical data available. Retrying in 60 seconds...")
                    time.sleep(60)
                    continue

                # Run backtest on the last year of data
                final_balance, new_trades = self.backtest(product_id, start_date, end_date, balance, risk_per_trade, trailing_stop_percent)

                # Check for new trades
                if new_trades:
                    latest_trade = new_trades[-1]
                    if not trades or latest_trade.date > trades[-1].date:
                        self.logger.info(f"New trade detected: {latest_trade}")
                        trades.append(latest_trade)

                        # Update balance and btc_balance based on the new trade
                        if latest_trade.action in ["BUY", "STRONG BUY"]:
                            btc_balance += latest_trade.amount
                            balance -= (latest_trade.amount * latest_trade.price + latest_trade.fee)
                        elif latest_trade.action in ["SELL", "STRONG SELL", "STOP LOSS", "TRAILING STOP"]:
                            balance += (latest_trade.amount * latest_trade.price - latest_trade.fee)
                            btc_balance -= latest_trade.amount


                # Log current state
                current_price = float(candles[-1]['close'])
                portfolio_value = balance + (btc_balance * current_price)
                self.logger.info(f"Current balance: {balance:.2f} USD")
                self.logger.info(f"Current BTC balance: {btc_balance:.8f} BTC")
                self.logger.info(f"Current portfolio value: {portfolio_value:.2f} USD")
                self.logger.info(f"Current BTC price: {current_price:.2f} USD")

                # Sleep for the specified update interval
                self.logger.info(f"Sleeping for {update_interval} seconds before next update...")
                time.sleep(update_interval)

            except Exception as e:
                self.logger.error(f"An error occurred during continuous backtesting: {e}", exc_info=True)
                # Wait and continue running
                time.sleep(60)

    def set_granularity_settings(self, granularity: str) -> None:
        """Apply settings for the specified granularity."""
        settings = self.granularity_settings.get(granularity, self.granularity_settings["ONE_HOUR"])
        for key, value in settings.items():
            setattr(self, key, value)


