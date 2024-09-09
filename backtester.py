from datetime import datetime
import os
import json
import logging
from tqdm import tqdm
import numpy as np
from datetime import timedelta
import time
import requests

class Backtester:
    def __init__(self, trader):
        self.trader = trader
        self.logger = logging.getLogger(__name__)
        self.cooldown_period = 24 * 60 * 60 * 1  # 3 days in seconds
        self.max_trades_per_day = 1
        self.min_price_change = 0.08  # 8% minimum price change
        self.drawdown_threshold = 0.1  # 10% drawdown threshold
        self.strong_buy_percentage = 0.8  # 80% of balance for strong buy
        self.buy_percentage = 0.25  # 25% of balance for regular buy

    def backtest(self, product_id: str, start_date, end_date, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float):
        try:
            # Convert start_date and end_date to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

            self.logger.info(f"Starting backtest for {product_id} from {start_date} to {end_date}.")

            # Create a directory for candle files if it doesn't exist
            candle_dir = "candle_data"
            os.makedirs(candle_dir, exist_ok=True)

            # Create a filename based on the product_id and date range
            filename = f"{product_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            filepath = os.path.join(candle_dir, filename)
            
            # Check if we need the most recent data
            need_recent_data = end_date.date() == datetime.now().date()

            # if os.path.exists(filepath):
            #     # Load existing historical data from file
            #     self.logger.info(f"Loading historical data from {filepath}...")
            #     with open(filepath, 'r') as f:
            #         candles = json.load(f)
            #     self.logger.info(f"Loaded {len(candles)} candles from file.")
            # else:
                # If file doesn't exist, fetch all historical data
            self.logger.info(f"Fetching all historical data from {start_date} to {end_date}...")
            candles = self.trader.get_historical_data(product_id, start_date, end_date)
            
            # Save the fetched data to a file
            with open(filepath, 'w') as f:
                json.dump(candles, f)
            self.logger.info(f"Saved {len(candles)} candles to {filepath}")

            if not candles:
                self.logger.warning("No historical data available for backtesting.")
                return initial_balance, []

            balance = initial_balance
            btc_balance = 0
            trades = []
            last_trade_time = None
            last_trade_price = None
            
            # Define a minimum number of candles required for analysis
            min_candles = 50  # Adjust this value based on your longest indicator period

            # Define constraints
            trades_today = 0
            last_trade_date = None
            last_buy_price = None

            # Initialize tqdm progress bar for the entire loop
            with tqdm(total=len(candles), desc="Processing candles") as pbar:
                highest_price_since_buy = 0  # Track the highest price since the last buy
                for i, candle in enumerate(candles):
                    close_price = float(candle['close'])
                    current_time = int(candle['start'])
                    current_date = datetime.utcfromtimestamp(current_time).date()
                    
                    # Reset trades_today if it's a new day
                    if last_trade_date != current_date:
                        trades_today = 0
                        last_trade_date = current_date

                    if current_date > datetime(2024, 9, 8).date():
                        if 'combined_signal' in locals() and combined_signal is not None:
                            self.logger.info(f"Combined signal for today: {combined_signal}")
                            combined_signal = None  # Reset combined_signal after logging

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
                                            trades.append(self.trader.create_trade_record(current_time, combined_signal, close_price, btc_to_buy, fee))
                                            last_trade_time = current_time
                                            last_trade_price = close_price
                                            highest_price_since_buy = close_price  # Reset highest price after buying
                                            trades_today += 1

                                elif combined_signal in ["SELL", "STRONG SELL"] and btc_balance > 0:
                                    # Implement a trailing stop
                                    if last_buy_price and (close_price < last_buy_price * (1 - self.drawdown_threshold) or combined_signal == "STRONG SELL"):
                                        amount_to_sell = btc_balance
                                        balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(amount_to_sell * close_price, close_price, is_buy=False)
                                        balance += balance_to_add
                                        btc_balance = 0
                                        trades.append(self.trader.create_trade_record(current_time, "STOP LOSS" if close_price < last_buy_price else combined_signal, close_price, amount_to_sell, fee))
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
                            trades.append(self.trader.create_trade_record(current_time, "TRAILING STOP", close_price, amount_to_sell, fee))
                            last_trade_time = current_time
                            last_trade_price = close_price
                            highest_price_since_buy = 0  # Reset after selling
                            trades_today += 1

                    # Update the progress bar
                    pbar.update(1)

            # Calculate final portfolio value
            final_value = balance + (btc_balance * float(candles[-1]['close']))
            self.logger.info(f"Backtest completed. Final value: {final_value:.2f}")
            self.logger.info("Trades:")
            for trade in trades:
                usd_value = trade['amount'] * trade['price']
                self.logger.info(f"Date: {datetime.utcfromtimestamp(trade['date']).strftime('%Y-%m-%d %H:%M:%S')}, "
                            f"Action: {trade['action']}, Price: {trade['price']:.2f}, "
                            f"Amount: {trade['amount']:.8f}, Fee: {trade['fee']:.2f}, "
                            f"USD Value: {(usd_value - trade['fee']):.2f}")
            return final_value, trades
        except Exception as e:
            self.logger.error(f"An error occurred during backtesting: {e}", exc_info=True)
            return initial_balance, []

    def save_state(self, product_id, balance, btc_balance, last_trade_time, last_trade_price, last_buy_price, highest_price_since_buy, trades):
        state = {
            'product_id': product_id,
            'balance': balance,
            'btc_balance': btc_balance,
            'last_trade_time': last_trade_time,
            'last_trade_price': last_trade_price,
            'last_buy_price': last_buy_price,
            'highest_price_since_buy': highest_price_since_buy,
            'trades': trades
        }
        with open('live_trading_state.json', 'w') as f:
            json.dump(state, f)
        self.logger.info("State saved to disk")

    def load_state(self):
        if not os.path.exists('live_trading_state.json'):
            # Create the file if it doesn't exist
            with open('live_trading_state.json', 'w') as f:
                json.dump({}, f)  # Initialize with an empty JSON object
            self.logger.info("State file created as it did not exist.")
        
        with open('live_trading_state.json', 'r') as f:
            state = json.load(f)
        self.logger.info("State loaded from disk")
        return state

    def run_live(self, product_id: str, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float):
        try:
            # Try to load saved state
            saved_state = self.load_state()
            if saved_state and saved_state['product_id'] == product_id:
                balance = saved_state['balance']
                btc_balance = saved_state['btc_balance']
                last_trade_time = saved_state['last_trade_time']
                last_trade_price = saved_state['last_trade_price']
                last_buy_price = saved_state['last_buy_price']
                highest_price_since_buy = saved_state['highest_price_since_buy']
                trades = saved_state['trades']
                self.logger.info("Resuming from saved state")
            else:
                balance = initial_balance
                btc_balance = 0
                last_trade_time = None
                last_trade_price = None
                last_buy_price = None
                highest_price_since_buy = 0
                trades = []

            save_interval = 300  # Save state every 5 minutes
            last_save_time = time.time()

            while True:  # Run continuously
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # Get the last 1 hour of data

                try:
                    # Fetch the most recent candles
                    candles = self.trader.get_historical_data(product_id, start_date, end_date)
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    self.logger.error(f"Network error occurred: {e}. Retrying in 60 seconds...")
                    time.sleep(60)  # Wait for 60 seconds before retrying
                    continue

                if not candles:
                    self.logger.warning("No recent historical data available. Retrying in 60 seconds...")
                    time.sleep(60)  # Wait before trying again
                    continue  # Skip to the next iteration if no data

                # Initialize variables for trading logic
                min_candles = 50  # Adjust this value based on your longest indicator period
                trades_today = 0
                last_trade_date = None

                # Process the most recent candles
                for i, candle in enumerate(candles):
                    close_price = float(candle['close'])
                    current_time = int(candle['start'])
                    current_date = datetime.utcfromtimestamp(current_time).date()

                    # Reset trades_today if it's a new day
                    if last_trade_date != current_date:
                        trades_today = 0
                        last_trade_date = current_date

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
                                trade_size_multiplier = 1.0  # Default multiplier
                                if market_conditions == "Bullish":
                                    trade_size_multiplier = 1.2
                                elif market_conditions == "Bull Market":
                                    trade_size_multiplier = 1.5
                                elif market_conditions == "Bearish":
                                    trade_size_multiplier = 0.8
                                elif market_conditions == "Bear Market":
                                    trade_size_multiplier = 0.5
                                else:
                                    trade_size_multiplier = 1.0

                                # Simulate trade execution based on signal
                                if combined_signal in ["BUY", "STRONG BUY"] and balance > 0:
                                    if last_buy_price is None or close_price < last_buy_price:
                                        # Check for additional entry conditions
                                        trend = self.trader.identify_trend(product_id, candles[:i+1])
                                        volume_signal = self.trader.technical_analysis.analyze_volume(candles[:i+1])
                                        if trend == "Uptrend" and volume_signal == "High":
                                            if combined_signal == "STRONG BUY":
                                                btc_to_buy, fee = self.trader.calculate_trade_amount_and_fee(balance * self.strong_buy_percentage * trade_size_multiplier, close_price, is_buy=True)
                                            else:  # Regular BUY
                                                btc_to_buy, fee = self.trader.calculate_trade_amount_and_fee(balance * self.buy_percentage * trade_size_multiplier, close_price, is_buy=True)
                                            
                                            balance -= (btc_to_buy * close_price + fee)
                                            btc_balance += btc_to_buy
                                            trades.append(self.trader.create_trade_record(current_time, combined_signal, close_price, btc_to_buy, fee))
                                            last_trade_time = current_time
                                            last_trade_price = close_price
                                            last_buy_price = close_price
                                            highest_price_since_buy = close_price
                                            trades_today += 1

                                elif combined_signal in ["SELL", "STRONG SELL"] and btc_balance > 0:
                                    if last_buy_price and (close_price < last_buy_price * (1 - self.drawdown_threshold) or combined_signal == "STRONG SELL"):
                                        amount_to_sell = btc_balance
                                        balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(amount_to_sell * close_price, close_price, is_buy=False)
                                        balance += balance_to_add
                                        btc_balance = 0
                                        trades.append(self.trader.create_trade_record(current_time, "STOP LOSS" if close_price < last_buy_price else combined_signal, close_price, amount_to_sell, fee))
                                        last_trade_time = current_time
                                        last_trade_price = close_price
                                        highest_price_since_buy = 0
                                        trades_today += 1

                    # Update highest price for trailing stop
                    if btc_balance > 0:
                        highest_price_since_buy = max(highest_price_since_buy, close_price)
                        if close_price <= highest_price_since_buy * (1 - trailing_stop_percent):
                            # Trigger trailing stop
                            amount_to_sell = btc_balance
                            balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(amount_to_sell * close_price, close_price, is_buy=False)
                            balance += balance_to_add
                            btc_balance = 0
                            trades.append(self.trader.create_trade_record(current_time, "TRAILING STOP", close_price, amount_to_sell, fee))
                            last_trade_time = current_time
                            last_trade_price = close_price
                            highest_price_since_buy = 0
                            trades_today += 1

                # Save state periodically
                current_time = time.time()
                if current_time - last_save_time >= save_interval:
                    self.save_state(product_id, balance, btc_balance, last_trade_time, last_trade_price, last_buy_price, highest_price_since_buy, trades)
                    last_save_time = current_time

                # Log the current state
                self.logger.info(f"Current balance: {balance:.2f}, BTC balance: {btc_balance:.8f}")
                self.logger.info(f"Last trade: {trades[-1] if trades else 'No trades yet'}")
                for trade in trades:
                    usd_value = trade['amount'] * trade['price']
                    self.logger.info(f"Date: {datetime.utcfromtimestamp(trade['date']).strftime('%Y-%m-%d %H:%M:%S')}, "
                                f"Action: {trade['action']}, Price: {trade['price']:.2f}, "
                                f"Amount: {trade['amount']:.8f}, Fee: {trade['fee']:.2f}, "
                                f"USD Value: {(usd_value - trade['fee']):.2f}")
                # Sleep for a while before the next iteration (e.g., 60 seconds)
                time.sleep(60)

        except Exception as e:
            self.logger.error(f"An error occurred during live trading: {e}", exc_info=True)
            # Save state before exiting due to error
            self.save_state(product_id, balance, btc_balance, last_trade_time, last_trade_price, last_buy_price, highest_price_since_buy, trades)

