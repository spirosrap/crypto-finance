from datetime import datetime
import os
import json
import logging
from tqdm import tqdm
import numpy as np

class Backtester:
    def __init__(self, trader):
        self.trader = trader
        self.logger = logging.getLogger(__name__)

    def backtest(self, product_id: str, start_date, end_date, initial_balance: float, risk_per_trade: float, trailing_stop_percent: float):
        try:
            # Convert start_date and end_date to datetime objects if they're strings
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

            self.logger.info(f"Starting backtest for {product_id} from {start_date} to {end_date}.")

            # Create a directory for candle files if it doesn't exist
            candle_dir = "candle_data"
            os.makedirs(candle_dir, exist_ok=True)

            # Create a filename based on the product_id and date range
            filename = f"{product_id}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            filepath = os.path.join(candle_dir, filename)
            
            if os.path.exists(filepath):
                # Load historical data from file if it exists
                self.logger.info(f"Loading historical data from {filepath}...")
                with open(filepath, 'r') as f:
                    candles = json.load(f)
            else:
                # Fetch historical data if file doesn't exist
                self.logger.info(f"Fetching historical data from {start_date} to {end_date}...")
                candles = self.trader.get_historical_data(product_id, start_date, end_date)
                
                # Save the fetched data to a file
                with open(filepath, 'w') as f:
                    json.dump(candles, f)
                self.logger.info(f"Saved historical data to {filepath}")
            self.logger.info(f"Fetched {len(candles)} candles.")

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
            cooldown_period = 24 * 60 * 60 * 3  # 3 days in seconds
            max_trades_per_day = 1
            min_price_change = 0.07  # 7% minimum price change
            trades_today = 0
            last_trade_date = None
            last_buy_price = None
            drawdown_threshold = 0.1  # 10% drawdown threshold

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

                    # Only generate signals if we have enough historical data and haven't exceeded max trades for the day
                    if i >= min_candles and trades_today < max_trades_per_day:
                        # Check if enough time has passed since the last trade
                        if last_trade_time is None or (current_time - last_trade_time) >= cooldown_period:
                            # Check if the price has changed enough since the last trade
                            if last_trade_price is None or abs(close_price - last_trade_price) / last_trade_price >= min_price_change:
                                # Calculate indicators and generate signal
                                rsi = self.trader.compute_rsi_for_backtest(candles[:i+1])
                                macd, signal, histogram = self.trader.compute_macd_for_backtest(candles[:i+1])
                                combined_signal = self.trader.generate_combined_signal(rsi, macd, signal, histogram, candles[:i+1])

                                # Execute trade based on signal
                                if combined_signal in ["BUY", "STRONG BUY"] and balance > 0:
                                    # Only buy if the current price is lower than the last buy price or if it's the first buy
                                    if last_buy_price is None or close_price < last_buy_price:
                                        # Check for additional entry conditions
                                        trend = self.trader.identify_trend(product_id, candles[:i+1])
                                        volume_signal = self.trader.technical_analysis.analyze_volume(candles[:i+1])
                                        if trend == "Uptrend" and volume_signal == "High":  # Ensure trend and volume conditions are met
                                            if combined_signal == "STRONG BUY":
                                                btc_to_buy, fee = self.trader.calculate_trade_amount_and_fee(balance * 0.7, close_price, is_buy=True)
                                                balance -= (btc_to_buy * close_price + fee)
                                            else:  # Regular BUY
                                                btc_to_buy, fee = self.trader.calculate_trade_amount_and_fee(balance * 0.3, close_price, is_buy=True)
                                                balance -= (btc_to_buy * close_price + fee)

                                            btc_balance += btc_to_buy
                                            trades.append(self.trader.create_trade_record(current_time, combined_signal, close_price, btc_to_buy, fee))
                                            last_trade_time = current_time
                                            last_trade_price = close_price
                                            highest_price_since_buy = close_price  # Reset highest price after buying
                                            trades_today += 1

                                elif combined_signal in ["SELL", "STRONG SELL"] and btc_balance > 0:
                                    # Implement a trailing stop
                                    if last_buy_price and (close_price < last_buy_price * (1 - drawdown_threshold) or combined_signal == "STRONG SELL"):
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
