from datetime import datetime, timezone, timedelta
import os
import json
import logging
from tqdm import tqdm
import time
from typing import List, Tuple, Dict, Any, Optional

from trading.models import TradeRecord, PerformanceMetrics
from trading.granularity_settings import GRANULARITY_SETTINGS
from trading.visualization import plot_trades  # Add this import

class Backtester:
    def __init__(self, trader: Any):
        self.trader = trader
        self.logger = logging.getLogger(__name__)
        self.granularity_settings = GRANULARITY_SETTINGS
        
        # Initialize drawdown thresholds first
        self.drawdown_thresholds = {
            "ONE_MINUTE": 0.05,   # 5% for very short timeframes
            "FIVE_MINUTE": 0.07,  # 7% for short timeframes
            "FIFTEEN_MINUTE": 0.09, # 9% for medium-short timeframes
            "ONE_HOUR": 0.15,     # 15% for hourly
            "SIX_HOUR": 0.20,     # 20% for medium timeframes
            "ONE_DAY": 0.25       # 25% for daily timeframes
        }
        self.drawdown_threshold = self.drawdown_thresholds["ONE_HOUR"]  # default
        
        # Default settings
        self.atr_period = 14
        self.atr_multiplier = 2
        
        # Apply granularity settings last, as it will update drawdown_threshold
        self.set_granularity_settings("ONE_HOUR")

    def create_trade_record(self, date: int, action: str, price: float, amount: float, fee: float) -> TradeRecord:
        return TradeRecord(date, action, price, amount, fee)

    def calculate_dynamic_stop_loss(self, candles: List[Dict], entry_price: float) -> float:
        if entry_price is None:
            return None
        atr = self.trader.technical_analysis.compute_atr(candles, self.atr_period)
        return entry_price - (atr * self.atr_multiplier)

    def calculate_dynamic_take_profit(self, candles: List[Dict], entry_price: float) -> Tuple[float, float, float]:
        """Calculate multiple dynamic take profit levels based on market conditions.
        Returns a tuple of (tp1, tp2, tp3) representing different take profit targets.
        """
        if entry_price is None:
            return None, None, None
        
        # Calculate ATR for volatility-based take profit
        atr = self.trader.technical_analysis.compute_atr(candles, self.atr_period)
        
        # Calculate trend strength using ADX
        adx = self.trader.technical_analysis.compute_adx(candles, period=14)
        
        # Get RSI for momentum analysis
        rsi = self.trader.compute_rsi_for_backtest(candles)
        
        # Analyze volume profile
        recent_volume = sum(float(candle['volume']) for candle in candles[-5:]) / 5
        volume_sma = sum(float(candle['volume']) for candle in candles[-20:]) / 20
        volume_ratio = recent_volume / volume_sma if volume_sma > 0 else 1
        
        # Base multipliers for different take profit levels
        tp1_multiplier = 1.003  # First target: 0.3% profit
        tp2_multiplier = 1.007  # Second target: 0.7% profit
        tp3_multiplier = 1.015  # Third target: 1.5% profit
        
        # Adjust multipliers based on market conditions
        if adx > 40:  # Very strong trend
            tp1_multiplier += 0.002
            tp2_multiplier += 0.003
            tp3_multiplier += 0.005
        elif adx > 25:  # Strong trend
            tp1_multiplier += 0.001
            tp2_multiplier += 0.002
            tp3_multiplier += 0.003
        
        # Adjust based on RSI
        if rsi > 65:  # Overbought - take profits quicker
            tp1_multiplier *= 0.95
            tp2_multiplier *= 0.97
            tp3_multiplier *= 0.98
        elif rsi < 35:  # Oversold - allow for more upside
            tp1_multiplier *= 1.02
            tp2_multiplier *= 1.03
            tp3_multiplier *= 1.05
        
        # Volume-based adjustments
        if volume_ratio > 1.3:
            tp1_multiplier *= 1.01
            tp2_multiplier *= 1.02
            tp3_multiplier *= 1.03
        
        # Calculate take profit levels
        tp1 = entry_price * tp1_multiplier
        tp2 = entry_price * tp2_multiplier
        tp3 = entry_price * tp3_multiplier
        
        # Look for nearby resistance levels
        recent_highs = [float(candle['high']) for candle in candles[-5:]]  # Shortened to 5 candles
        if recent_highs:
            highs_above_entry = [high for high in recent_highs if high > entry_price]
            if highs_above_entry:
                nearest_resistance = min(highs_above_entry)
                # Adjust take profit levels if resistance is nearby
                if nearest_resistance < tp3:
                    resistance_level = nearest_resistance * 0.999
                    tp3 = min(tp3, resistance_level)
                    tp2 = min(tp2, resistance_level * 0.997)
                    tp1 = min(tp1, resistance_level * 0.995)
        
        # Ensure minimum profit levels
        min_tp1 = entry_price * 1.002  # Minimum 0.2% profit
        min_tp2 = entry_price * 1.004  # Minimum 0.4% profit
        min_tp3 = entry_price * 1.008  # Minimum 0.8% profit
        
        return (
            max(min_tp1, tp1),
            max(min_tp2, tp2),
            max(min_tp3, tp3)
        )

    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float, risk_per_trade: float) -> float:
        """Calculate position size with enhanced risk management"""
        # Base risk calculation
        risk_amount = balance * risk_per_trade
        
        # Calculate volatility
        volatility = abs(entry_price - stop_loss) / entry_price
        
        # Adjust position size based on volatility
        if volatility > 0.05:  # High volatility
            risk_amount *= 0.7  # Reduce position size
        elif volatility < 0.02:  # Low volatility
            risk_amount *= 1.2  # Increase position size
            
        # Ensure minimum position size
        min_position = balance * 0.01  # 1% of balance minimum
        risk_per_coin = entry_price - stop_loss
        position_size = max(risk_amount / risk_per_coin, min_position / entry_price)
        
        return position_size

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
            
            # Fetch historical data
            self.logger.info(f"Fetching all historical data from {start_date} to {end_date} with granularity {granularity}...")
            candles = self.trader.get_historical_data(product_id, start_date, end_date, granularity)
            
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

            # Add tracking for average buy price
            total_btc_bought = 0
            total_cost = 0
            average_buy_price = None
            take_profit1, take_profit2, take_profit3 = None, None, None

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
                
                    # Only generate signals if we have enough historical data and haven't exceeded max trades for the day
                    if i >= min_candles and trades_today < self.max_trades_per_day:
                        # Check if enough time has passed since the last trade
                        if last_trade_time is None or (current_time - last_trade_time) >= self.cooldown_period:
                            if last_trade_price is None or abs(close_price - last_trade_price) / last_trade_price >= self.min_price_change:
                            # Check if price has changed enough from average buy price
                                price_change_ok = (
                                    average_buy_price is None or 
                                    abs(close_price - average_buy_price) / average_buy_price >= self.min_price_change
                                )

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
                                
                                # Check for drawdown
                                drawdown_metrics = PerformanceMetrics.calculate_drawdown_metrics(portfolio_values)                    
                                # Additional risk management based on drawdown
                                if drawdown_metrics["current_drawdown"] > self.drawdown_threshold * 100:
                                    # Reduce position sizes or take defensive actions
                                    trade_size_multiplier *= 1  # Placeholder, maybe change later.
                                        
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
                                            take_profit1, take_profit2, take_profit3 = self.calculate_dynamic_take_profit(candles[:i+1], close_price)
                                            position_size = self.calculate_position_size(balance, close_price, stop_loss, risk_per_trade)
                                            btc_to_buy = min(position_size, balance / close_price)  # Ensure we don't exceed available balance
                                                                                        
                                            # When buying, update the average buy price
                                            if btc_to_buy > 0:
                                                # Update average buy price
                                                total_btc_bought += btc_to_buy
                                                total_cost += (btc_to_buy * close_price)
                                                average_buy_price = total_cost / total_btc_bought

                                elif combined_signal in ["SELL", "STRONG SELL"] and btc_balance > 0:
                                    # Implement a trailing stop
                                    should_sell = False
                                    sell_reason = ""
                                    sell_amount = 0

                                    # Check all sell conditions
                                    if close_price < last_trade_price * (1 - self.drawdown_threshold) or combined_signal == "STRONG SELL":
                                        should_sell = True
                                        sell_amount = btc_balance
                                        sell_reason = "STOP LOSS"
                                    elif close_price <= stop_loss:
                                        should_sell = True
                                        sell_amount = btc_balance
                                        sell_reason = "STOP LOSS"
                                    # Use average buy price for take profit levels
                                    elif average_buy_price is not None:  # Only check take profits if we have an average buy price
                                        if close_price >= average_buy_price * 1.015:  # TP3: 1.5% above average entry
                                            should_sell = True
                                            sell_amount = btc_balance  # Sell 100% of position
                                            sell_reason = "TAKE_PROFIT_3"
                                        elif close_price >= average_buy_price * 1.007:  # TP2: 0.7% above average entry
                                            should_sell = True
                                            sell_amount = btc_balance * 0.5  # Sell 50% of position
                                            sell_reason = "TAKE_PROFIT_2"
                                        elif close_price >= average_buy_price * 1.003:  # TP1: 0.3% above average entry
                                            should_sell = True
                                            sell_amount = btc_balance * 0.3  # Sell 30% of position
                                            sell_reason = "TAKE_PROFIT_1"

                                    # Execute sell if any condition is met
                                    if should_sell and sell_amount > 0:
                                        balance_to_add, fee = self.trader.calculate_trade_amount_and_fee(sell_amount * close_price, close_price, is_buy=False)
                                        balance += balance_to_add
                                        btc_balance -= sell_amount
                                        trades.append(self.create_trade_record(current_time, sell_reason, close_price, sell_amount, fee))
                                        last_trade_time = current_time
                                        last_trade_price = close_price
                                        trades_today += 1

                                        # Update the totals after selling
                                        remaining_btc = total_btc_bought - sell_amount
                                        if remaining_btc > 0:
                                            total_btc_bought = remaining_btc
                                            total_cost = total_cost * (remaining_btc / (remaining_btc + sell_amount))
                                        else:
                                            total_btc_bought = 0
                                            total_cost = 0
                                            average_buy_price = None

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

            # Update the take profit logging at the end
            if take_profit1 is not None and take_profit2 is not None and take_profit3 is not None:
                self.logger.info(f"Final take profit levels - TP1: {take_profit1:.2f} USD | TP2: {take_profit2:.2f} USD | TP3: {take_profit3:.2f} USD | Combined Signal: {combined_signal}")
                print(f"Min price change: {abs(close_price - last_trade_price) / last_trade_price} | Min price change threshold: {self.min_price_change}")
            else:
                self.logger.info("No take profit levels set (no buy trades executed)")

            # Print the current trailing stop value in Bitcoin terms
            last_btc_price = float(candles[-1]['close'])
            if highest_price_since_buy > 0:
                trailing_stop_btc_value = highest_price_since_buy * (1 - trailing_stop_percent)
                self.logger.info(
                    f"Current trailing stop: {trailing_stop_btc_value:.2f} USD | "
                    f"Peak price since buy: {highest_price_since_buy:.2f} USD | "
                    f"Stop triggers at: {trailing_stop_percent * 100:.2f}% drop from peak"
                )
            else:
                self.logger.info("No buy trades executed, so no trailing stop was set.")
            
            metrics = PerformanceMetrics.calculate_metrics(portfolio_values, buy_and_hold_values)
            self.logger.info(
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f} | "
                f"Sortino Ratio: {metrics['sortino_ratio']:.4f} | "
                f"Max Drawdown: {metrics['max_drawdown']:.2f}% | "
                f"Sharpe Ratio Buy and Hold: {metrics['sharpe_ratio_buy_and_hold']:.4f} | "
                f"Sortino Ratio Buy and Hold: {metrics['sortino_ratio_buy_and_hold']:.4f}"
            )

            # After updating portfolio_values...
            if len(portfolio_values) > 0:
                drawdown_metrics = PerformanceMetrics.calculate_drawdown_metrics(portfolio_values)
                
                # Log drawdown metrics if they exceed certain thresholds
                if drawdown_metrics["current_drawdown"] > 5:  # Alert on 5% drawdown
                    self.logger.warning(
                        f"Current drawdown: {drawdown_metrics['current_drawdown']:.2f}% | "
                        f"Duration: {drawdown_metrics['drawdown_duration']} periods | "
                        f"Max drawdown: {drawdown_metrics['max_drawdown']:.2f}%"
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
        
        # Update drawdown threshold based on granularity
        self.drawdown_threshold = self.drawdown_thresholds.get(granularity, self.drawdown_thresholds["ONE_HOUR"])



