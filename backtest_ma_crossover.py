#!/usr/bin/env python3
"""
Moving Average Crossover Strategy Backtest
Tests EMA crossovers with trend confirmation and volume filters
Period: 2021-2025
"""

from simplified_trading_bot import (
    CoinbaseService, TechnicalAnalysis, GRANULARITY,
    get_perp_product, get_price_precision
)
from datetime import datetime, timedelta, UTC
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import talib

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CrossoverSignal:
    timestamp: datetime
    price: float
    fast_ema: float
    slow_ema: float
    signal_type: str  # 'golden_cross' or 'death_cross'
    trend_strength: float
    volume_confirmation: bool

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    profit: Optional[float]
    exit_reason: Optional[str]
    signal_type: str
    trend_strength: float
    max_profit: float
    max_loss: float

class MACrossoverStrategy:
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                 signal_period: int = 9, trend_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.trend_period = trend_period
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # EMAs for crossover
        df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.fast_period)
        df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.slow_period)
        df['ema_trend'] = talib.EMA(df['close'], timeperiod=self.trend_period)
        
        # Additional EMAs for multi-timeframe analysis
        df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        # MACD for confirmation
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'], 
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ADX for trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # RSI for overbought/oversold
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Calculate crossover signals
        df['ema_diff'] = df['ema_fast'] - df['ema_slow']
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        return df
    
    def identify_crossovers(self, df: pd.DataFrame) -> List[CrossoverSignal]:
        """Identify MA crossover signals"""
        df = self.calculate_indicators(df)
        signals = []
        
        for i in range(self.trend_period, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check for golden cross (bullish)
            golden_cross = (prev_row['ema_diff'] <= 0 and row['ema_diff'] > 0)
            
            # Check for death cross (bearish) - for exit signals
            death_cross = (prev_row['ema_diff'] >= 0 and row['ema_diff'] < 0)
            
            if golden_cross or death_cross:
                # Additional filters for golden cross
                if golden_cross:
                    # Must be above long-term trend
                    above_trend = row['close'] > row['ema_trend']
                    
                    # Volume confirmation
                    volume_confirm = row['volume_ratio'] > 1.2
                    
                    # Not overbought
                    not_overbought = row['rsi'] < 70
                    
                    # Trend strength (ADX > 20 indicates trending market)
                    trending = row['adx'] > 20
                    
                    if above_trend and not_overbought and trending:
                        trend_strength = self._calculate_trend_strength(row)
                        
                        signals.append(CrossoverSignal(
                            timestamp=row.name,
                            price=row['close'],
                            fast_ema=row['ema_fast'],
                            slow_ema=row['ema_slow'],
                            signal_type='golden_cross',
                            trend_strength=trend_strength,
                            volume_confirmation=volume_confirm
                        ))
                
                # Death cross signals (for exits or shorts)
                elif death_cross:
                    trend_strength = self._calculate_trend_strength(row)
                    
                    signals.append(CrossoverSignal(
                        timestamp=row.name,
                        price=row['close'],
                        fast_ema=row['ema_fast'],
                        slow_ema=row['ema_slow'],
                        signal_type='death_cross',
                        trend_strength=trend_strength,
                        volume_confirmation=row['volume_ratio'] > 1.0
                    ))
        
        return signals
    
    def _calculate_trend_strength(self, row: pd.Series) -> float:
        """Calculate overall trend strength"""
        strength = 0.0
        
        # EMA alignment (all EMAs in order)
        if row['ema_9'] > row['ema_21'] > row['ema_50'] > row['ema_trend']:
            ema_score = 1.0
        elif row['ema_9'] < row['ema_21'] < row['ema_50'] < row['ema_trend']:
            ema_score = -1.0
        else:
            ema_score = 0.0
        strength += abs(ema_score) * 0.3
        
        # ADX strength
        adx_score = min(row['adx'] / 40.0, 1.0) if not pd.isna(row['adx']) else 0.0
        strength += adx_score * 0.3
        
        # MACD histogram momentum
        if not pd.isna(row['macd_hist']):
            macd_score = min(abs(row['macd_hist']) / (row['close'] * 0.001), 1.0)
            strength += macd_score * 0.2
        
        # Price distance from trend EMA
        distance_score = abs(row['close'] - row['ema_trend']) / row['ema_trend']
        distance_score = min(distance_score * 10, 1.0)
        strength += distance_score * 0.2
        
        return min(strength, 1.0)

def backtest_strategy(product_id: str = 'BTC-USDC', 
                     start_date: str = '2021-01-01',
                     end_date: str = '2025-01-01',
                     initial_balance: float = 10000,
                     position_size_pct: float = 0.2,
                     stop_loss_atr: float = 2.0,
                     take_profit_atr: float = 3.0):
    """Run the MA crossover strategy backtest"""
    
    logger.info(f"Starting MA Crossover Backtest for {product_id}")
    logger.info(f"Period: {start_date} to {end_date}")
    
    # Initialize services
    from config import API_KEY_PERPS, API_SECRET_PERPS
    cb = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
    
    # Fetch historical data
    start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=UTC)
    end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=UTC)
    
    logger.info("Fetching historical data...")
    raw_data = cb.historical_data.get_historical_data(product_id, start, end, GRANULARITY)
    df = pd.DataFrame(raw_data)
    
    # Convert columns
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_datetime(df['start'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    
    # Initialize strategy
    strategy = MACrossoverStrategy()
    
    # Generate signals
    logger.info("Identifying crossover signals...")
    signals = strategy.identify_crossovers(df)
    
    # Backtest execution
    balance = initial_balance
    trades = []
    current_trade = None
    
    df_with_indicators = strategy.calculate_indicators(df)
    
    for signal in signals:
        # Handle golden cross signals (entry)
        if signal.signal_type == 'golden_cross' and current_trade is None:
            # Calculate position size
            position_value = balance * position_size_pct
            position_size = position_value / signal.price
            
            # Get ATR for stop/target calculation
            signal_idx = df_with_indicators.index.get_loc(signal.timestamp)
            atr = df_with_indicators.iloc[signal_idx]['atr']
            
            # Create trade
            current_trade = Trade(
                entry_time=signal.timestamp,
                exit_time=None,
                entry_price=signal.price,
                exit_price=None,
                position_size=position_size,
                profit=None,
                exit_reason=None,
                signal_type=signal.signal_type,
                trend_strength=signal.trend_strength,
                max_profit=0,
                max_loss=0
            )
            
            # Set initial stop loss and take profit
            stop_loss = signal.price - (atr * stop_loss_atr)
            take_profit = signal.price + (atr * take_profit_atr)
            
        # Handle death cross signals (exit if in position)
        elif signal.signal_type == 'death_cross' and current_trade is not None:
            current_trade.exit_time = signal.timestamp
            current_trade.exit_price = signal.price
            current_trade.exit_reason = "Death Cross"
            current_trade.profit = (signal.price - current_trade.entry_price) * current_trade.position_size
            balance += current_trade.profit
            trades.append(current_trade)
            current_trade = None
            continue
        
        # Manage open position
        if current_trade is not None:
            entry_idx = df_with_indicators.index.get_loc(current_trade.entry_time)
            
            for j in range(entry_idx + 1, len(df_with_indicators)):
                row = df_with_indicators.iloc[j]
                
                # Track max profit/loss
                current_profit = (row['high'] - current_trade.entry_price) / current_trade.entry_price * 100
                current_loss = (row['low'] - current_trade.entry_price) / current_trade.entry_price * 100
                current_trade.max_profit = max(current_trade.max_profit, current_profit)
                current_trade.max_loss = min(current_trade.max_loss, current_loss)
                
                # Check stop loss
                if row['low'] <= stop_loss:
                    current_trade.exit_time = row.name
                    current_trade.exit_price = stop_loss
                    current_trade.exit_reason = "Stop Loss"
                    current_trade.profit = (stop_loss - current_trade.entry_price) * current_trade.position_size
                    balance += current_trade.profit
                    trades.append(current_trade)
                    current_trade = None
                    break
                
                # Check take profit
                elif row['high'] >= take_profit:
                    current_trade.exit_time = row.name
                    current_trade.exit_price = take_profit
                    current_trade.exit_reason = "Take Profit"
                    current_trade.profit = (take_profit - current_trade.entry_price) * current_trade.position_size
                    balance += current_trade.profit
                    trades.append(current_trade)
                    current_trade = None
                    break
                
                # Trail stop loss after significant profit
                if current_profit > 2.0:  # After 2% profit
                    new_stop = current_trade.entry_price + (current_trade.entry_price * 0.005)  # Trail to 0.5% profit
                    stop_loss = max(stop_loss, new_stop)
                
                # Check for trend reversal exit
                if row['ema_fast'] < row['ema_slow'] * 0.995:  # Fast EMA crosses below slow with margin
                    current_trade.exit_time = row.name
                    current_trade.exit_price = row['close']
                    current_trade.exit_reason = "Trend Reversal"
                    current_trade.profit = (row['close'] - current_trade.entry_price) * current_trade.position_size
                    balance += current_trade.profit
                    trades.append(current_trade)
                    current_trade = None
                    break
    
    # Calculate statistics
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.profit > 0])
    losing_trades = len([t for t in trades if t.profit < 0])
    
    if total_trades > 0:
        win_rate = winning_trades / total_trades * 100
        avg_profit = sum(t.profit for t in trades) / total_trades
        total_profit = sum(t.profit for t in trades)
        
        avg_win = sum(t.profit for t in trades if t.profit > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t.profit for t in trades if t.profit < 0) / losing_trades if losing_trades > 0 else 0
        
        profit_factor = abs(sum(t.profit for t in trades if t.profit > 0) / 
                           sum(t.profit for t in trades if t.profit < 0)) if losing_trades > 0 else float('inf')
        
        # Calculate max drawdown
        equity_curve = [initial_balance]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.profit)
        
        peak = initial_balance
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
    else:
        win_rate = avg_profit = total_profit = profit_factor = avg_win = avg_loss = max_drawdown = 0
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("MA CROSSOVER STRATEGY BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"Final Balance: ${balance:,.2f}")
    logger.info(f"Total Profit: ${total_profit:,.2f}")
    logger.info(f"Return: {(balance/initial_balance - 1)*100:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Winning Trades: {winning_trades}")
    logger.info(f"Losing Trades: {losing_trades}")
    logger.info(f"Win Rate: {win_rate:.2f}%")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info(f"Average Win: ${avg_win:.2f}")
    logger.info(f"Average Loss: ${avg_loss:.2f}")
    
    # Export trades
    if trades:
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'profit': t.profit,
            'profit_pct': t.profit / (t.entry_price * t.position_size) * 100,
            'exit_reason': t.exit_reason,
            'signal_type': t.signal_type,
            'trend_strength': t.trend_strength,
            'max_profit_pct': t.max_profit,
            'max_loss_pct': t.max_loss,
            'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600
        } for t in trades])
        
        filename = f"ma_crossover_backtest_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"\nTrades exported to: {filename}")
    
    return {
        'strategy': 'MA Crossover',
        'period': f"{start_date} to {end_date}",
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_profit': total_profit,
        'return_pct': (balance/initial_balance - 1)*100,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

if __name__ == "__main__":
    backtest_strategy()