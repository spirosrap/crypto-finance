#!/usr/bin/env python3
"""
Momentum Breakout Strategy Backtest
Tests breakouts above resistance levels with volume confirmation
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
class BreakoutSignal:
    timestamp: datetime
    price: float
    resistance_level: float
    volume_surge: float
    momentum_score: float
    breakout_strength: float

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    profit: Optional[float]
    exit_reason: Optional[str]
    breakout_strength: float
    volume_surge: float
    momentum_score: float

class MomentumBreakoutStrategy:
    def __init__(self, lookback_period: int = 20, volume_multiplier: float = 2.0,
                 momentum_threshold: float = 0.02, atr_multiplier: float = 2.0):
        self.lookback_period = lookback_period
        self.volume_multiplier = volume_multiplier
        self.momentum_threshold = momentum_threshold
        self.atr_multiplier = atr_multiplier
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Price levels
        df['resistance'] = df['high'].rolling(window=self.lookback_period).max()
        df['support'] = df['low'].rolling(window=self.lookback_period).min()
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum indicators
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Moving averages for trend
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # Price strength
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        return df
    
    def identify_breakouts(self, df: pd.DataFrame) -> List[BreakoutSignal]:
        """Identify momentum breakout opportunities"""
        df = self.calculate_indicators(df)
        signals = []
        
        for i in range(max(self.lookback_period, 200), len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Breakout conditions:
            # 1. Price breaks above recent resistance
            # 2. Volume surge (above multiplier)
            # 3. Strong momentum (ROC above threshold)
            # 4. RSI not overbought (< 70)
            # 5. Price above key EMAs
            
            price_breakout = (prev_row['close'] <= prev_row['resistance'] and 
                            row['close'] > row['resistance'])
            
            volume_surge = row['volume_ratio'] > self.volume_multiplier
            strong_momentum = row['roc'] > self.momentum_threshold * 100
            rsi_room = row['rsi'] < 70
            trend_alignment = (row['close'] > row['ema_20'] > row['ema_50'] > row['ema_200'])
            
            if price_breakout and volume_surge and strong_momentum and rsi_room and trend_alignment:
                breakout_strength = self._calculate_breakout_strength(row, prev_row)
                momentum_score = self._calculate_momentum_score(row)
                
                signals.append(BreakoutSignal(
                    timestamp=row.name,
                    price=row['close'],
                    resistance_level=row['resistance'],
                    volume_surge=row['volume_ratio'],
                    momentum_score=momentum_score,
                    breakout_strength=breakout_strength
                ))
        
        return signals
    
    def _calculate_breakout_strength(self, row: pd.Series, prev_row: pd.Series) -> float:
        """Calculate the strength of the breakout"""
        # Price breakout magnitude
        breakout_pct = (row['close'] - row['resistance']) / row['resistance']
        
        # Volume confirmation
        volume_strength = min(row['volume_ratio'] / 3.0, 1.0)
        
        # Momentum strength
        momentum_strength = min(row['roc'] / 5.0, 1.0)
        
        # Combine factors
        strength = (breakout_pct * 0.4 + volume_strength * 0.4 + momentum_strength * 0.2)
        
        return min(strength, 1.0)
    
    def _calculate_momentum_score(self, row: pd.Series) -> float:
        """Calculate overall momentum score"""
        score = 0.0
        
        # ROC component
        roc_score = min(max(row['roc'] / 10.0, 0), 1.0)
        score += roc_score * 0.3
        
        # MACD component
        macd_score = 1.0 if row['macd'] > row['macd_signal'] and row['macd_hist'] > 0 else 0.5
        score += macd_score * 0.2
        
        # RSI component (50-70 is ideal)
        if 50 <= row['rsi'] <= 70:
            rsi_score = 1.0
        else:
            rsi_score = 0.5
        score += rsi_score * 0.2
        
        # Price position component
        position_score = row['price_position'] if not pd.isna(row['price_position']) else 0.5
        score += position_score * 0.3
        
        return min(score, 1.0)

def backtest_strategy(product_id: str = 'BTC-USDC', 
                     start_date: str = '2021-01-01',
                     end_date: str = '2025-01-01',
                     initial_balance: float = 10000,
                     position_size_pct: float = 0.15,
                     trailing_stop_atr: float = 2.0):
    """Run the momentum breakout strategy backtest"""
    
    logger.info(f"Starting Momentum Breakout Backtest for {product_id}")
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
    strategy = MomentumBreakoutStrategy()
    
    # Generate signals
    logger.info("Identifying breakout opportunities...")
    signals = strategy.identify_breakouts(df)
    
    # Backtest execution
    balance = initial_balance
    trades = []
    current_trade = None
    
    df_with_indicators = strategy.calculate_indicators(df)
    
    for signal in signals:
        # Skip if we already have a position
        if current_trade is not None:
            continue
            
        # Calculate position size
        position_value = balance * position_size_pct
        position_size = position_value / signal.price
        
        # Create trade
        current_trade = Trade(
            entry_time=signal.timestamp,
            exit_time=None,
            entry_price=signal.price,
            exit_price=None,
            position_size=position_size,
            profit=None,
            exit_reason=None,
            breakout_strength=signal.breakout_strength,
            volume_surge=signal.volume_surge,
            momentum_score=signal.momentum_score
        )
        
        # Find exit with trailing stop
        entry_idx = df_with_indicators.index.get_loc(signal.timestamp)
        entry_atr = df_with_indicators.iloc[entry_idx]['atr']
        
        highest_price = signal.price
        trailing_stop = signal.price - (entry_atr * trailing_stop_atr)
        
        for j in range(entry_idx + 1, len(df_with_indicators)):
            row = df_with_indicators.iloc[j]
            
            # Update trailing stop if price moved up
            if row['high'] > highest_price:
                highest_price = row['high']
                new_trailing_stop = highest_price - (row['atr'] * trailing_stop_atr)
                trailing_stop = max(trailing_stop, new_trailing_stop)
            
            # Check if trailing stop hit
            if row['low'] <= trailing_stop:
                current_trade.exit_time = row.name
                current_trade.exit_price = trailing_stop
                current_trade.exit_reason = "Trailing Stop"
                current_trade.profit = (trailing_stop - current_trade.entry_price) * current_trade.position_size
                balance += current_trade.profit
                trades.append(current_trade)
                current_trade = None
                break
            
            # Exit if momentum reverses (RSI > 80 or price below EMA20)
            if row['rsi'] > 80:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "RSI Overbought"
                current_trade.profit = (row['close'] - current_trade.entry_price) * current_trade.position_size
                balance += current_trade.profit
                trades.append(current_trade)
                current_trade = None
                break
            
            if row['close'] < row['ema_20']:
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
    else:
        win_rate = avg_profit = total_profit = profit_factor = avg_win = avg_loss = 0
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("MOMENTUM BREAKOUT STRATEGY BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"Final Balance: ${balance:,.2f}")
    logger.info(f"Total Profit: ${total_profit:,.2f}")
    logger.info(f"Return: {(balance/initial_balance - 1)*100:.2f}%")
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
            'exit_reason': t.exit_reason,
            'breakout_strength': t.breakout_strength,
            'volume_surge': t.volume_surge,
            'momentum_score': t.momentum_score
        } for t in trades])
        
        filename = f"momentum_breakout_backtest_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"\nTrades exported to: {filename}")
    
    return {
        'strategy': 'Momentum Breakout',
        'period': f"{start_date} to {end_date}",
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_profit': total_profit,
        'return_pct': (balance/initial_balance - 1)*100,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

if __name__ == "__main__":
    backtest_strategy()