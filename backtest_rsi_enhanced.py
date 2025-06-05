#!/usr/bin/env python3
"""
Enhanced RSI Mean Reversion Strategy Backtest
Tests RSI oversold bounces with additional volume and trend filters
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
class TradeSignal:
    timestamp: datetime
    price: float
    rsi: float
    volume_ratio: float
    trend_strength: float
    signal_strength: float

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    profit: Optional[float]
    exit_reason: Optional[str]
    rsi_at_entry: float
    volume_ratio: float
    trend_strength: float

class EnhancedRSIStrategy:
    def __init__(self, rsi_period: int = 14, rsi_oversold: float = 30, 
                 rsi_overbought: float = 70, volume_threshold: float = 1.5,
                 trend_ema: int = 200):
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        self.trend_ema = trend_ema
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Trend analysis
        df['ema_200'] = talib.EMA(df['close'], timeperiod=self.trend_ema)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['trend_strength'] = (df['ema_50'] - df['ema_200']) / df['ema_200'] * 100
        
        # Price action
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals based on enhanced RSI strategy"""
        df = self.calculate_indicators(df)
        signals = []
        
        for i in range(self.trend_ema, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Entry conditions:
            # 1. RSI crosses below oversold level
            # 2. Volume spike (above threshold)
            # 3. Price above 200 EMA (uptrend)
            # 4. Price near lower Bollinger Band
            
            rsi_oversold_cross = (prev_row['rsi'] > self.rsi_oversold and 
                                 row['rsi'] <= self.rsi_oversold)
            
            volume_spike = row['volume_ratio'] > self.volume_threshold
            uptrend = row['close'] > row['ema_200']
            near_bb_lower = row['close'] <= row['bb_lower'] * 1.02
            
            if rsi_oversold_cross and volume_spike and uptrend and near_bb_lower:
                signal_strength = self._calculate_signal_strength(row)
                signals.append(TradeSignal(
                    timestamp=row.name,
                    price=row['close'],
                    rsi=row['rsi'],
                    volume_ratio=row['volume_ratio'],
                    trend_strength=row['trend_strength'],
                    signal_strength=signal_strength
                ))
        
        return signals
    
    def _calculate_signal_strength(self, row: pd.Series) -> float:
        """Calculate signal strength based on multiple factors"""
        strength = 0.0
        
        # RSI component (more oversold = stronger)
        rsi_component = (self.rsi_oversold - row['rsi']) / self.rsi_oversold
        strength += rsi_component * 0.3
        
        # Volume component
        volume_component = min(row['volume_ratio'] / 3.0, 1.0)
        strength += volume_component * 0.3
        
        # Trend component
        trend_component = min(max(row['trend_strength'] / 5.0, 0), 1.0)
        strength += trend_component * 0.2
        
        # Bollinger Band component
        bb_component = max(0, (row['bb_lower'] - row['close']) / row['bb_lower'])
        strength += bb_component * 0.2
        
        return min(strength, 1.0)

def backtest_strategy(product_id: str = 'BTC-USDC', 
                     start_date: str = '2021-01-01',
                     end_date: str = '2025-01-01',
                     initial_balance: float = 10000,
                     position_size_pct: float = 0.1,
                     take_profit: float = 0.02,
                     stop_loss: float = 0.01):
    """Run the enhanced RSI strategy backtest"""
    
    logger.info(f"Starting Enhanced RSI Backtest for {product_id}")
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
    strategy = EnhancedRSIStrategy()
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = strategy.generate_signals(df)
    
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
            rsi_at_entry=signal.rsi,
            volume_ratio=signal.volume_ratio,
            trend_strength=signal.trend_strength
        )
        
        # Find exit
        entry_idx = df_with_indicators.index.get_loc(signal.timestamp)
        tp_price = signal.price * (1 + take_profit)
        sl_price = signal.price * (1 - stop_loss)
        
        for j in range(entry_idx + 1, len(df_with_indicators)):
            row = df_with_indicators.iloc[j]
            
            # Check stop loss
            if row['low'] <= sl_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = sl_price
                current_trade.exit_reason = "Stop Loss"
                current_trade.profit = (sl_price - current_trade.entry_price) * current_trade.position_size
                balance += current_trade.profit
                trades.append(current_trade)
                current_trade = None
                break
                
            # Check take profit
            elif row['high'] >= tp_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = tp_price
                current_trade.exit_reason = "Take Profit"
                current_trade.profit = (tp_price - current_trade.entry_price) * current_trade.position_size
                balance += current_trade.profit
                trades.append(current_trade)
                current_trade = None
                break
                
            # Check RSI exit (overbought)
            elif row['rsi'] >= strategy.rsi_overbought:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "RSI Overbought"
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
        profit_factor = abs(sum(t.profit for t in trades if t.profit > 0) / 
                           sum(t.profit for t in trades if t.profit < 0)) if losing_trades > 0 else float('inf')
    else:
        win_rate = avg_profit = total_profit = profit_factor = 0
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("ENHANCED RSI STRATEGY BACKTEST RESULTS")
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
    logger.info(f"Average Profit per Trade: ${avg_profit:.2f}")
    
    # Export trades
    if trades:
        trades_df = pd.DataFrame([{
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'profit': t.profit,
            'exit_reason': t.exit_reason,
            'rsi_at_entry': t.rsi_at_entry,
            'volume_ratio': t.volume_ratio,
            'trend_strength': t.trend_strength
        } for t in trades])
        
        filename = f"enhanced_rsi_backtest_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"\nTrades exported to: {filename}")
    
    return {
        'strategy': 'Enhanced RSI',
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