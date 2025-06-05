#!/usr/bin/env python3
"""
Volatility Breakout Strategy Backtest
Tests Bollinger Band squeezers and ATR-based volatility expansions
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
class VolatilitySignal:
    timestamp: datetime
    price: float
    signal_type: str  # 'bb_squeeze', 'volatility_expansion', 'range_breakout'
    volatility_ratio: float
    bb_width: float
    atr: float
    strength: float

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
    volatility_ratio: float
    max_profit_pct: float
    max_loss_pct: float

class VolatilityBreakoutStrategy:
    def __init__(self, bb_period: int = 20, bb_dev: float = 2.0, 
                 atr_period: int = 14, squeeze_threshold: float = 0.015):
        self.bb_period = bb_period
        self.bb_dev = bb_dev
        self.atr_period = atr_period
        self.squeeze_threshold = squeeze_threshold
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev
        )
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_width_sma'] = df['bb_width'].rolling(window=50).mean()
        
        # ATR and volatility metrics
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        df['atr_sma'] = df['atr'].rolling(window=20).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_sma']
        
        # Keltner Channels (for squeeze detection)
        df['kc_middle'] = talib.EMA(df['close'], timeperiod=20)
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * 1.5)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * 1.5)
        
        # Price range analysis
        df['range'] = df['high'] - df['low']
        df['range_sma'] = df['range'].rolling(window=20).mean()
        df['range_ratio'] = df['range'] / df['range_sma']
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Trend
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # Historical volatility
        df['returns'] = df['close'].pct_change()
        df['hv_20'] = df['returns'].rolling(window=20).std() * np.sqrt(365 * 24 * 12)  # Annualized for 5min bars
        
        return df
    
    def identify_volatility_setups(self, df: pd.DataFrame) -> List[VolatilitySignal]:
        """Identify volatility-based trading opportunities"""
        df = self.calculate_indicators(df)
        signals = []
        
        for i in range(max(self.bb_period, 200), len(df)):
            row = df.iloc[i]
            prev_rows = df.iloc[i-10:i]  # Look at recent history
            
            # 1. Bollinger Band Squeeze Breakout
            bb_squeeze = row['bb_upper'] < row['kc_upper'] and row['bb_lower'] > row['kc_lower']
            squeeze_release = (prev_rows['bb_width'].iloc[-2] < self.squeeze_threshold and 
                             row['bb_width'] > self.squeeze_threshold)
            
            if squeeze_release and row['volume_ratio'] > 1.5:
                # Determine direction based on price action
                if row['close'] > row['bb_upper'] and row['close'] > row['ema_20']:
                    strength = self._calculate_signal_strength(row, 'bb_squeeze')
                    signals.append(VolatilitySignal(
                        timestamp=row.name,
                        price=row['close'],
                        signal_type='bb_squeeze',
                        volatility_ratio=row['volatility_ratio'],
                        bb_width=row['bb_width'],
                        atr=row['atr'],
                        strength=strength
                    ))
            
            # 2. Volatility Expansion Breakout
            volatility_expanding = (row['volatility_ratio'] > 1.5 and 
                                  prev_rows['volatility_ratio'].mean() < 1.0)
            
            if volatility_expanding and row['adx'] > 25:
                # Check for directional breakout
                if row['close'] > prev_rows['high'].max() and row['volume_ratio'] > 1.3:
                    strength = self._calculate_signal_strength(row, 'volatility_expansion')
                    signals.append(VolatilitySignal(
                        timestamp=row.name,
                        price=row['close'],
                        signal_type='volatility_expansion',
                        volatility_ratio=row['volatility_ratio'],
                        bb_width=row['bb_width'],
                        atr=row['atr'],
                        strength=strength
                    ))
            
            # 3. Range Breakout (after consolidation)
            low_volatility_period = prev_rows['atr_percent'].mean() < 0.5
            range_breakout = (row['range_ratio'] > 2.0 and 
                            row['close'] > prev_rows['high'].max())
            
            if low_volatility_period and range_breakout and row['volume_ratio'] > 2.0:
                strength = self._calculate_signal_strength(row, 'range_breakout')
                signals.append(VolatilitySignal(
                    timestamp=row.name,
                    price=row['close'],
                    signal_type='range_breakout',
                    volatility_ratio=row['volatility_ratio'],
                    bb_width=row['bb_width'],
                    atr=row['atr'],
                    strength=strength
                ))
        
        return signals
    
    def _calculate_signal_strength(self, row: pd.Series, signal_type: str) -> float:
        """Calculate signal strength based on multiple factors"""
        strength = 0.0
        
        if signal_type == 'bb_squeeze':
            # BB width expansion rate
            bb_expansion = min(row['bb_width'] / self.squeeze_threshold, 2.0) / 2.0
            strength += bb_expansion * 0.4
            
            # Volume confirmation
            volume_score = min(row['volume_ratio'] / 2.0, 1.0)
            strength += volume_score * 0.3
            
            # ADX strength
            adx_score = min(row['adx'] / 40.0, 1.0) if not pd.isna(row['adx']) else 0.5
            strength += adx_score * 0.3
            
        elif signal_type == 'volatility_expansion':
            # Volatility ratio
            vol_score = min(row['volatility_ratio'] / 2.0, 1.0)
            strength += vol_score * 0.4
            
            # Trend alignment
            if row['ema_20'] > row['ema_50'] > row['ema_200']:
                strength += 0.3
            
            # Momentum (RSI)
            if 40 < row['rsi'] < 70:
                strength += 0.3
                
        elif signal_type == 'range_breakout':
            # Range expansion
            range_score = min(row['range_ratio'] / 3.0, 1.0)
            strength += range_score * 0.4
            
            # Volume surge
            volume_score = min(row['volume_ratio'] / 3.0, 1.0)
            strength += volume_score * 0.4
            
            # Price position
            if row['close'] > row['bb_upper']:
                strength += 0.2
        
        return min(strength, 1.0)

def backtest_strategy(product_id: str = 'BTC-USDC', 
                     start_date: str = '2021-01-01',
                     end_date: str = '2025-01-01',
                     initial_balance: float = 10000,
                     position_size_pct: float = 0.15,
                     atr_stop_multiplier: float = 2.0,
                     atr_target_multiplier: float = 3.0):
    """Run the volatility breakout strategy backtest"""
    
    logger.info(f"Starting Volatility Breakout Backtest for {product_id}")
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
    strategy = VolatilityBreakoutStrategy()
    
    # Generate signals
    logger.info("Identifying volatility setups...")
    signals = strategy.identify_volatility_setups(df)
    
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
            signal_type=signal.signal_type,
            volatility_ratio=signal.volatility_ratio,
            max_profit_pct=0,
            max_loss_pct=0
        )
        
        # Set dynamic stops based on ATR
        stop_loss = signal.price - (signal.atr * atr_stop_multiplier)
        take_profit = signal.price + (signal.atr * atr_target_multiplier)
        
        # Find exit
        entry_idx = df_with_indicators.index.get_loc(signal.timestamp)
        
        for j in range(entry_idx + 1, len(df_with_indicators)):
            row = df_with_indicators.iloc[j]
            
            # Track max profit/loss
            current_profit_pct = (row['high'] - current_trade.entry_price) / current_trade.entry_price * 100
            current_loss_pct = (row['low'] - current_trade.entry_price) / current_trade.entry_price * 100
            current_trade.max_profit_pct = max(current_trade.max_profit_pct, current_profit_pct)
            current_trade.max_loss_pct = min(current_trade.max_loss_pct, current_loss_pct)
            
            # Dynamic stop adjustment based on volatility
            current_atr = row['atr']
            if signal.signal_type == 'bb_squeeze' and row['bb_width'] < strategy.squeeze_threshold:
                # Tighten stop if volatility contracts again
                stop_loss = max(stop_loss, signal.price - (current_atr * 1.5))
            
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
            
            # Exit on volatility contraction
            if row['volatility_ratio'] < 0.5 and current_profit_pct > 0:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "Volatility Contraction"
                current_trade.profit = (row['close'] - current_trade.entry_price) * current_trade.position_size
                balance += current_trade.profit
                trades.append(current_trade)
                current_trade = None
                break
            
            # Trail stop after significant move
            if current_profit_pct > 1.5:
                new_stop = current_trade.entry_price + (current_atr * 0.5)
                stop_loss = max(stop_loss, new_stop)
    
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
        
        # Strategy-specific stats
        bb_squeeze_trades = [t for t in trades if t.signal_type == 'bb_squeeze']
        vol_expansion_trades = [t for t in trades if t.signal_type == 'volatility_expansion']
        range_breakout_trades = [t for t in trades if t.signal_type == 'range_breakout']
        
        bb_win_rate = (len([t for t in bb_squeeze_trades if t.profit > 0]) / 
                      len(bb_squeeze_trades) * 100) if bb_squeeze_trades else 0
        vol_win_rate = (len([t for t in vol_expansion_trades if t.profit > 0]) / 
                       len(vol_expansion_trades) * 100) if vol_expansion_trades else 0
        range_win_rate = (len([t for t in range_breakout_trades if t.profit > 0]) / 
                         len(range_breakout_trades) * 100) if range_breakout_trades else 0
    else:
        win_rate = avg_profit = total_profit = profit_factor = avg_win = avg_loss = 0
        bb_win_rate = vol_win_rate = range_win_rate = 0
        bb_squeeze_trades = vol_expansion_trades = range_breakout_trades = []
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("VOLATILITY BREAKOUT STRATEGY BACKTEST RESULTS")
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
    logger.info("\nSignal Type Breakdown:")
    logger.info(f"BB Squeeze: {len(bb_squeeze_trades)} trades, {bb_win_rate:.1f}% win rate")
    logger.info(f"Volatility Expansion: {len(vol_expansion_trades)} trades, {vol_win_rate:.1f}% win rate")
    logger.info(f"Range Breakout: {len(range_breakout_trades)} trades, {range_win_rate:.1f}% win rate")
    
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
            'volatility_ratio': t.volatility_ratio,
            'max_profit_pct': t.max_profit_pct,
            'max_loss_pct': t.max_loss_pct,
            'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600
        } for t in trades])
        
        filename = f"volatility_breakout_backtest_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"\nTrades exported to: {filename}")
    
    return {
        'strategy': 'Volatility Breakout',
        'period': f"{start_date} to {end_date}",
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_profit': total_profit,
        'return_pct': (balance/initial_balance - 1)*100,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'bb_squeeze_trades': len(bb_squeeze_trades),
        'vol_expansion_trades': len(vol_expansion_trades),
        'range_breakout_trades': len(range_breakout_trades)
    }

if __name__ == "__main__":
    backtest_strategy()