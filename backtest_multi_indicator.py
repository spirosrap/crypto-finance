#!/usr/bin/env python3
"""
Multi-Indicator Confluence Strategy Backtest
Combines multiple technical indicators for high-probability trading signals
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
class ConfluenceSignal:
    timestamp: datetime
    price: float
    confluence_score: float
    indicator_signals: Dict[str, bool]
    signal_strength: float
    market_condition: str

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    profit: Optional[float]
    exit_reason: Optional[str]
    confluence_score: float
    active_indicators: int
    market_condition: str

class MultiIndicatorStrategy:
    def __init__(self, min_confluence_score: float = 0.6,
                 rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.min_confluence_score = min_confluence_score
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.indicator_weights = {
            'rsi': 0.15,
            'macd': 0.15,
            'bb': 0.15,
            'stoch': 0.10,
            'volume': 0.15,
            'trend': 0.15,
            'momentum': 0.15
        }
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = df.copy()
        
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
        
        # Moving Averages
        df['ema_9'] = talib.EMA(df['close'], timeperiod=9)
        df['ema_21'] = talib.EMA(df['close'], timeperiod=21)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_ema'] = talib.EMA(df['obv'], timeperiod=20)
        
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Momentum indicators
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # ADX for trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Support/Resistance
        df['resistance'] = df['high'].rolling(window=20).max()
        df['support'] = df['low'].rolling(window=20).min()
        
        # Market structure
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_lows'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['trend_score'] = df['higher_highs'].rolling(10).sum() + df['higher_lows'].rolling(10).sum()
        
        return df
    
    def check_indicator_signals(self, row: pd.Series, prev_row: pd.Series) -> Dict[str, bool]:
        """Check individual indicator signals"""
        signals = {}
        
        # RSI signal (oversold bounce)
        signals['rsi'] = (row['rsi'] < self.rsi_oversold and 
                         row['rsi'] > prev_row['rsi'])  # RSI turning up from oversold
        
        # MACD signal (bullish crossover)
        signals['macd'] = (row['macd'] > row['macd_signal'] and 
                          prev_row['macd'] <= prev_row['macd_signal'])
        
        # Bollinger Band signal (bounce from lower band)
        signals['bb'] = (row['close'] > row['bb_lower'] and 
                        prev_row['close'] <= prev_row['bb_lower'] and
                        row['bb_position'] < 0.3)
        
        # Stochastic signal (oversold bounce)
        signals['stoch'] = (row['stoch_k'] < 20 and row['stoch_k'] > row['stoch_d'] and
                           row['stoch_k'] > prev_row['stoch_k'])
        
        # Volume signal (volume expansion)
        signals['volume'] = (row['volume_ratio'] > 1.5 and 
                           row['obv'] > row['obv_ema'])
        
        # Trend signal (price above key MAs)
        signals['trend'] = (row['close'] > row['ema_21'] and 
                          row['ema_9'] > row['ema_21'] and
                          row['close'] > row['ema_200'])
        
        # Momentum signal (positive momentum)
        signals['momentum'] = (row['roc'] > 0 and row['cci'] > -100 and
                             row['williams_r'] > -80)
        
        return signals
    
    def calculate_confluence_score(self, signals: Dict[str, bool]) -> float:
        """Calculate weighted confluence score"""
        score = 0.0
        for indicator, signal in signals.items():
            if signal:
                score += self.indicator_weights.get(indicator, 0)
        return score
    
    def identify_market_condition(self, row: pd.Series) -> str:
        """Identify current market condition"""
        if row['adx'] > 25:
            if row['plus_di'] > row['minus_di']:
                return "STRONG_UPTREND"
            else:
                return "STRONG_DOWNTREND"
        elif row['adx'] < 20:
            return "RANGING"
        else:
            if row['trend_score'] > 10:
                return "WEAK_UPTREND"
            elif row['trend_score'] < 5:
                return "WEAK_DOWNTREND"
            else:
                return "CHOPPY"
    
    def generate_signals(self, df: pd.DataFrame) -> List[ConfluenceSignal]:
        """Generate confluence-based trading signals"""
        df = self.calculate_indicators(df)
        signals = []
        
        for i in range(200, len(df)):  # Start after all indicators are calculated
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check all indicator signals
            indicator_signals = self.check_indicator_signals(row, prev_row)
            
            # Calculate confluence score
            confluence_score = self.calculate_confluence_score(indicator_signals)
            
            # Identify market condition
            market_condition = self.identify_market_condition(row)
            
            # Generate signal if confluence is high enough
            if confluence_score >= self.min_confluence_score:
                # Additional filters based on market condition
                if market_condition in ["STRONG_DOWNTREND", "RANGING"]:
                    continue  # Skip signals in unfavorable conditions
                
                signal_strength = self._calculate_signal_strength(
                    row, indicator_signals, market_condition)
                
                signals.append(ConfluenceSignal(
                    timestamp=row.name,
                    price=row['close'],
                    confluence_score=confluence_score,
                    indicator_signals=indicator_signals,
                    signal_strength=signal_strength,
                    market_condition=market_condition
                ))
        
        return signals
    
    def _calculate_signal_strength(self, row: pd.Series, 
                                 indicator_signals: Dict[str, bool],
                                 market_condition: str) -> float:
        """Calculate overall signal strength"""
        strength = 0.0
        
        # Base strength from confluence
        active_indicators = sum(indicator_signals.values())
        strength += (active_indicators / len(indicator_signals)) * 0.4
        
        # Market condition bonus
        if market_condition in ["STRONG_UPTREND", "WEAK_UPTREND"]:
            strength += 0.2
        
        # Volatility factor
        if 0.3 < row['atr_percent'] < 1.0:  # Moderate volatility is ideal
            strength += 0.2
        
        # Distance from support
        support_distance = (row['close'] - row['support']) / row['support']
        if support_distance < 0.02:  # Close to support
            strength += 0.1
        
        # Trend alignment
        if row['ema_9'] > row['ema_21'] > row['ema_50']:
            strength += 0.1
        
        return min(strength, 1.0)

def backtest_strategy(product_id: str = 'BTC-USDC', 
                     start_date: str = '2021-01-01',
                     end_date: str = '2025-01-01',
                     initial_balance: float = 10000,
                     position_size_pct: float = 0.12,
                     stop_loss_pct: float = 0.015,
                     take_profit_pct: float = 0.025):
    """Run the multi-indicator confluence strategy backtest"""
    
    logger.info(f"Starting Multi-Indicator Confluence Backtest for {product_id}")
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
    strategy = MultiIndicatorStrategy()
    
    # Generate signals
    logger.info("Generating confluence signals...")
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
        
        # Count active indicators
        active_indicators = sum(signal.indicator_signals.values())
        
        # Create trade
        current_trade = Trade(
            entry_time=signal.timestamp,
            exit_time=None,
            entry_price=signal.price,
            exit_price=None,
            position_size=position_size,
            profit=None,
            exit_reason=None,
            confluence_score=signal.confluence_score,
            active_indicators=active_indicators,
            market_condition=signal.market_condition
        )
        
        # Dynamic stops based on ATR and confluence
        entry_idx = df_with_indicators.index.get_loc(signal.timestamp)
        atr = df_with_indicators.iloc[entry_idx]['atr']
        
        # Tighter stops for higher confluence
        stop_multiplier = 1.0 - (signal.confluence_score - 0.6) * 0.5
        take_profit_multiplier = 1.0 + (signal.confluence_score - 0.6) * 0.5
        
        stop_loss = signal.price - (atr * 1.5 * stop_multiplier)
        take_profit = signal.price + (atr * 2.5 * take_profit_multiplier)
        
        # Find exit
        for j in range(entry_idx + 1, len(df_with_indicators)):
            row = df_with_indicators.iloc[j]
            
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
                
            # Exit on signal deterioration
            # Recalculate indicators
            indicator_signals = strategy.check_indicator_signals(row, df_with_indicators.iloc[j-1])
            current_confluence = strategy.calculate_confluence_score(indicator_signals)
            
            if current_confluence < 0.3:  # Exit if confluence drops significantly
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "Signal Deterioration"
                current_trade.profit = (row['close'] - current_trade.entry_price) * current_trade.position_size
                balance += current_trade.profit
                trades.append(current_trade)
                current_trade = None
                break
                
            # Trailing stop after profit
            current_profit_pct = (row['close'] - current_trade.entry_price) / current_trade.entry_price
            if current_profit_pct > 0.01:  # 1% profit
                new_stop = current_trade.entry_price + (current_trade.entry_price * 0.002)  # Trail to 0.2% profit
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
        
        # Confluence analysis
        high_confluence_trades = [t for t in trades if t.confluence_score > 0.7]
        high_confluence_win_rate = (len([t for t in high_confluence_trades if t.profit > 0]) / 
                                   len(high_confluence_trades) * 100) if high_confluence_trades else 0
        
        # Market condition analysis
        trend_trades = [t for t in trades if 'TREND' in t.market_condition]
        trend_win_rate = (len([t for t in trend_trades if t.profit > 0]) / 
                         len(trend_trades) * 100) if trend_trades else 0
    else:
        win_rate = avg_profit = total_profit = profit_factor = avg_win = avg_loss = 0
        high_confluence_win_rate = trend_win_rate = 0
        high_confluence_trades = []
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("MULTI-INDICATOR CONFLUENCE STRATEGY BACKTEST RESULTS")
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
    logger.info(f"\nConfluence Analysis:")
    logger.info(f"High Confluence (>0.7) Trades: {len(high_confluence_trades)}")
    logger.info(f"High Confluence Win Rate: {high_confluence_win_rate:.1f}%")
    logger.info(f"Trending Market Win Rate: {trend_win_rate:.1f}%")
    
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
            'confluence_score': t.confluence_score,
            'active_indicators': t.active_indicators,
            'market_condition': t.market_condition,
            'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600
        } for t in trades])
        
        filename = f"multi_indicator_backtest_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"\nTrades exported to: {filename}")
    
    return {
        'strategy': 'Multi-Indicator Confluence',
        'period': f"{start_date} to {end_date}",
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_profit': total_profit,
        'return_pct': (balance/initial_balance - 1)*100,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'high_confluence_trades': len(high_confluence_trades),
        'high_confluence_win_rate': high_confluence_win_rate
    }

if __name__ == "__main__":
    backtest_strategy()