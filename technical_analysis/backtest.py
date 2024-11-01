from dataclasses import dataclass
from typing import List, Dict, Type, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from .base import BaseTechnicalAnalysis, SignalType, SignalResult
import logging

@dataclass
class TradePosition:
    entry_price: float
    entry_time: datetime
    position_size: float
    stop_loss: float
    take_profit: float
    position_type: str  # 'long' or 'short'

@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_profit_per_trade: float
    avg_loss_per_trade: float
    trades_history: List[Dict]

class Backtester:
    def __init__(
        self,
        strategy_class: Type[BaseTechnicalAnalysis],
        initial_capital: float = 10000.0,
        position_size: float = 0.1,  # 10% of initial capital
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,  # 4% take profit
        maker_fee: float = 0.001,  # 0.1% maker fee
        taker_fee: float = 0.002,  # 0.2% taker fee
    ):
        self.strategy = strategy_class()
        self.initial_capital = initial_capital
        self.base_position_size = initial_capital * position_size  # Store fixed position size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.logger = logging.getLogger(__name__)
        
        # Runtime variables
        self.capital = initial_capital
        self.current_position: Optional[TradePosition] = None
        self.trades_history: List[Dict] = []
        
    def run(self, candles: List[Dict]) -> BacktestResult:
        """Run backtest on historical data."""
        self.capital = self.initial_capital
        self.current_position = None
        self.trades_history = []
        self.latest_signal = None
        
        for i in range(len(candles) - 1):
            # Get subset of candles up to current point
            current_candles = candles[:i+1]
            next_candle = candles[i+1]
            
            # Skip if not enough data for analysis
            if len(current_candles) < 50:  # Minimum required candles
                continue
                
            try:
                # Get signal from strategy
                signal = self.strategy.analyze(current_candles)
                self.latest_signal = signal
                
                # Print current trade status if in a position
                if self.current_position:
                    self._print_trade_update(current_candles[-1])
                    
                    # Print market status
                    market_status = self.get_market_status(current_candles[-1])
                    print("\nMarket Status:")
                    print(f"Risk/Reward Ratio: {market_status['risk_reward_ratio']:.2f}")
                    print(f"Distance to Stop Loss: {market_status['distance_to_stop_loss']*100:.2f}%")
                    print(f"Distance to Take Profit: {market_status['distance_to_take_profit']*100:.2f}%")
                
                # Process existing position if any
                if self.current_position:
                    self._check_position_exit(next_candle)
                
                # Process new signal if no position
                if not self.current_position:
                    self._process_signal(signal, next_candle)
                    
            except Exception as e:
                self.logger.error(f"Error processing candle: {str(e)}")
                
        return self._calculate_results()
    
    def _process_signal(self, signal: SignalResult, next_candle: Dict):
        """Process trading signal."""
        # More lenient signal thresholds
        if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY] and signal.confidence > 0.3:  # Changed from 0.5
            self._open_long_position(next_candle)
        elif signal.signal_type in [SignalType.STRONG_SELL, SignalType.SELL] and signal.confidence > 0.3:  # Changed from 0.5
            self._open_short_position(next_candle)
    
    def _open_long_position(self, candle: Dict):
        """Open a long position."""
        entry_price = float(candle['open'])
        position_size = self.base_position_size  # Use fixed position size
        stop_loss = entry_price * (1 - self.stop_loss_pct)
        take_profit = entry_price * (1 + self.take_profit_pct)
        
        # Convert Unix timestamp to datetime
        entry_time = datetime.fromtimestamp(candle['timestamp'])
        
        self.current_position = TradePosition(
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_type='long'
        )
        
        # Apply entry fee
        self.capital -= position_size * self.taker_fee
    
    def _open_short_position(self, candle: Dict):
        """Open a short position."""
        entry_price = float(candle['open'])
        position_size = self.base_position_size  # Use fixed position size
        stop_loss = entry_price * (1 + self.stop_loss_pct)
        take_profit = entry_price * (1 - self.take_profit_pct)
        
        # Convert Unix timestamp to datetime
        entry_time = datetime.fromtimestamp(candle['timestamp'])
        
        self.current_position = TradePosition(
            entry_price=entry_price,
            entry_time=entry_time,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_type='short'
        )
        
        # Apply entry fee
        self.capital -= position_size * self.taker_fee
    
    def _check_position_exit(self, candle: Dict):
        """Check if position should be closed."""
        if not self.current_position:
            return
            
        high = float(candle['high'])
        low = float(candle['low'])
        close = float(candle['close'])
        
        exit_price = None
        exit_type = None
        
        if self.current_position.position_type == 'long':
            if low <= self.current_position.stop_loss:
                exit_price = self.current_position.stop_loss
                exit_type = 'stop_loss'
            elif high >= self.current_position.take_profit:
                exit_price = self.current_position.take_profit
                exit_type = 'take_profit'
        else:  # short position
            if high >= self.current_position.stop_loss:
                exit_price = self.current_position.stop_loss
                exit_type = 'stop_loss'
            elif low <= self.current_position.take_profit:
                exit_price = self.current_position.take_profit
                exit_type = 'take_profit'
        
        if exit_price:
            # Convert Unix timestamp to datetime
            exit_time = datetime.fromtimestamp(candle['timestamp'])
            self._close_position(exit_price, exit_time, exit_type)
    
    def _close_position(self, exit_price: float, exit_time: datetime, exit_type: str):
        """Close current position and record trade."""
        if not self.current_position:
            return
            
        # Calculate profit/loss
        if self.current_position.position_type == 'long':
            pnl = (exit_price - self.current_position.entry_price) / self.current_position.entry_price
        else:
            pnl = (self.current_position.entry_price - exit_price) / self.current_position.entry_price
            
        # Apply fees
        pnl -= (self.maker_fee + self.taker_fee)
        
        # Update capital
        trade_pnl = self.current_position.position_size * pnl
        self.capital += self.current_position.position_size + trade_pnl
        
        # Print trade completion
        print("\nTrade Closed:")
        print(f"Entry Time: {self.current_position.entry_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Exit Time: {exit_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Position: {self.current_position.position_type.upper()}")
        print(f"Entry Price: ${self.current_position.entry_price:.2f}")
        print(f"Exit Price: ${exit_price:.2f}")
        print(f"P&L: {pnl*100:.2f}% (${trade_pnl:.2f})")
        print(f"Exit Type: {exit_type}")
        print(f"Current Capital: ${self.capital:.2f}")
        print("-" * 50)
        
        # Record trade
        self.trades_history.append({
            'entry_time': self.current_position.entry_time,
            'exit_time': exit_time,
            'entry_price': self.current_position.entry_price,
            'exit_price': exit_price,
            'position_type': self.current_position.position_type,
            'position_size': self.current_position.position_size,
            'pnl': trade_pnl,
            'pnl_pct': pnl,
            'exit_type': exit_type
        })
        
        self.current_position = None
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if not self.trades_history:
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_profit_per_trade=0.0,
                avg_loss_per_trade=0.0,
                trades_history=[]
            )
        
        # Calculate metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = pd.Series([trade['pnl_pct'] for trade in self.trades_history])
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
        
        # Calculate trade statistics
        winning_trades = len([t for t in self.trades_history if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades_history if t['pnl'] <= 0])
        total_trades = len(self.trades_history)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [t['pnl'] for t in self.trades_history if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trades_history if t['pnl'] <= 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = sum(profits) / sum(losses) if losses else float('inf')
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_profit_per_trade=avg_profit,
            avg_loss_per_trade=avg_loss,
            trades_history=self.trades_history
        ) 

    def _print_trade_update(self, candle: Dict):
        """Print current trade status."""
        if self.current_position:
            current_price = float(candle['close'])
            unrealized_pnl = 0
            
            if self.current_position.position_type == 'long':
                unrealized_pnl = (current_price - self.current_position.entry_price) / self.current_position.entry_price
            else:  # short
                unrealized_pnl = (self.current_position.entry_price - current_price) / self.current_position.entry_price
                
            print("\nCurrent Trade Status:")
            print(f"Time: {datetime.fromtimestamp(candle['timestamp']).strftime('%Y-%m-%d %H:%M')}")
            print(f"Position: {self.current_position.position_type.upper()}")
            print(f"Entry Price: ${self.current_position.entry_price:.2f}")
            print(f"Current Price: ${current_price:.2f}")
            print(f"Stop Loss: ${self.current_position.stop_loss:.2f}")
            print(f"Take Profit: ${self.current_position.take_profit:.2f}")
            print(f"Unrealized P&L: {unrealized_pnl*100:.2f}%")
            print(f"Position Size: ${self.current_position.position_size:.2f}")

    def get_current_position(self) -> Optional[Dict]:
        """Get information about the current open position."""
        if not self.current_position:
            return None
        
        return {
            'position_type': self.current_position.position_type,
            'entry_price': self.current_position.entry_price,
            'entry_time': self.current_position.entry_time,
            'position_size': self.current_position.position_size,
            'stop_loss': self.current_position.stop_loss,
            'take_profit': self.current_position.take_profit,
            'current_capital': self.capital
        }

    def get_latest_signal(self) -> Optional[SignalResult]:
        """Get the most recent trading signal."""
        return getattr(self, 'latest_signal', None)

    def get_market_status(self, current_candle: Dict) -> Dict:
        """Get current market status including indicators."""
        if not self.current_position:
            return None
        
        current_price = float(current_candle['close'])
        unrealized_pnl = 0
        
        if self.current_position.position_type == 'long':
            unrealized_pnl = (current_price - self.current_position.entry_price) / self.current_position.entry_price
        else:  # short
            unrealized_pnl = (self.current_position.entry_price - current_price) / self.current_position.entry_price
            
        distance_to_sl = abs(current_price - self.current_position.stop_loss) / current_price
        distance_to_tp = abs(current_price - self.current_position.take_profit) / current_price
        
        return {
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_amount': unrealized_pnl * self.current_position.position_size,
            'distance_to_stop_loss': distance_to_sl,
            'distance_to_take_profit': distance_to_tp,
            'risk_reward_ratio': distance_to_tp / distance_to_sl if distance_to_sl > 0 else 0
        }

    def modify_position(self, 
                       new_stop_loss: Optional[float] = None, 
                       new_take_profit: Optional[float] = None,
                       reduce_size_pct: Optional[float] = None) -> bool:
        """Modify current position parameters."""
        if not self.current_position:
            return False
        
        if new_stop_loss is not None:
            self.current_position.stop_loss = new_stop_loss
        
        if new_take_profit is not None:
            self.current_position.take_profit = new_take_profit
        
        if reduce_size_pct is not None and 0 < reduce_size_pct < 1:
            # Partially close position
            reduction_amount = self.current_position.position_size * reduce_size_pct
            self.current_position.position_size -= reduction_amount
            self.capital += reduction_amount  # Add reduced portion back to capital
        
        return True

    def close_position(self, current_candle: Dict) -> Optional[Dict]:
        """Force close the current position at market price."""
        if not self.current_position:
            return None
        
        exit_price = float(current_candle['close'])
        exit_time = datetime.fromtimestamp(current_candle['timestamp'])
        
        # Record the trade with 'manual_close' exit type
        self._close_position(exit_price, exit_time, 'manual_close')
        
        return {
            'exit_price': exit_price,
            'exit_time': exit_time,
            'pnl': self.trades_history[-1]['pnl'] if self.trades_history else None
        }

    def get_open_opportunities(self, current_candles: List[Dict]) -> Dict:
        """Analyze current market conditions and return potential trade opportunities."""
        try:
            signal = self.strategy.analyze(current_candles)
            current_price = float(current_candles[-1]['close'])
            
            opportunity = {
                'timestamp': datetime.fromtimestamp(current_candles[-1]['timestamp']).strftime('%Y-%m-%d %H:%M'),
                'current_price': current_price,
                'signal_type': signal.signal_type.value,
                'confidence': signal.confidence,
                'market_regime': signal.market_regime.value,
                'indicators': signal.indicators,
                'potential_entry': current_price,
                'potential_stop_loss': current_price * (1 - self.stop_loss_pct) if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY] 
                                     else current_price * (1 + self.stop_loss_pct),
                'potential_take_profit': current_price * (1 + self.take_profit_pct) if signal.signal_type in [SignalType.STRONG_BUY, SignalType.BUY]
                                      else current_price * (1 - self.take_profit_pct),
                'potential_position_size': self.base_position_size,
                'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct
            }
            
            return opportunity
        except Exception as e:
            self.logger.error(f"Error getting trade opportunities: {str(e)}")
            return None

    def simulate_unresolved_trades(self, candles: List[Dict]) -> List[Dict]:
        """Simulate trades that would still be open at the end of the backtest period."""
        unresolved_trades = []
        
        for i in range(max(0, len(candles) - 50), len(candles)):
            current_candles = candles[:i+1]
            
            if len(current_candles) < 50:
                continue
                
            opportunity = self.get_open_opportunities(current_candles)
            if opportunity and opportunity['confidence'] > 0.3:  # Same threshold as _process_signal
                unresolved_trades.append(opportunity)
        
        return unresolved_trades