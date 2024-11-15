from typing import List, Dict, Optional
from .base import BaseTechnicalAnalysis, SignalType, SignalResult
import pandas as pd
from dataclasses import dataclass
import logging

@dataclass
class BacktestResults:
    trades_history: List[Dict]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

class Backtester:
    def __init__(
        self,
        strategy_class,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ):
        self.strategy = strategy_class()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.current_position = None
        self.trades_history = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def process_candle(self, candle: Dict, historical_candles: List[Dict]) -> None:
        """Process a single candle and execute trades based on signals."""
        try:
            # Get signal from strategy
            signal_result = self.strategy.analyze(historical_candles)
            
            # Skip if no signal or invalid signal
            if signal_result is None:
                return
                
            current_price = float(candle['close'])
            
            # Handle existing position
            if self.current_position:
                self._manage_existing_position(candle, signal_result)
                
            # Handle new position entry
            elif signal_result.signal_type in [SignalType.LONG, SignalType.SHORT]:
                self._enter_position(candle, signal_result)
                
        except Exception as e:
            self.logger.error(f"Error processing candle: {str(e)}")

    def _enter_position(self, candle: Dict, signal: SignalResult) -> None:
        """Enter a new position based on signal."""
        current_price = float(candle['close'])
        position_size = self.current_capital * self.position_size
        
        stop_loss = current_price * (1 - self.stop_loss_pct) if signal.signal_type == SignalType.LONG else current_price * (1 + self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct) if signal.signal_type == SignalType.LONG else current_price * (1 - self.take_profit_pct)
        
        # Convert timestamp to milliseconds if it's in seconds
        timestamp = float(candle['timestamp'])
        if timestamp < 1e12:  # If timestamp is in seconds
            timestamp *= 1000
        
        self.current_position = {
            'entry_time': timestamp,
            'entry_price': current_price,
            'position_type': signal.signal_type,
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market_regime': signal.market_regime
        }

    def _manage_existing_position(self, candle: Dict, signal: SignalResult) -> None:
        """Manage existing position including exits."""
        current_price = float(candle['close'])
        position = self.current_position
        
        # Check for exit conditions
        exit_price = None
        exit_type = None
        
        if position['position_type'] == SignalType.LONG:
            if current_price <= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_type = 'stop_loss'
            elif current_price >= position['take_profit']:
                exit_price = position['take_profit']
                exit_type = 'take_profit'
            elif signal.signal_type == SignalType.SHORT:
                exit_price = current_price
                exit_type = 'signal_reversal'
                
        elif position['position_type'] == SignalType.SHORT:
            if current_price >= position['stop_loss']:
                exit_price = position['stop_loss']
                exit_type = 'stop_loss'
            elif current_price <= position['take_profit']:
                exit_price = position['take_profit']
                exit_type = 'take_profit'
            elif signal.signal_type == SignalType.LONG:
                exit_price = current_price
                exit_type = 'signal_reversal'
        
        # Execute exit if conditions met
        if exit_price:
            self._exit_position(candle['timestamp'], exit_price, exit_type)

    def _exit_position(self, exit_time: float, exit_price: float, exit_type: str) -> None:
        """Exit current position and record trade."""
        position = self.current_position
        
        # Calculate P&L
        if position['position_type'] == SignalType.LONG:
            pnl = (exit_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) / position['entry_price']
            
        pnl_amount = position['size'] * pnl
        
        # Update capital
        self.current_capital += pnl_amount
        
        # Convert exit timestamp to milliseconds if it's in seconds
        if exit_time < 1e12:  # If timestamp is in seconds
            exit_time *= 1000
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'position_type': position['position_type'].value,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'size': position['size'],
            'pnl': pnl_amount,
            'pnl_pct': pnl,
            'exit_type': exit_type,
            'market_regime': position['market_regime'].value
        }
        self.trades_history.append(trade)
        
        # Clear current position
        self.current_position = None

    def run(self, candles: List[Dict]) -> BacktestResults:
        """Run backtest on historical candles."""
        self.current_capital = self.initial_capital
        self.current_position = None
        self.trades_history = []
        
        for i in range(self.strategy.required_history, len(candles)):
            historical_candles = candles[i-self.strategy.required_history:i+1]
            self.process_candle(candles[i], historical_candles)
            
        # Calculate final metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        if not self.trades_history:
            return BacktestResults(
                trades_history=[],
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0
            )
            
        # Calculate other metrics
        returns = pd.Series([trade['pnl_pct'] for trade in self.trades_history])
        sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calculate win rate
        winning_trades = len([t for t in self.trades_history if t['pnl'] > 0])
        win_rate = winning_trades / len(self.trades_history)
        
        # Calculate profit factor
        gross_profits = sum([t['pnl'] for t in self.trades_history if t['pnl'] > 0])
        gross_losses = abs(sum([t['pnl'] for t in self.trades_history if t['pnl'] < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return BacktestResults(
            trades_history=self.trades_history,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor
        )

    def simulate_unresolved_trades(self, candles: List[Dict]) -> List[Dict]:
        """Simulate potential trades based on current market conditions."""
        opportunities = []
        
        try:
            latest_candles = candles[-self.strategy.required_history:]
            signal = self.strategy.analyze(latest_candles)
            
            if signal and signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                current_price = float(latest_candles[-1]['close'])
                
                # Convert timestamp to milliseconds if needed
                timestamp = float(latest_candles[-1]['timestamp'])
                if timestamp < 1e12:  # If timestamp is in seconds
                    timestamp *= 1000
                
                stop_loss = current_price * (1 - self.stop_loss_pct) if signal.signal_type == SignalType.LONG else current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct) if signal.signal_type == SignalType.LONG else current_price * (1 - self.take_profit_pct)
                
                risk = abs(current_price - stop_loss)
                reward = abs(take_profit - current_price)
                risk_reward_ratio = reward / risk if risk != 0 else 0
                
                opportunity = {
                    'timestamp': pd.Timestamp.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M'),
                    'signal_type': signal.signal_type.value,
                    'current_price': current_price,
                    'potential_entry': current_price,
                    'potential_stop_loss': stop_loss,
                    'potential_take_profit': take_profit,
                    'potential_position_size': self.current_capital * self.position_size,
                    'confidence': signal.confidence,
                    'risk_reward_ratio': risk_reward_ratio,
                    'market_regime': signal.market_regime.value,
                    'indicators': signal.indicators
                }
                
                opportunities.append(opportunity)
        
        except Exception as e:
            self.logger.error(f"Error simulating trades: {str(e)}")
        
        return opportunities

    def get_market_status(self, current_candle: Dict) -> Dict:
        """Get current market status including position information."""
        if not self.current_position:
            return {'in_position': False}
            
        current_price = float(current_candle['close'])
        position = self.current_position
        
        if position['position_type'] == SignalType.LONG:
            unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price']
            
        return {
            'in_position': True,
            'position_type': position['position_type'].value,
            'entry_price': position['entry_price'],
            'current_price': current_price,
            'unrealized_pnl': unrealized_pnl,
            'stop_loss': position['stop_loss'],
            'take_profit': position['take_profit']
        }

    def get_current_position(self) -> Optional[Dict]:
        """Get current position information."""
        return self.current_position

    def modify_position(self, new_stop_loss: Optional[float] = None, reduce_size_pct: Optional[float] = None) -> None:
        """Modify current position parameters."""
        if not self.current_position:
            return
            
        if new_stop_loss is not None:
            self.current_position['stop_loss'] = new_stop_loss
            
        if reduce_size_pct is not None and 0 < reduce_size_pct <= 1:
            reduction_amount = self.current_position['size'] * reduce_size_pct
            self.current_position['size'] -= reduction_amount

    def get_latest_signal(self) -> Optional[SignalResult]:
        """Get the latest signal from the strategy."""
        return self.strategy.latest_signal if hasattr(self.strategy, 'latest_signal') else None