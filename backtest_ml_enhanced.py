#!/usr/bin/env python3
"""
Machine Learning Enhanced Strategy Backtest
Uses XGBoost to predict profitable trading opportunities
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MLSignal:
    timestamp: datetime
    price: float
    prediction: float
    probability: float
    feature_importance: Dict[str, float]

@dataclass
class Trade:
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    profit: Optional[float]
    exit_reason: Optional[str]
    ml_probability: float
    prediction_score: float

class MLEnhancedStrategy:
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50],
                 target_return: float = 0.01, probability_threshold: float = 0.65):
        self.lookback_periods = lookback_periods
        self.target_return = target_return
        self.probability_threshold = probability_threshold
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for ML model"""
        df = df.copy()
        
        # Price-based features
        for period in self.lookback_periods:
            df[f'returns_{period}'] = df['close'].pct_change(period)
            df[f'high_low_ratio_{period}'] = (df['high'] / df['low']).rolling(period).mean()
            df[f'close_range_position_{period}'] = (
                (df['close'] - df['low'].rolling(period).min()) / 
                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
            )
        
        # Technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['rsi_slope'] = df['rsi'].diff(5)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
        df['macd_hist_slope'] = df['macd_hist'].diff(3)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR and volatility
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['atr_percent'] = df['atr'] / df['close']
        df['volatility_ratio'] = df['atr'] / df['atr'].rolling(20).mean()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_trend'] = df['volume_sma'].pct_change(10)
        
        # Moving averages
        for ma in [10, 20, 50, 200]:
            df[f'ema_{ma}'] = talib.EMA(df['close'], timeperiod=ma)
            df[f'distance_from_ema_{ma}'] = (df['close'] - df[f'ema_{ma}']) / df[f'ema_{ma}']
        
        # Market microstructure
        df['spread'] = df['high'] - df['low']
        df['spread_ratio'] = df['spread'] / df['close']
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Momentum indicators
        df['roc'] = talib.ROC(df['close'], timeperiod=10)
        df['mom'] = talib.MOM(df['close'], timeperiod=10)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        
        # Pattern recognition features
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_lows'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['trend_strength'] = df['higher_highs'].rolling(10).sum() + df['higher_lows'].rolling(10).sum()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_asian_session'] = (df['hour'] >= 0) & (df['hour'] < 8)
        df['is_european_session'] = (df['hour'] >= 8) & (df['hour'] < 16)
        df['is_us_session'] = (df['hour'] >= 16) & (df['hour'] < 24)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training"""
        df = self.calculate_features(df)
        
        # Create target variable: 1 if price increases by target_return in next N bars
        lookahead_bars = 20  # Look 20 bars ahead (100 minutes)
        df['future_return'] = df['close'].shift(-lookahead_bars) / df['close'] - 1
        df['target'] = (df['future_return'] > self.target_return).astype(int)
        
        # Select features
        feature_cols = [col for col in df.columns if col not in 
                       ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return',
                        'start', 'timestamp']]
        
        # Remove rows with NaN
        df_clean = df.dropna()
        
        return df_clean[feature_cols], df_clean['target']
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train XGBoost model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X_train.columns.tolist()
        
        # Log feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("\nTop 10 Most Important Features:")
        for feature, importance in top_features:
            logger.info(f"{feature}: {importance:.4f}")
    
    def generate_signals(self, df: pd.DataFrame) -> List[MLSignal]:
        """Generate ML-based trading signals"""
        if self.model is None:
            raise ValueError("Model must be trained before generating signals")
        
        df_features = self.calculate_features(df)
        
        # Select same features used in training
        feature_cols = self.feature_names
        df_features = df_features[feature_cols].dropna()
        
        # Scale features
        X_scaled = self.scaler.transform(df_features)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Generate signals where probability exceeds threshold
        signals = []
        for i, (idx, prob) in enumerate(zip(df_features.index, probabilities)):
            if prob > self.probability_threshold and predictions[i] == 1:
                # Get feature importance for this prediction
                feature_values = X_scaled[i]
                feature_importance = dict(zip(
                    self.feature_names[:5],  # Top 5 features
                    self.model.feature_importances_[:5]
                ))
                
                signals.append(MLSignal(
                    timestamp=idx,
                    price=df.loc[idx, 'close'],
                    prediction=predictions[i],
                    probability=prob,
                    feature_importance=feature_importance
                ))
        
        return signals

def backtest_strategy(product_id: str = 'BTC-USDC', 
                     start_date: str = '2021-01-01',
                     end_date: str = '2025-01-01',
                     initial_balance: float = 10000,
                     position_size_pct: float = 0.1,
                     stop_loss_pct: float = 0.02,
                     take_profit_pct: float = 0.03,
                     train_size: float = 0.6):
    """Run the ML enhanced strategy backtest"""
    
    logger.info(f"Starting ML Enhanced Backtest for {product_id}")
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
    strategy = MLEnhancedStrategy()
    
    # Prepare training data
    logger.info("Preparing training data...")
    X, y = strategy.prepare_training_data(df)
    
    # Split data into train and test
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train model
    logger.info("Training ML model...")
    strategy.train_model(X_train, y_train)
    
    # Evaluate model on test set
    X_test_scaled = strategy.scaler.transform(X_test)
    y_pred = strategy.model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    logger.info(f"\nModel Performance on Test Set:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    
    # Generate signals on test period only
    test_start_idx = df.index.get_loc(X_test.index[0])
    df_test = df.iloc[test_start_idx:]
    
    logger.info("\nGenerating trading signals...")
    signals = strategy.generate_signals(df_test)
    
    # Backtest execution
    balance = initial_balance
    trades = []
    current_trade = None
    
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
            ml_probability=signal.probability,
            prediction_score=signal.prediction
        )
        
        # Set stops
        stop_loss = signal.price * (1 - stop_loss_pct)
        take_profit = signal.price * (1 + take_profit_pct)
        
        # Find exit
        entry_idx = df_test.index.get_loc(signal.timestamp)
        
        for j in range(entry_idx + 1, min(entry_idx + 100, len(df_test))):  # Max 100 bars
            row = df_test.iloc[j]
            
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
                
            # Time-based exit after 20 bars
            elif j - entry_idx >= 20:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "Time Exit"
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
        
        # ML-specific metrics
        avg_ml_prob = sum(t.ml_probability for t in trades) / total_trades
        high_prob_trades = [t for t in trades if t.ml_probability > 0.7]
        high_prob_win_rate = (len([t for t in high_prob_trades if t.profit > 0]) / 
                             len(high_prob_trades) * 100) if high_prob_trades else 0
    else:
        win_rate = avg_profit = total_profit = profit_factor = avg_win = avg_loss = 0
        avg_ml_prob = high_prob_win_rate = 0
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("ML ENHANCED STRATEGY BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Period: {X_test.index[0]} to {end_date} (Test Period Only)")
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
    logger.info(f"\nML Metrics:")
    logger.info(f"Average ML Probability: {avg_ml_prob:.3f}")
    logger.info(f"High Probability (>0.7) Win Rate: {high_prob_win_rate:.1f}%")
    logger.info(f"Total Signals Generated: {len(signals)}")
    
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
            'ml_probability': t.ml_probability,
            'duration_hours': (t.exit_time - t.entry_time).total_seconds() / 3600
        } for t in trades])
        
        filename = f"ml_enhanced_backtest_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        logger.info(f"\nTrades exported to: {filename}")
    
    return {
        'strategy': 'ML Enhanced',
        'period': f"{X_test.index[0]} to {end_date}",
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_profit': total_profit,
        'return_pct': (balance/initial_balance - 1)*100,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'model_accuracy': accuracy,
        'model_precision': precision,
        'model_recall': recall
    }

if __name__ == "__main__":
    backtest_strategy()