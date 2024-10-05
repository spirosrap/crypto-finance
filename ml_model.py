import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import logging
import joblib
from datetime import datetime, timedelta
import os

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        self.weights = None

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        
        # Calculate weights based on individual model performance
        self.weights = []
        for model in self.models:
            y_pred = model.predict(X)
            score = f1_score(y, y_pred)
            self.weights.append(score)
        
        # Normalize weights
        self.weights = np.array(self.weights) / sum(self.weights)
        return self

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_predictions = np.average(predictions, axis=0, weights=self.weights)
        return (weighted_predictions > 0.5).astype(int)

    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)

class MLSignal:
    def __init__(self, logger, historical_data):
        self.logger = logger
        self.ml_model = None
        self.historical_data = historical_data
        self.model_file = 'ml_model.joblib'

    def prepare_features(self, candles: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        if len(candles) < 50:
            self.logger.warning(f"Not enough candles for ML features. Got {len(candles)}, need at least 50.")
            return np.array([]), np.array([])

        df = pd.DataFrame(candles)
        
        try:
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Calculate technical indicators
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['macd'], _, _ = talib.MACD(df['close'])
            df['sma_short'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_long'] = talib.SMA(df['close'], timeperiod=30)
            df['returns'] = df['close'].pct_change()
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['bbw'] = (talib.BBANDS(df['close'], timeperiod=20)[0] - talib.BBANDS(df['close'], timeperiod=20)[2]) / df['close']
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['rsi_lag1'] = df['rsi'].shift(1)
            df['macd_lag1'] = df['macd'].shift(1)
            df['trend_strength'] = np.where(df['adx'] > 25, 1, 0)
            df['ema_fast'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_slow'] = talib.EMA(df['close'], timeperiod=26)
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Add more features
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            df['willr'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['mom'] = talib.MOM(df['close'], timeperiod=10)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)

            # New features
            df['ema_crossover'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
            df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
            df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            df['macd_signal'] = np.where(df['macd'] > 0, 1, -1)
            df['bbw_high'] = np.where(df['bbw'] > df['bbw'].rolling(window=20).mean(), 1, 0)

            # Create target variable (1 if price goes up, 0 if it goes down)
            df['target'] = (df['returns'].shift(-1) > 0).astype(int)
            
            # Handle missing values
            # First, forward fill
            df = df.ffill()
            
            # Then, backward fill any remaining NaNs at the beginning
            df = df.bfill()
            
            # If there are still NaNs, drop those rows
            df = df.dropna()
            
            if df.empty:
                self.logger.warning("All rows removed after feature calculation and NaN removal.")
                return np.array([]), np.array([])
            
            features = ['rsi', 'macd', 'sma_short', 'sma_long', 'volume', 'returns', 'atr', 'bbw', 'roc', 'mfi', 'adx', 
                        'rsi_lag1', 'macd_lag1', 'trend_strength', 'ema_fast', 'ema_slow', 'cci', 'obv',
                        'stoch_k', 'stoch_d', 'willr', 'mom', 'log_return', 'volatility',
                        'ema_crossover', 'rsi_overbought', 'rsi_oversold', 'macd_signal', 'bbw_high']
            
            X = df[features].values
            y = df['target'].values

            return X, y
        except Exception as e:
            self.logger.error(f"Error in prepare_features: {str(e)}")
            return np.array([]), np.array([])

    def train_model(self):
        # Get 4 years of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*4)
        candles = self.historical_data.get_historical_data('BTC-USD', start_date, end_date, granularity="ONE_HOUR")
        
        X, y = self.prepare_features(candles)
        
        if X.size == 0 or y.size == 0:
            self.logger.warning("Not enough data to train ML model.")
            return

        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        self.logger.info(f"Class distribution: {class_distribution}")

        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        # Split the data into training, validation, and testing sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Create a preprocessing pipeline
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define base models with regularization
        class_counts = np.bincount(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1

        models = {
            'lr': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'rf': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'xgb': XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
        }

        # Simplified hyperparameter search spaces
        param_distributions = {
            'lr': {
                'classifier__C': [0.1, 1, 10],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear', 'saga'],
                'classifier__max_iter': [1000, 2000]
            },
            'rf': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            },
            'xgb': {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5],
                'classifier__min_child_weight': [1, 5],
                'classifier__subsample': [0.8, 1.0],
                'classifier__colsample_bytree': [0.8, 1.0]
            }
        }

        best_models = {}
        for name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline, param_distributions[name], n_iter=10,
                cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1
            )
            random_search.fit(X_train_resampled, y_train_resampled)
            best_models[name] = random_search.best_estimator_

        # Create ensemble with weighted voting
        self.ml_model = EnsembleClassifier([model for model in best_models.values()])
        self.ml_model.fit(X_train_resampled, y_train_resampled)

        # Evaluate the model on training, validation, and test sets
        sets = [
            ("Training", X_train_resampled, y_train_resampled),
            ("Validation", X_val, y_val),
            ("Test", X_test, y_test)
        ]

        for set_name, X_set, y_set in sets:
            y_pred = self.ml_model.predict(X_set)
            y_pred_proba = self.ml_model.predict_proba(X_set)[:, 1]  # Probability of positive class

            mse = mean_squared_error(y_set, y_pred_proba)
            mae = mean_absolute_error(y_set, y_pred_proba)
            r2 = r2_score(y_set, y_pred_proba)

            self.logger.info(f"{set_name} set - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Log model weights
        for model, weight in zip(self.ml_model.models, self.ml_model.weights):
            self.logger.info(f"Model {type(model).__name__} weight: {weight:.4f}")

        # Save the trained model
        joblib.dump(self.ml_model, self.model_file)
        self.logger.info(f"ML model trained and saved to {self.model_file}")

    def load_model(self):
        try:
            self.ml_model = joblib.load(self.model_file)
            model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.model_file))
            if model_age > timedelta(days=7):  # Retrain weekly
                self.logger.info("Model is over a week old. Retraining...")
                self.train_model()
            else:
                self.logger.info(f"ML model loaded from {self.model_file}")
        except FileNotFoundError:
            self.logger.warning(f"ML model file not found. Training a new model.")
            self.train_model()

    def predict_signal(self, candles: List[Dict]) -> int:
        if self.ml_model is None:
            self.load_model()
        
        X, _ = self.prepare_features(candles[-50:])  # Use the last 50 candles for prediction
        
        if X.size == 0:
            self.logger.warning("Not enough data to make ML prediction. Returning neutral signal.")
            return 0
        
        try:
            self.logger.debug(f"X shape: {X.shape}")
            
            X_processed = self.ml_model.models[0].named_steps['preprocessor'].transform(X)
            
            probability = self.ml_model.predict_proba(X_processed)
            
            # Use the last prediction (most recent)
            last_probability = probability[-1]
            self.logger.debug(f"Last ML Prediction probability: {last_probability}")
            
            # Adjust scaling to make signal more pronounced and ensure it can be negative
            signal = int((last_probability[1] - 0.5) * 20)  # Scale from -10 to 10
            
            self.logger.debug(f"ML signal: {signal}")
            return signal
        except Exception as e:
            self.logger.exception(f"Error in ML prediction: {str(e)}. Returning neutral signal.")
            raise  # Re-raise the exception for debugging
            return 0

    def evaluate_performance(self, candles: List[Dict]) -> float:
        X, y = self.prepare_features(candles[-100:])  # Use last 100 candles for evaluation
        if X.size == 0 or y.size == 0 or self.ml_model is None:
            return 1.0  # Default weight if we can't evaluate

        y_pred = self.ml_model.predict(X)
        recent_accuracy = accuracy_score(y, y_pred)
        return min(max(recent_accuracy, 0.5), 2.0)  # Scale between 0.5 and 2