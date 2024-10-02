import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import logging

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0).round().astype(int)

    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probas, axis=0)

class MLSignal:
    def __init__(self, logger):
        self.logger = logger
        self.ml_model = None

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

            # Create target variable (1 if price goes up, 0 if it goes down)
            df['target'] = (df['returns'].shift(-1) > 0).astype(int)
            
            # Drop the last row as it won't have a target
            df = df.iloc[:-1]
            
            # Drop rows with NaN values
            df.dropna(inplace=True)
            
            if df.empty:
                self.logger.warning("All rows removed after feature calculation and NaN removal.")
                return np.array([]), np.array([])
            
            features = ['rsi', 'macd', 'sma_short', 'sma_long', 'volume', 'returns', 'atr', 'bbw', 'roc', 'mfi', 'adx', 
                        'rsi_lag1', 'macd_lag1', 'trend_strength', 'ema_fast', 'ema_slow', 'cci', 'obv']
            
            X = df[features].values
            y = df['target'].values

            return X, y
        except Exception as e:
            self.logger.error(f"Error in prepare_features: {str(e)}")
            return np.array([]), np.array([])

    def train_model(self, candles: List[Dict]):
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

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Create a preprocessing pipeline
        preprocessor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Define base models with regularization
        models = {
            'lr': LogisticRegression(random_state=42, class_weight='balanced'),
            'rf': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced'),
            'xgb': XGBClassifier(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=5, 
                                 scale_pos_weight=class_weights[1]/class_weights[0] if len(class_weights) > 1 else 1)
        }

        # Hyperparameter search spaces
        param_distributions = {
            'lr': {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                'classifier__penalty': ['l1', 'l2'],
                'classifier__solver': ['liblinear', 'saga']
            },
            'rf': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 20],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            'xgb': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5],
                'classifier__min_child_weight': [1, 5, 10]
            }
        }

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        best_models = {}
        for name, model in models.items():
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            random_search = RandomizedSearchCV(
                pipeline, param_distributions[name], n_iter=10, 
                cv=tscv, scoring='f1', random_state=42, n_jobs=-1
            )
            random_search.fit(X_train, y_train)
            best_models[name] = random_search.best_estimator_

        # Create ensemble
        self.ml_model = EnsembleClassifier([model for model in best_models.values()])
        self.ml_model.fit(X_train, y_train)

        # Evaluate the model on the test set
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, self.ml_model.predict_proba(X_test)[:, 1])
        self.logger.info(f"ML Model test set - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

        # Evaluate on training set for comparison
        y_train_pred = self.ml_model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        self.logger.info(f"ML Model training set - Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

        # Log feature importances if available
        for model in self.ml_model.models:
            if hasattr(model.named_steps['classifier'], 'feature_importances_'):
                importances = model.named_steps['classifier'].feature_importances_
                feature_names = ['rsi', 'macd', 'sma_short', 'sma_long', 'volume', 'returns', 'atr', 'bbw', 'roc', 'mfi', 'adx', 
                                 'rsi_lag1', 'macd_lag1', 'trend_strength', 'ema_fast', 'ema_slow', 'cci', 'obv']
                for name, importance in zip(feature_names, importances):
                    self.logger.info(f"Feature importance - {name}: {importance}")

    def predict_signal(self, candles: List[Dict]) -> int:
        if self.ml_model is None:
            self.logger.info("Training ML model...")
            self.train_model(candles[:-1])  # Train on all but the last candle
        
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
            self.logger.error(f"Error in ML prediction: {str(e)}. Returning neutral signal.")
            return 0

    def evaluate_performance(self, candles: List[Dict]) -> float:
        X, y = self.prepare_features(candles[-100:])  # Use last 100 candles for evaluation
        if X.size == 0 or y.size == 0 or self.ml_model is None:
            return 1.0  # Default weight if we can't evaluate

        y_pred = self.ml_model.predict(X)
        recent_accuracy = accuracy_score(y, y_pred)
        return min(max(recent_accuracy, 0.5), 2.0)  # Scale between 0.5 and 2