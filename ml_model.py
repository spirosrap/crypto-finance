import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import logging
import joblib
from datetime import datetime, timedelta
import os
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.classes_ = None

    def fit(self, X, y):
        # Store unique class labels
        self.classes_ = np.unique(y)

        # Train base models
        self.fitted_base_models_ = []
        for model in self.base_models:
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_base_models_.append(fitted_model)

        # Generate meta-features
        meta_features = np.column_stack([
            cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
            for model in self.base_models
        ])

        # Train meta-model
        self.fitted_meta_model_ = clone(self.meta_model)
        self.fitted_meta_model_.fit(meta_features, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            model.predict_proba(X)[:, 1] for model in self.fitted_base_models_
        ])
        return self.fitted_meta_model_.predict_proba(meta_features)

    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5

    def get_params(self, deep=True):
        return {
            "base_models": self.base_models,
            "meta_model": self.meta_model
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def classes_(self):
        return self._classes

    @classes_.setter
    def classes_(self, value):
        self._classes = value

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
        class_counts = np.bincount(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1

        # Define base models
        base_models = [
            ('lr', LogisticRegression(random_state=42, class_weight='balanced', max_iter=3000)),
            ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
            ('xgb', XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight))
        ]

        # Define meta-model
        meta_model = LogisticRegression(random_state=42)

        # Create stacking ensemble
        stacking_ensemble = StackingEnsemble(base_models=[model for _, model in base_models], meta_model=meta_model)

        # Create a pipeline with preprocessing and stacking ensemble
        self.ml_model = Pipeline([
            ('preprocessor', preprocessor),
            ('stacking_ensemble', stacking_ensemble)
        ])

        # Simplified hyperparameter search space
        param_distributions = {
            'stacking_ensemble__base_models__0__C': [0.1, 1, 10],
            'stacking_ensemble__base_models__1__n_estimators': [100, 200],
            'stacking_ensemble__base_models__1__max_depth': [10, 20, None],
            'stacking_ensemble__base_models__2__n_estimators': [100, 200],
            'stacking_ensemble__base_models__2__learning_rate': [0.01, 0.1],
            'stacking_ensemble__meta_model__C': [0.1, 1, 10]
        }

        # Perform randomized search
        random_search = RandomizedSearchCV(
            self.ml_model, param_distributions, n_iter=10,
            cv=TimeSeriesSplit(n_splits=3), scoring='neg_log_loss',
            random_state=42, n_jobs=-1
        )
        random_search.fit(X_train_resampled, y_train_resampled)

        # Set the best model
        self.ml_model = random_search.best_estimator_

        self.logger.info(f"Best parameters: {random_search.best_params_}")
        self.logger.info(f"Best score: {-random_search.best_score_:.4f}")

        # Evaluate the model on training, validation, and test sets
        sets = [
            ("Training", X_train_resampled, y_train_resampled),
            ("Validation", X_val, y_val),
            ("Test", X_test, y_test)
        ]

        for set_name, X_set, y_set in sets:
            y_pred = self.ml_model.predict(X_set)
            y_pred_proba = self.ml_model.predict_proba(X_set)[:, 1]

            mse = mean_squared_error(y_set, y_pred_proba)
            mae = mean_absolute_error(y_set, y_pred_proba)
            r2 = r2_score(y_set, y_pred_proba)

            self.logger.info(f"{set_name} set - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

        # Save the trained model
        joblib.dump(self.ml_model, self.model_file)
        self.logger.info(f"ML model trained and saved to {self.model_file}")

    def evaluate_feature_importance(self, X_train, y_train, X_val, y_val):
        feature_names = ['rsi', 'macd', 'sma_short', 'sma_long', 'volume', 'returns', 'atr', 'bbw', 'roc', 'mfi', 'adx', 
                         'rsi_lag1', 'macd_lag1', 'trend_strength', 'ema_fast', 'ema_slow', 'cci', 'obv',
                         'stoch_k', 'stoch_d', 'willr', 'mom', 'log_return', 'volatility',
                         'ema_crossover', 'rsi_overbought', 'rsi_oversold', 'macd_signal', 'bbw_high']

        # Permutation importance
        perm_importance = permutation_importance(self.ml_model, X_val, y_val, n_repeats=10, random_state=42)

        # Sort features by importance
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': perm_importance.importances_mean})
        feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

        # Log feature importance
        self.logger.info("Feature Importance:")
        for idx, row in feature_importance.iterrows():
            self.logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Identify low importance features
        low_importance_threshold = 0.001  # Adjust this threshold as needed
        low_importance_features = feature_importance[feature_importance['importance'] < low_importance_threshold]['feature'].tolist()
        
        if low_importance_features:
            self.logger.info(f"Consider removing these low importance features: {', '.join(low_importance_features)}")
        else:
            self.logger.info("No low importance features identified.")

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
            
            X_processed = self.ml_model.named_steps['preprocessor'].transform(X)
            
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