import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime, timedelta
import os
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance

# Add these constants near the top of the file
GRANULARITY_SETTINGS = {
    'ONE_MINUTE': {'training_days': 10, 'feature_window': 100},
    'FIVE_MINUTE': {'training_days': 60, 'feature_window': 200},
    'TEN_MINUTE': {'training_days': 75, 'feature_window': 250},
    'FIFTEEN_MINUTE': {'training_days': 90, 'feature_window': 300},
    'THIRTY_MINUTE': {'training_days': 180, 'feature_window': 400},
    'ONE_HOUR': {'training_days': 365*4, 'feature_window': 500},
    'SIX_HOUR': {'training_days': 730, 'feature_window': 1000},
    'ONE_DAY': {'training_days': 2000, 'feature_window': 2000},
}

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self._classes = None
        self.fitted_base_models_ = None
        self.fitted_meta_model_ = None

    def __sklearn_is_fitted__(self):
        """Return whether the model is fitted."""
        return (hasattr(self, 'fitted_base_models_') and 
                hasattr(self, 'fitted_meta_model_') and 
                self.fitted_base_models_ is not None and 
                self.fitted_meta_model_ is not None)

    def get_feature_names_out(self, feature_names_in=None):
        """Get output feature names."""
        return np.array([f'meta_feature_{i}' for i in range(len(self.base_models))])

    def fit(self, X, y):
        """Fit the stacking ensemble."""
        # Store unique class labels
        self._classes = np.unique(y)
        
        # Train base models
        self.fitted_base_models_ = []
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # Train each base model and get predictions
        for i, model in enumerate(self.base_models):
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_base_models_.append(fitted_model)
            meta_features[:, i] = fitted_model.predict_proba(X)[:, 1]

        # Train meta-model
        self.fitted_meta_model_ = clone(self.meta_model)
        self.fitted_meta_model_.fit(meta_features, y)
        
        return self

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        if not self.__sklearn_is_fitted__():
            raise RuntimeError("Model must be fitted before making predictions")
        
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models_)))
        for i, model in enumerate(self.fitted_base_models_):
            meta_features[:, i] = model.predict_proba(X)[:, 1]
            
        return self.fitted_meta_model_.predict_proba(meta_features)

    def predict(self, X):
        """Predict class labels for X."""
        probas = self.predict_proba(X)
        return np.where(probas[:, 1] > 0.5, 1, 0)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            "base_models": self.base_models,
            "meta_model": self.meta_model
        }
        if deep:
            base_params = {}
            for i, model in enumerate(self.base_models):
                model_params = model.get_params(deep=True)
                base_params.update({f'base_models__{i}__{key}': value 
                                  for key, value in model_params.items()})
            params.update(base_params)
            params.update({f'meta_model__{key}': value 
                          for key, value in self.meta_model.get_params(deep=True).items()})
        return params

    def set_params(self, **parameters):
        """Set the parameters of this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def classes_(self):
        """Get the classes seen during fit."""
        if self._classes is None:
            raise AttributeError("Model not fitted yet")
        return self._classes

    @classes_.setter
    def classes_(self, value):
        """Set the classes."""
        self._classes = value

class MLSignal:
    def __init__(self, logger, historical_data, product_id='BTC-USDC', granularity='ONE_HOUR'):
        self.logger = logger
        self.ml_model = None
        self.historical_data = historical_data
        self.product_id = product_id
        self.granularity = granularity
        self.settings = GRANULARITY_SETTINGS.get(granularity, GRANULARITY_SETTINGS['ONE_HOUR'])
        self.model_file = os.path.join('models', f'ml_model_{product_id.lower().replace("-", "_")}_{granularity.lower()}.joblib')

    def prepare_features(self, candles: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        if len(candles) < self.settings['feature_window']:
            self.logger.debug(f"Not enough candles for ML features. Got {len(candles)}, need at least {self.settings['feature_window']}.")
            return np.array([]), np.array([])

        # print("\nFirst candle structure:", candles[0])
        # print("Converting candles to DataFrame format...")
        
        try:
            # Convert the list of dictionaries to a list of lists for DataFrame creation
            candle_data = []
            for i, candle in enumerate(candles):
                candle_data.append([
                    float(candle['close']),
                    float(candle['high']),
                    float(candle['low']),
                    float(candle['volume']),
                    float(candle['open']),
                    candle['start']
                ])
            
            # print("Creating DataFrame...")
            df = pd.DataFrame(
                candle_data,
                columns=['close', 'high', 'low', 'volume', 'open', 'start']
            )
            
            # print("\nCalculating technical indicators...")
            # print("- Calculating RSI...")
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # print("- Calculating MACD...")
            df['macd'], _, _ = talib.MACD(df['close'])
            
            # print("- Calculating SMAs...")
            df['sma_short'] = talib.SMA(df['close'], timeperiod=10)
            df['sma_long'] = talib.SMA(df['close'], timeperiod=30)
            
            # print("- Calculating returns...")
            df['returns'] = df['close'].pct_change()
            
            # print("- Calculating ATR...")
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # print("- Calculating Bollinger Bands...")
            df['bbw'] = (talib.BBANDS(df['close'], timeperiod=20)[0] - talib.BBANDS(df['close'], timeperiod=20)[2]) / df['close']
            
            # print("- Calculating additional indicators...")
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
            
            # print("- Calculating stochastic oscillator...")
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])
            
            # print("- Calculating final indicators...")
            df['willr'] = talib.WILLR(df['high'], df['low'], df['close'])
            df['mom'] = talib.MOM(df['close'], timeperiod=10)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)

            # print("- Creating derivative features...")
            df['ema_crossover'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
            df['rsi_overbought'] = np.where(df['rsi'] > 70, 1, 0)
            df['rsi_oversold'] = np.where(df['rsi'] < 30, 1, 0)
            df['macd_signal'] = np.where(df['macd'] > 0, 1, -1)
            df['bbw_high'] = np.where(df['bbw'] > df['bbw'].rolling(window=20).mean(), 1, 0)

            # print("- Creating target variable...")
            df['target'] = (df['returns'].shift(-1) > 0).astype(int)
            
            # print("Handling missing values...")
            df = df.ffill().bfill()
            df = df.dropna()
            
            if df.empty:
                print("Warning: All rows removed after feature calculation and NaN removal.")
                return np.array([]), np.array([])
            
            features = ['rsi', 'macd', 'sma_short', 'sma_long', 'volume', 'returns', 'atr', 'bbw', 'roc', 'mfi', 'adx', 
                        'rsi_lag1', 'macd_lag1', 'trend_strength', 'ema_fast', 'ema_slow', 'cci', 'obv',
                        'stoch_k', 'stoch_d', 'willr', 'mom', 'log_return', 'volatility',
                        'ema_crossover', 'rsi_overbought', 'rsi_oversold', 'macd_signal', 'bbw_high']
            
            # print("Preparing final X and y arrays...")
            X = df[features].values
            y = df['target'].values
            
            # print(f"Final dataset shape - X: {X.shape}, y: {y.shape}")
            return X, y
        
        except Exception as e:
            self.logger.error(f"Error in prepare_features: {str(e)}")
            raise  # Re-raise the exception to see the full traceback

    def train_model(self):
        # Get historical data for the specified number of days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.settings['training_days'])
        print(f"\nFetching historical data from {start_date.date()} to {end_date.date()}")
        print(f"Training window: {self.settings['training_days']} days")
        print(f"Feature window size: {self.settings['feature_window']} candles")
        
        candles = self.historical_data.get_historical_data(self.product_id, start_date, end_date, granularity=self.granularity)
        print(f"Retrieved {len(candles)} candles")
        
        print("\nPreparing features...")
        X, y = self.prepare_features(candles)
        
        if X.size == 0 or y.size == 0:
            print("Error: Not enough data to train ML model.")
            return

        # Log class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        print(f"\nClass distribution:")
        print(f"Upward movements (1): {class_distribution.get(1, 0)}")
        print(f"Downward movements (0): {class_distribution.get(0, 0)}")

        print("\nSplitting data into train/validation/test sets...")
        # Split the data into training, validation, and testing sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

        print("\nApplying SMOTE to balance classes...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print("\nSelecting important features...")
        feature_selector = self.select_features(X_train_resampled, y_train_resampled)

        # Create base pipeline components
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Create base models with simpler configuration
        base_models = [
            LogisticRegression(random_state=42, max_iter=3000),
            RandomForestClassifier(random_state=42),
            XGBClassifier(random_state=42)
        ]
        
        # Create meta model
        meta_model = LogisticRegression(random_state=42)
        
        # Create the full pipeline
        self.ml_model = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', feature_selector),
            ('classifier', StackingEnsemble(base_models=base_models, meta_model=meta_model))
        ])

        # Simplified parameter grid
        param_distributions = {
            'classifier__base_models__0__C': [0.1, 1.0, 10.0],
            'classifier__base_models__1__n_estimators': [100, 200],
            'classifier__base_models__2__learning_rate': [0.01, 0.1],
            'classifier__meta_model__C': [0.1, 1.0, 10.0]
        }

        # Create and fit random search
        random_search = RandomizedSearchCV(
            estimator=self.ml_model,
            param_distributions=param_distributions,
            n_iter=5,  # Reduced number of iterations for faster testing
            cv=TimeSeriesSplit(n_splits=3),
            scoring='accuracy',
            random_state=42,
            n_jobs=-1,
            verbose=2
        )

        # Fit the random search
        random_search.fit(X_train_resampled, y_train_resampled)

        # Set the best model from random search
        self.ml_model = random_search.best_estimator_

        print("\nModel Performance Summary:")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best cross-validation score: {-random_search.best_score_:.4f}")

        # Evaluate the model on training, validation, and test sets
        sets = [
            ("Training", X_train_resampled, y_train_resampled),
            ("Validation", X_val, y_val),
            ("Test", X_test, y_test)
        ]

        print("\nPerformance metrics:")
        for set_name, X_set, y_set in sets:
            try:
                y_pred = self.ml_model.predict(X_set)
                y_pred_proba = self.ml_model.predict_proba(X_set)[:, 1]

                accuracy = accuracy_score(y_set, y_pred)
                mse = mean_squared_error(y_set, y_pred_proba)
                mae = mean_absolute_error(y_set, y_pred_proba)
                r2 = r2_score(y_set, y_pred_proba)

                print(f"\n{set_name} set metrics:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  MSE: {mse:.4f}")
                print(f"  MAE: {mae:.4f}")
                print(f"  R2: {r2:.4f}")
            except Exception as e:
                print(f"Error evaluating {set_name} set: {str(e)}")

        # Save the trained model
        try:
            print(f"\nSaving model to {self.model_file}")
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
            joblib.dump(self.ml_model, self.model_file)
            print("Model training completed!")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def select_features(self, X, y):
        # Train a simple model for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Calculate feature importance
        importances = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
        feature_importance = pd.DataFrame({'feature': self.get_feature_names(), 'importance': importances.importances_mean})
        feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)

        # Log feature importance
        self.logger.info("Feature Importance:")
        for idx, row in feature_importance.iterrows():
            self.logger.info(f"{row['feature']}: {row['importance']:.4f}")

        # Select features with importance above the mean
        importance_threshold = feature_importance['importance'].mean()
        selected_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()

        self.logger.info(f"Selected {len(selected_features)} features out of {len(self.get_feature_names())}")
        self.logger.info(f"Selected features: {', '.join(selected_features)}")

        # Create a selector based on the selected features
        selector = SelectFromModel(estimator=rf, prefit=False, threshold=importance_threshold)
        selector.fit(X, y)  # Fit the selector here
        return selector

    def get_feature_names(self):
        return ['rsi', 'macd', 'sma_short', 'sma_long', 'volume', 'returns', 'atr', 'bbw', 'roc', 'mfi', 'adx', 
                'rsi_lag1', 'macd_lag1', 'trend_strength', 'ema_fast', 'ema_slow', 'cci', 'obv',
                'stoch_k', 'stoch_d', 'willr', 'mom', 'log_return', 'volatility',
                'ema_crossover', 'rsi_overbought', 'rsi_oversold', 'macd_signal', 'bbw_high']

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
                self.logger.info(f"Model for {self.product_id} with granularity {self.granularity} is over a week old. Retraining...")
                self.train_model()
            else:
                self.logger.info(f"ML model for {self.product_id} with granularity {self.granularity} loaded from {self.model_file}")
        except FileNotFoundError:
            self.logger.warning(f"ML model file for {self.product_id} with granularity {self.granularity} not found. Training a new model.")
            self.train_model()

    def predict_signal(self, candles: List[Dict]) -> int:
        if self.ml_model is None:
            self.load_model()
        
        X, _ = self.prepare_features(candles[-self.settings['feature_window']:])  # Use the last feature_window candles for prediction. was 50
        
        if X.size == 0:
            self.logger.debug("Not enough data to make ML prediction. Returning neutral signal.")
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
        X, y = self.prepare_features(candles[-self.settings['feature_window']:])  # Use last feature_window candles for evaluation was 100
        if X.size == 0 or y.size == 0 or self.ml_model is None:
            return 1.0  # Default weight if we can't evaluate

        y_pred = self.ml_model.predict(X)
        recent_accuracy = accuracy_score(y, y_pred)
        return min(max(recent_accuracy, 0.5), 2.0)  # Scale between 0.5 and 2