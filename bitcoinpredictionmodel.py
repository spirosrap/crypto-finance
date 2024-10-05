import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from coinbaseservice import CoinbaseService
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor, ExtraTreesRegressor
from external_data import ExternalDataFetcher
from historicaldata import HistoricalData
from config import API_KEY, API_SECRET
import schedule
import time
from datetime import datetime, timedelta
import joblib
import os
import logging
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import Lasso

# COINGECKO PROVIDES ONLY UP TO ONE YEAR OF DATA FOR THE FREE TIER
DAYS_TO_TEST_MODEL = 100  # Increase from 50 to 100

class BitcoinPredictionModel:
    def __init__(self, coinbase_service):
        self.model = None
        self.scaler_X = StandardScaler()  # Scaler for features
        self.scaler_y = StandardScaler()  # Scaler for target variable
        self.params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }
        self.coinbase_service = coinbase_service
        self.model_file = 'bitcoin_prediction_model.joblib'
        self.logger = logging.getLogger(__name__)
        self.selected_features = None
        self.ensemble = None

    def prepare_data(self, candles, external_data=None):
        df = pd.DataFrame(candles)

        if df.empty:
            self.logger.error("Candles data is empty. Cannot proceed with data preparation.")
            return None, None, None

        df['date'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s').dt.date
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Calculate RSI (14 * 1hr = 14hr)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (12 * 1hr = 12hr, 26 * 1hr = 26hr, 9 * 1hr = 9hr)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Add percentage change
        df['pct_change'] = df['close'].pct_change()
        
        # Add volatility (20 * 1hr = 20hr rolling standard deviation of returns)
        df['volatility'] = df['pct_change'].rolling(window=20).std()
        
        # Add direction (1 for up, 0 for down or no change)
        df['direction'] = (df['close'].shift(-24) > df['close']).astype(int)  # Shift by 24 periods (24 hours)

        # Add market condition (this is a placeholder, you may want to calculate it based on your analysis)
        df['market_condition'] = 0  # Default value, you can modify this later based on your analysis

        # Add lagged features
        df['lagged_close'] = df['close'].shift(1)
        df['lagged_volume'] = df['volume'].shift(1)
        df['lagged_rsi'] = df['rsi'].shift(1)
        df['lagged_macd'] = df['macd'].shift(1)
        df['lagged_signal'] = df['signal'].shift(1)
        df['lagged_pct_change'] = df['pct_change'].shift(1)
        df['lagged_volatility'] = df['volatility'].shift(1)

        if external_data is not None:
            # Convert external_data 'date' column to date type if it's not already
            external_data['date'] = pd.to_datetime(external_data['date']).dt.date

            # Add external data
            df = pd.merge(df, external_data, on='date', how='left')

            # Add new features
            if 'btc_market_cap' in df.columns and 'total_crypto_market_cap' in df.columns:
                df['btc_dominance'] = df['btc_market_cap'] / df['total_crypto_market_cap']
            else:
                print("Warning: Unable to calculate btc_dominance due to missing columns")
                df['btc_dominance'] = np.nan

            if 'hash_rate' in df.columns:
                df['hash_rate_ma'] = df['hash_rate'].rolling(window=24).mean()  # 24-hour moving average
            else:
                print("Warning: Unable to calculate hash_rate_ma due to missing hash_rate column")
                df['hash_rate_ma'] = np.nan

            # Add percentage change for external data
            df['hash_rate_pct_change'] = df['hash_rate'].pct_change()
            df['total_market_cap_pct_change'] = df['total_crypto_market_cap'].pct_change()
            df['sp500_pct_change'] = df['sp500'].pct_change()

            # Add lagged external features
            df['lagged_hash_rate'] = df['hash_rate'].shift(1)
            df['lagged_btc_dominance'] = df['btc_dominance'].shift(1)
            df['lagged_total_crypto_market_cap'] = df['total_crypto_market_cap'].shift(1)
            df['lagged_sp500'] = df['sp500'].shift(1)

        df = df.dropna().reset_index(drop=True)

        if df.empty:
            self.logger.error("Dataframe is empty after preparation. Check for NaN values or insufficient data.")
            return None, None, None

        # Define all potential features
        all_features = ['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                        'market_condition', 'lagged_close', 'lagged_volume', 
                        'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                        'lagged_pct_change', 'lagged_volatility']

        if external_data is not None:
            all_features.extend(['btc_dominance', 'hash_rate_ma', 'hash_rate_pct_change',
                                 'total_market_cap_pct_change', 'sp500_pct_change',
                                 'lagged_hash_rate', 'lagged_btc_dominance',
                                 'lagged_total_crypto_market_cap', 'lagged_sp500'])

        X = df[all_features].copy()

        y = df['close'].shift(-1)

        df = df.iloc[:-1]
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        return df, X, y

    def select_features(self, X, y):
        estimator = GradientBoostingRegressor(**self.params)
        selector = RFECV(estimator=estimator, step=1, cv=TimeSeriesSplit(n_splits=5), 
                         scoring='neg_mean_squared_error', n_jobs=-1)
        selector = selector.fit(X, y)
        
        self.selected_features = X.columns[selector.support_].tolist()
        print("Selected features:", self.selected_features)
        
        return X[self.selected_features]

    def train(self):
        historical_data = HistoricalData(self.coinbase_service.client)

        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS_TO_TEST_MODEL)
        candles = historical_data.get_historical_data("BTC-USDC", start_date, end_date, granularity="ONE_HOUR")

        # Fetch external data
        external_data_fetcher = ExternalDataFetcher()
        external_data = external_data_fetcher.get_data(start_date, end_date)

        # Prepare the data
        df, X, y = self.prepare_data(candles, external_data)
        
        if X is None or y is None:
            self.logger.error("Data preparation failed. Unable to train the model.")
            return

        # Perform feature selection
        X = self.select_features(X, y)

        if len(X) < 100:  # Adjust this threshold as needed
            self.logger.error(f"Insufficient data for training. Only {len(X)} samples available.")
            return

        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        # Fit and transform the training data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Transform validation and test data
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        # Replace GridSearchCV with RandomizedSearchCV
        param_distributions = {
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 4, 8]
        }

        tscv = TimeSeriesSplit(n_splits=5)
        random_search = RandomizedSearchCV(GradientBoostingRegressor(loss='huber', n_estimators=100), 
                                           param_distributions, 
                                           n_iter=10,
                                           cv=tscv, 
                                           scoring='neg_mean_squared_error',
                                           n_jobs=-1, 
                                           verbose=2)
        
        try:
            random_search.fit(X_train_scaled, y_train_scaled)
            print("Best parameters:", random_search.best_params_)
            best_params = random_search.best_params_
        except Exception as e:
            print(f"An error occurred during random search: {e}")
            print("Falling back to default parameters.")
            best_params = self.params

        # Implement early stopping without warm start
        max_estimators = 1000
        best_val_score = float('inf')
        best_iteration = 0
        patience = 10
        model = None

        for n_estimators in range(1, max_estimators + 1):
            current_model = GradientBoostingRegressor(n_estimators=n_estimators, **best_params)
            current_model.fit(X_train_scaled, y_train_scaled)
            val_pred = current_model.predict(X_val_scaled)
            val_score = mean_squared_error(y_val_scaled, val_pred)
            
            if val_score < best_val_score:
                best_val_score = val_score
                best_iteration = n_estimators
                model = current_model
            elif n_estimators - best_iteration > patience:
                print(f"Early stopping at iteration {n_estimators}")
                break

        self.model = model
        print(f"Best number of estimators: {best_iteration}")

        # Create and train the ensemble
        gb = GradientBoostingRegressor(n_estimators=best_iteration, **best_params)
        rf = RandomForestRegressor(n_estimators=100)
        et = ExtraTreesRegressor(n_estimators=100)

        self.ensemble = VotingRegressor([('gb', gb), ('rf', rf), ('et', et)])
        self.ensemble.fit(X_train_scaled, y_train_scaled)

        # Use the ensemble for predictions
        val_predictions = self.ensemble.predict(X_val_scaled)
        test_predictions = self.ensemble.predict(X_test_scaled)

        val_mse = mean_squared_error(y_val_scaled, val_predictions)
        val_mae = mean_absolute_error(y_val_scaled, val_predictions)
        val_r2 = r2_score(y_val_scaled, val_predictions)
        print(f"Validation set MSE: {val_mse:.4f}")
        print(f"Validation set MAE: {val_mae:.4f}")
        print(f"Validation set R2: {val_r2:.4f}")

        test_mse = mean_squared_error(y_test_scaled, test_predictions)
        test_mae = mean_absolute_error(y_test_scaled, test_predictions)
        test_r2 = r2_score(y_test_scaled, test_predictions)
        print(f"Test set MSE: {test_mse:.4f}")
        print(f"Test set MAE: {test_mae:.4f}")
        print(f"Test set R2: {test_r2:.4f}")

        feature_importance = self.model.feature_importances_
        feature_names = X.columns
        for importance, name in sorted(zip(feature_importance, feature_names), reverse=True):
            print(f"{name}: {importance:.4f}")

        # Save the trained ensemble
        joblib.dump({
            'ensemble': self.ensemble,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'selected_features': self.selected_features
        }, self.model_file)
        self.logger.info(f"Bitcoin prediction model trained and saved to {self.model_file}")

    def fit_arima(self, df):
        model = sm.tsa.ARIMA(df['close'], order=(5, 1, 0))
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

    def predict_arima(self, model_fit, steps=1):
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def load_model(self):
        try:
            model_data = joblib.load(self.model_file)
            self.ensemble = model_data['ensemble']
            self.scaler_X = model_data['scaler_X']
            self.scaler_y = model_data['scaler_y']
            self.selected_features = model_data['selected_features']
            model_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.model_file))
            if model_age > timedelta(days=7):  # Retrain weekly
                self.logger.info("Bitcoin prediction model is over a week old. Retraining...")
                self.train()
            else:
                self.logger.info(f"Bitcoin prediction model loaded from {self.model_file}")
        except FileNotFoundError:
            self.logger.warning(f"Bitcoin prediction model file not found. Training a new model.")
            self.train()

    def predict(self, features):
        if self.ensemble is None:
            self.load_model()

        if self.selected_features is None:
            self.logger.error("No features have been selected. The model may not have been properly trained.")
            return None

        features = features[self.selected_features]
        
        features_scaled = self.scaler_X.transform(features)
        predictions_scaled = self.ensemble.predict(features_scaled)
        return self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()