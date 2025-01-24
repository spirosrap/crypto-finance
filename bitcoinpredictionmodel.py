import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor, RandomForestRegressor, ExtraTreesRegressor
from external_data import ExternalDataFetcher
from historicaldata import HistoricalData
from config import API_KEY, API_SECRET

from datetime import datetime, timedelta
import joblib
import os
import logging
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import Lasso
from tqdm import tqdm
import time

# Replace the existing DAYS_TO_TEST_MODEL dictionary with this updated version
DAYS_TO_TEST_MODEL = {
    "ONE_MINUTE": 7,    # 1 week of minute data
    "FIVE_MINUTE": 30,  # 1 month of 5-minute data
    "TEN_MINUTE": 45,   # 1.5 months of 10-minute data
    "FIFTEEN_MINUTE": 60,  # 2 months of 15-minute data
    "THIRTY_MINUTE": 90,   # 3 months of 30-minute data
    "ONE_HOUR": 180,       # 6 months of hourly data
    "SIX_HOUR": 365,       # 1 year of 6-hour data
    "ONE_DAY": 730         # 2 years of daily data
}

class BitcoinPredictionModel:
    def __init__(self, coinbase_service, product_id="BTC-USDC", granularity="ONE_HOUR"):
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
        self.product_id = product_id
        self.granularity = granularity
        self.days_to_test = DAYS_TO_TEST_MODEL.get(granularity, 180)  # Default to 180 days if granularity not found
        self.model_file = os.path.join('models', f'{product_id.lower().replace("-", "_")}_{granularity.lower()}_prediction_model.joblib')
        self.logger = logging.getLogger(__name__)
        self.selected_features = None
        self.ensemble = None

    def prepare_data(self, candles, external_data=None):
        # First, check if candles data is valid and log its structure
        if not candles or len(candles) == 0:
            self.logger.error("Empty candles data received")
            return None, None, None

        try:
            # Create a list of dictionaries with the correct structure
            data = []
            for i, candle in enumerate(candles):
                # The candles are already dictionaries, just extract the data
                try:
                    data.append({
                        'start': str(candle['start']),
                        'low': str(candle['low']),
                        'high': str(candle['high']),
                        'open': str(candle['open']),
                        'close': str(candle['close']),
                        'volume': str(candle['volume'])
                    })
                except Exception as e:
                    self.logger.error(f"Error processing candle {i}: {e}")
                    continue
            
            if not data:
                self.logger.error("No valid candles data after processing")
                return None, None, None
            
            # Create DataFrame from the structured data
            df = pd.DataFrame(data)
            
            # Convert numeric columns to float, handling non-numeric values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
            # Convert 'start' column to datetime
            if 'start' in df.columns:
                df['start'] = pd.to_datetime(df['start'].astype(int), unit='s')
        
            if df.empty:
                self.logger.error("DataFrame is empty after conversion")
                return None, None, None
            
            df['date'] = df['start'].dt.date
            
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
        except Exception as e:
            print(f"An error occurred during data preparation: {e}")
            return None, None, None

    def select_features(self, X, y, max_time=600):  # max_time in seconds (10 minutes)
        print("Starting feature selection...")
        estimator = GradientBoostingRegressor(**self.params)
        n_features = X.shape[1]
        
        class RFECV_with_progress(RFECV):
            def _fit(self, X, y, step, estimator, n_jobs):
                self.n_features_ = X.shape[1]
                self.support_ = np.ones(self.n_features_, dtype=bool)
                self.ranking_ = np.ones(self.n_features_, dtype=int)
                
                start_time = time.time()
                with tqdm(total=self.n_features_, desc="Feature Selection Progress") as pbar:
                    while np.sum(self.support_) > 1:
                        if time.time() - start_time > max_time:
                            print(f"Feature selection timed out after {max_time} seconds.")
                            break
                        
                        features_before = np.sum(self.support_)
                        super()._fit(X, y, step, estimator, n_jobs)
                        features_after = np.sum(self.support_)
                        features_removed = features_before - features_after
                        pbar.update(features_removed)
                        
                        print(f"Features remaining: {features_after}")
        
        selector = RFECV_with_progress(estimator=estimator, step=1, cv=TimeSeriesSplit(n_splits=5), 
                                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        
        try:
            selector = selector.fit(X, y)
        except Exception as e:
            print(f"An error occurred during feature selection: {e}")
            print("Falling back to using all features.")
            self.selected_features = X.columns.tolist()
            return X

        self.selected_features = X.columns[selector.support_].tolist()
        print(f"Feature selection complete. Selected {len(self.selected_features)} out of {n_features} features.")
        print("Selected features:", self.selected_features)
        
        return X[self.selected_features]

    def train(self):
        print("Starting training process...")
        historical_data = HistoricalData(self.coinbase_service.client)
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_to_test)
        print(f"Fetching historical data from {start_date} to {end_date}")
        candles = historical_data.get_historical_data(self.product_id, start_date, end_date, granularity=self.granularity)
        print(f"Fetched {len(candles)} candles")
        
        # Fetch external data
        print("Fetching external data...")
        external_data_fetcher = ExternalDataFetcher()
        external_data = external_data_fetcher.get_data(start_date, end_date)
        print(f"Fetched external data with shape: {external_data.shape if external_data is not None else 'None'}")
        
        # Prepare the data
        print("Preparing data...")
        df, X, y = self.prepare_data(candles, external_data)
        
        if X is None or y is None:
            self.logger.error("Data preparation failed. Unable to train the model.")
            return

        print(f"Prepared data shape: X: {X.shape}, y: {y.shape}")

        # Perform feature selection
        print("Performing feature selection...")
        X = self.select_features(X, y)
        print(f"Selected features: {X.columns.tolist()}")

        if len(X) < 100:  # Adjust this threshold as needed
            self.logger.error(f"Insufficient data for training. Only {len(X)} samples available.")
            return

        print("Splitting data...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        print("Scaling data...")
        # Fit and transform the training data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Transform validation and test data
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

        print("Starting RandomizedSearchCV...")
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

        print("Training final model with early stopping...")
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

            if n_estimators % 10 == 0:  # Print every 10 iterations
                print(f"Iteration {n_estimators}: MSE = {val_score:.4f}")

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
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)  # Create the 'models' directory if it doesn't exist
        joblib.dump({
            'ensemble': self.ensemble,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'selected_features': self.selected_features
        }, self.model_file)
        self.logger.info(f"{self.product_id} prediction model trained and saved to {self.model_file}")

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
                self.logger.info(f"{self.product_id} prediction model is over a week old. Retraining...")
                self.train()
            else:
                self.logger.info(f"{self.product_id} prediction model loaded from {self.model_file}")
        except FileNotFoundError:
            self.logger.warning(f"{self.product_id} prediction model file not found. Training a new model.")
            self.train()

    def predict(self, features):
        if self.ensemble is None:
            self.load_model()

        if self.selected_features is None:
            self.logger.error("No features have been selected. The model may not have been properly trained.")
            return None

        # Create a new DataFrame with only the selected features, filling missing ones with 0
        # Use float64 dtype to avoid issues with integer/float incompatibility
        features_selected = pd.DataFrame(0.0, index=features.index, columns=self.selected_features, dtype=np.float64)
        
        # Update the features, ensuring all values are converted to float
        for col in self.selected_features:
            if col in features.columns:
                features_selected[col] = features[col].astype(np.float64)

        features_scaled = self.scaler_X.transform(features_selected)
        
        try:
            predictions_scaled = self.ensemble.predict(features_scaled)
            return self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return None