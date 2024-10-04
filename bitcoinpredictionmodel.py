import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from technicalanalysis import TechnicalAnalysis
from coinbaseservice import CoinbaseService
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

class BitcoinPredictionModel:
    def __init__(self, coinbase_service):
        self.model = None
        self.scaler_X = StandardScaler()  # Scaler for features
        self.scaler_y = StandardScaler()  # Scaler for target variable
        self.tech_analysis = TechnicalAnalysis(coinbase_service)
        self.params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        }

    def prepare_data(self, df):
        df['lagged_close'] = df['close'].shift(1)
        df['lagged_volume'] = df['volume'].shift(1)
        df['lagged_rsi'] = df['rsi'].shift(1)
        df['lagged_macd'] = df['macd'].shift(1)
        df['lagged_signal'] = df['signal'].shift(1)
        df['lagged_pct_change'] = df['pct_change'].shift(1)
        df['lagged_volatility'] = df['volatility'].shift(1)

        df['lagged_hash_rate'] = df['hash_rate'].shift(1)
        df['lagged_btc_dominance'] = df['btc_dominance'].shift(1)
        df['lagged_total_crypto_market_cap'] = df['total_crypto_market_cap'].shift(1)
        df['lagged_sp500'] = df['sp500'].shift(1)

        df = df.dropna().reset_index(drop=True)

        X = df[['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                 'market_condition', 'lagged_close', 'lagged_volume', 
                 'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                 'lagged_pct_change', 'lagged_volatility']].copy()

        y = df['close'].shift(-1)

        df = df.iloc[:-1]
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        
        return df, X, y

    def train(self, historical_data):
        df, X, y = self.prepare_data(historical_data)
        
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

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 0.9],  # Changed 1.0 to 0.9
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(GradientBoostingRegressor(), 
                                   param_grid, 
                                   cv=tscv, 
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1, 
                                   verbose=2)
        
        try:
            grid_search.fit(X_train_scaled, y_train_scaled)
            print("Best parameters:", grid_search.best_params_)
            self.model = grid_search.best_estimator_
        except Exception as e:
            print(f"An error occurred during grid search: {e}")
            print("Falling back to default parameters.")
            self.model = GradientBoostingRegressor(**self.params)
            self.model.fit(X_train_scaled, y_train_scaled)

        val_predictions = self.model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val_scaled, val_predictions)
        val_mae = mean_absolute_error(y_val_scaled, val_predictions)
        val_r2 = r2_score(y_val_scaled, val_predictions)
        print(f"Validation set MSE: {val_mse:.4f}")
        print(f"Validation set MAE: {val_mae:.4f}")
        print(f"Validation set R2: {val_r2:.4f}")

        test_predictions = self.model.predict(X_test_scaled)
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

    def fit_arima(self, df):
        model = sm.tsa.ARIMA(df['close'], order=(5, 1, 0))
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

    def predict_arima(self, model_fit, steps=1):
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def predict(self, features):
        required_features = ['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                             'market_condition', 'lagged_close', 'lagged_volume', 
                             'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                             'lagged_pct_change', 'lagged_volatility']
        
        for feature in required_features:
            if feature not in features.columns:
                features[feature] = 0
        
        features = features[required_features]
        
        features_scaled = self.scaler_X.transform(features)
        predictions_scaled = self.model.predict(features_scaled)
        return self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()