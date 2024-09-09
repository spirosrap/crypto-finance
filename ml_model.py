import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit  # Import cross_val_score, GridSearchCV, and TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from technicalanalysis import TechnicalAnalysis  # Import TechnicalAnalysis
from coinbaseservice import CoinbaseService

class BitcoinPredictionModel:
    def __init__(self, coinbase_service):
        self.model = None
        self.scaler = StandardScaler()
        self.tech_analysis = TechnicalAnalysis(coinbase_service)  # Initialize TechnicalAnalysis
        # Add more hyperparameters for tuning
        self.params = {
            'objective': 'binary:logistic',
            'n_estimators': 200,
            'max_depth': 5,  # New hyperparameter
            'learning_rate': 0.1,  # New hyperparameter
            'subsample': 0.8,  # New hyperparameter
            'colsample_bytree': 0.8  # New hyperparameter
        }

    def prepare_data(self, df):
        # Create lagged features first
        df['lagged_close'] = df['close'].shift(1)
        df['lagged_volume'] = df['volume'].shift(1)
        df['lagged_rsi'] = df['rsi'].shift(1)
        df['lagged_macd'] = df['macd'].shift(1)
        df['lagged_signal'] = df['signal'].shift(1)
        df['lagged_pct_change'] = df['pct_change'].shift(1)
        df['lagged_volatility'] = df['volatility'].shift(1)

        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)

        # Prepare features and target
        X = df[['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                 'market_condition', 'lagged_close', 'lagged_volume', 
                 'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                 'lagged_pct_change', 'lagged_volatility']].copy()

        y = df['direction']  # Predict direction (1 for up, 0 for down or no change)
        
        return X, y  # Return X and y without further modification

    def train(self, historical_data):
        X, y = self.prepare_data(historical_data)
        X = self.scaler.fit_transform(X)

        # Define the parameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Initialize GridSearchCV with TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'), param_grid, cv=tscv, scoring='accuracy')
        grid_search.fit(X, y)

        # Print the best parameters found
        print("Best parameters:", grid_search.best_params_)

        # Fit the model on the entire dataset using the best parameters
        self.model = grid_search.best_estimator_
        self.model.fit(X, y)

        # Perform cross-validation using TimeSeriesSplit
        cv_scores = cross_val_score(self.model, X, y, cv=tscv)  # Time series cross-validation
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f}")

    def predict(self, features):
        # Ensure all required features are present
        required_features = ['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                             'market_condition', 'lagged_close', 'lagged_volume', 
                             'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                             'lagged_pct_change', 'lagged_volatility']
        
        # Add missing features with default values (e.g., 0)
        for feature in required_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Reorder columns to match the order used during training
        features = features[required_features]
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]  # Return probability of upward movement