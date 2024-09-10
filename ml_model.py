import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit  # Import cross_val_score, GridSearchCV, and TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from technicalanalysis import TechnicalAnalysis  # Import TechnicalAnalysis
from coinbaseservice import CoinbaseService
import statsmodels.api as sm  # Import statsmodels for ARIMA
from sklearn.utils.class_weight import compute_class_weight  # Import for class weights
from sklearn.metrics import make_scorer

class BitcoinPredictionModel:
    def __init__(self, coinbase_service):
        self.model = None
        self.scaler = StandardScaler()
        self.tech_analysis = TechnicalAnalysis(coinbase_service)  # Initialize TechnicalAnalysis
        # Add regularization parameters
        self.params = {
            'objective': 'binary:logistic',
            'n_estimators': 200,
            'max_depth': 5,  # New hyperparameter
            'learning_rate': 0.1,  # New hyperparameter
            'subsample': 0.8,  # New hyperparameter
            'colsample_bytree': 0.8,  # New hyperparameter
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'gamma': 0,  # Minimum loss reduction required to make a further partition
            'min_child_weight': 1  # Minimum sum of instance weight needed in a child
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

        # Add lagged features for new external data
        df['lagged_hash_rate'] = df['hash_rate'].shift(1)
        df['lagged_btc_dominance'] = df['btc_dominance'].shift(1)
        df['lagged_total_crypto_market_cap'] = df['total_crypto_market_cap'].shift(1)
        df['lagged_sp500'] = df['sp500'].shift(1)

        # Drop rows with NaN values
        df = df.dropna().reset_index(drop=True)

        # Prepare features and target
        X = df[['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                 'market_condition', 'lagged_close', 'lagged_volume', 
                 'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                 'lagged_pct_change', 'lagged_volatility']].copy()
                #  'hash_rate', 'btc_dominance', 'total_crypto_market_cap', 'sp500',
                #  'lagged_hash_rate', 'lagged_btc_dominance', 'lagged_total_crypto_market_cap', 'lagged_sp500']].copy()

        y = df['direction']  # Predict direction (1 for up, 0 for down or no change)
        
        return df, X, y  # Return the adjusted DataFrame along with X and y

    def direction_accuracy(self, y_true, y_pred):
        """Custom scoring function to evaluate direction prediction."""
        return (y_true == (y_pred >= 0.5).astype(int)).mean()

    def train(self, historical_data):
        df, X, y = self.prepare_data(historical_data)
        
        # Split the data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)

        # Scale the features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Calculate class weights
        class_weights = dict(zip(np.unique(y_train), compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

        # Update the parameter grid for tuning
        param_grid = {
            'n_estimators': [200],
            'max_depth': [5, 10],
            'learning_rate': [0.001],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [3.0],
            'reg_lambda': [1.0],
            'gamma': [0.1],
            'min_child_weight': [1, 3],
            'scale_pos_weight': [1, class_weights[1]/class_weights[0]]
        }

        # Create a custom scorer
        direction_scorer = make_scorer(self.direction_accuracy, greater_is_better=True)

        # Initialize GridSearchCV with TimeSeriesSplit and custom scorer
        tscv = TimeSeriesSplit(n_splits=5)
        grid_search = GridSearchCV(xgb.XGBClassifier(objective='binary:logistic'), 
                                   param_grid, 
                                   cv=tscv, 
                                   scoring=direction_scorer,
                                   refit='direction_accuracy',
                                   n_jobs=-1, 
                                   verbose=2)
        grid_search.fit(X_train_scaled, y_train)

        # Print the best parameters found
        print("Best parameters:", grid_search.best_params_)

        # Fit the model on the entire training dataset using the best parameters
        self.model = grid_search.best_estimator_
        self.model.fit(X_train_scaled, y_train)

        # Evaluate on validation set
        val_predictions = self.model.predict(X_val_scaled)
        val_accuracy = self.direction_accuracy(y_val, val_predictions)
        print(f"Validation set accuracy: {val_accuracy:.4f}")

        # Evaluate on test set
        test_predictions = self.model.predict(X_test_scaled)
        test_accuracy = self.direction_accuracy(y_test, test_predictions)
        print(f"Test set accuracy: {test_accuracy:.4f}")

        # Perform cross-validation using TimeSeriesSplit and custom scorer on the training set
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=tscv, scoring=direction_scorer)
        
        print(f"Cross-validation scores (Direction Accuracy): {cv_scores}")
        print(f"Mean CV score (Direction Accuracy): {cv_scores.mean():.4f}")

        # Print feature importances
        feature_importance = self.model.feature_importances_
        feature_names = X.columns
        for importance, name in sorted(zip(feature_importance, feature_names), reverse=True):
            print(f"{name}: {importance:.4f}")

    def fit_arima(self, df):
        # Fit an ARIMA model
        model = sm.tsa.ARIMA(df['close'], order=(5, 1, 0))  # Adjust order as needed
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

    def predict_arima(self, model_fit, steps=1):
        # Forecast the next 'steps' time points
        forecast = model_fit.forecast(steps=steps)
        return forecast

    def predict(self, features):
        # Ensure all required features are present
        required_features = ['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                             'market_condition', 'lagged_close', 'lagged_volume', 
                             'lagged_rsi', 'lagged_macd', 'lagged_signal', 
                             'lagged_pct_change', 'lagged_volatility']
                            #  'hash_rate', 'btc_dominance', 'total_crypto_market_cap', 'sp500',
                            #  'lagged_hash_rate', 'lagged_btc_dominance', 'lagged_total_crypto_market_cap', 'lagged_sp500']
        
        # Add missing features with default values (e.g., 0)
        for feature in required_features:
            if feature not in features.columns:
                features[feature] = 0
        
        # Reorder columns to match the order used during training
        features = features[required_features]
        
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]  # Return probability of upward movement