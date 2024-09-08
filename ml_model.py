import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score  # Import cross_val_score
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
        X = df[['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility']].copy()  # Create a copy to avoid SettingWithCopyWarning
        
        # Add market condition features
        market_condition = self.tech_analysis.analyze_market_conditions(df.to_dict(orient='records'))
        X['market_condition'] = market_condition  # Add market condition as a feature
        
        # Convert market condition to numerical values
        X['market_condition'] = X['market_condition'].map({
            "Bull Market": 1,
            "Bear Market": -1,
            "Neutral": 0
        }).fillna(0)  # Fill NaN with 0 for neutral

        y = df['direction']  # Predict direction (1 for up, 0 for down or no change)
        return X, y

    def train(self, historical_data):
        X, y = self.prepare_data(historical_data)
        X = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        self.model = xgb.XGBClassifier(**self.params)  # Use hyperparameters
        cv_scores = cross_val_score(self.model, X, y, cv=5)  # 5-fold cross-validation
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f}")
        
        # Fit the model on the entire dataset after cross-validation
        self.model.fit(X, y)
        
        # You can add evaluation metrics here if needed

    def predict(self, features):
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]  # Return probability of upward movement