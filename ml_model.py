import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BitcoinPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        X = df[['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility']]
        y = df['direction']  # Predict direction (1 for up, 0 for down or no change)
        return X, y

    def train(self, historical_data):
        X, y = self.prepare_data(historical_data)
        X = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # You can add evaluation metrics here if needed

    def predict(self, features):
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)[:, 1]  # Return probability of upward movement