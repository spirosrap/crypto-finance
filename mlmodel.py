from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
        self.scaler = StandardScaler()

    def train(self, X, y):
        # Implement model training
        pass

    def predict(self, X):
        # Implement prediction
        pass

    def generate_ml_signal(self, product_id, start=None, end=None):
        try:
            # Fetch historical data
            end = int(datetime.utcnow().timestamp()) if end is None else int(end)
            start = end - 86400*14 if start is None else int(start)  # 30 days of data

            candles = market_data.get_candles(
                self.client,
                product_id=product_id,
                start=start,
                end=end,
                granularity="ONE_HOUR"
            )
            
            # Prepare features
            df = pd.DataFrame(candles['candles'])
            if df.empty:
                print("No data available for the specified time range.")
                return "HOLD"
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Ensure we have enough data points
            required_data_points = 100
            if len(df) < required_data_points:
                print(f"Insufficient data points for analysis. Required: {required_data_points}, Available: {len(df)}")
                return "HOLD"
            
            # Calculate technical indicators
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['RSI'] = self.compute_rsi_from_prices(df['close'].tolist())
            df['MACD'], df['Signal'], _ = self.compute_macd(df['close'])
            df['ATR'] = self.compute_atr(df['high'], df['low'], df['close'])
            
            # Drop rows with NaN values
            df = df.dropna()
            
            # Check again after dropping NaN values
            if len(df) < required_data_points:
                print(f"Insufficient data points after processing. Required: {required_data_points}, Available: {len(df)}")
                return "HOLD"
            
            # Prepare features for prediction
            X = df[['close', 'volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal', 'ATR']].values
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Prepare target variable
            y = np.where(df['close'].shift(-1) > df['close'] * 1.015, 1, 0)[:-1]  # 1 for 1.5% price increase, 0 otherwise
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-1], y, test_size=0.2, random_state=42)
            
            # Train a Random Forest model with adjusted parameters
            model = RandomForestClassifier(n_estimators=200, min_samples_split=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate model performance
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            print(f"Train accuracy: {train_accuracy:.2f}, Test accuracy: {test_accuracy:.2f}")
            
            # Make prediction for the latest data point
            latest_features = X_scaled[-1].reshape(1, -1)
            prediction_prob = model.predict_proba(latest_features)[0]
            
            # Generate signal based on prediction probability and model performance
            if test_accuracy < 0.6:  # If model performance is poor, default to HOLD
                return "HOLD"
            elif prediction_prob[1] > 0.7:  # Threshold for BUY signal
                return "BUY"
            elif prediction_prob[0] > 0.7:  # Threshold for SELL signal
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            print(f"Error generating ML signal: {e}")
            return "HOLD"  # Default to HOLD if there's an error
