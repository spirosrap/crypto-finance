import pandas as pd
import numpy as np
from ml_model import BitcoinPredictionModel
from datetime import datetime, timedelta
from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY, API_SECRET
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def prepare_historical_data(candles):
    df = pd.DataFrame(candles)
    df['date'] = pd.to_datetime(df['start'], unit='s')
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Calculate RSI (14 hours)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (12, 26, 9 hours)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Add percentage change
    df['pct_change'] = df['close'].pct_change()
    
    # Add volatility (20-hour rolling standard deviation of returns)
    df['volatility'] = df['pct_change'].rolling(window=20).std()
    
    # Add direction (1 for up, 0 for down or no change)
    df['direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    return df.dropna().reset_index(drop=True)

def main():
    # Initialize necessary classes
    coinbase_service = CoinbaseService(API_KEY, API_SECRET)
    historical_data = HistoricalData(coinbase_service.client)

    # Fetch historical data (5 months of hourly data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=150)  # Get 5 months of data
    candles = historical_data.get_historical_data("BTC-USD", start_date, end_date)
    
    # Prepare the data
    df = prepare_historical_data(candles)
    print("Historical data fetched and prepared:")
    print(df.head())
    print("\nShape of data:", df.shape)

    # Create and train the model
    model = BitcoinPredictionModel()
    model.train(df)
    print("\nModel trained successfully.")

    # Make prediction for the next day
    last_known_values = df.iloc[-1][['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility']]
    future_prediction = model.predict(pd.DataFrame([last_known_values]))
    print(future_prediction)
    print("\nPrediction for the next day:")
    future_date = df['date'].iloc[-1] + timedelta(hours=1)
    if future_prediction[0] >= 0.7:
        predicted_direction = "Up"
    elif future_prediction[0] <= 0.3:
        predicted_direction = "Down"
    else:
        predicted_direction = "Uncertain"
    print(f"{future_date}: Predicted direction: {predicted_direction} (Probability: {future_prediction[0]:.2f})")

    # Calculate and print model performance metrics
    X, y = model.prepare_data(df)
    y_pred = model.predict(X)
    y_pred_class = (y_pred >= 0.7).astype(int)  # Only consider "Up" if probability is 0.7 or higher
    
    # Ensure all arrays have the same length
    min_len = min(len(df) - 1, len(y), len(y_pred_class))
    
    accuracy = accuracy_score(y[:min_len], y_pred_class[:min_len])
    precision = precision_score(y[:min_len], y_pred_class[:min_len])
    recall = recall_score(y[:min_len], y_pred_class[:min_len])
    f1 = f1_score(y[:min_len], y_pred_class[:min_len])
    
    print("\nModel Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot actual vs predicted directions
    plt.figure(figsize=(12, 6))
    plot_df = df.iloc[1:min_len+1].copy()
    plot_df['actual_up'] = y[:min_len].values
    plot_df['predicted_up'] = y_pred_class[:min_len]

    plt.plot(plot_df['date'], plot_df['close'], label='Bitcoin Price')
    plt.scatter(plot_df['date'][plot_df['actual_up'] == 1], 
                plot_df['close'][plot_df['actual_up'] == 1], 
                color='green', label='Actual Up', alpha=0.5)
    plt.scatter(plot_df['date'][plot_df['predicted_up'] == 1], 
                plot_df['close'][plot_df['predicted_up'] == 1], 
                color='lime', label='Predicted Up (≥0.7)', alpha=0.5)
    plt.title('Bitcoin Price and Direction Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('bitcoin_direction_prediction.png')
    print("\nDirection prediction plot saved as 'bitcoin_direction_prediction.png'")

if __name__ == "__main__":
    main()