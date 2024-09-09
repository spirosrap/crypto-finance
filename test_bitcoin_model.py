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
    df['date'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s')  # Explicitly convert to numeric
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

    return df.dropna().reset_index(drop=True)

def main():
    # Initialize necessary classes
    coinbase_service = CoinbaseService(API_KEY, API_SECRET)  # Create CoinbaseService instance
    historical_data = HistoricalData(coinbase_service.client)

    # Fetch historical data (e.g., 1 month of data with 5-minute granularity)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200)  # Get 1 month of data
    candles = historical_data.get_historical_data("BTC-USD", start_date, end_date, granularity="ONE_HOUR")  # Fetch data with 5-minute granularity

    # Prepare the data
    df = prepare_historical_data(candles)
    print("Historical data fetched and prepared:")
    print("\nShape of data:", df.shape)

    # Create and train the model
    model = BitcoinPredictionModel(coinbase_service)  # Pass coinbase_service to the model
    model.train(df)
    print("\nModel trained successfully.")

    # Make prediction for the next 24 hours
    last_known_values = df.iloc[-1][['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 'market_condition']]
    # Add lagged values
    for col in ['close', 'volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility']:
        last_known_values[f'lagged_{col}'] = df.iloc[-2][col]

    # Create a DataFrame for prediction
    future_prediction = model.predict(pd.DataFrame([last_known_values]))
    print(future_prediction)
    print("\nPrediction for the next 24 hours:")
    future_date = df['date'].iloc[-1] + timedelta(hours=24)  # Adjust for 24 hours ahead
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
    y_pred_class = (y_pred >= 0.5).astype(int)  # Only consider "Up" if probability is 0.5 or higher
    
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

    # # Fit ARIMA model
    # arima_model = model.fit_arima(df)

    # # Make a prediction for the next hour
    # arima_forecast = model.predict_arima(arima_model, steps=1)
    # print(f"ARIMA forecast for the next hour: {arima_forecast}")

if __name__ == "__main__":
    main()