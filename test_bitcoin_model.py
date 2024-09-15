import pandas as pd
import numpy as np
from ml_model import BitcoinPredictionModel
from datetime import datetime, timedelta
from historicaldata import HistoricalData
from coinbaseservice import CoinbaseService
from technicalanalysis import TechnicalAnalysis
from config import API_KEY, API_SECRET
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from external_data import ExternalDataFetcher
from sklearn.model_selection import train_test_split
import schedule
import time

DAYS_TO_TEST_MODEL = 200  # Global variable to define the number of days to test the model
LOOK_AHEAD_HOURS = 24 # Global variable to define the number of hours to look ahead for prediction

def prepare_historical_data(candles, external_data):
    df = pd.DataFrame(candles)
    df['date'] = pd.to_datetime(pd.to_numeric(df['start']), unit='s').dt.date  # Convert to date
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

    return df.dropna().reset_index(drop=True)

def calculate_normalized_metrics(y_true, y_pred, scaler):
    y_true_array = y_true.values.reshape(-1, 1)
    y_pred_array = y_pred.reshape(-1, 1)
    y_true_normalized = scaler.transform(y_true_array).ravel()
    y_pred_normalized = scaler.transform(y_pred_array).ravel()
    mse = mean_squared_error(y_true_normalized, y_pred_normalized)
    mae = mean_absolute_error(y_true_normalized, y_pred_normalized)
    r2 = r2_score(y_true_normalized, y_pred_normalized)
    return mse, mae, r2

def main():
    print("Starting main function")
    # Initialize necessary classes
    coinbase_service = CoinbaseService(API_KEY, API_SECRET)  # Create CoinbaseService instance
    historical_data = HistoricalData(coinbase_service.client)

    # Fetch historical data (e.g., 1 month of data with 5-minute granularity)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_TEST_MODEL)  # Get 2 months of data
    candles = historical_data.get_historical_data("BTC-USD", start_date, end_date, granularity="ONE_HOUR")  # Fetch data with 5-minute granularity

    # Fetch external data
    external_data_fetcher = ExternalDataFetcher()
    external_data = external_data_fetcher.get_data(start_date, end_date)

    # Prepare the data
    df = prepare_historical_data(candles, external_data)
    print("Historical data fetched and prepared:")
    print("\nShape of data:", df.shape)

    # Split the data into training, validation, and testing sets
    train_df, temp_df = train_test_split(df, test_size=0.3, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, shuffle=False)

    # Create and train the model
    model = BitcoinPredictionModel(coinbase_service)
    try:
        model.train(train_df)
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return

    # Evaluate the model on the training set
    train_df, X_train, y_train = model.prepare_data(train_df)
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100  # Convert to percentage
    train_mse_norm, train_mae_norm, train_r2_norm = calculate_normalized_metrics(y_train, y_train_pred, model.scaler_y)
    print(f"\nMetrics on Training Set:")
    print(f"MSE: {train_mse:.4f} (Normalized: {train_mse_norm:.4f})")
    print(f"MAE: {train_mae:.4f} (Normalized: {train_mae_norm:.4f})")
    print(f"R2: {train_r2:.4f} (Normalized: {train_r2_norm:.4f})")
    print(f"MAPE: {train_mape:.2f}%")

    # Evaluate the model on the validation set
    val_df, X_val, y_val = model.prepare_data(val_df)
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100  # Convert to percentage
    val_mse_norm, val_mae_norm, val_r2_norm = calculate_normalized_metrics(y_val, y_val_pred, model.scaler_y)
    print(f"\nMetrics on Validation Set:")
    print(f"MSE: {val_mse:.4f} (Normalized: {val_mse_norm:.4f})")
    print(f"MAE: {val_mae:.4f} (Normalized: {val_mae_norm:.4f})")
    print(f"R2: {val_r2:.4f} (Normalized: {val_r2_norm:.4f})")
    print(f"MAPE: {val_mape:.2f}%")

    # Evaluate the model on the test set
    test_df, X_test, y_test = model.prepare_data(test_df)
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100  # Convert to percentage
    test_mse_norm, test_mae_norm, test_r2_norm = calculate_normalized_metrics(y_test, y_test_pred, model.scaler_y)
    print(f"\nMetrics on Test Set:")
    print(f"MSE: {test_mse:.4f} (Normalized: {test_mse_norm:.4f})")
    print(f"MAE: {test_mae:.4f} (Normalized: {test_mae_norm:.4f})")
    print(f"R2: {test_r2:.4f} (Normalized: {test_r2_norm:.4f})")
    print(f"MAPE: {test_mape:.2f}%")

    # Plot actual vs predicted prices for the test set
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['date'], y_test, label='Actual Price')
    plt.plot(test_df['date'], y_test_pred, label='Predicted Price')
    plt.title('Bitcoin Price: Actual vs Predicted (Test Set)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('bitcoin_price_prediction_test.png')
    print("\nPrice prediction plot for test set saved as 'bitcoin_price_prediction_test.png'")

    # Make prediction for the next hour
    last_known_values = df.iloc[-1][['volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility', 
                                     'market_condition', 'hash_rate', 'btc_dominance', 
                                     'total_crypto_market_cap', 'sp500']]
    # Add lagged values
    for col in ['close', 'volume', 'rsi', 'macd', 'signal', 'pct_change', 'volatility',
                'hash_rate', 'btc_dominance', 'total_crypto_market_cap', 'sp500']:
        last_known_values[f'lagged_{col}'] = df.iloc[-2][col]

    # Create a DataFrame for prediction
    future_prediction = model.predict(pd.DataFrame([last_known_values]))
    print("\nPrediction for the next hour:")
    future_date = df['date'].iloc[-1] + timedelta(hours=1)
    print(f"{future_date}: Predicted price: ${future_prediction[0]:.2f}")

    # Calculate and print model performance metrics
    df, X, y = model.prepare_data(df)
    y_pred = model.predict(X)
    
    # Define min_len
    min_len = len(y)

    # Calculate direction accuracy
    y_pred_class = (y_pred >= 0.5).astype(int)
    direction_accuracy = (y == y_pred_class).mean()


    # Plot actual vs predicted directions
    plt.figure(figsize=(12, 6))
    plot_df = df.copy()
    plot_df['actual_up'] = y.values
    plot_df['predicted_up'] = y_pred_class

    plt.plot(plot_df['date'], plot_df['close'], label='Bitcoin Price')
    plt.scatter(plot_df['date'][plot_df['actual_up'] == 1], 
                plot_df['close'][plot_df['actual_up'] == 1], 
                color='green', label='Actual Up', alpha=0.5)
    plt.scatter(plot_df['date'][plot_df['predicted_up'] == 1], 
                plot_df['close'][plot_df['predicted_up'] == 1], 
                color='lime', label='Predicted Up (≥0.5)', alpha=0.5)
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
    # Schedule the main function to run every 1 minute
    schedule.every(1).minute.do(main)
    
    # Keep the script running
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"Network error: {e}. Retrying in 1 minute...")
            time.sleep(60)  # Wait for 1 minute before retrying
        else:
            time.sleep(3)