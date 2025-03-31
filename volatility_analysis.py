from historicaldata import HistoricalData
from coinbase.rest import RESTClient
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict
import os

from config import API_KEY_PERPS, API_SECRET_PERPS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_hourly_volatility(
    client: RESTClient,
    product_id: str,
    days_back: int = 30
) -> pd.DataFrame:
    """
    Analyze hourly volatility patterns for a given trading pair.
    
    Args:
        client: Coinbase REST client
        product_id: Trading pair (e.g., 'BTC-USD')
        days_back: Number of days of historical data to analyze
    
    Returns:
        DataFrame with hourly volatility statistics
    """
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    # Initialize historical data fetcher
    historical_data = HistoricalData(client)
    
    # Fetch candlestick data
    logger.info(f"Fetching {days_back} days of hourly candlestick data for {product_id}")
    candles = historical_data.get_historical_data(
        product_id=product_id,
        start_date=start_date,
        end_date=end_date,
        granularity="ONE_HOUR"
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(candles)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['start'], unit='s')
    
    # Convert string values to float
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    
    # Calculate hourly range (volatility)
    df['hour'] = df['datetime'].dt.hour
    df['range'] = df['high'] - df['low']
    
    # Group by hour and calculate statistics
    hourly_stats = df.groupby('hour').agg({
        'range': ['mean', 'std', 'count']
    }).round(4)
    
    # Flatten column names
    hourly_stats.columns = ['avg_range', 'std_range', 'num_candles']
    
    # Sort by average range (volatility)
    hourly_stats = hourly_stats.sort_values('avg_range', ascending=False)
    
    return hourly_stats

def plot_hourly_volatility(stats: pd.DataFrame, product_id: str):
    """
    Create a bar chart of average volatility by hour.
    
    Args:
        stats: DataFrame with hourly volatility statistics
        product_id: Trading pair identifier for the plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    bars = plt.bar(stats.index, stats['avg_range'])
    
    # Customize the plot
    plt.title(f'Hourly Volatility Pattern - {product_id}')
    plt.xlabel('Hour of Day (UTC)')
    plt.ylabel('Average Range (High - Low)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # Set x-axis ticks to show all hours
    plt.xticks(range(24))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_dir = 'volatility_analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = f'{output_dir}/{product_id.replace("-", "_")}_hourly_volatility.png'
    plt.savefig(filename)
    plt.close()
    
    logger.info(f"Plot saved as {filename}")

def main():
    # Initialize Coinbase client with API credentials
    client = RESTClient(
        api_key=API_KEY_PERPS,
        api_secret=API_SECRET_PERPS
    )
    
    # Example trading pair
    product_id = "BTC-USD"
    
    # Analyze volatility patterns
    stats = analyze_hourly_volatility(client, product_id)
    
    # Print summary table
    print("\nHourly Volatility Summary (sorted by average range):")
    print("=" * 80)
    print(f"{'Hour':>4} {'Avg Range':>12} {'Std Dev':>12} {'Num Candles':>12}")
    print("-" * 80)
    
    for hour, row in stats.iterrows():
        print(f"{hour:4d} {row['avg_range']:12.4f} {row['std_range']:12.4f} {row['num_candles']:12.0f}")
    
    # Create visualization
    plot_hourly_volatility(stats, product_id)

if __name__ == "__main__":
    main() 