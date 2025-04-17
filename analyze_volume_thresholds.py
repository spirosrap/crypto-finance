import pandas as pd
import numpy as np
from tabulate import tabulate

def load_and_process_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert 'Exit Trade' and 'ENTRY' to numeric
    df['Exit Trade'] = pd.to_numeric(df['Exit Trade'], errors='coerce')
    df['ENTRY'] = pd.to_numeric(df['ENTRY'], errors='coerce')
    
    # Calculate PnL for each trade
    df['PnL'] = (df['Exit Trade'] - df['ENTRY']) / df['ENTRY'] * 100
    
    # Convert Outcome to binary (1 for SUCCESS, 0 for STOP LOSS)
    df['Outcome_Binary'] = (df['Outcome'] == 'SUCCESS').astype(int)
    
    return df

def analyze_volume_buckets(df):
    # Define volume buckets
    buckets = [0, 0.8, 1.0, 1.2, 1.4, float('inf')]
    labels = ['< 0.8', '0.8-1.0', '1.0-1.2', '1.2-1.4', '> 1.4']
    
    # Create volume bucket column
    df['Volume_Bucket'] = pd.cut(df['Relative Volume'], bins=buckets, labels=labels)
    
    results = []
    
    for bucket in labels:
        bucket_data = df[df['Volume_Bucket'] == bucket]
        
        if len(bucket_data) == 0:
            continue
            
        trade_count = len(bucket_data)
        win_rate = (bucket_data['Outcome_Binary'].mean() * 100)
        
        # Calculate profit factor
        gains = bucket_data[bucket_data['PnL'] > 0]['PnL'].sum()
        losses = abs(bucket_data[bucket_data['PnL'] < 0]['PnL'].sum())
        profit_factor = gains / losses if losses != 0 else float('inf')
        
        results.append({
            'Volume Range': bucket,
            'Trade Count': trade_count,
            'Win Rate (%)': round(win_rate, 2),
            'Profit Factor': round(profit_factor, 2)
        })
    
    return pd.DataFrame(results)

def find_optimal_threshold(results_df):
    # Filter for buckets with at least 20 trades
    valid_buckets = results_df[results_df['Trade Count'] >= 20]
    
    if len(valid_buckets) == 0:
        return None
    
    # Find bucket with highest profit factor
    optimal_bucket = valid_buckets.loc[valid_buckets['Profit Factor'].idxmax()]
    return optimal_bucket

def main():
    # Load and process data
    df = load_and_process_data('automated_trades.csv')
    
    # Analyze volume buckets
    results = analyze_volume_buckets(df)
    
    # Find optimal threshold
    optimal = find_optimal_threshold(results)
    
    # Print results
    print("\nVolume Threshold Analysis:")
    print(tabulate(results, headers='keys', tablefmt='grid', floatfmt='.2f'))
    
    if optimal is not None:
        print("\nOptimal Volume Threshold:")
        print(f"Volume Range: {optimal['Volume Range']}")
        print(f"Trade Count: {optimal['Trade Count']}")
        print(f"Win Rate: {optimal['Win Rate (%)']}%")
        print(f"Profit Factor: {optimal['Profit Factor']}")
    else:
        print("\nNo volume bucket meets the minimum trade count requirement (20 trades)")

if __name__ == "__main__":
    main() 