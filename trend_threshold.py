#!/usr/bin/env python3
"""
trend_threshold.py - Calculate optimal trend threshold based on historical trend slope data
"""

import numpy as np
import pandas as pd
import os

def get_trend_threshold(slope_values, percentile=70):
    """
    Calculate a trend threshold based on the specified percentile of absolute slope values.
    
    Args:
        slope_values: List or array of historical trend slope values
        percentile: Percentile to use (default: 70)
        
    Returns:
        float: Calculated trend threshold value
        
    Raises:
        ValueError: If the input slope_values is empty
    """
    if len(slope_values) == 0:
        raise ValueError("Input slope_values cannot be empty")
    
    # Take absolute values to measure trend strength regardless of direction
    abs_slopes = np.abs(slope_values)
    
    # Calculate the specified percentile
    threshold = np.percentile(abs_slopes, percentile)
    
    return threshold

def load_slope_data_from_csv(filepath='automated_trades.csv'):
    """
    Load trend slope data from the automated trades CSV file.
    
    Args:
        filepath: Path to the CSV file (default: automated_trades.csv)
        
    Returns:
        numpy.ndarray: Array of trend slope values
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV file doesn't contain trend slope data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    # Load the CSV file
    trades_df = pd.read_csv(filepath)
    
    # Check if 'Trend Slope' column exists
    if 'Trend Slope' not in trades_df.columns:
        raise ValueError("CSV file does not contain 'Trend Slope' column")
    
    # Extract trend slope data and convert to numeric, drop NaN values
    slope_values = pd.to_numeric(trades_df['Trend Slope'], errors='coerce').dropna().values
    
    if len(slope_values) == 0:
        raise ValueError("No valid trend slope data found in CSV file")
    
    return slope_values

def main():
    try:
        # Load trend slope data from CSV
        slope_values = load_slope_data_from_csv()
        
        # Calculate the trend threshold
        threshold = get_trend_threshold(slope_values)
        
        # Print the result rounded to 6 decimal places
        print(f"TREND_THRESHOLD = {threshold:.6f}")
        
        # Also print the number of trades analyzed
        print(f"Analyzed {len(slope_values)} trades")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print("Using sample data instead...")
        
        # Fallback to sample data
        sample_slope_data = [-0.0012, 0.0025, -0.0018, 0.0031, -0.0022, 0.0016, 
                             0.0037, -0.0009, 0.0029, 0.0013, -0.0026, 0.0032]
        
        threshold = get_trend_threshold(sample_slope_data)
        print(f"TREND_THRESHOLD = {threshold:.6f} (from sample data)")
    
if __name__ == "__main__":
    main() 