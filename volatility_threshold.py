#!/usr/bin/env python3
"""
volatility_threshold.py - Calculate dynamic volatility threshold for trading
using the 70th percentile of historical ATR percentage values.
"""

import numpy as np
import sys
import argparse
import csv
import os

def get_volatility_threshold(atr_percent_values, percentile=70):
    """
    Calculate a dynamic volatility threshold based on historical ATR percentage values.
    
    Args:
        atr_percent_values (list/array): List of historical ATR percentage values
        percentile (int): Percentile to use (default: 70)
        
    Returns:
        float: The calculated volatility threshold rounded to 4 decimal places
    
    Raises:
        ValueError: If input list is empty
    """
    if not atr_percent_values:
        raise ValueError("Input list cannot be empty")
    
    if not isinstance(atr_percent_values, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array")
    
    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100")
    
    # Calculate the specified percentile
    threshold = np.percentile(atr_percent_values, percentile)
    
    # Round to 4 decimal places
    return round(threshold, 4)

def read_data_from_csv(file_path, atr_column='ATR %'):
    """
    Read ATR percentage values from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        atr_column (str): Column name containing ATR percentage values
        
    Returns:
        list: List of ATR percentage values
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file has invalid data or missing column
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        atr_values = []
        with open(file_path, 'r', newline='') as csvfile:
            # First check if file is empty
            if os.stat(file_path).st_size == 0:
                raise ValueError("CSV file is empty")
            
            # Try to read file with csv module
            reader = csv.DictReader(csvfile)
            
            # Check if the required column exists
            if reader.fieldnames and atr_column not in reader.fieldnames:
                available_columns = ', '.join(reader.fieldnames)
                raise ValueError(f"Column '{atr_column}' not found in CSV. Available columns: {available_columns}")
            
            for row in reader:
                value = row[atr_column].strip()
                if value:  # Skip empty cells
                    # Handle percentage values (remove % if present)
                    value = value.replace('%', '')
                    atr_values.append(float(value))
        
        if not atr_values:
            raise ValueError(f"No valid data found in '{atr_column}' column")
            
        return atr_values
    except csv.Error as e:
        raise ValueError(f"CSV parsing error: {e}")
    except ValueError as e:
        # Re-raise value errors with more context
        raise ValueError(f"Invalid data in CSV file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Calculate dynamic volatility threshold from ATR percentage values in a CSV file.')
    parser.add_argument('file', help='Path to CSV file containing trading data')
    parser.add_argument('-c', '--column', default='ATR %', 
                        help='Column name containing ATR percentage values (default: "ATR %")')
    parser.add_argument('-p', '--percentile', type=int, default=70, 
                        help='Percentile to use (default: 70)')
    args = parser.parse_args()
    
    try:
        # Read data from CSV file
        atr_values = read_data_from_csv(args.file, args.column)
        
        # Calculate threshold
        threshold = get_volatility_threshold(atr_values, args.percentile)
        print(f"Dynamic Volatility Threshold ({args.percentile}th percentile): {threshold}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 