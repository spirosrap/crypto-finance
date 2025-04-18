#!/usr/bin/env python3
"""
volatility_threshold.py - Calculate dynamic volatility threshold for trading
using the 70th percentile of historical ATR percentage values.
"""

import numpy as np
import sys

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

def main():
    # Sample data for testing
    sample_data = [0.18, 0.22, 0.19, 0.25, 0.3, 0.24, 0.27, 0.29, 0.21, 0.31, 0.23, 0.28, 0.26, 0.32, 0.2]
    
    try:
        threshold = get_volatility_threshold(sample_data)
        print(f"Dynamic Volatility Threshold (70th percentile): {threshold}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 