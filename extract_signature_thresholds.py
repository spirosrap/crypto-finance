# cursor: create a file called extract_signature_thresholds.py

"""
This script extracts statistical signature thresholds (Q1–Q3) from past successful trades.
It reads a CSV file with trade data and outputs the 25th, 50th, and 75th percentiles
for key technical indicators: ATR %, Trend Slope, RSI at Entry, Relative Volume, MAE, and MFE.
"""

import pandas as pd

def extract_signature_thresholds(file_path="tagged_trades_with_regimes.csv"):
    df = pd.read_csv(file_path)

    # Filter only successful trades
    df["Is_Win"] = df["Outcome"].str.contains("SUCCESS").astype(int)
    successful = df[df["Is_Win"] == 1]

    # Select technical features
    features = successful[[
        "ATR %", "Trend Slope", "RSI at Entry", "Relative Volume", "MAE", "MFE"
    ]].copy()

    # Compute Q1, Q2, Q3
    quantiles = features.quantile([0.25, 0.50, 0.75])
    print("=== Signature Thresholds (25th, 50th, 75th percentiles) ===")
    print(quantiles)

    # Optional: save to CSV
    quantiles.to_csv("signature_thresholds.csv")

# Example usage:
if __name__ == "__main__":
    extract_signature_thresholds()