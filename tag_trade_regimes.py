# cursor: create a file called regime_classifier.py

"""
This script tags each trade with a regime signature using 5 factors:
1. ATR %
2. Trend Slope
3. RSI at Entry
4. Relative Volume
5. TP Hit Style (Wick or Clean)
"""

import pandas as pd
import argparse

def classify_regime(row):
    regime = []

    # 1. ATR %
    if row["ATR %"] > 0.7:
        regime.append("High ATR")
    elif row["ATR %"] > 0.3:
        regime.append("Moderate ATR")
    else:
        regime.append("Low ATR")

    # 2. Trend Slope
    if row["Trend Slope"] > 0.002:
        regime.append("Strong Uptrend")
    elif row["Trend Slope"] < -0.002:
        regime.append("Strong Downtrend")
    else:
        regime.append("Flat Trend")

    # 3. RSI at Entry
    if row["RSI at Entry"] < 20:
        regime.append("Oversold")
    elif row["RSI at Entry"] < 30:
        regime.append("RSI Dip")
    else:
        regime.append("Normal RSI")

    # 4. Relative Volume
    if row["Relative Volume"] > 2.0:
        regime.append("High Volume")
    elif row["Relative Volume"] > 1.0:
        regime.append("Moderate Volume")
    else:
        regime.append("Low Volume")

    # 5. TP Mode
    if "wick" in str(row["TP Mode"]).lower():
        regime.append("Wick Exit")
    else:
        regime.append("Clean Exit")

    return " | ".join(regime)

def tag_trade_regimes(csv_file):
    df = pd.read_csv(csv_file)
    df["Regime Signature"] = df.apply(classify_regime, axis=1)
    df.to_csv("tagged_trades_with_regimes.csv", index=False)
    print("Tagged trades saved to tagged_trades_with_regimes.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag trades with regime signatures")
    parser.add_argument("file", nargs="?", default="automated_trades.csv", help="CSV file containing trade data (default: automated_trades.csv)")
    args = parser.parse_args()
    
    tag_trade_regimes(args.file)