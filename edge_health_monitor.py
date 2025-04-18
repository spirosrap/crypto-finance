# cursor: create a file called edge_health_monitor.py

"""
Edge Health Monitor: Summarizes key metrics to evaluate if your trading system 
is operating within normal, degraded, or suppressed edge conditions.
"""

import pandas as pd

def evaluate_edge_health(file_path):
    df = pd.read_csv(file_path)

    # Normalize outcomes
    df["Is_Win"] = df["Outcome"].str.contains("SUCCESS").astype(int)
    df["Is_Loss"] = df["Outcome"].str.contains("STOP LOSS").astype(int)

    # Rolling windows
    rolling_window = 20
    recent = df.tail(rolling_window)

    win_rate = recent["Is_Win"].mean() * 100
    profit_factor = recent.loc[recent["Is_Win"] == 1, "Outcome %"].sum() / abs(
        recent.loc[recent["Is_Loss"] == 1, "Outcome %"].sum() + 1e-9
    )
    max_dd = recent["Outcome %"].min()
    mae_on_wins = recent[recent["Is_Win"] == 1]["MAE"].mean()
    mfe_on_losses = recent[recent["Is_Loss"] == 1]["MFE"].mean()

    # Regime medians
    atr_median = recent["ATR %"].median()
    slope_median = recent["Trend Slope"].median()
    rel_vol_median = recent["Relative Volume"].median()

    print("=== Edge Health Summary ===")
    print(f"Win Rate (last {rolling_window}):      {win_rate:.2f}%")
    print(f"Profit Factor (last {rolling_window}): {profit_factor:.2f}")
    print(f"Max Drawdown (last {rolling_window}):  {max_dd:.2f}%")
    print(f"Avg MAE on Wins:                       {mae_on_wins:.2f}%")
    print(f"Avg MFE on Losses:                     {mfe_on_losses:.2f}%")

    print("\n=== Regime Signal Check ===")
    print(f"ATR % Median:          {atr_median:.2f}")
    print(f"Trend Slope Median:    {slope_median:.4f}")
    print(f"Relative Volume Median:{rel_vol_median:.2f}")

    # Red flag indicators
    flags = []
    if win_rate < 35: flags.append("Low Win Rate")
    if profit_factor < 1.2: flags.append("Weak PF")
    if abs(max_dd) > 10: flags.append("High Drawdown")
    if atr_median < 0.2 and slope_median > -0.001 and rel_vol_median < 1.5:
        flags.append("Suppressed Regime")

    print("\n=== Flags ===")
    if flags:
        for flag in flags:
            print(f"⚠️  {flag}")
    else:
        print("✅ System operating within expected parameters.")

# Example usage:
evaluate_edge_health("tagged_trades_with_regimes.csv")