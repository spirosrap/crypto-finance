#!/usr/bin/env python3
"""
Regime Change Detector – Adaptive Exposure for RSI Dip Strategy

This script analyzes the last N live trades to detect edge deterioration.
It outputs a regime signal: STABLE or DETERIORATING, based on system metrics.
"""

import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Regime Change Detector")
    parser.add_argument('--trades', type=str, default='automated_trades.csv', help='Path to trade log CSV')
    parser.add_argument('--window', type=int, default=10, help='Number of recent trades to evaluate')
    parser.add_argument('--log', action='store_true', help='Save regime status to regime_log.csv')
    return parser.parse_args()

def evaluate_regime(df):
    win_rate = (df["Outcome %"] > 0).mean()
    avg_mae = df["MAE"].mean()
    avg_mfe = df["MFE"].mean()
    avg_volume = df["Relative Volume"].mean()
    avg_atr = df["ATR %"].mean()

    # Rule-based regime conditions
    regime_deteriorating = (
        win_rate < 0.5 or
        (avg_mae > 0.5 and avg_mfe < 1.0) or
        avg_volume < 0.5 or
        avg_atr < 0.2
    )

    regime = "DETERIORATING" if regime_deteriorating else "STABLE"
    metrics = {
        "Regime": regime,
        "Win Rate": round(win_rate, 2),
        "Avg MAE": round(avg_mae, 2),
        "Avg MFE": round(avg_mfe, 2),
        "Avg Volume": round(avg_volume, 2),
        "Avg ATR%": round(avg_atr, 2)
    }
    return metrics

def log_metrics(metrics):
    log_file = "regime_log.csv"
    df = pd.DataFrame([metrics])
    df.to_csv(log_file, mode='a', index=False, header=not os.path.exists(log_file))

def main():
    args = parse_args()

    if not os.path.exists(args.trades):
        print(f"❌ Error: {args.trades} not found.")
        return

    df = pd.read_csv(args.trades)
    if len(df) < args.window:
        print(f"⚠️ Not enough trades to evaluate (found {len(df)}, need {args.window})")
        return

    recent = df.tail(args.window)
    metrics = evaluate_regime(recent)

    print(f"📊 Regime Status: {metrics['Regime']}")
    for k, v in metrics.items():
        if k != "Regime":
            print(f"{k}: {v}")

    if args.log:
        log_metrics(metrics)

if __name__ == "__main__":
    main() 