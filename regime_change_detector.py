#!/usr/bin/env python3
"""
Regime Change Detector – RSI Dip Strategy Context
This script monitors recent trading metrics to decide whether the system should tighten filters based on regime deterioration.
"""

import pandas as pd
import argparse
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Regime Change Detector for RSI Dip Strategy')
    parser.add_argument('--trades', type=str, default='automated_trades.csv',
                      help='Path to the trades CSV file (default: automated_trades.csv)')
    parser.add_argument('--window', type=int, default=10,
                      help='Number of recent trades to analyze (default: 10)')
    parser.add_argument('--log', action='store_true',
                      help='Log results to regime_log.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if trades file exists
    if not os.path.exists(args.trades):
        print(f"Error: Trade log file '{args.trades}' not found")
        return
    
    # Load trade log
    df = pd.read_csv(args.trades)
    
    # Make sure we have at least some trades
    if len(df) == 0:
        print("No trades found in the log")
        return
    
    # Get only completed trades
    completed_df = df[df['Outcome'] != 'PENDING']
    
    # Check if we have enough completed trades
    if len(completed_df) < args.window:
        print(f"Warning: Only {len(completed_df)} completed trades available (requested window: {args.window})")
        print("Using all available completed trades for analysis")
        recent = completed_df
    else:
        # Rolling window of recent trades
        recent = completed_df.tail(args.window)
    
    # Calculate metrics
    win_rate = (recent["Outcome %"] > 0).mean()
    avg_mae = recent["MAE"].mean()
    avg_mfe = recent["MFE"].mean()
    avg_volume = recent["Relative Volume"].mean() if "Relative Volume" in recent.columns else 0
    avg_atr = recent["ATR %"].mean() if "ATR %" in recent.columns else 0
    
    # Calculate additional health metrics
    mfe_mae_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
    avg_rr_ratio = recent["R/R Ratio"].mean() if "R/R Ratio" in recent.columns else 0
    
    # Deterioration rules - multiple levels
    severe_deterioration = (
        (win_rate < 0.4) or
        (avg_mae > 0.6 and avg_mfe < 0.8) or
        (mfe_mae_ratio < 1.2) or
        (avg_volume < 0.4 and avg_atr < 0.2)
    )
    
    moderate_deterioration = (
        (win_rate < 0.5) or
        (avg_mae > 0.5 and avg_mfe < 1.0) or
        (mfe_mae_ratio < 1.5) or
        (avg_volume < 0.5) or
        (avg_atr < 0.2)
    )
    
    # Determine regime status
    if severe_deterioration:
        regime_status = "SEVERE_DETERIORATION"
        message = "⚠️ SEVERE Regime Deterioration – Activate Strict Filters or Pause Trading"
    elif moderate_deterioration:
        regime_status = "MODERATE_DETERIORATION"
        message = "⚠️ Regime Deteriorating – Activate Moderate Filters"
    else:
        regime_status = "STABLE"
        message = "✅ Regime Stable – Keep Filters Relaxed"
    
    # Output regime signal
    print("\n" + "="*50)
    print(f"REGIME CHANGE DETECTOR - ANALYSIS OF LAST {len(recent)} TRADES")
    print("="*50)
    print(message)
    print("-"*50)
    
    # Print metrics
    print(f"Win Rate: {win_rate:.2f}")
    print(f"Avg MAE: {avg_mae:.2f}, Avg MFE: {avg_mfe:.2f}, MFE/MAE Ratio: {mfe_mae_ratio:.2f}")
    print(f"Relative Volume: {avg_volume:.2f}, ATR%: {avg_atr:.2f}")
    print(f"Avg R/R Ratio: {avg_rr_ratio:.2f}")
    
    # Print recommendations based on regime
    print("\nRECOMMENDED ACTIONS:")
    if regime_status == "SEVERE_DETERIORATION":
        print("1. Consider pausing trading temporarily")
        print("2. Increase RSI threshold to 35")
        print("3. Require minimum volume ratio of 1.8")
        print("4. Only trade during high volatility (ATR% > 0.3)")
        print("5. Add trend slope filter (reject if slope < -0.002)")
    elif regime_status == "MODERATE_DETERIORATION":
        print("1. Increase RSI threshold to 32")
        print("2. Require minimum volume ratio of 1.5")
        print("3. Consider reducing position size by 30%")
    else:
        print("1. Maintain standard RSI threshold (30)")
        print("2. Maintain standard volume ratio filter (1.4)")
        print("3. Continue normal position sizing")
    
    # Log results if requested
    if args.log:
        log_file = "regime_log.csv"
        log_exists = os.path.exists(log_file)
        
        log_data = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Regime': regime_status,
            'Win_Rate': win_rate,
            'Avg_MAE': avg_mae,
            'Avg_MFE': avg_mfe,
            'MFE_MAE_Ratio': mfe_mae_ratio,
            'Avg_Volume': avg_volume,
            'Avg_ATR': avg_atr,
            'Avg_RR_Ratio': avg_rr_ratio,
            'Trades_Analyzed': len(recent)
        }
        
        log_df = pd.DataFrame([log_data])
        if log_exists:
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
        
        print(f"\nResults logged to {log_file}")

if __name__ == "__main__":
    main() 