#!/usr/bin/env python3
import subprocess
import re
import os
import time
import signal
from datetime import datetime

def run_backtest(start_date, end_date, timeout=300):
    """Run a backtest with the given date range and return the last report."""
    print(f"\n{'='*80}")
    print(f"Running backtest for period: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    # Run the backtest command
    cmd = f"python backtest_trading_bot.py --start_date {start_date} --end_date {end_date}"
    
    try:
        # Use a timeout to prevent hanging
        process = subprocess.run(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout
        )
        
        # Print the output in real-time
        print(process.stdout)
        
        # Check for errors
        if process.returncode != 0:
            print(f"Error running backtest: {process.stderr}")
            return None
        
        # Find the last report in the output
        output_lines = process.stdout.splitlines()
        
        # Find the start of the report
        report_start_idx = -1
        for i, line in enumerate(output_lines):
            if "=== BACKTEST RESULTS ===" in line:
                report_start_idx = i
                break
        
        if report_start_idx == -1:
            print("No report found in the output")
            return None
        
        # Extract the report
        report = output_lines[report_start_idx:]
        
        return '\n'.join(report)
        
    except subprocess.TimeoutExpired:
        print(f"Backtest timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error running backtest: {e}")
        return None

def main():
    # List of date ranges to backtest
    date_ranges = [
        ("2023-01-01", "2023-02-01"),
        ("2023-03-01", "2023-04-01"),
        ("2023-06-01", "2023-07-01"),
        ("2023-11-01", "2023-12-01"),
        ("2024-01-01", "2024-02-01"),
        ("2024-03-01", "2024-04-01"),
        ("2025-03-01", "2025-04-01"),
        ("2025-01-01", "2025-02-01"),
        ("2025-02-01", "2025-03-01")
    ]
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"all_backtest_reports_{timestamp}.txt"
    
    # Create the file and write the header
    with open(output_file, 'w') as f:
        f.write(f"ALL BACKTEST REPORTS - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    # Run each backtest and append the report to the file
    for start_date, end_date in date_ranges:
        report = run_backtest(start_date, end_date)
        
        # Append to the file
        with open(output_file, 'a') as f:
            if report:
                # Write a separator and the date range
                f.write(f"\n{'='*80}\n")
                f.write(f"BACKTEST PERIOD: {start_date} to {end_date}\n")
                f.write(f"{'='*80}\n\n")
                
                # Write the report
                f.write(report)
                f.write("\n\n")
                
                print(f"\nReport for {start_date} to {end_date} saved to {output_file}")
            else:
                # Write a note that the report failed
                f.write(f"\n{'='*80}\n")
                f.write(f"BACKTEST PERIOD: {start_date} to {end_date} - FAILED\n")
                f.write(f"{'='*80}\n\n")
                
                print(f"\nReport for {start_date} to {end_date} failed")
        
        # Add a small delay between backtests to avoid overwhelming the system
        time.sleep(1)
    
    print(f"\nAll backtest reports saved to {output_file}")

if __name__ == "__main__":
    main() 