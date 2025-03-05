import csv
from datetime import datetime
import os
import shutil
from trade_outcome_analyzer import parse_single_trade, analyze_trade_outcome
from coinbase.rest import RESTClient
from historicaldata import HistoricalData
from config import API_KEY_PERPS, API_SECRET_PERPS

def read_trade_output(trade_no, timestamp_str, side, entry_price):
    """Find and read the corresponding trade from trade_output.txt."""
    with open('trade_output.txt', 'r') as f:
        content = f.read()
    
    # Split content into individual trade sections
    trade_sections = content.split('====== 🤖 AI Trading Recommendation')
    
    # Convert timestamp string to datetime for comparison
    target_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    
    for section in trade_sections:
        if not section.strip():
            continue
            
        section = '====== 🤖 AI Trading Recommendation' + section
        
        # Try to parse timestamp from section
        try:
            timestamp_start = section.find('(') + 1
            timestamp_end = section.find(')')
            section_timestamp_str = section[timestamp_start:timestamp_end]
            section_timestamp = datetime.strptime(section_timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            # Check if this is the trade we're looking for
            if section_timestamp == target_timestamp:
                # Parse the trade and verify entry price
                trade_details = parse_single_trade(section)
                if (trade_details and 
                    trade_details['side'] == side and 
                    abs(trade_details['entry_price'] - float(entry_price)) < 0.01):
                    return trade_details
        except Exception as e:
            print(f"Error processing section: {e}")
            continue
    
    return None

def reprocess_trades():
    # Initialize Coinbase client and historical data
    client = RESTClient(API_KEY_PERPS, API_SECRET_PERPS)
    historical_data = HistoricalData(client)
    
    # Read existing trades
    trades = []
    with open('automated_trades.csv', 'r') as f:
        reader = csv.DictReader(f)
        trades = list(reader)
    
    # Create new CSV with updated calculations
    new_csv = 'automated_trades_new.csv'
    with open(new_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'No.', 'Timestamp', 'SIDE', 'ENTRY', 'Take Profit', 'Stop Loss',
            'Probability', 'Confidence', 'R/R Ratio', 'Volume Strength',
            'Outcome', 'Outcome %', 'Leverage', 'Margin'
        ])
        
        for trade in trades:
            print(f"\nReprocessing trade {trade['No.']} from {trade['Timestamp']}")
            
            # Find original trade details from trade_output.txt
            trade_details = read_trade_output(
                trade['No.'], 
                trade['Timestamp'],
                trade['SIDE'],
                trade['ENTRY']
            )
            
            if not trade_details:
                print(f"Could not find original trade details for trade {trade['No.']} - keeping original values")
                # Keep original values if we can't find the trade
                writer.writerow([
                    trade['No.'],
                    trade['Timestamp'],
                    trade['SIDE'],
                    trade['ENTRY'],
                    trade['Take Profit'],
                    trade['Stop Loss'],
                    trade['Probability'],
                    trade['Confidence'],
                    trade['R/R Ratio'],
                    trade['Volume Strength'],
                    trade['Outcome'],
                    trade['Outcome %'],
                    trade['Leverage'],
                    trade['Margin']
                ])
                continue
            
            # Analyze outcome with new calculations
            outcome_details = analyze_trade_outcome(trade_details, historical_data)
            
            if not outcome_details:
                print(f"Could not determine outcome for trade {trade['No.']} - keeping original outcome")
                outcome_details = {
                    'outcome': trade['Outcome'],
                    'outcome_pct': float(trade['Outcome %'])
                }
            
            # Write updated trade details
            writer.writerow([
                trade['No.'],
                trade_details['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                trade_details['side'],
                trade_details['entry_price'],
                trade_details['take_profit'],
                trade_details['stop_loss'],
                f"{trade_details['probability']}%",
                trade_details['confidence'],
                trade_details['r_r_ratio'],
                trade_details['volume_strength'],
                outcome_details['outcome'],
                round(outcome_details['outcome_pct'], 1),
                f"{trade_details['effective_leverage']}x",
                trade_details['position_size'] / trade_details['effective_leverage']
            ])
    
    # Replace old CSV with new one
    shutil.move(new_csv, 'automated_trades.csv')
    print("\nTrades reprocessed successfully!")

if __name__ == '__main__':
    reprocess_trades() 