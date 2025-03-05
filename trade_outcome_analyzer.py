import json
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from historicaldata import HistoricalData
from coinbase.rest import RESTClient
import os
from config import API_KEY_PERPS, API_SECRET_PERPS

def parse_single_trade(trade_section: str) -> Optional[Dict]:
    """Parse a single trade section and extract trade details."""
    try:
        # Extract the JSON part
        json_start = trade_section.find('{')
        json_end = trade_section.find('}') + 1
        trade_json = json.loads(trade_section[json_start:json_end])
        
        # Extract timestamp from the header
        timestamp_str = trade_section[trade_section.find('(') + 1:trade_section.find(')')]
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        
        # Extract other details from the Order Summary section
        lines = trade_section.split('\n')
        position_size = None
        initial_margin = None
        raw_leverage = None
        side = 'SHORT' if 'SELL AT' in str(trade_json) else 'LONG'
        
        # First pass to get initial margin and raw leverage
        for line in lines:
            if 'Initial Margin:' in line:
                initial_margin = float(line.split('$')[1].strip())
            elif 'Leverage:' in line and 'Reducing' not in line:
                # Extract leverage value, handling format like "Leverage: 10x"
                lev_str = line.split(':')[1].strip()
                raw_leverage = float(lev_str.replace('x', ''))
            elif 'Required Margin:' in line and not initial_margin:
                initial_margin = float(line.split('$')[1].strip())
                
        # Second pass to get position size and check for reduction
        position_reduced = False
        for line in lines:
            if 'Reducing position size by' in line:
                position_reduced = True
            elif 'Position Size:' in line:
                # Extract position size, handling both formats ($560.0 or $560.0 (≈0.0067 BTC))
                pos_str = line.split('$')[1].strip()
                position_size = float(pos_str.split()[0].strip())
        
        # Calculate effective leverage
        if position_size and initial_margin:
            effective_leverage = round(position_size / initial_margin, 1)
        else:
            effective_leverage = raw_leverage
            
        print(f"Trade details: Position Size=${position_size}, Initial Margin=${initial_margin}")
        print(f"Raw Leverage={raw_leverage}x, Position Reduced={position_reduced}, Effective Leverage={effective_leverage}x")
        
        # Handle both BUY and SELL trades
        if side == 'LONG':
            entry_price = float(trade_json['BUY AT'])
            take_profit = float(trade_json['SELL BACK AT'])
        else:
            entry_price = float(trade_json['SELL AT'])
            take_profit = float(trade_json['BUY BACK AT'])
            
        return {
            'timestamp': timestamp,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': float(trade_json['STOP LOSS']),
            'probability': float(trade_json['PROBABILITY']),
            'confidence': trade_json['CONFIDENCE'],
            'r_r_ratio': float(trade_json['R/R_RATIO']),
            'volume_strength': trade_json['VOLUME_STRENGTH'],
            'side': side,
            'position_size': position_size,
            'initial_margin': initial_margin,
            'effective_leverage': effective_leverage,
            'position_reduced': position_reduced
        }
    except Exception as e:
        print(f"Error parsing trade section: {e}")
        return None

def parse_trade_output(file_path: str) -> List[Dict]:
    """Parse the trade output file and extract all trades."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split content into individual trade sections
    trade_sections = content.split('====== 🤖 AI Trading Recommendation')
    trade_sections = [section.strip() for section in trade_sections if section.strip()]
    
    trades = []
    for section in trade_sections:
        section = '====== 🤖 AI Trading Recommendation' + section  # Add back the header
        trade_details = parse_single_trade(section)
        if trade_details:
            trades.append(trade_details)
        
    print(f"Found {len(trades)} trades in the file")
    return trades

def analyze_trade_outcome(trade_details: Dict, historical_data: HistoricalData) -> Optional[Dict]:
    """Analyze if the trade hit take profit or stop loss based on historical data."""
    try:
        # Use datetime objects directly
        start_date = trade_details['timestamp']
        end_date = start_date + timedelta(hours=24)
        
        # Get historical data from trade timestamp onwards
        candles = historical_data.get_historical_data(
            product_id='BTC-PERP-INTX',  # Updated to match the trade product
            start_date=start_date,
            end_date=end_date,
            granularity='ONE_MINUTE'
        )
        
        if not candles:
            print("No historical data received")
            return None
            
        entry_price = trade_details['entry_price']
        take_profit = trade_details['take_profit']
        stop_loss = trade_details['stop_loss']
        is_short = trade_details['side'] == 'SHORT'
        position_size = trade_details['position_size']
        effective_leverage = trade_details['effective_leverage']
        
        print(f"\nAnalyzing trade from {start_date} to {end_date}")
        print(f"Entry: {entry_price}, TP: {take_profit}, SL: {stop_loss}, Side: {'SHORT' if is_short else 'LONG'}")
        print(f"Position Size: ${position_size}, Effective Leverage: {effective_leverage}x")
        
        for candle in candles:
            try:
                low_price = float(candle['low'])
                high_price = float(candle['high'])
                candle_time = datetime.fromtimestamp(int(candle['start']))
                
                if is_short:
                    # For SHORT positions
                    if low_price <= take_profit:
                        # Calculate profit based on position size and effective leverage
                        price_change_pct = ((entry_price - take_profit) / entry_price)
                        profit_pct = price_change_pct * 100 * effective_leverage
                        print(f"SHORT SUCCESS - Price change: {price_change_pct*100:.2f}%, Leverage: {effective_leverage}x, Total: {profit_pct:.2f}%")
                        return {
                            'outcome': 'SUCCESS',
                            'outcome_pct': profit_pct,
                            'hit_price': take_profit,
                            'hit_time': candle_time
                        }
                    elif high_price >= stop_loss:
                        # Calculate loss based on position size and effective leverage
                        price_change_pct = ((stop_loss - entry_price) / entry_price)
                        loss_pct = price_change_pct * 100 * effective_leverage
                        print(f"SHORT STOP LOSS - Price change: {price_change_pct*100:.2f}%, Leverage: {effective_leverage}x, Total: {-loss_pct:.2f}%")
                        return {
                            'outcome': 'STOP LOSS',
                            'outcome_pct': -loss_pct,
                            'hit_price': stop_loss,
                            'hit_time': candle_time
                        }
                else:
                    # For LONG positions
                    if high_price >= take_profit:
                        # Calculate profit based on position size and effective leverage
                        price_change_pct = ((take_profit - entry_price) / entry_price)
                        profit_pct = price_change_pct * 100 * effective_leverage
                        print(f"LONG SUCCESS - Price change: {price_change_pct*100:.2f}%, Leverage: {effective_leverage}x, Total: {profit_pct:.2f}%")
                        return {
                            'outcome': 'SUCCESS',
                            'outcome_pct': profit_pct,
                            'hit_price': take_profit,
                            'hit_time': candle_time
                        }
                    elif low_price <= stop_loss:
                        # Calculate loss based on position size and effective leverage
                        price_change_pct = ((entry_price - stop_loss) / entry_price)
                        loss_pct = price_change_pct * 100 * effective_leverage
                        print(f"LONG STOP LOSS - Price change: {price_change_pct*100:.2f}%, Leverage: {effective_leverage}x, Total: {-loss_pct:.2f}%")
                        return {
                            'outcome': 'STOP LOSS',
                            'outcome_pct': -loss_pct,
                            'hit_price': stop_loss,
                            'hit_time': candle_time
                        }
            except (KeyError, ValueError) as e:
                print(f"Error processing candle: {e}")
                continue
        
        print("No outcome found in the analyzed timeframe")
        return None  # Trade hasn't completed yet
        
    except Exception as e:
        print(f"Error analyzing trade outcome: {e}")
        print(f"Details: {trade_details}")
        return None

def is_trade_recorded(trade_details: Dict, csv_file: str) -> bool:
    """Check if the trade is already recorded in the CSV file."""
    if not os.path.exists(csv_file):
        return False
        
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        trade_timestamp = trade_details['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        trade_entry = str(trade_details['entry_price'])
        
        for row in reader:
            # Check if timestamp and entry price match (unique identifier for a trade)
            if row['Timestamp'] == trade_timestamp and str(float(row['ENTRY'])) == trade_entry:
                return True
    
    return False

def add_to_csv(trade_details: Dict, outcome_details: Dict):
    """Add the trade results to automated_trades.csv if not already present."""
    csv_file = 'automated_trades.csv'
    
    # Check if trade is already recorded
    if is_trade_recorded(trade_details, csv_file):
        print("Trade already recorded in CSV, skipping...")
        return
    
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                'No.', 'Timestamp', 'SIDE', 'ENTRY', 'Take Profit', 'Stop Loss',
                'Probability', 'Confidence', 'R/R Ratio', 'Volume Strength',
                'Outcome', 'Outcome %', 'Leverage', 'Margin'
            ])
        
        # Get the next trade number
        if file_exists:
            with open(csv_file, 'r') as f:
                next_trade_no = sum(1 for line in f)
        else:
            next_trade_no = 1
            
        writer.writerow([
            next_trade_no,
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
            round(outcome_details['outcome_pct'], 2),
            f"{trade_details['effective_leverage']}x",
            trade_details['position_size'] / trade_details['effective_leverage']
        ])

def main():
    # Initialize Coinbase client and historical data with API keys
    client = RESTClient(API_KEY_PERPS, API_SECRET_PERPS)
    historical_data = HistoricalData(client)
    
    # Parse all trades from the output file
    trades = parse_trade_output('trade_output.txt')
    if not trades:
        print("No trades found in the file")
        return
    
    # Process each trade
    for trade_details in trades:
        print(f"\nProcessing trade from {trade_details['timestamp']} ({trade_details['side']} at {trade_details['entry_price']})")
        
        # Analyze outcome
        outcome_details = analyze_trade_outcome(trade_details, historical_data)
        if not outcome_details:
            print("Trade hasn't completed yet")
            continue
        
        # Add to CSV if not already recorded
        add_to_csv(trade_details, outcome_details)
        print(f"Trade recorded with outcome: {outcome_details['outcome']} ({outcome_details['outcome_pct']}%)")

if __name__ == "__main__":
    main() 