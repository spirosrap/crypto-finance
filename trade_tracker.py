import os
from datetime import datetime
import json
import re
from coinbaseservice import CoinbaseService
import pandas as pd

from config import API_KEY_PERPS, API_SECRET_PERPS

class TradeTracker:
    def __init__(self, api_key, api_secret):
        self.coinbase_service = CoinbaseService(api_key, api_secret)
        self.trades_file = "automated_trades.md"
        self.ensure_trades_file_exists()

    def ensure_trades_file_exists(self):
        """Create the trades file with headers if it doesn't exist"""
        if not os.path.exists(self.trades_file):
            headers = """# AI Trading Recommendations

| Timestamp           | SIDE   | ENTRY    | Take Profit | Stop Loss  | Probability | Confidence | R/R Ratio | Volume Strength | Outcome  | Outcome % | Leverage | Margin |
|:-------------------|:-------|:---------|:------------|:-----------|:------------|:-----------|:----------|:----------------|:---------|:----------|:---------|:-------|"""
            with open(self.trades_file, 'w') as file:
                file.write(headers)

    def parse_trade_execution(self, output_text):
        """Parse the trade execution output text and extract relevant information"""
        try:
            # Find and parse the JSON data
            json_match = re.search(r'{.*}', output_text)
            if not json_match:
                raise ValueError("No JSON data found in output")
            
            trade_data = json.loads(json_match.group())
            
            # Extract timestamp using regex
            timestamp_match = re.search(r'\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\)', output_text)
            timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Extract side from the output
            side_match = re.search(r'Side: (BUY|SELL)', output_text)
            if side_match:
                side = "LONG" if side_match.group(1) == "BUY" else "SHORT"
            else:
                side = None
            
            # Extract margin and leverage
            margin_match = re.search(r'Initial Margin: \$(\d+\.?\d*)', output_text)
            margin = int(float(margin_match.group(1))) if margin_match else 50
            
            leverage_match = re.search(r'Leverage: (\d+(?:\.\d+)?)', output_text)
            leverage = leverage_match.group(1) if leverage_match else "10"
            
            # Extract entry price
            entry_price_match = re.search(r'Entry Price: \$(\d+\.?\d*)', output_text)
            if entry_price_match:
                entry_price = float(entry_price_match.group(1))
            else:
                # Try to get from current price if entry price not found
                current_price_match = re.search(r'Current Price: \$(\d+\.?\d*)', output_text)
                if current_price_match:
                    entry_price = float(current_price_match.group(1))
                else:
                    entry_price = float(trade_data.get("BUY AT" if side == "BUY" else "SELL AT", 0))
            
            # Extract take profit and stop loss from Order Summary if available
            tp_match = re.search(r'Take Profit Price: \$(\d+\.?\d*)', output_text)
            sl_match = re.search(r'Stop Loss Price: \$(\d+\.?\d*)', output_text)
            
            take_profit = float(tp_match.group(1)) if tp_match else float(trade_data.get("SELL BACK AT" if side == "BUY" else "BUY BACK AT", 0))
            stop_loss = float(sl_match.group(1)) if sl_match else float(trade_data.get("STOP LOSS", 0))
            
            # Extract outcome information if available
            outcome = "OPEN"
            outcome_pct = "0.00"
            
            # Look for success/failure indicators in the entire output
            if "SUCCESS" in output_text.upper():
                outcome = "SUCCESS"
                # Calculate success percentage based on entry and take profit prices
                if side == "LONG":
                    pct_change = ((take_profit - entry_price) / entry_price) * 100
                else:  # SHORT
                    pct_change = ((entry_price - take_profit) / entry_price) * 100
                # Apply leverage
                outcome_pct = f"{pct_change * float(leverage):.2f}"
            elif "STOP LOSS HIT" in output_text.upper():
                outcome = "STOP LOSS"
                # Calculate loss percentage based on entry and stop loss prices
                if side == "LONG":
                    pct_change = ((stop_loss - entry_price) / entry_price) * 100
                else:  # SHORT
                    pct_change = ((entry_price - stop_loss) / entry_price) * 100
                # Apply leverage and make it negative for losses
                outcome_pct = f"{pct_change * float(leverage):.2f}"
            elif "CLOSED" in output_text.upper():
                outcome = "CLOSED"
                outcome_pct = "0.00"
            
            # Calculate position size and actual leverage if available
            position_size_match = re.search(r'Position Size: \$(\d+\.?\d*)', output_text)
            if position_size_match and margin_match:
                position_size = float(position_size_match.group(1))
                margin_amount = float(margin_match.group(1))
                if margin_amount > 0:
                    actual_leverage = int(position_size / margin_amount)
                    leverage = f"{actual_leverage}"
            
            # Extract R/R ratio improvements
            if "R/R Ratio" in output_text:
                rr_match = re.search(r'R/R Ratio: (\d+\.?\d*)', output_text)
                if rr_match:
                    rr_ratio = float(rr_match.group(1))
                else:
                    rr_ratio = float(trade_data.get("R/R_RATIO", 0))
            else:
                rr_ratio = float(trade_data.get("R/R_RATIO", 0))
            
            # Compile trade information
            trade_info = {
                'timestamp': timestamp,
                'side': side,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'probability': float(trade_data.get("PROBABILITY", 0)),
                'confidence': trade_data.get("CONFIDENCE", "Moderate"),
                'rr_ratio': rr_ratio,
                'volume_strength': trade_data.get("VOLUME_STRENGTH", "Moderate"),
                'leverage': leverage,
                'margin': margin,
                'outcome': outcome,
                'outcome_pct': outcome_pct
            }
            
            return trade_info
            
        except Exception as e:
            print(f"Error parsing trade execution: {str(e)}")
            return None

    def format_trade_entry(self, trade_info):
        """Format a trade into the markdown table format"""
        try:
            # Format the entry with exact spacing
            fmt = "| {:<19} | {:<5} | {:<9.2f} | {:<11.2f} | {:<10.2f} | {:>4.1f}%       | {:<8}     |{:>5.2f}      | {:<8}        | {:<9}  | {:<9} | {}x      | {:<d}      |"
            
            # Format the entry using the format string
            entry = fmt.format(
                trade_info['timestamp'],
                trade_info['side'],
                trade_info['entry_price'],
                trade_info['take_profit'],
                trade_info['stop_loss'],
                trade_info['probability'],
                trade_info['confidence'],
                trade_info['rr_ratio'],
                trade_info['volume_strength'],
                trade_info['outcome'],
                trade_info['outcome_pct'],
                trade_info['leverage'],
                trade_info['margin']
            )
            
            return entry
            
        except Exception as e:
            print(f"Error formatting trade entry: {str(e)}")
            return None

    def add_trade_to_file(self, trade_entry):
        """Add a trade entry to the markdown file"""
        try:
            # Ensure file exists
            self.ensure_trades_file_exists()
            
            # Append the new trade entry to the end of the file
            with open(self.trades_file, 'a') as file:
                file.write('\n' + trade_entry)
                
            return True
        except Exception as e:
            print(f"Error adding trade to file: {str(e)}")
            return False

    def add_trade_from_execution(self, execution_output):
        """Process trade execution output and add it to the trades file"""
        try:
            # Parse the execution output
            trade_info = self.parse_trade_execution(execution_output)
            if not trade_info:
                print("Failed to parse trade execution output")
                return False
            
            # Format the trade entry
            trade_entry = self.format_trade_entry(trade_info)
            if not trade_entry:
                print("Failed to format trade entry")
                return False
            
            # Add the entry to the file
            success = self.add_trade_to_file(trade_entry)
            if success:
                print(f"Successfully added trade from {trade_info['timestamp']}")
            else:
                print(f"Failed to add trade to file")
            
            return success
            
        except Exception as e:
            print(f"Error adding trade from execution: {str(e)}")
            return False

def main():
    # Get API credentials
    api_key = API_KEY_PERPS
    api_secret = API_SECRET_PERPS
    
    if not api_key or not api_secret:
        print("Error: Please set API_KEY_PERPS and API_SECRET_PERPS in config.py")
        return
    
    tracker = TradeTracker(api_key, api_secret)
    
    # Example usage with trade execution output
    example_output = """
    ====== 🤖 AI Trading Recommendation (2025-02-26 23:08:52) ======
    {"BUY AT":84351.85,"SELL BACK AT":86491.46,"STOP LOSS":82237.93,"PROBABILITY":60.7,"CONFIDENCE":"Moderate","R/R_RATIO":1.012,"VOLUME_STRENGTH":"Weak","IS_VALID":true}
    
    Executing trade with parameters:
    Product: BTC-PERP-INTX
    Side: BUY
    Initial Margin: $50.0
    Leverage: 10x
    Position Size: $500.0
    Entry Price: $84351.85
    Take Profit: $86491.46
    Stop Loss: $82237.93
    """
    
    tracker.add_trade_from_execution(example_output)

if __name__ == "__main__":
    main() 