from trade_tracker import TradeTracker
from config import API_KEY_PERPS, API_SECRET_PERPS
import sys

def process_trade_from_file(filename):
    try:
        with open(filename, 'r') as file:
            trade_output = file.read()
            
        tracker = TradeTracker(API_KEY_PERPS, API_SECRET_PERPS)
        success = tracker.add_trade_from_execution(trade_output)
        
        if success:
            print("Trade successfully added to automated_trades.md")
        else:
            print("Failed to add trade")
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"Error processing trade: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_trade.py <trade_output_file>")
        sys.exit(1)
        
    process_trade_from_file(sys.argv[1]) 