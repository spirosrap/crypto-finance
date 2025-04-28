import subprocess
import time
import logging
import argparse
import sys
from datetime import datetime, UTC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='ETH Trading Scheduler')
    parser.add_argument('--product_id', type=str, default='ETH-USDC',
                      help='Product ID to trade (default: ETH-USDC)')
    parser.add_argument('--margin', type=float, default=100,
                      help='Position size in USD (default: 100)')
    parser.add_argument('--leverage', type=int, default=5,
                      help='Trading leverage (default: 5)')
    return parser.parse_args()

def run_trading_bot(product_id, margin, leverage):
    try:
        # Run simplified_trading_bot.py with specified parameters
        cmd = [
            'python', 'simplified_trading_bot.py',
            '--product_id', product_id,
            '--margin', str(margin),
            '--leverage', str(leverage)
        ]
        
        logger.info(f"Running trading bot at {datetime.now(UTC)}")
        print("\n" + "=" * 50)  # Visual separator
        
        # Run the command and show output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read both stdout and stderr in real-time
        while True:
            # Read stdout
            output = process.stdout.readline()
            if output:
                print(output.strip(), flush=True)
            
            # Read stderr
            error = process.stderr.readline()
            if error:
                print(error.strip(), file=sys.stderr, flush=True)
            
            # Check if process has finished
            if output == '' and error == '' and process.poll() is not None:
                break
        
        # Get the return code
        return_code = process.poll()
        
        print("=" * 50 + "\n")  # Visual separator
        
        if return_code != 0:
            logger.error(f"Trading bot exited with code {return_code}")
            # Check if it's a connection error
            error_output = process.stderr.read()
            if "connection" in error_output.lower() or "timeout" in error_output.lower():
                logger.warning("Connection error detected, will retry after delay")
                time.sleep(30)  # Wait 30 seconds before retrying
                return False
        else:
            logger.info("Trading bot completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            logger.warning("Connection error detected, will retry after delay")
            time.sleep(30)  # Wait 30 seconds before retrying
        return False

def main():
    args = parse_args()
    logger.info(f"Starting trading scheduler for {args.product_id}")
    logger.info(f"Using margin: ${args.margin}, leverage: {args.leverage}x")
    
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while True:
        success = run_trading_bot(args.product_id, args.margin, args.leverage)
        
        if success:
            consecutive_failures = 0
            time.sleep(120)  # Wait 2 minutes before next run
        else:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                logger.error(f"Too many consecutive failures ({max_consecutive_failures}). Exiting...")
                break
            # Exponential backoff for retries
            wait_time = min(30 * (2 ** (consecutive_failures - 1)), 300)  # Max 5 minutes
            logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)

if __name__ == "__main__":
    main() 

    