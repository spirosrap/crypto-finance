import subprocess
import argparse
from datetime import datetime, timedelta

# Add argument parsing
parser = argparse.ArgumentParser(description="Run backtesting commands with custom product ID, granularity, and monthly options.")
parser.add_argument("--product_id", default="BTC-USDC", help="Product ID to use for backtesting (default: BTC-USDC)")
parser.add_argument("--granularity", default="ONE_HOUR", help="Granularity in seconds (default: ONE_HOUR)")
parser.add_argument("--months", type=int, default=0, help="Number of months to backtest individually (default: 0)")
args = parser.parse_args()

# List of commands to run
commands = [
    f"python base.py --start_date 2024-08-23 --product_id {args.product_id} --granularity {args.granularity}",
    f"python base.py --start_date 2024-05-01 --product_id {args.product_id} --granularity {args.granularity}",
    f"python base.py --bearmarket --product_id {args.product_id} --granularity {args.granularity}",
    f"python base.py --start_date 2023-06-01 --product_id {args.product_id} --granularity {args.granularity}",
    f"python base.py --ytd --product_id {args.product_id} --granularity {args.granularity}",
    f"python base.py --product_id {args.product_id} --granularity {args.granularity}",
    f"python base.py --bullmarket --product_id {args.product_id} --granularity {args.granularity}",
]

# Add monthly commands if --months argument is provided
if args.months > 0:
    current_date = datetime.now()
    commands = []
    for i in range(args.months):
        target_date = current_date - timedelta(days=30 * i)
        month_year = target_date.strftime("%m-%y")  # Changed to MM-YY format
        commands.append(f"python base.py --month-year {month_year} --product_id {args.product_id} --granularity {args.granularity}")

# Run each command and capture the output in real-time
for command in commands:
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    print(f"COMMAND: {command}")
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            # Print lines containing specific metrics
            if any(metric in output for metric in ["Total return:", "Number of trades:", "Sharpe Ratio:", "Maximum Drawdown:"]):
                print(f"RESULTS: {output.strip()}")
    
    print("\n" + "="*50 + "\n")
