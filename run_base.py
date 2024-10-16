import subprocess
import time
from datetime import datetime, date

def run_command():
    # Get the first day of the current month
    today = date.today()
    first_day_of_month = date(today.year, today.month, 1)
    start_date = first_day_of_month.strftime("%Y-%m-%d")

    command = [f"python base.py --start_date {start_date} --product_id BTC-USDC --granularity ONE_MINUTE"]

    return subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

def main():
    all_output = []

    while True:  # Outer loop to keep the script running continuously
        process = run_command()
        
        # Wait for the command to finish without showing any output
        process.wait()
        
        # Collect all output after the command has finished
        all_output = process.stdout.readlines()
        
        # Clear the console and print the last 8 lines
        print("\033c", end="")  # Clear console
        print("Last 8 lines of output:")
        for line in all_output[-10:]:
            print(line.strip())
        
        print("Command finished. Restarting in 5 seconds...")
        time.sleep(5)  # Wait for 5 seconds before restarting the command

if __name__ == "__main__":
    main()
