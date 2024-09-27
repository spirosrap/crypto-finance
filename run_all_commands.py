import subprocess

# List of commands to run
commands = [
    "python base.py --start_date 2024-08-23",
    "python base.py --start_date 2024-05-01",
    "python base.py --bearmarket",
    "python base.py --start_date 2023-06-01",
    "python base.py --ytd",
    "python base.py",
    "python base.py --bullmarket",
]

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