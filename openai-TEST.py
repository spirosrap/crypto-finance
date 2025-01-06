from openai import OpenAI
from config import OPENAI_KEY
import subprocess
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Analyze market data and get AI trading recommendations')
parser.add_argument('--product_id', type=str, default='BTC-USDC', help='Trading pair to analyze (e.g., BTC-USDC, ETH-USDC)')
args = parser.parse_args()

# Run market analyzer as a subprocess and capture output, redirecting stderr to devnull
try:
    with open(os.devnull, 'w') as devnull:
        market_analysis_output = subprocess.check_output(
            ['python', 'market_analyzer.py', '--product_id', args.product_id, '--granularity', 'ONE_HOUR'],
            text=True,
            stderr=devnull
        )
except subprocess.CalledProcessError as e:
    market_analysis_output = f"Error running market analyzer: {str(e)}"

client = OpenAI(api_key=OPENAI_KEY)

# Now include the market analysis in your OpenAI prompt
response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",  # Using the latest available model
    messages=[
        {"role": "system", "content": "You are a financial advisor."},
        {"role": "user", "content": f"Here's the latest market analysis:\n{market_analysis_output}\nGive me only a SELL AT <PRICE> or BUY AT <PRICE> and a one paragraph rationale about the decision."}
    ]
)

print(response.choices[0].message.content)