from openai import OpenAI
from config import OPENAI_KEY
import subprocess
import os
import argparse
import json
from typing import Dict, Optional

def run_market_analysis(product_id: str, granularity: str) -> Optional[Dict]:
    """Run market analysis and return the output as a dictionary."""
    try:
        with open(os.devnull, 'w') as devnull:
            result = subprocess.check_output(
                ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', granularity],
                text=True,
                stderr=devnull
            )
        return {'success': True, 'data': result}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': str(e)}

# Set up argument parser
parser = argparse.ArgumentParser(description='Analyze market data and get AI trading recommendations')
parser.add_argument('--product_id', type=str, default='BTC-USDC', help='Trading pair to analyze (e.g., BTC-USDC, ETH-USDC)')
parser.add_argument('--granularity', type=str, default='ONE_HOUR', help='Time granularity for analysis (e.g., ONE_MINUTE, FIVE_MINUTES, ONE_HOUR, ONE_DAY)')
args = parser.parse_args()

# Run market analysis
analysis_result = run_market_analysis(args.product_id, args.granularity)

if not analysis_result['success']:
    print(f"Error running market analyzer: {analysis_result['error']}")
    exit(1)

market_analysis_output = analysis_result['data']

client = OpenAI(api_key=OPENAI_KEY)

# Comprehensive system prompt for better trading recommendations
SYSTEM_PROMPT = """You are an expert cryptocurrency trading advisor. Your role is to:
1. Analyze market data and technical indicators
2. Provide clear, actionable trading recommendations
3. Specify exact entry, exit, and risk management levels
4. Consider market context and volatility
5. Format recommendations consistently as 'ACTION AT PRICE' or 'HOLD UNTIL PRICE'

Your response should be structured as:
1. Primary recommendation (SELL/BUY/HOLD with specific price)
2. Brief rationale (2-3 sentences maximum)
3. Risk management levels (stop-loss and take-profit)"""

try:
    response = client.chat.completions.create(
        model="gpt-4",  # Using the stable GPT-4 model
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here's the latest market analysis for {args.product_id}:\n{market_analysis_output}\nBased on this analysis, provide a trading recommendation."}
        ],
        temperature=0.3,  # Lower temperature for more consistent outputs
        max_tokens=300,   # Limit response length for conciseness
        presence_penalty=0.1,  # Slight penalty to avoid repetition
        frequency_penalty=0.1  # Slight penalty to avoid repetition
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error getting trading recommendation: {str(e)}")