from openai import OpenAI
from config import OPENAI_KEY
import subprocess
import os
import argparse
import json
import time
from typing import Dict, Optional
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='prompt_market.log'
)

def validate_api_key() -> bool:
    """Validate that the OpenAI API key is set and well-formed."""
    if not OPENAI_KEY or not isinstance(OPENAI_KEY, str):
        logging.error("OpenAI API key is not set or invalid")
        return False
    if not OPENAI_KEY.startswith('sk-'):
        logging.error("OpenAI API key appears to be malformed")
        return False
    return True

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
        logging.error(f"Market analyzer error: {str(e)}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logging.error(f"Unexpected error in market analysis: {str(e)}")
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception)),
    before_sleep=lambda retry_state: logging.warning(
        f"Attempt {retry_state.attempt_number} failed, retrying..."
    )
)
def get_trading_recommendation(client: OpenAI, market_analysis: str, product_id: str) -> Optional[str]:
    """Get trading recommendation with improved retry logic."""
    SYSTEM_PROMPT = """You are an expert cryptocurrency trading advisor. Your role is to:
1. Analyze market data and technical indicators
2. Provide clear, actionable trading recommendations
3. Specify exact entry, exit, and risk management levels
4. Consider market context and volatility
5. Format recommendations consistently as 'ACTION AT PRICE' or 'HOLD UNTIL PRICE'

Your response should be structured as:
1. Primary recommendation (SELL/BUY/HOLD with specific price)
2. Brief rationale (2-3 sentences maximum)
3. Risk management levels (stop-loss and take-profit)
4. Market alerts and warnings (if any)
5. Key technical levels to watch"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here's the latest market analysis for {product_id}:\n{market_analysis}\nBased on this analysis, provide a trading recommendation."}
            ],
            temperature=0.3,
            max_tokens=300,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Failed to get trading recommendation: {str(e)}")
        raise

def format_output(recommendation: str, analysis_result: Dict) -> None:
    """Format and print the trading recommendation with enhanced market insights."""
    print("\n====== 🤖 AI Trading Recommendation ======")
    print(recommendation)

    if 'data' in analysis_result:
        try:
            # Parse the market analysis data
            analysis_data = json.loads(analysis_result['data'])
            
            # Print market alerts if available
            if 'alerts' in analysis_data:
                print("\n====== 🔔 Market Alerts ======")
                alerts = analysis_data['alerts']
                if alerts.get('triggered_alerts'):
                    print("\n🚨 Active Alerts:")
                    for alert in alerts['triggered_alerts']:
                        print(f"• [{alert['priority']}] {alert['type']}: {alert['message']}")

            # Print key levels if available
            if 'key_levels' in analysis_data:
                print("\n====== 🎯 Key Price Levels ======")
                levels = analysis_data['key_levels']
                print(f"• Support: ${levels.get('support', 'N/A')}")
                print(f"• Resistance: ${levels.get('resistance', 'N/A')}")
                print(f"• Pivot: ${levels.get('pivot', 'N/A')}")

            # Print risk metrics if available
            if 'risk_metrics' in analysis_data:
                print("\n====== ⚠️ Risk Analysis ======")
                risk = analysis_data['risk_metrics']
                print(f"• Risk Level: {risk.get('dynamic_risk', 0)*100:.1f}%")
                print(f"• Volatility: {risk.get('volatility', 0)*100:.1f}%")
                print(f"• Risk/Reward: {risk.get('risk_reward_ratio', 'N/A')}")

        except json.JSONDecodeError:
            logging.warning("Could not parse market analysis JSON data")
        except Exception as e:
            logging.error(f"Error formatting output: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze market data and get AI trading recommendations')
    parser.add_argument('--product_id', type=str, default='BTC-USDC', help='Trading pair to analyze (e.g., BTC-USDC, ETH-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR', help='Time granularity for analysis (e.g., ONE_MINUTE, FIVE_MINUTES, ONE_HOUR, ONE_DAY)')
    args = parser.parse_args()

    try:
        # Validate API key first
        if not validate_api_key():
            print("Invalid or missing OpenAI API key. Please check your configuration.")
            exit(1)

        # Run market analysis
        analysis_result = run_market_analysis(args.product_id, args.granularity)
        if not analysis_result['success']:
            print(f"Error running market analyzer: {analysis_result['error']}")
            exit(1)

        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_KEY)

        # Get trading recommendation
        recommendation = get_trading_recommendation(client, analysis_result['data'], args.product_id)
        if recommendation is None:
            print("Failed to get trading recommendation. Check the logs for details.")
            exit(1)

        # Format and display the output
        format_output(recommendation, analysis_result)

    except Exception as e:
        logging.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred. Check the logs for details.")
        exit(1)

if __name__ == "__main__":
    main()