from openai import OpenAI
from config import OPENAI_KEY, DEEPSEEK_KEY, XAI_KEY, OPENROUTER_API_KEY
import subprocess
import os
import argparse
import json
import time
from typing import Dict, Optional
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='prompt_market.log'
)

# Initialize client at module level
client = None

# Add at the top with other constants
MODEL_CONFIG = {
    'grok': 'grok-2-latest',
    'reasoner': 'deepseek-reasoner',
    'deepseek': 'deepseek-chat',
    'default': 'gpt-4o-mini',
    'o1-mini': 'o1-mini',
    'o3-mini': 'o3-mini',  # Add O3 Mini model
    'gpt4o': 'gpt-4o',
    'deepseek-r1': 'deepseek/deepseek-r1'  # Add DeepSeek R1 model
}

# Add color constants at the top
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'cyan': '\033[96m',
    'bold': '\033[1m',
    'end': '\033[0m'
}

def initialize_client(use_deepseek: bool = False, use_reasoner: bool = False, use_grok: bool = False, use_deepseek_r1: bool = False):
    global client
    try:
        if use_deepseek_r1:
            if not OPENROUTER_API_KEY:
                logging.error("OpenRouter API key is not set in the configuration")
                print(f"{COLORS['red']}Error: OpenRouter API key is not set. Please add it to your configuration.{COLORS['end']}")
                return False
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                default_headers={
                    "HTTP-Referer": "https://github.com/cursor-ai",
                    "X-Title": "Cursor AI Trading Bot"
                }
            )
            return True
            
        provider = 'X AI' if use_grok else ('DeepSeek' if (use_deepseek or use_reasoner) else 'OpenAI')
        api_key = XAI_KEY if use_grok else (DEEPSEEK_KEY if (use_deepseek or use_reasoner) else OPENAI_KEY)
        
        if not api_key:
            logging.error(f"{provider} API key is not set in the configuration")
            print(f"{COLORS['red']}Error: {provider} API key is not set. Please add it to your configuration.{COLORS['end']}")
            return False
            
        if provider == 'OpenAI' and not api_key.startswith('sk-'):
            logging.error("Invalid OpenAI API key format")
            print(f"{COLORS['red']}Error: Invalid OpenAI API key format. Key should start with 'sk-'{COLORS['end']}")
            return False

        if use_grok:
            client = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")
        elif use_deepseek or use_reasoner:
            client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        else:
            client = OpenAI(api_key=OPENAI_KEY)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if 'unauthorized' in error_msg or 'invalid api key' in error_msg:
            print(f"{COLORS['red']}Error: Invalid {provider} API key. Please check your credentials.{COLORS['end']}")
        elif 'connection' in error_msg:
            print(f"{COLORS['red']}Error: Could not connect to {provider} API. Please check your internet connection.{COLORS['end']}")
        else:
            print(f"{COLORS['red']}Error initializing {provider} client: {str(e)}{COLORS['end']}")
        logging.error(f"Failed to initialize API client: {str(e)}")
        return False

def validate_api_key(use_deepseek: bool = False, use_reasoner: bool = False, use_grok: bool = False, use_deepseek_r1: bool = False) -> bool:
    """Validate that the API key is set and well-formed."""
    if use_deepseek_r1:
        api_key = OPENROUTER_API_KEY
        provider = 'OpenRouter'
    elif use_grok:
        api_key = XAI_KEY
        provider = 'X AI'
    elif use_deepseek or use_reasoner:
        api_key = DEEPSEEK_KEY
        provider = 'DeepSeek'
    else:
        api_key = OPENAI_KEY
        provider = 'OpenAI'

    if not api_key or not isinstance(api_key, str):
        logging.error(f"{provider} API key is not set or invalid")
        return False
    if provider == 'OpenAI' and not api_key.startswith('sk-'):
        logging.error("OpenAI API key appears to be malformed")
        return False
    return True

def validate_inputs(product_id: str, granularity: str) -> bool:
    """Validate product ID and granularity parameters."""
    valid_products = ['BTC-USDC', 'ETH-USDC', "DOGE-USDC"]  # Add more as needed
    valid_granularities = ['ONE_MINUTE', 'FIVE_MINUTE', "FIFTEEN_MINUTE", 'ONE_HOUR', 'ONE_DAY']
    
    if product_id not in valid_products:
        logging.error(f"Invalid product ID: {product_id}")
        return False
    if granularity not in valid_granularities:
        logging.error(f"Invalid granularity: {granularity}")
        return False
    return True

def run_market_analysis(product_id: str, granularity: str) -> Optional[Dict]:
    """Run market analysis and return the output as a dictionary."""
    try:
        with open(os.devnull, 'w') as devnull:
            result = subprocess.check_output(
                ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', granularity],
                text=True,
                stderr=devnull,
                timeout=300  # 5 minute timeout
            )
        return {'success': True, 'data': result}
    except subprocess.TimeoutExpired:
        logging.error("Market analyzer timed out after 5 minutes")
        return {'success': False, 'error': "Analysis timed out"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Market analyzer error: {str(e)}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logging.error(f"Unexpected error in market analysis: {str(e)}")
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((
        TimeoutError,
        ConnectionError,
        openai.APIError,
        openai.APIConnectionError,
        openai.RateLimitError
    )),
    before_sleep=lambda retry_state: logging.warning(
        f"Attempt {retry_state.attempt_number} failed, retrying in {retry_state.next_action.sleep} seconds..."
    )
)
def get_trading_recommendation(client: OpenAI, market_analysis: str, product_id: str, use_deepseek: bool = False, use_reasoner: bool = False, use_grok: bool = False, use_o1_mini: bool = False, use_o3_mini: bool = False, use_gpt4o: bool = False, use_deepseek_r1: bool = False) -> tuple[Optional[str], Optional[str]]:
    """Get trading recommendation with improved retry logic."""
    if client is None:
        raise ValueError("API client not properly initialized")

    # Define provider at the start of the function
    provider = 'OpenRouter' if use_deepseek_r1 else ('X AI' if use_grok else ('DeepSeek' if (use_deepseek or use_reasoner) else 'OpenAI'))

    SYSTEM_PROMPT = (
        "Reply only with: \"BUY AT <PRICE> and SELL AT <PRICE> with STOP LOSS at <PRICE>. Probability of success: <PROBABILITY>. Signal Confidence: <CONFIDENCE>. R/R: <R/R_RATIO>.\" or "
        "\"SELL AT <PRICE> and BUY BACK AT <PRICE> with STOP LOSS at <PRICE>. Probability of success: <PROBABILITY>. Signal Confidence: <CONFIDENCE>. R/R: <R/R_RATIO>.\""
        "Instruction 1: Compute R/R ratio as: R/R = (|Entry Price - Target Price|) / (|Entry Price - Stop Loss Price|), "
        "ensuring a correct risk-to-reward calculation for both buy and sell signals."
        "Instruction 2: Signal confidence should be between Strong, Moderate, Weak"
    )
    
    # "Instruction 3: Only if when there's a >85% probability for reversal reply with \"HOLD Probability of reversal: <PROBABILITY>. Signal Confidence: <CONFIDENCE>. Pattern Detected: <PATTERN>.\""
    # "Instruction 4: Create another modified suggestion for stop and target for 20x leverage."
    
    try:
        # Simplified model selection
        model = MODEL_CONFIG['default']
        if use_grok:
            model = MODEL_CONFIG['grok']
        elif use_reasoner:
            model = MODEL_CONFIG['reasoner']
        elif use_deepseek:
            model = MODEL_CONFIG['deepseek']
        elif use_o1_mini:
            model = MODEL_CONFIG['o1-mini']
        elif use_o3_mini:
            model = MODEL_CONFIG['o3-mini']
        elif use_gpt4o:
            model = MODEL_CONFIG['gpt4o']
        elif use_deepseek_r1:
            model = MODEL_CONFIG['deepseek-r1']
            
        logging.info(f"Using model: {model} from provider: {provider}")
        
        # For models that don't support system messages, combine with user message
        messages = []
        user_content = f"Here's the latest market analysis for {product_id}:\n{market_analysis}\nBased on this analysis, provide a trading recommendation."
        
        if model in [MODEL_CONFIG['o1-mini'], MODEL_CONFIG['o3-mini']]:  # Add o3-mini to the list of models that need combined messages
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
        params = {
            "model": model,
            "messages": messages
        }
            
        response = client.chat.completions.create(**params)
        if not response.choices:
            logging.error("API response contained no choices")
            return None, None
            
        # Handle different response formats
        reasoning = None
        if use_reasoner:
            recommendation = response.choices[0].message.content
            reasoning = response.choices[0].message.reasoning_content
            logging.info(f"Reasoning behind recommendation: {reasoning}")
        else:
            recommendation = response.choices[0].message.content

        # Validate recommendation format
        if not (("BUY AT" in recommendation and "SELL AT" in recommendation) or 
                ("SELL AT" in recommendation and "BUY BACK AT" in recommendation) or
                ("HOLD" in recommendation)):
            logging.error(f"Invalid recommendation format: {recommendation}")
            return None, None
        return recommendation, reasoning
    except openai.RateLimitError:
        error_msg = f"{COLORS['red']}Error: {provider} API rate limit exceeded. Please wait a few minutes and try again.{COLORS['end']}"
        print(error_msg)
        logging.error(f"{provider} rate limit exceeded")
        raise
    except openai.APIConnectionError as e:
        error_msg = f"{COLORS['red']}Error: Could not connect to {provider} API. Please check your internet connection.{COLORS['end']}"
        print(error_msg)
        logging.error(f"API connection error: {str(e)}")
        raise
    except openai.AuthenticationError as e:
        error_msg = f"{COLORS['red']}Error: Invalid {provider} API key or authentication failed.{COLORS['end']}"
        print(error_msg)
        logging.error(f"Authentication error: {str(e)}")
        raise
    except openai.APIError as e:
        error_msg = f"{COLORS['red']}Error: {provider} API error occurred: {str(e)}{COLORS['end']}"
        print(error_msg)
        logging.error(f"API error: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"{COLORS['red']}Error: Unexpected error while getting trading recommendation: {str(e)}{COLORS['end']}"
        print(error_msg)
        logging.error(f"Failed to get trading recommendation: {str(e)}")
        raise

def format_output(recommendation: str, analysis_result: Dict, reasoning: Optional[str] = None) -> None:
    """Format and print the trading recommendation with enhanced market insights."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{COLORS['cyan']}====== 🤖 AI Trading Recommendation ({current_time}) ======{COLORS['end']}")
    print(f"{COLORS['bold']}{recommendation}{COLORS['end']}")

    if reasoning:
        print(f"\n{COLORS['yellow']}====== 🧠 Reasoning ======{COLORS['end']}")
        print(' '.join(reasoning.split()[:80]) + '...')  # Add ellipsis for truncation

    if 'data' in analysis_result:
        try:
            # Parse the market analysis data
            analysis_data = analysis_result['data']
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

        except json.JSONDecodeError as e:
            logging.warning(f"Could not parse market analysis JSON data: {str(e)}")
            logging.debug(f"Invalid JSON data received: {analysis_result['data'][:200]}...")  # Truncated for brevity
        except KeyError as e:
            logging.error(f"Missing expected key in analysis data: {str(e)}")
        except Exception as e:
            logging.error(f"Error formatting output: {str(e)}")

def validate_configuration() -> bool:
    """Validate all required configuration parameters."""
    required_env_vars = {
        'OPENAI_KEY': OPENAI_KEY,
        'DEEPSEEK_KEY': DEEPSEEK_KEY,
        'XAI_KEY': XAI_KEY
    }
    
    for name, value in required_env_vars.items():
        if not value or not isinstance(value, str):
            logging.error(f"Missing required environment variable: {name}")
            return False
    return True

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze market data and get AI trading recommendations')
    parser.add_argument('--product_id', type=str, default='BTC-USDC', help='Trading pair to analyze (e.g., BTC-USDC, ETH-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR', help='Time granularity for analysis (e.g., ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY)')
    parser.add_argument('--use_deepseek', action='store_true', help='Use DeepSeek Chat API instead of OpenAI')
    parser.add_argument('--use_reasoner', action='store_true', help='Use DeepSeek Reasoner API (includes reasoning steps)')
    parser.add_argument('--use_grok', action='store_true', help='Use X AI Grok API')
    parser.add_argument('--use_o1_mini', action='store_true', help='Use O1 Mini model')
    parser.add_argument('--use_o3_mini', action='store_true', help='Use O3 Mini model')
    parser.add_argument('--use_gpt4o', action='store_true', help='Use GPT-4O model')
    parser.add_argument('--use_deepseek_r1', action='store_true', help='Use DeepSeek R1 model from OpenRouter')
    args = parser.parse_args()

    if sum([args.use_deepseek, args.use_reasoner, args.use_grok, args.use_o1_mini, args.use_o3_mini, args.use_gpt4o, args.use_deepseek_r1]) > 1:
        print("Please choose only one of --use_deepseek, --use_reasoner, --use_grok, --use_o1_mini, --use_o3_mini, --use_gpt4o, or --use_deepseek_r1.")
        exit(1)

    try:
        # Validate API key first
        if not validate_api_key(args.use_deepseek, args.use_reasoner, args.use_grok, args.use_deepseek_r1):
            provider = 'OpenRouter' if args.use_deepseek_r1 else ('X AI' if args.use_grok else ('DeepSeek' if (args.use_deepseek or args.use_reasoner) else 'OpenAI'))
            print(f"{COLORS['red']}Error: Invalid or missing {provider} API key. Please check your configuration.{COLORS['end']}")
            exit(1)

        # Initialize the client
        if not initialize_client(args.use_deepseek, args.use_reasoner, args.use_grok, args.use_deepseek_r1):
            exit(1)  # Error message already printed in initialize_client

        # Add input validation
        if not validate_inputs(args.product_id, args.granularity):
            print("Invalid input parameters")
            exit(1)

        # Add configuration validation
        if not validate_configuration():
            print("Missing required configuration parameters")
            exit(1)

        # Run market analysis
        analysis_result = run_market_analysis(args.product_id, args.granularity)
        if not analysis_result['success']:
            print(f"Error running market analyzer: {analysis_result['error']}")
            exit(1)

        # Get trading recommendation
        recommendation, reasoning = get_trading_recommendation(client, analysis_result['data'], args.product_id, 
                                                            args.use_deepseek, args.use_reasoner, args.use_grok, 
                                                            args.use_o1_mini, args.use_o3_mini, args.use_gpt4o, args.use_deepseek_r1)
        if recommendation is None:
            print("Failed to get trading recommendation. Check the logs for details.")
            exit(1)

        # Format and display the output
        format_output(recommendation, analysis_result, reasoning)

    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}Operation cancelled by user.{COLORS['end']}")
        exit(0)
    except Exception as e:
        print(f"{COLORS['red']}An unexpected error occurred: {str(e)}{COLORS['end']}")
        logging.error(f"Unexpected error in main: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()