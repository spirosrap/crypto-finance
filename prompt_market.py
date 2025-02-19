from openai import OpenAI
import subprocess
import os
import argparse
import json
import time
import requests
from typing import Dict, Optional
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

# Use environment variables or config.py
try:
    from config import OPENAI_KEY, DEEPSEEK_KEY, XAI_KEY, OPENROUTER_API_KEY, HYPERBOLIC_KEY
except ImportError:
    OPENAI_KEY = os.getenv('OPENAI_KEY')
    DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
    XAI_KEY = os.getenv('XAI_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    HYPERBOLIC_KEY = os.getenv('HYPERBOLIC_KEY')

if not (OPENAI_KEY and DEEPSEEK_KEY and XAI_KEY and OPENROUTER_API_KEY and HYPERBOLIC_KEY):
    print("Error: Missing one or more required API keys. Please set OPENAI_KEY, DEEPSEEK_KEY, XAI_KEY, OPENROUTER_API_KEY, and HYPERBOLIC_KEY in your config or environment variables.")


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
    'deepseek-r1': 'deepseek/deepseek-r1',  # Add DeepSeek R1 model
    'ollama': 'deepseek-r1:7b',  # Add Ollama model
    'hyperbolic': 'deepseek-ai/DeepSeek-R1'  # Add Hyperbolic model
}

# Add Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Add Hyperbolic API configuration
HYPERBOLIC_API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"

# Add color constants at the top
COLORS = {
    'red': '\033[91m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'cyan': '\033[96m',
    'bold': '\033[1m',
    'end': '\033[0m'
}

def initialize_client(use_deepseek: bool = False, use_reasoner: bool = False, use_grok: bool = False, use_deepseek_r1: bool = False, use_ollama: bool = False):
    global client
    try:
        if use_ollama:
            # For Ollama, we don't need to initialize a client
            return True
            
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

def validate_api_key(use_deepseek: bool = False, use_reasoner: bool = False, use_grok: bool = False, use_deepseek_r1: bool = False, use_hyperbolic: bool = False) -> bool:
    """Validate that the API key is set and well-formed."""
    if use_hyperbolic:
        api_key = HYPERBOLIC_KEY
        provider = 'Hyperbolic'
    elif use_deepseek_r1:
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
    valid_products = ['BTC-USDC', 'ETH-USDC', "DOGE-USDC", "SOL-USDC", "SHIB-USDC"]  # Add more as needed
    valid_granularities = ['ONE_MINUTE', 'FIVE_MINUTE', "FIFTEEN_MINUTE", 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY']
    
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
                ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', granularity, '--console_logging', 'false'],
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

def get_ollama_response(prompt: str, model: str = "deepseek-r1:7b") -> Optional[str]:
    """Get response from Ollama API."""
    try:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        response = requests.post(OLLAMA_API_URL, json=data, stream=True)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    parsed_data = json.loads(line.decode("utf-8"))
                    full_response += parsed_data.get("response", "")
                except json.JSONDecodeError as json_err:
                    logging.error(f"Error parsing Ollama JSON response: {json_err}")
                    continue
        
        return full_response.strip()
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making Ollama API request: {str(e)}")
        print(f"{COLORS['red']}Error: Could not connect to Ollama API. Is Ollama running locally?{COLORS['end']}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in Ollama API call: {str(e)}")
        return None

def get_hyperbolic_response(messages: list, model: str = "deepseek-ai/DeepSeek-R1") -> Optional[str]:
    """Get response from Hyperbolic API."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HYPERBOLIC_KEY}"
        }
        
        data = {
            "messages": messages,
            "model": model,
            # "max_tokens": 508,
            # "temperature": 0.1,
            # "top_p": 0.9
        }
        
        response = requests.post(HYPERBOLIC_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        
        return None
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error making Hyperbolic API request: {str(e)}")
        print(f"{COLORS['red']}Error: Could not connect to Hyperbolic API.{COLORS['end']}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error in Hyperbolic API call: {str(e)}")
        return None

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
def get_trading_recommendation(client: OpenAI, market_analysis: str, product_id: str, use_deepseek: bool = False, use_reasoner: bool = False, use_grok: bool = False, use_o1_mini: bool = False, use_o3_mini: bool = False, use_gpt4o: bool = False, use_deepseek_r1: bool = False, use_ollama: bool = False, use_hyperbolic: bool = False) -> tuple[Optional[str], Optional[str]]:
    """Get trading recommendation with improved retry logic."""
    if not (use_ollama or use_hyperbolic) and client is None:
        raise ValueError("API client not properly initialized")

    # Add timeout parameter for API calls
    TIMEOUT = 30  # 30 seconds timeout

    # Define provider at the start of the function
    provider = 'Hyperbolic' if use_hyperbolic else ('Ollama' if use_ollama else ('OpenRouter' if use_deepseek_r1 else ('X AI' if use_grok else ('DeepSeek' if (use_deepseek or use_reasoner) else 'OpenAI'))))

    SYSTEM_PROMPT = (
        "Reply only with a valid JSON object in a single line (without any markdown code block) representing one of the following signals: "
        "For a SELL signal: {\"SELL AT\": <PRICE>, \"BUY BACK AT\": <PRICE>, \"STOP LOSS\": <PRICE>, \"PROBABILITY\": <PROBABILITY>, \"CONFIDENCE\": \"<CONFIDENCE>\", \"R/R_RATIO\": <R/R_RATIO>, \"VOLUME_STRENGTH\": \"<VOLUME_STRENGTH>\"} "
        "or for a BUY signal: {\"BUY AT\": <PRICE>, \"SELL BACK AT\": <PRICE>, \"STOP LOSS\": <PRICE>, \"PROBABILITY\": <PROBABILITY>, \"CONFIDENCE\": \"<CONFIDENCE>\", \"R/R_RATIO\": <R/R_RATIO>, \"VOLUME_STRENGTH\": \"<VOLUME_STRENGTH>\"}. "
        "Instruction 1: Use code to calculate the R/R ratio. "
        "Instruction 2: Signal confidence should be one of: 'Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak'. "
        "Instruction 3: Volume strength should be one of: 'Very High', 'High', 'Moderate', 'Low', 'Very Low'."
    )    
    
    try:
        if use_hyperbolic:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here's the latest market analysis for {product_id}:\n{market_analysis}\nBased on this analysis, provide a trading recommendation."}
            ]
            recommendation = get_hyperbolic_response(messages, MODEL_CONFIG['hyperbolic'])
            if recommendation is None:
                raise Exception("Failed to get response from Hyperbolic API")
            return recommendation, None
            
        if use_ollama:
            # Format prompt for Ollama
            full_prompt = f"{SYSTEM_PROMPT}\n\nHere's the latest market analysis for {product_id}:\n{market_analysis}\nBased on this analysis, provide a trading recommendation."
            recommendation = get_ollama_response(full_prompt, MODEL_CONFIG['ollama'])
            if recommendation is None:
                raise Exception("Failed to get response from Ollama API")
            return recommendation, None
            
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
            "messages": messages,
            "timeout": TIMEOUT
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
        if not (("BUY AT" in recommendation and "SELL BACK AT" in recommendation) or 
                ("SELL AT" in recommendation and "BUY BACK AT" in recommendation) or
                ("HOLD" in recommendation)):
            logging.error(f"Invalid recommendation format: {recommendation}")
            return None, None
        return recommendation, reasoning
    except TimeoutError:
        error_msg = f"{COLORS['red']}Error: Request timed out after {TIMEOUT} seconds.{COLORS['end']}"
        print(error_msg)
        logging.error("API request timed out")
        raise
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
        'XAI_KEY': XAI_KEY,
        'OPENROUTER_API_KEY': OPENROUTER_API_KEY,
        'HYPERBOLIC_KEY': HYPERBOLIC_KEY
    }
    
    for name, value in required_env_vars.items():
        if not value or not isinstance(value, str):
            logging.error(f"Missing required environment variable: {name}")
            return False
    return True

def validate_model_availability(model: str, provider: str) -> bool:
    """Validate that the selected model is available from the provider."""
    try:
        if provider == 'Ollama':
            # Check if Ollama server is running
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        elif provider == 'Hyperbolic':
            # Verify Hyperbolic API connectivity
            headers = {"Authorization": f"Bearer {HYPERBOLIC_KEY}"}
            response = requests.get("https://api.hyperbolic.xyz/v1/models", headers=headers, timeout=5)
            available_models = response.json().get('data', [])
            return any(m['id'] == model for m in available_models)
        elif provider == 'OpenRouter':
            # Verify OpenRouter model availability
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
            available_models = response.json()
            return model in [m['id'] for m in available_models]
        
        return True  # Default to True for other providers
        
    except Exception as e:
        logging.warning(f"Could not validate model availability for {provider}: {str(e)}")
        return True  # Default to True if check fails

def execute_trade(recommendation: str, product_id: str) -> None:
    """Execute trade based on recommendation if probability > 30% and sets stop loss from recommendation"""
    try:
        # Parse the JSON recommendation
        rec_dict = json.loads(recommendation.replace("'", '"'))  # Convert single quotes to double quotes
        
        # Extract probability (now a float without % symbol)
        prob = float(rec_dict['PROBABILITY'])
        
        # Check probability threshold
        if prob <= 60:
            print(f"{COLORS['yellow']}Trade not executed: Probability {prob:.1f}% is below threshold of 60%{COLORS['end']}")
            return
            
        # Check R/R ratio threshold
        rr_ratio = float(rec_dict['R/R_RATIO'])
        if rr_ratio < 0.2:
            print(f"{COLORS['yellow']}Trade not executed: R/R ratio {rr_ratio:.3f} is below minimum threshold of 0.2{COLORS['end']}")
            return
            
        # Determine trade direction and prices
        if 'SELL AT' in rec_dict:
            side = 'SELL'
            entry_price = float(rec_dict['SELL AT'])
            target_price = float(rec_dict['BUY BACK AT'])
            stop_loss = float(rec_dict['STOP LOSS'])
        elif 'BUY AT' in rec_dict:
            side = 'BUY'
            entry_price = float(rec_dict['BUY AT'])
            target_price = float(rec_dict['SELL BACK AT'])
            stop_loss = float(rec_dict['STOP LOSS'])
        else:
            print(f"{COLORS['red']}Invalid recommendation format{COLORS['end']}")
            return
            
        # Prepare trade parameters
        margin = 100
        leverage = 20   # Using 20x leverage        
        size_usd = margin * leverage  # 100$ margin with 20x leverage

        # Calculate stop loss percentage
        stop_loss_pct = abs((stop_loss - entry_price) / entry_price * 100)
        
        # Calculate potential profit percentage
        profit_pct = abs((target_price - entry_price) / entry_price * 100)
        
        # Adjust stop loss if it would result in larger loss than potential gain
        if stop_loss_pct > profit_pct:
            # Calculate new stop loss price that matches the profit distance
            if side == 'BUY':
                stop_loss = entry_price * (1 - profit_pct/100)  # For long positions
            else:
                stop_loss = entry_price * (1 + profit_pct/100)  # For short positions
            print(f"{COLORS['yellow']}Stop loss adjusted to match take profit distance (${stop_loss:.2f}){COLORS['end']}")
            stop_loss_pct = profit_pct  # Update the percentage since we adjusted it

        # Map product_id to perpetual format
        perp_product_map = {
            'BTC-USDC': 'BTC-PERP-INTX',
            'ETH-USDC': 'ETH-PERP-INTX',
            'DOGE-USDC': 'DOGE-PERP-INTX',
            'SOL-USDC': 'SOL-PERP-INTX',
            'SHIB-USDC': '1000SHIB-PERP-INTX'
        }
        
        perp_product = perp_product_map.get(product_id)
        if not perp_product:
            print(f"{COLORS['red']}Unsupported product for perpetual trading: {product_id}{COLORS['end']}")
            return

        # Calculate potential profit/loss in USD based on initial margin
        margin = size_usd  # Initial margin amount
        profit_usd = margin * (profit_pct / 100)
        loss_usd = margin * (stop_loss_pct / 100)
        
        # Round prices to integers for BTC trades in the command
        cmd_target_price = int(target_price) if product_id == 'BTC-USDC' else target_price
        cmd_stop_loss = int(stop_loss) if product_id == 'BTC-USDC' else stop_loss
        
        # Execute trade using subprocess
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', perp_product,
            '--side', side,
            '--size', str(size_usd),
            '--leverage', str(leverage),
            '--tp', str(cmd_target_price),
            '--sl', str(cmd_stop_loss),
            '--no-confirm'
        ]
        
        # Print trade details with additional information
        print(f"\n{COLORS['cyan']}Executing trade with parameters:{COLORS['end']}")
        print(f"Product: {perp_product}")
        print(f"Side: {side}")
        print(f"Initial Margin: ${margin}")
        print(f"Leverage: {leverage}x")
        print(f"Position Size: ${size_usd}")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Take Profit: ${target_price:.2f} ({profit_pct:.2f}% / ${profit_usd:.2f})")
        print(f"Stop Loss: ${stop_loss:.2f} ({stop_loss_pct:.2f}% / ${loss_usd:.2f})")
        print(f"Probability: {prob:.1f}%")
        print(f"Signal Confidence: {rec_dict['CONFIDENCE']}")
        print(f"R/R Ratio: {rec_dict['R/R_RATIO']:.3f}")
        print(f"Volume Strength: {rec_dict['VOLUME_STRENGTH']}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"{COLORS['green']}Trade executed successfully!{COLORS['end']}")
            if result.stdout.strip():
                print(result.stdout)
        else:
            print(f"{COLORS['red']}Error executing trade:{COLORS['end']}")
            if result.stderr.strip():
                print(f"{COLORS['red']}{result.stderr.strip()}{COLORS['end']}")
            if result.stdout.strip():  # Sometimes error details are in stdout
                print(f"{COLORS['red']}{result.stdout.strip()}{COLORS['end']}")
            
    except json.JSONDecodeError as e:
        print(f"{COLORS['red']}Error parsing recommendation JSON: {str(e)}{COLORS['end']}")
        logging.error(f"JSON parse error in execute_trade: {str(e)}\nRecommendation: {recommendation}")
    except Exception as e:
        error_details = str(e)
        print(f"{COLORS['red']}Error processing trade: {error_details}{COLORS['end']}")
        logging.error(f"Error in execute_trade: {error_details}")
        
        # Log additional context if available
        if hasattr(e, 'cmd'):
            logging.error(f"Failed command: {' '.join(e.cmd)}")
        if hasattr(e, 'output'):
            logging.error(f"Command output: {e.output}")
        if hasattr(e, 'stderr'):
            logging.error(f"Command stderr: {e.stderr}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze market data and get AI trading recommendations')
    parser.add_argument('--product_id', type=str, default='BTC-USDC', help='Trading pair to analyze (e.g., BTC-USDC, ETH-USDC, SOL-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR', help='Time granularity for analysis (e.g., ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY)')
    parser.add_argument('--use_deepseek', action='store_true', help='Use DeepSeek Chat API instead of OpenAI')
    parser.add_argument('--use_reasoner', action='store_true', help='Use DeepSeek Reasoner API (includes reasoning steps)')
    parser.add_argument('--use_grok', action='store_true', help='Use X AI Grok API')
    parser.add_argument('--use_o1_mini', action='store_true', help='Use O1 Mini model')
    parser.add_argument('--use_o3_mini', action='store_true', help='Use O3 Mini model')
    parser.add_argument('--use_gpt4o', action='store_true', help='Use GPT-4O model')
    parser.add_argument('--use_deepseek_r1', action='store_true', help='Use DeepSeek R1 model from OpenRouter')
    parser.add_argument('--use_ollama', action='store_true', help='Use Ollama model')
    parser.add_argument('--use_hyperbolic', action='store_true', help='Use Hyperbolic API')
    parser.add_argument('--execute_trades', action='store_true', help='Execute trades automatically when probability > 60%')
    args = parser.parse_args()

    if sum([args.use_deepseek, args.use_reasoner, args.use_grok, args.use_o1_mini, args.use_o3_mini, args.use_gpt4o, args.use_deepseek_r1, args.use_ollama, args.use_hyperbolic]) > 1:
        print("Please choose only one of --use_deepseek, --use_reasoner, --use_grok, --use_o1_mini, --use_o3_mini, --use_gpt4o, --use_deepseek_r1, --use_ollama, or --use_hyperbolic.")
        exit(1)

    try:
        # Validate API key first
        if not validate_api_key(args.use_deepseek, args.use_reasoner, args.use_grok, args.use_deepseek_r1, args.use_hyperbolic):
            provider = 'OpenRouter' if args.use_deepseek_r1 else ('X AI' if args.use_grok else ('DeepSeek' if (args.use_deepseek or args.use_reasoner) else 'OpenAI'))
            print(f"{COLORS['red']}Error: Invalid or missing {provider} API key. Please check your configuration.{COLORS['end']}")
            exit(1)

        # Initialize the client
        if not initialize_client(args.use_deepseek, args.use_reasoner, args.use_grok, args.use_deepseek_r1, args.use_ollama):
            exit(1)  # Error message already printed in initialize_client

        # Add input validation
        if not validate_inputs(args.product_id, args.granularity):
            print("Invalid input parameters")
            exit(1)

        # Add configuration validation
        if not validate_configuration():
            print("Missing required configuration parameters")
            exit(1)

        # Validate model availability
        model = MODEL_CONFIG.get('default')
        provider = 'OpenAI'
        
        if args.use_hyperbolic:
            model = MODEL_CONFIG['hyperbolic']
            provider = 'Hyperbolic'
        elif args.use_ollama:
            model = MODEL_CONFIG['ollama']
            provider = 'Ollama'
        elif args.use_deepseek_r1:
            model = MODEL_CONFIG['deepseek-R1']
            provider = 'OpenRouter'
        
        if not validate_model_availability(model, provider):
            print(f"{COLORS['red']}Error: Model {model} is not available from {provider}.{COLORS['end']}")
            exit(1)

        # Run market analysis
        analysis_result = run_market_analysis(args.product_id, args.granularity)
        if not analysis_result['success']:
            print(f"Error running market analyzer: {analysis_result['error']}")
            exit(1)

        # Get trading recommendation
        recommendation, reasoning = get_trading_recommendation(client, analysis_result['data'], args.product_id, 
                                                            args.use_deepseek, args.use_reasoner, args.use_grok, 
                                                            args.use_o1_mini, args.use_o3_mini, args.use_gpt4o, 
                                                            args.use_deepseek_r1, args.use_ollama, args.use_hyperbolic)
        if recommendation is None:
            print("Failed to get trading recommendation. Check the logs for details.")
            exit(1)

        # Format and display the output
        format_output(recommendation, analysis_result, reasoning)

        # Execute trade only if --execute_trades flag is provided
        if recommendation and args.execute_trades:
            execute_trade(recommendation, args.product_id)
        # elif recommendation and not args.execute_trades:
        #     print(f"\n{COLORS['yellow']}Trade not executed: --execute_trades flag not provided{COLORS['end']}")
        #     print(f"{COLORS['yellow']}To execute trades automatically, run with --execute_trades flag{COLORS['end']}")

    except KeyboardInterrupt:
        print(f"\n{COLORS['yellow']}Operation cancelled by user.{COLORS['end']}")
        exit(0)
    except Exception as e:
        print(f"{COLORS['red']}An unexpected error occurred: {str(e)}{COLORS['end']}")
        logging.error(f"Unexpected error in main: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()