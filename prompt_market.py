from openai import OpenAI
import subprocess
import os
import argparse
import json
import time
import requests
import csv
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
from coinbaseservice import CoinbaseService
# Use environment variables or config.py
try:
    from config import OPENAI_KEY, DEEPSEEK_KEY, XAI_KEY, OPENROUTER_API_KEY, HYPERBOLIC_KEY, API_KEY_PERPS, API_SECRET_PERPS
except ImportError:
    OPENAI_KEY = os.getenv('OPENAI_KEY')
    DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
    XAI_KEY = os.getenv('XAI_KEY')
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    HYPERBOLIC_KEY = os.getenv('HYPERBOLIC_KEY')

if not (OPENAI_KEY and DEEPSEEK_KEY and XAI_KEY and OPENROUTER_API_KEY and HYPERBOLIC_KEY and API_KEY_PERPS and API_SECRET_PERPS):
    print("Error: Missing one or more required API keys. Please set OPENAI_KEY, DEEPSEEK_KEY, XAI_KEY, OPENROUTER_API_KEY, API_KEY_PERPS, and API_SECRET_PERPS and HYPERBOLIC_KEY in your config or environment variables.")


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
    'gpt45-preview': 'gpt-4.5-preview',  # Add GPT-4.5 Preview model
    'o1': 'o1',  # Add O1 model
    'o1-mini': 'o1-mini',
    'o3-mini': 'o3-mini',  # Add O3 Mini model
    'o3-mini-effort': 'o3-mini-2025-01-31',  # Add new O3 Mini model with effort
    'o4-mini': 'o4-mini',  # Add O4 Mini model
    'gpt4o': 'gpt-4o',
    'gpt41': 'gpt-4.1',  # Add GPT-4.1 Turbo model
    'deepseek-r1': 'deepseek/deepseek-r1',  # Add DeepSeek R1 model
    'ollama': 'deepseek-r1:7b',  # Add Ollama model
    'hyperbolic': 'deepseek-ai/DeepSeek-R1',  # Add Hyperbolic model
    'ollama-1.5b': 'deepseek-r1:1.5b',  # Add DeepSeek R1 1.5B model
    'ollama-8b': 'deepseek-r1:8b',  # Add DeepSeek R1 8B model
    'ollama-14b': 'deepseek-r1:14b',  # Add DeepSeek R1 14B model
    'ollama-32b': 'deepseek-r1:32b',  # Add DeepSeek R1 32B model
    'ollama-70b': 'deepseek-r1:70b',  # Add DeepSeek R1 70B model
    'ollama-671b': 'deepseek-r1:671b'  # Add DeepSeek R1 671B model
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
        # Debug prints removed for cleaner output
        # Logging can be used for any necessary debugging
        logging.debug("Initializing client")
        logging.debug(f"OPENAI_KEY exists: {bool(OPENAI_KEY)}")
        logging.debug(f"DEEPSEEK_KEY exists: {bool(DEEPSEEK_KEY)}")
        logging.debug(f"XAI_KEY exists: {bool(XAI_KEY)}")
        logging.debug(f"OPENROUTER_API_KEY exists: {bool(OPENROUTER_API_KEY)}")
        logging.debug(f"HYPERBOLIC_KEY exists: {bool(HYPERBOLIC_KEY)}")
        logging.debug(f"API_KEY_PERPS exists: {bool(API_KEY_PERPS)}")
        logging.debug(f"API_SECRET_PERPS exists: {bool(API_SECRET_PERPS)}")
        
        if use_ollama:
            logging.debug("Using Ollama - no client initialization needed")
            # For Ollama, we don't need to initialize a client
            return True
            
        if use_deepseek_r1:
            logging.debug("Using DeepSeek R1 via OpenRouter")
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
            logging.debug("OpenRouter client initialized successfully")
            return True
            
        provider = 'X AI' if use_grok else ('DeepSeek' if (use_deepseek or use_reasoner) else 'OpenAI')
        api_key = XAI_KEY if use_grok else (DEEPSEEK_KEY if (use_deepseek or use_reasoner) else OPENAI_KEY)
        
        logging.debug(f"Using provider: {provider}")
        
        if not api_key:
            logging.error(f"{provider} API key is not set in the configuration")
            print(f"{COLORS['red']}Error: {provider} API key is not set. Please add it to your configuration.{COLORS['end']}")
            return False
            
        if provider == 'OpenAI' and not api_key.startswith('sk-'):
            logging.error("Invalid OpenAI API key format")
            print(f"{COLORS['red']}Error: Invalid OpenAI API key format. Key should start with 'sk-'{COLORS['end']}")
            return False

        if use_grok:
            logging.debug("Initializing X AI client")
            client = OpenAI(api_key=XAI_KEY, base_url="https://api.x.ai/v1")
            logging.debug("X AI client initialized successfully")
        elif use_deepseek or use_reasoner:
            logging.debug("Initializing DeepSeek client")
            client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
            logging.debug("DeepSeek client initialized successfully")
        else:
            logging.debug("Initializing OpenAI client")
            client = OpenAI(api_key=OPENAI_KEY)
            logging.debug("OpenAI client initialized successfully")
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

def run_market_analysis(product_id: str, granularity: str, 
                     alignment_timeframe_1: Optional[str] = None,
                     alignment_timeframe_2: Optional[str] = None,
                     alignment_timeframe_3: Optional[str] = None) -> Optional[Dict]:
    """Run market analysis and return the output as a dictionary.
    
    Args:
        product_id: Trading pair to analyze
        granularity: Primary timeframe to analyze
        alignment_timeframe_1: Optional additional timeframe for alignment
        alignment_timeframe_2: Optional additional timeframe for alignment
        alignment_timeframe_3: Optional additional timeframe for alignment
    """
    results = {}
    
    # Collect all timeframes to analyze
    timeframes = [granularity]
    if alignment_timeframe_1:
        timeframes.append(alignment_timeframe_1)
    if alignment_timeframe_2:
        timeframes.append(alignment_timeframe_2)
    if alignment_timeframe_3:
        timeframes.append(alignment_timeframe_3)
    
    # Remove duplicates (if any)
    timeframes = list(set(timeframes))
    
    # Run primary timeframe analysis first
    try:
        with open(os.devnull, 'w') as devnull:
            result = subprocess.check_output(
                ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', granularity, '--console_logging', 'false'],
                text=True,
                stderr=devnull,
                timeout=60  # 1 minute timeout
            )
        results[granularity] = result
        
        # If alignment timeframes are provided, run analysis for each
        for tf in timeframes:
            if tf != granularity:  # Skip primary which was already analyzed
                logging.info(f"Running alignment analysis for {tf} timeframe")
                print(f"Running alignment analysis for {tf} timeframe...")
                try:
                    with open(os.devnull, 'w') as devnull:
                        tf_result = subprocess.check_output(
                            ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', tf, '--console_logging', 'false'],
                            text=True,
                            stderr=devnull,
                            timeout=60  # 1 minute timeout
                        )
                    results[tf] = tf_result
                except Exception as e:
                    logging.warning(f"Error in alignment analysis for {tf}: {str(e)}")
                    results[tf] = f"Error: {str(e)}"
        
        # Calculate timeframe alignment score if multiple timeframes
        alignment_score = 100 if len(results) <= 1 else calculate_alignment_score(results)
        
        return {
            'success': True, 
            'data': results[granularity],  # Main timeframe result
            'alignment_results': results,   # All results including alignment timeframes
            'alignment_score': alignment_score,  # 0-100 score for alignment
            'alignment_timeframes': timeframes  # List of all timeframes analyzed
        }
        
    except subprocess.TimeoutExpired:
        logging.error("Market analyzer timed out after 1 minute")
        return {'success': False, 'error': "Analysis timed out"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Market analyzer error: {str(e)}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logging.error(f"Unexpected error in market analysis: {str(e)}")
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}

def calculate_alignment_score(results: Dict[str, str]) -> int:
    """Calculate a timeframe alignment score based on multiple timeframe analyses.
    
    Scores range from 0 (completely misaligned) to 100 (perfect alignment)
    """
    try:
        # Extract market conditions from each timeframe's result
        directions = []
        
        for timeframe, result in results.items():
            # Skip results with errors
            if result.startswith("Error:"):
                continue
                
            if "BULLISH" in result.upper():
                directions.append(1)  # Bullish
            elif "BEARISH" in result.upper():
                directions.append(-1)  # Bearish
            else:
                directions.append(0)  # Neutral/Choppy
        
        # If we have no valid directions
        if not directions:
            return 50  # Default neutral score
            
        # Calculate alignment based on consistency of directions
        unique_directions = set(directions)
        
        if len(unique_directions) == 1:
            # All timeframes agree
            return 100
        elif len(unique_directions) == 2 and 0 in unique_directions:
            # Some timeframes are neutral, but no conflicting directions
            return 75
        elif len(unique_directions) == 2:
            # Some conflicting directions
            return 50
        else:
            # Completely conflicting (has positive, negative, and neutral)
            return 25
            
    except Exception as e:
        logging.error(f"Error calculating alignment score: {str(e)}")
        return 50  # Default to moderate score on error

def get_ollama_response(prompt: str, model: str = "deepseek-r1:7b", **kwargs) -> Optional[str]:
    """Get response from Ollama API."""
    try:
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        # Add deterministic parameters if provided
        if 'temperature' in kwargs:
            data["temperature"] = kwargs.get("temperature", 0.0)
        if 'top_p' in kwargs:
            data["top_p"] = kwargs.get("top_p", 0.0)
        if 'seed' in kwargs:
            data["seed"] = kwargs.get("seed", 42)
        
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

def get_hyperbolic_response(messages: list, model: str = "deepseek-ai/DeepSeek-R1", **kwargs) -> Optional[str]:
    """Get response from Hyperbolic API."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HYPERBOLIC_KEY}"
        }
        
        data = {
            "messages": messages,
            "model": model
        }
        
        # Add deterministic parameters if provided
        if 'temperature' in kwargs:
            data["temperature"] = kwargs.get("temperature", 0.0)
        if 'top_p' in kwargs:
            data["top_p"] = kwargs.get("top_p", 0.0)
        if 'seed' in kwargs:
            data["seed"] = kwargs.get("seed", 42)
        
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
def get_trading_recommendation(client: OpenAI, market_analysis: str, product_id: str, 
                              use_deepseek: bool = False, use_reasoner: bool = False, 
                              use_grok: bool = False, use_gpt45_preview: bool = False, 
                              use_o1: bool = False, use_o1_mini: bool = False, 
                              use_o3_mini: bool = False, use_o3_mini_effort: bool = False, 
                              use_o4_mini: bool = False, use_gpt4o: bool = False, use_gpt41: bool = False, 
                              use_deepseek_r1: bool = False, use_ollama: bool = False, use_ollama_1_5b: bool = False,
                              use_ollama_8b: bool = False, use_ollama_14b: bool = False,
                              use_ollama_32b: bool = False, use_ollama_70b: bool = False,
                              use_ollama_671b: bool = False, use_hyperbolic: bool = False,
                              alignment_score: int = 50) -> tuple[Optional[str], Optional[str]]:
    """
    Get trading recommendation with improved retry logic and debug output.
    
    This function enforces deterministic behavior using temperature=0, top_p=0 and a fixed seed=42
    for models that support these parameters. For models that don't support these parameters
    (like o4-mini), it falls back to using default values.
    
    Note that even with these settings, LLMs may produce slightly different outputs between runs,
    though critical trading fields (signal type, prices, stop loss) should remain consistent.
    """
    # Debug output moved to logging
    logging.debug("Starting trading recommendation request")
    """Get trading recommendation with improved retry logic."""
    if not (use_ollama or use_ollama_1_5b or use_ollama_8b or use_ollama_14b or 
            use_ollama_32b or use_ollama_70b or use_ollama_671b or use_hyperbolic) and client is None:
        raise ValueError("API client not properly initialized")

    # Add timeout parameter for API calls
    TIMEOUT = 90  # 90 seconds timeout

    # Define provider at the start of the function
    provider = 'Hyperbolic' if use_hyperbolic else ('Ollama' if (use_ollama or use_ollama_1_5b or 
              use_ollama_8b or use_ollama_14b or use_ollama_32b or use_ollama_70b or 
              use_ollama_671b) else ('OpenRouter' if use_deepseek_r1 else ('X AI' if use_grok else 
              ('DeepSeek' if (use_deepseek or use_reasoner) else 'OpenAI'))))

    # Models that don't support certain deterministic parameters
    # Based on error: "temperature does not support 0.0 with this model. Only the default (1) value is supported."
    models_without_deterministic_support = {
        'o4-mini': ['temperature', 'top_p']
    }

    # Check if the current model doesn't support deterministic parameters
    current_model_key = None
    if use_o4_mini:
        current_model_key = 'o4-mini'
    # Add other model checks as needed...
    
    # List of parameters not supported by the current model
    unsupported_params = []
    if current_model_key and current_model_key in models_without_deterministic_support:
        unsupported_params = models_without_deterministic_support[current_model_key]
        logging.warning(f"Model {current_model_key} doesn't support deterministic parameters: {', '.join(unsupported_params)}")
        print(f"{COLORS['yellow']}Note: Model {current_model_key} doesn't fully support deterministic parameters.{COLORS['end']}")
        print(f"{COLORS['yellow']}Using default values for: {', '.join(unsupported_params)}{COLORS['end']}")

    # SYSTEM_PROMPT = """
    # You are a professional crypto trading advisor with expertise in technical analysis and market psychology.

    # Reply only with a valid JSON object in a single line (without any markdown code block) representing one of the following signals:

    # For a SELL signal: 
    # {
    #     "SIGNAL_TYPE": "SELL",
    #     "SELL AT": <PRICE>,
    #     "BUY BACK AT": <PRICE>,
    #     "STOP LOSS": <PRICE>,
    #     "PROBABILITY": <PROBABILITY_0_TO_100>,
    #     "CONFIDENCE": "<CONFIDENCE>",
    #     "R/R_RATIO": <R/R_RATIO>,
    #     "VOLUME_STRENGTH": "<VOLUME_STRENGTH>",
    #     "VOLATILITY": "<VOLATILITY>",
    #     "MARKET_REGIME": "<MARKET_REGIME>",
    #     "REGIME_CONFIDENCE": "<REGIME_CONFIDENCE>",
    #     "TIMEFRAME_ALIGNMENT": <TIMEFRAME_ALIGNMENT_SCORE>,
    #     "REASONING": "<CONCISE_REASONING>",
    #     "IS_VALID": <IS_VALID>
    # }

    # For a BUY signal:
    # {
    #     "SIGNAL_TYPE": "BUY", 
    #     "BUY AT": <PRICE>,
    #     "SELL BACK AT": <PRICE>,
    #     "STOP LOSS": <PRICE>,
    #     "PROBABILITY": <PROBABILITY_0_TO_100>,
    #     "CONFIDENCE": "<CONFIDENCE>",
    #     "R/R_RATIO": <R/R_RATIO>,
    #     "VOLUME_STRENGTH": "<VOLUME_STRENGTH>",
    #     "VOLATILITY": "<VOLATILITY>",
    #     "MARKET_REGIME": "<MARKET_REGIME>",
    #     "REGIME_CONFIDENCE": "<REGIME_CONFIDENCE>",
    #     "TIMEFRAME_ALIGNMENT": <TIMEFRAME_ALIGNMENT_SCORE>,
    #     "REASONING": "<CONCISE_REASONING>",
    #     "IS_VALID": <IS_VALID>
    # }

    # For a HOLD recommendation when no trade is advisable:
    # {
    #     "SIGNAL_TYPE": "HOLD",
    #     "PRICE": <CURRENT_PRICE>,
    #     "PROBABILITY": <PROBABILITY_0_TO_100>,
    #     "CONFIDENCE": "<CONFIDENCE>",
    #     "MARKET_REGIME": "<MARKET_REGIME>",
    #     "REGIME_CONFIDENCE": "<REGIME_CONFIDENCE>",
    #     "REASONING": "<CONCISE_REASONING>",
    #     "IS_VALID": true
    # }

    # ## Market Analysis Requirements

    # 1. **Market Regime Analysis**
    # - Identify the current market regime as one of: 'Strong Bullish', 'Bullish', 'Choppy Bullish', 'Choppy', 'Choppy Bearish', 'Bearish', or 'Strong Bearish'
    # - Set REGIME_CONFIDENCE as: 'Very High', 'High', 'Moderate', 'Low', 'Very Low'
    # - Only generate BUY signals in Bullish or Strong Bullish regimes unless a clear reversal pattern is detected
    # - Only generate SELL signals in Bearish or Strong Bearish regimes unless a clear reversal pattern is detected
    # - Reversal patterns must have pattern confidence ≥ 70% and pattern completion ≥ 70%
    # - Reject any trade signal if pattern completion is < 70%
    # - In Choppy regimes, require PROBABILITY ≥ 80 and R/R_RATIO ≥ 2.0

    # 2. **Conflict Resolution Rules**
    # - If RSI > 70 and volume change < -50%, subtract 15 from PROBABILITY and downgrade CONFIDENCE by one level
    # - If RSI > 75 and volume is falling, override any BUY or SELL signal with HOLD unless all other indicators are strongly aligned
    # - If a detected pattern conflicts with the active trend or market regime, downgrade CONFIDENCE by one level

    # 3. **Risk/Reward Calculation**
    # - For a SELL signal: R/R_RATIO = (SELL AT - BUY BACK AT) / (STOP LOSS - SELL AT)
    # - For a BUY signal: R/R_RATIO = (SELL BACK AT - BUY AT) / (BUY AT - STOP LOSS)
    # - Ensure R/R ratio > 1.0 for valid trades, and > 2.0 in choppy regimes
    # - Target a realistic profit zone of 1.5–3% under most conditions

    # 4. **Timeframe Alignment**
    # - Provide a TIMEFRAME_ALIGNMENT score between 0–100
    # - 100 = all timeframes align in signal direction; 0 = fully conflicted timeframes

    # 5. **Validation Criteria**
    # - For SELL: STOP LOSS > SELL AT
    # - For BUY: STOP LOSS < BUY AT
    # - Set IS_VALID to false if these conditions are not met
    # - Return HOLD if IS_VALID = false or PROBABILITY < 50
    # - Provide a concise REASONING (30–50 words) that explains the signal

    # 6. **Rating System**
    # - CONFIDENCE: 'Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak'
    # - VOLUME_STRENGTH: 'Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak'
    # - VOLATILITY: 'Very Low', 'Low', 'Medium', 'High', 'Very High'
    # - PROBABILITY: 0–100 (% chance of success)

    # Remember to reply with only one JSON object (no markdown or explanation) and follow the format strictly.
    # """

    SYSTEM_PROMPT = """
    You are a professional crypto trading advisor with expertise in technical analysis and market psychology.

    Your task is to evaluate BOTH long and short opportunities based on the data. You must return ONE valid JSON object in a single line representing a BUY, SELL, or HOLD recommendation.

    Avoid directional bias. Assess both sides, then select the most favorable setup. If neither meets criteria, return HOLD.

    Reply only with a valid JSON object in a single line (without any markdown code block) representing one of the following signals:

    For a SELL signal: 
    {
        "SIGNAL_TYPE": "SELL",
        "SELL AT": <PRICE>,
        "BUY BACK AT": <PRICE>,
        "STOP LOSS": <PRICE>,
        "PROBABILITY": <PROBABILITY_0_TO_100>,
        "CONFIDENCE": "<CONFIDENCE>",
        "R/R_RATIO": <R/R_RATIO>,
        "VOLUME_STRENGTH": "<VOLUME_STRENGTH>",
        "VOLATILITY": "<VOLATILITY>",
        "MARKET_REGIME": "<MARKET_REGIME>",
        "REGIME_CONFIDENCE": "<REGIME_CONFIDENCE>",
        "TIMEFRAME_ALIGNMENT": <TIMEFRAME_ALIGNMENT_SCORE>,
        "REASONING": "<CONCISE_REASONING>",
        "IS_VALID": <IS_VALID>
    }

    For a BUY signal:
    {
        "SIGNAL_TYPE": "BUY", 
        "BUY AT": <PRICE>,
        "SELL BACK AT": <PRICE>,
        "STOP LOSS": <PRICE>,
        "PROBABILITY": <PROBABILITY_0_TO_100>,
        "CONFIDENCE": "<CONFIDENCE>",
        "R/R_RATIO": <R/R_RATIO>,
        "VOLUME_STRENGTH": "<VOLUME_STRENGTH>",
        "VOLATILITY": "<VOLATILITY>",
        "MARKET_REGIME": "<MARKET_REGIME>",
        "REGIME_CONFIDENCE": "<REGIME_CONFIDENCE>",
        "TIMEFRAME_ALIGNMENT": <TIMEFRAME_ALIGNMENT_SCORE>,
        "REASONING": "<CONCISE_REASONING>",
        "IS_VALID": <IS_VALID>
    }

    For a HOLD recommendation when no trade is advisable:
    {
        "SIGNAL_TYPE": "HOLD",
        "PRICE": <CURRENT_PRICE>,
        "PROBABILITY": <PROBABILITY_0_TO_100>,
        "CONFIDENCE": "<CONFIDENCE>",
        "MARKET_REGIME": "<MARKET_REGIME>",
        "REGIME_CONFIDENCE": "<REGIME_CONFIDENCE>",
        "REASONING": "<CONCISE_REASONING>",
        "IS_VALID": true
    }

    Key Adjustments:
    1. Market Regime Rules (revised)
    - BUY signals are favored in Bullish/Strong Bullish regimes  
    - SELL signals are favored in Bearish/Strong Bearish regimes  
    - In Choppy regimes, assess both sides. Choose the one with better PROBABILITY and R/R.  
    - Reversal trades are allowed in any regime if pattern confidence ≥ 75% and pattern completion ≥ 70%.

    2. Conflict Resolution (adjusted)
    - If RSI > 75 AND falling volume → downgrade confidence by one level  
    - Remove automatic override to HOLD  
    - Do not subtract from PROBABILITY unless multiple indicators conflict

    3. Risk/Reward + Filtering (unchanged)
    - Maintain R/R > 1.0 generally, > 2.0 in Choppy  
    - Reject trades with invalid STOP/ENTRY logic  
    - Return HOLD if PROBABILITY < 50 or IS_VALID = false

    4. Selection Logic (new rule)
    - After evaluating both BUY and SELL opportunities, return only the higher-quality signal  
    - Use REASONING field to explain why the chosen side is better

    All other formatting, validation, and scoring rules remain unchanged.
    """    
    
    try:
        logging.debug("Starting recommendation - checking model type")
        if use_hyperbolic:
            logging.debug("Using Hyperbolic API")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Here's the latest market analysis for {product_id}:\n{market_analysis}\nTimeframe alignment score: {alignment_score}/100\nBased on this analysis and the timeframe alignment score, provide a trading recommendation."}
            ]
            logging.debug("Calling Hyperbolic API...")
            # Set deterministic parameters for Hyperbolic
            hyperbolic_params = {}
            if 'temperature' not in unsupported_params:
                hyperbolic_params["temperature"] = 0
            if 'top_p' not in unsupported_params:
                hyperbolic_params["top_p"] = 0
            if 'seed' not in unsupported_params:
                hyperbolic_params["seed"] = 42
            
            recommendation = get_hyperbolic_response(messages, MODEL_CONFIG['hyperbolic'], **hyperbolic_params)
            logging.debug("Hyperbolic API response received")
            if recommendation is None:
                raise Exception("Failed to get response from Hyperbolic API")
            return recommendation, None
            
        if use_ollama or use_ollama_1_5b or use_ollama_8b or use_ollama_14b or use_ollama_32b or use_ollama_70b or use_ollama_671b:
            logging.debug("Using Ollama API")
            # Format prompt for Ollama
            full_prompt = f"{SYSTEM_PROMPT}\n\nHere's the latest market analysis for {product_id}:\n{market_analysis}\nTimeframe alignment score: {alignment_score}/100\nBased on this analysis and the timeframe alignment score, provide a trading recommendation."
            logging.debug("Calling Ollama API...")
            
            # Select the appropriate model size
            model_key = 'ollama'  # default 7B
            if use_ollama_1_5b:
                model_key = 'ollama-1.5b'
            elif use_ollama_8b:
                model_key = 'ollama-8b'
            elif use_ollama_14b:
                model_key = 'ollama-14b'
            elif use_ollama_32b:
                model_key = 'ollama-32b'
            elif use_ollama_70b:
                model_key = 'ollama-70b'
            elif use_ollama_671b:
                model_key = 'ollama-671b'
            
            # Deterministic parameters for Ollama
            ollama_params = {}
            if 'temperature' not in unsupported_params:
                ollama_params["temperature"] = 0.0
            if 'top_p' not in unsupported_params:
                ollama_params["top_p"] = 0.0
            if 'seed' not in unsupported_params:
                ollama_params["seed"] = 42
                
            recommendation = get_ollama_response(full_prompt, MODEL_CONFIG[model_key], **ollama_params)
            logging.debug("Ollama API response received")
            if recommendation is None:
                raise Exception("Failed to get response from Ollama API")
            return recommendation, None
            
        # Simplified model selection
        logging.debug("Using standard API model")
        model = MODEL_CONFIG['default']
        if use_grok:
            model = MODEL_CONFIG['grok']
        elif use_reasoner:
            model = MODEL_CONFIG['reasoner']
        elif use_deepseek:
            model = MODEL_CONFIG['deepseek']
        elif use_gpt45_preview:
            model = MODEL_CONFIG['gpt45-preview']
        elif use_o1:
            model = MODEL_CONFIG['o1']
        elif use_o1_mini:
            model = MODEL_CONFIG['o1-mini']
        elif use_o3_mini:
            model = MODEL_CONFIG['o3-mini']
        elif use_o3_mini_effort:
            model = MODEL_CONFIG['o3-mini-effort']
        elif use_o4_mini:
            model = MODEL_CONFIG['o4-mini']
        elif use_gpt4o:
            model = MODEL_CONFIG['gpt4o']
        elif use_gpt41:
            model = MODEL_CONFIG['gpt41']
        elif use_deepseek_r1:
            model = MODEL_CONFIG['deepseek-r1']
            
        logging.debug(f"Selected model: {model} from provider: {provider}")
        logging.info(f"Using model: {model} from provider: {provider}")
        
        # For models that don't support system messages, combine with user message
        messages = []
        user_content = f"Here's the latest market analysis for {product_id}:\n{market_analysis}\nTimeframe alignment score: {alignment_score}/100\nBased on this analysis and the timeframe alignment score, provide a trading recommendation."
        
        if model in [MODEL_CONFIG['o1-mini'], MODEL_CONFIG['o3-mini'], MODEL_CONFIG['o3-mini-effort'], MODEL_CONFIG['o4-mini']]:  # Add o4-mini to the list of models that need combined messages
            messages = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"}
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ]
            
        # Set up parameters for the API call
        params = {
            "model": model,
            "messages": messages,
            "timeout": TIMEOUT
        }

        # Add deterministic parameters if supported by the model
        if 'temperature' not in unsupported_params:
            params["temperature"] = 0.0  # Set temperature to 0 for deterministic output
        if 'top_p' not in unsupported_params:
            params["top_p"] = 0.0        # Set top_p to 0 for deterministic output
        if 'seed' not in unsupported_params:
            params["seed"] = 42           # Set fixed seed for deterministic output

        # Add reasoning_effort parameter for o3-mini-effort model
        if use_o3_mini_effort:
            params["reasoning_effort"] = "medium"
            
        # Log debug info before API call
        logging.debug(f"Making API request to {provider} with model {model}")
        logging.debug(f"Request timeout: {TIMEOUT}s")
        
        # Log which deterministic parameters are being used
        deterministic_params_used = []
        if 'temperature' in params:
            deterministic_params_used.append(f"temperature={params['temperature']}")
        if 'top_p' in params:
            deterministic_params_used.append(f"top_p={params['top_p']}")
        if 'seed' in params:
            deterministic_params_used.append(f"seed={params['seed']}")
            
        if deterministic_params_used:
            logging.debug(f"Using deterministic parameters: {', '.join(deterministic_params_used)}")
        else:
            logging.debug("No deterministic parameters set for this model")
        
        # Make the API call with timing
        import time
        start_time = time.time()
        logging.debug("Starting API call...")
        
        try:
            response = client.chat.completions.create(**params)
            end_time = time.time()
            logging.debug(f"API call completed in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"API call failed with error: {str(e)}")
            raise
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
        # Move debug output to logging
        logging.debug("FINISHED TRADING RECOMMENDATION REQUEST")
        logging.debug(f"Recommendation type: {type(recommendation)}")
        logging.debug(f"Recommendation length: {len(recommendation) if recommendation else 0}")
        logging.debug(f"Recommendation content: {recommendation}")
        
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
    # Clear divider to separate from any previous output
    print("\n\n")
    
    # Trade recommendation header
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{COLORS['cyan']}====== 🤖 AI TRADING RECOMMENDATION ({current_time}) ======{COLORS['end']}")
    print(f"{COLORS['cyan']}{'=' * 60}{COLORS['end']}")
    
    try:
        # Try to parse the recommendation as JSON for better formatting
        rec_dict = json.loads(recommendation.replace("'", '"'))
        
        # Get signal type
        signal_type = rec_dict.get('SIGNAL_TYPE', 'UNKNOWN')
        if signal_type == 'UNKNOWN':
            # Legacy format detection
            if 'SELL AT' in rec_dict:
                signal_type = 'SELL'
            elif 'BUY AT' in rec_dict:
                signal_type = 'BUY'
            elif 'HOLD' in rec_dict:
                signal_type = 'HOLD'
                
        # Color code based on signal type
        if signal_type == 'BUY':
            header_color = COLORS['green']
        elif signal_type == 'SELL':
            header_color = COLORS['red']
        else:  # HOLD or other
            header_color = COLORS['yellow']
            
        # Print signal header with extra visibility
        if signal_type == 'HOLD':
            print(f"\n{header_color}{'*' * 20}{COLORS['end']}")
            print(f"{header_color}𝗦𝗜𝗚𝗡𝗔𝗟: {signal_type}{COLORS['end']}")
            print(f"{header_color}{'*' * 20}{COLORS['end']}")
        else:
            print(f"\n{header_color}𝗦𝗜𝗚𝗡𝗔𝗟: {signal_type}{COLORS['end']}")
        
        # Print probability and confidence
        if 'PROBABILITY' in rec_dict:
            prob = float(rec_dict['PROBABILITY'])
            confidence = rec_dict.get('CONFIDENCE', 'N/A')
            print(f"{COLORS['bold']}Probability:{COLORS['end']} {prob:.1f}% | {COLORS['bold']}Confidence:{COLORS['end']} {confidence}")
        
        # Print pricing information based on signal type
        if signal_type == 'BUY':
            entry = float(rec_dict.get('BUY AT', 0))
            target = float(rec_dict.get('SELL BACK AT', 0))
            stop = float(rec_dict.get('STOP LOSS', 0))
            print(f"{COLORS['bold']}Entry:{COLORS['end']} ${entry:.2f} | {COLORS['bold']}Target:{COLORS['end']} ${target:.2f} | {COLORS['bold']}Stop:{COLORS['end']} ${stop:.2f}")
            
            if entry > 0 and stop > 0:
                risk_pct = abs((stop - entry) / entry * 100)
                reward_pct = abs((target - entry) / entry * 100)
                print(f"{COLORS['bold']}Risk:{COLORS['end']} {risk_pct:.2f}% | {COLORS['bold']}Reward:{COLORS['end']} {reward_pct:.2f}%")
        
        elif signal_type == 'SELL':
            entry = float(rec_dict.get('SELL AT', 0))
            target = float(rec_dict.get('BUY BACK AT', 0))
            stop = float(rec_dict.get('STOP LOSS', 0))
            print(f"{COLORS['bold']}Entry:{COLORS['end']} ${entry:.2f} | {COLORS['bold']}Target:{COLORS['end']} ${target:.2f} | {COLORS['bold']}Stop:{COLORS['end']} ${stop:.2f}")
            
            if entry > 0 and stop > 0:
                risk_pct = abs((stop - entry) / entry * 100)
                reward_pct = abs((target - entry) / entry * 100)
                print(f"{COLORS['bold']}Risk:{COLORS['end']} {risk_pct:.2f}% | {COLORS['bold']}Reward:{COLORS['end']} {reward_pct:.2f}%")
        
        elif signal_type == 'HOLD':
            price = rec_dict.get('PRICE', 'N/A')
            print(f"{COLORS['bold']}Current Price:{COLORS['end']} ${price}")
        
        # Print market regime information
        if 'MARKET_REGIME' in rec_dict:
            regime = rec_dict['MARKET_REGIME']
            regime_confidence = rec_dict.get('REGIME_CONFIDENCE', 'N/A')
            print(f"{COLORS['bold']}Market Regime:{COLORS['end']} {regime} ({regime_confidence} confidence)")
        
        # Print additional metrics
        metrics = []
        if 'R/R_RATIO' in rec_dict:
            rr_ratio = float(rec_dict['R/R_RATIO'])
            metrics.append(f"R/R Ratio: {rr_ratio:.2f}")
        
        if 'VOLUME_STRENGTH' in rec_dict:
            metrics.append(f"Volume: {rec_dict['VOLUME_STRENGTH']}")
            
        if 'VOLATILITY' in rec_dict:
            metrics.append(f"Volatility: {rec_dict['VOLATILITY']}")
            
        if 'TIMEFRAME_ALIGNMENT' in rec_dict:
            alignment = rec_dict['TIMEFRAME_ALIGNMENT']
            metrics.append(f"Timeframe Alignment: {alignment}/100")
            
        if metrics:
            print(f"{COLORS['bold']}Metrics:{COLORS['end']} {' | '.join(metrics)}")
        
        # Print reasoning if available in the structured response
        if 'REASONING' in rec_dict and rec_dict['REASONING']:
            print(f"\n{COLORS['cyan']}Reasoning:{COLORS['end']} {rec_dict['REASONING']}")
    
    except json.JSONDecodeError:
        # If parsing fails, fall back to printing the raw recommendation
        print(f"{COLORS['bold']}{recommendation}{COLORS['end']}")

    # Print reasoning from the model if available (separate from structured response)
    if reasoning:
        print(f"\n{COLORS['yellow']}====== 🧠 Model Reasoning ======{COLORS['end']}")
        # Format reasoning: limit to 100 words for conciseness
        words = reasoning.split()
        if len(words) > 100:
            print(' '.join(words[:100]) + '...')  # Add ellipsis for truncation
        else:
            print(reasoning)

    # Process and display market analysis data
    if 'data' in analysis_result:
        try:
            # Parse the market analysis data
            analysis_data = analysis_result['data']
            
            # Print market condition summary if available
            if 'market_condition' in analysis_data:
                condition = analysis_data['market_condition']
                print(f"\n{COLORS['cyan']}====== 📊 Market Condition ======{COLORS['end']}")
                print(f"• Trend: {condition.get('trend', 'N/A')}")
                print(f"• Momentum: {condition.get('momentum', 'N/A')}")
                print(f"• Volatility: {condition.get('volatility', 'N/A')}")
            
            # Print market alerts if available
            if 'alerts' in analysis_data:
                alerts = analysis_data['alerts']
                if alerts.get('triggered_alerts'):
                    print(f"\n{COLORS['cyan']}====== 🔔 Market Alerts ======{COLORS['end']}")
                    print("\n🚨 Active Alerts:")
                    for alert in alerts['triggered_alerts']:
                        print(f"• [{alert['priority']}] {alert['type']}: {alert['message']}")

            # Print key levels if available
            if 'key_levels' in analysis_data:
                print(f"\n{COLORS['cyan']}====== 🎯 Key Price Levels ======{COLORS['end']}")
                levels = analysis_data['key_levels']
                print(f"• Support: ${levels.get('support', 'N/A')}")
                print(f"• Resistance: ${levels.get('resistance', 'N/A')}")
                print(f"• Pivot: ${levels.get('pivot', 'N/A')}")
                
                # Print additional levels if available
                if 'fibonacci_levels' in levels:
                    print("\nFibonacci Levels:")
                    for level, value in levels['fibonacci_levels'].items():
                        print(f"  • {level}: ${value}")

            # Print risk metrics if available
            if 'risk_metrics' in analysis_data:
                print(f"\n{COLORS['cyan']}====== ⚠️ Risk Analysis ======{COLORS['end']}")
                risk = analysis_data['risk_metrics']
                print(f"• Risk Level: {risk.get('dynamic_risk', 0)*100:.1f}%")
                print(f"• Volatility: {risk.get('volatility', 0)*100:.1f}%")
                print(f"• Risk/Reward: {risk.get('risk_reward_ratio', 'N/A')}")
                
                if 'max_drawdown' in risk:
                    print(f"• Max Drawdown: {risk['max_drawdown']*100:.1f}%")
                    
                if 'optimal_position_size' in risk:
                    print(f"• Optimal Position: ${risk['optimal_position_size']:.2f}")
            
            # Print indicator values if available
            if 'indicators' in analysis_data:
                print(f"\n{COLORS['cyan']}====== 📈 Technical Indicators ======{COLORS['end']}")
                indicators = analysis_data['indicators']
                
                # Group indicators by type
                momentum_indicators = ['rsi', 'macd', 'stochastic', 'cci']
                trend_indicators = ['sma', 'ema', 'adx', 'dmi', 'ichimoku']
                volatility_indicators = ['bollinger_bands', 'atr', 'standard_deviation']
                
                # Print momentum indicators
                momentum_vals = {k: v for k, v in indicators.items() if k in momentum_indicators and v is not None}
                if momentum_vals:
                    print("Momentum Indicators:")
                    for name, value in momentum_vals.items():
                        if isinstance(value, dict):
                            # Handle complex indicators like MACD
                            values_str = ", ".join([f"{subkey}: {subval}" for subkey, subval in value.items()])
                            print(f"  • {name.upper()}: {values_str}")
                        else:
                            print(f"  • {name.upper()}: {value}")
                
                # Print trend indicators
                trend_vals = {k: v for k, v in indicators.items() if k in trend_indicators and v is not None}
                if trend_vals:
                    print("\nTrend Indicators:")
                    for name, value in trend_vals.items():
                        if isinstance(value, dict):
                            values_str = ", ".join([f"{subkey}: {subval}" for subkey, subval in value.items()])
                            print(f"  • {name.upper()}: {values_str}")
                        else:
                            print(f"  • {name.upper()}: {value}")
                
                # Print volatility indicators
                vol_vals = {k: v for k, v in indicators.items() if k in volatility_indicators and v is not None}
                if vol_vals:
                    print("\nVolatility Indicators:")
                    for name, value in vol_vals.items():
                        if isinstance(value, dict):
                            values_str = ", ".join([f"{subkey}: {subval}" for subkey, subval in value.items()])
                            print(f"  • {name.upper()}: {values_str}")
                        else:
                            print(f"  • {name.upper()}: {value}")

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
        'HYPERBOLIC_KEY': HYPERBOLIC_KEY,
        'API_KEY_PERPS': API_KEY_PERPS,
        'API_SECRET_PERPS': API_SECRET_PERPS
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

def execute_trade(recommendation: str, product_id: str, margin: float = 100, leverage: int = 20, use_limit_order: bool = False) -> None:
    """
    Execute trade based on model recommendation and advanced trade filters.
    
    Args:
        recommendation: JSON string containing the trading recommendation
        product_id: Trading pair to trade (e.g. BTC-USDC)
        margin: Initial margin amount in USD
        leverage: Leverage multiplier (1-20)
        use_limit_order: Whether to use limit orders instead of market orders
    """
    try:
        # Parse the JSON recommendation
        rec_dict = json.loads(recommendation.replace("'", '"'))  # Convert single quotes to double quotes
        
        # Check if it's a HOLD recommendation
        if rec_dict.get('SIGNAL_TYPE') == 'HOLD':
            print(f"{COLORS['yellow']}HOLD recommendation: {rec_dict.get('REASONING', 'No trade advised at this time.')}{COLORS['end']}")
            return
        
        # Extract probability and signal type
        prob = float(rec_dict['PROBABILITY'])
        side = rec_dict.get('SIGNAL_TYPE', 'UNKNOWN')
        
        if side == 'UNKNOWN':
            # Legacy format compatibility
            side = 'SELL' if 'SELL AT' in rec_dict else 'BUY'
        
        # Display reasoning if available
        if 'REASONING' in rec_dict:
            print(f"{COLORS['cyan']}Signal reasoning: {rec_dict['REASONING']}{COLORS['end']}")
        
        # Check validity flag
        if rec_dict.get('IS_VALID', True) is False:
            print(f"{COLORS['red']}Trade not executed: Model flagged recommendation as invalid{COLORS['end']}")
            return
        
        # Check probability threshold
        if prob < 65:
            print(f"{COLORS['yellow']}Trade not executed: Probability {prob:.1f}% is below threshold of 65%{COLORS['end']}")
            return
            
        # Check market regime conditions
        market_regime = rec_dict.get('MARKET_REGIME', 'Choppy')
        regime_confidence = rec_dict.get('REGIME_CONFIDENCE', 'Low')
        
        # Check timeframe alignment if available
        timeframe_alignment = rec_dict.get('TIMEFRAME_ALIGNMENT', 50)
        if timeframe_alignment < 60:
            print(f"{COLORS['yellow']}Trade not executed: Timeframe alignment score ({timeframe_alignment}) is below threshold of 60{COLORS['end']}")
            return
            
        # Validate trade direction against market regime
        if side == 'BUY' and market_regime in ['Bearish', 'Strong Bearish', 'Choppy Bearish']:
            if regime_confidence in ['High', 'Very High'] and prob < 85:
                print(f"{COLORS['red']}Trade not executed: {side} signal in {market_regime} regime requires >85% probability (current: {prob:.1f}%){COLORS['end']}")
                return
        elif side == 'SELL' and market_regime in ['Bullish', 'Strong Bullish', 'Choppy Bullish']:
            if regime_confidence in ['High', 'Very High'] and prob < 85:
                print(f"{COLORS['red']}Trade not executed: {side} signal in {market_regime} regime requires >85% probability (current: {prob:.1f}%){COLORS['end']}")
                return
                
        # Adjust thresholds for choppy markets
        if 'Choppy' in market_regime:
            if prob < 80:
                print(f"{COLORS['yellow']}Trade not executed: Probability {prob:.1f}% is below choppy market threshold of 80%{COLORS['end']}")
                return
            if float(rec_dict['R/R_RATIO']) < 2.0:
                print(f"{COLORS['yellow']}Trade not executed: R/R ratio {float(rec_dict['R/R_RATIO']):.2f} is below choppy market threshold of 2.0{COLORS['end']}")
                return
            
            # Increase timeframe alignment requirement for choppy markets
            if timeframe_alignment < 75:
                print(f"{COLORS['yellow']}Trade not executed: Timeframe alignment score ({timeframe_alignment}) is below choppy market threshold of 75{COLORS['end']}")
                return
        
        # Check volatility conditions
        volatility = rec_dict.get('VOLATILITY', 'Medium')
        if volatility in ['Very High']:
            if prob < 85:
                print(f"{COLORS['yellow']}Trade not executed: Market volatility '{volatility}' requires probability >85% (current: {prob:.1f}%){COLORS['end']}")
                return
        elif volatility in ['Very Low']:
            print(f"{COLORS['yellow']}Trade not executed: Market volatility '{volatility}' is too low for reliable execution{COLORS['end']}")
            return
        
        # Check volume strength
        volume_strength = rec_dict['VOLUME_STRENGTH']
        if volume_strength in ['Very Weak']:
            print(f"{COLORS['red']}Trade not executed: Volume strength '{volume_strength}' is too low{COLORS['end']}")
            return
            
        # Check confidence level
        confidence = rec_dict['CONFIDENCE']
        if confidence in ['Weak', 'Very Weak']:
            print(f"{COLORS['yellow']}Trade not executed: Signal confidence '{confidence}' is too low{COLORS['end']}")
            return
            
        # Check R/R ratio thresholds
        rr_ratio = float(rec_dict['R/R_RATIO'])
        if rr_ratio < 1.0:
            print(f"{COLORS['yellow']}Trade not executed: R/R ratio {rr_ratio:.3f} is below minimum threshold of 1.0{COLORS['end']}")
            return
        
        if rr_ratio > 5.0:
            # Very high R/R ratios often indicate unrealistic targets
            if prob < 85:
                print(f"{COLORS['yellow']}Trade not executed: R/R ratio {rr_ratio:.3f} is above maximum threshold of 5.0 (requires probability >85%){COLORS['end']}")
                return
            
        # Determine trade direction and prices
        if side == 'SELL':
            entry_price = float(rec_dict['SELL AT'])
            target_price = float(rec_dict['BUY BACK AT'])
            stop_loss = float(rec_dict['STOP LOSS'])
        elif side == 'BUY':
            entry_price = float(rec_dict['BUY AT'])
            target_price = float(rec_dict['SELL BACK AT'])
            stop_loss = float(rec_dict['STOP LOSS'])
        else:
            print(f"{COLORS['red']}Invalid signal type: {side}{COLORS['end']}")
            return

        try:
            # Initialize CoinbaseService with API keys
            cb_service = CoinbaseService(API_KEY_PERPS, API_SECRET_PERPS)
            
            # Get current price from market trades (most accurate real-time price)
            trades = cb_service.client.get_market_trades(product_id=product_id, limit=1)
            current_price = float(trades['trades'][0]['price'])
            
            # Calculate price deviation percentage
            price_deviation = abs((entry_price - current_price) / current_price * 100)
            
            # Determine whether to use market or limit order based on price conditions
            should_use_market = False
            # Adaptive price threshold based on volatility
            PRICE_THRESHOLD = 0.2  # Default 0.2% threshold
            if volatility == 'High':
                PRICE_THRESHOLD = 0.3  # Increase threshold to 0.3% for high volatility
            elif volatility == 'Very High':
                PRICE_THRESHOLD = 0.5  # Increase threshold to 0.5% for very high volatility
            
            if use_limit_order:
                # For limit orders, check if price conditions are favorable
                if side == 'SELL':
                    if entry_price < current_price:
                        # Entry price is below current price, unfavorable for limit SELL
                        print(f"{COLORS['yellow']}Switching to market order: Limit SELL price (${entry_price:.2f}) is below current price (${current_price:.2f}){COLORS['end']}")
                        should_use_market = True
                    elif price_deviation <= PRICE_THRESHOLD:
                        # Entry price is very close to current price, use market for immediate execution
                        print(f"{COLORS['yellow']}Switching to market order: Limit price deviation ({price_deviation:.2f}%) is below threshold ({PRICE_THRESHOLD}%){COLORS['end']}")
                        should_use_market = True
                elif side == 'BUY':
                    if entry_price > current_price:
                        # Entry price is above current price, unfavorable for limit BUY
                        print(f"{COLORS['yellow']}Switching to market order: Limit BUY price (${entry_price:.2f}) is above current price (${current_price:.2f}){COLORS['end']}")
                        should_use_market = True
                    elif price_deviation <= PRICE_THRESHOLD:
                        # Entry price is very close to current price, use market for immediate execution
                        print(f"{COLORS['yellow']}Switching to market order: Limit price deviation ({price_deviation:.2f}%) is below threshold ({PRICE_THRESHOLD}%){COLORS['end']}")
                        should_use_market = True
                
                if should_use_market and price_deviation > PRICE_THRESHOLD:
                    # If using market order but price deviation is too high, don't execute
                    print(f"{COLORS['red']}Market order not executed: Price deviation ({price_deviation:.2f}%) exceeds threshold ({PRICE_THRESHOLD}%){COLORS['end']}")
                    return
            else:
                # If market order is specified, check if price deviation is acceptable
                should_use_market = True
                if price_deviation > PRICE_THRESHOLD:
                    print(f"{COLORS['red']}Market order not executed: Price deviation ({price_deviation:.2f}%) exceeds threshold ({PRICE_THRESHOLD}%){COLORS['end']}")
                    return
                
        except Exception as e:
            print(f"{COLORS['red']}Error getting current price: {str(e)}{COLORS['end']}")
            return
            
        # Calculate stop loss percentage
        stop_loss_pct = abs((stop_loss - entry_price) / entry_price * 100)
        
        # Calculate potential profit percentage
        profit_pct = abs((target_price - entry_price) / entry_price * 100)
        
        # Check if stop loss is less than minimum threshold
        if stop_loss_pct < 0.5:
            print(f"{COLORS['red']}Trade not executed: Stop loss percentage ({stop_loss_pct:.2f}%) is less than minimum threshold of 0.5%{COLORS['end']}")
            return
        
        # Validate stop loss direction based on trade direction
        if side == 'SELL':
            if stop_loss <= entry_price:
                print(f"{COLORS['red']}Invalid stop loss for SELL order: Stop loss (${stop_loss:.2f}) must be above entry price (${entry_price:.2f}){COLORS['end']}")
                return
            if target_price >= entry_price:
                print(f"{COLORS['red']}Invalid take profit for SELL order: Take profit (${target_price:.2f}) must be below entry price (${entry_price:.2f}){COLORS['end']}")
                return
        else:  # BUY
            if stop_loss >= entry_price:
                print(f"{COLORS['red']}Invalid stop loss for BUY order: Stop loss (${stop_loss:.2f}) must be below entry price (${entry_price:.2f}){COLORS['end']}")
                return
            if target_price <= entry_price:
                print(f"{COLORS['red']}Invalid take profit for BUY order: Take profit (${target_price:.2f}) must be above entry price (${entry_price:.2f}){COLORS['end']}")
                return

        # Validate risk-reward ratio
        if stop_loss_pct > profit_pct:
            print(f"{COLORS['red']}Invalid risk-reward: Stop loss distance ({stop_loss_pct:.2f}%) is larger than take profit distance ({profit_pct:.2f}%){COLORS['end']}")
            return
            
        # NEW: Check for conservative targets (3-5% profit target maximum)
        MAX_PROFIT_TARGET = 5.0  # Maximum 5% profit target
        if profit_pct > MAX_PROFIT_TARGET:
            # Calculate a more conservative target price
            if side == 'BUY':
                conservative_target = entry_price * (1 + (MAX_PROFIT_TARGET / 100))
                print(f"{COLORS['yellow']}Adjusting to conservative target: Original target ${target_price:.2f} ({profit_pct:.2f}%) → ${conservative_target:.2f} ({MAX_PROFIT_TARGET:.2f}%){COLORS['end']}")
                target_price = conservative_target
            else:  # SELL
                conservative_target = entry_price * (1 - (MAX_PROFIT_TARGET / 100))
                print(f"{COLORS['yellow']}Adjusting to conservative target: Original target ${target_price:.2f} ({profit_pct:.2f}%) → ${conservative_target:.2f} ({MAX_PROFIT_TARGET:.2f}%){COLORS['end']}")
                target_price = conservative_target
            
            # Recalculate profit percentage with conservative target
            profit_pct = MAX_PROFIT_TARGET
            
            # Recalculate R/R ratio with new target
            if side == 'BUY':
                rr_ratio = (target_price - entry_price) / (entry_price - stop_loss)
            else:  # SELL
                rr_ratio = (entry_price - target_price) / (stop_loss - entry_price)
                
            print(f"{COLORS['yellow']}Updated R/R ratio with conservative target: {rr_ratio:.2f}{COLORS['end']}")
            
            # If new R/R ratio is too low, don't execute
            if rr_ratio < 1.0:
                print(f"{COLORS['red']}Trade not executed: Conservative target results in R/R ratio below 1.0{COLORS['end']}")
                return

        # Calculate position size with improved risk management
        base_size_usd = margin * leverage
        size_usd = base_size_usd  # Default starting position size
        position_sizing_message = ""
        
        # Advanced position sizing rules based on multiple factors
        sizing_factor = 1.0  # Start with full position size
        
        # 1. Adjust based on stop loss percentage
        if stop_loss_pct <= 1.5:
            # Tight stop loss allows for full position
            position_sizing_message = f"{COLORS['green']}Full position size - Tight stop loss ≤ 1.5%{COLORS['end']}"
        elif stop_loss_pct <= 3.0:
            # Medium stop loss - reduce by 20%
            sizing_factor *= 0.8
            position_sizing_message = f"{COLORS['yellow']}Reducing position by 20% - Medium stop loss ({stop_loss_pct:.2f}%){COLORS['end']}"
        else:
            # Wide stop loss - reduce by 40%
            sizing_factor *= 0.6
            position_sizing_message = f"{COLORS['yellow']}Reducing position by 40% - Wide stop loss ({stop_loss_pct:.2f}%){COLORS['end']}"
            
        # 2. Adjust based on market regime
        if 'Choppy' in market_regime:
            sizing_factor *= 0.8
            position_sizing_message += f"\n{COLORS['yellow']}Additional 20% reduction due to choppy market{COLORS['end']}"
        elif 'Strong' in market_regime:
            if ('Bullish' in market_regime and side == 'BUY') or ('Bearish' in market_regime and side == 'SELL'):
                # Strong trend in favorable direction - maintain position size
                pass
            else:
                # Counter-trend trade in strong trend - reduce position
                sizing_factor *= 0.7
                position_sizing_message += f"\n{COLORS['yellow']}Counter-trend trade - reducing position by 30%{COLORS['end']}"
                
        # 3. Adjust based on timeframe alignment
        if timeframe_alignment >= 90:
            # Excellent alignment - can increase position slightly
            sizing_factor *= 1.1
            position_sizing_message += f"\n{COLORS['green']}Increasing position by 10% - High timeframe alignment ({timeframe_alignment}){COLORS['end']}"
        elif timeframe_alignment < 70:
            # Poor alignment - reduce position
            sizing_factor *= 0.9
            position_sizing_message += f"\n{COLORS['yellow']}Reducing position by 10% - Low timeframe alignment ({timeframe_alignment}){COLORS['end']}"
            
        # Apply total position sizing adjustment
        size_usd = base_size_usd * sizing_factor
        
        # Cap position size at base size in case multiple positive adjustments exceed 100%
        size_usd = min(size_usd, base_size_usd)
        
        # Ensure minimum position size (at least 30% of base size)
        size_usd = max(size_usd, base_size_usd * 0.3)

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

        # Calculate potential profit/loss in USD based on adjusted position size
        profit_usd = size_usd * (profit_pct / 100)
        loss_usd = size_usd * (stop_loss_pct / 100)
        
        # Format prices according to asset precision
        cmd_target_price = format_price_by_asset(target_price, product_id)
        cmd_stop_loss = format_price_by_asset(stop_loss, product_id)
        cmd_entry_price = format_price_by_asset(entry_price, product_id)
        
        # Execute trade using subprocess
        cmd = [
            'python', 'trade_btc_perp.py',
            '--product', perp_product,
            '--side', side,
            '--size', str(size_usd),
            '--leverage', str(leverage),
            '--tp', str(cmd_target_price),
            '--sl', str(cmd_stop_loss)
        ]
        
        # Add limit price only if using limit order and conditions are met
        if use_limit_order and not should_use_market:
            cmd.extend(['--limit', str(cmd_entry_price)])
        
        # Add no-confirm flag
        cmd.append('--no-confirm')
        
        # Print trade details with enhanced information
        print(f"\n{COLORS['cyan']}===== Trade Execution Summary ====={COLORS['end']}")
        print(f"{COLORS['bold']}Asset & Market:{COLORS['end']}")
        print(f"  Product: {perp_product}")
        print(f"  Market Regime: {market_regime} (Confidence: {regime_confidence})")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Volatility: {volatility}")
        
        print(f"\n{COLORS['bold']}Trade Parameters:{COLORS['end']}")
        print(f"  Side: {side}")
        print(f"  Entry Price: ${entry_price:.2f} (Deviation: {price_deviation:.2f}%)")
        print(f"  Take Profit: ${target_price:.2f} ({profit_pct:.2f}% / ${profit_usd:.2f})")
        print(f"  Stop Loss: ${stop_loss:.2f} ({stop_loss_pct:.2f}% / ${loss_usd:.2f})")
        print(f"  Order Type: {'Limit' if (use_limit_order and not should_use_market) else 'Market'}")
        
        print(f"\n{COLORS['bold']}Risk Management:{COLORS['end']}")
        print(f"  Initial Margin: ${margin:.2f}")
        print(f"  Leverage: {leverage}x")
        print(f"  Position Size: ${size_usd:.2f} ({(sizing_factor * 100):.0f}% of max)")
        print(position_sizing_message)
        
        print(f"\n{COLORS['bold']}Signal Metrics:{COLORS['end']}")
        print(f"  Probability: {prob:.1f}%")
        print(f"  R/R Ratio: {rr_ratio:.3f}")
        print(f"  Signal Confidence: {confidence}")
        print(f"  Volume Strength: {volume_strength}")
        print(f"  Timeframe Alignment: {timeframe_alignment}/100")
        
        # Execute the trade
        print(f"\n{COLORS['cyan']}Executing trade...{COLORS['end']}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"\n{COLORS['green']}✅ Trade executed successfully!{COLORS['end']}")
            
            # Record trade to history file
            record_trade_to_history(side, entry_price, target_price, stop_loss, 
                                   prob, rr_ratio, product_id, market_regime)
                                   
            # Record trade to trade_output.txt with the requested format
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            execution_summary = ""
            
            # Add order execution information to the summary
            if use_limit_order and not should_use_market:
                execution_summary += f"Limit order placed at ${entry_price:.2f}\n"
            else:
                # Switching to market order messages for both BUY and SELL directions
                if side == 'SELL':
                    execution_summary += f"Switching to market order: Limit price (${entry_price:.2f}) is below or close to current price (${current_price:.2f})\n"
                else:  # BUY
                    execution_summary += f"Switching to market order: Limit price (${entry_price:.2f}) is above or close to current price (${current_price:.2f})\n"
                
            # Add execution parameters
            execution_summary += f"\nExecuting trade with parameters:\n"
            execution_summary += f"Product: {perp_product}\n"
            execution_summary += f"Side: {side}\n"
            execution_summary += f"Market Regime: {market_regime}\n"
            execution_summary += f"Regime Confidence: {regime_confidence}\n"
            execution_summary += f"Initial Margin: ${margin:.2f}\n"
            execution_summary += f"Leverage: {leverage}x\n"
            
            # Add position sizing info
            sizing_message = "Using standard position sizing (full margin)"
            if stop_loss_pct <= 1.5:
                sizing_message += " - SL ≤ 1.5%"
            elif stop_loss_pct <= 3.0:
                sizing_message += " - Medium SL"
            else:
                sizing_message += " - Wide SL"
            execution_summary += f"{sizing_message}\n"
            
            # Add trade details
            execution_summary += f"Position Size: ${size_usd:.2f}\n"
            execution_summary += f"Entry Price: ${entry_price:.2f}\n"
            execution_summary += f"Current Price: ${current_price:.2f}\n"
            execution_summary += f"Price Deviation: {price_deviation:.2f}%\n"
            execution_summary += f"Take Profit: ${target_price:.2f} ({profit_pct:.2f}% / ${profit_usd:.2f})\n"
            execution_summary += f"Stop Loss: ${stop_loss:.2f} ({stop_loss_pct:.2f}% / ${loss_usd:.2f})\n"
            execution_summary += f"Probability: {prob:.1f}%\n"
            execution_summary += f"Signal Confidence: {confidence}\n"
            execution_summary += f"R/R Ratio: {rr_ratio:.3f}\n"
            execution_summary += f"Volume Strength: {volume_strength}\n"
            execution_summary += f"Order Type: {'Limit' if (use_limit_order and not should_use_market) else 'Market'}\n"
            
            # Add the order summary from the output
            if result.stdout.strip():
                execution_summary += f"\n{result.stdout.strip()}\n"
                
            # Add the order placed message
            execution_summary += "\nOrder placed successfully!\n"
            
            # Record to trade_output.txt
            record_trade_to_output_txt(recommendation, timestamp, execution_summary)
            
            # Print output to console as before
            if result.stdout.strip():
                print(result.stdout)
        else:
            print(f"\n{COLORS['red']}❌ Error executing trade:{COLORS['end']}")
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

def format_price_by_asset(price: float, product_id: str) -> str:
    """Format price according to asset precision rules"""
    if product_id == 'BTC-USDC':
        return str(int(price))  # BTC uses whole numbers
    elif product_id == 'ETH-USDC':
        return f"{price:.1f}"  # ETH uses 1 decimal place
    elif product_id == 'SOL-USDC':
        return f"{price:.2f}"  # SOL uses 2 decimal places
    elif product_id in ['DOGE-USDC', 'SHIB-USDC']:
        return f"{price:.6f}"  # DOGE and SHIB use 6 decimal places
    else:
        return str(price)  # Default formatting

def record_trade_to_output_txt(recommendation: str, timestamp: str, execution_summary: str) -> None:
    """
    Record trade details to trade_output.txt in the requested format.
    
    Args:
        recommendation: JSON string containing the trading recommendation
        timestamp: Timestamp string in format YYYY-MM-DD HH:MM:SS
        execution_summary: Order execution summary text
    """
    try:
        with open('trade_output.txt', 'a') as file:
            # Add the header and recommendation in the requested format
            file.write(f"\n====== 🤖 AI Trading Recommendation ({timestamp}) ======\n")
            file.write(f"{recommendation}\n")
            file.write(execution_summary)
            file.write("\n\n")
        
        logging.info(f"Trade recorded to trade_output.txt: {timestamp}")
    except Exception as e:
        logging.error(f"Error recording trade to trade_output.txt: {str(e)}")
        # Don't raise the exception since this is a non-critical operation

def record_trade_to_history(side: str, entry_price: float, target_price: float, 
                           stop_loss: float, probability: float, rr_ratio: float,
                           product_id: str, market_regime: str) -> None:
    """Record trade details to a CSV history file for later analysis"""
    try:
        # Create a timestamp for the trade
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create the CSV file if it doesn't exist
        file_exists = os.path.isfile('trade_history.csv')
        
        with open('trade_history.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(['Timestamp', 'Product', 'Side', 'Entry Price', 'Target Price', 
                                 'Stop Loss', 'Probability', 'R/R Ratio', 'Market Regime', 'Status'])
            
            # Write trade data
            writer.writerow([
                timestamp, 
                product_id, 
                side, 
                f"{entry_price:.2f}", 
                f"{target_price:.2f}", 
                f"{stop_loss:.2f}", 
                f"{probability:.1f}", 
                f"{rr_ratio:.2f}", 
                market_regime, 
                'OPEN'  # Initial status is OPEN
            ])
            
        logging.info(f"Trade recorded to history: {side} {product_id} at {entry_price:.2f}")
    except Exception as e:
        logging.error(f"Error recording trade to history: {str(e)}")
        # Don't raise the exception since this is a non-critical operation

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='''
        AI-Powered Crypto Market Analyzer and Trading Recommendation System

        This tool analyzes cryptocurrency market data using various AI models and provides trading recommendations.
        It supports multiple AI providers including OpenAI, DeepSeek, X AI (Grok), and others.
        The analysis includes technical indicators, market sentiment, and risk metrics.

        Example usage:
          python prompt_market.py --product_id BTC-USDC --granularity ONE_HOUR
          python prompt_market.py --use_deepseek --margin 200 --leverage 10
          python prompt_market.py --use_grok --execute_trades
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Trading parameters
    trading_group = parser.add_argument_group('Trading Parameters')
    trading_group.add_argument('--product_id', type=str, default='BTC-USDC',
                        help='Trading pair to analyze (e.g., BTC-USDC, ETH-USDC, SOL-USDC, DOGE-USDC, SHIB-USDC)')
    trading_group.add_argument('--granularity', type=str, default='ONE_HOUR',
                        help='Time granularity for analysis (ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, TWO_HOUR, SIX_HOUR, ONE_DAY)')
    trading_group.add_argument('--margin', type=float, default=100,
                        help='Initial margin amount in USD for trading (default: 100)')
    trading_group.add_argument('--leverage', type=int, default=20,
                        help='Leverage multiplier for trading (default: 20, range: 1-20)')
    trading_group.add_argument('--execute_trades', action='store_true',
                        help='Execute trades automatically when probability exceeds 60%% and other conditions are met')
    trading_group.add_argument('--limit_order', action='store_true',
                        help='Use limit orders instead of market orders, using the SELL_AT or BUY_AT price from the recommendation')

    # AI Model Selection
    model_group = parser.add_argument_group('AI Model Selection (choose one)')
    model_group.add_argument('--use_deepseek', action='store_true',
                        help='Use DeepSeek Chat API for market analysis')
    model_group.add_argument('--use_reasoner', action='store_true',
                        help='Use DeepSeek Reasoner API (includes detailed reasoning steps)')
    model_group.add_argument('--use_grok', action='store_true',
                        help='Use X AI Grok API for analysis')
    model_group.add_argument('--use_gpt45_preview', action='store_true',
                        help='Use GPT-4.5 Preview model for analysis')
    model_group.add_argument('--use_o1', action='store_true',
                        help='Use o1 model for analysis')
    model_group.add_argument('--use_o1_mini', action='store_true',
                        help='Use o1 Mini model for faster, lighter analysis')
    model_group.add_argument('--use_o3_mini', action='store_true',
                        help='Use o3 Mini model for enhanced performance')
    model_group.add_argument('--use_o3_mini_effort', action='store_true',
                        help='Use o3-mini-2025-01-31 model with medium reasoning effort')
    model_group.add_argument('--use_o4_mini', action='store_true',
                        help='Use O4-mini model for advanced reasoning')
    model_group.add_argument('--use_gpt4o', action='store_true',
                        help='Use GPT-4o model for advanced analysis')
    model_group.add_argument('--use_gpt41', action='store_true',
                        help='Use GPT-4.1 Turbo model for enhanced analysis')
    model_group.add_argument('--use_deepseek_r1', action='store_true',
                        help='Use DeepSeek R1 model via OpenRouter API')
    model_group.add_argument('--use_ollama', action='store_true',
                        help='Use local Ollama model (7B) for offline analysis')
    model_group.add_argument('--use_ollama_1_5b', action='store_true',
                        help='Use local Ollama DeepSeek R1 1.5B model')
    model_group.add_argument('--use_ollama_8b', action='store_true',
                        help='Use local Ollama DeepSeek R1 8B model')
    model_group.add_argument('--use_ollama_14b', action='store_true',
                        help='Use local Ollama DeepSeek R1 14B model')
    model_group.add_argument('--use_ollama_32b', action='store_true',
                        help='Use local Ollama DeepSeek R1 32B model')
    model_group.add_argument('--use_ollama_70b', action='store_true',
                        help='Use local Ollama DeepSeek R1 70B model')
    model_group.add_argument('--use_ollama_671b', action='store_true',
                        help='Use local Ollama DeepSeek R1 671B model')
    model_group.add_argument('--use_hyperbolic', action='store_true',
                        help='Use Hyperbolic API for market analysis')
                        
    # Timeframe Alignment Options
    alignment_group = parser.add_argument_group('Timeframe Alignment (optional)')
    alignment_group.add_argument('--alignment_timeframe_1', type=str,
                        help='Additional timeframe #1 for alignment (e.g., ONE_HOUR)')
    alignment_group.add_argument('--alignment_timeframe_2', type=str,
                        help='Additional timeframe #2 for alignment (e.g., FIFTEEN_MINUTE)')
    alignment_group.add_argument('--alignment_timeframe_3', type=str,
                        help='Additional timeframe #3 for alignment (e.g., FIVE_MINUTE)')

    args = parser.parse_args()

    if sum([args.use_deepseek, args.use_reasoner, args.use_grok, args.use_gpt45_preview, 
            args.use_o1, args.use_o1_mini, args.use_o3_mini, args.use_o3_mini_effort, 
            args.use_o4_mini, args.use_gpt4o, args.use_gpt41, args.use_deepseek_r1, args.use_ollama, args.use_ollama_1_5b,
            args.use_ollama_8b, args.use_ollama_14b, args.use_ollama_32b, args.use_ollama_70b,
            args.use_ollama_671b, args.use_hyperbolic]) > 1:
        print("Please choose only one of --use_deepseek, --use_reasoner, --use_grok, --use_gpt45_preview, " +
              "--use_o1, --use_o1_mini, --use_o3_mini, --use_o3_mini_effort, --use_o4_mini, --use_gpt4o, --use_gpt41, " +
              "--use_deepseek_r1, --use_ollama, --use_ollama_1_5b, --use_ollama_8b, " +
              "--use_ollama_14b, --use_ollama_32b, --use_ollama_70b, --use_ollama_671b, or --use_hyperbolic.")
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
            model = MODEL_CONFIG['deepseek-r1']
            provider = 'OpenRouter'
        
        if not validate_model_availability(model, provider):
            print(f"{COLORS['red']}Error: Model {model} is not available from {provider}.{COLORS['end']}")
            exit(1)

        # Run market analysis
        analysis_result = run_market_analysis(
            args.product_id, 
            args.granularity,
            args.alignment_timeframe_1,
            args.alignment_timeframe_2,
            args.alignment_timeframe_3
        )
        if not analysis_result['success']:
            print(f"Error running market analyzer: {analysis_result['error']}")
            exit(1)
            
        # Calculate alignment score but display it in a more concise format
        if any([args.alignment_timeframe_1, args.alignment_timeframe_2, args.alignment_timeframe_3]):
            alignment_score = analysis_result.get('alignment_score', 0)
            timeframes = analysis_result.get('alignment_timeframes', [args.granularity])
            
            # Use a more concise format for alignment information
            print("\n🔍 Timeframe Alignment:")
            print(f"- Primary: {args.granularity}")
            print(f"- Others: {', '.join([tf for tf in timeframes if tf != args.granularity])}")
            print(f"- Score: {alignment_score}/100")
            
            # Interpret alignment score with symbols
            if alignment_score >= 90:
                print("✅ Excellent alignment (very strong signal)")
            elif alignment_score >= 75:
                print("✅ Good alignment (strong signal)")
            elif alignment_score >= 50:
                print("✓ Moderate alignment (decent signal)")
            else:
                print("⚠️ Poor alignment (weak/conflicting signals)")
                
            print("Continuing with analysis...\n")

        # Get trading recommendation
        alignment_score = analysis_result.get('alignment_score', 50)
        recommendation, reasoning = get_trading_recommendation(
            client, analysis_result['data'], args.product_id, 
            args.use_deepseek, args.use_reasoner, args.use_grok, 
            args.use_gpt45_preview, args.use_o1, args.use_o1_mini, 
            args.use_o3_mini, args.use_o3_mini_effort, args.use_o4_mini, 
            args.use_gpt4o, args.use_gpt41, args.use_deepseek_r1, args.use_ollama, args.use_ollama_1_5b,
            args.use_ollama_8b, args.use_ollama_14b, args.use_ollama_32b,
            args.use_ollama_70b, args.use_ollama_671b, args.use_hyperbolic,
            alignment_score=alignment_score
        )
        if recommendation is None:
            print("Failed to get trading recommendation. Check the logs for details.")
            exit(1)

        # Format and display the output with recommendation
        format_output(recommendation, analysis_result, reasoning)

        # Execute trade only if --execute_trades flag is provided
        if recommendation and args.execute_trades:
            execute_trade(recommendation, args.product_id, args.margin, args.leverage, args.limit_order)
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