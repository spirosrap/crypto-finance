import os
import subprocess
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel
import argparse
from config import DEEPSEEK_KEY, SERPAPI_KEY
import math
from typing import Dict
import json  # Add json import for pretty printing
import re

# Add SerpAPI search tool if available
try:
    from smolagents import SerpAPISearchTool
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# Retrieve API token from environment variables
#hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Define the model ID (optional, use default if not specified)
#model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Replace with your preferred model or remove for default

# Initialize the model with the API key
#model = HfApiModel(model_id=model_id, token=hf_api_key)

# Initialize model based on command line arguments
def get_model(model_name):
    if model_name == "o3-mini":
        return OpenAIServerModel(
            model_id="o3-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif model_name == "o4-mini":
        return OpenAIServerModel(
            model_id="o4-mini",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif model_name == "gpt-4.1":
        return OpenAIServerModel(
            model_id="gpt-4.1",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif model_name == "o3":
        return OpenAIServerModel(
            model_id="o3",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    elif model_name == "deepseek-chat":
        return OpenAIServerModel(
            model_id="deepseek-chat",
            api_base="https://api.deepseek.com",
            api_key=DEEPSEEK_KEY,
        )
    elif model_name == "deepseek-reasoner":
        return OpenAIServerModel(
            model_id="deepseek-reasoner",
            api_base="https://api.deepseek.com",
            api_key=DEEPSEEK_KEY,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

model = get_model("o3-mini")

def run_analysis(product_id='BTC-USDC', granularity='ONE_HOUR', model_name='o3-mini', search_tool='duckduckgo'):
    # Run market_analyzer.py as a subprocess
    with open(os.devnull, 'w') as devnull:
        try:
            result = subprocess.check_output(
                ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', granularity, '--console_logging', 'false'],
                text=True,
                stderr=devnull,
                timeout=300  # 5 minute timeout
            )
        except subprocess.TimeoutExpired:
            print("Analysis timed out after 5 minutes")
            return
        except subprocess.CalledProcessError as e:
            print(f"Error running market analysis: {e}")
            return

    # Create prompt using the subprocess output
    prompt = f"""
    {result}

    — TASK —
    Produce an **objective, data‑driven trade assessment** for {product_id}.
    Return **LONG / SHORT / NEUTRAL** (no preference wording, just the label).

    — METHODS & REQUIRED OUTPUT —
    1. **Market bias**  
    • Determine LONG / SHORT / NEUTRAL from quantified technical signals **and** a timestamped sentiment index built from reputable sources (news headlines, on‑chain data, social chatter).  
    • If evidence strength < 60 % (per your own scoring rubric), output **NEUTRAL**.

    2. **Price targets & risk control**  
    • Compute statistically derived TP and SL: median of the last N comparable setups ± 1 std‑dev.  
    • Report Risk/Reward ratio.

    3. **Probabilities & confidence metrics**  
    • Win‑probability: historical hit‑rate of comparable setups (0‑100 %).  
    • Confidence level: {{"Low", "Moderate", "High"}} mapped to p‑value bands (<0.32, 0.05, 0.01).

    4. **Volume & sentiment scores**  
    • Volume‑confirmation score (0‑100 %) from z‑score of current vs 30‑bar mean.  
    • Aggregate market sentiment index (–1 = max bearish, +1 = max bullish).

    5. **P&L simulation (Python)**  
    • Starting margin: $100, leverage 20× ⇒ position size $2 000.  
    • Show potential **profit** and **loss** in USD for both:  
        – Conservative targets (median TP/SL).  
        – Aggressive targets (upper‑quartile TP, lower‑quartile SL).  
    • Print a small table with TP, SL, P&L‑conservative, P&L‑aggressive.

    — EVIDENCE REQUIREMENTS —
    • Cite every external dataset or headline with source + timestamp.  
    • No subjective language ("looks good", "feels bullish", etc.).  
    • If data are insufficient, return: **"NO TRADE – INSUFFICIENT EVIDENCE"**.

    — OUTPUT FORMAT —
    Return your analysis as structured JSON with the EXACT format shown below.
    Do not deviate from this structure. Keys must match EXACTLY as shown:

    ```json
    {{
        "SIGNAL_TYPE": "LONG, SHORT, or NEUTRAL",
        "TP_PRICE": 00000.00,
        "SL_PRICE": 00000.00,
        "PROBABILITY": 00.0,
        "RR_RATIO": 0.0,
        "CONFIDENCE_LEVEL": "Low/Moderate/High",
        "VOLUME_SCORE": 00,
        "SENTIMENT_SCORE": 0.0,
        "PNL_CONSERVATIVE": {{
            "TP": 000.00,  
            "SL": -000.00  
        }},
        "PNL_AGGRESSIVE": {{
            "TP": 000.00,
            "SL": -000.00
        }},
        "SOURCES": [
            "Source 1 with timestamp",
            "Source 2 with timestamp"
        ]
    }}
    ```

    Note that profit values should be positive numbers and loss values should be negative numbers.
    All calculations should be done carefully to ensure mathematical accuracy.
    """

    # Initialize the model and agent
    try:
        model = get_model(model_name)
        
        # Set up agent with the selected search tool
        tools = []
        if search_tool != 'none':
            if search_tool == 'serpapi' and SERPAPI_AVAILABLE and SERPAPI_KEY:
                print("Using SerpAPI for web search")
                tools = [SerpAPISearchTool(api_key=SERPAPI_KEY)]
            elif search_tool == 'duckduckgo':
                print("Using DuckDuckGo for web search")
                tools = [DuckDuckGoSearchTool()]
            else:
                print(f"Search tool '{search_tool}' not available or configured. Running without web search.")
        
        agent = CodeAgent(tools=tools, model=model, max_steps=10)
        
        # Run the agent with the prompt
        result = agent.run(prompt)
        
        # Extract and validate the trade recommendation
        validated_result = validate_trade_recommendation(result)
        
        return validated_result
        
    except Exception as e:
        if model_name == "deepseek-reasoner":
            print(f"Error with deepseek-reasoner: {e}")
            print("\nNote: deepseek-reasoner may have compatibility issues with web search tools.")
            print("You can try using --model deepseek-chat instead or use --search-tool none to disable web search.")
        else:
            print(f"Error running analysis with {model_name}: {e}")
            if "DuckDuckGoSearchException" in str(e) or "Ratelimit" in str(e):
                print("\nYou've hit DuckDuckGo rate limits. Try using --search-tool serpapi or --search-tool none.")

def validate_trade_recommendation(agent_output):
    """
    Validate the trade recommendation calculations to ensure they're mathematically correct.
    
    Args:
        agent_output (str or dict): The output from the agent containing the trade recommendation
    
    Returns:
        str: The validated output with added validation results
    """
    print("Validating trade calculations...", end="")
    
    # Handle dictionary output from the agent (most likely already parsed JSON)
    if isinstance(agent_output, dict):
        trade_data = agent_output
        original_output = json.dumps(trade_data, indent=4)
    else:
        # Extract JSON from the agent output if it's a string
        try:
            json_pattern = r'```json\s*(.*?)\s*```'
            json_matches = re.findall(json_pattern, agent_output, re.DOTALL)
            
            if not json_matches:
                # Try to find JSON without markdown code blocks
                try:
                    trade_data = json.loads(agent_output)
                    original_output = agent_output
                except json.JSONDecodeError:
                    print(" Failed (No valid JSON found)")
                    return agent_output
            else:
                trade_data = json.loads(json_matches[0].strip())
                original_output = agent_output
        except Exception as e:
            print(f" Failed (Error parsing JSON: {e})")
            return agent_output
    
    try:
        # Standard position parameters
        margin = 100
        leverage = 20
        position_size = margin * leverage
        
        signal_type = trade_data.get("SIGNAL_TYPE")
        tp_price = trade_data.get("TP_PRICE")
        sl_price = trade_data.get("SL_PRICE")
        rr_ratio = trade_data.get("RR_RATIO")
        
        # Skip validation for NEUTRAL signals
        if signal_type == "NEUTRAL":
            print(" Skipped (NEUTRAL signal)")
            return original_output
            
        # Calculate the implied entry price from the RR ratio
        # RR Ratio = Reward / Risk
        if signal_type == "LONG":
            # For LONG: RR = (TP - Entry) / (Entry - SL)
            # Solving for Entry: Entry = (TP + RR*SL) / (1 + RR)
            implied_entry = (tp_price + rr_ratio * sl_price) / (1 + rr_ratio)
        else:  # SHORT
            # For SHORT: RR = (Entry - TP) / (SL - Entry)
            # Solving for Entry: Entry = (RR*SL + TP) / (1 + RR)
            implied_entry = (rr_ratio * sl_price + tp_price) / (1 + rr_ratio)
        
        # Calculate expected P&L values
        if signal_type == "LONG":
            expected_tp_pnl = ((tp_price - implied_entry) / implied_entry) * position_size
            expected_sl_pnl = ((sl_price - implied_entry) / implied_entry) * position_size
        else:  # SHORT
            expected_tp_pnl = ((implied_entry - tp_price) / implied_entry) * position_size
            expected_sl_pnl = -1 * abs(((implied_entry - sl_price) / implied_entry) * position_size)
        
        # Recalculate Risk/Reward ratio
        if signal_type == "LONG":
            recalculated_rr = (tp_price - implied_entry) / (implied_entry - sl_price)
        else:  # SHORT
            recalculated_rr = (implied_entry - tp_price) / (sl_price - implied_entry)
        
        # Round values for comparison
        expected_tp_pnl = round(expected_tp_pnl, 2)
        expected_sl_pnl = round(expected_sl_pnl, 2)
        recalculated_rr = round(recalculated_rr, 2)
        
        # Get the actual P&L values from the trade data
        actual_tp_pnl = trade_data.get("PNL_CONSERVATIVE", {}).get("TP")
        actual_sl_pnl = trade_data.get("PNL_CONSERVATIVE", {}).get("SL")
        
        # Check if calculated values match reported values with increased tolerance
        # Increase tolerance to handle minor discrepancies from different calculation methods
        tolerance_pnl = 1.0  # Allow up to $1 difference in P&L values
        tolerance_rr = 0.1   # Allow up to 0.1 difference in RR ratio
        
        tp_pnl_valid = abs(expected_tp_pnl - actual_tp_pnl) <= tolerance_pnl
        sl_pnl_valid = abs(expected_sl_pnl - actual_sl_pnl) <= tolerance_pnl
        rr_valid = abs(recalculated_rr - rr_ratio) <= tolerance_rr
        
        all_valid = tp_pnl_valid and sl_pnl_valid and rr_valid
        
        # Prepare validation results
        validation_results = {
            "Implied Entry Price": round(implied_entry, 2),
            "Recalculated RR Ratio": recalculated_rr,
            "Expected TP P&L": expected_tp_pnl,
            "Expected SL P&L": expected_sl_pnl,
            "Calculations Valid": all_valid
        }
        
        # Format the final output
        if all_valid:
            print(" ✅ Valid")
            return original_output
        else:
            discrepancies = []
            if not tp_pnl_valid:
                discrepancies.append(f"TP P&L: expected ${expected_tp_pnl} vs reported ${actual_tp_pnl}")
            if not sl_pnl_valid:
                discrepancies.append(f"SL P&L: expected ${expected_sl_pnl} vs reported ${actual_sl_pnl}")
            if not rr_valid:
                discrepancies.append(f"RR Ratio: calculated {recalculated_rr} vs reported {rr_ratio}")
                
            discrepancy_details = ", ".join(discrepancies)
            print(f" ⚠️ Invalid: {discrepancy_details}")
            
            # Add simplified validation message to output
            validation_message = f"\n\n⚠️ VALIDATION: Calculations contain discrepancies: {discrepancy_details}"
            return original_output + validation_message
        
    except (KeyError, TypeError, ZeroDivisionError) as e:
        print(f" ⚠️ Error: {str(e)}")
        return original_output + f"\n\n⚠️ VALIDATION ERROR: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run market analysis for crypto trading')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Trading pair to analyze (default: BTC-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR',
                      help='Candle interval (default: ONE_HOUR)')
    parser.add_argument('--model', type=str, default='o3-mini',
                      choices=['o3-mini', 'o3', 'o4-mini', 'gpt-4.1', 'deepseek-chat', 'deepseek-reasoner'],
                      help='Model to use for analysis (default: o3-mini)')
    parser.add_argument('--search-tool', type=str, default='duckduckgo',
                      choices=['duckduckgo', 'serpapi', 'none'],
                      help='Web search tool to use (default: duckduckgo)')
    
    args = parser.parse_args()
    
    run_analysis(
        product_id=args.product_id, 
        granularity=args.granularity, 
        model_name=args.model,
        search_tool=args.search_tool
    )