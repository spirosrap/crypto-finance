import argparse
import json
import hashlib
import time
import copy
from prompt_market import initialize_client, get_trading_recommendation, MODEL_CONFIG, COLORS

def compare_json_recommendations(rec1, rec2):
    """
    Compare two trading recommendation JSON objects for critical fields.
    Returns tuple (is_identical, differences)
    """
    try:
        # Convert to dict if needed
        if isinstance(rec1, str):
            rec1 = json.loads(rec1.replace("'", '"'))
        if isinstance(rec2, str):
            rec2 = json.loads(rec2.replace("'", '"'))
        
        # Critical fields that must match exactly
        critical_fields = [
            'SIGNAL_TYPE',
            'BUY AT', 'SELL AT',
            'SELL BACK AT', 'BUY BACK AT', 
            'STOP LOSS'
        ]
        
        # Fields that can have small variations
        tolerance_fields = {
            'PROBABILITY': 5,  # Allow 5% difference
            'R/R_RATIO': 0.2,  # Allow 0.2 difference
        }
        
        # Less critical fields we can ignore
        ignore_fields = ['REASONING']
        
        differences = []
        
        # Check critical fields first
        for field in critical_fields:
            # Skip fields not relevant to this signal type
            if field not in rec1 and field not in rec2:
                continue
                
            # One has field and other doesn't
            if (field in rec1 and field not in rec2) or (field in rec2 and field not in rec1):
                differences.append(f"Field '{field}' exists in only one recommendation")
                continue
                
            # Both have field, compare values
            if field in rec1 and field in rec2 and rec1[field] != rec2[field]:
                differences.append(f"Field '{field}' differs: {rec1[field]} vs {rec2[field]}")
        
        # Check tolerance fields
        for field, tolerance in tolerance_fields.items():
            if field in rec1 and field in rec2:
                # Handle numeric comparisons with tolerance
                try:
                    val1 = float(rec1[field])
                    val2 = float(rec2[field])
                    if abs(val1 - val2) > tolerance:
                        differences.append(f"Field '{field}' differs beyond tolerance: {val1} vs {val2}")
                except (ValueError, TypeError):
                    # If not numeric, do exact comparison
                    if rec1[field] != rec2[field]:
                        differences.append(f"Field '{field}' differs: {rec1[field]} vs {rec2[field]}")
        
        # Check remaining fields except ignored ones
        all_fields = set(list(rec1.keys()) + list(rec2.keys()))
        remaining_fields = [f for f in all_fields if f not in critical_fields 
                           and f not in tolerance_fields and f not in ignore_fields]
        
        for field in remaining_fields:
            # Skip if field doesn't exist in both
            if field not in rec1 or field not in rec2:
                continue
                
            # Compare values
            if rec1[field] != rec2[field]:
                differences.append(f"Non-critical field '{field}' differs: {rec1[field]} vs {rec2[field]}")
        
        return len(differences) == 0, differences
    
    except Exception as e:
        return False, [f"Error comparing recommendations: {str(e)}"]

def get_model_flags(model_name):
    """
    Convert model name to the appropriate use_* flags for the get_trading_recommendation function.
    """
    use_flags = {
        "use_deepseek": False,
        "use_reasoner": False,
        "use_grok": False,
        "use_gpt45_preview": False,
        "use_o1": False,
        "use_o1_mini": False,
        "use_o3_mini": False,
        "use_o3_mini_effort": False,
        "use_o4_mini": False,
        "use_gpt4o": False,
        "use_gpt41": False,
        "use_deepseek_r1": False,
        "use_ollama": False,
        "use_ollama_1_5b": False,
        "use_ollama_8b": False,
        "use_ollama_14b": False,
        "use_ollama_32b": False,
        "use_ollama_70b": False,
        "use_ollama_671b": False,
        "use_hyperbolic": False,
    }
    
    # Map model name to the appropriate use flag
    model_map = {
        "deepseek": "use_deepseek",
        "reasoner": "use_reasoner",
        "grok": "use_grok",
        "gpt-4.5": "use_gpt45_preview",
        "o1-mini": "use_o1_mini",
        "o1": "use_o1",
        "o3-mini": "use_o3_mini",
        "o3-mini-effort": "use_o3_mini_effort",
        "o4-mini": "use_o4_mini",
        "gpt-4o": "use_gpt4o",
        "gpt-4.1": "use_gpt41",
        "deepseek-r1": "use_deepseek_r1",
        "ollama": "use_ollama",
    }
    
    # Find the matching flag
    for key, flag in model_map.items():
        if key.lower() in model_name.lower():
            use_flags[flag] = True
            break
    
    return use_flags

def run_deterministic_test(model_name="gpt-4o-mini", iterations=10, use_seed=True):
    """
    Run deterministic test by calling the model multiple times with temperature=0 and top_p=0.
    
    Args:
        model_name: The model to test
        iterations: Number of times to run the test
        use_seed: Whether to use a fixed seed
    """
    print(f"{COLORS['cyan']}Testing deterministic behavior of {model_name}{COLORS['end']}")
    print(f"{COLORS['cyan']}Parameters: temperature=0, top_p=0, use_seed={use_seed}{COLORS['end']}")
    print(f"{COLORS['cyan']}Running {iterations} iterations...{COLORS['end']}\n")
    
    # Get model flags based on model name
    use_model_flags = get_model_flags(model_name)
    
    # Print what model we're using
    model_flag_str = ", ".join([flag for flag, value in use_model_flags.items() if value])
    if model_flag_str:
        print(f"Using model flags: {model_flag_str}")
    else:
        print("Using default model: gpt-4o-mini")
    
    # Initialize the client
    if not initialize_client(**{k: v for k, v in use_model_flags.items() if k in [
        "use_deepseek", "use_reasoner", "use_grok", "use_deepseek_r1", "use_ollama"
    ]}):
        print(f"{COLORS['red']}Failed to initialize client{COLORS['end']}")
        return
    
    # Sample market analysis data for testing - using a fixed example
    market_analysis = """
    Market Analysis for BTC-USDC (ONE_HOUR):
    
    Technical Indicators:
    - RSI: 58.2 (Neutral)
    - MACD: Bullish (MACD: 145.2, Signal: 112.8, Histogram: 32.4)
    - Bollinger Bands: Price in mid-range (Upper: 63840, Middle: 62450, Lower: 61060)
    - Moving Averages: Price above 20 EMA (62100) and 50 EMA (61200)
    - Volume: Moderate, +5% change from previous period
    
    Market Conditions:
    - Current Price: $62,450
    - 24h High: $63,120
    - 24h Low: $61,780
    - Support Levels: $61,800, $61,200, $60,500
    - Resistance Levels: $63,000, $63,800, $64,500
    
    Pattern Analysis:
    - Trend: Bullish
    - Pattern: Ascending triangle forming
    - Pattern Completion: 70%
    - Pattern Confidence: 75%
    
    Volatility Analysis:
    - 24h Volatility: 3.2% (Moderate)
    - ATR: 1250 (Moderate)
    
    Market Summary:
    BTC is currently in a bullish trend, trading at $62,450. Technical indicators are mostly positive with RSI at 58.2 (neutral), bullish MACD, and price trading above key moving averages. An ascending triangle pattern is forming with 70% completion, suggesting potential for an upward breakout. Support at $61,800 appears strong with moderate volume. Volatility is moderate at 3.2% over 24 hours.
    """
    
    # Create fixed product_id
    product_id = "BTC-USDC"
    
    # Store responses
    responses = []
    hashes = []
    parsed_responses = []
    
    # Before running the test, print the specific model that will be used
    if hasattr(MODEL_CONFIG, 'get'):
        for flag_name, is_enabled in use_model_flags.items():
            if is_enabled and flag_name.startswith("use_"):
                model_key = flag_name[4:]  # Remove "use_" prefix
                if model_key in MODEL_CONFIG:
                    print(f"Will use model: {MODEL_CONFIG[model_key]}")
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        
        # Override the OpenAI chat.completions.create method to inject our parameters
        # Save the original method
        from openai import OpenAI
        from prompt_market import client
        original_create = None
        if client and hasattr(client, 'chat') and hasattr(client.chat, 'completions'):
            original_create = client.chat.completions.create
        
            # Define the wrapper function to inject our parameters
            def wrapped_create(*args, **kwargs):
                # Force zero temperature and top_p
                kwargs["temperature"] = 0.0
                kwargs["top_p"] = 0.0
                if use_seed:
                    kwargs["seed"] = 42
                return original_create(*args, **kwargs)
            
            # Replace the method with our wrapper
            client.chat.completions.create = wrapped_create
        
        try:
            # Create a copy of the model flags to avoid modifying the original
            get_rec_flags = copy.copy(use_model_flags)
            
            # Get recommendation using the official function
            recommendation, reasoning = get_trading_recommendation(
                client=client,
                market_analysis=market_analysis,
                product_id=product_id,
                alignment_score=80,
                **get_rec_flags
            )
            
            # Restore the original method if it was modified
            if original_create:
                client.chat.completions.create = original_create
            
            if recommendation:
                # Store response and its hash
                responses.append(recommendation)
                response_hash = hashlib.md5(recommendation.encode()).hexdigest()
                hashes.append(response_hash)
                
                print(f"  Response hash: {response_hash}")
                
                # Try to parse as JSON to check for consistency
                try:
                    rec_dict = json.loads(recommendation.replace("'", '"'))
                    parsed_responses.append(rec_dict)
                    
                    signal_type = rec_dict.get('SIGNAL_TYPE', 'UNKNOWN')
                    probability = rec_dict.get('PROBABILITY', 'N/A')
                    
                    # Print decision details
                    if signal_type == 'BUY':
                        print(f"  Signal: {signal_type}, Buy at: {rec_dict.get('BUY AT', 'N/A')}, " +
                              f"Sell at: {rec_dict.get('SELL BACK AT', 'N/A')}, " +
                              f"Stop: {rec_dict.get('STOP LOSS', 'N/A')}, " +
                              f"Prob: {probability}")
                    elif signal_type == 'SELL':
                        print(f"  Signal: {signal_type}, Sell at: {rec_dict.get('SELL AT', 'N/A')}, " +
                              f"Buy at: {rec_dict.get('BUY BACK AT', 'N/A')}, " +
                              f"Stop: {rec_dict.get('STOP LOSS', 'N/A')}, " +
                              f"Prob: {probability}")
                    else:  # HOLD
                        print(f"  Signal: {signal_type}, Price: {rec_dict.get('PRICE', 'N/A')}, " +
                              f"Prob: {probability}")
                except json.JSONDecodeError:
                    print(f"  {COLORS['yellow']}Could not parse response as JSON{COLORS['end']}")
            else:
                print(f"  {COLORS['red']}Failed to get recommendation{COLORS['end']}")
        
        except Exception as e:
            print(f"  {COLORS['red']}Error: {str(e)}{COLORS['end']}")
        
        # Wait a moment between requests
        if i < iterations - 1:
            time.sleep(2)
    
    # Check if all hashes are identical
    if hashes:
        all_same = all(h == hashes[0] for h in hashes)
        
        print(f"\n{COLORS['cyan']}--- Hash Comparison Results ---{COLORS['end']}")
        print(f"All responses identical (by hash): {'✅ Yes' if all_same else '❌ No'}")
        
        # Check if all critical fields are identical even if the text varies
        if len(parsed_responses) > 1 and not all_same:
            print(f"\n{COLORS['cyan']}--- Critical Field Comparison ---{COLORS['end']}")
            base_response = parsed_responses[0]
            all_fields_same = True
            
            for i, response in enumerate(parsed_responses[1:], 1):
                is_same, differences = compare_json_recommendations(base_response, response)
                all_fields_same = all_fields_same and is_same
                
                if not is_same:
                    print(f"Response {i+1} has critical differences: {', '.join(differences)}")
            
            if all_fields_same:
                print(f"✅ All critical trading fields are consistent despite text differences")
            else:
                print(f"❌ Critical trading fields differ between responses")
        
        if not all_same:
            print(f"\n{COLORS['yellow']}Found differences in responses:{COLORS['end']}")
            # Find first different response
            first_diff_idx = next((i for i, h in enumerate(hashes) if h != hashes[0]), None)
            if first_diff_idx is not None:
                print(f"\n{COLORS['cyan']}First response:{COLORS['end']}")
                print(f"{responses[0]}")
                print(f"\n{COLORS['cyan']}Different response (iteration {first_diff_idx+1}):{COLORS['end']}")
                print(f"{responses[first_diff_idx]}")
    else:
        print(f"\n{COLORS['red']}No valid responses were collected{COLORS['end']}")

def main():
    parser = argparse.ArgumentParser(description="Test LLM deterministic behavior with temperature=0 and top_p=0")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to test")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to run")
    parser.add_argument("--no-seed", action="store_true", help="Disable fixed seed")
    
    args = parser.parse_args()
    
    run_deterministic_test(
        model_name=args.model,
        iterations=args.iterations,
        use_seed=not args.no_seed
    )

if __name__ == "__main__":
    main() 