#!/usr/bin/env python3
import hashlib
import json
import inspect
from prompt_market import get_trading_recommendation, COLORS

def test_deterministic_parameters():
    """
    Verify that deterministic parameters (temperature=0, top_p=0, seed=42) 
    are properly set in the get_trading_recommendation function.
    """
    print(f"{COLORS['cyan']}Testing deterministic parameters in prompt_market.py{COLORS['end']}")
    
    # Get the source code of the function
    source_code = inspect.getsource(get_trading_recommendation)
    
    # Check for deterministic parameters in the source code
    checks = {
        "temperature=0": any(["temperature=0" in source_code, "temperature=0.0" in source_code]),
        "top_p=0": any(["top_p=0" in source_code, "top_p=0.0" in source_code]),
        "seed=42": "seed=42" in source_code,
        "hyperbolic deterministic": "hyperbolic_params" in source_code,
        "ollama deterministic": "ollama_params" in source_code
    }
    
    # Print results
    print(f"\n{COLORS['cyan']}--- Results ---{COLORS['end']}")
    
    all_checks_passed = True
    for check_name, passed in checks.items():
        status = f"{COLORS['green']}✅ Pass{COLORS['end']}" if passed else f"{COLORS['red']}❌ Fail{COLORS['end']}"
        print(f"{check_name}: {status}")
        all_checks_passed = all_checks_passed and passed
    
    # Check for specific parameters in API calls
    openai_params_check = '"temperature": 0' in source_code and '"top_p": 0' in source_code and '"seed": 42' in source_code
    status = f"{COLORS['green']}✅ Pass{COLORS['end']}" if openai_params_check else f"{COLORS['red']}❌ Fail{COLORS['end']}"
    print(f"API parameter settings: {status}")
    all_checks_passed = all_checks_passed and openai_params_check
    
    print(f"\n{COLORS['cyan']}--- Conclusion ---{COLORS['end']}")
    if all_checks_passed:
        print(f"{COLORS['green']}All deterministic parameters are properly set in the get_trading_recommendation function.{COLORS['end']}")
        print(f"{COLORS['green']}The function should now produce more consistent results.{COLORS['end']}")
    else:
        print(f"{COLORS['yellow']}Some deterministic parameters may not be properly set.{COLORS['end']}")
        print(f"{COLORS['yellow']}Please check the function implementation to ensure deterministic behavior.{COLORS['end']}")
    
    print(f"\n{COLORS['cyan']}--- Note on LLM Determinism ---{COLORS['end']}")
    print("Even with temperature=0, top_p=0, and a fixed seed, LLMs may still produce")
    print("slightly different outputs across invocations. However, the critical decision")
    print("fields (like signal type, prices, and stop loss) should remain consistent.")
    
    # Return overall result
    return all_checks_passed

if __name__ == "__main__":
    test_deterministic_parameters() 