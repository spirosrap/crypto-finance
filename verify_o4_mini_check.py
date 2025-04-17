#!/usr/bin/env python3
import inspect
from prompt_market import get_trading_recommendation, COLORS

def verify_o4_mini_handling():
    """
    Verify specifically how the o4-mini model is handled with regard to deterministic parameters.
    """
    print(f"{COLORS['cyan']}Checking o4-mini model handling in prompt_market.py{COLORS['end']}")
    
    # Get the source code of the function
    source_code = inspect.getsource(get_trading_recommendation)
    
    # Check for o4-mini in models_without_deterministic_support
    has_o4_mini_check = "'o4-mini': ['temperature', 'top_p']" in source_code
    
    # Check for model key detection
    has_current_model_check = "if use_o4_mini:" in source_code and "current_model_key = 'o4-mini'" in source_code
    
    # Check for unsupported params check
    has_unsupported_params_check = "if current_model_key and current_model_key in models_without_deterministic_support:" in source_code
    
    # Check for conditional parameter setting
    has_conditional_params = (
        "if 'temperature' not in unsupported_params:" in source_code and
        "if 'top_p' not in unsupported_params:" in source_code and
        "if 'seed' not in unsupported_params:" in source_code
    )
    
    # Print results
    print(f"\n{COLORS['cyan']}--- Results ---{COLORS['end']}")
    
    def print_check(name, passed):
        status = f"{COLORS['green']}✅ Pass{COLORS['end']}" if passed else f"{COLORS['red']}❌ Fail{COLORS['end']}"
        print(f"{name}: {status}")
        
    print_check("o4-mini in models_without_deterministic_support", has_o4_mini_check)
    print_check("o4-mini model detection", has_current_model_check)
    print_check("Unsupported parameters check", has_unsupported_params_check)
    print_check("Conditional parameter setting", has_conditional_params)
    
    all_passed = has_o4_mini_check and has_current_model_check and has_unsupported_params_check and has_conditional_params
    
    print(f"\n{COLORS['cyan']}--- Conclusion ---{COLORS['end']}")
    if all_passed:
        print(f"{COLORS['green']}The code correctly handles o4-mini model's limitations with deterministic parameters.{COLORS['end']}")
        print(f"{COLORS['green']}For the o4-mini model, temperature and top_p parameters will be skipped.{COLORS['end']}")
    else:
        print(f"{COLORS['yellow']}Some checks for o4-mini model handling failed.{COLORS['end']}")
        print(f"{COLORS['yellow']}Please review the implementation to ensure proper handling.{COLORS['end']}")
        
    # Look for warning message about o4-mini
    has_warning = "f\"{COLORS['yellow']}Note: Model {current_model_key} doesn't fully support deterministic parameters.{COLORS['end']}\"" in source_code
    print(f"\n{COLORS['cyan']}User Notification:{COLORS['end']}")
    if has_warning:
        print(f"{COLORS['green']}✅ The code displays a warning when o4-mini is used, notifying users about limited deterministic support.{COLORS['end']}")
    else:
        print(f"{COLORS['red']}❌ No user notification found about o4-mini's limited deterministic support.{COLORS['end']}")
    
    return all_passed

if __name__ == "__main__":
    verify_o4_mini_handling() 