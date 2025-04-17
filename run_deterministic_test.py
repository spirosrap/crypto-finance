#!/usr/bin/env python3
import os
import json
import datetime
import subprocess
from typing import Dict, List, Any

def run_test(model: str, iterations: int = 10, use_seed: bool = True) -> Dict[str, Any]:
    """
    Run the deterministic test and capture the results
    """
    cmd = ["python", "verify_deterministic_llm.py", 
           "--model", model, 
           "--iterations", str(iterations)]
    
    if not use_seed:
        cmd.append("--no-seed")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse the output to extract results
    output = result.stdout
    
    # Determine if all hashes were identical
    identical_hashes = "All responses identical (by hash): ✅ Yes" in output
    
    # Determine if critical fields were consistent
    critical_fields_consistent = "✅ All critical trading fields are consistent" in output
    
    # Extract signal type and decision from output
    signal_type = None
    buy_at = None
    sell_at = None
    stop_loss = None
    probability = None
    
    for line in output.split('\n'):
        if "Signal:" in line:
            parts = line.split(',')
            if len(parts) >= 4:
                signal_type = parts[0].split(':')[1].strip()
                
                for part in parts:
                    if "Buy at:" in part:
                        buy_at = part.split(':')[1].strip()
                    if "Sell at:" in part:
                        sell_at = part.split(':')[1].strip()
                    if "Stop:" in part:
                        stop_loss = part.split(':')[1].strip()
                    if "Prob:" in part:
                        probability = part.split(':')[1].strip()
            break
    
    # Check if any JSON responses were found
    responses_found = "Response hash:" in output
    
    return {
        "model": model,
        "iterations": iterations,
        "use_seed": use_seed,
        "timestamp": datetime.datetime.now().isoformat(),
        "identical_hashes": identical_hashes,
        "critical_fields_consistent": critical_fields_consistent,
        "signal_type": signal_type,
        "buy_at": buy_at,
        "sell_at": sell_at, 
        "stop_loss": stop_loss,
        "probability": probability,
        "responses_found": responses_found,
        "raw_output": output
    }

def save_results(results: List[Dict[str, Any]], filename: str = "deterministic_test_results.json"):
    """
    Save test results to a JSON file
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filename}")

def main():
    # Models to test
    models = [
        "gpt-4o-mini",  # Default model
        "gpt-4o",       # Full GPT-4o
    ]
    
    results = []
    
    print(f"Running deterministic tests on {len(models)} models...")
    
    for model in models:
        print(f"\n{'-'*50}")
        print(f"Testing model: {model}")
        print(f"{'-'*50}\n")
        
        # Test with seed
        print(f"Running test with seed=42...")
        result_with_seed = run_test(model, iterations=10, use_seed=True)
        results.append(result_with_seed)
        
        # Print summary
        print(f"Results for {model} with seed=42:")
        print(f"  Identical hashes: {result_with_seed['identical_hashes']}")
        print(f"  Critical fields consistent: {result_with_seed['critical_fields_consistent']}")
        print(f"  Signal: {result_with_seed['signal_type']}")
        print(f"  Buy at: {result_with_seed['buy_at']}")
        print(f"  Sell at: {result_with_seed['sell_at']}")
        print(f"  Stop loss: {result_with_seed['stop_loss']}")
        print(f"  Probability: {result_with_seed['probability']}")
        
        # Test without seed
        print(f"\nRunning test with no seed...")
        result_without_seed = run_test(model, iterations=10, use_seed=False)
        results.append(result_without_seed)
        
        # Print summary
        print(f"Results for {model} without seed:")
        print(f"  Identical hashes: {result_without_seed['identical_hashes']}")
        print(f"  Critical fields consistent: {result_without_seed['critical_fields_consistent']}")
        print(f"  Signal: {result_without_seed['signal_type']}")
        print(f"  Buy at: {result_without_seed['buy_at']}")
        print(f"  Sell at: {result_without_seed['sell_at']}")
        print(f"  Stop loss: {result_without_seed['stop_loss']}")
        print(f"  Probability: {result_without_seed['probability']}")
    
    # Save all results
    save_results(results)
    
    # Print final summary
    print("\n=== FINAL SUMMARY ===")
    for result in results:
        seed_status = "with seed=42" if result["use_seed"] else "without seed"
        identical = "✅" if result["identical_hashes"] else "❌"
        consistent = "✅" if result["critical_fields_consistent"] else "❌"
        print(f"{result['model']} {seed_status}: Identical hashes: {identical}, Consistent fields: {consistent}")

if __name__ == "__main__":
    main() 