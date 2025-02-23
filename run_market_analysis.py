import argparse
import subprocess

def run_market_analysis(product_id="BTC-USDC", granularity="ONE_HOUR"):
    # Command to run market analyzer
    cmd = [
        "python", "market_analyzer.py",
        "--product_id", product_id,
        "--granularity", granularity,
        "--console_log", "false"
    ]

    # Additional text to append
    append_text = """====  \n \n Reply only with a valid JSON object in a single line (without any markdown code block) representing one of the following signals: For a SELL signal: {\"SELL AT\": <PRICE>, \"BUY BACK AT\": <PRICE>, \"STOP LOSS\": <PRICE>, \"PROBABILITY\": <PROBABILITY>, \"CONFIDENCE\": \"<CONFIDENCE>\", \"R/R_RATIO\": <R/R_RATIO>, \"VOLUME_STRENGTH\": \"<VOLUME_STRENGTH>\", \"IS_VALID\": <IS_VALID>} or for a BUY signal: {\"BUY AT\": <PRICE>, \"SELL BACK AT\": <PRICE>, \"STOP LOSS\": <PRICE>, \"PROBABILITY\": <PROBABILITY>, \"CONFIDENCE\": \"<CONFIDENCE>\", \"R/R_RATIO\": <R/R_RATIO>, \"VOLUME_STRENGTH\": \"<VOLUME_STRENGTH>\", \"IS_VALID\": <IS_VALID>}. Instruction 1: Use code to calculate the R/R ratio. Instruction 2: Signal confidence should be one of: 'Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak'. Instruction 3: Volume strength should be one of: 'Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak'. Instruction 4: If Stop Loss is below current price for a SELL signal, set IS_VALID to False. (Default is True) Instruction 5: If Stop Loss is above current price for a BUY signal, set IS_VALID to False. (Default is True)"""
    try:
        # Run market analyzer and redirect output to file
        with open("market_analysis.txt", "w") as f:
            subprocess.run(cmd, stdout=f, check=True)
        
        # Append the additional text
        with open("market_analysis.txt", "a") as f:
            f.write(append_text)
            
        print(f"Analysis completed successfully for {product_id} with {granularity} granularity")
        print("Results saved to market_analysis.txt")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running market analyzer: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run market analysis with configurable parameters")
    parser.add_argument("--product_id", default="BTC-USDC", help="Product ID (e.g., BTC-USDC, ETH-USDC)")
    parser.add_argument("--granularity", default="ONE_HOUR", 
                      choices=["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "SIX_HOUR", "ONE_DAY"],
                      help="Time granularity for the analysis")
    
    args = parser.parse_args()
    run_market_analysis(args.product_id, args.granularity)

if __name__ == "__main__":
    main() 