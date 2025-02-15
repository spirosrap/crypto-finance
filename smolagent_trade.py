import os
import subprocess
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel
import argparse

# Retrieve API token from environment variables
#hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Define the model ID (optional, use default if not specified)
#model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"  # Replace with your preferred model or remove for default

# Initialize the model with the API key
#model = HfApiModel(model_id=model_id, token=hf_api_key)


model = OpenAIServerModel(
    model_id="gpt-4o",
    api_base="https://api.openai.com/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    # custom_role_conversions={
    #     "system": "user",  # Convert system to user
    #     "tool-response": "user",  # Convert tool-response to assistant instead of tool
    # }
)

# model = OpenAIServerModel(
#     model_id="deepseek-reasoner",
#     api_base="https://api.deepseek.com", # Leave this blank to query OpenAI servers.
#     api_key=DEEPSEEK_KEY, # Switch to the API key for the server you're targeting.
# )

def run_analysis(product_id='BTC-USDC', granularity='ONE_HOUR'):
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

    Based on this analysis and using any other information you find online for sentiment analysis or news, suggest a SHORT or LONG position for {product_id} and a price target/Stop Loss with probability of success (0-100%) and report the 
    risk/reward ratio and signa confidence level. Also report volume confirmation signal level and general sentiment level of the market.
    """

    # Initialize the agent
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

    # Run the agent with the prompt
    agent.run(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run market analysis for crypto trading')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Trading pair to analyze (default: BTC-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR',
                      help='Candle interval (default: ONE_HOUR)')
    
    args = parser.parse_args()
    
    run_analysis(product_id=args.product_id, granularity=args.granularity)