import os
import subprocess
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel
import argparse
from config import DEEPSEEK_KEY, SERPAPI_KEY

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

    Based on this analysis and using any other information you find online for sentiment analysis or news, suggest a SHORT or LONG position for {product_id} 
    and a price target/Stop Loss with probability of success (0-100%) and report the risk/reward ratio and signal confidence level. 
    Also report volume confirmation signal level and general sentiment level of the market. Use python to calculate and report the loss and profit in USD with 20x leverage and 100$ initial margin (total 2000).
    Show potential profit and loss and conservative/aggressive target prices.
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
        agent.run(prompt)
    except Exception as e:
        if model_name == "deepseek-reasoner":
            print(f"Error with deepseek-reasoner: {e}")
            print("\nNote: deepseek-reasoner may have compatibility issues with web search tools.")
            print("You can try using --model deepseek-chat instead or use --search-tool none to disable web search.")
        else:
            print(f"Error running analysis with {model_name}: {e}")
            if "DuckDuckGoSearchException" in str(e) or "Ratelimit" in str(e):
                print("\nYou've hit DuckDuckGo rate limits. Try using --search-tool serpapi or --search-tool none.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run market analysis for crypto trading')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Trading pair to analyze (default: BTC-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR',
                      help='Candle interval (default: ONE_HOUR)')
    parser.add_argument('--model', type=str, default='o3-mini',
                      choices=['o3-mini', 'o4-mini', 'gpt-4.1', 'deepseek-chat', 'deepseek-reasoner'],
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