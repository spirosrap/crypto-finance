import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, OpenAIServerModel
from market_analyzer import MarketAnalyzer
from config import DEEPSEEK_KEY
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

# Initialize the market analyzer
def run_analysis(product_id='BTC-USDC', candle_interval='ONE_HOUR'):
    # Initialize the market analyzer
    analyzer = MarketAnalyzer(product_id=product_id, candle_interval=candle_interval)

    # Get market analysis
    analysis = analyzer.get_market_signal()

    # Create prompt using analysis results
    prompt = f"""
    Current Price: ${analysis['current_price']:.4f}
    Signal: {analysis['signal']}
    Position: {analysis['position']}
    Confidence: {analysis['confidence']*100:.1f}%
    Market Condition: {analysis['market_condition']}

    Technical Indicators:
    - RSI: {analysis['indicators']['rsi']:.2f}
    - MACD: {analysis['indicators']['macd']:.4f}
    - ADX: {analysis['indicators']['adx']:.2f}
    - Trend Direction: {analysis['indicators']['trend_direction']}

    Volume Analysis:
    - Volume Change: {analysis['volume_analysis']['change']:.1f}%
    - Volume Trend: {analysis['volume_analysis']['trend']}
    - Volume Strength: {analysis['volume_analysis']['strength']}

    Risk Metrics:
    - Dynamic Risk: {analysis['risk_metrics']['dynamic_risk']*100:.1f}%

    Trading Recommendation:
    {analysis['recommendation']}

    Based on this analysis and any other information you find online, suggest a SHORT or LONG position for {product_id} and a price target/Stop Loss with probability of success (0-100%) and report the 
    risk/reward ratio.
    """

    # Initialize the agent
    agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

    # Run the agent with the enhanced prompt
    agent.run(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run market analysis for crypto trading')
    parser.add_argument('--product_id', type=str, default='BTC-USDC',
                      help='Trading pair to analyze (default: BTC-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR',
                      help='Candle interval (default: ONE_HOUR)')
    
    args = parser.parse_args()
    
    run_analysis(product_id=args.product_id, candle_interval=args.granularity)