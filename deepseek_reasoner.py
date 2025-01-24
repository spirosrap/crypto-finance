from openai import OpenAI
import os
import sys
from typing import List, Dict, Optional
import logging
from config import DEEPSEEK_KEY
import argparse
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeepSeekReasoner:
    def __init__(self, market_analysis: str, product_id: str):
        if not DEEPSEEK_KEY:
            raise ValueError("DeepSeek API key is not set in config.py")
        self.client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")
        self.market_analysis = market_analysis
        self.product_id = product_id
        self.conversation_history: List[Dict[str, str]] = []
        
        # Get initial trading recommendation
        initial_messages = [
            {"role": "system", "content": "Reply only with \"BUY AT <PRICE> and SELL AT <PRICE>\" or \"SELL AT <PRICE> and BUY BACK AT <PRICE>\""},
            {"role": "user", "content": f"Here's the latest market analysis for {product_id}:\n{market_analysis}\nBased on this analysis, provide a trading recommendation."}
        ]
        
        # Get initial recommendation
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=initial_messages,
            stream=True
        )
        recommendation, initial_reasoning = self.process_stream(response)
        
        # Print initial recommendation
        print("\n🤖 Initial Trading Recommendation:", recommendation)
        if initial_reasoning:
            print("🧠 Reasoning:", initial_reasoning)
        
        # Initialize conversation history for follow-up questions
        self.conversation_history = [
            {"role": "system", "content": "You are a cryptocurrency market analysis assistant. Help analyze and explain the market data and trading recommendation."},
            {"role": "user", "content": f"Here's the latest market analysis for {product_id}:\n{market_analysis}"},
            {"role": "assistant", "content": f"I've analyzed the market data. The trading recommendation is: {recommendation}"}
        ]

    def process_stream(self, response) -> tuple[str, str]:
        """Process streaming response and return content and reasoning."""
        reasoning_content = ""
        content = ""
        
        for chunk in response:
            if chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
            elif chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
                
        return content.strip(), reasoning_content.strip()

    def get_response(self, question: str) -> tuple[str, Optional[str]]:
        """Get response from DeepSeek Reasoner model."""
        try:
            # Add user's question to conversation history
            self.conversation_history.append({"role": "user", "content": question})
            
            # Get response from model
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=self.conversation_history,
                stream=True
            )
            
            # Process the response
            answer, reasoning = self.process_stream(response)
            
            # Add assistant's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": answer})
            
            return answer, reasoning
            
        except Exception as e:
            logger.error(f"Error getting response from DeepSeek: {str(e)}")
            return f"Error: {str(e)}", None

    def clear_history(self):
        """Clear conversation history and reinitialize with market analysis."""
        # Get initial trading recommendation again
        initial_messages = [
            {"role": "system", "content": "Reply only with \"BUY AT <PRICE> and SELL AT <PRICE>\" or \"SELL AT <PRICE> and BUY BACK AT <PRICE>\""},
            {"role": "user", "content": f"Here's the latest market analysis for {self.product_id}:\n{self.market_analysis}\nBased on this analysis, provide a trading recommendation."}
        ]
        
        # Get fresh recommendation
        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=initial_messages,
            stream=True
        )
        recommendation, initial_reasoning = self.process_stream(response)
        
        # Print fresh recommendation
        print("\n🤖 Fresh Trading Recommendation:", recommendation)
        if initial_reasoning:
            print("🧠 Reasoning:", initial_reasoning)
        
        # Reset conversation history for follow-up questions
        self.conversation_history = [
            {"role": "system", "content": "You are a cryptocurrency market analysis assistant. Help analyze and explain the market data and trading recommendation."},
            {"role": "user", "content": f"Here's the latest market analysis for {self.product_id}:\n{self.market_analysis}"},
            {"role": "assistant", "content": f"I've analyzed the market data. The trading recommendation is: {recommendation}"}
        ]

def run_market_analysis(product_id: str, granularity: str) -> Optional[Dict]:
    """Run market analysis and return the output as a dictionary."""
    try:
        with open(os.devnull, 'w') as devnull:
            result = subprocess.check_output(
                ['python', 'market_analyzer.py', '--product_id', product_id, '--granularity', granularity],
                text=True,
                stderr=devnull,
                timeout=300  # 5 minute timeout
            )
        return {'success': True, 'data': result}
    except subprocess.TimeoutExpired:
        logger.error("Market analyzer timed out after 5 minutes")
        return {'success': False, 'error': "Analysis timed out"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Market analyzer error: {str(e)}")
        return {'success': False, 'error': str(e)}
    except Exception as e:
        logger.error(f"Unexpected error in market analysis: {str(e)}")
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description='Interactive market analysis with DeepSeek Reasoner')
    parser.add_argument('--product_id', type=str, default='BTC-USDC', help='Trading pair to analyze (e.g., BTC-USDC)')
    parser.add_argument('--granularity', type=str, default='ONE_HOUR', help='Time granularity for analysis')
    args = parser.parse_args()

    # Run market analysis first
    print(f"\nRunning market analysis for {args.product_id}...")
    analysis_result = run_market_analysis(args.product_id, args.granularity)
    
    if not analysis_result['success']:
        print(f"Error running market analysis: {analysis_result.get('error', 'Unknown error')}")
        return

    # Initialize reasoner with market analysis
    reasoner = DeepSeekReasoner(analysis_result['data'], args.product_id)
    
    print("\nWelcome to DeepSeek Reasoner Interactive Mode!")
    print("Market analysis has been loaded. You can now ask questions about it.")
    print("Type 'exit' to quit or 'clear' to start a fresh conversation.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYour question (or 'exit'/'clear'): ").strip()
            
            # Check for exit command
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
                
            # Check for clear command
            if user_input.lower() == 'clear':
                reasoner.clear_history()
                print("\nConversation history cleared. Market analysis reloaded!")
                continue
                
            # Skip empty input
            if not user_input:
                continue
                
            # Get response from model
            answer, reasoning = reasoner.get_response(user_input)
            
            # Print the response
            print("\n🤖 Answer:", answer)
            
            # Print reasoning if available
            if reasoning:
                print("\n🧠 Reasoning:", reasoning)
                
        except KeyboardInterrupt:
            print("\n\nExiting gracefully...")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 