import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, OpenAIServerModel

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

# Initialize the agent
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)

# Run the agent
agent.run("Find me the best memecoins to invest in right NOW.")