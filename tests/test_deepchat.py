from openai import OpenAI
from config import DEEPSEEK_KEY

try:
    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)

except Exception as e:
    error_msg = str(e).lower()
    print(f"Error occurred while calling Deepseek API: {str(e)}")
    
    if "api_key" in error_msg:
        print("Please check if your DEEPSEEK_KEY is correctly set in config.py")
    elif "insufficient_quota" in error_msg or "credits" in error_msg:
        print("Your Deepseek API credits have been depleted. Please recharge your account.")
    elif "connection" in error_msg:
        print("Connection error: Please check your internet connection and the API base URL")
    elif "rate limit" in error_msg:
        print("Rate limit exceeded. Please wait a moment before trying again.")
    elif "model" in error_msg:
        print("Invalid model specified. Please check if 'deepseek-chat' is the correct model name.")
    else:
        print(error_msg)
