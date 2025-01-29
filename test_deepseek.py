from openai import OpenAI
from openai import APIError
from config import DEEPSEEK_KEY

client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

def make_api_call(messages, round_num):
    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        return response
    except APIError as e:
        if "insufficient_quota" in str(e):
            print(f"Round {round_num} Error: You have run out of API credits")
        elif "invalid_api_key" in str(e):
            print(f"Round {round_num} Error: The API key is invalid")
        elif "rate_limit_exceeded" in str(e):
            print(f"Round {round_num} Error: Rate limit exceeded. Please wait before making more requests")
        else:
            print(f"Round {round_num} Error: {str(e)}")
        return None

# Round 1
messages = [{"role": "user", "content": "9.11 and 9.8, which is greater?"}]
response = make_api_call(messages, 1)

if response:
    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content

    # Round 2
    messages.append({'role': 'assistant', 'content': content})
    messages.append({'role': 'user', 'content': "How many Rs are there in the word 'strawberry'?"})
    response = make_api_call(messages, 2)

    if response:
        print(reasoning_content)
        print(content)