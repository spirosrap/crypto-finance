import requests
import json  # Import the json module for safe parsing

url = "http://localhost:11434/api/generate"
data = {
    "model": "deepseek-r1:1.5b",
    "prompt": "Tell me a joke about a rat in a maze.",
    "stream": True  # Ensure streaming output
}

response = requests.post(url, json=data, stream=True)

# Process streaming output
full_response = ""
for line in response.iter_lines():
    if line:
        json_data = line.decode("utf-8")
        try:
            # Use json.loads instead of eval for safe parsing
            parsed_data = json.loads(json_data)
            full_response += parsed_data.get("response", "")
        except json.JSONDecodeError:
            pass  # Skip malformed lines

print(full_response)  # Full concatenated response
