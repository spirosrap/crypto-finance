import requests
import json  # Import the json module for safe parsing

url = "http://localhost:11434/api/generate"
data = {
    "model": "deepseek-r1:7b",
    "prompt": "Tell me a joke about a rat in a maze.",
    "stream": True  # Ensure streaming output
}

try:
    response = requests.post(url, json=data, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Process streaming output
    full_response = ""
    for line in response.iter_lines():
        if line:
            json_data = line.decode("utf-8")
            try:
                # Use json.loads instead of eval for safe parsing
                parsed_data = json.loads(json_data)
                full_response += parsed_data.get("response", "")
            except json.JSONDecodeError as json_err:
                print(f"Error parsing JSON response: {json_err}")
                continue  # Skip malformed lines

    print(full_response)  # Full concatenated response

except requests.exceptions.RequestException as req_err:
    print(f"Error making API request: {req_err}")
except Exception as e:
    print(f"Unexpected error occurred: {e}")
