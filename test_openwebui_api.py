import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:1.5b",
    "prompt": "Tell me a joke.",
}
response = requests.post(url, json=payload)

try:
    # Split the response text into individual JSON objects
    response_lines = response.text.strip().split('\n')
    
    # Concatenate the "response" fields from each JSON object
    full_response = ""
    for line in response_lines:
        json_obj = requests.models.complexjson.loads(line)
        full_response += json_obj["response"]
    
    print(full_response)
except requests.exceptions.JSONDecodeError as e:
    # Handle JSON decoding errors
    print("Failed to decode JSON response:", e)
    print("Response content:", response.text)
