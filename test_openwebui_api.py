import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:1.5b",
    "prompt": "Tell me a joke.",
}
response = requests.post(url, json=payload)

try:
    # Attempt to parse the response as JSON
    response_data = response.json()
    print(response_data["response"])
except requests.exceptions.JSONDecodeError as e:
    # Handle JSON decoding errors
    print("Failed to decode JSON response:", e)
    print("Response content:", response.text)
