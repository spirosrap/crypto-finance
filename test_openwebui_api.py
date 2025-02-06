import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "deepseek-r1:1.5b",
    "prompt": "Tell me a joke.",
}
response = requests.post(url, json=payload)

print(response.json()["response"])
