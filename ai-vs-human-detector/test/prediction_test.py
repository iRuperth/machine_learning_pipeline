import requests

url = "http://127.0.0.1:8000/predict"

params = {"text": "Este es un texto escrito por un humano"}

response = requests.post(url, params=params)

if response.status_code == 200:
    print("success:")
    print(response.json())
else:
    print(f"Error {response.status_code}:")
    print(response.text) 