import requests

url = "http://localhost:3000/predict"
data = {
    "payload": {
        "Year": 2020,
        "Month": 5,
        "Day": 15,
        "Location": "Iceland-S",
        "Deaths": 2,
    }
    }

response = requests.post(url, json=data)  # 'json=' automatically sets Content-Type
print(response.text)
