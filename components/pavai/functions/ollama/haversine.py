## pip install haversine 

import requests
import json
import sys
from haversine import haversine

country = sys.argv[1]
mylat, mylon = 47.6455, -122.8258
schema = {
    "city": {
        "type": "string",
        "description": "Name of the city"
    },
    "lat": {
        "type": "float",
        "description": "Decimal Latitude of the city"
    },
    "lon": {
        "type": "float",
        "description": "Decimal longitude of the city"
    }
}

payload = {
    "model": "llama2",
    "messages": [
        {
            "role": "system",
            "content": f"You are a helpful AI assistant. The user will enter a country name and the assistant will return the decimal latitude and decimal longitude of the capital of the country. Output in JSON using the schema defined here: {json.dumps(schema)}."
        },
        {"role": "user", "content": "France"},
        {"role": "assistant", "content": json.dumps({"city": "Paris", "lat": 48.8566, "lon": 2.3522})},
        {"role": "user", "content": country}
    ],
    "format": "json",
    "stream": False
}

response = requests.post("http://localhost:11434/api/chat", json=payload)
cityinfo = json.loads(response.json()["message"]["content"])

distance = haversine((mylat, mylon), (cityinfo['lat'], cityinfo['lon']), unit='mi')
print(f"Bainbridge Island is about {int(round(distance, -1))} miles away from {cityinfo['city']}")

