## pip install forex-python

import requests
import json
import sys
from forex_python.converter import CurrencyRates

currency_code = "GBP"

schema = {
    "currency": {
        "type": "string",
        "description": "Currency code"
    },
    "rate": {
        "type": "float",
        "description": "Exchange rate against USD"
    },
    "date": {
        "type": "string",
        "description": "Date of the rate"
    }
}

payload = {
    "model": "openhermes:latest",
    "messages": [
        {
            "role": "system",
            "content": f"You are a helpful AI assistant. The user will enter a currency code and the assistant will return the exchange rate against USD and the date of the rate. Output in JSON using the schema defined here: {json.dumps(schema)}."
        },
        {"role": "user", "content": "EUR"},
        {"role": "assistant", "content": json.dumps({"currency": "EUR", "rate": 0.85, "date": "13-02-2024"})},
        {"role": "user", "content": currency_code}
    ],
    "format": "json",
    "stream": False
}

response = requests.post("http://192.168.0.18:12345/api/chat", json=payload)
currency_info = json.loads(response.json()["message"]["content"])

c = CurrencyRates()
rate_to_usd = c.get_rate(currency_info['currency'], 'USD')

print(f"The exchange rate for {currency_info['currency']} against USD is {rate_to_usd} as of {currency_info['date']}.")

