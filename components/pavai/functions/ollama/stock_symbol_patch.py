from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import yfinance as yf

import instructor

company = "Google"


class StockInfo(BaseModel):
    company: str = Field(..., description="Name of the company")
    ticker: str = Field(..., description="Ticker symbol of the company")


# enables `response_model` in create call
client = instructor.patch(
    OpenAI(
        base_url="http://192.168.0.18:12345/v1",
        api_key="ollama",
    ),
    mode=instructor.Mode.JSON,
)

resp = client.chat.completions.create(
    model="openhermes:latest",
    messages=[
        {
            "role": "user",
            "content": f"Return the company name and the ticker symbol of the {company}."
        }
    ],
    response_model=StockInfo,
    max_retries=10
)
print(resp.model_dump_json(indent=2))
stock = yf.Ticker(resp.ticker)
hist = stock.history(period="1d")
stock_price = hist['Close'].iloc[-1]
print(f"The stock price of the {resp.company} is {stock_price}. USD")
