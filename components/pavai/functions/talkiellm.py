from typing import Any, List, AnyStr, Union, Optional, Sequence, Mapping, Literal, Dict
from pavai.functions.functionary.prompt_template import get_prompt_template_from_tokenizer
import yfinance as yf
from collections.abc import Iterator, AsyncIterator
from chatlab import FunctionRegistry, tool_result
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from openai import OpenAI
import tiktoken
from termcolor import colored
from pydantic import Field
import requests
import random
import json
import asyncio
from rich.logging import RichHandler
import logging
import os
import time
import datetime
import sys
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)

### Singleton Class ###
class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(
                MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()

##
## Functions ##
##

def get_ticker(company_name: str, country: str = "United States"):
    """Get the company stock exchange ticket name"""
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/122.0 (Windows 12.0; Win64; x64)"
    params = {"q": company_name, "quotes_count": 1, "country": country}
    res = requests.get(url=url, params=params, headers={
                       'User-Agent': user_agent})
    data = res.json()
    company_code = data['quotes'][0]['symbol']
    return company_code


def get_ticket_price(stock_symbol: str = Field(description="the company stock symbol")):
    """get stock price"""
    ticker = yf.Ticker(stock_symbol)
    todays_data = ticker.history(period='1d')
    return round(todays_data['Close'][0], 2)


def get_stock_price(company_name: str = Field(description="the company stock symbol")):
    """find company stock ticket name then get current stock price"""
    try:
        stock_ticket = get_ticker(company_name)
        print(f"found ticket name: {stock_ticket}")
        price = get_ticket_price(stock_ticket)
        return f"current price {price}"
    except Exception as e:
        print(e)
        return f"unable get company stock price due error!"


def get_time_now():
    """Get time now"""
    return str(datetime.datetime.now())


def get_current_weather(
    location: str = Field(
        description="The city and state, e.g., San Francisco, CA"),
    unit: str = Field(description="C or F")
):
    """Get the current weather"""
    return {
        "temperature": 75 + random.randint(-5, 5),
        "units": unit,
        "weather": random.choice(["sunny", "cloudy", "rainy", "windy"]),
    }

# testing
# get_ticker("apple")
# get_stock_price(company_name="apple")

LLM_OPTIONS = {
        "n_ctx": 4096,
        "n_gpu_layers": 33,
        "embedding": True,
        "verbose": True,
        "n_threads": 6,
        "chat_format": "llama-2",
        "split_mode:": 0,
        "chat_handler": None
}

# API_HOST = "http://192.168.0.18:12345"
API_HOST = "http://192.168.0.29:8004"
# LLM_PROVIDER=config["LLM_PROVIDER"]

# API_HOST=config["API_HOST"]
# API_URL_BASE = f"{API_HOST}/v1"
# API_URL_CHAT = f"{API_URL_BASE}/chat/completions"


class TalkieAPI(Singleton):

    def __init__(self, api_base: str = None,
                 use_local: bool = True,
                 options: dict = None,
                 llm_provider: str = "llamacpp"):
        self.use_local = use_local
        self.llm_provider = llm_provider
        # api client
        if not use_local:
            self.client = OpenAI(
                base_url=api_base,
                api_key="pavai",  # required, but unused
            )
        # Model repository on the Hugging Face model hub
        model_repo = "meetkai/functionary-small-v2.2-GGUF"
        # File to download
        # file_name = "functionary-small-v2.2.f16.gguf"
        # file_name = "functionary-small-v2.2.q8_0.gguf"
        file_name = "functionary-small-v2.2.q4_0.gguf"
        # Download the file
        local_file_path = hf_hub_download(
            repo_id=model_repo, filename=file_name)
        # You can download gguf files from https://huggingface.co/meetkai/functionary-7b-v2-GGUF/tree/main
        self.llm = Llama(model_path=local_file_path, **options)
        # Create tokenizer from HF.
        # We found that the tokenizer from llama_cpp is not compatible with tokenizer from HF that we trained
        # The reason might be we added new tokens to the original tokenizer
        # So we will use tokenizer from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo, legacy=True)
        # prompt_template will be used for creating the prompt
        self.prompt_template = get_prompt_template_from_tokenizer(
            self.tokenizer)
        # registry
        self.registry = FunctionRegistry()
        # name to function mappping
        self.available_functions = {}
        # register defaults
        self._register_defaults()
        logger.info("TalkierAPI instance created")

    def create(self, messages: List = Field(default_factory=list), tools: List = Field(default_factory=list), model="functionary-small-v2.2"):
        """Creates a model response for the given chat conversation. OpenAI's `chat.create()` function."""
        # Create the prompt to use for inference
        prompt_str = self.prompt_template.get_prompt_from_messages(
            messages + [{"role": "assistant"}], tools
        )
        token_ids = self.tokenizer.encode(prompt_str)

        gen_tokens = []
        # Get list of stop_tokens
        stop_token_ids = [
            self.tokenizer.encode(token)[-1]
            for token in self.prompt_template.get_stop_tokens_for_generation()
        ]

        # We use function generate (instead of __call__) so we can pass in list of token_ids
        for token_id in self.llm.generate(token_ids, temp=0):
            if token_id in stop_token_ids:
                break
            gen_tokens.append(token_id)

        llm_output = self.tokenizer.decode(gen_tokens)
        # parse the message from llm_output
        response = self.prompt_template.parse_assistant_response(llm_output)
        return response

    def _execute_tool_calls(self, tool_calls: list) -> list:
        """executing function calls"""
        result = []
        i = 1
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_to_call = self.available_functions[function_name]
            function_args = json.loads(tool_call["function"]["arguments"])
            logger.debug("tool calls functions:")
            logger.debug(f"<tool_function_{i}:{function_name}>")
            logger.debug(f" |params: {function_args}")
            logger.debug(f"</tool_function_{i}:{function_name}>")
            match function_name:
                case "get_stock_price":
                    logger.info(f"________________________")
                    logger.info(f"<invoke>{function_name}</invoke>")
                    if len(function_args) > 0:
                        function_response = function_to_call(
                            company_name=function_args.get("company_name"))
                        logger.info(f"<result>{function_response}</result>")
                        logger.info(f"________________________")
                case "get_ticker":
                    logger.info(f"________________________")
                    logger.info(f"<invoke>{function_name}</invoke>")
                    if len(function_args) > 0:
                        function_response = function_to_call(
                            company_name=function_args.get("company_name"))
                        logger.info(f"<result>{function_response}</result>")
                        logger.info(f"________________________")
                case "get_current_weather":
                    logger.info(f"________________________")
                    logger.info(f"<invoke>{function_name}</invoke>")
                    if len(function_args) > 0:
                        function_response = function_to_call(
                            location=function_args.get("location"),
                            unit=function_args.get("unit"),)
                        logger.info(f"<result>{function_response}</result>")
                        logger.info(f"________________________")
                case "get_time_now":
                    logger.info(f"________________________")
                    logger.info(f"<invoke>{function_name}</invoke>")
                    function_response = function_to_call()
                    logger.info(f"<result>{function_response}</result>")
                    logger.info(f"________________________")
                case _:
                    logger.info(f"***********************************")
                    logger.warn(f"| Not function match found: {function_name}")
                    logger.info(f"***********************************")

            result.append(
                {
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
            i += 1
        return result

    def _register_defaults(self):
        self.register_functions("get_current_weather", get_current_weather)
        self.register_functions("get_time_now", get_time_now)
        self.register_functions("get_stock_price", get_stock_price)

    def register_functions(self, funname, pyfuncall):
        """register a function name and a callable python function"""
        self.registry.register(pyfuncall)
        self.available_functions[funname] = pyfuncall
        logger.info(f"register function: {funname}")
        logger.debug(json.dumps(self.registry.tools, indent=2))

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def count_messages_token(self, messages: list):
        """perform a count of tokens in a message list"""
        if messages is None:
            return 0
        content = ""
        if isinstance(messages, str):
            content = " ".join(messages)
        else:
            for m in messages:
                content = content+str(m)
        return self.num_tokens_from_string(content)

    def add_messages(self, history: list = None, system_prompt: str = None, user_prompt: str = None, ai_prompt: str = None, image_list: list = None, base64_image=None):
        """create a message list """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if image_list:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_list
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ], })
        if base64_image:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ], })
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
        if ai_prompt:
            messages.append({"role": "assistant", "content": ai_prompt})
        if history:
            return history+messages
        else:
            return messages

    def openai_chat_completion(self, prompt: str, history: list = None, model: str = "zephyr:latest"):
        """perform a openai chat call"""
        messages = self.add_messages(user_prompt=prompt, history=history)
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.2,
        )
        reply_text = response.choices[0].message.content
        reply_messages = self.add_messages(
            ai_prompt=reply_text, history=messages)
        return reply_text, reply_messages

    def _process_response_tools(self, response, messages):
        """process response with tool calling """
        tool_calls = response["tool_calls"]
        # Step 2: check if the model wanted to call a function
        if tool_calls:
            t0 = time.perf_counter()
            logger.info(f"Tool calls required {response}")
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            # messages.append(response)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            result = self._execute_tool_calls(tool_calls)
            if len(result) > 0:
                messages.extend(result)
            second_response = self.create(
                messages=messages, tools=self.registry.tools)
            # print(second_response["content"])
            # print(colored(f"Assistant: {second_response['content']}", "light_magenta", attrs=["bold"]))
            logger.debug(f"Assistant: {second_response['content']}")
            # extend conversation with assistant's reply
            messages.append(second_response)
            t1 = time.perf_counter()
            logger.info(f"chatapi-call-2 took {t1-t0:.2f} seconds")
            return second_response['content'], messages
        else:
            logger.info(f"No tool calling needed!")
            return response['content'], messages

    def chat(self, user_prompt: str, system_prompt: str = None, history: list = []):
        """chat with LLM"""
        print(colored(f"User: {user_prompt}", "light_blue", attrs=["bold"]))
        messages = self.add_messages(
            history=history, system_prompt=system_prompt, user_prompt=user_prompt)
        t0 = time.perf_counter()
        # Perform First Call
        response = self.create(messages=messages, tools=self.registry.tools)
        messages.append(response)
        if response.get("content") is not None:
            print(
                colored(f"Assistant: {response['content']}", "light_magenta", attrs=["bold"]))
        else:
            print(colored(
                f"Assistant: {response['tool_calls']}", "light_magenta", attrs=["bold"]))
        logger.debug(f"chatapi-call-1 response: {response}")
        t1 = time.perf_counter()
        logger.info(f"chatapi-call-1 took {t1-t0:.2f} seconds")
        # Perform Second Call for Tools
        reply, messages = self._process_response_tools(response, messages)
        logger.debug(f"chatapi-call-2 response: {response}")
        if reply is not None:
            print(colored(f"Assistant: {reply}", "green", attrs=["bold"]))
        return reply, messages


if __name__ == "__main__":
    # enum llama_split_mode {
    #     LLAMA_SPLIT_NONE    = 0, // single GPU
    #     LLAMA_SPLIT_LAYER   = 1, // split layers and KV across GPUs
    #     LLAMA_SPLIT_ROW     = 2, // split rows across GPUs
    # };

    functionary = TalkieAPI(api_base=API_HOST, use_local=False, options=LLM_OPTIONS)

    for i in range(3):
        functionary = TalkieAPI(api_base=API_HOST, use_local=False, options=LLM_OPTIONS)

        response, messages = functionary.chat("check apple stock price?")
        print(response)

        response, messages = functionary.chat("check Microsoft stock price?")
        print(response)

        response, messages = functionary.chat("how is the weather looks like in Toronto?")
        print(response)

        response, messages = functionary.chat("what's time is now?")
        print(response)

        response, messages = functionary.chat("happy monday")
        print(response)
