from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

# tested on llamacpp version==0.2.27
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" poetry run pip install llama-cpp-python==0.2.27 --force-reinstall --no-cache-dir
import sys
import datetime
import asyncio
import random
import requests
from termcolor import colored
import logging
import json
import time
from rich.logging import RichHandler
from pydantic import Field
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer
from collections.abc import Iterator, AsyncIterator
from typing import Any, List, Union, Optional, Sequence, Mapping, Literal, Dict
from rich import pretty
from huggingface_hub import hf_hub_download
from pavai.shared.llm.local.functionary.prompt_template import get_prompt_template_from_tokenizer
from chatlab import FunctionRegistry, tool_result
import os
import traceback
# from dotenv import dotenv_values
# system_config = {
#     **dotenv_values("env.shared"),  # load shared development variables
#     **dotenv_values("env.secret"),  # load sensitive variables
#     **os.environ,  # override loaded values with environment variables
# }

# pretty.install()
# logging.basicConfig(level=logging.INFO, format="%(message)s",
#                     datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
# logger = logging.getLogger(__name__)

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


max_allow_context = config["MODEL_CONTEXT_SIZE"]

LLM_OPTIONS = {
    "n_ctx": 7360,
    "n_gpu_layers": 33,
    "embedding": True,
    "verbose": True,
    "n_threads": 6,
    "chat_format": "llama-2",
    "split_mode:": 0,
    "verbose": False
}

FUNCTIONARY_OPTIONS = {
    "n_ctx": 7360,
    "n_gpu_layers": 33,
    "embedding": True,
    "verbose": True,
    "n_threads": 6,
    "chat_format": "functionary-v2",
    "split_mode:": 0,
    "verbose": False
}

LAVA_OPTIONS = {
    "n_ctx": 7360,
    "n_gpu_layers": 33,
    "embedding": True,
    "verbose": True,
    "n_threads": 8,
    #    "chat_format": "llava",
    "split_mode:": 0,
    "verbose": False,
}

CHAT_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "typical_p": 1,
    "stream": False,
    "max_tokens": 256
}

HF_CACHE_DIR = config["HF_CACHE_DIR"]  # "resources/models"

##
# Functions
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


class LlamaCppLocal(Singleton):

    def __init__(self, options: dict = None):
        """constructor"""
        self.options = options
        self.download_models()
        # So we will use tokenizer from HuggingFace
        # self.tokenizer = AutoTokenizer.from_pretrained(model_repo, legacy=True, cache_dir=cache_dir)
        # registry
        self.registry = FunctionRegistry()
        # name to function mappping
        self.available_functions = {}
        # register defaults
        self._register_defaults()
        self.last_used_model = None
        self.llm = None
        logger.info("LlamaCppLocal instance created")

    def load_model_multimodal(self, options: dict = LAVA_OPTIONS):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        self.download_multimodal_model()
        chat_handler = Llava15ChatHandler(clip_model_path=self.project_file)
        self.llm = Llama(model_path=self.model_file,
                         chat_handler=chat_handler, **options)

    def download_multimodal_model(self):
        # Download a single file
        self.model_file = hf_hub_download(
            repo_id=config["LOCAL_MM_REPO_ID"], filename=config["LOCAL_MM_REPO_MODEL_FILE"], cache_dir=HF_CACHE_DIR)
        self.project_file = hf_hub_download(
            repo_id=config["LOCAL_MM_REPO_ID"], filename=config["LOCAL_MM_REPO_PROJECT_FILE"], cache_dir=HF_CACHE_DIR)
        print("downloaded model file: ", self.model_file)
        print("downloaded project file: ", self.project_file)
        return self.model_file, self.project_file

    def load_default_model(self, options: dict = LLM_OPTIONS):
        from llama_cpp import Llama
        self.model_file = self.download_default_model()
        options = LLM_OPTIONS
        chat_format = config["LOCAL_DEFAULT_REPO_CHAT_FORMAT"]
        if chat_format is not None:
            options["chat_format"] = chat_format
        options["tokenizer"] = None
        options["chat_handler"] = None
        self.llm = Llama(model_path=self.model_file, **options)
        return self.llm

    def download_default_model(self):
        # Download a single file
        model_file = hf_hub_download(
            repo_id=config["LOCAL_DEFAULT_REPO_ID"], filename=config["LOCAL_DEFAULT_REPO_MODEL_FILE"], cache_dir=HF_CACHE_DIR)
        print("downloaded model file: ", model_file)
        return model_file

    def download_functionary_model(self):
        # Download a single file
        model_file = hf_hub_download(
            repo_id=config["LOCAL_FUN_REPO_ID"], filename=config["LOCAL_FUN_REPO_MODEL_FILE"], cache_dir=HF_CACHE_DIR)
        print("downloaded functionary model file: ", model_file)
        return model_file

    def download_models(self, options: dict = None):
        # multimodal
        self.download_multimodal_model()
        # text model
        self.download_default_model()
        # functionary model
        self.download_functionary_model()

    def switch_model_default(self, options: dict = LLM_OPTIONS):
        logger.warn("switch model default")
        self.llm = None
        self._release_model()
        self.load_default_model()

    def load_functionary_model(self, options: dict = LLM_OPTIONS):
        # from llama_cpp.llama_tokenizer import LlamaHFTokenizer
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        logger.warn("switch model functionary")
        cache_dir = config["HF_CACHE_DIR"]
        repo_id = config["LOCAL_FUN_REPO_ID"]
        filename = config["LOCAL_FUN_REPO_MODEL_FILE"]
        chat_format = config["LOCAL_FUN_REPO_CHAT_FORMAT"]
        # self.tokenizer=LlamaHFTokenizer.from_pretrained(repo_id)
        self.download_functionary_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id, legacy=True, cache_dir=cache_dir)
        if cache_dir is None:
            cache_dir = config["HF_CACHE_DIR"]
        if chat_format is not None:
            options["chat_format"] = chat_format
        self.llm = Llama(model_path=self.model_file, **options)
        self.prompt_template = get_prompt_template_from_tokenizer(
            self.tokenizer)

    def switch_model_functionary(self, options: dict = LLM_OPTIONS):
        from llama_cpp import Llama
        from transformers import AutoTokenizer
        logger.warn("switch model functionary")
        cache_dir = config["HF_CACHE_DIR"]
        repo_id = config["LOCAL_FUN_REPO_ID"]
        filename = config["LOCAL_FUN_REPO_MODEL_FILE"]
        chat_format = config["LOCAL_FUN_REPO_CHAT_FORMAT"]
        # self.tokenizer=LlamaHFTokenizer.from_pretrained(repo_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            repo_id, legacy=True, cache_dir=cache_dir)
        self.switch_model_dynamic(
            repo_id=repo_id,
            filename=filename,
            chat_format=chat_format,
            tokenizer=self.tokenizer,
            options=options)

        self.prompt_template = get_prompt_template_from_tokenizer(
            self.tokenizer)

    def switch_model_multimodal(self, options: dict = LAVA_OPTIONS):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        self.llm = None
        self._release_model()
        logger.warn("switch model multimodal")
        # chat_format = config["LOCAL_MM_REPO_CHAT_FORMAT"]
        self.download_multimodal_model()
        options = LAVA_OPTIONS
        chat_handler = Llava15ChatHandler(clip_model_path=self.project_file)
        self.llm = Llama(model_path=self.model_file,
                         chat_handler=chat_handler,
                         logits_all=True, **options)

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

    def openai_chat_completion(self, prompt: str, history: list = None, model: str = None):
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

    def text_chat(self, user_prompt: str, system_prompt: str = None, history: list = [], options: dict = None):
        """text chat with LLM"""
        t0 = time.perf_counter()
        reply = None
        messages = []
        try:
            # self.load_model(LLM_OPTIONS)
            print(colored(f"User: {user_prompt}",
                  "light_blue", attrs=["bold"]))
            messages = self.add_messages(
                history=history, system_prompt=system_prompt, user_prompt=user_prompt)
            # Perform First Call
            response = self.llm.create_chat_completion(
                messages=messages, **options)
            messages.append(response["choices"][0]["message"])
            reply = response["choices"][0]["message"]["content"]
            # if reply is not None:
            print(colored(f"Assistant: {reply}",
                  "light_magenta", attrs=["bold"]))
            # else:
            #     print(colored(f"Assistant: no response return", "light_red", attrs=["bold"]))
            logger.debug(f"chat-call-1 response: {response}")
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logger.error(f"textchat has an error occurred {e.args}")
            #raise e
        t1 = time.perf_counter()
        logger.info(f"chat-call-1 took {t1-t0:.2f} seconds")
        return reply, messages

    def image_chat(self, user_prompt: str = None, system_prompt: str = None, history: list = None, options: dict = None, image_url: str = None, base64_image=None):
        """multimodal chat with LLM"""
        t0 = time.perf_counter()
        reply = None
        messages = []
        try:
            # self.load_model_llava(LAVA_OPTIONS)
            print(colored(f"User: {user_prompt}",
                  "light_blue", attrs=["bold"]))
            if image_url is not None:
                image_list = image_url
            else:
                image_list = None
            messages = self.add_messages(history=history, system_prompt=system_prompt,
                                         user_prompt=user_prompt,
                                         base64_image=base64_image,
                                         image_list=image_list)
            response = self.llm.create_chat_completion(
                messages=messages, **options)
            messages.append(response["choices"][0]["message"])
            reply = response["choices"][0]["message"]["content"]
            print(colored(f"Assistant: {reply}",
                  "light_magenta", attrs=["bold"]))
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logger.error(f"image_chat has an error occurred {e.args}")
            #raise e
        t1 = time.perf_counter()
        logger.info(f"chat-call-1 took {t1-t0:.2f} seconds")
        return reply, messages

    def chat(self, user_prompt: str = None, system_prompt: str = None, history: list = None,
             image_url: str = None, base64_image=None, tools: list = None, options: dict = None):
        """multimodal chat with LLM"""
        t0 = time.perf_counter()
        reply = None
        messages = []
        if tools is not None:
            # functionary call
            if self.last_used_model != "functionarychat":
                self.switch_model_functionary()
            reply, messages = self.functionary_chat(
                user_prompt=user_prompt, system_prompt=system_prompt, history=history)
            self.last_used_model = "functionarychat"
        elif image_url is not None or base64_image is not None:
            # multimodalchat call
            if self.last_used_model != "multimodalchat":
                self.switch_model_multimodal()
            reply, messages = self.image_chat(
                user_prompt=user_prompt, image_url=image_url, base64_image=base64_image, system_prompt=system_prompt, history=history, options=options)
            self.last_used_model = "multimodalchat"
        else:
            # textchat call
            if self.last_used_model != "textchat":
                self.switch_model_default()
            reply, messages = self.text_chat(
                user_prompt=user_prompt, system_prompt=system_prompt, history=history, options=options)
            self.last_used_model = "textchat"
        return reply, messages

    def _release_model(self):
        # unload
        if self.llm is not None:
            self.llm.reset()
            self.llm = None
            del self.llm

    def switch_model_dynamic(self, repo_id: str = None, filename: str = None, chat_format: str = None, tokenizer=None, cache_dir: str = None, options: dict = None):
        from llama_cpp import Llama
        if cache_dir is None:
            cache_dir = config["HF_CACHE_DIR"]
        if chat_format is not None:
            options["chat_format"] = chat_format
        if tokenizer is not None:
            options["tokenizer"] = tokenizer
        # unload
        self._release_model()
        self.model_file = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        print("downloaded model file: ", self.model_file)
        self.llm = Llama(model_path=self.model_file, **options)

    def _functionary_create(self, messages: List = Field(default_factory=list), tools: List = Field(default_factory=list), model="functionary-small-v2.2"):
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
            second_response = self._functionary_create(
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

    def functionary_chat(self, user_prompt: str, system_prompt: str = None, history: list = []):
        """chat with function call"""
        print(colored(f"User: {user_prompt}", "light_blue", attrs=["bold"]))
        messages = self.add_messages(
            history=history, system_prompt=system_prompt, user_prompt=user_prompt)
        t0 = time.perf_counter()
        # Perform First Call
        response = self._functionary_create(
            messages=messages, tools=self.registry.tools)
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

    def load_image(self, image_url: str) -> bytes:
        if image_url.startswith("data:"):
            import base64
            image_bytes = base64.b64decode(image_url.split(",")[1])
            return image_bytes
        else:
            import urllib.request
            with urllib.request.urlopen(image_url) as f:
                image_bytes = f.read()
                return image_bytes

    def encode_image(image_path):
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def api_calling_v2(self,
                       api_host: str = None,
                       api_key: str = "EMPTY",
                       active_model: str = "zephyr:latest",
                       prompt: str = None,
                       history: list = None,
                       system_prompt: str = None,
                       top_p: int = 1,
                       max_tokens: int = 256,
                       temperature: float = 0.2,
                       stop_words=["<"],
                       presence_penalty: int = 0,
                       frequency_penalty: int = 0
                       ):
        if isinstance(stop_words, str):
            stop_words = [stop_words]

        chat_options = {
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "max_tokens": max_tokens
        }
        reply, messages = self.chat(user_prompt=prompt, system_prompt=system_prompt,
                                    history=history, options=chat_options)

        reply_status = "ok"
        reply_text = reply
        reply_messages = self.add_messages(
            ai_prompt=reply_text, history=messages)
        return reply_text, reply_messages, reply_status

    def distilled_history(self,
                          api_host: str = None,
                          api_key: str = "EMPTY",
                          active_model: str = "zephyr:latest",
                          prompt: str = None,
                          chatbot: list = [],
                          history: list = [],
                          system_prompt: str = None,
                          top_p: int = 1,
                          max_tokens: int = 256,
                          temperature: float = 0.2,
                          stop_words=["<"],
                          presence_penalty: int = 0,
                          frequency_penalty: int = 0,
                          ):
        logger.warn("step.1 distilled current conversation")
        output, output_messages, output_status = self.api_calling_v2(
            api_host=api_host,
            api_key=api_key,
            active_model=active_model,
            prompt="summarize current conversation in precise format without losing key information.",
            history=history,
            system_prompt=system_prompt,
            top_p=top_p,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_words=stop_words,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        return output, output_messages, output_status

    def message_and_history_v2(self,
                               api_host: str = None,
                               api_key: str = "EMPTY",
                               active_model: str = "zephyr:latest",
                               prompt: str = None,
                               chatbot: list = [],
                               history: list = [],
                               system_prompt: str = None,
                               top_p: int = 1,
                               max_tokens: int = 256,
                               temperature: float = 0.2,
                               stop_words=["<"],
                               presence_penalty: int = 0,
                               frequency_penalty: int = 0,
                               ):
        t0 = time.perf_counter()
        chatbot = chatbot or []
        logger.debug("#########################################")
        logger.info(prompt)
        output = ""
        output_messages = history
        try:
            conversation_tokens = self.count_messages_token(history)
            max_allow_context = config["MODEL_CONTEXT_SIZE"]
            if int(conversation_tokens) > int(max_allow_context):
                logger.warn(
                    "current conversation tokens exceed context window allow.")
                output, output_messages, output_status = self.distilled_history(
                    api_host=api_host,
                    api_key=api_key,
                    active_model=active_model,
                    prompt=prompt,
                    history=history,
                    system_prompt=system_prompt,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_words=stop_words,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )
                chatbot = []
                chatbot.append((prompt, output))
                history = output_messages

            logger.info("next continue process user prompt.")
            output, output_messages, output_status = self.api_calling_v2(
                api_host=api_host,
                api_key=api_key,
                active_model=active_model,
                prompt=prompt,
                history=history,
                system_prompt=system_prompt,
                top_p=top_p,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_words=stop_words,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
            )
        except Exception as e:
            print(e.args)
            print(traceback.print_exc())
            output = str(e.args)
            output_status="failed"

        chatbot.append((prompt, output))
        # logger.debug("------------------")
        logger.debug(output)
        # logger.info("*********************")
        # logger.debug(output_messages)
        logger.info("*********************")
        tokens = self.count_messages_token(history)
        t1 = time.perf_counter()
        took = (t1-t0)
        output_status = f"<i>conversation tokens {conversation_tokens} | api status: {output_status} | took {took:.2f} seconds</i>"
        logger.info(output_status)
        return chatbot, output_messages, output_status

    def list_models(self):
        modelnames = []
        modelnames.append(config["LOCAL_MM_REPO_ID"])
        modelnames.append(config["LOCAL_DEFAULT_REPO_ID"])
        modelnames.append(config["LOCAL_FUN_REPO"])
        return modelnames

    def _test_function_call(self):
        response, messages = self.chat(
            "what's time is now?", tools="Yes", options=CHAT_OPTIONS)
        print(response)
        return response

    def _test_multimodal_raw(self):
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        # raw test
        model_file, project_file = llama.download_multimodal_model()
        chat_handler = Llava15ChatHandler(clip_model_path=project_file)
        llm = Llama(model_path=model_file,
                    chat_handler=chat_handler,
                    n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
                    logits_all=True,  # needed to make llava work
                    )
        reply = llm.create_chat_completion(
            model="llava",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png",
                            },
                        },
                        {"type": "text", "text": "What does the image say. Format your response as a json object with a single 'text' key."},
                    ],
                }
            ],
        )
        print(reply["choices"][0]["message"]["content"])

    def _test_multimodal_weburl(self):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"
        # with image url
        image_url = "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png"
        reply_text, messages = llama.chat(user_prompt="what is this image about?",
                                          system_prompt=system_prompt,
                                          image_url=image_url,
                                          tools=None, history=[],
                                          options=CHAT_OPTIONS)
        print(reply_text)

    def _test_multimodal_localfile(self):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"
        # with image url
        image_url = "file:///home/pop/software_engineer/pavai-workspace/samples/Bill_Image_Receipt.png"
        reply_text, messages = llama.chat(user_prompt="what is this image about?",
                                          system_prompt=system_prompt,
                                          image_url=image_url,
                                          tools=None, history=[],
                                          options=CHAT_OPTIONS)
        print(reply_text)

    def _test_multimodal_base64(self):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"
        # with image base64
        image64 = "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"
        reply_text, messages = llama.chat(user_prompt="what is this image about?",
                                          system_prompt=system_prompt,
                                          base64_image=image64,
                                          tools=None, history=[],
                                          options=CHAT_OPTIONS)
        print(reply_text)

    def _test_mixed_chatting(self):
        system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. The assistant calls functions with appropriate input when necessary"
        # for i in range(10):
        # test UI call 1
        reply_text, reply_messages, reply_status = llama.api_calling_v2(
            prompt="hello", system_prompt=system_prompt, history=[])
        print(reply_text)

        # test UI call 2
        chatbot, output_messages, output_status = llama.message_and_history_v2(prompt="hello AI",
                                                                               system_prompt=system_prompt,
                                                                               history=[])
        print(chatbot, "\n", output_status)

        # test direct call - no tooling
        reply_text, messages = llama.chat(user_prompt="why the sky is blue",
                                          system_prompt=system_prompt,
                                          tools=None, history=[], options=CHAT_OPTIONS)
        print(reply_text)

        # with tool calling
        reply_text, messages = llama.chat(user_prompt="what time is now?",
                                          system_prompt=system_prompt,
                                          tools="Yes", history=[], options=CHAT_OPTIONS)
        print(reply_text)

        # image
        self._test_multimodal_base64()

    def encode_image(self, image_path):
        import base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def upload_image(self,
                     api_host: str = None,
                     api_key: str = "EMPTY",
                     active_model: str = None,
                     user_prompt: str = None,
                     image=None,
                     chatbot: list = None,
                     history: list = None,
                     system_prompt: list = None):
        """chat with image """
        if image is None:
            print("ignore, missing image removed!")
            return
        user_prompt = user_prompt.strip()
        if len(user_prompt) == 0 and image is not None:
            user_prompt = "What’s in this image?"
        if image is None:
            raise ValueError("missing image")

        image_data = self.encode_image(image)
        messages = self.add_messages(
            user_prompt=user_prompt, system_prompt=system_prompt, history=history, base64_image=image_data)
        t0 = time.perf_counter()
        reply_text, messages = self.chat(user_prompt=user_prompt,
                                          system_prompt=system_prompt,
                                          base64_image=image_data,
                                          tools=None, history=history,
                                          options=CHAT_OPTIONS)
        t1 = time.perf_counter()
        took_time = t1-t0
        reply_status = f"<p align='right'>api done. It took {took_time:.2f}s</p>"
        reply_text = reply_text
        chatbot.append((image, reply_text))
        return chatbot, messages


if __name__ == "__main__":
    llama = LlamaCppLocal(options=LAVA_OPTIONS)
    # llama._test_multimodal_raw()
    # llama._test_multimodal_weburl()
    # llama._test_multimodal_base64()
    # llama._test_multimodal_localfile()
    # llama._test_function_call()
    llama._test_mixed_chatting()
