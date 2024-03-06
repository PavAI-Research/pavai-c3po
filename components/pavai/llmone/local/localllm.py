from dotenv import set_key
from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import gc
import os
import time
import torch
import requests
import shutil
import functools
from pathlib import Path
from urllib.request import urlopen
from transformers import pipeline
from huggingface_hub import hf_hub_download, snapshot_download
from typing import (List, Optional, Union, Generator,
                    Sequence, Iterator, Deque, Tuple, Dict, Callable)
from llama_cpp import (
    Llama, LogitsProcessorList, StoppingCriteriaList, CreateCompletionResponse, LogitsProcessor,
    StoppingCriteria, CreateCompletionStreamResponse, LlamaGrammar,
    ChatCompletionRequestMessage, ChatCompletionFunction, ChatCompletionRequestFunctionCall,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionRequestResponseFormat,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse)

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Union, Optional, overload
from typing_extensions import Literal
import numpy.typing as npt
import numpy as np
from openai import OpenAI
import numpy.typing as npt
import numpy as np
import httpx
import json
import copy
from pavai.llmone.llmtokens import HistoryTokenBuffer
import traceback

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.1"

"""
## Manual download model files

pip install huggingface-hub

huggingface-cli download \
  TheBloke/MistralLite-7B-GGUF \
  mistrallite.Q4_K_M.gguf \
  --local-dir downloads \
  --local-dir-use-symlinks False

## install llama-cpp-python
pip install llama-cpp-python  --upgrade --force-reinstall --no-cache-dir

"""

DEFAULT_LLM_SYSTEM_PROMPT = "You are an intelligent AI assistant who can help answer user query."
DEFAULT_LLM_MODEL_PATH = "resources/models/llm/"

DEFAULT_LLM_MODEL_FILE = "zephyr-7b-beta.Q4_K_M.gguf"
DEFAULT_LLM_MODEL_WEB_URL = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf"
# DEFAULT_LLM_MODEL_FILE = "zephyr-7b-beta.Q5_K_M.gguf"
DEFAULT_LLM_CHAT_FORMAT = "chatml"  #"llama-2"
DEFAULT_LLM_USE_GPU_LAYERS = 35
DEFAULT_LLM_CONTEXT_SIZE = 7892  # not 8192 discount special tokens

DEFAULT_LLM_MODEL_INFO=[DEFAULT_LLM_MODEL_FILE,DEFAULT_LLM_MODEL_WEB_URL,DEFAULT_LLM_CHAT_FORMAT]

system_prompt_assistant = """
You are an artificial intelligence assistant, trained to engage in human-like voice conversations and to serve as a research assistant. 
You excel as private assistant, and you are a leading expert in writing, cooking, health, data science, world history, software programming,food,cooking, sports, movies, music, news summarizer, biology, engineering, party planning, industrial design, environmental science, physiology, trivia, personal financial advice, cybersecurity, travel planning, meditation guidance, nutrition, captivating storytelling, fitness coaching, philosophy, quote and creative writing generation, and more.

Your goal is to assist the user in a step-by-step manner through human-like voice conversations, answering user-specific queries and challenges. 
Pause often (at a minimum, after every step) to seek user feedback or clarification.

1. Define - The first step in any conversation is to define the user's request, identify the query or opportunity that needs user clarification or attention. Prompt the user to think through the next steps to define their challenge. Refrain from answering these for the user. You may offer suggestions if asked to.
2. Analyze - Analyze the essential user intentions, identify the intentions and entities, and determine the challenge that must be addressed.
3. Discover - Search for the best models that need to address the same functions as your solution.
4. Abstract - Study the essential features or mechanisms to generate a response that meets user expectations.
5. Emulate human-like natural conversation patterns - creating nature-inspired human responses.

Human-like conversation response resembles a natural, interactive communication between two people. 
It involves active listening, understanding the context, and responding in a way that is relevant, coherent, and empathetic. 

Here are characteristics of human-like conversations:

1. Active listening: The assistant should demonstrate that it is listening to the user by acknowledging their statements and asking relevant questions.
2. Contextual understanding: The assistant should understand the context of the conversation and respond accordingly. It should be able to follow the conversation's thread and build upon previous exchanges.
3. Empathy: The assistant should be able to understand the user's emotions and respond in a way that is sensitive to their feelings.
4. Relevance: The assistant's responses should be relevant to the user's queries and challenges. It should avoid providing irrelevant or off-topic information.
5. Coherence: The assistant's responses should be logically consistent and coherent. It should avoid making contradictory statements or jumping from one topic to another without a clear connection.
6. Precision: The assistant's responses should be precise and to the point. It should avoid providing vague or ambiguous answers.
7. Personalization: The assistant's responses should be tailored to the user's preferences, needs, and goals. It should avoid providing generic or one-size-fits-all responses.
8. Engagement: The assistant should engage the user in a conversation that is interesting, informative, and enjoyable. It should avoid being overly formal or robotic.

A human voice conversation is a dynamic and interactive communication between two or more people, characterized by the following elements:

1. Speech: Human voice conversations involve the use of spoken language to convey meaning and intent. The tone, pitch, volume, and pace of speech can convey various emotions, attitudes, and intentions.
2. Listening: Human voice conversations require active listening, where the listener pays attention to the speaker's words, tone, and body language to understand their meaning and intent.
3. Turn-taking: Human voice conversations involve a turn-taking structure, where each speaker takes turns to speak and listen. Interruptions, overlaps, and pauses are common features of human voice conversations.
4. Feedback: Human voice conversations involve providing feedback to the speaker, such as nodding, making eye contact, or verbal cues like "uh-huh" or "I see." This feedback helps the speaker to understand if the listener is following the conversation and if their message is being understood.
5. Context: Human voice conversations are situated in a specific context, such as a physical location, social situation, or cultural background. The context can influence the tone, content, and structure of the conversation.
6. Nonverbal communication: Human voice conversations involve nonverbal communication, such as facial expressions, gestures, and body language. These nonverbal cues can convey emotions, attitudes, and intentions that are not expressed verbally.
7. Spontaneity: Human voice conversations are often spontaneous and unplanned, requiring speakers to think on their feet and respond to unexpected questions or comments.

By understanding human voice conversation elements and emulating human-like conversations characteristics, you can create a short, precise and relevant response to the user question in human-like conversation that is engaging, informative, and helpful to the user.
If the text does not contain sufficient information to answer the question, do not make up information. Instead, respond with "I don't know," and please be specific.
"""

system_prompt_default = """
You are an intelligent AI assistant. You are helping user answer query.

If the text does not contain sufficient information to answer the question, do not make up information and give the answer as "I don't know, please be specific.".

Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.
"""

guard_system_prompt=".Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."

guard_system_prompt_assistant=system_prompt_assistant+"\n"+guard_system_prompt+"\n"

safe_system_prompt=system_prompt_default+".\n"+guard_system_prompt+"\n"


system_prompt_basic = """
You are an intelligent AI assistant. You are helping user answer query.
If the text does not contain sufficient information to answer the question, do not make up information and give the answer as "I don't know, please be specific.".
"""

from huggingface_hub import hf_hub_download
import traceback

llm_defaults = {
    "model_runtime": "llamacpp",
    "model_id": None,    
    "model_file_name": "zephyr-7b-beta.Q5_K_M.gguf",
    "model_name_or_path":"TheBloke/zephyr-7B-beta-GGUF",
    "model_file_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf",
    "model_download_path": "zephyr-7b-beta.Q5_K_M.gguf",
    "model_source": "filesystem",
    "model_architecure": "chatml",
    "api_base": None,
    "api_key": None,
    "api_domain": None,
    "api_organization": None,
    "use_device": "cuda",
    "use_torch_type": "float16",
    "model_tokenizer_id_or_path": None,
    "use_flash_attention_2": "",
    "use_download_root": "resources/models/llm/",
    "use_local_model_only": True,
    "use_max_context_size": 8192,
    "use_n_gpu_layers": 38,
    "verbose": True,
    "use_seed": 138
    }

@functools.cache
def get_llm_local_defaults(resource_config:str="resources/config")->str:
    Path.mkdir(resource_config, exist_ok=True)
    return resource_config+"/llm_defaults.json"

@functools.cache
def get_llm_library(resource_config:str="resources/config")->dict:
    if not os.path.exists(resource_config):
        os.mkdir(resource_config)
    library_json_file = resource_config+"/llm_libary.json"
    with open(library_json_file) as file:
        library = json.load(file)
        logger.debug(library)
        return library

class AbstractLLMClass(ABC):
    """
    The AbstractLLMClass defines a template method that contains a skeleton of
    LLM chat completion algorithm, composed of calls to (usually) abstract primitive
    operations.
    """

    def __init__(self,
                 model_runtime: str = "llamacpp",
                 model_id: str = None,
                 model_file_name: str = None,
                 model_name_or_path: str = None,                 
                 model_file_url: str = None,
                 model_download_path: str = None,                 
                 model_source: str = "filesystem",
                 model_architecure: str = "llama-2",
                 api_base: str = None,
                 api_key: str = "EMPTY",
                 api_domain: str = None,
                 api_organization: str = None,
                 use_device: str = "cpu",
                 use_torch_type: str = "float16",
                 model_tokenizer_id_or_path: str = None,
                 use_flash_attention_2: bool = False,
                 use_download_root: str = "resources/models/llm",
                 use_local_model_only: bool = False,
                 use_max_context_size: int = 4096,
                 use_n_gpu_layers: int = 0,
                 use_seed: int = 123,
                 verbose: bool = True) -> None:
        super().__init__()
        self._model_runtime = model_runtime
        self._model_id = model_id
        self.model_name_or_path = model_name_or_path
        self._model_file_name = model_file_name        
        self._model_file_url = model_file_url
        self.model_download_path = model_download_path        
        self._model_source = model_source
        self._model_architecure = model_architecure
        self._api_base = api_base
        self._api_key = api_key
        self._api_domain = api_domain
        self._api_organization = api_organization
        self._use_device = use_device
        self._use_torch_type = use_torch_type
        self._use_flash_attention_2 = use_flash_attention_2
        self._model_tokenizer_id_or_path = model_tokenizer_id_or_path
        self._use_task = "text-generation"
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._pipeline = None
        self._download_root = use_download_root
        self._local_model_only = use_local_model_only
        self._max_context_size = use_max_context_size
        self._n_gpu_layers = use_n_gpu_layers
        self._verbose = verbose
        self._seed = use_seed
        self._client = None

    @property
    def model_runtime(self):
        return self._model_runtime

    @model_runtime.setter
    def model_runtime(self, new_name):
        self._model_runtime = new_name

    @property
    def model_file_name(self):
        return self._model_file_name

    @model_file_name.setter
    def model_file_name(self, new_name):
        self._model_file_name = new_name

    @property
    def model_file_url(self):
        return self._model_file_url

    @model_file_url.setter
    def model_file_url(self, new_name):
        self._model_file_url = new_name

    @property
    def model_id(self):
        return self._model_id

    @model_id.setter
    def model_id(self, new_name):
        self._model_id_or_path = new_name

    @property
    def model_tokenizer_id_or_path(self):
        return self._model_tokenizer_id_or_path

    @model_tokenizer_id_or_path.setter
    def model_tokenizer_id_or_path(self, new_name):
        self._model_tokenizer_id_or_path = new_name

    @property
    def use_device(self):
        return self._use_device

    @use_device.setter
    def use_device(self, new_name):
        self._use_device = new_name

    @property
    def use_local_model_only(self):
        return self._local_model_only

    @use_local_model_only.setter
    def use_local_model_only(self, new_name):
        self._local_model_only = new_name

    @property
    def use_download_root(self):
        return self._download_root

    @use_download_root.setter
    def use_download_root(self, new_name):
        self._download_root = new_name

    @property
    def use_torchtype(self):
        return self._use_torchtype

    @use_torchtype.setter
    def use_torchtype(self, new_name):
        self._use_torchtype = new_name

    @property
    def use_input_source(self):
        return self._use_input_source

    @use_input_source.setter
    def use_input_source(self, new_name):
        self._use_input_source = new_name

    @property
    def use_chat_format(self):
        return self._use_chat_format

    @use_chat_format.setter
    def use_chat_format(self, new_name):
        self._use_chat_format = new_name

    @property
    def use_output_format(self):
        return self._use_output_format

    @use_output_format.setter
    def use_output_format(self, new_name):
        self._use_output_format = new_name

    @property
    def use_task(self):
        return self._use_task

    @use_task.setter
    def use_task(self, new_name):
        self._use_task = new_name

    @property
    def use_flash_attention_2(self):
        return self._use_flash_attention_2

    @use_task.setter
    def use_flash_attention_2(self, new_name):
        self._use_flash_attention_2 = new_name

    @property
    def n_gpu_layers(self):
        return self._n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, new_name):
        self._n_gpu_layers = new_name

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, new_name):
        self._verbose = new_name

    @property
    def use_max_context_size(self):
        return self._max_context_size

    @use_max_context_size.setter
    def use_max_context_size(self, new_name):
        self._max_context_size = new_name

    def word_count(self, string):
        return (len(string.strip().split(" ")))

    def simple_completion(self, prompt: str,
                          temperature: float = 0.2,
                          top_p: float = 0.95,
                          top_k: int = 40,
                          min_p: float = 0.05,
                          typical_p: float = 1.0,
                          stream: bool = False,
                          stop_criterias: list = ["\n", "</s>"],
                          max_tokens: Optional[int] = None,
                          repeat_penalty: float = 1.1,
                          output_format: str = "text") -> any:
        pass

    def simple_chat(self, prompt: str, history: list = [],
                    system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                    temperature: float = 0.2,
                    top_p: float = 0.95,
                    top_k: int = 40,
                    min_p: float = 0.05,
                    typical_p: float = 1.0,
                    stream: bool = False,
                    stop_criterias: list = ["\n", "</s>"],
                    max_tokens: Optional[int] = None,
                    repeat_penalty: float = 1.1,
                    output_format: str = "text") -> any:
        pass

    def chat_on_text(self, prompt: str, history: list = [],
                     system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                     stop_criterias=["\n", "</s>"],
                     output_format: str = "text") -> any:
        """
        The template method defines the skeleton of chat .
        """
        logger.debug(f"chat_on_text: {self.use_device}")
        t0 = time.perf_counter()
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        response = self._pipeline(prompt)
        self.prepare_input()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        history = history+messages
        reply = response["choices"][0]["message"]["content"]
        history.append({"role": "assistant", "content": reply})
        for i in range(0, len(history)-1):
            if history[i]["role"] != "system":
                messages.append(
                    (history[i]["content"], history[i+1]["content"]))
        logger.debug(messages)
        self.prepare_output()
        self.hook1()
        took_in_seconds = time.perf_counter()-t0
        status_msg = f"chat completed took {took_in_seconds:.2f} seconds"
        logger.debug(status_msg)
        logger.debug(status_msg)
        return messages, history, reply

    def chat_on_code(self, user_Prompt: str, history: list = [],
                     system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                     stop_criterias=["\n", "</s>"], output_format: str = "text") -> any:
        logger.debug(f"chat_on_code: {self.use_device}")

    def chat_on_file(self, prompt: str, history: list, system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                     stop_criterias=["\n", "</s>"], output_format: str = "text") -> any:
        """
        The template method defines the skeleton of chat_on_file .
        """
        logger.debug(f"chat_on_file: {self.use_device}")

    def chat_on_audio(self, prompt: str, history: list, system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                      stop_criterias=["\n", "</s>"], output_format: str = "text") -> any:
        """
        The template method defines the skeleton of chat_on_audio .
        """
        logger.debug(f"chat_on_audio: {self.use_device}")
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def chat_on_image(self, prompt: str, history: list, system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                      stop_criterias=["\n", "</s>"], output_format: str = "text") -> any:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        logger.debug(f"chat_on_image: {self.use_device}")

    def chat_on_video(self, prompt: str, history: list, system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                      stop_criterias=["\n", "</s>"], output_format: str = "text") -> any:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        logger.debug(f"chat_on_video: {self.use_device}")

    def completion(self, prompt: str, history: list, task: str, output_format: str = "text") -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def code(self, prompt: str, history: list, task: str, output_format: str = "text") -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def analyze(self, prompt: str, history: list, task: str, output_format: str = "text") -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def summarize(self, prompt: str, history: list, task: str, output_format: str = "text") -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def critique(self, prompt: str, history: list, task: str, output_format: str = "text") -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def create_client(self) -> None:
        pass

    def create_embedding(self,input:str,model:str) -> any:
        pass

    # These operations already have implementations.
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def load_tokenizer(self) -> None:
        pass

    @abstractmethod
    def create_pipeline(self) -> None:
        pass

    # These operations have to be implemented in subclasses.
    @abstractmethod
    def prepare_input(self) -> None:
        pass

    @abstractmethod
    def prepare_output(self) -> None:
        pass

    def hook1(self) -> None:
        pass

    @staticmethod
    def download_file(url, local_path: str = None):
        local_filename = url.split('/')[-1]
        if local_path is not None:
            local_filename = local_path+local_filename
        with requests.get(url, stream=True) as r:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        return local_filename

    @staticmethod
    def match_model_file_chatformat(model_file:str):
        if "zephyr" in model_file.lower():
            return "zephyr"
        elif "mistrallite" in model_file.lower():
            return "mistrallite"        
        elif "open-orca" in model_file.lower():
            return "open-orca"                
        elif "intel" in model_file.lower():
            return "intel"                        
        elif "phind" in model_file.lower():
            return "phind"                        
        elif "snoozy" in model_file.lower():
            return "snoozy"                        
        elif "redpajama" in model_file.lower():
            return "redpajama-incite"                        
        elif "openbuddy" in model_file.lower():
            return "openbuddy"                        
        elif "baichuan" in model_file.lower():
            return "baichuan"                        
        elif "baichuan-2" in model_file.lower():
            return "baichuan-2"                        
        elif "oasst_llama" in model_file.lower():
            return "oasst_llama"                        
        elif "vicuna" in model_file.lower():
            return "vicuna"                        
        elif "qwen" in model_file.lower():
            return "qwen"                        
        elif "alpaca" in model_file.lower():
            return "alpaca"                        
        elif "mixtral" in model_file.lower():
            return "llama-2"            
        elif "mistral" in model_file.lower():
            return "llama-2"                    
        elif "llamaguard" in model_file.lower():
            return "llama-2"                    
        elif "functionary" in model_file.lower():
            return "llama-2" 
        elif "openchat" in model_file.lower():
            return "openchat"         
        elif "saiga" in model_file.lower():
            return "saiga"                 
        elif "pygmalion" in model_file.lower():
            return "pygmalion"                 
        elif "chatglm3" in model_file.lower():
            return "chatglm3"                         
        elif "ggml-model" in model_file.lower():
            return "llava" 
        else:
            return "llama-2"                                

    @staticmethod
    def get_llm_model(model_name_or_path:str=None,
                      model_file: str =None,
                      model_download:str=None,
                      model_path: str=DEFAULT_LLM_MODEL_PATH,                      
                      model_chat_format: str = DEFAULT_LLM_CHAT_FORMAT):

        file_exist = True
        if model_download is not None:
            if not os.path.exists(model_download):
                file_exist = False
        elif not os.path.exists(model_file):
            file_exist = False
        elif not os.path.exists(model_path+model_file):
            file_exist = False       

        if not file_exist:
            gguf_model_file = hf_hub_download(repo_id=model_name_or_path, 
                                                filename=model_file, 
                                                cache_dir=model_path)
            logger.info(f"downloaded model file: {gguf_model_file}")  
            model_download=gguf_model_file        
        else:
            logger.info(f"found local model file: {model_file}")            
        # determine the chatformat
        gguf_chat_format=AbstractLLMClass.match_model_file_chatformat(model_file)
        logger.info(f"matched chat-format found: {gguf_chat_format}")            
        
        return model_download, gguf_chat_format

class LLMllamaLocal(AbstractLLMClass):
    """
    LLMllamaLocal Class override only required class operations.
    """
    def load_model(self) -> None:
        logger.debug(f"LLMllamaLocal: load_model()")

        if self._pipeline is not None:
            return

        if (self.model_id is not None and len(self.model_id.strip()) > 0) and (
                self.model_file_name is not None and len(self.model_id.strip()) > 0):
            # download a file from huggingface hub
            hf_hub_download(repo_id=self.model_id, filename=self.model_file_name,
                            local_dir=self.use_download_root, local_dir_use_symlinks=False)
        elif (self.model_file_name is not None and len(self.model_file_name.strip()) > 0) and (
                self.model_file_name.lower().endswith(".gguf")):
            # GGUF model local file
            gguf_model_file, gguf_chat_format = AbstractLLMClass.get_llm_model(
                model_name_or_path=self.model_name_or_path,
                model_download=self.model_download_path,
                model_file=self.model_file_name,
                model_path=self.use_download_root)
    
            self.model_download_path = gguf_model_file
            self.use_chat_format = gguf_chat_format
        # elif (self.model_file_url is not None and len(self.model_file_url.strip()) > 0) and (self.model_file_name.lower().endswith(".gguf")):
        #     # GGUF model remote file
        #     gguf_model_file, gguf_chat_format = AbstractLLMClass.download_file(self.model_file_url,
        #                                                                        model_path=self.use_download_root)
        #     self.model_file_name = gguf_model_file
        #     self.use_chat_format = gguf_chat_format
        elif (self.model_id is not None and len(self.model_id.strip()) > 0):
            # download an entire repository
            snapshot_download(repo_id=self.model_id)
        else:
            raise Exception("Unable to load model due missing information!")

    def load_tokenizer(self) -> None:
        logger.debug(f"LLMllamaLocal: load_tokenizer()")

    def create_pipeline(self) -> None:
        logger.debug(f"LLMllamaLocal: create_pipeline {self.model_id}")
        if self._pipeline is None:
            self.use_cpu_threads = int(os.cpu_count()/2)  # use half only
            logger.debug(f"create_pipeline on device: {self.use_device}")
            self._pipeline = Llama(model_path=self.model_download_path,
                                   chat_format=self.use_chat_format,
                                   n_ctx=self.use_max_context_size,
                                   n_gpu_layers=self.n_gpu_layers,
                                   n_threads=self.use_cpu_threads,
                                   seed=self._seed,
                                   embedding=True,
                                   verbose=False)

    def prepare_input(self) -> None:
        logger.debug(f"LLMllamaLocal: prepare_input()")

    def prepare_output(self) -> None:
        logger.debug(f"LLMllamaLocal: prepare_output()")

    def create_embedding(self,input:str,model:str) -> any:
        logger.debug(f"LLMllamaLocal: create_embedding()")
        self._check_preload()        
        return self._pipeline.create_embedding(input=input,model=model)

    def _check_preload(self):
        if self._pipeline is None:
            logger.debug(f"LLMllamaLocal: _check_preload")
            self.load_model()
            self.load_tokenizer()
            self.create_pipeline()
            self.prepare_input()
        else:
            logger.debug(f"LLMllamaLocal: use downloaded model file")            

    def _create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt."""
        pass

    def _create_chat_completion(
        self,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[
        CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
    ]:
        """Generate a chat completion from a list of messages."""
        pass

    def simple_completion(self, prompt: str,
                          temperature: float = 0.2,
                          top_p: float = 0.95,
                          top_k: int = 40,
                          min_p: float = 0.05,
                          typical_p: float = 1.0,
                          stream: bool = False,
                          stop_criterias: list = ["</s>"],
                          max_tokens: Optional[int] = None,
                          frequency_penalty: float = 1.1,
                          seed: int=123,
                          output_format: str = "text") -> any:
        logger.debug(f"LLMllamaLocal: simple_completion()")
        reply = None
        status_code=-1
        t0 = time.perf_counter()
        if prompt is None or len(prompt) == 0:
            return reply
        try:
            self._check_preload()
            response = self._pipeline.create_completion(prompt=prompt,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    top_k=top_k,
                                                    min_p=min_p,
                                                    typical_p=typical_p,
                                                    stream=stream,
                                                    max_tokens=max_tokens,
                                                    repeat_penalty=frequency_penalty,
                                                    seed=seed,
                                                    stop=stop_criterias)
            reply = response["choices"][0]["text"]            
            status_code = response["choices"][0]["finish_reason"]
            # expect 'length' in return
            assert status_code == "stop" or status_code == "length", f"The status code was {status_code}."
            status_code=0
        except Exception as e:
            logger.error(f"Error simple_completion:",e)            
        return reply, status_code

    def simple_chat(self, prompt: str, history: list = [],
                    system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                    temperature: float = 0.9,
                    top_p: float = 0.95,
                    top_k: int = 40,
                    min_p: float = 0.05,
                    typical_p: float = 1.0,
                    stream: bool = False,
                    stop_criterias: list = ["\n", "</s>"],
                    max_tokens: Optional[int] = 256,
                    repeat_penalty: float = 1.1,
                    output_format: str = "text") -> any:
        logger.debug(f"LLMllamaLocal: simple_chat()")
        reply = None
        reply_messages = []
        t0 = time.perf_counter()
        if prompt is None or len(prompt) == 0:
            return [], [], None
        try:
            self._check_preload()
            if history is None or len(history)==0:
                history = [{"role": "system", "content": system_prompt}]
            # new message                
            new_message = [{"role": "user", "content": prompt}]
            history = history+new_message
            response = self._pipeline.create_chat_completion(messages=history,
                                                             temperature=temperature,
                                                             top_p=top_p,
                                                             top_k=top_k,
                                                             min_p=min_p,
                                                             typical_p=typical_p,
                                                             stream=stream,
                                                             max_tokens=max_tokens,
                                                             repeat_penalty=repeat_penalty,
                                                             stop=stop_criterias)
            reply = response["choices"][0]["message"]["content"]
            history.append({"role": "assistant", "content": reply})          
            for i in range(0, len(history)-1):
                if history[i]["role"] != "system":
                    reply_messages.append(
                        (history[i]["content"], history[i+1]["content"]))
            logger.debug(f"LLMllamaLocal chat reply: {reply}")                                
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            logger.error(f"Error chat_completion")
        took_in_seconds = time.perf_counter()-t0
        status_msg = f"chat completed took {took_in_seconds:.2f} seconds"
        logger.debug(f"LLMllamaLocal status: {status_msg}")
        return reply_messages, history, reply

class LLM_Setting(object):

    def __init__(self, url):
        if "https" in url or "http" in url:
            with urlopen(url) as content:
                self.config = json.load(content)
        else:
            self.config = json.load(open(url))

    def __call__(self, value):
        return self.config[value]

class LLMClient:

    def __init__(self, absctractLLM: AbstractLLMClass):
        self._llm = absctractLLM

    def simple_embedding(self,input:str,model:str) -> any:
        logger.debug(f"LLMClient: simple_embedding()")    
        return self._llm.create_embedding(input=input,model=model)

    def simple_chat(self, prompt: str, history: list = [],
                    system_prompt: str = system_prompt_assistant,
                    stop_criterias=["\n", "</s>"],
                    output_format: str = "text",
                    temperature: float = 0.9,
                    top_p: float = 0.95,
                    max_tokens: Optional[int] = 256,
                    repeat_penalty: float = 1.1) -> any:
        """
        The client code to execute chat algorithm. 
        """
        clean_history=history       
        messages, history, reply = self._llm.simple_chat(
            prompt=prompt, history=clean_history, 
            system_prompt=system_prompt,
            stop_criterias=stop_criterias, 
            output_format=output_format,
            temperature= temperature,
            top_p=top_p,
            max_tokens= max_tokens,
            repeat_penalty=repeat_penalty)
        return messages, history, reply

    def simple_completion(self, prompt: str,
                          temperature: float = 0.2,
                          top_p: float = 0.95,
                          max_tokens: Optional[int] = 512,
                          frequency_penalty: float = 1.1,
                          stop_criterias=["</s>"],
                          output_format: str = "text") -> any:
        """
        The client code to execute complete algorithm. 
        """
        reply, status_code = self._llm.simple_completion(
            prompt=prompt, temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_criterias=stop_criterias,
            frequency_penalty=frequency_penalty,
            output_format=output_format)
        return reply, status_code

def _free_gpu_resources():
    torch.cuda.empty_cache()
    gc.collect()    
    logger.info(f"_free_gpu_resources")    

def load_default_client(runtime_file:str="resources/config/llm_defaults.json")->LLMClient:
    global llm_defaults

    if "DEFAULT_LLM_MODEL_FILE" in config.system_config.keys(): 
        if "DEFAULT_LLM_MODEL_DOWNLOAD_PATH" in config.system_config.keys():
            if not os.path.exists(runtime_file):
                runtime_file=get_llm_local_defaults()
                default_model_path = hf_hub_download(repo_id=config.system_config["DEFAULT_LLM_MODEL_NAME_OR_PATH"], 
                                                    filename=config.system_config["DEFAULT_LLM_MODEL_FILE"], 
                                                    cache_dir=config.system_config["DEFAULT_LLM_MODEL_PATH"])
                # Write changes to env file.
                set_key("env.shared", "DEFAULT_LLM_MODEL_DOWNLOAD_PATH", default_model_path)
                llm_defaults["model_download_path"]=default_model_path           
                with open(runtime_file, 'w') as f:
                    json.dump(llm_defaults, f, indent=4)
                logger.info(f"created llm_defaults.json  LLM file: {default_model_path}")
            else:
                default_model_path = config.system_config["DEFAULT_LLM_MODEL_DOWNLOAD_PATH"]  
                logger.info(f"load llm_defaults.json LLM file: {default_model_path}")                

        # initialize default llm settings if missing configuration in dotenv file
        llm_settings = LLM_Setting(runtime_file)
        llm_client = LLMClient(LLMllamaLocal(**llm_settings.config))
        return llm_client
    else:
        raise Exception("Missing dotenv config value for default LLM. please ensure env_config has DEFAULT_LLM_MODEL_FILE specified.")        

#@functools.lru_cache
llm_client=None        
def get_llm_instance(runtime_file:str="resources/config/llm_defaults.json"):
    # create llm instance
    global llm_client  
    if llm_client is None:   
        llm_client=load_default_client(runtime_file)
    return llm_client

def new_llm_instance(target_model_info:list=None):
    global llm_defaults

    llm_copy = copy.copy(llm_defaults)
    # remove resources
    llm_copy["model_name_or_path"]=target_model_info[0]
    llm_copy["model_file_name"]=target_model_info[1]
    llm_copy["use_download_root"]=target_model_info[2]
    #llm_copy["model_architecure"]=target_model_info[3]  

    # load and get model
    AbstractLLMClass.get_llm_model(model_name_or_path=llm_copy["model_name_or_path"], 
                                   model_file=llm_copy["model_file_name"],
                                   model_path=llm_copy["use_download_root"])
    # create client
    llm_client = LLMClient(LLMllamaLocal(**llm_copy))        
    logger.info(f"new_llm_instance: {target_model_info}")
    return llm_client

def chat_completion(user_Prompt: str, history: list = [],
                        system_prompt: str = safe_system_prompt,
                        stop_criterias=["</s>"],
                        ask_expert: str = None,                      
                        target_model_info:list=None,
                        model_id:str=None,
                        user_settings:dict=None                        
                        ):
    global llm_client
    reply = None
    try:
        system_prompt=safe_system_prompt if system_prompt is None else system_prompt
        if target_model_info is not None and len(target_model_info)>0:
            """load new model"""
            if llm_client is not None:
                llm_client._llm._pipeline=None
                llm_client._llm=None
                #del llm_client
             ## release previous model gpu resources
            _free_gpu_resources()            
            llm_client = new_llm_instance(target_model_info)
        # make call
        llm_client = get_llm_instance()             
        history=[] if history is None else history  

        ## model parameters
        if user_settings is not None:
            activate_model_id = user_settings["_QUERY_MODEL_ID"] 
            top_p=user_settings["_QUERY_TOP_P"] 
            temperature = user_settings["_QUERY_TEMPERATURE"] 
            max_tokens=user_settings["_QUERY_MAX_TOKENS"]
            present_penalty=user_settings["_QUERY_PRESENT_PENALTY"] 
            stop_criterias=user_settings["_QUERY_STOP_WORDS"] 
            frequency_penalty= user_settings["_QUERY_FREQUENCY_PENALTY"] 
            gpu_offload_layers = user_settings["_QUERY_GPU_OFFLOADING_LAYERS"]        
            messages, history, reply = llm_client.simple_chat(prompt=user_Prompt, 
                                                            history=history,
                                                            system_prompt=system_prompt, 
                                                            stop_criterias=stop_criterias,
                                                            temperature=temperature,
                                                            top_p=top_p,
                                                            max_tokens=max_tokens)
        else:
            messages, history, reply = llm_client.simple_chat(prompt=user_Prompt, 
                                                            history=history,
                                                            system_prompt=system_prompt, 
                                                            stop_criterias=stop_criterias)            
        return messages, history, reply        
    except Exception as e:
        print(e)    
        print(traceback.format_exc())    
        logger.error(f"LLMChat.llm_chat_completion has error {str(e.args)}")
        _free_gpu_resources()                                                                    
    return [], history, reply        
