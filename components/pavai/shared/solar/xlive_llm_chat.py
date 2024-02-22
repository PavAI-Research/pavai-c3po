from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
from rich.logging import RichHandler
from rich import print,pretty,console
from rich.pretty import (Pretty,pprint)
from rich.panel import Panel
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()
import warnings 
warnings.filterwarnings("ignore")
import gc
import os
import time
import torch
import requests
import shutil
import functools
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
from openai import OpenAI
from openai._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from openai import Stream, AsyncStream
from openai.types import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    completion_create_params)
import numpy.typing as npt
import numpy as np
from openai import OpenAI
import numpy.typing as npt
import numpy as np
import httpx
import json
from .xlive_track_tokens import HistoryTokenBuffer

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.1"

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')
# logger = logging.getLogger(__name__)

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
# DEFAULT_LLM_MODEL_FILE = "zephyr-7b-beta.Q5_K_M.gguf"
DEFAULT_LLM_CHAT_FORMAT = "chatml"  #"llama-2"
DEFAULT_LLM_USE_GPU_LAYERS = 35
DEFAULT_LLM_CONTEXT_SIZE = 7892  # not 8192 discount special tokens

## system_prompts
## https://github.com/mustvlad/ChatGPT-System-Prompts

### Good Chatbot Flow
# [Input from User] --> [Analyze User Request] --> [Identify intend and Entities] --> [Compose reply]  

system_prompt_science_explainer="""
You are an expert in various scientific disciplines, including physics, chemistry, and biology. Explain scientific concepts, theories, and phenomena in an engaging and accessible way. Use real-world examples and analogies to help users better understand and appreciate the wonders of science.
"""

system_prompt_historical_expert="""
You are an expert in world history, knowledgeable about different eras, civilizations, and significant events. Provide detailed historical context and explanations when answering questions. Be as informative as possible, while keeping your responses engaging and accessible.
"""

system_prompt_history_storyteller="""
You are a captivating storyteller who brings history to life by narrating the events, people, and cultures of the past. Share engaging stories and lesser-known facts that illuminate historical events and provide valuable context for understanding the world today. Encourage users to explore and appreciate the richness of human history.
"""

system_prompt_language_learning_coach="""
You are a language learning coach who helps users learn and practice new languages. Offer grammar explanations, vocabulary building exercises, and pronunciation tips. Engage users in conversations to help them improve their listening and speaking skills and gain confidence in using the language.
"""

system_prompt_philosopher="""
You are a philosopher, engaging users in thoughtful discussions on a wide range of philosophical topics, from ethics and metaphysics to epistemology and aesthetics. Offer insights into the works of various philosophers, their theories, and ideas. Encourage users to think critically and reflect on the nature of existence, knowledge, and values.
"""

system_prompt_fitness_coach="""
You are a knowledgeable fitness coach, providing advice on workout routines, nutrition, and healthy habits. Offer personalized guidance based on the user's fitness level, goals, and preferences, and motivate them to stay consistent and make progress toward their objectives.
"""

system_prompt_news_summarizer="""
You are a news summarizer, providing concise and objective summaries of current events and important news stories from around the world. Offer context and background information to help users understand the significance of the news, and keep them informed about the latest developments in a clear and balanced manner.
"""

system_prompt_nutritionist="""
You are a nutritionist AI, dedicated to helping users achieve their fitness goals by providing personalized meal plans, recipes, and daily updates. Begin by asking questions to understand the user's current status, needs, and preferences. Offer guidance on nutrition, exercise, and lifestyle habits to support users in reaching their objectives. Adjust your recommendations based on user feedback, and ensure that your advice is tailored to their individual needs, preferences, and constraints.
"""

system_prompt_cyber_security_specialist="""
You are a cyber security specialist, providing guidance on securing digital systems, networks, and data. Offer advice on best practices for protecting against threats, vulnerabilities, and breaches. Share recommendations for security tools, techniques, and policies, and help users stay informed about the latest trends and developments in the field.
"""

system_prompt_time_management_coach="""
You are a time management coach, helping users to manage their time more effectively and achieve their goals. Offer practical tips, strategies, and encouragement to help them stay focused, organized, and motivated.
"""

system_prompt_recipe_recommender="""
You are a recipe recommender, providing users with delicious and easy-to-follow recipes based on their dietary preferences, available ingredients, and cooking skill level. Offer step-by-step instructions and helpful tips for preparing each dish, and suggest creative variations to help users expand their culinary repertoire.
"""

system_prompt_travel_planner="""
You are a virtual travel planner, assisting users with their travel plans by providing information on destinations, accommodations, attractions, and transportation options. Offer tailored recommendations based on the user's preferences, budget, and travel goals, and share practical tips to help them have a memorable and enjoyable trip.
"""

system_prompt_progamming_assistant="""
You are an AI programming assistant. Follow the user's requirements carefully and to the letter. First, think step-by-step and describe your plan for what to build in pseudocode, written out in great detail. Then, output the code in a single code block. Minimize any other prose.
"""

system_prompt_personal_finance_advisor="""
You are a personal finance advisor, providing guidance on budgeting, saving, investing, and managing debt. Offer practical tips and strategies to help users achieve their financial goals, while considering their individual circumstances and risk tolerance. Encourage responsible money management and long-term financial planning.
"""

system_prompt_inspirational_quotes="""
You are an AI that generates original, thought-provoking, and inspiring quotes. Your quotes should be motivational, uplifting, and relevant to the user's input, encouraging them to reflect on their thoughts and actions.
"""

system_prompt_mediation_guide="""
You are a meditation guide, helping users to practice mindfulness and reduce stress. Provide step-by-step instructions for various meditation techniques, along with tips for cultivating a peaceful, focused mindset. Encourage users to explore the benefits of regular meditation practice for their mental and emotional well-being.
"""

system_prompt_social_media_influencer="""
You are a social media influencer, sharing your thoughts, experiences, and tips on various topics such as fashion, travel, technology, or personal growth. Provide insightful and engaging content that resonates with your followers, and offer practical advice or inspiration to help them improve their lives.
"""

system_prompt_diy_project_idea_generator="""
You are a DIY project idea generator, inspiring users with creative and practical ideas for home improvement, crafts, or hobbies. Provide step-by-step instructions, materials lists, and helpful tips for completing projects of varying difficulty levels. Encourage users to explore their creativity and develop new skills through hands-on activities.
"""

system_prompt_trivia_master="""
You are a trivia master, challenging users with fun and interesting questions across a variety of categories, including history, science, pop culture, and more. Provide multiple-choice questions or open-ended prompts, and offer explanations and interesting facts to supplement the answers. Encourage friendly competition and help users expand their general knowledge.
"""

system_prompt_poet="""
You are a poet, crafting original poems based on users' input, feelings, or themes. Experiment with various poetic forms and styles, from sonnets and haikus to free verse and spoken word. Share your passion for language, imagery, and emotions, and inspire users to appreciate the beauty and power of poetry.
"""

system_prompt_party_planner="""
You are a party planner, providing creative ideas and practical tips for organizing memorable events, from small gatherings to large celebrations. Offer suggestions for themes, decorations, food, and entertainment, and help users tailor their party plans to their budget, space, and guest list. Encourage users to create unique and enjoyable experiences for their guests.
"""

system_prompt_career_counselor="""
You are a career counselor, offering advice and guidance to users seeking to make informed decisions about their professional lives. Help users explore their interests, skills, and goals, and suggest potential career paths that align with their values and aspirations. Offer practical tips for job searching, networking, and professional development.
"""

system_prompt_math_tutor="""
You are a math tutor who helps students of all levels understand and solve mathematical problems. Provide step-by-step explanations and guidance for a range of topics, from basic arithmetic to advanced calculus. Use clear language and visual aids to make complex concepts easier to grasp.
"""

system_prompt_python_tutor="""
You are a math tutor who helps students of all levels understand and solve mathematical problems. Provide step-by-step explanations and guidance for a range of topics, from basic arithmetic to advanced calculus. Use clear language and visual aids to make complex concepts easier to grasp.
"""

system_prompt_machine_learning_tutor="""
You are a Machine Learning Tutor AI, dedicated to guiding senior software engineers in their journey to become proficient machine learning engineers. Provide comprehensive information on machine learning concepts, techniques, and best practices. Offer step-by-step guidance on implementing machine learning algorithms, selecting appropriate tools and frameworks, and building end-to-end machine learning projects. Tailor your instructions and resources to the individual needs and goals of the user, ensuring a smooth transition into the field of machine learning.
"""

system_prompt_data_scientist="""
I want you to act as a data scientist to analyze datasets. Do not make up information that is not in the dataset. For each analysis I ask for, provide me with the exact and definitive answer and do not provide me with code or instructions to do the analysis on other platforms.
"""

system_prompt_creative_writing_coach="""
You are a creative writing coach, guiding users to improve their storytelling skills and express their ideas effectively. Offer constructive feedback on their writing, suggest techniques for developing compelling characters and plotlines, and share tips for overcoming writer's block and staying motivated throughout the creative process.
"""

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

system_prompt_basic = """
You are an intelligent AI assistant. You are helping user answer query.
If the text does not contain sufficient information to answer the question, do not make up information and give the answer as "I don't know, please be specific.".
"""

system_roles={
    "basic": system_prompt_basic,
    "assistant": system_prompt_assistant,    
    "science_explainer":system_prompt_science_explainer,
    "data_scientist":system_prompt_data_scientist,    
    "creative_writing_coach":system_prompt_creative_writing_coach,
    "python_tutor":system_prompt_python_tutor,
    "math_tutor":system_prompt_math_tutor,    
    "machine_learning_tutor":system_prompt_machine_learning_tutor,        
    "career_counselor":system_prompt_career_counselor,          
    "party_planner":system_prompt_party_planner,               
    "poet":system_prompt_poet,               
    "trivia_master":system_prompt_trivia_master,                        
    "idea_generator":system_prompt_diy_project_idea_generator,               
    "social_media_influencer":system_prompt_social_media_influencer,                            
    "mediation_guide":system_prompt_mediation_guide,               
    "inspirational_quotes":system_prompt_inspirational_quotes,                                
    "personal_finance_advisor":system_prompt_personal_finance_advisor,                                    
    "progamming_assistant":system_prompt_progamming_assistant,                                
    "travel_planner":system_prompt_travel_planner,                                        
    "recipe_recommender":system_prompt_recipe_recommender,                                            
    "time_management_coach":system_prompt_time_management_coach,                                        
    "cyber_security_specialist":system_prompt_cyber_security_specialist,                                                
    "nutritionist":system_prompt_nutritionist,                                            
    "news_summarizer":system_prompt_news_summarizer,                                        
    "fitness_coach":system_prompt_fitness_coach,                                                   
    "philosopher":system_prompt_philosopher,                                        
    "language_learning_coach":system_prompt_language_learning_coach,                                                       
    "history_storyteller":system_prompt_history_storyteller,                                        
    "historical_expert":system_prompt_historical_expert,                                                       
    "science_explainer":system_prompt_science_explainer                                                            
}


gguf_map = {
    "zephyr-7b-beta.Q3_K_M.gguf": ["zephyr-7b-beta.Q3_K_M.gguf", "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q3_K_M.gguf", "chatml"],
    "zephyr-7b-beta.Q4_K_M.gguf": ["zephyr-7b-beta.Q4_K_M.gguf", "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf", "chatml"],
    "zephyr-7b-beta.Q5_K_M.gguf": ["zephyr-7b-beta.Q5_K_M.gguf", "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf", "chatml"],
    "yarn-mistral-7b-128k.Q3_K_M.gguf": ["yarn-mistral-7b-128k.Q3_K_M.gguf", "https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-GGUF/resolve/main/yarn-mistral-7b-128k.Q3_K_M.gguf", "llama-2"],
    "yarn-mistral-7b-128k.Q4_K_M.gguf": ["yarn-mistral-7b-128k.Q4_K_M.gguf", "https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-GGUF/resolve/main/yarn-mistral-7b-128k.Q4_K_M.gguf", "llama-2"],
    "yarn-mistral-7b-128k.Q5_K_M.gguf": ["yarn-mistral-7b-128k.Q5_K_M.gguf", "https://huggingface.co/TheBloke/Yarn-Mistral-7B-128k-GGUF/resolve/main/yarn-mistral-7b-128k.Q5_K_M.gguf", "llama-2"],
}

# "use_download_root": "resources/models/llm/local",
llm_local = {
    "model_runtime": "llamacpp",
    "model_id": None,
    "model_file_name": "zephyr-7b-beta.Q5_K_M.gguf",
    "model_file_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf",
    "model_source": "filesystem",
    "model_architecure": "llama-2",
    "api_base": None,
    "api_key": None,
    "api_domain": None,
    "api_organization": None,
    "use_device": "cuda",
    "use_torch_type": "float16",
    "model_tokenizer_id_or_path": None,
    "use_flash_attention_2": "",
    "use_download_root": "/home/pop/development/mclab/talking-llama/models/llm/",
    "use_local_model_only": True,
    "use_max_context_size": 4096*2,
    "use_n_gpu_layers": 38,
    "verbose": True,
    "use_seed": 123
}

llm_api = {
    "model_runtime": "llamacpp",
    "model_id": None,
    "model_file_name": "zephyr-7b-beta.Q4_K_M.gguf",
    "model_file_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
    "model_source": "filesystem",
    "model_architecure": "llama-2",
    "api_base": None,
    "api_key": None,
    "api_domain": None,
    "api_organization": None,
    "use_device": "cuda",
    "use_torch_type": "float16",
    "model_tokenizer_id_or_path": None,
    "use_flash_attention_2": "",
    "use_download_root": "/home/pop/development/mclab/talking-llama/models/llm/",
    "use_local_model_only": True,
    "use_max_context_size": 4096*2,
    "use_n_gpu_layers": 38,
    "verbose": True,
    "use_seed": 123
}

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
                 model_file_url: str = None,
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
        self._model_file_name = model_file_name
        self._model_file_url = model_file_url
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
    def get_llm_model(model_path: str=DEFAULT_LLM_MODEL_PATH,
                      model_file: str = "",
                      model_chat_format: str = ""):

        gguf_config = gguf_map[model_file]
        if gguf_config is None:
            gguf_model_file = model_path+DEFAULT_LLM_MODEL_FILE
            gguf_chat_format = DEFAULT_LLM_CHAT_FORMAT
        else:
            gguf_model = gguf_map[model_file]
            gguf_model_file = model_path+gguf_model[0]
            gguf_chat_format = gguf_model[2]

        if not os.path.isfile(gguf_model_file):
            gguf_model_url = gguf_model[1]
            logger.error(f"[Missing] local LLM gguf file: {gguf_model_file} / {gguf_chat_format}")            
            logger.warn("Attempt one-time downloading model file from web may take few minutes, depends on your internet connection.")
            logger.info(f"downloading LLM gguf file:{gguf_model_url}. please wait!!!")
            gguf_model_file = AbstractLLMClass.download_file(
                gguf_model_url, model_path)
            if os.path.isfile(gguf_model_file):
                logger.info(f"web download - Success!!!")                
            else:
                logger.info(f"web download - Failed!!!")                                                
        else:
            logger.debug(f"use exist local LLM gguf file: {gguf_model_file} | {gguf_chat_format}")

        return gguf_model_file, gguf_chat_format

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
            gguf_model_file, gguf_chat_format = AbstractLLMClass.get_llm_model(model_path=self.use_download_root,
                                                                   model_file=self.model_file_name)
            self.model_file_name = gguf_model_file
            self.use_chat_format = gguf_chat_format
        elif (self.model_file_url is not None and len(self.model_file_url.strip()) > 0) and (self.model_file_name.lower().endswith(".gguf")):
            # GGUF model remote file
            gguf_model_file, gguf_chat_format = AbstractLLMClass.download_file(self.model_file_url,
                                                                               model_path=self.use_download_root)
            self.model_file_name = gguf_model_file
            self.use_chat_format = gguf_chat_format
        elif (self.model_id is not None and len(self.model_id.strip()) > 0):
            # download an entire repository
            snapshot_download(repo_id=self.model_id)
        else:
            raise Exception("Unable to load model due missing information!")

    def load_tokenizer(self) -> None:
        logger.debug(f"LLMllamaLocal: load_tokenizer()")

    def create_pipeline(self) -> None:
        logger.debug(f"LLMllamaLocal: load_tokenizer {self.model_id}")
        if self._pipeline is None:
            self.use_cpu_threads = int(os.cpu_count()/2)  # use half only
            logger.debug(f"create_pipeline device: {self.use_device}")
            self._pipeline = Llama(model_path=self.model_file_name,
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
                    temperature: float = 0.2,
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            history = history+messages
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
            logger.error(f"Error chat_completion:",e)
        took_in_seconds = time.perf_counter()-t0
        status_msg = f"chat completed took {took_in_seconds:.2f} seconds"
        logger.debug(f"LLMllamaLocal status: {status_msg}")
        return reply_messages, history, reply

class LLMllamaOpenAI(AbstractLLMClass):
    """
    LLMllamaAPI Class override only required class operations.
    """

    def __init__(self, model_runtime: str = "llamacpp", model_id: str = None,
                 model_file_name: str = None, model_file_url: str = None,
                 model_source: str = "filesystem", model_architecure: str = "llama-2",
                 api_base: str = None, api_key: str = "EMPTY", api_domain: str = None,
                 api_organization: str = None, use_device: str = "cpu",
                 use_torch_type: str = "float16", model_tokenizer_id_or_path: str = None,
                 use_flash_attention_2: bool = False,
                 use_download_root: str = "resources/models/llm",
                 use_local_model_only: bool = False, use_max_context_size: int = 4096,
                 use_n_gpu_layers: int = 0, verbose: bool = True, use_seed: int = 123):
        super().__init__(model_runtime, model_id, model_file_name, model_file_url, model_source, model_architecure,
                         api_base, api_key, api_domain, api_organization, use_device, use_torch_type, model_tokenizer_id_or_path,
                         use_flash_attention_2, use_download_root, use_local_model_only, use_max_context_size,
                         use_n_gpu_layers, verbose, use_seed)
        self._client = None

    def load_model(self) -> None:
        logger.debug(f"LLMllamaAPI: load_model: None")

    def load_tokenizer(self) -> None:
        logger.debug(f"LLMllamaAPI: load_tokenizer: None")

    def create_client(self) -> None:
        logger.debug(f"LLMllamaAPI: create_client {self._api_base}")
        self._client = OpenAI(api_key=self._api_key, base_url=self._api_base)

    def create_pipeline(self) -> None:
        logger.debug(f"LLMllamaAPI: load_tokenizer {self.model_id}")

    def prepare_input(self) -> None:
        logger.debug(f"LLMllamaAPI: prepare_input: None")

    def prepare_output(self) -> None:
        logger.debug(f"LLMllamaAPI: prepare_output: None")

    def create_embedding(self,input:str,model:str="text-embedding-ada-002") -> any:
        logger.debug(f"LLMllamaAPI: create_embedding()")      
        self._check_preload()  
        response = self._client.embeddings.create(input=input,model=model)
        return response

    def _check_preload(self):
        if self._client is None:
            self.create_client()
        else:
            logger.debug(f"LLMllamaLocal _check_preload: use preload client")            
        if self._pipeline is None:
            self.load_model()
            self.load_tokenizer()
            self.create_pipeline()
            self.prepare_input()
        else:
            logger.debug(f"LLMllamaLocal _check_preload: use preload model")                        

    def _create_completion(self,
                           model: Union[
            str,
            Literal[
                "babbage-002",
                "davinci-002",
                "gpt-3.5-turbo-instruct",
                "text-davinci-003",
                "text-davinci-002",
                "text-davinci-001",
                "code-davinci-002",
                "text-curie-001",
                "text-babbage-001",
                "text-ada-001",
            ],
                               ],
        prompt: Union[str, List[str], List[int], List[List[int]], None],
        best_of: Optional[int] | NotGiven = NOT_GIVEN,
        echo: Optional[bool] | NotGiven = NOT_GIVEN,
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        logprobs: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str], None] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]
                         ] | Literal[True] | NotGiven = NOT_GIVEN,
        suffix: Optional[str] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> Completion | Stream[Completion]:
        """completion"""
        return self._client.completions.create(prompt=prompt, model=model, best_of=best_of, echo=echo, frequency_penalty=frequency_penalty,
                                               logit_bias=logit_bias, logprobs=logprobs, max_tokens=max_tokens, n=n,
                                               presence_penalty=presence_penalty, seed=seed, stop=stop, stream=stream,
                                               suffix=suffix, temperature=temperature, top_p=top_p, user=user,
                                               extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body,
                                               timeout=timeout)

    def _api_completion(self, prompt: str,
                        model: str = "gpt-3.5-turbo-instruct",
                        max_tokens: int = 256,
                        frequency_penalty: int = 1.1,
                        temperature: int = 0.2,
                        top_p: float = 0.95,
                        seed: int = 123,
                        stop: list = None,
                        best_of: int = 3,
                        user="user",
                        timeout: int = 120,
                        stream: bool = False,
                        verbose: bool = False):
        t0 = time.perf_counter()
        assert isinstance(prompt, str), "`prompt` should be a string"
        response = self._create_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            seed=seed,
            stream=stream,
            best_of=best_of,
            stop=stop,
            timeout=timeout,
            user=user)
        status_code = response.choices[0].finish_reason
        # expect "stop" but got 'length' in return
        assert status_code == "stop" or status_code == "length", f"The status code was {status_code}."
        reply = response.choices[0].text
        tx = time.perf_counter()-t0
        logger.info(f"LLMllamaLocal api_completion: {status_code} took {tx:.2f}")                                
        return reply, status_code

    def _create_chat_completion(self,
                                messages: List[ChatCompletionMessageParam],
                                model: Union[
            str,
            Literal[
                "gpt-4-1106-preview",
                "gpt-4-vision-preview",
                "gpt-4",
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-4-32k-0613",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-16k-0613",
            ],
                                    ],
        frequency_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        function_call: completion_create_params.FunctionCall | NotGiven = NOT_GIVEN,
        functions: List[completion_create_params.Function] | NotGiven = NOT_GIVEN,
        logit_bias: Optional[Dict[str, int]] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        n: Optional[int] | NotGiven = NOT_GIVEN,
        presence_penalty: Optional[float] | NotGiven = NOT_GIVEN,
        response_format: completion_create_params.ResponseFormat | NotGiven = NOT_GIVEN,
        seed: Optional[int] | NotGiven = NOT_GIVEN,
        stop: Union[Optional[str], List[str]] | NotGiven = NOT_GIVEN,
        stream: Optional[Literal[False]
                         ] | Literal[True] | NotGiven = NOT_GIVEN,
        temperature: Optional[float] | NotGiven = NOT_GIVEN,
        tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven = NOT_GIVEN,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        top_p: Optional[float] | NotGiven = NOT_GIVEN,
        user: str | NotGiven = NOT_GIVEN,
        # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
        # The extra values given here take precedence over values defined on the client or passed to this method.
        extra_headers: Headers | None = None,
        extra_query: Query | None = None,
        extra_body: Body | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        return self._client.chat.completions.create(messages=messages, model=model, frequency_penalty=frequency_penalty,
                                                    function_call=function_call, functions=functions, logit_bias=logit_bias,
                                                    max_tokens=max_tokens, n=n, presence_penalty=presence_penalty, response_format=response_format,
                                                    seed=seed, stop=stop, stream=stream, temperature=temperature, tool_choice=tool_choice,
                                                    tools=tools, top_p=top_p, user=user, extra_headers=extra_headers,
                                                    extra_query=extra_query, extra_body=extra_body, timeout=timeout)

    def _api_chat_completion(self, user_prompt: str, history: list = [],
                             system_prompt: str = None,
                             model: str = "gpt-3.5-turbo",
                             max_tokens: int = 256,
                             frequency_penalty: int = 1.1,
                             temperature: int = 0.2,
                             top_p: int = 0.95,
                             seed: int = 123,
                             stop: list = None,
                             user="user",
                             timeout: int = 120, verbose: bool = False):
        t0 = time.perf_counter()
        assert isinstance(user_prompt, str), "`user_prompt` should be a string"
        # assert isinstance(system_prompt, str), "`system_prompt` should be a string"
        assert isinstance(history, list), "`history` should be a list"
        user_msg = {"role": "user", "content": user_prompt}
        if history is None or len(history) == 0:
            history = []
            system_msg = {"role": "system", "content": system_prompt}
            history.append(system_msg)
        history.append(user_msg)
        response = self._create_chat_completion(
            messages=history,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            seed=seed,
            stop=stop,
            timeout=timeout,
            user=user)
        status_code = response.choices[0].finish_reason
        # expect "stop" but got 'length' in return
        assert status_code == "stop" or status_code == "length", f"The status code was {status_code}."
        reply = response.choices[0].message.content
        response_msg = {"role": "assistant", "content": reply}
        history.append(response_msg)
        tx = time.perf_counter()-t0
        logger.debug(f"LLMllamaLocal _api_chat_completion: {status_code} took {tx:.2f}")                                        
        return reply, history, status_code

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
                          output_format: str = "text") -> any:
        logger.debug(f"LLMllamaLocal: simple_complete()")
        t0 = time.perf_counter()
        reply = ""
        status_code = -1
        try:
            self._check_preload()
            reply, status_code = self._api_completion(prompt=prompt, temperature=temperature,
                                                      top_p=top_p,
                                                      stream=stream, stop=stop_criterias,
                                                      max_tokens=max_tokens,
                                                      frequency_penalty=frequency_penalty
                                                      )
        except Exception as e:
            logger.error(f"Error simple_complete:",e)
        tx = time.perf_counter()-t0
        logger.debug(f"LLMllamaLocal simple_complete: {status_code} took {tx:.2f}")                                                
        return reply, status_code

    def simple_chat(self, prompt: str, history: list = [],
                    system_prompt: str = DEFAULT_LLM_SYSTEM_PROMPT,
                    temperature: float = 0.2,
                    top_p: int = 0.95,
                    top_k: int = 40,
                    min_p: float = 0.05,
                    typical_p: float = 1.0,
                    stream: bool = False,
                    stop_criterias: list = ["\n", "</s>"],
                    max_tokens: Optional[int] = 256,
                    repeat_penalty: float = 1.1,
                    output_format: str = "text") -> any:
        logger.debug(f"LLMllamaAPI: simple_chat()")
        reply_messages = []
        reply = None
        try:
            self._check_preload()
            reply, history, status_code = self._api_chat_completion(
                user_prompt=prompt, history=history,
                system_prompt=system_prompt,
                temperature=temperature, top_p=top_p,
                stop=stop_criterias, max_tokens=max_tokens
            )
            for i in range(0, len(history)-1):
                if history[i]["role"] != "system":
                    reply_messages.append(
                        (history[i]["content"], history[i+1]["content"]))
        except Exception as e:
            logger.error(f"LLMllamaLocal chat_completion error: {e}")                                                            
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

class LLMClient():

    def __init__(self, absctractLLM: AbstractLLMClass):
        self._llm = absctractLLM

    def simple_embedding(self,input:str,model:str) -> any:
        logger.debug(f"LLMClient: simple_embedding()")    
        # "text-embedding-ada-002"    
        return self._llm.create_embedding(input=input,model=model)

    def simple_chat(self, prompt: str, history: list = [],
                    system_prompt: str = system_prompt_assistant,
                    stop_criterias=["\n", "</s>"],
                    output_format: str = "text",
                    temperature: float = 0.2,
                    top_p: float = 0.95,
                    max_tokens: Optional[int] = 256,
                    repeat_penalty: float = 1.1) -> any:
        """
        The client code to execute chat algorithm. 
        """
        ## remove invalid or duplicate system messages if exist
        clean_history=history
        if len(history)>0:
            clean_history=[{"role": "system", "content": system_prompt}]            
            for i in range(0, len(history)):
                try:
                    if history[i]["role"] != "system":
                        clean_history.append(history[i])
                except:
                    pass
        ## invoke chat                 
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

@functools.lru_cache
def get_llm(runtime_file:str="llm_local.json"):
    llm_settings = LLM_Setting(runtime_file)
    llm_client = LLMClient(LLMllamaLocal(**llm_settings.config))
    #llm_client = LLMClient(LLMllamaLocal(**llm_local))        
    #print(llm_settings.config)
    # attempt freeup unused resources
    _free_gpu_resources()
    return llm_client

# create llm instance
llm_client = None

def llm_chat_completion(user_Prompt: str, history: list = [],
                        system_prompt: str = system_prompt_assistant,
                        stop_criterias=["</s>"]):
    global llm_client    
    reply = None
    try:
        llm_client = get_llm() if llm_client is None else llm_client
        history=[] if history is None else history  
        messages, history, reply = llm_client.simple_chat(
            prompt=user_Prompt, history=history,system_prompt=system_prompt, 
            stop_criterias=stop_criterias)
        return messages, history, reply        
    except Exception as e:
        logger.error(f"LLMllamaLocal llm_chat_completion error: {e}")                                                                    
    return [], history, reply        

def llm_chat(input_text, chat_history):
    # messages, history, reply = voice_llm_api.text_chat_completion(input_text,chat_history)
    messages, history, reply = llm_chat_completion(input_text, chat_history)
    return messages, history, reply

client = OpenAI(api_key="EMPTY", base_url="http://localhost:7002/v1")

def llm_chat_openai(messages):
    response = client.chat.completions.create(
    model="gpt-3.5-turbo", messages=messages, stream=False,max_tokens=256)
    return response.choices[0].message.content

def process_chat_opeai(new_query:str,history:list, context_size:int=4096*2):
    character_prompt = 'You are a intelligent AI bot. your are helpful to answer user query.'
    history_buffer = HistoryTokenBuffer(history=history,max_tokens=context_size)
    history_buffer.update(new_query)
    logger.info(f"[gray]context token counts available: {history_buffer.token_count} / max {history_buffer.max_tokens}[/gray]")
    if len(history_buffer.overflow_summary)>0:
        # reduce and compress history content 
        logger.info(f"overflow summary: {history_buffer.overflow_summary}")
        summary="provide an summary of following text:\n "+history_buffer.overflow_summary
        user_messages=[{'role': 'system',  'content': character_prompt},
                       {'role': 'user',  'content': summary}]
        assistant_response = llm_chat_openai(user_messages)    
        ##text_speaker(sd,text=assistant_response)       
        history_buffer.update(summary)    
        history=history_buffer.history
        history_buffer.overflow_summary=""
        history.append({'role': 'assistant', 'content': assistant_response})                   

    # perform new query now
    history.append({'role': 'user', 'content': new_query})
    assistant_response = llm_chat_openai([{'role': 'system',  'content': character_prompt}] + history[-10:])
    #console.log(f"\nAI: {assistant_response}\n")
    history.append({'role': 'assistant', 'content': assistant_response})    
    return assistant_response, history 

def test_local_embedding():
    logger.info("***TEST EMBEDDING***")
    llm_client = LLMClient(LLMllamaLocal(**llm_local))
    reply = llm_client.simple_embedding(input="hello", model="text-embedding-ada-002")
    logger.info(reply)
    logger.info("\n")

def test_local_completion():
    logger.info("***TEST COMPLETION***")
    llm_client = LLMClient(LLMllamaLocal(**llm_local))
    reply, status_code = llm_client.simple_completion(
        prompt="planets in the solar system are ")
    logger.info(reply)
    logger.info("\n")
    reply, status_code = llm_client.simple_completion(
        prompt="write a quick sort in python")
    logger.info(reply)
    logger.info("\n")        

def test_local_multi_turn_conversation():
    logger.info("***TEST CONVERSATION MODE***")    
    llm_client = LLMClient(LLMllamaLocal(**llm_local))

    ## test-1 conversation start
    messages, history, reply = llm_client.simple_chat(prompt="hey", history=[],
                                                      stop_criterias=["</s>"],
                                                      system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    ## ------------------------------
    ## sample conversation  
    ## ------------------------------
    logger.info("---"*20)
   
    ## test ask question-1
    messages, history, reply = llm_client.simple_chat(prompt="I need some help on writing a formal resume. any suggestion?", history=[],
                                                    stop_criterias=["</s>"],
                                                    system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    
    ## test ask question-2
    messages, history, reply = llm_client.simple_chat(prompt="can you give me good sample of product manager profile summary?", history=history,
                                                    stop_criterias=["</s>"],
                                                    system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")        
    
    ## test ask question-3
    messages, history, reply = llm_client.simple_chat(prompt="Is adding references necessary?, if Yes, then how many references works the best", 
                                                      history=history,
                                                    stop_criterias=["</s>"],
                                                    system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")       

    ## test ask question-4
    messages, history, reply = llm_client.simple_chat(prompt="how many pages a good resume should be?", 
                                                      history=history,
                                                    stop_criterias=["</s>"],
                                                    system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")       

    ## test ask question-5
    messages, history, reply = llm_client.simple_chat(prompt="thanks for your help!", 
                                                      history=history,
                                                    stop_criterias=["</s>"],
                                                    system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n\n")          

def test_local_ask_standalone_questions():
    logger.info("***TEST ASK QUESTIONS including domain expert mode***")    
    #llm_settings = LLM_Setting("llm_local.json")
    #llm_client = LLMClient(LLMllamaLocal(**llm_settings.config))
    #logger.info(llm_settings.config)        
    llm_client = LLMClient(LLMllamaLocal(**llm_local)) 
    ## ------------------------------
    ## chatbot individual questions 
    ## ------------------------------
    logger.info("---"*20)
    
    ## general question-1      
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me about Toronto CN-Tower", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    
    ## domain expert: question-1      
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me about Toronto CN-Tower", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_trivia_master)
    logger.info(messages[-1])
    logger.info("\n")    
    
    logger.info("---"*20)    
    
    ## general question-2      
    messages, history, reply = llm_client.simple_chat(
        prompt="can you remind me, who was the 38th president of united states?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    
    ## domain expert: question-2     
    messages, history, reply = llm_client.simple_chat(
        prompt="can you remind me, who was the 38th president of united states?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_historical_expert)
    logger.info(messages[-1])
    logger.info("\n")    

    logger.info("---"*20)    
        
    ## general question-3      
    messages, history, reply = llm_client.simple_chat(
        prompt="why pi has infinite value?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    

    ## domain expert: question-3      
    messages, history, reply = llm_client.simple_chat(
        prompt="why pi has infinite value?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_math_tutor)
    logger.info(messages[-1])
    logger.info("\n")    

    logger.info("---"*20)    
    
    ## general question-4     
    messages, history, reply = llm_client.simple_chat(
        prompt="what is 4096 times 4?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_math_tutor)
    logger.info(messages[-1])
    logger.info("\n")    

    ## domain expert: question-4    
    messages, history, reply = llm_client.simple_chat(
        prompt="what is 4096 times 4?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_math_tutor)
    logger.info(messages[-1])
    logger.info("\n")    
    
    logger.info("---"*20)    
        
    ## general question-5     
    messages, history, reply = llm_client.simple_chat(
        prompt="who live longer human or dinosaur?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    

    ## domain expert: question-5     
    messages, history, reply = llm_client.simple_chat(
        prompt="who live longer human or dinosaur?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_trivia_master)
    logger.info(messages[-1])
    logger.info("\n")    

    logger.info("---"*20)    
        
    ## general question-6     
    messages, history, reply = llm_client.simple_chat(
        prompt="how to stay healhy?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    

    ## domain expert: question-6      
    messages, history, reply = llm_client.simple_chat(
        prompt="how to stay healhy?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_nutritionist)
    logger.info(messages[-1])
    logger.info("\n")    

    logger.info("---"*20)    

    ## general question-7     
    messages, history, reply = llm_client.simple_chat(
        prompt="any suggesttion on a fun birthday party on weekend?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    

    ## domain expert: question-7      
    messages, history, reply = llm_client.simple_chat(
        prompt="any suggesttion on a fun birthday party on weekend?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_party_planner)
    logger.info(messages[-1])
    logger.info("\n")    

    logger.info("---"*20)    

    ## general question-8     
    messages, history, reply = llm_client.simple_chat(
        prompt="can you provide step-by-step instruction on how to remove virus from my computer?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")        

    ## domain expert: question-8      
    messages, history, reply = llm_client.simple_chat(
        prompt="can you provide step-by-step instruction on how to remove virus from my computer?", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_cyber_security_specialist)
    logger.info(messages[-1])
    logger.info("\n")    
    
    logger.info("---"*20)
    
    ## general question-9      
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me planets in the solar system", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")    
    ## domain expert: question-9      
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me planets in the solar system", history=[],
        stop_criterias=["</s>"],system_prompt=system_prompt_science_explainer)
    logger.info(messages[-1])
    logger.info("\n")    
    
def test_api_calls():
    llm_settings = LLM_Setting("llm_api.json")
    llm_client = LLMClient(LLMllamaOpenAI(**llm_settings.config))
    logger.info(llm_settings.config)        
    messages, history, reply = llm_client.simple_chat(
        prompt="hello", history=[])
    logger.info(reply)
    logger.info("\n")
    reply, status_code = llm_client.simple_completion(
        prompt="write a quick sort in python")
    logger.info(reply)
    logger.info("api done!")
    
if __name__ == "__main__":
    from psutil import virtual_memory
    ram_gb = virtual_memory().total / 1e9
    logger.info("available RAM: ", ram_gb)
    # test_local_embedding()
    # test_local_completion()    
    test_local_multi_turn_conversation()
    # test_local_ask_standalone_questions()
    logger.info("local done!\n\n")

# system_prompt_chatbot1 = """
# You are an artificial intelligence assistant trained to have human-like voice conversations and research assistant, and a leading expert in time management coach, a math tutor, software designer and research assistant, and a leading expert in time management, cooking, health, cooking, software programming, food, biology, engineering,party planner,industrial design, environmental science, physiology, trivia master, personal financial adviser, cyber security specialist, travel planner, mediation guide, nutritionist, captive storyteller,knowledgeable fitness coach, philosopher, generate inspiring quotes and creative writing coach.

# Your goal is to help the user work in a step by step way through human-like voice conversations to answer user specific queries and challenges. 
# Stop often (at a minimum after every step) to ask the user for feedback or clarification.

# 1. Define - The first step in any conversation is to define the user request, identify query or opportunity that you want user to clarify or address. Prompt the user to think through the next steps to define their challenge. Don't try to answer these for the user. You may offer suggestions if asked to.
# 2. Analyze - Analyze the essential user intends, identify intend and entitiers and challenge must address
# 3. Discover - Look for best models that need to address the same functions as your solution
# 4. Abstract - Study the essential features or mechanisms to generate response that meet user expectation  
# 5. Emulate human-like natutal conversation Lessons - creating nature-inspired human response 

# A human voice conversation is a dynamic and interactive communication between two or more people, characterized by the following elements:

# 1. Speech: Human voice conversations involve the use of spoken language to convey meaning and intent. The tone, pitch, volume, and pace of speech can convey various emotions, attitudes, and intentions.
# 2. Listening: Human voice conversations require active listening, where the listener pays attention to the speaker's words, tone, and body language to understand their meaning and intent.
# 3. Turn-taking: Human voice conversations involve a turn-taking structure, where each speaker takes turns to speak and listen. Interruptions, overlaps, and pauses are common features of human voice conversations.
# 4. Feedback: Human voice conversations involve providing feedback to the speaker, such as nodding, making eye contact, or verbal cues like "uh-huh" or "I see." This feedback helps the speaker to understand if the listener is following the conversation and if their message is being understood.
# 5. Context: Human voice conversations are situated in a specific context, such as a physical location, social situation, or cultural background. The context can influence the tone, content, and structure of the conversation.
# 6. Nonverbal communication: Human voice conversations involve nonverbal communication, such as facial expressions, gestures, and body language. These nonverbal cues can convey emotions, attitudes, and intentions that are not expressed verbally.
# 7. Spontaneity: Human voice conversations are often spontaneous and unplanned, requiring speakers to think on their feet and respond to unexpected questions or comments.

# By understanding these elements, an artificial intelligence assistant can create a human-like voice conversation that is engaging, informative, and helpful to the user.

# Human-like conversation response patterns:
# - response is short and precise and relevant to user question.
# - If the text does not contain sufficient information to answer the question, do not make up information and give the answer as I don't know, please be specific.

# """