from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

# import os
# from dotenv import dotenv_values
# system_config = {
#     **dotenv_values("env.shared"),  # load shared development variables
#     **dotenv_values("env.secret"),  # load sensitive variables
#     **os.environ,  # override loaded values with environment variables
# }
# pip install ollama
# pip install gradio
# import os
# from dotenv import dotenv_values
# system_config = dotenv_values("env_config")

import base64
import time
import random
#from openai import OpenAI
import ollama
from ollama import Client
import asyncio
from ollama import AsyncClient

from ollama._types import Message, Options, RequestError, ResponseError
from typing import Any, AnyStr, Union, Optional, Sequence, Mapping, Literal
import tiktoken
#import pavai.shared.llm.solar.llmtype as llmtype

#openai.api_key = "YOUR_API_KEY"
#prompt = "Enter Your Query Here"
#LLM_PROVIDER=system_config["LLM_PROVIDER"]

API_HOST=config.system_config["SOLAR_LLM_OLLAMA_HOST"]
client = Client(host=API_HOST)
asclient = AsyncClient(host=API_HOST)

# client = Client(host='http://192.168.0.18:12345')
# asclient = AsyncClient(host='http://192.168.0.18:12345')

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_messages_token(messages:list):
    content=""
    if messages is None:
        return 0    
    if isinstance(messages, str):
        content=" ".join(messages)
    else:
        for m in messages:
            content=content+str(m) 
    return num_tokens_from_string(content)

def add_messages(history:list=None,system_prompt:str=None, 
                 user_prompt:str=None,ai_prompt:str=None,image_list:list=None):
    messages=[]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})    
    if ai_prompt:
        messages.append({"role": "assistant", "content": ai_prompt})  
    if image_list:
        messages.append({"role": "user","content": user_prompt,'images': image_list})    
    if history:
        return history+messages
    else:
        return messages

def embeddings(model:str,prompt:str):
    return client.embeddings(model=model, prompt=prompt)

def list_models():
    models = client.list()
    model_names=[]
    for model in models["models"]:
        model_names.append(model['name'])
    return model_names

def pull_models(model):
    return client.pull(model)

async def async_api_calling(prompt:str, history:list=None,
                model:str="zephyr",
                stream: bool = False,
                format: Literal['', 'json'] = '',
                options: Options | None = None,
                keep_alive: float | str | None = None): 
    
    messages = add_messages(user_prompt=prompt, history=history)

    response = await asclient.chat(
        model=model, 
        messages=messages,
        stream=stream,format=format,options=options,keep_alive=keep_alive
        )
    # response = client.chat(
    #     model=model, 
    #     messages=messages,
    #     stream=stream,format=format,options=options,keep_alive=keep_alive
    # )    
    reply_text = response["message"]["content"] 
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)    
    return reply_text, reply_messages

def api_calling(prompt:str, history:list=None,
                model:str="zephyr",
                stream: bool = False,
                format: Literal['', 'json'] = '',
                options: Options | None = None,
                keep_alive: float | str | None = None): 
    
    messages = add_messages(user_prompt=prompt, history=history)
    response = client.chat(
        model=model, 
        messages=messages,
        stream=stream,format=format,options=options,keep_alive=keep_alive
    )    
    reply_text = response["message"]["content"] 
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)    
    return reply_text, reply_messages

def message_and_history(input, chatbot, history): 
    history = history or [] 
    print(history) 
    s = list(sum(history, ())) 
    print(s) 
    s.append(input) 
    print('#########################################') 
    print(s) 
    prompt = ' '.join(s) 
    print(prompt) 
    ##output, output_messages = api_calling(prompt,history) 
    output, output_messages = asyncio.run(async_api_calling(prompt,history))      
    history.append((input, output)) 
    print('------------------') 
    print(history)
    print("*********************")     
    print(output_messages) 
    print("*********************") 
    return history, history

def api_calling_v2(
        api_host:str=None,
        api_key:str="EMPTY",
        active_model:str="zephyr:latest",    
        prompt:str=None,         
        history:list=None,
        system_prompt:list=None,         
        stream: bool = False,
        raw: bool = False,
        format: Literal['', 'json'] = '',
        options: Options | None = None,
        keep_alive: float | str | None = None): 
    
    if api_host is not None:
        client = Client(host=api_host)
        print(f"Use method API host: {api_host} and model: {active_model}")

    messages = add_messages(user_prompt=prompt,system_prompt=system_prompt, history=history)
    t0=time.perf_counter()
    response = client.chat(
        model=active_model, 
        messages=messages,
        stream=stream,format=format,options=options,keep_alive=keep_alive
    )  
    t1=time.perf_counter()
    took_time = t1-t0    
    #reply_status = f"<p align='right'>api done: {response['done']}. It took {took_time:.2f}s</p>"   
    reply_text = response["message"]["content"] 
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)    
    return reply_text, reply_messages, response['done']

def message_and_history_v2(
    api_host:str=None,
    api_key:str="EMPTY",
    active_model:str="zephyr:latest",           
    prompt: str=None,
    chatbot: list = [],
    history: list = [],
    system_prompt: str = None,
    top_p: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    stop_words=["<"],
    presence_penalty: int = 0,
    frequency_penalty: int = 0,
):
    t0=time.perf_counter()
    chatbot = chatbot or []
    print("#########################################")
    print(prompt)
    if isinstance(stop_words, str):
        stop_words=[stop_words]
    else:
        stop_words=["\n", "user:"]        

    options={
        "seed": 228,
        "top_p": top_p,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop_words,
        "numa": False,
        "num_ctx": max_tokens,
    }
    output, output_messages, output_status = api_calling_v2(
        api_host=api_host,
        api_key=api_key,
        active_model=active_model,            
        user_prompt=prompt,
        history=history,
        system_prompt=system_prompt,
        options=options
    )
    chatbot.append((prompt, output))
    print("------------------")
    print(chatbot)
    print("*********************")
    print(output_messages)
    print("*********************")
    tokens = count_messages_token(history)
    t1=time.perf_counter()
    took=(t1-t0)    
    output_status=f"<i>tokens:{tokens} | api status: {output_status} | took {took:.2f} seconds</i>"
    return chatbot, output_messages, output_status

def query_the_image(client, query: str, image_list: list[str], selected_model:str="llava:latest") -> ollama.chat:
    try:
        res = client.chat(
            model=selected_model,
            messages=[
                {
                'role': 'user',
                'content': query,
                'images': image_list,
                }
            ]
        )
    except Exception as e:
        print(f"Error: {e}")
        return None
    return res['message']['content']

def upload_image(
        api_host:str=None,
        api_key:str="EMPTY",
        active_model:str="llava:7b-v1.6-mistral-q5_0",    
        user_prompt:str=None,         
        image=None,
        chatbot:list=None,                 
        history:list=None,
        system_prompt:list=None): 
    
    if image is None:
        print("image removed!")
        return

    if api_host is not None:
        client = Client(host=api_host)
        print(f"Use method API host: {api_host} and model: {active_model}")

    user_prompt=user_prompt.strip()
    if len(user_prompt)==0 and image is not None:
        user_prompt="what is this image about?"

    messages = add_messages(user_prompt=user_prompt,system_prompt=system_prompt, history=history, image_list=[image])
    t0=time.perf_counter()
    response = client.chat(
        model=active_model, 
        messages=messages
    )  
    t1=time.perf_counter()
    took_time = t1-t0    
    reply_status = f"<p align='right'>api done: {response['done']}. It took {took_time:.2f}s</p>"   
    reply_text = response["message"]["content"] 
    #reply_messages = add_messages(ai_prompt=reply_text, history=messages)  
    chatbot.append((image, reply_text))
    return chatbot, messages

def message_and_history_v3(
    api_host:str=None,
    api_key:str="EMPTY",
    active_model:str="zephyr:latest",           
    user_prompt: str=None,
    chatbot: list = [],
    history: list = [],
    system_prompt: str = None,
    top_p: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    stop_words=["<"],
    presence_penalty: int = 0,
    frequency_penalty: int = 0
):
    t0=time.perf_counter()
    chatbot = chatbot or []
    print("#########################################")
    print(user_prompt)
    if isinstance(stop_words, str):
        stop_words=[stop_words]
    else:
        stop_words=["\n", "user:"]        

    options={
        "seed": 228,
        "top_p": top_p,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop_words,
        "numa": False,
        "num_ctx": max_tokens,
    }
    output, output_messages, output_status = api_calling_v2(
        api_host=api_host,
        api_key=api_key,
        active_model=active_model,            
        user_prompt=user_prompt,
        history=history,
        system_prompt=system_prompt,
        options=options
    )
    chatbot.append((user_prompt, output))
    print("------------------")
    print(chatbot)
    print("*********************")
    print(output_messages)
    print("*********************")
    tokens = count_messages_token(history)
    t1=time.perf_counter()
    took=(t1-t0)    
    output_status=f"<i>tokens:{tokens} api status: {output_status} took {took:.2f}s</i>"
    return chatbot, output_messages, output_status    

def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')


# if __name__=="__main__":
#     #client = Client(host=API_HOST)
#     client = Client(host='http://192.168.0.18:12345')
#     models = client.list()
#     model_names=[]
#     for model in models["models"]:
#         model_names.append(model['name'])

#     print(model_names)
#     # asclient = AsyncClient(host='http://192.168.0.18:12345')

