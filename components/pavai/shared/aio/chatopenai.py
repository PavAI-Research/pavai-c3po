# pip install openai
# pip install gradio
import os
from dotenv import dotenv_values
system_config = dotenv_values("env_config")

import time
import base64
import json
import requests
import io
import json
import binascii
from openai import OpenAI
from os import PathLike
from pathlib import Path
from hashlib import sha256
from base64 import b64encode, b64decode
from typing import Any, AnyStr, Union, Optional, Sequence, Mapping, Literal
import tiktoken
from rich import pretty
pretty.install()

# Streaming endpoint
#API_HOST = "http://192.168.0.18:12345"
#API_HOST = "http://192.168.0.29:8004"
#LLM_PROVIDER=system_config["LLM_PROVIDER"]

API_HOST=system_config["SOLAR_LLM_DEFAULT_HOST"]
API_URL_BASE = f"{API_HOST}/v1"
API_URL_CHAT = f"{API_URL_BASE}/chat/completions"


#prompt = "Enter Your Query Here"

client = OpenAI(
    base_url=API_URL_BASE,
    api_key="pavai",  # required, but unused
)

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_messages_token(messages:list):
    if messages is None:
        return 0
    content=""
    if isinstance(messages, str):
        content=" ".join(messages)
    else:
        for m in messages:
            content=content+str(m) 
    return num_tokens_from_string(content)

def add_messages_v1(
    history: list = None,
    system_prompt: str = None,
    user_prompt: str = None,
    ai_prompt: str = None,
):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    if ai_prompt:
        messages.append({"role": "assistant", "content": ai_prompt})
    if history:
        return history + messages
    else:
        return messages

def add_messages(history:list=None,system_prompt:str=None, 
                 user_prompt:str=None,ai_prompt:str=None,image_list:list=None, base64_image=None):
    messages=[]
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
            ],})            
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
            ],})                   
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})    
    if ai_prompt:
        messages.append({"role": "assistant", "content": ai_prompt})   
    if history:
        return history+messages
    else:
        return messages

def api_calling_streaming(
    prompt: str, chatbot: list = [], history: list = [], model: str = "zephyr:latest"
):
    messages = add_messages(user_prompt=prompt, history=history)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.2,
        stream=True,
    )
    print(f"Logging : streaming_api response code - {response}")
    # chatbot = list(sum(messages, ()))
    # chatbot.append((prompt,""))
    if len(chatbot) == 0:
        chatbot = [[]]
    else:
        chatbot[-1][1] = ""
    history = messages
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content != "<":
            partial_message += chunk.choices[0].delta.content
            time.sleep(0.05)
            history.append({"role": "assistant", "content": partial_message})
            yield [(prompt, partial_message)], history

def api_calling(prompt: str, history: list = None, model: str = "zephyr:latest"):
    messages = add_messages(user_prompt=prompt, history=history)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.2,
    )
    reply_text = response.choices[0].message.content
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)
    return reply_text, reply_messages

def message_and_history(input: str, chatbot: list = [], history: list = []):
    history = history or []
    print(history)
    s = list(sum(history, ()))
    print(s)
    s.append(input)
    print("#########################################")
    print(s)
    prompt = " ".join(s)
    print(prompt)
    output, output_messages = api_calling(prompt, history)
    history.append((input, output))
    print("------------------")
    print(history)
    print("*********************")
    print(output_messages)
    print("*********************")
    return history, history

def api_calling_v2(
    api_host:str=None,
    api_key:str="EMPTY",
    active_model:str="zephyr:latest",    
    prompt: str=None,
    history: list = None,
    system_prompt: str = None,
    top_p: int = 1,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    stop_words=["<"],
    presence_penalty: int = 0,
    frequency_penalty: int = 0
):
    if api_host is not None:
        client = OpenAI(
            base_url=f"{api_host}/v1",
            api_key=api_key,  # required, but unused
        )
        print(f"Use method API host: {api_host} and model: {active_model}")

    if isinstance(stop_words,str):
        stop_words=[stop_words]

    messages = add_messages(
        user_prompt=prompt, system_prompt=system_prompt, history=history
    )
    response = client.chat.completions.create(
        model=active_model,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        top_p=top_p,
        stop=stop_words,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
    )
    reply_status = response.choices[0].finish_reason
    reply_text = response.choices[0].message.content
    reply_messages = add_messages(ai_prompt=reply_text, history=messages)
    return reply_text, reply_messages, reply_status

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

    output, output_messages, output_status = api_calling_v2(
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

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
        client = OpenAI(
            base_url=f"{api_host}/v1",
            api_key=api_key,  # required, but unused
        )        
        print(f"Use method API host: {api_host} and model: {active_model}")

    user_prompt=user_prompt.strip()
    if len(user_prompt)==0 and image is not None:
        user_prompt="What’s in this image?"
    if image is None:
        raise ValueError("missing image")

    if LLM_PROVIDER=="ollama":
        ## deletegate to ollama client due incompatability with openai
        import chatollama
        return chatollama.upload_image(api_host=api_host,api_key=api_key,active_model=active_model,user_prompt=user_prompt,
                            image=image,chatbot=chatbot,history=history,system_prompt=system_prompt)
    else:
        image_data=encode_image(image)
        messages = add_messages(user_prompt=user_prompt,system_prompt=system_prompt, history=history,
                                base64_image=image_data)
    t0=time.perf_counter()
    response = client.chat.completions.create(
        model=active_model, 
        messages=messages
    )  
    t1=time.perf_counter()
    took_time = t1-t0    
    reply_status = f"<p align='right'>api done: {response.choices[0].finish_reason}. It took {took_time:.2f}s</p>"   
    reply_text = response.choices[0].message.content    
    #reply_messages = add_messages(ai_prompt=reply_text, history=messages)  
    chatbot.append((image, reply_text))
    return chatbot, messages

def chatstream(
    openai_gpt4_key: str,
    system_msg: str,
    inputs,
    top_p,
    temperature,
    chat_counter,
    chatbot=[],
    history=[],
    model: str = "zephyr",
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_gpt4_key}",  # Users will provide their own OPENAI_API_KEY
    }
    print(f"system message is ^^ {system_msg}")
    if system_msg.strip() == "":
        initial_message = [
            {"role": "user", "content": f"{inputs}"},
        ]
        multi_turn_message = []
    else:
        initial_message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{inputs}"},
        ]
        multi_turn_message = [
            {"role": "system", "content": system_msg},
        ]

    if chat_counter == 0:
        payload = {
            "model": model,
            "messages": initial_message,
            "temperature": 1.0,
            "top_p": 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }
        print(f"chat_counter - {chat_counter}")
    else:  # if chat_counter != 0 :
        messages = multi_turn_message  # Of the type of - [{"role": "system", "content": system_msg},]
        for data in chatbot:
            user = {}
            user["role"] = "user"
            user["content"] = data[0]
            assistant = {}
            assistant["role"] = "assistant"
            assistant["content"] = data[1]
            messages.append(user)
            messages.append(assistant)
        temp = {}
        temp["role"] = "user"
        temp["content"] = inputs
        messages.append(temp)
        # messages
        payload = {
            "model": model,
            "messages": messages,  # Of the type of [{"role": "user", "content": f"{inputs}"}],
            "temperature": temperature,  # 1.0,
            "top_p": top_p,  # 1.0,
            "n": 1,
            "stream": True,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        }

    chat_counter += 1

    history.append(inputs)
    print(f"Logging : payload is - {payload}")
    # make a POST request to the API endpoint using the requests.post method, passing in stream=True
    response = requests.post(API_URL_CHAT, headers=headers, json=payload, stream=True)
    print(f"Logging : response code - {response.status_code}")
    token_counter = 0
    partial_words = ""

    counter = 0
    for chunk in response.iter_lines():
        # Skipping first chunk
        if counter == 0:
            counter += 1
            continue
        # check whether each line is non-empty
        if chunk.decode():
            chunk = chunk.decode()
            # decode each line as response data is in bytes
            if (
                len(chunk) > 12
                and "content" in json.loads(chunk[6:])["choices"][0]["delta"]
            ):
                partial_words = (
                    partial_words
                    + json.loads(chunk[6:])["choices"][0]["delta"]["content"]
                )
                if token_counter == 0:
                    if partial_words != "," or partial_words != "!":
                        history.append(" " + partial_words)
                else:
                    history[-1] = partial_words
                chat = [
                    (history[i], history[i + 1]) for i in range(0, len(history) - 1, 2)
                ]  # convert to tuples of list
                token_counter += 1
                yield chat, history, chat_counter, f"Response status: {response.status_code}"  # resembles {chatbot: chat, state: history}

def list_models():
    if LLM_PROVIDER=="ollama":
        API_URL_MODELS = f"{API_HOST}/api/tags"
        model_names=list_models_ollama(API_URL_MODELS)
    else:
        API_URL_MODELS = f"{API_HOST}/v1/models"
        model_names=list_models_openai(API_URL_MODELS)
    return model_names

def list_models_openai(api_url_models:str):
    response = requests.get(api_url_models)
    models = json.loads(response.text)
    model_names = []
    for model in models["data"]:
        model_names.append(model["id"])
    print(f"Logging : response code - {model_names}")
    return model_names

def list_models_ollama(api_url_models:str):
    response = requests.get(api_url_models)
    result = json.loads(response.text)
    model_names = []
    for model in result["models"]:
        model_names.append(model["model"])
    print(f"Logging : response code - {model_names}")
    return model_names

# Image to Base 64 Converter
def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode("utf-8")

def _as_path(s: Optional[Union[str, PathLike]]) -> Union[Path, None]:
  if isinstance(s, str) or isinstance(s, Path):
    try:
      if (p := Path(s)).exists():
        return p
    except Exception:
      ...
  return None

def _as_bytesio(s: Any) -> Union[io.BytesIO, None]:
  if isinstance(s, io.BytesIO):
    return s
  elif isinstance(s, bytes):
    return io.BytesIO(s)
  return None

def _encode_image(image) -> str:
  """
  >>> _encode_image(b'ollama')
  'b2xsYW1h'
  >>> _encode_image(io.BytesIO(b'ollama'))
  'b2xsYW1h'
  >>> _encode_image('LICENSE')
  'TUlUIExpY2Vuc2UKCkNvcHlyaWdodCAoYykgT2xsYW1hCgpQZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYSBjb3B5Cm9mIHRoaXMgc29mdHdhcmUgYW5kIGFzc29jaWF0ZWQgZG9jdW1lbnRhdGlvbiBmaWxlcyAodGhlICJTb2Z0d2FyZSIpLCB0byBkZWFsCmluIHRoZSBTb2Z0d2FyZSB3aXRob3V0IHJlc3RyaWN0aW9uLCBpbmNsdWRpbmcgd2l0aG91dCBsaW1pdGF0aW9uIHRoZSByaWdodHMKdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLCBhbmQvb3Igc2VsbApjb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGUgU29mdHdhcmUgaXMKZnVybmlzaGVkIHRvIGRvIHNvLCBzdWJqZWN0IHRvIHRoZSBmb2xsb3dpbmcgY29uZGl0aW9uczoKClRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluIGFsbApjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLgoKVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEICJBUyBJUyIsIFdJVEhPVVQgV0FSUkFOVFkgT0YgQU5ZIEtJTkQsIEVYUFJFU1MgT1IKSU1QTElFRCwgSU5DTFVESU5HIEJVVCBOT1QgTElNSVRFRCBUTyBUSEUgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFksCkZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuIElOIE5PIEVWRU5UIFNIQUxMIFRIRQpBVVRIT1JTIE9SIENPUFlSSUdIVCBIT0xERVJTIEJFIExJQUJMRSBGT1IgQU5ZIENMQUlNLCBEQU1BR0VTIE9SIE9USEVSCkxJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HIEZST00sCk9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFNPRlRXQVJFIE9SIFRIRSBVU0UgT1IgT1RIRVIgREVBTElOR1MgSU4gVEhFClNPRlRXQVJFLgo='
  >>> _encode_image(Path('LICENSE'))
  'TUlUIExpY2Vuc2UKCkNvcHlyaWdodCAoYykgT2xsYW1hCgpQZXJtaXNzaW9uIGlzIGhlcmVieSBncmFudGVkLCBmcmVlIG9mIGNoYXJnZSwgdG8gYW55IHBlcnNvbiBvYnRhaW5pbmcgYSBjb3B5Cm9mIHRoaXMgc29mdHdhcmUgYW5kIGFzc29jaWF0ZWQgZG9jdW1lbnRhdGlvbiBmaWxlcyAodGhlICJTb2Z0d2FyZSIpLCB0byBkZWFsCmluIHRoZSBTb2Z0d2FyZSB3aXRob3V0IHJlc3RyaWN0aW9uLCBpbmNsdWRpbmcgd2l0aG91dCBsaW1pdGF0aW9uIHRoZSByaWdodHMKdG8gdXNlLCBjb3B5LCBtb2RpZnksIG1lcmdlLCBwdWJsaXNoLCBkaXN0cmlidXRlLCBzdWJsaWNlbnNlLCBhbmQvb3Igc2VsbApjb3BpZXMgb2YgdGhlIFNvZnR3YXJlLCBhbmQgdG8gcGVybWl0IHBlcnNvbnMgdG8gd2hvbSB0aGUgU29mdHdhcmUgaXMKZnVybmlzaGVkIHRvIGRvIHNvLCBzdWJqZWN0IHRvIHRoZSBmb2xsb3dpbmcgY29uZGl0aW9uczoKClRoZSBhYm92ZSBjb3B5cmlnaHQgbm90aWNlIGFuZCB0aGlzIHBlcm1pc3Npb24gbm90aWNlIHNoYWxsIGJlIGluY2x1ZGVkIGluIGFsbApjb3BpZXMgb3Igc3Vic3RhbnRpYWwgcG9ydGlvbnMgb2YgdGhlIFNvZnR3YXJlLgoKVEhFIFNPRlRXQVJFIElTIFBST1ZJREVEICJBUyBJUyIsIFdJVEhPVVQgV0FSUkFOVFkgT0YgQU5ZIEtJTkQsIEVYUFJFU1MgT1IKSU1QTElFRCwgSU5DTFVESU5HIEJVVCBOT1QgTElNSVRFRCBUTyBUSEUgV0FSUkFOVElFUyBPRiBNRVJDSEFOVEFCSUxJVFksCkZJVE5FU1MgRk9SIEEgUEFSVElDVUxBUiBQVVJQT1NFIEFORCBOT05JTkZSSU5HRU1FTlQuIElOIE5PIEVWRU5UIFNIQUxMIFRIRQpBVVRIT1JTIE9SIENPUFlSSUdIVCBIT0xERVJTIEJFIExJQUJMRSBGT1IgQU5ZIENMQUlNLCBEQU1BR0VTIE9SIE9USEVSCkxJQUJJTElUWSwgV0hFVEhFUiBJTiBBTiBBQ1RJT04gT0YgQ09OVFJBQ1QsIFRPUlQgT1IgT1RIRVJXSVNFLCBBUklTSU5HIEZST00sCk9VVCBPRiBPUiBJTiBDT05ORUNUSU9OIFdJVEggVEhFIFNPRlRXQVJFIE9SIFRIRSBVU0UgT1IgT1RIRVIgREVBTElOR1MgSU4gVEhFClNPRlRXQVJFLgo='
  >>> _encode_image('YWJj')
  'YWJj'
  >>> _encode_image(b'YWJj')
  'YWJj'
  """

  if p := _as_path(image):
    return b64encode(p.read_bytes()).decode('utf-8')

  try:
    b64decode(image, validate=True)
    return image if isinstance(image, str) else image.decode('utf-8')
  except (binascii.Error, TypeError):
    ...

  if b := _as_bytesio(image):
    return b64encode(b.read()).decode('utf-8')

  raise ValueError('image must be bytes, path-like object, or file-like object')

