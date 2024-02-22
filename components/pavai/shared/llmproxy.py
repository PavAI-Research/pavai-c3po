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
import time
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
logger.info(os.getcwd())
import functools

#import shared.solar.llmprompt as llmprompt 
# system_prompt_assistant
import shared.solar.llmcognitive as llmcognitive 
from shared.aio.llmchat import get_llm_instance,llm_chat_completion
from shared.solar.llmprompt import system_prompt_assistant
import functools

from shared.solar.llmchat import multimodalchat
import shared.solar.llmchat as llmchat
import shared.solar.llmcognitive as llmcognitive 
import shared.solar.llmprompt as llmprompt 
from openai import OpenAI

print("--GLOBAL SYSTEM MODE----")
print(system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=system_config["GLOBAL_TTS"]
_GLOBAL_TTS_LIBRETTS_VOICE=system_config["GLOBAL_TTS_LIBRETTS_VOICE"]
_GLOBAL_STT=system_config["GLOBAL_STT"]
_GLOBAL_DEFAULT_LLM_MODEL_FILE=system_config["DEFAULT_LLM_MODEL_FILE"]

_GLOBAL_DEFAULT_LLM_MODEL_INFO=[system_config["DEFAULT_LLM_MODEL_FILE"],
                        system_config["DEFAULT_LLM_MODEL_WEB_URL"],
                        system_config["DEFAULT_LLM_MODEL_CHAT_FORMAT"]
                        ]
# global
llmsolar = None

def chatbot_ui_client(input_text:str, chat_history:list,
                        system_prompt: str = None,
                        ask_expert: str = None,                      
                        target_model_info:list=None,
                        user_override:bool=False,
                        skip_content_safety_check:bool=True,
                        skip_data_security_check:bool=True,
                        skip_self_critique_check:bool=True                      
                      ):
    t0=time.perf_counter()    
    query_system_prompt = system_prompt if system_prompt is not None else system_prompt_assistant
    query_ask_expert = ask_expert if ask_expert is not None else None    
    if _GLOBAL_SYSTEM_MODE=="SOLAR":
        if target_model_info is not None:
            reply_text,chat_history = chat_service (user_prompt=input_text, 
                                                     history=chat_history, 
                                                     system_prompt=query_system_prompt,
                                                     ask_expert = query_ask_expert,
                                                     target_model_info=target_model_info,
                                                     user_override=user_override,
                                                     skip_content_safety_check=skip_content_safety_check,
                                                     skip_data_security_check=skip_data_security_check,
                                                     skip_self_critique_check=skip_self_critique_check                                                                           
                                                     )
        else:
            reply_text,chat_history = chat_service (user_prompt=input_text, 
                                                     history=chat_history, 
                                                     ask_expert = query_ask_expert,
                                                     system_prompt=query_system_prompt,
                                                     user_override=user_override,
                                                     skip_content_safety_check=skip_content_safety_check,
                                                     skip_data_security_check=skip_data_security_check,
                                                     skip_self_critique_check=skip_self_critique_check                                                                                                                                
                                                     )
    else: 
        ## ask experts 
        system_prompt = system_prompt_assistant
        chatbot_ui_messages, chat_history, reply_text = llm_chat_completion(user_Prompt=input_text, 
                                                                            history= chat_history,
                                                                            system_prompt = system_prompt,
                                                                            ask_expert = query_ask_expert, 
                                                                            target_model_info=target_model_info                                                                           
                                                                            )
        logger.debug(f"[blue]{reply_text}[/blue]",extra=dict(markup=True))    

    # cleanup messages 
    clean_chatbot_ui_messages=[]  
    if chat_history is not None and len(chat_history)>1:  
        for i in range(0, len(chat_history)-1):
            if "system" != chat_history[i]["role"]:
                clean_chatbot_ui_messages.append((chat_history[i]["content"], chat_history[i+1]["content"]))                                

    t1=time.perf_counter()-t0
    logger.info(f"llmproxy.chatbot_ui_client: took {t1:.6f} seconds")
    return clean_chatbot_ui_messages, chat_history, reply_text    

def multimodal_ui_client(input_text:str, chat_history:list,
                        system_prompt: str = None,
                        user_query:str = "What does the image say?",
                        ask_expert: str = None,                      
                        target_model_info:list=None,
                        user_override:bool=False,
                        skip_content_safety_check:bool=True,
                        skip_data_security_check:bool=True,
                        skip_self_critique_check:bool=True                      
                      ):
    t0=time.perf_counter()    

    logger.debug("-----multimodal_ui_client----")
    # image_url: file:////tmp/gradio/26f63cc5e677d6e4b106a13413ac942c021ecb67/invoice1.png
    # image_url:"https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/02/Left-Handed-Male-Writing-1296x728-Header.jpg?w=1155&h=1528"
    user_image_url=input_text
    domain_client = OpenAI(
        api_key=f"{system_config['SOLAR_LLM_DOMAIN_API_KEY']}",
        base_url=f"{system_config['SOLAR_LLM_DOMAIN_SERVER_URL']}"
    )
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            history=[],
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "text"}, model_id="llava-v1.5-7b")
    logger.debug(reply_object["response"])
    history=normalize_imagechat_history(history)
    reply_messages=[]
    for i in range(0, len(history)-1):
        if history[i]["role"] != "system":
            reply_messages.append((history[i]["content"], history[i+1]["content"]))
    t1=time.perf_counter()-t0
    logger.info(f"llmproxy.multimodal_ui_client: took {t1:.6f} seconds")
    return reply_messages, history, reply_object["response"]    


@functools.lru_cache
def get_solarclient(user_override:bool=False,
                    skip_content_safety_check:bool=True,
                    skip_data_security_check:bool=True,
                    skip_self_critique_check:bool=True
                    ):
    global llmsolar
    #if llmsolar is not None:
    #    return llmsolar
    default_url=system_config["SOLAR_LLM_DEFAULT_SERVER_URL"] 
    default_api_key=system_config["SOLAR_LLM_DEFAULT_API_KEY"]             
    domain_url=system_config["SOLAR_LLM_DOMAIN_SERVER_URL"] 
    domain_api_key=system_config["SOLAR_LLM_DOMAIN_API_KEY"]     
    if not user_override:
        skip_content_safety_check=system_config["SOLAR_SKIP_CONTENT_SAFETY_CHECK"]    
        skip_content_safety_check = True if skip_content_safety_check=="true" else False
        skip_data_security_check=system_config["SOLAR_SKIP_DATA_SECURITY_CHECK"] 
        skip_data_security_check = True if skip_data_security_check=="true" else False            
        skip_self_critique_check=system_config["SOLAR_SKIP_SELF_CRITIQUE_CHECK"]
        skip_self_critique_check = True if skip_self_critique_check=="true" else False       
    logger.debug("---LLMSolarClient:Settings---")
    logger.debug(f"SOLAR_LLM_DEFAULT_SERVER_URL: {default_url}")
    logger.debug(f"SOLAR_LLM_DEFAULT_API_KEY: {default_api_key}")
    logger.debug(f"SOLAR_LLM_DOMAIN_SERVER_URL: {domain_url}")
    logger.debug(f"SOLAR_LLM_DOMAIN_API_KEY: {domain_api_key}")
    logger.debug(f"SOLAR_SKIP_CONTENT_SAFETY_CHECK: {skip_content_safety_check}")
    logger.debug(f"SOLAR_SKIP_DATA_SECURITY_CHECK: {skip_data_security_check}")
    logger.debug(f"SOLAR_SKIP_SELF_CRITIQUE_CHECK: {skip_self_critique_check}")
    llmsolar = llmcognitive.LLMSolarClient(default_url=default_url,
                                            default_api_key=default_api_key,
                                            domain_url=domain_url,
                                            domain_api_key=domain_api_key,
                                            skip_content_safety_check=skip_content_safety_check,
                                            skip_data_security_check=skip_data_security_check,
                                            skip_self_critique_check=skip_self_critique_check)       
    return llmsolar    
  
def chat_service(user_prompt:str=None, history:list=[], 
                   system_prompt: str = system_prompt_assistant,
                   stop_criterias=["</s>"],
                   ask_expert:str=None,                   
                   target_model_info:list=None,
                   user_override:bool=False,
                    skip_content_safety_check:bool=True,
                    skip_data_security_check:bool=True,
                    skip_self_critique_check:bool=True                   
                   ):    
    global llmsolar
    t0=time.perf_counter()
    if _GLOBAL_SYSTEM_MODE=="SOLAR":
        llmsolar=get_solarclient(user_override,skip_content_safety_check,
                    skip_data_security_check,skip_self_critique_check)
        cleanup=False
        input_data = {"input_query": user_prompt}        
        if ask_expert is not None:
            input_data["ask_expert"]= ask_expert                  
        if target_model_info is not None:
            input_data["input_model_id"]= target_model_info[0]                              
            input_data["input_model_web_url"]= target_model_info[1]                                          
            input_data["input_model_web_chatformat"]= target_model_info[2]                                                      
        output,history=llmsolar.chat(input_data,history,cleanup)
        logger.debug(f"[blue]{output['output_text']}[/blue]",extra=dict(markup=True))
        reply =output['output_text']
    else:
        messages, history, reply = llm_chat_completion(user_Prompt=user_prompt, 
                                                       history= history,system_prompt = system_prompt,
                                                       stop_criterias=stop_criterias,
                                                        ask_expert = ask_expert, 
                                                        target_model_info=target_model_info                                                                                                                                  
                                                       )
        logger.debug(f"[blue]{reply}[/blue]",extra=dict(markup=True))
    t1=time.perf_counter()-t0
    logger.debug(f"llmproxy.chat_service: took {t1} seconds")
    return reply,history

def normalize_imagechat_history(history:list):
    clean_history=history
    for i in range(0, len(history)-1):
        if history[i]["role"] != "system":
            if history[i]["role"] == "user":
                if isinstance(history[i]["content"],list):   
                    image_question= history[i]["content"][1]['text']   
                    image_url = history[i]["content"][0]['image_url']['url'] 
                    #image_response = history[i+1]["content"]   
                    #clean_history.append((image_question,image_response))
                    image_question="<p><image src='"+image_url+"' alt='show uploaded image'/>"+image_question+"</p>"
                    history[i]["content"]=image_question
    return clean_history

def word_count(text):
    return (len(text.strip().split(" ")))

def chat_count_tokens(text):
    llm_client = get_llm_instance()
    if llm_client._llm is not None and llm_client._llm._pipeline is not None:
        model = llm_client._llm._pipeline
        return len(model.tokenize(text.encode() if isinstance(text, str) else text))
    else:
        return word_count(text)