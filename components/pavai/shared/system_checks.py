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

# from dotenv import dotenv_values
# system_config = dotenv_values("env_config")
# import logging
# from rich.logging import RichHandler
from rich import print,pretty,console
from rich.pretty import (Pretty,pprint)
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress
#logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])

#logger = logging.getLogger(__name__)

#pretty.install()
import sys,os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
logger.info(os.getcwd())

import warnings 
warnings.filterwarnings("ignore")
import nltk
import os
import time
import spacy
import torch
import platform
import gradio as gr
import psutil
from psutil import virtual_memory
from transformers.utils import is_flash_attn_2_available
import random
import sounddevice as sd
from openai import OpenAI
from pavai.shared.grammar import (get_or_download_grammar_model_snapshot,init_grammar_correction_model,
                          fix_grammar_error, DEFAULT_GRAMMAR_MODEL_SIZE)
from pavai.shared.audio.transcribe import (get_or_download_whisper_model_snapshot,get_transcriber,speech_to_text, DEFAULT_WHISPER_MODEL_SIZE)


#from pavai.shared.audio.stt_vad import init_vad_model
# from pavai.shared.audio.voices_piper import espeak,get_voice_model_file
#from pavai.shared.audio.tts_client import text_speaker_ai
from pavai.shared.audio.tts_client import system_tts_local
#import pavai.shared.solar.llmchat as llmchat
import pavai.llmone.llmproxy as llmproxy
#from pavai.llmone.local.localllm import (get_llm_instance, local_chat_completion,  LLM_Setting, LLMClient,LLMllamaLocal, AbstractLLMClass)
# import pavai.llmone.local.localllm as localllm

from pathlib import Path
from pavai.shared.styletts2.download_models import get_styletts2_model_files
import traceback

__RELEASE_VERSION__="alpha-0.0.3"
__RELEASE_DATE__="2024/01/07"

DEFAULT_SYSTEM_MODE = "system_mode_all_in_one"

logger.info("--GLOBAL SYSTEM MODE----")
logger.info(config.system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=config.system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=config.system_config["GLOBAL_TTS"]
_GLOBAL_TTS_LIBRETTS_VOICE=config.system_config["GLOBAL_TTS_LIBRETTS_VOICE"]
_GLOBAL_STT=config.system_config["GLOBAL_STT"]
_GLOBAL_TTS_API_ENABLE=config.system_config["GLOBAL_TTS_API_ENABLE"]
_GLOBAL_TTS_API_URL=config.system_config["GLOBAL_TTS_API_URL"]
_GLOBAL_TTS_API_LANGUAGE=config.system_config["GLOBAL_TTS_API_LANGUAGE"]
_GLOBAL_TTS_API_SPEAKER_MODEL=config.system_config["GLOBAL_TTS_API_SPEAKER_MODEL"]

SYSTEM_THEME = "theme"
SYSTEM_THEME_SOFT = "soft"
SYSTEM_THEME_GLASS = "glass"
SYSTEM_THEME_DEFAULT = "default"

PRODUCT_NAME = "PAVAI"
#DEFAULT_SYSTEM_AI_NAME = "Amy"

PAVAI_APP_VOCIE="PAVAI C3PO"
DEFAULT_PAVAI_VOCIE_AGENT="Jane"

DEFAULT_PAVAI_STARTUP_MSG_INTRO = " your personal multilingual AI assistant for everyday tasks, how may I help you today?"
DEFAULT_PAVAI_STARTUP_MSG_INTRO2 = " an AI assistant. I can help you find answers on everyday tasks, do you have a question for me?"
DEFAULT_PAVAI_STARTUP_MSG_INTRO3 = " I am not a human being but rather an advanced artificial intelligence system designed to understand and respond to your queries. do you have any questions for me?"
DEFAULT_PAVAI_STARTUP_MSG_INTRO4 = " I can perform a wide range of tasks to assist you, including Answering questions, Providing recommendations, Generating text, Translating text, Solving math problems, Providing definitions so these are just a few examples of what I can do"
DEFAULT_PAVAI_STARTUP_MSG_INTRO5 = " how may I help you today?"
#DEFAULT_PAVAI_STARTUP_MSG_INTRO6 = """an AI-powered assistant, and my primary role is to answer questions and provide information to users like you. what can I do for you today?"""

DEFAULT_PAVAI_VOCIE_STARTUP_MSG_NEXT_STEP = "ready? let's open your browsers and type in web url http://localhost:7860; or enter a secured url to start using the system UI."

PAVAI_APP_TALKIE="PavAI Talkie"
DEFAULT_PAVAI_TALKIE_AGENT="Jane"
DEFAULT_PAVAI_TALKIE_STARTUP_MSG_INTRO = "Hi,I am Jane your personal multilingual PavAI.Talkie AI assistant for everyday tasks, how may I help you today?"
# DEFAULT_PAVAI_TALKIE_STARTUP_MSG_INTRO_2 = "Hello, Ryan is here. how may I help you today?"
# DEFAULT_PAVAI_TALKIE_STARTUP_MSG_INTRO_3 = f"Greetings, I am an AI language model designed to assist you with information, answers, and suggestions based on the text provided. I can't perform tasks outside of our text interactions, but I strive to provide helpful and accurate responses in a timely manner.?"
# DEFAULT_PAVAI_TALKIE_STARTUP_MSG_INTRO_4 = f"Hey, I am Ryan your personal a multilingual PavAI.Talkie AI assistant for everyday tasks, how may I help you today?"

DEFAULT_PAVAI_TALKIE_STARTUP_MSG_NEXT_STEP = "ready, I am listening."
### IMPORTANT: localhost is required for use of microphone in the web browser.

DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_RUNNING ="working on system startup checks!"
DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_SUCCESS ="System startup check success!"
DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED = "Oops, system startup check failed!"
DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED_RECOVER = "please check the console log for cause of failure, fix the issue then try start again."

DEFAULT_SYSTEM_WORKSPACE = "workspace"
DEFAULT_SYSTEM_DOWNLOADS_PATH = "workspace/downloads"
DEFAULT_SYSTEM_LLM_MODEL_PATH = "resources/models/llm"
DEFAULT_SYSTEM_VOICE_MODEL_PATH = "resources/models/llm"
DEFAULT_SYSTEM_VOICE_MODEL_PATH = "resouces/models/voices"
DEFAULT_SYSTEM_VAD_MODEL_PATH = "resources/models/silero-vad"

# LLM Model and hyper-parameters
DEFAULT_LLM_MODEL_PATH = "resources/models/llm"
DEFAULT_LLM_MODEL_FILE = "zephyr-7b-beta.Q5_K_M.gguf"
DEFAULT_LLM_MODEL_CHAT_FORMAT = "chatml"
DEFAULT_LLM_OFFLOAD_GPU_LAYERS = 35

# Grammar synthesis model
DEFAULT_GRAMMAR_MODEL_ID = "pszemraj/grammar-synthesis-small"

# Image Generation Model
DEFAULT_TEXT_TO_IMAGE_MODEL_ID = "segmind/SSD-1B"

# speech-to-text Whisper Model
DEFAULT_STT_MODEL_SIZE = "large"
DEFAULT_STT_MODEL_ID = "Systran/faster-whisper-large-v3"

# text-to-speech Voice model
# DEFAULT_TTS_VOICE_MODEL_PATH=/home/pop/development/mclab/talking-llama/models/voices/
DEFAULT_TTS_VOICE_MODEL_PATH = "resources/models/voices/"
DEFAULT_TTS_VOICE_MODEL_LANGUAGE = "en"
# DEFAULT_TTS_VOICE_MODEL_GENDER = "Amy"
# DEFAULT_TTS_VOICE_MODEL_ONNX_FILE = "en_US-amy-medium.onnx"
# DEFAULT_TTS_VOICE_MODEL_ONNX_JSON = "en_US-amy-medium.onnx.json"

DEFAULT_TTS_VOICE_MODEL_VOCIE="Amy"
DEFAULT_TTS_VOICE_MODEL_VOCIE_ONNX_FILE="en_US-amy-medium.onnx"
DEFAULT_TTS_VOICE_MODEL_VOCIE_ONNX_JSON="en_US-amy-medium.onnx.json"

DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT="Ryan"
DEFAULT_TTS_VOICE_MODEL_TALKIE_ONNX_FILE="en_US-ryan-medium.onnx"
DEFAULT_TTS_VOICE_MODEL_TALKIE_ONNX_JSON="en_US-ryan-medium.onnx.json"

## Application Activated TTS on startup
ACTIVE_TTS_VOICE_MODEL_AGENT=""
ACTIVE_TTS_VOICE_MODEL_ONNX_FILE=""
ACTIVE_ESPEAK_VOICE_LANGUAGE = ""


"""INSTRUCTIONS"""
VOICE_PROMPT_INSTRUCTIONS_TEXT = """
            #### Voice Query Instructions:

            LLM.Vocie is a audio-based voice assistant for everyday tasks. 
            It support voice input for query, image generation and chatbot on data. 

            1. Press `Record from microphone to start 
            2. Speak your query input.
            3. System automatically detected and transcribe speech to text
            4. Upon completion, the system send transcribed text to LLM
            5. Wait for the LLM text response and AI generated voice reponse.
            6. If audio doesn't start automatically, Press the `Play` button in
            the `Output` box.
            7. When ready to provide another input, under `User Audio Input`, press the
            `X` in the top-right corner to clear the last recording. Then press
            `Record from microphone` again and speak your next input.

            Note: When you press `X` you may see a red `Error` in a box. That is normal.
            
            #### Enable Voice-to-Image Generation Instruction

                Click User Options: 
                > click user Options, select enable image generation. 
                > click record to capture your voice prompt.    
                > click logs to most recent list of queries                    

            #### Enable Single-shot Grammar sysnthesis correction model
                this model apply grammar correction to transcribed text.
                hence, correction result show in separate window.
                Click User Options: 
                > click user Options, select enable grammar correction

            #### Start New query
            > click Next Qeury button to start a new query             
            """  
table = Table(title="PAVAI System Health Checks")    
table.add_column("Task", justify="left", style="magenta", no_wrap=True)     
table.add_column("Description", style="blue")
table.add_column("Status", justify="right")    

def wakeup_time(output_voice:str="en"):
    t = time.localtime()
    logger.info(f'Local Time is {t}')
    current_time = time.strftime("%H:%M:%S", t)
    current_time_message = f'Current time is {current_time}'
    return current_time_message

def cpu_usage_bar():
    from tqdm import tqdm
    from time import sleep
    import psutil
    with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
        while True:
            rambar.n=psutil.virtual_memory().percent
            cpubar.n=psutil.cpu_percent()
            rambar.refresh()
            cpubar.refresh()
            sleep(0.5)

def version_info():
    info={}
    info["PAVAI_VERSION"]="Alpha 0.0.3"    
    info["PAVAI_RELEASE_DATE"]="2024/01/08"            
    print(Panel(Pretty(info))) 
    return info   

def environment_info():
    env_info={}
    ram_gb = virtual_memory().total / 1e9
    env_info["platform_system"]=platform.system()    
    env_info["platform_release"]=platform.release()     
    env_info["memory_available"]=ram_gb
    env_info["memory_percent"]=psutil.virtual_memory().percent               
    env_info["cpu_count"]=psutil.cpu_count()
    env_info["cpu_usage"]=psutil.cpu_percent()     
    env_info["os_name"]=os.name      
    env_info["gradio_version"]=gr.__version__
    env_info["torch_version"]=torch.__version__
    # pip install gradio==4.7.1
    use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    use_flash_attention_2 = is_flash_attn_2_available()
    env_info["use device"]=use_device
    if use_device == 'cuda' or use_device == 'cuda:0':
        env_info["gpu_name"]=torch.cuda.get_device_name(0)        
        env_info["gpu_memory_allocated"]=round(torch.cuda.memory_allocated(0)/1024**3, 1)                
        env_info["gpu_memory_reserved"]=round(torch.cuda.memory_reserved(0)/1024**3, 1)                        
    env_info["torch_dtype"]=use_torch_dtype        
    env_info["use_flash_attention_2"]=use_flash_attention_2            
    logger.info("-----------------------------------------------------")
    logger.info("ENVIRONMENT INFO")
    logger.info("-----------------------------------------------------")    
    logger.info(f"PAVAI version: {__RELEASE_VERSION__}")    
    logger.info(f"PAVAI release date: {__RELEASE_DATE__}")        
    logger.info("-----------------------------------------------------")
    logger.info(env_info)      
    logger.info("-----------------------------------------------------")    
    return env_info

def set_default_system_voice(system_agent:str=None):
    global ACTIVE_TTS_VOICE_MODEL_AGENT
    global ACTIVE_TTS_VOICE_MODEL_ONNX_FILE
    global ACTIVE_ESPEAK_VOICE_LANGUAGE
    if system_agent==PAVAI_APP_VOCIE:
        ACTIVE_TTS_VOICE_MODEL_AGENT=DEFAULT_TTS_VOICE_MODEL_VOCIE
        ACTIVE_TTS_VOICE_MODEL_ONNX_FILE=DEFAULT_TTS_VOICE_MODEL_VOCIE_ONNX_FILE
        ACTIVE_ESPEAK_VOICE_LANGUAGE = DEFAULT_TTS_VOICE_MODEL_VOCIE_ONNX_JSON
    elif system_agent==PAVAI_APP_TALKIE:        
        ACTIVE_TTS_VOICE_MODEL_AGENT=DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT
        ACTIVE_TTS_VOICE_MODEL_ONNX_FILE=DEFAULT_TTS_VOICE_MODEL_TALKIE_ONNX_FILE
        ACTIVE_ESPEAK_VOICE_LANGUAGE = DEFAULT_TTS_VOICE_MODEL_TALKIE_ONNX_JSON        
    else: 
        raise Exception ("unknow app name")

def get_speaker_audio_file(workspace_temp:str="workspace/temp")->str:
    Path.mkdir(workspace_temp, exist_ok=True)
    return workspace_temp+"/espeak_text_to_speech.mp3"

def download_spacy_model(model_size:str="en_core_web_sm",storage_dir:str="resources/models/spacy"):
    cache_dir=os.getenv("cache_dir", storage_dir)
    model_path=model_size
    try:
        #creating a new directory if not exist
        Path(storage_dir).mkdir(parents=True, exist_ok=True)        
        nlp = spacy.load(os.path.join(cache_dir,model_path))
    except OSError:
        spacy.cli.download(model_path)
        nlp = spacy.load(model_path)
        nlp.to_disk(os.path.join(cache_dir,model_path))  
   
def system_resources_check(output_voice:str="jane"):
    logger.info("***running system resources checks***")    
    with Progress(transient=True) as progress: 
        try:       
            task = progress.add_task("checking system resources...", total=7)

            # 1. nltk downloads
            logger.info("download nltk [punk] resource files")                                              
            nltk.download('punkt')
            table.add_row("resource_check", f"get nltk:punkt", "[green]Found[/]")
            logger.info(f"1.Found nltk:punkt downloads ")               
            progress.advance(task)

            # 2. spacy en_core_web_lg model downloads
            logger.info("download spacy en_core_web_lg model")  
            download_spacy_model()                                            
            # python -m spacy download en_core_web_lg
            table.add_row("resource_check", f"get spacy en_core_web_lg model", "[green]Found[/]")
            logger.info(f"2.Found spacy model downloads")               
            progress.advance(task)

            logger.info("checking text-to-speech file exist")                                  
            # 3. voice model file        
            # local_voice_path = system_config["DEFAULT_TTS_VOICE_MODEL_PATH"]
            # isExist = os.path.exists(local_voice_path)
            # if not isExist:
            #     os.makedirs(local_voice_path)
            # voice_onnx_file, voice_json_file = get_voice_model_file(local_voice_path=system_config["DEFAULT_TTS_VOICE_MODEL_PATH"],
            #                                                         spoken_language="en")
            # if voice_onnx_file is not None and voice_json_file is not None:
            #     table.add_row("resource_check", f"get text-to-speech model file: {voice_json_file}", "[green]Found[/]")
            #     logger.info(f"3.Found text-to-speech file {voice_json_file}")               
            # else:
            #     table.add_row("resource_check", f"get text-to-speech model file: {voice_json_file}", "[red]Missing[/]")        
            #     logger.error(f"Missing text-to-speech file {voice_json_file}")                           
            progress.advance(task)

            # 4. LLM model file        
            if _GLOBAL_SYSTEM_MODE=="solar-openai":
                logger.info("checking SOLAR LLM server configuration exist")
                default_url=config.system_config["SOLAR_LLM_DEFAULT_SERVER_URL"] 
                default_api_key=config.system_config["SOLAR_LLM_DEFAULT_API_KEY"]             
                default_model_id=config.system_config["SOLAR_LLM_DEFAULT_MODEL_ID"]                             
                skip_content_safety_check=config.system_config["SOLAR_SKIP_CONTENT_SAFETY_CHECK"]    
                skip_data_security_check=config.system_config["SOLAR_SKIP_DATA_SECURITY_CHECK"] 
                skip_self_critique_check=config.system_config["SOLAR_SKIP_SELF_CRITIQUE_CHECK"]
            elif _GLOBAL_SYSTEM_MODE=="ollama-openai":
                logger.info("checking SOLAR Ollama server configuration exist")
                default_url=config.system_config["SOLAR_LLM_OLLAMA_SERVER_URL"] 
                default_api_key=config.system_config["SOLAR_LLM_OLLAMA_API_KEY"]             
                default_model_id=config.system_config["SOLAR_LLM_OLLAMA_MODEL_ID"]             
                skip_content_safety_check=config.system_config["SOLAR_SKIP_CONTENT_SAFETY_CHECK"]    
                skip_data_security_check=config.system_config["SOLAR_SKIP_DATA_SECURITY_CHECK"] 
                skip_self_critique_check=config.system_config["SOLAR_SKIP_SELF_CRITIQUE_CHECK"]                
            else:
                logger.info("checking LLM file exist")                    
                # setup LLM model file if not exist
                local_llm_name_or_path = config.system_config["DEFAULT_LLM_MODEL_NAME_OR_PATH"]                
                local_llm_path = config.system_config["DEFAULT_LLM_MODEL_PATH"]
                local_llm_file = config.system_config["DEFAULT_LLM_MODEL_FILE"]
                local_llm_download = config.system_config["DEFAULT_LLM_MODEL_DOWNLOAD_PATH"]                
                isExist = os.path.exists(local_llm_path)
                if not isExist:
                    os.makedirs(local_llm_path)
                gguf_model_file, gguf_chat_format = llmproxy.chat_models_local(
                    model_name_or_path=local_llm_name_or_path,
                    model_file=local_llm_file,
                    model_download=local_llm_download,                    
                    model_path=local_llm_path)
                if gguf_model_file is not None and gguf_chat_format is not None:
                    table.add_row("resource_check", f"get LLM model file: {gguf_model_file}", "[green]Found[/]")
                    logger.info(f"4.Found LLM file {gguf_model_file}")                           
                else:
                    table.add_row("resource_check", f"get LLM model file: {gguf_model_file}", "[red]Missing[/]")        
                    logger.error(f"Missing LLM file {gguf_model_file}")                                       
            
            progress.advance(task)

            # 5. speech-to-text model file (Whisper)        
            logger.info("checking speech-to-text (whisper) model file exist")     
            model_file= get_or_download_whisper_model_snapshot()
            if model_file:
                table.add_row("resource_check", f"get speech-to-text model file: {model_file}", "[green]Found[/]")
                logger.info(f"5.Found speech-to-text model file {model_file}")                           
            else:
                table.add_row("resource_check", f"get speech-to-text model file", "[red]Missing[/]")
                logger.error(f"Missing speech-to-text model file!!!")                                                                                 
            progress.advance(task)

            # 6. grammar_model
            logger.info("checking grammar synthesis model file exist")     
            model_file= get_or_download_grammar_model_snapshot()
            if model_file:
                table.add_row("resource_check", f"get grammar synthesis model file: {model_file}", "[green]Found[/]")
                logger.info(f"6.Found grammar synthesis model file {model_file}")                           
            else:
                table.add_row("resource_check", f"get grammar synthesis model file", "[red]Missing[/]")
                logger.error(f"Missing grammar synthesis model file!!!")                                                                                 
            progress.advance(task)

            # 7. styletts2 model files
            logger.info("download styletts2 model files")                                                          
            get_styletts2_model_files()
            progress.advance(task)              

        except Exception as e:
            print(e)
            logger.error("system_resources_check error.")
            # # 7. voice activity detection_model
            # logger.info("checking VAD model file exist")                 
            # model,utils=init_vad_model() 
            # #console.print("checking silero-vad ", end="")        
            # if (model is not None) and os.path.exists(DEFAULT_SYSTEM_VAD_MODEL_PATH):
            #     table.add_row("resource_check", f"get VAD model file path {DEFAULT_SYSTEM_VAD_MODEL_PATH}", "[green]Found[/]")
            #     logger.info(f"7.Found VAD model file {DEFAULT_SYSTEM_VAD_MODEL_PATH}")                           
            # else:
            #     table.add_row("resource_check", f"get VAD model file", "[red]Missing[/]")
            #     logger.error(f"Missing VAD model file!!!")                                                                                 
            # progress.advance(task)  
            current_system_mode="oops, system_resources_check error. please check the log "
            system_tts_local(text=current_system_mode,output_voice=output_voice)

def system_sanity_tests(output_voice:str="jane"):
    global system_is_ready    
    logger.info("***running system functional checks***")    
    if "DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_SUCCESS" not in config.system_config.keys():
        startup_message = DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_SUCCESS    
    else:
        startup_message = config.system_config["DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_SUCCESS"]    
        
    with Progress(transient=True) as progress:        
        task = progress.add_task("running system sanity test...", total=4)
        try:        
           # 1.text-to-speech model
            logger.info("[test#1] text-to-speech model")                    
            current_time_message=wakeup_time()   
            system_tts_local(text=current_time_message,output_voice=output_voice)

            current_system_mode="running in system mode: "+_GLOBAL_SYSTEM_MODE
            system_tts_local(text=current_system_mode,output_voice=output_voice)

            if "DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_RUNNING" not in config.system_config.keys():
                startup_message = DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_RUNNING
            else:
                startup_message = config.system_config["DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_RUNNING"]
            system_tts_local(text=startup_message,output_voice=output_voice)
            logger.info(f"[test#1] text-to-speech model: Passed")                               
            table.add_row("functional_check", f"sanity test:text-to-speech model", "[green]Passed[/]")        
            progress.advance(task)        

            if "DEFAULT_SYSTEM_MODE" not in config.system_config.keys():
                system_mode = DEFAULT_SYSTEM_MODE
            else:
                system_mode = config.system_config["DEFAULT_SYSTEM_MODE"]
            logger.info(f"system mode:{system_mode}")

            # 2.transcription model
            logger.info("[test#2] speech-to-text transcribe")     
            test_audio_file=config.system_config["DEFAULT_SYSTEM_AUDIO_TEST_FILE"]         
            speech_to_text(input_audio=test_audio_file) 

            logger.info(f"[test#2] Whisper model {DEFAULT_WHISPER_MODEL_SIZE}: Passed")                        
            table.add_row("functional_check", f"sanity test:speech-to-text model", "[green]Passed[/]")           
            progress.advance(task)        

            # 3.llm model
            logger.info("[test#3] LLM model")   
            if _GLOBAL_SYSTEM_MODE=="solar-openai" or _GLOBAL_SYSTEM_MODE=="ollama-openai":
                current_message="testing Solar LLM"                
                #system_tts_local(text=current_message,output_voice=output_voice)
                if _GLOBAL_SYSTEM_MODE=="ollama-openai":
                    default_url=str(config.system_config["SOLAR_LLM_DEFAULT_SERVER_URL"]).strip() 
                    default_api_key=str(config.system_config["SOLAR_LLM_DEFAULT_API_KEY"]).strip()            
                    default_api_key=str(config.system_config["SOLAR_LLM_DEFAULT_MODEL_ID"]).strip()                                
                else:
                    default_url=str(config.system_config["SOLAR_LLM_OLLAMA_SERVER_URL"]).strip() 
                    default_api_key=str(config.system_config["SOLAR_LLM_OLLAMA_API_KEY"]).strip()            
                    default_api_key=str(config.system_config["SOLAR_LLM_OLLAMA_MODEL_ID"]).strip()                                

                # skip_content_safety_check=system_config["SOLAR_SKIP_CONTENT_SAFETY_CHECK"]    
                #skip_data_security_check=config.system_config["SOLAR_SKIP_DATA_SECURITY_CHECK"] 
                #skip_self_critique_check=config.system_config["SOLAR_SKIP_SELF_CRITIQUE_CHECK"] 
                   
                reply_text, reply_messages = llmproxy.chat_api_remote(user_prompt="hello")
                #history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, 
                #                                                      query=user_query, history=[])
                logger.info(f"Solar Response:\n{reply_text}")
                logger.info(f"[test#3] LLM Model: {reply_text} status: OK")                
                table.add_row("functional_check", f"sanity test:LLM model", "[green]Passed[/]")                           
            else:
                current_message="testing Local LLM"                
                #system_tts_local(text=current_message,output_voice=output_voice)                
                llmproxy.create_llm_local()
                messages, xhistory, reply = llmproxy.chat_api_local("hello", history=[])
                logger.info(f"[test#3] LLM Model: {reply} status: OK")
                table.add_row("functional_check", f"sanity test:LLM model", "[green]Passed[/]")           
            progress.advance(task)        

            # 4.grammar model
            logger.info("[test#4] grammar model")        
            init_grammar_correction_model()
            raw_text = "Iwen 2the store yesturday to bye some food. I needd milk, bread, andafew otter things. The $$tore was reely crowed and I had a hard time finding everyting I needed. I finaly madeit t0 dacheck 0ut line and payed for my stuff."
            fix_grammar_error(raw_text)
            logger.info(f"[test#4] grammar model {DEFAULT_GRAMMAR_MODEL_SIZE}: Passed")                   
            table.add_row("functional_check", f"sanity test:grammar model", "[green]Passed[/]")    
            progress.advance(task)        
            logger.info("------------------------")
            logger.info(startup_message)
            logger.info("------------------------")        
        except Exception as e:
            system_is_ready = False
            print(e.args)
            print(traceback.format_exc())
            logger.error("system funtional check - error")        
            if "DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED" not in config.system_config.keys():
                startup_message = DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED
            else:
                startup_message = config.system_config["DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED"]
            if "DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED_RECOVER" not in config.system_config.keys():
                failed_message = DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED_RECOVER
            else:
                failed_message = config.system_config["DEFAULT_SYSTEM_STARTUP_MSG_SYSTEM_CHECK_FAILED_RECOVER"]
            startup_message = startup_message+failed_message
    return startup_message

def activate_system_agent(system_agent:str=None, startup_message:str=None, system_mode:str=None,output_voice:str="en"):
    logger.info(f"activate_system_agent: {system_agent}")         
    DEFAULT_PAVAI_STARTUP_MSG_INTROS = [
        DEFAULT_PAVAI_STARTUP_MSG_INTRO,
        DEFAULT_PAVAI_STARTUP_MSG_INTRO2,
        DEFAULT_PAVAI_STARTUP_MSG_INTRO3,
        DEFAULT_PAVAI_STARTUP_MSG_INTRO4,
        DEFAULT_PAVAI_STARTUP_MSG_INTRO5
    ]
    if system_agent==PAVAI_APP_VOCIE:    
        if "DEFAULT_PAVAI_VOCIE_AGENT" not in config.system_config.keys():
            agen_name = DEFAULT_PAVAI_VOCIE_AGENT
        else:
            agen_name = config.system_config["DEFAULT_PAVAI_VOCIE_AGENT"]
        agen_greeting = f"hi, i am {agen_name} from PAVAI a galaxy far far away."
        system_tts_local(text=agen_greeting,output_voice=output_voice)
        #time.sleep(0.5)              
        #rand_idx = random.randrange(len(DEFAULT_PAVAI_STARTUP_MSG_INTROS))
        #intro_message = DEFAULT_PAVAI_STARTUP_MSG_INTROS[rand_idx]                    
        #time.sleep(0.5)              
        #system_tts_local(sd,text=startup_message,output_voice=output_voice)
        #time.sleep(0.25)       
        #system_tts_local(sd,text=intro_message,output_voice=output_voice)
        if "DEFAULT_SYSTEM_STARTUP_MSG_OPEN_BROWSER" not in config.system_config.keys():
            launch_message = DEFAULT_PAVAI_VOCIE_STARTUP_MSG_NEXT_STEP
        else:
            launch_message = config.system_config["DEFAULT_SYSTEM_STARTUP_MSG_OPEN_BROWSER"]
        #time.sleep(0.25)       
        system_tts_local(text=launch_message,output_voice=output_voice)    
    elif system_agent==PAVAI_APP_TALKIE:
        if "DEFAULT_PAVAI_VOCIE_AGENT" not in config.system_config.keys():
            agen_name = DEFAULT_PAVAI_TALKIE_AGENT
        else:
            agen_name = config.system_config["DEFAULT_PAVAI_TALKIE_AGENT"]
        agen_greeting = f"hi, i am {agen_name} from PAVAI a galaxy far far away."
        system_tts_local(text=agen_greeting,output_voice=output_voice)
        #time.sleep(0.25)                      
        if "DEFAULT_PAVAI_STARTUP_MSG_INTRO" not in config.system_config.keys():
            intro_message = DEFAULT_PAVAI_STARTUP_MSG_INTRO
        else:
            intro_message = config.system_config["DEFAULT_PAVAI_STARTUP_MSG_INTRO"]        
        ##system_tts_local(text=startup_message,output_voice=output_voice)
        #time.sleep(0.25)       
        rand_idx = random.randrange(len(DEFAULT_PAVAI_STARTUP_MSG_INTROS))
        intro_message = DEFAULT_PAVAI_STARTUP_MSG_INTROS[rand_idx]                    
        ##intro_message=".your personal multilingual AI assistant for everyday tasks." 
        ##system_tts_local(text=intro_message,output_voice=output_voice)        
        if "DEFAULT_SYSTEM_STARTUP_MSG_OPEN_BROWSER" not in config.system_config.keys():
            launch_message = DEFAULT_PAVAI_TALKIE_STARTUP_MSG_NEXT_STEP
        else:
            launch_message = config.system_config["DEFAULT_SYSTEM_STARTUP_MSG_OPEN_BROWSER"]
        ##system_tts_local(text=launch_message,output_voice=output_voice)            

    logger.info(f"system funtional check - completed")         

def pavai_vocie_system_health_check(output_voice:str="en_amy"):
    System_ready = False    
    try:    
        environment_info()
        system_resources_check()
        startup_message=system_sanity_tests()
        activate_system_agent(system_agent=PAVAI_APP_VOCIE,startup_message=startup_message)    
        logger.info("pavai_vocie_system_health_check results")                                
        Console().print(table)
    except Exception as e:
        logger.error("pavai_talkie_system_health_check error!",e)
    return System_ready                     

def pavai_talkie_system_health_check(output_voice:str="jane"):
    System_ready = False
    try:
        environment_info()
        set_default_system_voice(system_agent=PAVAI_APP_TALKIE)    
        system_resources_check()
        startup_message=system_sanity_tests(output_voice=output_voice)
        activate_system_agent(system_agent=PAVAI_APP_TALKIE,startup_message=startup_message,output_voice=output_voice)    
        logger.info("pavai_talkie_system_health_check results")                                
        Console().print(table)
        Console().print(":white_check_mark: Ready? I am listening. how may I help you today?")
        System_ready = True
    except Exception as e:
        logger.error("pavai_talkie_system_health_check error!",e)
    return System_ready                     

"""MAIN"""
if __name__ == "__main__":
    #pavai_vocie_system_health_check(output_voice="en_amy")
    pavai_talkie_system_health_check(output_voice="jane")
    #intro_message="hi, I am Ryan, your personal multilingual AI assistant for everyday tasks, how may I help you today?" 
    #speak_instruction(intro_message,output_voice="en_ryan")