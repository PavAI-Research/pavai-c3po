# pip install python-dotenv
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
from rich.logging import RichHandler
from rich import pretty
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()
import warnings 
warnings.filterwarnings("ignore")

import os, sys
import gradio as gr
import torch
import pandas as pd
import numpy as np
import pavai.shared.datasecurity as datasecurity
from typing import BinaryIO, Union
from transformers.utils import is_flash_attn_2_available
from pavai.shared.audio.voices_piper import (text_to_speech, speak_acknowledge,speak_wait, speak_done, speak_instruction)
from pavai.shared.system_checks import (pavai_vocie_system_health_check, DEFAULT_SYSTEM_MODE, SYSTEM_THEME_SOFT,SYSTEM_THEME_GLASS,VOICE_PROMPT_INSTRUCTIONS_TEXT)
from pavai.shared.audio.transcribe import (speech_to_text, FasterTranscriber,DEFAULT_WHISPER_MODEL_SIZE)
from pavai.shared.llmproxy import chatbot_ui_client,chat_count_tokens,multimodal_ui_client
from pavai.shared.image.text2image import (StableDiffusionXL, image_generation_client,DEFAULT_TEXT_TO_IMAGE_MODEL)
from pavai.shared.fileutil import get_text_file_content
from pavai.shared.commands import filter_commmand_keywords
from pavai.shared.grammar import (fix_grammar_error)
from pavai.shared.audio.tts_client import get_speaker_audio_file

## remove Iterable, List, NamedTuple, Optional, Tuple,
## removed from os.path import dirname, join, abspath
from pavai.shared.aio.llmchat import (system_prompt_assistant, DEFAULT_LLM_CONTEXT_SIZE)
from pavai.shared.llmcatalog import LLM_MODEL_KX_CATALOG_TEXT
from pavai.shared.solar.llmprompt import knowledge_experts_system_prompts
from pavai.shared.aio.llmchat import get_llm_library


__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2024"
__version__ = "0.0.3"

print("--GLOBAL SYSTEM MODE----")
print(system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=system_config["GLOBAL_TTS"]

# tested version gradio version: 4.7.1
# pip install gradio==4.7.1
# whisper model
DEFAULT_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
DEFAULT_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
INT16_MAX_ABS_VALUE = 32768.0
DEFAULT_MAX_MIC_RECORD_LENGTH_IN_SECONDS = 30*60*60  # 30 miniutes
DEFAULT_WORKING_DIR = "./workspace"

# Global variables
system_is_ready = True
use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()

faster_transcriber = FasterTranscriber(model_id_or_path=DEFAULT_WHISPER_MODEL_SIZE)
stablediffusion_model = StableDiffusionXL(model_id_or_path=DEFAULT_TEXT_TO_IMAGE_MODEL)

# Knowledge experts and Domain Models
knowledge_experts = list(knowledge_experts_system_prompts.keys())
domain_models = list(get_llm_library().keys())
# global settings
_QUERY_ASK_EXPERT_ID=None  # planner
_QUERY_ASK_EXPERT_PROMPT=None  # default
_QUERY_LLM_MODEL_ID=None #"zephyr-7b-beta.Q4_K_M.gguf"
_QUERY_LLM_MODEL_INFO=None # list of [id,id,ful_url] 
# data security check - PII
_QUERY_ENABLE_PII_ANALYSIS=False 
_QUERY_ENABLE_PII_ANONYMIZATION=False
# content safety check
_QUERY_CONTENT_SAFETY_GUARD=False

# turn-off chatbot voice response
_TURN_OFF_CHATBOT_RESPONSE_VOICE=False

# uploaded image file
_USER_UPLOADED_IMAGE_FILE=None
_USER_UPLOADED_IMAGE_DEFAULT_QUERY="describe what does the image say?"
_USER_UPLOADED_IMAGE_FILE_DESCRIPTION=None
_TURN_OFF_IMAGE_CHAT_MODE=True

def turn_off_chatbot_voice_options(x):
    global _TURN_OFF_CHATBOT_RESPONSE_VOICE
    logger.warn(f"change turn_off_voice_response_options: {x}")
    _TURN_OFF_CHATBOT_RESPONSE_VOICE = x

def turn_off_image_chat_options(x):
    global _TURN_OFF_IMAGE_CHAT_MODE
    logger.warn(f"change turn_off_image_chat_options: {x}")
    _USER_UPLOADED_IMAGE_FILE = None
    _TURN_OFF_IMAGE_CHAT_MODE = x

def select_content_safety_options(x):
    global _QUERY_CONTENT_SAFETY_GUARD
    logger.warn(f"change content_safety_options: {x}")
    _QUERY_CONTENT_SAFETY_GUARD = x

def pii_data_analysis_options(x):
    global _QUERY_ENABLE_PII_ANALYSIS
    logger.warn(f"change pii_data_analysis_options: {x}")
    _QUERY_ENABLE_PII_ANALYSIS = x

def pii_data_amomymization_options(x):
    global _QUERY_ENABLE_PII_ANONYMIZATION    
    logger.warn(f"change pii_data_amomymization_options: {x}")
    _QUERY_ENABLE_PII_ANONYMIZATION=x    

def select_expert_option(x):
    global _QUERY_ASK_EXPERT_PROMPT
    global _QUERY_ASK_EXPERT_ID
    logger.warn(f"change knowledge expertise: {x}")
    expert_system_prompt = knowledge_experts_system_prompts[x]
    logger.debug("system prompts", expert_system_prompt)
    _QUERY_ASK_EXPERT_PROMPT = expert_system_prompt
    _QUERY_ASK_EXPERT_ID=x

def select_model_option(x):
    global _QUERY_LLM_MODEL_INFO    
    global _QUERY_LLM_MODEL_ID    
    logger.warn(f"change domain model: {x}")
    domain_model = get_llm_library()[x]
    logger.debug(f"system prompts {domain_model}")    
    _QUERY_LLM_MODEL_INFO=domain_model    
    _QUERY_LLM_MODEL_ID=x

def vc_text_to_image(user_prompt: str,
                     neg_prompt: str = "ugly, blurry, poor quality",
                     output_filename="new_text_to_image_1.png",
                     storage_path: str = "workspace/text-to-image") -> str:
    if user_prompt is None or len(user_prompt) == 0:
        gr.Warning("empty prompt text!")
        return None
    # load on-demand to reduce memory usage at cost of lower performance
    output_filename = image_generation_client(abstract_image_generator=stablediffusion_model,
                                              user_prompt=user_prompt, neg_prompt=neg_prompt,
                                              output_filename=output_filename, storage_path=storage_path)
    return output_filename


def vc_speech_to_text(input_audio: Union[str, BinaryIO, np.ndarray],
                      task_mode="transcribe",
                      model_size: str = "large",
                      beam_size: int = 5,
                      vad_filter: bool = True,
                      language: str = None,
                      include_timestamp_seg=False) -> str:
    transcription, language = speech_to_text(input_audio=input_audio, task_mode=task_mode, model_size=model_size,
                                             beam_size=beam_size, vad_filter=vad_filter, language=language,
                                             include_timestamp_seg=include_timestamp_seg)
    return transcription, language


def vc_unhide_outputs(enable_text_to_image: bool = False):
    if enable_text_to_image:
        return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=True), gr.Button(visible=True), gr.Button(visible=True)
    else:
        return gr.Textbox(visible=True), gr.Audio(visible=True), gr.Image(visible=False), gr.Button(visible=False), gr.Button(visible=False)


def vc_hide_outputs(enable_text_to_image: bool = False):
    if enable_text_to_image:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=False)

def vc_text_to_speech(text: str, output_voice: str = "en", mute=False, autoplay=False):
    out_file = text_to_speech(
        text=text, output_voice=output_voice, mute=mute, autoplay=autoplay)
    return out_file

def vc_fix_grammar_error(raw_text: str,
                         enabled: bool = False,
                         max_length: int = 128,
                         repetition_penalty: float = 1.05,
                         num_beams: int = 4):
    corrected_text, raw_text = fix_grammar_error(raw_text=raw_text, enabled=enabled, max_length=max_length,
                                                 repetition_penalty=repetition_penalty, num_beams=num_beams)
    return corrected_text, raw_text

def vc_set_task_mode(choice, task_mode_state):
    if choice == "translate":
        task_mode_state = choice
    else:
        task_mode_state = "transcribe"
    return task_mode_state

def cb_clear_input_text(text):
    return gr.update(value="", interactive=True)

def _mute_voice_speaker_for_these_commands_output(task_command: str) -> bool:
    logger.debug(f"skip_command_output: {task_command}")
    transcribe_youtube_command = "/disable_transcribe_youtube:"
    translate_youtube_command = "/disable_translate_youtube:"
    load_text_file_command = "/disable_load_text:"
    # speak_wait()
    if transcribe_youtube_command in task_command \
        or translate_youtube_command in task_command \
            or load_text_file_command in task_command:
        speak_instruction("process command: "+task_command)
        return True
    else:
        return False

def chat_client(input_text:str, chat_history:list):
    #_GLOBAL
    global _QUERY_ASK_EXPERT_ID
    global _QUERY_ASK_EXPERT_PROMPT
    global _QUERY_LLM_MODEL_ID
    global _QUERY_LLM_MODEL_INFO
    global _QUERY_ENABLE_PII_ANALYSIS
    global _QUERY_ENABLE_PII_ANONYMIZATION
    # content handling flags
    user_override=False
    skip_content_safety_check=True
    skip_data_security_check=True            
    skip_self_critique_check=True
    gr.Info("working on it...")    
    if _QUERY_ENABLE_PII_ANALYSIS or _QUERY_ENABLE_PII_ANONYMIZATION:                
        skip_data_security_check=False 
        user_override=True  
    if _QUERY_CONTENT_SAFETY_GUARD:
        skip_content_safety_check=False
        user_override=True               
    ## data security check on input
    if _QUERY_ENABLE_PII_ANALYSIS:
        pii_results = datasecurity.analyze_text(input_text=input_text)
        logger.info(pii_results,extra=dict(markup=True))  
        gr.Warning('PII detected in INPUT text.')        
    if _QUERY_ASK_EXPERT_PROMPT is None:
        chatbot_ui_messages, chat_history, reply_text = chatbot_ui_client(input_text=input_text, 
                                                                      chat_history=chat_history,
                                                                      ask_expert=_QUERY_ASK_EXPERT_ID,
                                                                      target_model_info=_QUERY_LLM_MODEL_INFO,
                                                                      user_override=user_override,
                                                                      skip_content_safety_check=skip_content_safety_check,
                                                                      skip_data_security_check=skip_data_security_check,
                                                                      skip_self_critique_check=skip_self_critique_check)
    else:
        gr.Info("routing request to domain experts")    
        chatbot_ui_messages, chat_history, reply_text = chatbot_ui_client(input_text=input_text, 
                                                                      chat_history=chat_history,
                                                                      system_prompt=_QUERY_ASK_EXPERT_PROMPT,
                                                                      ask_expert=_QUERY_ASK_EXPERT_ID,
                                                                      target_model_info=_QUERY_LLM_MODEL_INFO,
                                                                      user_override=user_override,
                                                                      skip_content_safety_check=skip_content_safety_check,
                                                                      skip_data_security_check=skip_data_security_check,
                                                                      skip_self_critique_check=skip_self_critique_check)                        
    ## data security check on output
    pii_results=None
    if _QUERY_ENABLE_PII_ANALYSIS:
        pii_results = datasecurity.analyze_text(input_text=reply_text)
        logger.warn(pii_results,extra=dict(markup=True))
        gr.Warning('PII detected in OUTPUT text.')                           

    if _QUERY_ENABLE_PII_ANONYMIZATION:
        if pii_results:
            reply_text = datasecurity.anonymize_text(input_text=reply_text,analyzer_results=pii_results)                
        else:
            reply_text = datasecurity.anonymize_text(input_text=reply_text)
        logger.info(pii_results,extra=dict(markup=True))
        gr.Warning('OUTPUT text. anonymized')

    return chatbot_ui_messages, chat_history, reply_text

def cb_botchat(bot_history, chat_history=[], task_command="fn_botchat"):
    global _USER_UPLOADED_IMAGE_FILE
    logger.info("cb_botchat")
    if bot_history is None or len(bot_history) == 0:
        return [], [], ""
    input_text = bot_history[-1][0]
    logger.info(f"cb_botchat: {input_text} task_command:{task_command}")
    if _mute_voice_speaker_for_these_commands_output(task_command):
        return bot_history, chat_history, input_text
    if input_text is not None and len(input_text) > 0:        
        ## image chat
        if _USER_UPLOADED_IMAGE_FILE is not None:
            reply_messages, chat_history,reply_text = imagechat(image_file_path=_USER_UPLOADED_IMAGE_FILE,
                                                    user_query=input_text,
                                                    history=chat_history)
            bot_messages=reply_messages  
            #logger.info(f"cb_botchat->image_chat_history:{bot_messages[-1][1]}")            
            return bot_messages, chat_history, reply_text
        else:    
            bot_messages, chat_history, reply_text = chat_client(input_text, chat_history)
            ## respond        
            if bot_messages is not None and len(bot_messages) > 0:
                logger.info(f"cb_botchat->bot_history:{bot_messages[-1][1]}")
                return bot_messages, chat_history, reply_text
            else:
                logger.info(f"cb_botchat-> reply error: {reply_text}")
                _USER_UPLOADED_IMAGE_FILE = None
    
    return bot_history, chat_history, ""

def cb_user_chat(user_message, chatbot_history, task_command):
    if user_message is None or len(user_message) == 0:
        gr.Warning("Empty input text! enter text then try again.")
        return
    if _mute_voice_speaker_for_these_commands_output(task_command):
        return "", chatbot_history
    chatbot_history = chatbot_history + [[user_message, None]]
    return "", chatbot_history

def vc_ask_assistant_to_image(enable_image_generation: bool, input_text: str, bot_history: list, chat_history: list):
    logger.info(f"ask_assistant_to_image: {input_text}")
    new_image_file = None
    reply = ""
    if input_text is not None and len(input_text) > 0:
        if enable_image_generation:
            new_image_file = vc_text_to_image(input_text)
            return bot_history, chat_history, "AI image generated.", new_image_file
        else:
            bot_history, chat_history, reply = chat_client(input_text, chat_history)
            return bot_history, chat_history, reply, new_image_file
    return bot_history, chat_history, reply, new_image_file

def vc_voice_chat(user_message, chatbot_history):
    chatbot_history = chatbot_history + [[user_message, None]]
    return user_message, chatbot_history

def vc_add_query_log_entry(logdf: pd.DataFrame, query_text: str, response_text: str, duration: int = 0):
    if query_text is None or len(query_text) == 0:
        return
    idx = 1 if len(logdf) == 0 else len(logdf)+1
    new_row = {'index': idx,
               'query_text': query_text, 'query_length': len(query_text),
               'response_text': response_text, 'response_length': len(response_text),
               "took": duration}
    logdf.loc[len(logdf)] = new_row
    return logdf

def xrun_task(task_name: str, task_input: str, task_output: str, output_voice: str = "en", autoplay=False):
    logger.info(
        f"run_task: {task_name} with input: {task_input} and output: {task_output}")
    if "text_to_speech" in task_name:
        out_file = get_speaker_audio_file#'espeak_text_to_speech.mp3'
        out_file = text_to_speech(task_input, output_voice)
        return out_file
    else:
        return None

def cb_text_to_speech(text: str, output_voice: str = "en", mute=False, autoplay=False):
    out_file=None
    if not _TURN_OFF_CHATBOT_RESPONSE_VOICE:
        out_file = text_to_speech(text=text, output_voice=output_voice, mute=mute, autoplay=autoplay)
    return out_file

def cb_hide_outputs(enable_text_to_image: bool = False):
    if enable_text_to_image:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=False)

def cb_filter_commmand_keywords(input_text, history):
    global _USER_UPLOADED_IMAGE_FILE
    global _USER_UPLOADED_IMAGE_DEFAULT_QUERY
    output_text, history, output_filename, task_command, language = filter_commmand_keywords(
        input_text, history)
    ### redirect image chat command to multimodal chat
    _image_chat_command = "/imagechat?"
    if task_command==_image_chat_command:
        result_text=output_text.split("???")
        _USER_UPLOADED_IMAGE_FILE = result_text[1]
        _USER_UPLOADED_IMAGE_DEFAULT_QUERY = result_text[0].replace('%20',' ')
        output_text=_USER_UPLOADED_IMAGE_DEFAULT_QUERY

    return output_text, history, output_filename, task_command, language

def cb_add_file(history, file):
    filename = None
    if file is not None:
        if isinstance(file, str):
            filename = file
        else:
            filename = file.name
        history = history + [((filename), None)]
        logger.info(f"add_file: {file} | {history}")
        return history, filename
    else:
        return history, None

def imagechat(image_file_path:str=None, user_query:str=None, history:list=[]):
    # reject conditions
    if image_file_path is None:
        return history, "missing image file path" , "image chat"
    if user_query is not None:
        if image_file_path == ("file:///".join(user_query)):
            user_query = _USER_UPLOADED_IMAGE_DEFAULT_QUERY
        if user_query=="file upload success":
            user_query = _USER_UPLOADED_IMAGE_DEFAULT_QUERY            
    else:
        user_query = _USER_UPLOADED_IMAGE_DEFAULT_QUERY        
    # process request    :file upload success
    gr.Info(f"imagechat with file {image_file_path}")        
    if image_file_path.lower().startswith("http://") or image_file_path.lower().startswith("https://"):
        reply_messages, history, reply_text = multimodal_ui_client(input_text=image_file_path,
                                                    user_query=user_query,
                                                    chat_history=history)
        return reply_messages, history, reply_text        
    elif image_file_path.lower().endswith(".png") or image_file_path.lower().endswith(".jpg") or image_file_path.lower().endswith(".jpeg"):
        reply_messages, history, reply_text = multimodal_ui_client(input_text=image_file_path,
                                                     user_query=user_query,
                                                     chat_history=history)
        return reply_messages, history, reply_text
    else:
        return [], history, "imagechat url is not valid. the link should end with .png or .jpeg."
    

def cb_process_upload_image(text: str, history:list, output_voice: str = "en", mute=False,
                            autoplay=False, prefix_text="multimodal image file:\n"):
    global _USER_UPLOADED_IMAGE_FILE
    logger.info(f"process_upload_image_file: {text}")
    gr.Info("processing upload image file.")        
    if mute or text is None or len(text.strip()) == 0:
        return None
    output_voice = "en" if output_voice is None else output_voice
    history = [] if history is None else history
    if text.lower().endswith(".png") or text.lower().endswith(".jpg") or text.lower().endswith(".jpeg"):
        gr.Info(f"working on upload file {text}")
        _USER_UPLOADED_IMAGE_FILE = "file:///"+text
        file_contents = "file upload success"
        history = history + [[file_contents, None]]
        return history, text, "process_upload_image"        
    else:
        return history, text, "process_upload_image"

def cb_process_upload_audio(text: str, history, output_voice: str = "en", mute=False,
                            autoplay=False, prefix_text="transcribed audio file:\n"):
    logger.info(f"process_upload_file_audio: {text}")
    gr.Info("processing upload audio file.")        
    if mute or text is None or len(text.strip()) == 0:
        return None
    output_voice = "en" if output_voice is None else output_voice
    history = [] if history is None else history
    if text.endswith(".mp3") or text.endswith(".wav"):
        file_path = text
        file_contents, detected_language = speech_to_text(input_audio=text, task_mode="transcribe")
        file_contents = prefix_text+file_contents
        history = history + [[file_contents, None]]
        return history, "processed upload file, any question?", "process_upload_audio"
    else:
        return history, text, "process_upload_audio"

def cb_process_upload_text(text: str, history, language: str = "en"):
    logger.info(f"process_upload_text: {text}")
    gr.Info("processing upload text file.")    
    if text is None or len(text.strip()) == 0:
        return None
    language = "en" if language is None else language
    history = [] if history is None else history
    text_file = text.lower()
    if text_file.endswith(".txt") or text_file.endswith(".md") \
            or text_file.endswith(".html") or text_file.endswith(".pdf"):
        file_contents, history, file_path = get_text_file_content(
            file_path=text_file, history=history, prefix_text="get file content:\n")
        return history, "processed upload file, any question?", "process_upload_text"
    else:
        return history, text, "process_upload_text"


def cb_update_chat_session(chat_history: list = [],
                           llm_max_context: int = DEFAULT_LLM_CONTEXT_SIZE,
                           system_prompt: str = system_prompt_assistant):
    """
    history.append({"role": "assistant", "content": reply})
    """
    gr.Info("update_chat_session...")
    # global llm
    if chat_history is None or len(chat_history) == 0:
        logger.debug("empty chat history")
        return [], chat_history, 123
    llm_max_context = llm_max_context-(llm_max_context*0.10)  # keep 10% free
    chat_tokens_size = 0
    for row in chat_history:
        if isinstance(row["content"],str):
            chat_tokens_size += chat_count_tokens(row["content"])
    logger.info(f"session tokens: {chat_tokens_size}")
    if chat_tokens_size > llm_max_context:
        # reduce exceed tokens
        gr.Warning("summarized conversation due tokens exceed model max context size")
        while chat_tokens_size > llm_max_context:
            chat_row = chat_history.pop(0)
            row_tokens = chat_count_tokens(chat_row["content"])
            chat_tokens_size = chat_tokens_size-row_tokens
        # summarize history to avoid hitting LLM max content
        summary_Prompt = "summarize current conversations history: " 
        messages, chat_history, reply = chat_client(input_text=summary_Prompt, chat_history=chat_history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": summary_Prompt},
            {"role": "assistant", "content": reply}
        ]
        chat_history = []
        chat_history = chat_history+messages
    messages = []
    for i in range(0, len(chat_history)-1):
        if chat_history[i]["role"] != "system":
            messages.append((chat_history[i]["content"], chat_history[i+1]["content"]))
    logger.debug(chat_history)
    logger.info(messages)
    return messages, chat_history, session_tokens


def cb_reset_chat_outputs():
    return gr.update(value="", visible=False, interactive=True), gr.update(value="", visible=False, interactive=True)

"""
MAIN User Interface
"""
title = '🤖 PavAI.Vocie - an advance private multilingual LLM-based AI Voice Assistant'
description = """an advance private multilingual AI Voice Assistant."""

theme = gr.themes.Default()
if "SYSTEM_THEME" in system_config.keys():
    system_theme = system_config["SYSTEM_THEME"]
    if system_theme == SYSTEM_THEME_SOFT:
        theme = gr.themes.Soft()
    elif system_theme == SYSTEM_THEME_GLASS:
        theme = gr.themes.Glass()

if "DEFAULT_SYSTEM_MODE" not in system_config.keys():
    system_mode = DEFAULT_SYSTEM_MODE
else:
    system_mode = system_config["DEFAULT_SYSTEM_MODE"]

"""UI-SYSTEM SETTINGS"""
with gr.Blocks(title="SYSTEM SETTINGS-UI", analytics_enabled=False) as llm_setting_ui:
    with gr.Group():        
        textbox_workspace = gr.Text(
            label="workspace", value="workspace",placeholder="data storage location",interactive=True)
        gr.Markdown("#### LLM Model and hyper-parameters")
        with gr.Row():
            llm_model_path = gr.Text(
                label="model_path", value="resources/models/llm", interactive=True)
            llm_model_file = gr.Text(
                label="model_file", value="zephyr-7b-beta.Q5_K_M.gguf", interactive=True)
        with gr.Row():
            llm_model_chat_format = gr.Text(
                label="model_chat_format", value="chatml", interactive=True)
            llm_offload_gpu_layers = gr.Textbox(
                value=35, label="offload gpu layers", interactive=True)

        gr.Markdown("#### Whisper Model")
        with gr.Row():
            stt_model_size = gr.Text(
                label="stt model size", value="large", interactive=True)
            stt_model_name = gr.Text(
                label="stt model name", value="Systran/faster-whisper-large-v3", interactive=True)

        gr.Markdown("#### Grammar synthesis model")
        with gr.Row():
            grammar_model_size = gr.Text(
                label="grammar model size", value="pszemraj/grammar-synthesis-small", interactive=True)

        gr.Markdown("#### AI Voice model")
        with gr.Row():
            voice_model_language = gr.Dropdown(
                choices=["en", "fr", "zh", "es"], value="en", label="Language", interactive=True)
            voice_model_gender = gr.Dropdown(
                choices=["Amy"], value="Amy", label="model gender", interactive=True)
        with gr.Row():
            voice_model_onnx_file = gr.Textbox(
                label="model onnx file", value="./models/voices/en_US-amy-medium.onnx", interactive=True)
            voice_model_json_file = gr.Textbox(
                label="model json file", value="./models/voices/en_US-amy-medium.onnx.json", interactive=True)

    def save_settings(settings):
        pass

    btn_save_settings = gr.Button(value="Save Changes")
    btn_save_settings.click(save_settings,
                            inputs=[textbox_workspace, llm_model_path,
                                    llm_model_file, llm_model_chat_format],
                            outputs=None, queue=False)

"""UI-VOICE-PROMPT"""
with gr.Blocks(title="VOICE-QUERY-UI", analytics_enabled=False) as llm_voice_query_ui:
    with gr.Group():    
        history_state = gr.State([])
        listening_status_state = gr.State(0)
        listening_sentence_state = gr.State([])
        # Session state box containing all user/system messages, hidden
        activities_state = gr.State([])
        task_mode_state = gr.State("transcribe")
        detected_lang_state = gr.State("en")
        session_tokens = gr.State(0)
        """Configuration"""
        with gr.Accordion("Click here for User Options", open=False):
            with gr.Row():
                with gr.Column(1):
                    radio_task_mode = gr.Radio(choices=["transcribe", "translate"], value="transcribe", interactive=True,
                                            label="transcription options",
                                            info="[transcribe] preserved spoken language while [translate] converts all spoken languages to english only.]")
                    radio_task_mode.change(fn=vc_set_task_mode, inputs=[
                                        radio_task_mode, task_mode_state], outputs=task_mode_state)
                with gr.Column(1):
                    checkbox_show_transcribed_text = gr.Checkbox(value=True,
                                                                label="show transcribed voice text")
                    checkbox_show_response_text = gr.Checkbox(value=True,
                                                            label="show response text")
                    checkbox_enable_grammar_fix = gr.Checkbox(value=False,
                                                            label="enable single-shot grammar synthesis correction model",
                                                            info="grammar synthesis model it does not semantically change text/information that is grammatically correct.")
                with gr.Column(1):
                    textbox_detected_spoken_language = gr.Textbox(
                        label="auto detect spoken language", value="en")
                    slider_max_query_log_entries = gr.Slider(
                        5, 30, step=1, label="Max logs", interactive=True)
                    checkbox_enable_text_to_image = gr.Checkbox(
                        value=False, label="enable image generation",
                        info="speak a shorter prompts to generate creative AI image.")
            with gr.Row():
                """content safety options"""                                
                checkbox_content_safety = gr.Checkbox(
                        value=False, label="enable content safety guard",
                        info="enforce content safety check on input and output")    
                checkbox_content_safety.change(fn=select_content_safety_options,inputs=checkbox_content_safety)                                                                           
                """data security options"""                
                checkbox_enable_PII_analysis = gr.Checkbox(
                    value=False, label="enable PII data analysis",
                    info="analyze and report PII data on query input and outputs")
                checkbox_enable_PII_analysis.change(fn=pii_data_analysis_options,
                                                    inputs=checkbox_enable_PII_analysis)
                checkbox_enable_PII_anonymization = gr.Checkbox(
                    value=False, label="enable PII data anonymization",
                    info="apply anonymization of PII data on input and output")    
                checkbox_enable_PII_anonymization.change(fn=pii_data_amomymization_options,
                                                    inputs=checkbox_enable_PII_anonymization)                                       
            with gr.Row():                
                """knowledge and domain model experts"""                
                expert_options = gr.Dropdown(knowledge_experts, label="Ask Knowledge Experts (AI)", info="[optional] route question to subject with domain knowledge.")
                expert_options.change(fn=select_expert_option,inputs=expert_options)        
                model_options = gr.Dropdown(domain_models,label="Custom LLM Model",info="[optional] use specialized model")
                model_options.change(fn=select_model_option,inputs=model_options)        

        """AUDIO INPUT/OUTPUTS"""
    # with gr.Group():
        default_max_microphone_recording_length_30_min = DEFAULT_MAX_MIC_RECORD_LENGTH_IN_SECONDS        
            # Audio Input Box
        with gr.Row():
            speaker_audio_input = gr.Audio(scale=2, sources=["microphone", "upload"], elem_id="user_speaker",
                                            type="filepath", label="press [record] to start speaking",
                                            show_download_button=True, show_share_button=True, visible=True,
                                            format="mp3", max_length=default_max_microphone_recording_length_30_min,
                                            waveform_options={"waveform_progress_color": "green", "waveform_progress_color": "green"})

            speaker_transcribed_input = gr.Textbox(label="transcribed query",
                                                    info="press [shipt]+[enter] key to re-submit transcribe text",
                                                    scale=2, lines=3,
                                                    show_copy_button=True, interactive=True,
                                                    container=True, visible=True)

            speaker_original_input = gr.Textbox(label="without grammar fix version",
                                                scale=2, lines=3, show_copy_button=True, interactive=True,
                                                container=True, visible=False)

            # handle change configuration option checkbox
            def vc_show_or_hide_input_textbox(checked):
                return gr.Textbox(visible=checked)
            checkbox_enable_grammar_fix.change(
                vc_show_or_hide_input_textbox, checkbox_enable_grammar_fix, speaker_original_input, queue=False)

            # clear transcribed textbox
            speaker_audio_input.clear(lambda: gr.update(
                value=""), None, speaker_transcribed_input)
            # handle audio recording as input
            speaker_input_mic = speaker_audio_input.stop_recording(
                speak_wait, None, None, queue=False
            ).then(
                vc_speech_to_text,
                inputs=[speaker_audio_input, task_mode_state],
                outputs=[speaker_transcribed_input,
                            textbox_detected_spoken_language],
                queue=False
            ).then(
                vc_fix_grammar_error,
                inputs=[speaker_transcribed_input,
                        checkbox_enable_grammar_fix],
                outputs=[speaker_transcribed_input,
                            speaker_original_input], queue=False
            )

            # handle audio file upload as input
            speaker_input_file = speaker_audio_input.upload(
                speak_wait, None, None, queue=False
            ).then(
                vc_speech_to_text,
                inputs=[speaker_audio_input, task_mode_state],
                outputs=[speaker_transcribed_input,
                            textbox_detected_spoken_language],
                queue=False
            ).then(
                vc_fix_grammar_error,
                inputs=[speaker_transcribed_input,
                        checkbox_enable_grammar_fix],
                outputs=[speaker_transcribed_input,
                            speaker_original_input], queue=False
            )
        with gr.Row():
            bot_audio_output = gr.Audio(scale=1, label="Assistant response audio",
                                        format="mp3", elem_id="ai_speaker", autoplay=True, visible=False,
                                        waveform_options={"waveform_progress_color": "orange", "waveform_progress_color": "orange"})
            bot_output_txt = gr.Textbox(
                label="Assistant response text", scale=2, lines=3,
                info="AI generated response text",
                placeholder="", container=True, visible=False, show_copy_button=True
            )
        with gr.Row():
            bot_output_image = gr.Image(
                label="generated image", type="filepath", interactive=False, visible=False)
        with gr.Row():
            # --summarize response
            btn_generate_request_image = gr.Button(
                value="Generate transcribed text image", size="sm", visible=False)
            btn_generate_request_image.click(fn=vc_text_to_image, inputs=[speaker_transcribed_input],
                                                outputs=[bot_output_image], queue=False)

            btn_generate_response_image = gr.Button(
                value="Generate response text image", size="sm", visible=False)
            btn_generate_response_image.click(fn=vc_text_to_image, inputs=[bot_output_txt],
                                                outputs=[bot_output_image], queue=False)
            bot_output_image.change(vc_unhide_outputs, inputs=[checkbox_enable_text_to_image], outputs=[
                                    bot_output_txt, bot_audio_output, bot_output_image], queue=False)

            btn_clear = gr.ClearButton([speaker_audio_input, bot_audio_output, bot_output_image,
                                        speaker_transcribed_input, bot_output_txt, activities_state],
                                        value="🎙️ Next Query")
            btn_clear.click(
                vc_hide_outputs, inputs=[checkbox_enable_text_to_image],
                outputs=[bot_output_txt, bot_audio_output, bot_output_image], queue=False)

        with gr.Accordion("Click here for Instructions", open=False):
            gr.Markdown(VOICE_PROMPT_INSTRUCTIONS_TEXT)

        """QUERY LOGS"""
        with gr.Accordion("Click here for Query Logs:", open=False):
            # Creating dataframe to log queries
            df_query_log = pd.DataFrame(
                {"index": [], "query_text": [], "query_length": [],
                    "response_text": [], "response_length": [], "took": []}
            )
            styler = df_query_log.style.highlight_between(color='lightgreen', axis=0)
            gr_query_log = gr.Dataframe(value=styler, interactive=True, wrap=True)
            max_values = 10
            def vc_update_log_entries(max_values):
                return gr.update(interactive=True, row_count=max_values)
            slider_max_query_log_entries.change(vc_update_log_entries,
                                                inputs=[
                                                    slider_max_query_log_entries],
                                                outputs=[gr_query_log], queue=False)

            # handle speaker input mic
        speaker_input_mic.then(
            speak_acknowledge, None, None, queue=False
        ).then(
            vc_voice_chat,
            inputs=[speaker_transcribed_input, activities_state],
            outputs=[speaker_transcribed_input, activities_state], queue=False
        ).then(
            vc_unhide_outputs,
            inputs=[checkbox_enable_text_to_image],
            outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                    btn_generate_request_image, btn_generate_response_image], queue=False
        ).then(
            vc_ask_assistant_to_image,
            inputs=[checkbox_enable_text_to_image,
                    speaker_transcribed_input, activities_state, history_state],
            outputs=[activities_state, history_state,
                    bot_output_txt, bot_output_image], queue=False
        ).then(
            vc_add_query_log_entry,
            inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
            outputs=[gr_query_log], queue=True
        ).then(
            lambda: gr.update(visible=True), None, textbox_detected_spoken_language, queue=True
        ).then(
            vc_text_to_speech,
            inputs=[bot_output_txt, textbox_detected_spoken_language],
            outputs=[bot_audio_output], queue=False
        )

        # handle speaker input file
        speaker_input_file.then(
            speak_acknowledge, None, None, queue=False
        ).then(
            vc_voice_chat,
            inputs=[speaker_transcribed_input, activities_state],
            outputs=[speaker_transcribed_input, activities_state], queue=False
        ).then(
            vc_unhide_outputs,
            inputs=[checkbox_enable_text_to_image],
            outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                    btn_generate_request_image, btn_generate_response_image], queue=False
        ).then(
            vc_ask_assistant_to_image,
            inputs=[checkbox_enable_text_to_image,
                    speaker_transcribed_input, activities_state, history_state],
            outputs=[activities_state, history_state,
                    bot_output_txt, bot_output_image], queue=False
        ).then(
            vc_add_query_log_entry,
            inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
            outputs=[gr_query_log], queue=True
        ).then(
            vc_text_to_speech,
            inputs=[bot_output_txt, textbox_detected_spoken_language],
            outputs=[bot_audio_output], queue=False
        )

        # handle transcribed text submit event
        speaker_transcribed_input.submit(
            speak_acknowledge, None, None, queue=False
        ).then(
            vc_voice_chat,
            inputs=[speaker_transcribed_input, activities_state],
            outputs=[speaker_transcribed_input, activities_state], queue=False
        ).then(
            vc_unhide_outputs,
            inputs=None,
            outputs=[bot_output_txt, bot_audio_output], queue=False
        ).then(
            vc_ask_assistant_to_image,
            inputs=[checkbox_enable_text_to_image,
                    speaker_transcribed_input, activities_state, history_state],
            outputs=[activities_state, history_state,
                    bot_output_txt, bot_output_image], queue=False
        ).then(
            vc_add_query_log_entry,
            inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
            outputs=[gr_query_log], queue=True
        ).then(
            vc_text_to_speech,
            inputs=[bot_output_txt, textbox_detected_spoken_language],
            outputs=[bot_audio_output], queue=False
        )

        # unhide buttons
        checkbox_enable_text_to_image.change(
            vc_unhide_outputs,
            inputs=[checkbox_enable_text_to_image],
            outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                    btn_generate_request_image, btn_generate_response_image], queue=False
        )

        # "realtime/gradio_tutor/audio/hp0.wav"
        # "realtime/examples/jfk.wav"
        gr.Examples(
        examples=["workspace/samples/jfk.wav"],
        inputs=[speaker_audio_input],
        outputs=[speaker_transcribed_input, textbox_detected_spoken_language],
        fn=speech_to_text,
        cache_examples=True,
        label="example audio file: president JFK speech"
    )

"""UI-CHATBOT SPEAKER"""
with gr.Blocks(title="CHATBOT_ON_UNSTRUCTURED_DATA", analytics_enabled=False) as llm_chatbot_ui:
    with gr.Group():
        with gr.Accordion("Click here for User Options", open=False):
            checkbox_turn_off_voice_response = gr.Checkbox(
                label="Turn off Voice Response", interactive=True)
            checkbox_turn_off_voice_response.change(fn=turn_off_chatbot_voice_options,inputs=checkbox_turn_off_voice_response)               

            llm_textbox_system_prompt = gr.Textbox(system_prompt_assistant, lines=3, label="System Prompt", interactive=True)
            llm_slider_temperature = gr.Slider(
                0, 10, label="Temperature", interactive=True)
            with gr.Row():
                llm_max_tokens = gr.Textbox(
                    value=256, label="Max tokens", interactive=True)
                llm_context_size = gr.Textbox(
                    value=6000, label="Context size", interactive=True)
            with gr.Row():
                """content safety options"""                                
                cbcheckbox_content_safety = gr.Checkbox(
                        value=False, label="enable content safety guard",
                        info="enforce content safety check on input and output")    
                cbcheckbox_content_safety.change(fn=select_content_safety_options,inputs=cbcheckbox_content_safety)                                                                           

        """chatbot knowledge and domain model experts"""
        with gr.Row():
            cb_expert_options = gr.Dropdown(knowledge_experts, label="Ask Knowledge Experts (AI)", info="[optional] route question to subject with domain knowledge.")
            cb_expert_options.change(fn=select_expert_option,inputs=cb_expert_options)        
            cb_model_options = gr.Dropdown(domain_models, label="Custom LLM Model",info="[optional] use specialized model")
            cb_model_options.change(fn=select_model_option,inputs=model_options)
            """image chat options"""                                
            cbcheckbox_turnoff_image_chat_option = gr.Checkbox(
                    value=False, label="turn-off image chat",
                    info="turn-off to reset image chat mode")    
            cbcheckbox_turnoff_image_chat_option.change(fn=turn_off_image_chat_options,
                                                        inputs=cbcheckbox_turnoff_image_chat_option)                                                                           

        chatbot_ui = gr.Chatbot(
            value=[], elem_id="chatbot",
            height=490, bubble_full_width=False,
            layout="bubble", show_copy_button=True, show_share_button=True,
        )
        with gr.Row():
            textbox_user_prompt = gr.Textbox(scale=4, show_label=True,
                                             label="User input",
                                             placeholder="Enter text and press enter, or record an audio",
                                             container=True, autofocus=True, interactive=True, show_copy_button=True
                                             )
        with gr.Row():
            textbox_local_filename = gr.Textbox(scale=5, show_label=True, visible=False,
                                                label="local filename",
                                                show_copy_button=True
                                                )
            textbox_chatbot_response = gr.Textbox(scale=5, show_label=False, visible=False,
                                                  label="chatbot response",
                                                  show_copy_button=True)
        with gr.Row():
            btn_new_conversation = gr.ClearButton(components=[textbox_user_prompt, chatbot_ui, history_state,
                                                              speaker_audio_input, bot_audio_output, speaker_transcribed_input,
                                                              bot_output_txt], value="🗨 New Conversation", scale=2
                                                  )
            
            btn_upload_chat_file = gr.UploadButton(
                label="📁 Upload a File", size="sm", file_types=["image", "video", "audio"])
            
            btn_upload_chat_file.click(lambda: gr.update(
                visible=True), None, textbox_local_filename, queue=False)
            # Connect upload button to chatbot
            file_msg = btn_upload_chat_file.upload(
                cb_add_file,
                inputs=[chatbot_ui, btn_upload_chat_file],
                outputs=[chatbot_ui, textbox_local_filename], queue=False
            ).then(
                speak_wait, None, None, queue=False
            )
            logger.info(f"process uploaded file.")                        
            file_msg.then(
                cb_process_upload_audio,
                inputs=[textbox_local_filename, chatbot_ui],
                outputs=[chatbot_ui, textbox_user_prompt,
                         textbox_chatbot_response], queue=False  
            ).then(
                cb_process_upload_text,
                inputs=[textbox_local_filename, chatbot_ui],
                outputs=[chatbot_ui, textbox_user_prompt,
                         textbox_chatbot_response], queue=False 
            ).then(
                cb_process_upload_image,
                inputs=[textbox_local_filename, chatbot_ui],
                outputs=[chatbot_ui, textbox_user_prompt,
                         textbox_chatbot_response], queue=False 
            ).then(            
                cb_botchat,
                inputs=[chatbot_ui, history_state, textbox_chatbot_response],
                outputs=[chatbot_ui, history_state,
                         textbox_chatbot_response], queue=False
            ).then(
                cb_update_chat_session,
                inputs=[history_state],
                outputs=[chatbot_ui, history_state,
                         session_tokens], queue=False
            ).then(
                speak_done, None, None, queue=False
            ).then(
                cb_reset_chat_outputs, outputs=[
                    textbox_local_filename, textbox_chatbot_response], queue=True
            )

            # Connect new conversation button
            btn_new_conversation.click(cb_hide_outputs,
                                       inputs=None,
                                       outputs=[bot_output_txt,
                                                bot_audio_output]
                                       ).then(lambda: None, None, chatbot_ui, queue=False
                                              ).then(lambda: None, [], history_state, queue=False
                                                     )
        # 2.Connect user input to chatbot
        cb_txt_msg = textbox_user_prompt.submit(cb_filter_commmand_keywords,
                                                inputs=[
                                                    textbox_user_prompt, chatbot_ui],
                                                outputs=[textbox_user_prompt, chatbot_ui,
                                                         textbox_local_filename, textbox_chatbot_response,
                                                         textbox_detected_spoken_language], queue=False
                                                ).then(
            cb_add_file,
            inputs=[chatbot_ui,
                    textbox_local_filename],
            outputs=[
                chatbot_ui, textbox_local_filename], queue=False
        )

        cb_txt_msg.then(cb_user_chat,
                        inputs=[textbox_user_prompt, chatbot_ui,
                                textbox_chatbot_response],
                        outputs=[textbox_user_prompt, chatbot_ui], queue=False
                        ).then(cb_botchat,
                               inputs=[chatbot_ui, history_state,
                                       textbox_chatbot_response],
                               outputs=[chatbot_ui, history_state,
                                        textbox_chatbot_response], queue=False
                        ).then(cb_update_chat_session,
                                inputs=[history_state],
                                outputs=[chatbot_ui, history_state,session_tokens], queue=False                                        
                        ).then(cb_text_to_speech,
                                inputs=[textbox_chatbot_response, textbox_detected_spoken_language, checkbox_turn_off_voice_response],
                                outputs=[bot_audio_output], queue=False
                        ).then(cb_reset_chat_outputs, 
                               outputs=[textbox_local_filename, textbox_chatbot_response], queue=True
                        )
        #.then(lambda: gr.update(interactive=True), None, textbox_user_prompt)

        with gr.Accordion("Click here for Available Commands", open=False):
            # gr.Markdown(config.INSTRUCTIONS_TEXT)
            with gr.Row():
                with gr.Column(1):
                    gr.Markdown("#### Youtube Commands:")
                    gr.Markdown("""
                    | command            | parameters    |  example                                                                 |
                    |----------          |:-------------:|:------                                                                   |
                    | /ytranscribe video|  url          | ```/ytranscribe:https://www.youtube.com/watch?v=E9lAeMz1DaM (english)```|
                    | /ysummarize audio |  url          | ```/ysummarize:https://www.youtube.com/watch?v=E9lAeMz1DaM (english)```|
                    | /ytranslate audio to english |  url          | ```/ytranslate:https://www.youtube.com/watch?v=hz5xWgjSUlk (french)```|

                    **Random Examples: English** 
                    ``` 
                    15 Tips To Manage Your Time Better
                    /ytranscribe:https://www.youtube.com/watch?v=GBM2k2zp-MQ
                                                                
                    Avril Lavigne - Breakaway
                    /ytranscribe:https://www.youtube.com/watch?v=oc7JLUvY9xI               
                    ```            
                    **Random Examples: Non-english** 
                    ``` 
                    > Case of Mixed languages
                    /ytranscribe:https://www.youtube.com/watch?v=zHGCJD64Djo                                
                    /ytranslate:https://www.youtube.com/watch?v=zHGCJD64Djo   
                                          
                    > Case of Non-english Talk show
                    (French talk-show: Raymond Devos-Parler pour ne rien dire)
                    /ytranslate:https://www.youtube.com/watch?v=hz5xWgjSUlk                                  
                    /ytranscribe:https://www.youtube.com/watch?v=hz5xWgjSUlk                                                   
                    /ysummarize:https://www.youtube.com/watch?v=hz5xWgjSUlk      
                    
                    > More Case of Non-english songs                                                                                                             
                    (Japanese: original song 彼岸花(マンジュシャカ) by Yamaguchi Momoe)
                    /ytranscribe:https://www.youtube.com/watch?v=JmS6zsR3UBs
                    /ytranslate:https://www.youtube.com/watch?v=JmS6zsR3UBs  
                    (Spanish song ASI ES LA VIDA by Enrique Iglesias,Maria Becerra)            
                    /ytranscribe:https://www.youtube.com/watch?v=XUoXE3bmDJY                        
                    /ytranslate:https://www.youtube.com/watch?v=XUoXE3bmDJY                                                          
                    (Cantonese song 蔓珠莎華 by various singers)
                    /ytranslate:https://www.youtube.com/watch?v=fxruu8cZams                                
                    /ytranslate:https://www.youtube.com/watch?v=f0Bn_wKLL5M                                        
                    ```            
                    `/ytranscribe` - extract audio from youtube video and transcribe audio to text

                    `/ysummarize` - extract audio from youtube video, pass to LLM to generate a summary of key points

                    `/ytranslate` - translate foreign language audio into English to english text.
                                             
                    """)
                with gr.Column(1):
                    gr.Markdown("#### Web Commands:")
                    gr.Markdown("""
                    | command            |      parameters                        |  example            |
                    |----------          |:-------------:                         |------:              |
                    | /ddgsearch? |  query=CN towe&max_results=5                       | ```/ddgsearch?query=CN towe&max_results=5``` |
                    | /ddgnews?   |  query=holiday&region=us&max_results=5&timelimit=d| ```/ddgnews?query=IBM stock price&max_results=5&backend=api``` |
                    | /ddgimage?  |  query,region="us-en",max_results=5 | ```/ddgimage?query=butterfly&max_results=5&backend=api&size=small&color=color&license_image=Public```    |
                    | /ddgvideo?  |  query,region="us-en",max_results=5 | ```/ddgvideo?query=butterfly&max_results=5&backend=api&resolution=standard&duration=short&license_videos=youtube``` |
                    | /ddganswer? |  query | ```/ddganswer?query=sun``` |                   
                    | /ddgsuggest? |  query | ```/ddgsuggest?query=butterfly``` |                                        
                    | /ddgtranslate? |  query | ```/ddgtranslate?query=butterfly&to_lang=fr``` |                                                            
                    """)
                    gr.Markdown("#### Summarize text file:")
                    gr.Markdown("""
                                ```/tfsummarize``` - generate a summary of the text content
                                **usage**
                                ```/tfsummarize:https://cdn.serc.carleton.edu/files/teaching_computation/workshop_2018/activities/plain_text_version_declaration_inde.txt```                        
                    """)

"""APP UI"""
app_ui = gr.TabbedInterface(
    theme=theme,
    interface_list=[llm_voice_query_ui,llm_chatbot_ui,llm_setting_ui],
    tab_names=["Voice Prompt", "Chatbot Speaker","System Settings"],
    title="🎙️VOCIE: private assistant 💬",
    css=".gradio-container {background: url('file=pavai_logo_large.png')}",
    analytics_enabled=False
)

with app_ui:
    with gr.Accordion("Debug", open=False):
        json = gr.JSON()
        btn_load = gr.Button("Dump session info to JSON")
        btn_load.click(lambda x: x, history_state, json)
    with gr.Group():
        gr.HTML(show_label=False, value="PavAI-Vocie prototype(1).  alpha 0.0.3. copyright@2024")

def update_gc_threshold():
    """optimize default memory settings"""
    import gc
    allocs,g1,g2=gc.get_threshold()
    gc.set_threshold(50_000,g1*5,g2*10)

"""MAIN"""
if __name__ == "__main__":
    import os
    background_image="resources/images/pavai_logo_large.png"
    authorized_users=[("abc:123"),("admin:123"),("john:smith"),("hello:hello")]      
    auth=[tuple(cred.split(':')) for cred in authorized_users] if authorized_users else None 
    try:
        absolute_path = os.path.abspath(background_image)
        pavai_vocie_system_health_check()
        server_name = "0.0.0.0" if "VOCIE_APP_HOST" not in system_config.keys() else system_config["VOCIE_APP_HOST"]
        server_port = 7860 if "VOCIE_APP_PORT" not in system_config.keys() else int(system_config["VOCIE_APP_PORT"])        
        share=False if "VOCIE_APP_SHARE" not in system_config.keys() else bool(system_config["VOCIE_APP_SHARE"])
        update_gc_threshold()        
        app_ui.queue()
        app_ui.launch(share=False,
                        auth=auth,
                    allowed_paths=[absolute_path],
                    server_name=server_name,
                    server_port=server_port)
    except Exception as ex:
        print("An error has occurred ",ex)
        gr.Error("Something went wrong! see console or log file for more details")
        speak_instruction(instruction="oops!, An error has occurred. start up failed. please check the console and logs.")
        speak_instruction(instruction="error message says "+str(ex.args))        
    