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
import gc
import traceback
import os, sys
import gradio as gr
import torch
import pandas as pd
import numpy as np
#import pavai.shared.datasecurity as datasecurity
from typing import BinaryIO, Union
from transformers.utils import is_flash_attn_2_available
from pavai.shared.system_checks import (pavai_vocie_system_health_check,DEFAULT_SYSTEM_MODE, SYSTEM_THEME_SOFT,SYSTEM_THEME_GLASS,VOICE_PROMPT_INSTRUCTIONS_TEXT)
from pavai.shared.audio.transcribe import (speech_to_text, FasterTranscriber,DEFAULT_WHISPER_MODEL_SIZE)
#from pavai.shared.llmproxy import chatbot_ui_client,chat_count_tokens,multimodal_ui_client
#from pavai.shared.llmproxy import chat_count_tokens,multimodal_ui_client
from pavai.shared.image.text2image import (StableDiffusionXL, image_generation_client,DEFAULT_TEXT_TO_IMAGE_MODEL)
# from pavai.shared.fileutil import get_text_file_content
# from pavai.shared.commands import filter_commmand_keywords
# from pavai.shared.grammar import (fix_grammar_error)
from pavai.shared.audio.tts_client import speak_instruction
# get_speaker_audio_file, 
#from pavai.shared.audio.voices_piper import (text_to_speech, speak_acknowledge,speak_wait, speak_done, speak_instruction)
#from pavai.shared.audio.voices_styletts2 import (text_to_speech, speak_acknowledge,speak_wait, speak_done, speak_instruction)

## remove Iterable, List, NamedTuple, Optional, Tuple,
## removed from os.path import dirname, join, abspath
from pavai.shared.aio.llmchat import (system_prompt_assistant, DEFAULT_LLM_CONTEXT_SIZE)
from pavai.shared.llmcatalog import LLM_MODEL_KX_CATALOG_TEXT
from pavai.shared.aio.chatprompt import knowledge_experts_system_prompts
from pavai.shared.aio.llmchat import get_llm_library
from pavai.vocei_web.translator_ui import CommunicationTranslator,ScratchPad
#from pavai.vocei_web.system_settings_ui import SystemSetting
from pavai.vocei_web.voice_prompt_ui import VoicePrompt
#from pavai.vocei_web.chatbot_speaker_ui import ChatbotSpeaker


__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2024"
__version__ = "0.0.3"

logger.warning("--GLOBAL SYSTEM MODE----")
logger.warning(system_config["GLOBAL_SYSTEM_MODE"])
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


"""
MAIN User Interface
"""
title = '🤖 PAvAI C3PO - an advance private multilingual LLM-based AI Voice Assistant'
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

class VoceiApp(VoicePrompt,CommunicationTranslator,ScratchPad):
    
    def __init__(self,name) -> None:
        super().__init__()
        self.name=name

    def update_gc_threshold(self):
        """optimize default memory settings"""
        import gc
        allocs,g1,g2=gc.get_threshold()
        gc.set_threshold(50_000,g1*5,g2*10)

    def main(self):
        #system_settings_ui=self.build_system_setting_ui()
        voice_prompt_ui=self.build_voice_prompt_ui()
        #chatbot_speaker_ui=self.build_chatbot_speaker_ui()
        translator_ui=self.build_translator_ui()
        scratchpad_ui=self.build_scratchpad_ui()        
        """APP UI"""
        # css_code='body{background-image:url("https://picsum.photos/seed/picsum/200/300");}'
        # css_image='div {margin-left: auto; margin-right: auto; width: 100%;\
        #     background-image: url("file=pavai_logo_large.png"); repeat 0 0;}'

        self.app_ui = gr.TabbedInterface(
            theme=theme,
            interface_list=[voice_prompt_ui,translator_ui,scratchpad_ui],
            tab_names=["Voice Prompt", "Multilingual Communication","Scratch Pad"],
            title="[C-3PO] Real Voice Assistant 💬",
            css=".gradio-container {background: url('file=pavai_logo_large.png')}",
            analytics_enabled=False            
        )
        with self.app_ui:
            with gr.Group():
                gr.HTML(show_label=False, value="PAvAI-VOCIE prototype(1).  alpha 0.0.3. copyright@2024")

        return self.app_ui

    def wipe_memory(self,objects:list=[]): # DOES WORK
        try:
            for obj in objects:
                del obj
            collected = gc.collect()
            print("Garbage collector: collected","%d objects." % collected)
            torch.cuda.empty_cache()
        except:
            pass

    def launch(self,server_name:str="0.0.0.0",server_port:int=7868,share:bool=False,**kwargs):
        background_image="resources/images/pavai_logo_large.png"
        authorized_users=[("abc:123"),("admin:123"),("john:smith"),("hello:hello")]      
        auth=[tuple(cred.split(':')) for cred in authorized_users] if authorized_users else None 
        try:
            self.main()
            absolute_path = os.path.abspath(background_image)
            pavai_vocie_system_health_check()
            #self.update_gc_threshold()       
            self.wipe_memory()  
            self.app_ui.queue()
            self.app_ui.launch(share=False,auth=None,allowed_paths=[absolute_path],server_name=server_name,server_port=server_port)
        except Exception as ex:
            print("An error has occurred ",ex)
            print(traceback.format_exc())
            gr.Error("Something went wrong! see console or log file for more details")
            speak_instruction(instruction="oops!, An error has occurred. start up failed. please check the console and logs.")
            speak_instruction(instruction="error message says "+str(ex.args))

"""MAIN"""
if __name__ == "__main__":
    server_name = "0.0.0.0" if "VOCIE_APP_HOST" not in system_config.keys() else system_config["VOCIE_APP_HOST"]
    server_port = 7860 if "VOCIE_APP_PORT" not in system_config.keys() else int(system_config["VOCIE_APP_PORT"])        
    share=False if "VOCIE_APP_SHARE" not in system_config.keys() else bool(system_config["VOCIE_APP_SHARE"])

    voiceapp=VoceiApp("Real-C3PO-Vocie")
    voiceapp.launch(server_name=server_name,server_port=server_port,share=share)
