from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import gc
import traceback
import os, sys
import gradio as gr
import torch
from transformers.utils import is_flash_attn_2_available
from pavai.shared.system_checks import (pavai_vocie_system_health_check,DEFAULT_SYSTEM_MODE, SYSTEM_THEME_SOFT,SYSTEM_THEME_GLASS,VOICE_PROMPT_INSTRUCTIONS_TEXT)
from pavai.shared.audio.transcribe import (speech_to_text, FasterTranscriber,DEFAULT_WHISPER_MODEL_SIZE)
from pavai.shared.image.text2image import (StableDiffusionXL, image_generation_client,DEFAULT_TEXT_TO_IMAGE_MODEL)
from pavai.shared.audio.tts_client import speak_instruction
from pavai.vocei_web.translator_ui import CommunicationTranslator
from pavai.vocei_web.scratchpad_ui import ScratchPad
from pavai.vocei_web.voice_prompt_ui import VoicePrompt
import pavai.setup.versions as versions

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2024"
__version__ = "0.0.3"

logger.warning("--GLOBAL SYSTEM MODE----")
logger.warning(config.system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=config.system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=config.system_config["GLOBAL_TTS"]

# tested version gradio version: 4.7.1
# pip install gradio==4.7.1

# whisper model
DEFAULT_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
DEFAULT_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
INT16_MAX_ABS_VALUE = 32768.0
DEFAULT_MAX_MIC_RECORD_LENGTH_IN_SECONDS = 30*60*60  # 30 miniutes

# Global variables
system_is_ready = True
use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()

faster_transcriber = FasterTranscriber(model_id_or_path=DEFAULT_WHISPER_MODEL_SIZE)
stablediffusion_model = StableDiffusionXL(model_id_or_path=DEFAULT_TEXT_TO_IMAGE_MODEL)

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

js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '1em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '10px';

    var text = 'Welcome to Vocie(C-3P0)';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];
                container.appendChild(letter);
                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

"""
MAIN User Interface
"""
title = '🤖 Pavai-Vocei(C3PO) - an advance private multilingual LLM-based AI Voice Assistant'
description = """an advance private multilingual AI Voice Assistant."""

theme = gr.themes.Default()
if "SYSTEM_THEME" in config.system_config.keys():
    system_theme = config.system_config["SYSTEM_THEME"]
    if system_theme == SYSTEM_THEME_SOFT:
        theme = gr.themes.Soft()
    elif system_theme == SYSTEM_THEME_GLASS:
        theme = gr.themes.Glass()

if "DEFAULT_SYSTEM_MODE" not in config.system_config.keys():
    system_mode = DEFAULT_SYSTEM_MODE
else:
    system_mode = config.system_config["DEFAULT_SYSTEM_MODE"]

class VoceiApp(VoicePrompt,CommunicationTranslator,ScratchPad):
    
    def __init__(self,name) -> None:
        super().__init__()
        self.name=name

    def welcome(name):
        return f"Welcome to Vocei, {name}!"

    def update_gc_threshold(self):
        """optimize default memory settings"""
        import gc
        allocs,g1,g2=gc.get_threshold()
        gc.set_threshold(50_000,g1*5,g2*10)

    def main(self):
        voice_prompt_ui=self.build_voice_prompt_ui()
        translator_ui=self.build_translator_ui()
        scratchpad_ui=self.build_scratchpad_ui()        
        """APP UI""" 
        ##    title="Vocie (C-3PO real assistant) 💬",
        ##    css=".gradio-container {background: url('file=pavai_logo_large.png')}",
        css = ".gradio-container {background: url(https://w0.peakpx.com/wallpaper/249/289/HD-wallpaper-c3po-and-r2d2-star-wars-c3po-movies-r2d2.jpg)}"       
        self.app_ui = gr.TabbedInterface(
            theme=theme,
            interface_list=[voice_prompt_ui,translator_ui,scratchpad_ui],
            tab_names=["Human to Machine Voice Prompt", "Inter-Human Communication (Multilingual)","Scratch Pad"],
            analytics_enabled=False,
            js=js,
            css=css,
            head="Pavai-Vocei"            
        )
        with self.app_ui:
            with gr.Group():
                gr.HTML(show_label=False, value=versions.version_full)

        return self.app_ui

    def wipe_memory(self,objects:list=[]): # DOES WORK
        try:
            for obj in objects:
                del obj
            collected = gc.collect()
            logger.debug("Garbage collector: collected","%d objects." % collected)
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
            self.update_gc_threshold()       
            self.wipe_memory()  
            self.app_ui.queue()
            self.app_ui.launch(share=False,auth=None,allowed_paths=[absolute_path],server_name=server_name,server_port=server_port)
        except Exception as ex:
            print("An error has occurred ",ex)
            logger.error(traceback.format_exc())
            gr.Error("Something went wrong! see console or log file for more details")
            speak_instruction(instruction="oops!, An error has occurred. start up failed. please check the console and logs.")
            speak_instruction(instruction="error message says "+str(ex.args))

"""MAIN"""
if __name__ == "__main__":
    server_name = "0.0.0.0" if "VOCIE_APP_HOST" not in config.system_config.keys() else config.system_config["VOCIE_APP_HOST"]
    server_port = 7860 if "VOCIE_APP_PORT" not in config.system_config.keys() else int(config.system_config["VOCIE_APP_PORT"])        
    share=False if "VOCIE_APP_SHARE" not in config.system_config.keys() else bool(config.system_config["VOCIE_APP_SHARE"])

    voiceapp=VoceiApp("Pavai-Vocie(C3PO)")
    voiceapp.launch(server_name=server_name,server_port=server_port,share=share)
