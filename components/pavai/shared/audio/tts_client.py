from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
import random
from rich.logging import RichHandler
from rich import print,pretty,console
from rich.pretty import (Pretty,pprint)
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import Progress
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
#pretty.install()
import warnings 
warnings.filterwarnings("ignore")

from pathlib import Path
from pavai.shared.audio.stt_vad import init_vad_model
from pavai.shared.audio.voices_piper import espeak,get_voice_model_file
#from pavai.shared.audio.voices_styletts2 import librispeak
from pavai.shared.styletts2 import LibriSpeech, LJSpeech
from pavai.shared.audio.vosk_client import api_speaker
from pavai.shared.audio.tts_gtts import text_to_speech_gtts
#from pavai.shared.styletts2 import (ljspeech,ljspeech_v2,test_lj_speech,test_lj_speech_v2)
import time
import numpy
import sounddevice as sd

print("--GLOBAL SYSTEM MODE----")
print(system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=system_config["GLOBAL_TTS"]
_GLOBAL_TTS_LIBRETTS_VOICE=system_config["GLOBAL_TTS_LIBRETTS_VOICE"]

_GLOBAL_STT=system_config["GLOBAL_STT"]
_GLOBAL_TTS_API_ENABLE=system_config["GLOBAL_TTS_API_ENABLE"]
_GLOBAL_TTS_API_URL=system_config["GLOBAL_TTS_API_URL"]
_GLOBAL_TTS_API_LANGUAGE=system_config["GLOBAL_TTS_API_LANGUAGE"]
_GLOBAL_TTS_API_SPEAKER_MODEL=system_config["GLOBAL_TTS_API_SPEAKER_MODEL"]

## global instance
onelibrispeech = LibriSpeech()
#librispeech = LJSpeech()

def get_speaker_audio_file(workspace_temp:str="workspace/temp")->str:
    Path.mkdir(workspace_temp, exist_ok=True)
    # if not os.path.exists(workspace_temp):
    #     os.mkdir(workspace_temp)
    return workspace_temp+"/espeak_text_to_speech.mp3"

def system_tts_local(sd,text:str,output_voice:str=None,vosk_params=None,autoplay:bool=True):
    if _GLOBAL_TTS_API_ENABLE=="true":
        ## use vosk api - piper ai-voice 
        vosk_params = {
            "sentence":text,
            "api_url":system_config["GLOBAL_TTS_API_URL"],  
            "language": system_config["GLOBAL_TTS_API_LANGUAGE"] ,    
            "speaker_model":system_config["GLOBAL_TTS_API_SPEAKER_MODEL"]
        }        
        api_speaker(vosk_params,text)
    else:
        if _GLOBAL_TTS=="LIBRETTS":
            ## human-liked custom voices        
            compute_style=system_config["GLOBAL_TTS_LIBRETTS_VOICE"]  
            print("compute_style: ",compute_style)
            ##librispeak(text=text,compute_style="jane")
            speaker_file_v2(text=text,autoplay=True)  
        elif _GLOBAL_TTS=="GTTS":            
            # google voice
            text_to_speech_gtts(text=text,autoplay=True)
        elif _GLOBAL_TTS=="LINUX":            
            # linux default voice
            import os
            os.system(f"spd-say {text}")
        else:
            if output_voice is None:
                output_voice=system_config["GLOBAL_TTS_PIPER_VOICE"]
            espeak(sd,text,output_voice=output_voice)

def speak_acknowledge():
    acknowledges = ["Nice,",
                    "Sure thing,",
                    "Yes! It's great",
                    "okay! or I like it!",
                    "So true",
                    "Got it.",
                    "Please stay online.",
                    "Do you mind waiting for a moment while I look up this information.",
                    "Please give me a moment while I look into this for you.",
                    "It will take me just a moment to process your request.",
                    "Oh! I had no idea",
                    "I totally get what you're saying",
                    "yea, Love it.",
                    "Good one"]
    wait_phases = ["please wait ", " one moment", "thanks!",
                   ", Please stay online.", " Thank you!"]
    waiting = str(random.choice(wait_phases))
    ack_text = str(random.choice(acknowledges))+waiting
    print("speak_acknowledge: ", ack_text)
    #text_to_speech(text=ack_text, output_voice="en", autoplay=True)
    system_tts_local(sd,text=ack_text,autoplay=True)  
    #text_speaker_ai(sd,text=ack_text)      

def speak_wait():
    acknowledges = ["okay! please wait",
                    "got it!, one moment",
                    "certainly!, one moment",
                    "sure, one moment",
                    "process your request."]
    ack_text = str(random.choice(acknowledges))
    print("speak_wait: ", ack_text)
    #text_to_speech(text=ack_text, output_voice="en", autoplay=True)
    system_tts_local(sd,text=ack_text,autoplay=True)    
    #librispeak(text=ack_text,compute_style="jane")
    #text_speaker_ai(sd,text=ack_text)          

def speak_done():
    acknowledges = ["all done!, please check.",
                    "process complete, please check",
                    "finish processing!, please check",
                    "completed your request!, please check"
                    ]
    ack_text = str(random.choice(acknowledges))
    # print("speak_done: ", ack_text)
    #text_to_speech(text=ack_text, output_voice="en", autoplay=True)
    system_tts_local(sd,text=ack_text,autoplay=True)    
    #text_speaker_ai(sd,text=ack_text)              

def speak_instruction(instruction: str, output_voice: str = "en"):
    logger.info(f"speak_instruction: {instruction}")
    ##text_to_speech(text=instruction, output_voice=output_voice, autoplay=True)
    system_tts_local(sd,text=instruction,autoplay=True)    
    #text_speaker_ai(sd,text=instruction)                  

def speaker_file(text:str,autoplay:bool=True)->str:
    wav_file = LJSpeech().ljspeech_v2(text=text,autoplay=autoplay)
    return wav_file

def speaker_file_v2(text:str,output_voice:str="jane",vosk_params=None,chunk_size:int=500,autoplay=False)->str:
    global onelibrispeech
    wav_file = onelibrispeech.librispeech_v3(text=text,compute_style=output_voice,autoplay=autoplay)
    return wav_file
