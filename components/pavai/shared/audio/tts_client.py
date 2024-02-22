from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
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
from shared.audio.stt_vad import init_vad_model
from shared.audio.voices_piper import espeak,get_voice_model_file
from shared.audio.voices_styletts2 import librispeak
from shared.audio.vosk_client import api_speaker
from shared.audio.tts_gtts import text_to_speech_gtts
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

def get_speaker_audio_file(workspace_temp:str="workspace/temp")->str:
    Path.mkdir(workspace_temp, exist_ok=True)
    # if not os.path.exists(workspace_temp):
    #     os.mkdir(workspace_temp)
    return workspace_temp+"/espeak_text_to_speech.mp3"

def text_speaker_ai(sd,text:str,output_voice:str="en_ryan",vosk_params=None):
    if _GLOBAL_TTS_API_ENABLE=="true":
        ## use vosk api - piper ai-voice 
        vosk_params = {
            "sentence":text,
            "api_url":_GLOBAL_TTS_API_URL,
            "language": _GLOBAL_TTS_API_LANGUAGE,    
            "speaker_model":_GLOBAL_TTS_API_SPEAKER_MODEL
        }        
        api_speaker(vosk_params,text)
    else:
        if _GLOBAL_TTS=="LIBRETTS":
            ## human-liked custom voices             
            librispeak(text,compute_style=_GLOBAL_TTS_LIBRETTS_VOICE)
        elif _GLOBAL_TTS=="GTTS":            
            # google voice
            text_to_speech_gtts(text=text,autoplay=True)
        elif _GLOBAL_TTS=="LINUX":            
            # linux default voice
            import os
            os.system(f"spd-say {text}")
        else:
            espeak(sd,text,output_voice=output_voice)