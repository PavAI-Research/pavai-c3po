from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import random
from pathlib import Path
from pavai.shared.styletts2 import LibriSpeech, LJSpeech
import sounddevice as sd

_GLOBAL_SYSTEM_MODE=config.system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=config.system_config["GLOBAL_TTS"]
_GLOBAL_TTS_LIBRETTS_VOICE=config.system_config["GLOBAL_TTS_LIBRETTS_VOICE"]

_GLOBAL_STT=config.system_config["GLOBAL_STT"]
_GLOBAL_TTS_API_ENABLE=config.system_config["GLOBAL_TTS_API_ENABLE"]
_GLOBAL_TTS_API_URL=config.system_config["GLOBAL_TTS_API_URL"]
_GLOBAL_TTS_API_LANGUAGE=config.system_config["GLOBAL_TTS_API_LANGUAGE"]
_GLOBAL_TTS_API_SPEAKER_MODEL=config.system_config["GLOBAL_TTS_API_SPEAKER_MODEL"]

## global instance
onelibrispeech = LibriSpeech()
#librispeech = LJSpeech()

def lpad_text(text:str, max_length:int=43, endingchar:str="c")->str:
    if len(text) < max_length:
        text=text.ljust(max_length, '…')
    return text+"."

def get_speaker_audio_file(workspace_temp:str="workspace/temp")->str:
    Path.mkdir(workspace_temp, exist_ok=True)
    return workspace_temp+"/espeak_text_to_speech.mp3"

def system_tts_local(text:str,output_voice:str=None,vosk_params=None,autoplay:bool=True):
    compute_style=config.system_config["GLOBAL_TTS_LIBRETTS_VOICE"]  
    logger.info(f"tts: {compute_style}")
    logger.info(f"tts: {text}")    
    speaker_file_v2(text=text,output_voice=output_voice,autoplay=autoplay)  

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
    system_tts_local(text=ack_text,autoplay=True)  

def speak_wait():
    acknowledges = ["okay! please wait",
                    "got it!, one moment",
                    "certainly!, one moment",
                    "sure, one moment",
                    "process your request."]
    ack_text = str(random.choice(acknowledges))
    print("speak_wait: ", ack_text)
    system_tts_local(text=ack_text,autoplay=True)    

def speak_done():
    acknowledges = ["all done!, please check.",
                    "process complete, please check",
                    "finish processing!, please check",
                    "completed your request!, please check"
                    ]
    ack_text = str(random.choice(acknowledges))
    system_tts_local(text=ack_text,autoplay=True)    

def speak_instruction(instruction: str, output_voice: str = "jane"):
    logger.info(f"speak_instruction: {instruction}")
    system_tts_local(text=instruction,autoplay=True, output_voice=output_voice)    

def speaker_file(text:str,autoplay:bool=True)->str:
    wav_file = LJSpeech().ljspeech_v2(text=text,autoplay=autoplay)
    return wav_file

def speaker_text(text:str,output_voice:str=None,autoplay=True)->str:
    global onelibrispeech
    if output_voice is None:
        output_voice = config.system_config["GLOBAL_TTS_LIBRETTS_VOICE"]
    wav = onelibrispeech.librispeech(text=text,compute_style=output_voice,autoplay=autoplay)
    return wav

def speaker_file_v2(text:str,output_voice:str=None,output_emotion:str=None,vosk_params=None,chunk_size:int=500,autoplay=False)->str:
    global onelibrispeech
    if output_voice is None:
        output_voice = config.system_config["GLOBAL_TTS_LIBRETTS_VOICE"]
    if output_emotion is not None:
        wav_file = onelibrispeech.librispeech_v2(text=text,compute_style=output_voice,emotion=output_emotion,autoplay=autoplay)
    else:
        wav_file = onelibrispeech.librispeech_v3(text=text,compute_style=output_voice,autoplay=autoplay)
    return wav_file
