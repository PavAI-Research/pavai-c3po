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
import os
import time
import shutil
import requests
import functools
import numpy as np
import pydub
import sounddevice as sd
from pathlib import Path
import os, sys
# from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), './shared')))
## google
#from gtts import gTTS  
## stylett2
#from pavai.shared.styletts2 import LibriSpeech, LJSpeech
from pavai.shared.audio.tts_piper import load_speech_synthesizer_model
# load_speech_synthesizer
#from pavai.shared.system_checks import get_speaker_audio_file
# from styletts2 import  ljspeech
# ljspeech(text=text,device="cuda")

from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from io import BytesIO
from pydub.playback import play
from pydub import AudioSegment
# _voice_model_name = None
# _voice_synthesize = None
# _local_voice_path = "/home/pop/development/mclab/talking-llama/models/voices/"

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

_local_voice_path = system_config["DEFAULT_TTS_VOICE_MODEL_PATH"]

# pick up active voice file
DEFAULT_ESPEAK_VOICE_AGENT = system_config["ACTIVE_TTS_VOICE_MODEL_AGENT"]
DEFAULT_ESPEAK_VOICE_MODEL = system_config["ACTIVE_TTS_VOICE_MODEL_ONNX_FILE"]
DEFAULT_ESPEAK_VOICE_LANGUAGE = system_config["ACTIVE_TTS_VOICE_MODEL_LANGUAGE"]

# # styleTTs2 reference voices
# human_reference_voices = {
#     "Ryan": compute_style("resources/models/styletts2/reference_audio/Ryan.wav"),
#     "Jane": compute_style("resources/models/styletts2/reference_audio/Jane.wav"),
#     "Me1": compute_style("resources/models/styletts2/reference_audio/Me1.wav"),
#     "Me2": compute_style("resources/models/styletts2/reference_audio/Me2.wav"),
#     "Me3": compute_style("resources/models/styletts2/reference_audio/Me3.wav"),
#     "Vinay": compute_style("resources/models/styletts2/reference_audio/Vinay.wav"),
#     "Nima": compute_style("resources/models/styletts2/reference_audio/Nima.wav"),
#     "Yinghao": compute_style("resources/models/styletts2/reference_audio/Yinghao.wav"),
#     "Keith": compute_style("resources/models/styletts2/reference_audio/Keith.wav"),
#     "May": compute_style("resources/models/styletts2/reference_audio/May.wav"),
#     "June": compute_style("resources/models/styletts2/reference_audio/June.wav")
# }

# demo-1
# librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0.3, beta=0.5, diffusion_steps=10)

# cached data
cache_voices_files = {}
cache_voices_models = {}

# def librispeak(text: str, compute_style:str="Jane",alpha=0.3, beta=0.7, diffusion_steps=10,output_voice_lang: str = "en"):
#     ref_s2=human_reference_voices[compute_style]
#     wav=librispeech(text=text,compute_style=ref_s2, voice=compute_style,alpha=alpha, 
#                     beta=beta, diffusion_steps=diffusion_steps)
#     return wav

# #librispeak("hello world")
def get_speaker_audio_file(workspace_temp:str="workspace/temp")->str:
    if not os.path.exists(workspace_temp):
        os.mkdir(workspace_temp)
    return workspace_temp+"/espeak_text_to_speech.mp3"

def download_file(url, local_path: str = None):
    local_filename = url.split('/')[-1]
    if local_path is not None:
        local_filename = local_path+local_filename
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def get_voice_model_file(local_voice_path: str, spoken_language: str = "en") -> str:
    spoken_language = "en" if spoken_language is None or len(
        spoken_language.strip()) == 0 else spoken_language
    voice_map = {
        "ar": ["ar_JO-kareem-medium.onnx",
               "ar_JO-kareem-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/ar/ar_JO/kareem/medium/"],
        "cs": ["cs_CZ-jirka-medium.onnx",
               "cs_CZ-jirka-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/cs/cs_CZ/jirka/medium/"],
        "da": ["da_DK-talesyntese-medium.onnx",
               "da_DK-talesyntese-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/da/da_DK/talesyntese/medium/"],
        "de": ["de_DE-thorsten-medium.onnx",
               "de_DE-thorsten-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/de/de_DE/thorsten/medium/"],
        "en": ["en_US-amy-medium.onnx",
               "en_US-amy-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/amy/medium/"],
        "en_ryan": ["en_US-ryan-medium.onnx",
                           "en_US-ryan-medium.onnx.json",
                           "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/ryan/medium/"],               
        "en_amy": ["en_US-amy-medium.onnx",
                   "en_US-amy-medium.onnx.json",
                   "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/amy/medium/"],
        "en_amy_low": ["en_US-amy-low.onnx",
                       "en_US-amy-low.onnx.json",
                       "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/amy/low/"],
        "en_ryan_medium": ["en_US-ryan-medium.onnx",
                           "en_US-ryan-medium.onnx.json",
                           "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/ryan/medium/"],
        "en_ryan_low": ["en_US-ryan-low.onnx",
                        "en_US-ryan-low.onnx.json",
                        "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/ryan/low/"],
        "en_kusal": ["en_US-kusal-medium.onnx",
                     "en_US-kusal-medium.onnx.json",
                     "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/kusal/medium/"],
        "en_lessac": ["en_US-lessac-medium.onnx",
                      "en_US-lessac-medium.onnx.json",
                      "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/lessac/medium/"],
        "en_lessac_low": ["en_US-lessac-low.onnx",
                          "en_US-lessac-low.onnx.json",
                          "https://huggingface.co/mychen76/piper-voices/resolve/main/en/en_US/lessac/low/"],
        "ru": ["ru_RU-irina-medium.onnx",
               "ru_RU-irina-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/ru/ru_RU/irina/medium/"],
        "zh": ["zh_CN-huayan-medium.onnx",
               "zh_CN-huayan-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/zh/zh_CN/huayan/medium/"],
        "fr": ["fr_FR-siwis-medium.onnx",
               "fr_FR-siwis-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/fr/fr_FR/siwis/medium/"],
        "uk": ["uk_UA-ukrainian_tts-medium.onnx",
               "uk_UA-ukrainian_tts-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/uk/uk_UA/ukrainian_tts/medium/"],
        "es": ["es_MX-ald-medium.onnx",
               "es_MX-ald-medium.onnx.json",
               "https://huggingface.co/mychen76/piper-voices/resolve/main/es/es_MX/ald/medium/"]
    }
    try:
        voice_config = voice_map[spoken_language]
    except Exception as e:
        # fallback to english
        voice_config = None

    if voice_config is None:
        voice_config = voice_map["en"]
        voice_onnx_file = local_voice_path+voice_config[0]
    else:
        voice_config = voice_map[spoken_language]
        voice_onnx_file = local_voice_path+voice_config[0]
        voice_json_file = local_voice_path+voice_config[1]
    # attempt download file if not exist locally
    if not os.path.isfile(voice_onnx_file):
        print("download voice files: ", voice_onnx_file, " | ", voice_json_file)
        voice_onnx_url = voice_config[2]+voice_config[0]
        voice_json_url = voice_config[2]+voice_config[1]
        voice_onnx_file = download_file(voice_onnx_url, local_voice_path)
        voice_json_file = download_file(voice_json_url, local_voice_path)
    return voice_onnx_file, voice_json_file


def convert_text_to_speech(text: str, output_voice: str = "en_us"):
    global _local_voice_path
    t0 = time.perf_counter()
    output_voice_lang = "en" if output_voice is None else output_voice
    # attempt retrieve from cache file
    if output_voice_lang in cache_voices_files.keys():
        output_voice_file = cache_voices_files[output_voice_lang]
        voice_onnx_file = output_voice_file[0]
        voice_json_file = output_voice_file[1]
    else:
        voice_onnx_file, voice_json_file = get_voice_model_file(
            _local_voice_path, output_voice_lang)
        cache_voices_files[output_voice_lang] = [
            voice_onnx_file, voice_json_file]

    # print("Found voice files: ", voice_onnx_file, " / ", voice_json_file)
    if voice_onnx_file is None:
        voice_onnx_file = DEFAULT_ESPEAK_VOICE_MODEL
        print(
            f"voice files not found for voice:{output_voice} - fallback to english.")
    if voice_onnx_file in cache_voices_models.keys():
        voice_synthesize = cache_voices_models[voice_onnx_file]
    else:
        voice_synthesize = load_speech_synthesizer_model(voice_onnx_file)
        cache_voices_models[voice_onnx_file] = voice_synthesize
        logger.info(f"loaded speech synthesizer model:{voice_onnx_file}")
    audio_norm, sample_rate = voice_synthesize(text)
    return {
        'data': audio_norm.tolist(),
        'sample-rate': sample_rate,
        'inference':  (time.perf_counter() - t0),
    }


def write_audio_file(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    voice = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    voice.export(f, format="mp3", bitrate="320k")

def slice_text_into_chunks(input_text:str, chunk_size:int=130):
    if len(input_text)<chunk_size:
        return [input_text]
    else:
        K = int(len(input_text)/chunk_size)    
        logger.info(f"The original string is: {str(input_text)} size of {len(input_text)} in chunks: {K}")
        chnk_len = len(input_text) // K
        result = []
        for idx in range(0, len(input_text), chnk_len):
            result.append(input_text[idx : idx + chnk_len])
        logger.debug(f"The K chunked list {str(result)}") 
        return result

def espeak(sd, text: str, output_voice: str = "en"):
    text_chunks = slice_text_into_chunks(text)    
    for chunk in text_chunks:    
        response = convert_text_to_speech(chunk, output_voice)
        npa = np.asarray(response['data'], dtype=np.int16)
        sd.play(npa, response['sample-rate'], blocking=True)
        sd.wait()

def speech_to_file(text, out_file, output_voice):
    response = convert_text_to_speech(text, output_voice=output_voice)
    npa = np.asarray(response['data'], dtype=np.int16)
    write_audio_file(out_file, sr=response['sample-rate'], x=npa)
    return out_file

def text_to_speech(text: str, output_voice: str = "en", mute=False, autoplay=False):
    #logger.debug(f"text_to_speech: {text}")
    if mute or text is None or len(text.strip()) == 0:
        return None
    if output_voice is None:
        output_voice = "en"
    out_file = get_speaker_audio_file()
    try:
        if isinstance(output_voice, str):
            outfile = speech_to_file(text, out_file, output_voice)
        else:
            file = speech_to_file(text, out_file, output_voice.value)
        if autoplay:
            espeak(sd, text, output_voice=output_voice)
    except Exception as e:
        # retry default voice if non-english voice file failed
        print("text_to_speech error: ", e)
        logger.error("text_to_speech error")
    return out_file

# def text_to_speech_gtts(text, autoplay=False):
#     print("text_to_speech: ", text)
#     out_file = 'gtts_text_to_speech.mp3'
#     if text is None or len(text) == 0:
#         return None
#     # Initialize gTTS with the text to convert removed: lang='en',
#     tts = gTTS(text, slow=False, tld='com')
#     tts.save(out_file)
#     if autoplay:
#         audio_bytes = BytesIO()
#         tts.write_to_fp(audio_bytes)
#         audio_bytes.seek(0)
#         audio_data = AudioSegment.from_file(audio_bytes, format="mp3")
#         play(audio_data)
#     return out_file
#     # linux Play the audio file
#     # os.system('afplay ' + speech_file)


def speak_acknowledge():
    import random
    acknowledges = ["Great question!",
                    "Nice,",
                    "Sure thing,",
                    "Yes! It's great/good",
                    "Fantastic! or I like it!",
                    "So true",
                    "Got it.",
                    " Do you mind waiting for a moment while I look up this information.",
                    "Please give me a moment while I look into this for you.",
                    "It will take me just a moment to process your request.",
                    "Oh! I had no idea",
                    "I totally get what you're saying",
                    "Love it.",
                    "Good one"]

    wait_phases = [", please wait...", "one moment...", ". thanks!",
                   ", Please stay online.", " Thank you for waiting."]
    waiting = str(random.choice(wait_phases))

    ack_text = str(random.choice(acknowledges))+waiting
    print("speak_acknowledge: ", ack_text)
    text_to_speech(text=ack_text, output_voice="en", autoplay=True)


def speak_wait():
    import random
    acknowledges = ["okay! please wait...",
                    "got it!, one moment...",
                    "certainly!, one moment...",
                    "sure, one moment...",
                    "process your request.",
                    "working on it."]
    ack_text = str(random.choice(acknowledges))
    print("speak_wait: ", ack_text)
    text_to_speech(text=ack_text, output_voice="en", autoplay=True)


def speak_done():
    import random
    acknowledges = ["all done!, please check.",
                    "process complete!..., please check",
                    "finish processing!, please check",
                    "completed your request! , please check"
                    ]
    ack_text = str(random.choice(acknowledges))
    # print("speak_done: ", ack_text)
    text_to_speech(text=ack_text, output_voice="en", autoplay=True)


def speak_instruction(instruction: str, output_voice: str = "en"):
    logger.debug(f"speak_instruction: {instruction}")
    text_to_speech(text=instruction, output_voice=output_voice, autoplay=True)
