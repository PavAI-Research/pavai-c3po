from pavai.shared.fileutil import download_file
from pavai.shared.styletts2 import (librispeech, librispeech_v2, librispeech_v3, compute_style,
                                    test_libris_speech, test_libris_speech_emotions, test_libris_speech_longspeech)
from pavai.shared.styletts2 import (
    ljspeech, ljspeech_v2, test_lj_speech, test_lj_speech_v2)
import gc
import torch
import cleantext
from pavai.shared.audio.tts_piper import load_speech_synthesizer, load_speech_synthesizer_model
from pavai.shared.styletts2.download_models import get_styletts2_model_files
from pydub.playback import play
from pydub import AudioSegment
import pydub
from gtts import gTTS
import sounddevice as sd
import numpy as np
from pathlib import Path
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from io import BytesIO
import os
import sys
import time
import json
import logging
import traceback
from rich import print, pretty, console
from rich.logging import RichHandler
from rich.pretty import (Pretty, pprint)
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)

# import functools
# import requests
# import shutil
# from rich.panel import Panel

pretty.install()
# from os.path import dirname, join, abspath
# sys.path.insert(0, abspath(join(dirname(__file__), '../shared')))

# stylett2
# from styletts2 import  ljspeech
# ljspeech(text=text,device="cuda")

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

try:
    with open('resources/config/reference_voices.json') as handle:
        reference_voices = json.loads(handle.read())
        print(reference_voices)
except Exception as e:
    print(e)
    raise ValueError("Missing reference voices config file. please check!")

# styleTTs2 reference voices


def lookup_voice(name: str = "Jane"):
    global reference_voices
    if reference_voices is None:
        reference_voices = {
            "ryan": "resources/models/styletts2/reference_audio/Ryan.wav",
            "jane": "resources/models/styletts2/reference_audio/Jane.wav",
            "me1": "resources/models/styletts2/reference_audio/Me1.wav",
            "me2": "resources/models/styletts2/reference_audio/Me2.wav",
            "me3": "resources/models/styletts2/reference_audio/Me3.wav",
            "vinay": "resources/models/styletts2/reference_audio/Vinay.wav",
            "nima": "resources/models/styletts2/reference_audio/Nima.wav",
            "yinghao": "resources/models/styletts2/reference_audio/Yinghao.wav",
            "keith": "resources/models/styletts2/reference_audio/Keith.wav",
            "may": "resources/models/styletts2/reference_audio/May.wav",
            "anthony": "resources/models/styletts2/reference_audio/anthony.wav",
            "c3p013": "resources/models/styletts2/reference_audio/c3p013.wav",
            "c3p0voice8": "resources/models/styletts2/reference_audio/c3p0_voice8.wav",
            "c3p0voice13": "resources/models/styletts2/reference_audio/c3p0_voice13.wav",
            "c3p0voice1": "resources/models/styletts2/reference_audio/c3p0_voice1.wav"
        }
    if name in reference_voices.keys():
        voice_path = reference_voices[name.lower()]
        voice = compute_style(voice_path)
    else:
        print(f" Error Missing voice file {name}, fallback to default")
        name = "Jane"
        voice_path = reference_voices[name.lower()]
        voice = compute_style(voice_path)
    return voice

# librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0.3, beta=0.5, diffusion_steps=10)


# cached data
cache_voices_files = {}
cache_voices_models = {}


def get_speaker_audio_file(workspace_temp: str = "workspace/temp") -> str:
    Path.mkdir(workspace_temp, exist_ok=True)
    # if not os.path.exists(workspace_temp):
    #     os.mkdir(workspace_temp)
    return workspace_temp+"/espeak_text_to_speech.mp3"


def slice_text_into_chunks(input_text: str, chunk_size: int = 100):
    if len(input_text) < chunk_size:
        return [input_text]
    else:
        K = int(len(input_text)/100)
        logger.info(
            f"The original string is: {str(input_text)} size of {len(input_text)} in chunks: {K}")
        # compute chunk length
        chnk_len = len(input_text) // K
        result = []
        for idx in range(0, len(input_text), chnk_len):
            result.append(input_text[idx: idx + chnk_len])
        logger.info(f"The K chunked list {str(result)}")
        return result


def free_memory(to_delete: list = None, debug: bool = False):
    import gc
    import torch
    import inspect
    # print("Before:")
    # memory_stats()
    gc.collect()
    torch.cuda.empty_cache()
    # print("After:")
    memory_stats()


def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_cached()/1024**2)


def librispeak_v1(text: str, compute_style: str = "jane", chunk_size=350, output_voice_lang: str = "en", autoplay: bool = True):
    ref_voice = lookup_voice(compute_style)
    result = librispeech(text=text, compute_style=ref_voice, alpha=0.3, beta=0.7, autoplay=autoplay)
    free_memory
    return result


def librispeak_v2(text: str, compute_style: str = "jane", emotion: str = None, chunk_size=350, output_voice_lang: str = "en", autoplay: bool = True):
    ref_voice = lookup_voice(compute_style)
    result= librispeech_v2(text=text, compute_style=ref_voice, emotion=emotion, autoplay=autoplay)
    free_memory
    return result

def librispeak_v3(text: str, compute_style: str = "jane", alpha=0.3, beta=0.7, chunk_size=350, output_voice_lang: str = "en", autoplay: bool = True):
    ref_voice = lookup_voice(compute_style)
    result = librispeech_v3(text=text, compute_style=ref_voice, alpha=0.3, beta=0.7, autoplay=autoplay)
    free_memory
    return result

def librispeak(text: str, compute_style: str = "jane", emotion: str = None, chunk_size: int = 450, autoplay: bool = True, output_voice_lang: str = "en"):
    text = cleantext.clean(text, extra_spaces=True, punct=True)
    text = text+"."
    wavs = []
    try:
        if len(text) > chunk_size:
            text_chunks = slice_text_into_chunks(
                input_text=text, chunk_size=chunk_size)
            for chunks in text_chunks:
                wav = librispeak_v1(
                    text=chunks, compute_style=compute_style, autoplay=autoplay)
                wavs.append(wav)
        else:
            if emotion is not None:
                wav = librispeak_v2(
                    text=text, compute_style=compute_style, emotion=emotion, autoplay=autoplay)
                wavs.append(wav)
            else:
                wav = librispeak_v1(
                    text=text, compute_style=compute_style, autoplay=autoplay)
                wavs.append(wav)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        free_memory()
        # fallback to system default speaker
        wavs = system_speak(text, autoplay=autoplay)
    finally:
        free_memory()
    return wavs


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


def espeak(sd, text: str, output_voice: str = "en"):
    if system_config["GLOBAL_TTS"] == "LIBRETTS":
        librispeak(text=text, compute_style="jane",
                   emotion="happy", autoplay=True)
    else:
        response = convert_text_to_speech(text, output_voice)
        npa = np.asarray(response['data'], dtype=np.int16)
        sd.play(npa, response['sample-rate'], blocking=True)
        sd.wait()


def speech_to_file(text, out_file, output_voice):
    response = convert_text_to_speech(text, output_voice=output_voice)
    npa = np.asarray(response['data'], dtype=np.int16)
    write_audio_file(out_file, sr=response['sample-rate'], x=npa)
    return out_file


def text_to_speech(text: str, output_voice: str = "en", mute=False, autoplay=False):
    # logger.info(f"text_to_speech: {text}")
    if mute or text is None or len(text.strip()) == 0:
        return None
    if output_voice is None:
        output_voice = "en"
    out_file = get_speaker_audio_file()  # 'espeak_text_to_speech.mp3'
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


def text_to_speech_gtts(text, autoplay=False):
    print("text_to_speech: ", text)
    out_file = 'gtts_text_to_speech.mp3'
    if text is None or len(text) == 0:
        return None
    # Initialize gTTS with the text to convert removed: lang='en',
    tts = gTTS(text, slow=False, tld='com')
    tts.save(out_file)
    if autoplay:
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        audio_data = AudioSegment.from_file(audio_bytes, format="mp3")
        play(audio_data)
    return out_file
    # linux Play the audio file
    # os.system('afplay ' + speech_file)


def speak_acknowledge():
    import random
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
    wait_phases = [", please wait", " one moment", " thanks!",
                   ", Please stay online", " Thank you!"]
    waiting = str(random.choice(wait_phases))
    ack_text = str(random.choice(acknowledges))+waiting
    print("speak_acknowledge: ", ack_text)
    # text_to_speech(text=ack_text, output_voice="en", autoplay=True)
    system_speak(text=ack_text, autoplay=True)


def speak_wait():
    import random
    acknowledges = ["okay! please wait.",
                    "got it!, one moment.",
                    "certainly!, one moment.",
                    "sure, one moment.",
                    "process your request.",
                    "working on it."]
    ack_text = str(random.choice(acknowledges))
    print("speak_wait: ", ack_text)
    # text_to_speech(text=ack_text, output_voice="en", autoplay=True)
    system_speak(text=ack_text, autoplay=True)


def speak_done():
    import random
    acknowledges = ["all done, please check.",
                    "process complete, please check",
                    "finish processing, please check",
                    "completed your request, please check"
                    ]
    ack_text = str(random.choice(acknowledges))
    # print("speak_done: ", ack_text)
    # text_to_speech(text=ack_text, output_voice="en", autoplay=True)
    system_speak(text=ack_text, autoplay=True)


def speak_instruction(instruction: str, output_voice: str = "en"):
    logger.info(f"speak_instruction: {instruction}")
    # text_to_speech(text=instruction, output_voice=output_voice, autoplay=True)
    system_speak(text=instruction, autoplay=True)


def system_speak(text: str, autoplay: bool = True) -> list:
    return ljspeech_v2(text=text, autoplay=autoplay)


"""MAIN"""
if __name__ == "__main__":
    # test_libris_speech()

    sample_text = """
    Thank you for this! I have one more question if anyone can bite: I am trying to take the average of the first elements in these datapoints(i.e. datapoints[0][0]). Just to list them, I tried doing datapoints[0:5][0] but all I get is the first datapoint with both elements as opposed to wanting to get the first 5 datapoints containing only the first element. Is there a way to do this?
    """
    sample_text2="""
    Hi there! I'm programmed to assist you with your requests. How may I help you today? Feel free to ask any question or provide any challenge, and I'll do my best to provide the most accurate and helpful response possible. Let's begin!
    """
    print("text size:", len(sample_text2))
    # system_speak(text=sample_text)
    librispeak(text=sample_text2, compute_style="jane", autoplay=True)
    librispeak(text=sample_text2, compute_style="jane",
               emotion="happy", autoplay=True)

    # librispeak_v1(text=sample_text)
    # librispeak_v2(text=sample_text, emotion="happy", autoplay=True)
    # librispeak_v3(text=sample_text, autoplay=True)
    # test_lj_speech()
    # test_lj_speech_v2()
    # test_libris_speech()
    # test_libris_speech_emotions()
    # test_libris_speech_longspeech(1)

    # librispeak("hello world")

# def download_file(url, local_path: str = None):
#     local_filename = url.split('/')[-1]
#     if local_path is not None:
#         local_filename = local_path+local_filename
#     with requests.get(url, stream=True) as r:
#         with open(local_filename, 'wb') as f:
#             shutil.copyfileobj(r.raw, f)
#     return local_filename

# # @functools.lru_cache
# # def load_speech_synthesize_model(voice_model_name):
# #     global _voice_synthesize
# #     _voice_synthesize = load_speech_synthesizer(_voice_model_name)
# #     return _voice_synthesize

# def get_styletts2_model_files(local_voice_path: str="resources/models/styletts2", remote_folder: str = "https://huggingface.co/mychen76") -> str:
#     """download styletts model file from remote location"""
#     try:
#         LibriTTS=local_voice_path+"/LibriTTS"
#         if not os.path.exists(LibriTTS):
#             os.mkdir(LibriTTS)
#             LibriTTS_model_config_url=f"{remote_folder}/styletts2/resolve/main/Models/LibriTTS/config.yml"
#             download_file(url=LibriTTS_model_config_url,local_path=LibriTTS)
#             LibriTTS_model_bin=f"{remote_folder}/styletts2/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth"
#             download_file(url=LibriTTS_model_bin,local_path=LibriTTS)
#             print(f"styletts2_model downloaded {LibriTTS}")
#         else:
#             print(f"styletts2_model already exist: {LibriTTS}")

#         LJSpeech=local_voice_path+"/LJSpeech"
#         if not os.path.exists(LJSpeech):
#             os.mkdir(LJSpeech)
#             LJSpeech_model_config_url=f"{remote_folder}/styletts2/resolve/main/Models/LJSpeech/config.yml"
#             download_file(url=LJSpeech_model_config_url,local_path=LJSpeech)
#             LJSpeech_model_bin=f"{remote_folder}/styletts2/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth"
#             download_file(url=LJSpeech_model_bin,local_path=LJSpeech)
#             print(f"styletts2_model downloaded {LJSpeech}")
#         else:
#             print(f"styletts2_model already exist: {LJSpeech}")
#     except Exception as e:
#         print("Exception occured ",e.args)
#         print(traceback.format_exc())
#         raise Exception("Failed to download styletts2 model files!")
