""" vosk_client.py """
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
from rich.logging import RichHandler
logging.basicConfig(level=logging.WARN, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
#pretty.install()
import warnings 
warnings.filterwarnings("ignore")

import requests
import numpy as np
from nltk.tokenize import sent_tokenize
import sounddevice as sd
from ..load_config import load_voice_config
from .voices_piper import espeak

#from rich import print, pretty
#import json
#pretty.install()

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

## global 
_config = None
_api_url = None 
_language = None 
_speaker_model = None 

# def local_tts(sd, text: str, output_voice: str = "en"):
#     espeak(sd,text,output_voice)

vosk_params = {
    "sentence":"hello",
    "api_url":"http://192.168.0.29:7000/predict",
    "language": "en_US",    
    "speaker_model":"./models/voices/en_US-ryan-medium.onnx"
}

def remote_tts(api_url:str,
                           sentence:str,
                           language:str="en_US",
                           speaker_model:str="./models/voices/en_US-amy-medium.onnx"):
    params = {"text": sentence, 
              "language": language,
              "speaker_model":speaker_model}
    response = requests.post(api_url, 
                         params=params, 
                         headers={"accept": "application/json"})
    #print(f"convert_to_speech response: {response.status_code} / {response.reason}")
    data = response.json()
    #print("took ",data["inference"])
    return data 

# def speak_text(sd,text:str,output_voice:str="en"):
#     espeak(sd,text,output_voice)    
    # data=remote_tts(api_url,sentence,language,speaker_model)
    # npa = np.asarray(data['data'], dtype=np.int16)
    # sd.play(npa, data['sample-rate'], blocking=True)
    # #sd.wait()

def slice_text_into_chunks(input_text:str, chunk_size:int=150):
    if len(input_text)<chunk_size:
        return [input_text]
    else:
        K = int(len(input_text)/chunk_size)    
        #logger.info(f"The original string is: {str(input_text)} size of {len(input_text)} in chunks: {K}")
        chnk_len = len(input_text) // K
        result = []
        for idx in range(0, len(input_text), chnk_len):
            result.append(input_text[idx : idx + chnk_len])
        #logger.debug(f"The K chunked list {str(result)}") 
        return result

def api_speaker(vosk_params,assistant_response):
    vosk_params["sentence"]=assistant_response
    api_convert_text_to_speech(**vosk_params)    

def ai_speaker(api_url:str,sentence:str,language:str="en_US",
                           speaker_model:str="./models/voices/en_US-amy-medium.onnx"):
    text_chunks = slice_text_into_chunks(sentence)    
    for chunk in text_chunks:        
        data=remote_tts(api_url,chunk,language,speaker_model)
        npa = np.asarray(data['data'], dtype=np.int16)
        sd.play(npa, data['sample-rate'], blocking=True)
        #sd.wait()

def api_convert_text_to_speech(sentence:str,
                   language:str = "en_US",
                   api_url:str="http://192.168.0.29:7000/predict",
                   speaker_model:str="./models/voices/en_US-amy-medium.onnx"):
    global _config
    if _config is None:
        _config = load_voice_config()
    # api_url = _config["api_base"] if _api_url is None else _api_url
    # language = _config["language"] if _language is None else _language
    # speaker_model= _config["model"] if _speaker_model is None else _speaker_model
    ai_speaker(api_url,sentence,language,speaker_model)

# def text_to_speech(sd,sentence:str,language:str = None):
#     global _config
#     if _config is None:
#         _config = load_voice_config()
#     api_url = _config["api_base"] if _api_url is None else _api_url
#     language = _config["language"] if _language is None else _language
#     speaker_model= _config["model"] if _speaker_model is None else _speaker_model
#     convert_and_speak(sd,api_url,sentence,language,speaker_model)
    
# if __name__ == '__main__':
#     api_url="http://192.168.0.29:7000/predict"
#     language = "en_US",    
#     speaker_model="./models/voices/en_US-amy-medium.onnx"    
#     text="Vosk is a speech recognition toolkit."
#     convert_text_to_speech(sentence=text,
#                            language=language,
#                            api_url=api_url,
#                            speaker_model=speaker_model)    
#     #data=convert_text_to_speech(api_url,sentence)
#     #npa = np.asarray(data['data'], dtype=np.int16)
#     #sd.play(npa, data['sample-rate'], blocking=True)    
#     speaker_model="./models/voices/en_US-lessac-medium.onnx"    
#     convert_text_to_speech(sentence=text,
#                            language=language,
#                            api_url=api_url,
#                            speaker_model=speaker_model)
#     speaker_model="./models/voices/en_US-ryan-medium.onnx"        
#     convert_text_to_speech(sentence=text,
#                            language=language,
#                            api_url=api_url,
#                            speaker_model=speaker_model)
