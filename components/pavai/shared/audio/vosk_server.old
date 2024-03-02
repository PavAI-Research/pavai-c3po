from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
import io
import json
from pydub import AudioSegment
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union
import numpy as np
import onnxruntime
from espeak_phonemizer import Phonemizer
from functools import partial
import logging
import logging.config
import time
## get configs
from ..load_config import (load_config_file, load_voice_config,load_llm_config, load_provider_config, 
                         _BOS,_EOS,_PAD)
from .tts_piper import load_speech_synthesize,create_speech_synthesize

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

_FILE = Path(__file__)
_DIR = _FILE.parent

FORMAT = "%(levelname)s:%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
model = None
config_file=load_config_file()
model_name = load_voice_config()["model"]
synthesize = load_speech_synthesize(model_name)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_name
    global synthesize    
    logging.debug("Run at startup!")
    if model_name is None:
        model_name = load_voice_config()["model"]
    if synthesize is None:
        synthesize = load_speech_synthesize(model_name)
        logging.debug(f'voice synthesize created: {model_name}')
    yield
    logging.debug("Run on shutdown.")    

def check_file_exist(path):
    import os.path
    check_file = os.path.isfile(path)
    return check_file

## ----------------------------
## API 
## ----------------------------
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "voice synthesizer API is live!"}

@app.post("/predict")
async def predict(text: str,language:str="en_US",speaker_model:str="./models/voices/en_US-amy-medium.onnx"):
    global model_name
    global synthesize    
    print("text:", text)
    print("voice_model:", speaker_model)    
    t0 = time.time()
    if check_file_exist(speaker_model):
        model_name = speaker_model      
    if model_name is None:
        model_name = load_voice_config()["model"]            
    #if synthesize is None:
    ##synthesize = load_speech_synthesize(model_name)
    synthesize = create_speech_synthesize(model_name)    
    print("synthesize:", model_name)                               
    audio_norm, sample_rate = synthesize(text)
    t1 = time.time()
    tt=round(t1-t0,2)
    print("took ",tt)
    return {
        'data': audio_norm.tolist(),
        'sample-rate': sample_rate,
        'inference':  tt,
    }

if __name__ == '__main__':
    uvicorn.run("__main__:app", host='0.0.0.0', port=7000)
