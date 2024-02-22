#!/usr/bin/env python
"""speech_synthesizer.py"""
import json
import logging
import numpy as np
import onnxruntime
from espeak_phonemizer import Phonemizer
## pip install espeak_phonemizer
from functools import partial
import functools
from pathlib import Path
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Union
from ..load_config import (load_config_file, _BOS,_EOS,_PAD)

_FILE = Path(__file__)
_DIR = _FILE.parent

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
logger = logging.getLogger(__name__)

model = None
synthesize = None

@dataclass
class PiperConfig:
    num_symbols: int
    num_speakers: int  #value=1
    sample_rate: int   #value=1
    espeak_voice: str
    length_scale: float
    noise_scale: float  #value=0.667
    noise_w: float      #value=1
    phoneme_id_map: Mapping[str, Sequence[int]]

class Piper:
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ):
        if config_path is None:
            config_path = f"{model_path}.json"

        self.config = get_model_config(config_path)
        self.phonemizer = Phonemizer(self.config.espeak_voice)
        # print("config path: ",model_path)
        logging.debug(f"config path: {model_path}")        
        # Session options        
        sess_options = onnxruntime.SessionOptions()
        # use the default graph optimization and just to increase the severity of the logger to filter out noisy warnings:       
        # onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        # onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL        
        self.model = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            ## comment out why cuda is no longer exist
            providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider", "AzureExecutionProvider"],    
            ## use cpu only for non-cuda GPU
            #providers=["CPUExecutionProvider"],        
        )

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
        """Synthesize WAV audio from text."""
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phonemes_str = self.phonemizer.phonemize(text)
        phonemes = [_BOS] + list(phonemes_str)
        phoneme_ids: List[int] = []

        for phoneme in phonemes:
            phoneme_ids.extend(self.config.phoneme_id_map[phoneme])
            phoneme_ids.extend(self.config.phoneme_id_map[_PAD])

        phoneme_ids.extend(self.config.phoneme_id_map[_EOS])

        phoneme_ids_array = np.expand_dims(
            np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array(
            [phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )
        # if (self.config.num_speakers > 1) and (speaker_id is not None):
        #     # Default speaker
        #     speaker_id = 0
        sid = None
        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)

        # Synthesize through Onnx
        audio = self.model.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))
        audio = audio_float_to_int16(audio.squeeze())
        return audio, self.config.sample_rate

def get_model_config(config_path: Union[str, Path]) -> PiperConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)
        inference = config_dict.get("inference", {})

        return PiperConfig(
            num_symbols=config_dict["num_symbols"],
            num_speakers=config_dict["num_speakers"],
            sample_rate=config_dict["audio"]["sample_rate"],
            espeak_voice=config_dict["espeak"]["voice"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            phoneme_id_map=config_dict["phoneme_id_map"],
        )

def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm

def load_speech_synthesizer(model_path):
    global synthesize
    if synthesize:
       return synthesize
    logging.debug(f"load_speech_synthesizer: {model_path}")    
    synthesize=load_speech_synthesizer_model(model_path)
    return synthesize

def load_speech_synthesizer_model(model_path):
    logging.debug(f"realtime_speech_synthesize - model_path: {model_path}")    
    speaker_id = None 
    voice = Piper(model_path=model_path, config_path=f"{model_path}.json")
    synthesize = partial(
        voice.synthesize,
        speaker_id=speaker_id,
        length_scale=None,
        noise_scale=0.5,
        noise_w=0.2,)
    logging.debug("loaded Voice Synthesizer.")
    return synthesize

def create_speech_synthesize(model_path):
    logging.debug(f"model_path: {model_path}")    
    speaker_id = None 
    voice = Piper(model_path=model_path, config_path=f"{model_path}.json")
    synthesize = partial(
        voice.synthesize,
        speaker_id=speaker_id,
        length_scale=None,
        noise_scale=0.5,
        noise_w=0.2,)
    logging.debug("Voice synthesize loaded.")
    return synthesize
