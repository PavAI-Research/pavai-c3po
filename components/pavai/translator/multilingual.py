## m4s seamless communication
from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import gradio as gr
import numpy as np
import torch
import torchaudio
from transformers import SeamlessM4Tv2ForSpeechToSpeech
from transformers import AutoProcessor
from typing import Any, Dict
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
import pavai.shared.system_types as system_types
from pavai.translator.lang_list import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2ST_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)

SPEAKER_ID = 7
AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 360  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
modelv2=None
processorv2=None

class SeamlessM4T(system_types.Singleton):

    def __init__(self, 
                 default_target_lang:str="French", 
                 default_speaker_id:int=7,
                 max_input_audio_length:int=3600, 
                 model_repo:str = "facebook/seamless-m4t-v2-large",
                 cache_dir:str="resources/models/translator"):
        self._model_repo = model_repo
        self._cache_dir = cache_dir
        self.CUDA_OUT_OF_MEMORY=False
        self.DEFAULT_TARGET_LANGUAGE=default_target_lang
        self.MAX_INPUT_AUDIO_LENGTH=max_input_audio_length
        self.AUDIO_SAMPLE_RATE=16000.0  ## mono
        self.SPEAKER_ID=default_speaker_id

    def download_model(self,model_repo:str = "facebook/seamless-m4t-v2-large",filename:str=None):
        # File to download
        logger.info("WARNING: downloading a very large translator model (25 GB)...\n")
        logger.info("---------may take sometime depends on your internet connection")
        self._local_model_path=snapshot_download(repo_id=model_repo,  cache_dir=self._cache_dir)
        return self._local_model_path

    def load_model(self,mode_name_or_path:str="facebook/seamless-m4t-v2-large", device:str="cpu"):
        gr.Info("Getting translation model...")
        self._processorv2 = AutoProcessor.from_pretrained(mode_name_or_path, cache_dir=self._cache_dir)
        self._modelv2 = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(mode_name_or_path, cache_dir=self._cache_dir)
        self._modelv2 = self._modelv2.to(device)
        return self._modelv2, self._processorv2

    def update_value(self,val):
        self.SPEAKER_ID = val
        logger.info(f"updated speaker {self.SPEAKER_ID}")

    def transcribe(self,
        input_audio_array: list,
        target_lang="spa",
        speaker_id: int = 6,
        text_num_beams: int = 6,
        speech_temperature: float = 0.6,    
        device:str="cpu",    
    ):
        gr.Info("Processing translation, please wait!")
        self._modelv2, self._processorv2 = self.load_model(device=device)
        audio_inputs = self._processorv2(audios=input_audio_array, return_tensors="pt", sampling_rate=self._modelv2.config.sampling_rate).to(device)

        ## generate
        gr.Info("Generating translation voice and text. Please wait!")    
        outputv2 = self._modelv2.generate(
            **audio_inputs,
            return_intermediate_token_ids=True,
            tgt_lang=target_lang,
            speaker_id=speaker_id,
            text_num_beams=text_num_beams,
            speech_do_sample=True,
            speech_temperature=speech_temperature
        )
        ## text
        text_tokens = outputv2[2]
        out_texts = self._processorv2.decode(text_tokens.tolist()[0], skip_special_tokens=True)
        logger.info(f"TRANSLATION: {out_texts}")
        ## audio
        out_audios = outputv2[0].cpu().numpy().squeeze()
        sample_rate = self._modelv2.config.sampling_rate
        return out_texts, out_audios, sample_rate

    def preprocess_audio(self,input_audio_filepath: str) -> None:
        """process audio file"""
        arr, org_sr = torchaudio.load(input_audio_filepath)
        new_arr = torchaudio.functional.resample(
            arr, orig_freq=org_sr, new_freq=self.AUDIO_SAMPLE_RATE
        )
        max_length = int(self.MAX_INPUT_AUDIO_LENGTH * self.AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(
                f"Input audio is too long. Only the first {self.MAX_INPUT_AUDIO_LENGTH} seconds is used."
            )
        return new_arr

    def run_s2st(self,
        input_audio_filepath: str,
        source_language: str,
        target_language: str,
        speaker_id: int = 7,
    ) -> tuple[tuple[int, np.ndarray] | None, str]:
        if input_audio_filepath is None:
            logger.warn("missing input audio file")
            return 
        audio_array = self.preprocess_audio(input_audio_filepath)
        source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
        target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
        logger.info(f"source lang: {source_language_code}")
        logger.info(f"target lang: {target_language_code}")

        ## determine optimal device to use
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.CUDA_OUT_OF_MEMORY:
            device="cpu"
        gr.Warning(f"Attempt use optimal device: {device} for translation!")  
        out_texts=""  
        try:
            out_texts, out_audios, sample_rate = self.transcribe(
                input_audio_array=audio_array,
                speaker_id=speaker_id,
                target_lang=target_language_code,
                device=device,        
            )
        except Exception as e:
            logger.error("Exception occurred ", e.args)
            if "CUDA out of memory" in str(e.args):
                self.CUDA_OUT_OF_MEMORY=True
                gr.Warning("Retry with device: CPU only, please wait!")
                out_texts, out_audios, sample_rate = self.transcribe(
                input_audio_array=audio_array,
                speaker_id=speaker_id,
                target_lang=target_language_code,
                device="cpu",        
            )

        out_texts = str(out_texts)
        return (int(sample_rate), out_audios), out_texts
