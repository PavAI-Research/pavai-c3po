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
import pavai.translator.lang_list as lang_list
# import os
# import pathlib

SPEAKER_ID = 7

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 360  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

modelv2=None
processorv2=None

def load_model(mode_name_or_path:str="facebook/seamless-m4t-v2-large", device:str="cpu"):
    global modelv2 
    global processorv2

    gr.Info("Getting translation model...")
    processorv2 = AutoProcessor.from_pretrained(mode_name_or_path)
    modelv2 = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(mode_name_or_path)
    modelv2 = modelv2.to(device)
    return modelv2, processorv2

def update_value(val):
    global SPEAKER_ID
    SPEAKER_ID = val
    print("updated speaker ", SPEAKER_ID)

def transcribe(
    input_audio_array: list,
    target_lang="spa",
    speaker_id: int = 6,
    text_num_beams: int = 6,
    speech_temperature: float = 0.6,    
    device:str="cpu",    
):
    gr.Info("Processing translation, please wait!")
    modelv2, processorv2 = load_model(device=device)
    audio_inputs = processorv2(audios=input_audio_array, return_tensors="pt", sampling_rate=modelv2.config.sampling_rate).to(device)

    ## generate
    gr.Info("Generating translation voice and text. Please wait!")    
    outputv2 = modelv2.generate(
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
    out_texts = processorv2.decode(text_tokens.tolist()[0], skip_special_tokens=True)
    logger.info(f"TRANSLATION: {out_texts}")
    ## audio
    out_audios = outputv2[0].cpu().numpy().squeeze()
    sample_rate = modelv2.config.sampling_rate
    return out_texts, out_audios, sample_rate

def preprocess_audio(input_audio_filepath: str) -> None:
    arr, org_sr = torchaudio.load(input_audio_filepath)
    new_arr = torchaudio.functional.resample(
        arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE
    )
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(
            f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used."
        )
    # torchaudio.save(input_audio_filepath, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    return new_arr

CUDA_OUT_OF_MEMORY=False
def run_s2st(
    input_audio_filepath: str,
    source_language: str,
    target_language: str,
    speaker_id: int = 7,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    global CUDA_OUT_OF_MEMORY 

    audio_array = preprocess_audio(input_audio_filepath)
    source_language_code = lang_list.LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = lang_list.LANGUAGE_NAME_TO_CODE[target_language]
    logger.info(f"source lang: {source_language_code}")
    logger.info(f"target lang: {target_language_code}")

    ## determine optimal device to use
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if CUDA_OUT_OF_MEMORY:
        device="cpu"
    gr.Warning(f"Attempt use optimal device: {device} for translation!")    
    out_texts=None
    sample_rate=24000
    out_audios=None
    try:
        out_texts, out_audios, sample_rate = transcribe(
            input_audio_array=audio_array,
            speaker_id=speaker_id,
            target_lang=target_language_code,
            device=device,        
        )
    except Exception as e:
        logger.error(f"Exception occurred {e.args}")
        if "CUDA out of memory" in str(e.args):
            CUDA_OUT_OF_MEMORY=True
            gr.Warning("Retry with device: CPU only, please wait!")
            out_texts, out_audios, sample_rate = transcribe(
            input_audio_array=audio_array,
            speaker_id=speaker_id,
            target_lang=target_language_code,
            device="cpu",        
        )

    out_texts = str(out_texts)
    return (int(sample_rate), out_audios), out_texts
