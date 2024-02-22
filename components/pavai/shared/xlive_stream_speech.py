import gradio as gr
import time
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.audio_utils import ffmpeg_read
import numpy as np
import torch
# pip install faster-whisper==0.10.0
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model, format_timestamp, get_logger
from pytube import YouTube

# -------------------------------------
# Environment
# -------------------------------------
# tested version gradio version: 4.7.1
print(f"use gradio version: {gr.__version__}")
print(f"use torch version: {torch.__version__}")
# pip install gradio==4.7.1

use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()
print(f"use device: {use_device}")    
if use_device == 'cuda' or use_device == 'cuda:0':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('|_ Allocated:', round(
        torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('|_ Reserved :', round(
        torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
print(f"use torch_dtype: {use_torch_dtype}")    
print(f"use_flash_attention_2: {use_flash_attention_2}") 

# -------------------------------------
# whisper
# pip install transformers optimum accelerate
# https://github.com/Vaibhavs10/insanely-fast-whisper
# -------------------------------------
whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
# openai/whisper-base
# openai/whisper-small.en
whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=use_torch_dtype, 
    low_cpu_mem_usage=True, use_safetensors=True, 
    use_flash_attention_2=use_flash_attention_2
)
whisper_transcriber = pipeline("automatic-speech-recognition", 
                               model=whisper_model,
                               tokenizer=whisper_processor.tokenizer,
                               feature_extractor=whisper_processor.feature_extractor,                               
                               device=use_device,
                               torch_dtype=use_torch_dtype,
                               max_new_tokens=128,
                               chunk_length_s=15, #default 15    
                               batch_size=16,     #default 16                           
                               generate_kwargs={"task": "transcribe"},
                               model_kwargs={"use_flash_attention_2": use_flash_attention_2},
                               return_timestamps=True                                                                                                     
                               )

def fn_whisper_transcribe(stream, new_chunk):
    if not isinstance(new_chunk, str):        
        sr, y = new_chunk
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        if stream is not None:
            stream = np.concatenate([stream, y])
        else:
            stream = y
        text = whisper_transcriber({"sampling_rate": sr, "raw": stream})["text"]                        
    else:
        stream = new_chunk      
        print(f"whisper_transcribe file: {new_chunk}")             
        text = whisper_transcriber(stream,return_timestamps=True)["text"] 
        text = text.strip() if text is not None else ""      
    return stream,text

if not use_flash_attention_2:
    # use flash attention from pytorch sdpa 
    # required: pip install optimum=1.14.1
    whisper_model = whisper_model.to_bettertransformer()
    print("use_flash_attention_2: to_bettertransformer")      

# -------------------------------------
# whisper-distilled
# -------------------------------------

distilled_processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
distilled_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "distil-whisper/distil-large-v2", torch_dtype=use_torch_dtype, 
    low_cpu_mem_usage=True, use_safetensors=True, 
    use_flash_attention_2=use_flash_attention_2
)
 
if not use_flash_attention_2:
    # use flash attention from pytorch sdpa 
    # required: pip install optimum=1.14.1
    distilled_model = distilled_model.to_bettertransformer()  
    print("use_flash_attention_2: to_bettertransformer")      

distilled_model.to(use_device)
distilled_transcriber = pipeline(
    "automatic-speech-recognition",
    model=distilled_model,
    tokenizer=distilled_processor.tokenizer,
    feature_extractor=distilled_processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15, #default 15
    batch_size=16,     #default 16
    torch_dtype=use_torch_dtype,
    device=use_device,
    generate_kwargs={"task": "transcribe"},
    model_kwargs={"use_flash_attention_2": use_flash_attention_2},
    return_timestamps=True                                       
)
transcriber_forward = distilled_transcriber._forward

# def distilled_transcribe_file(filename):
#     print (f"Transcribing New file: {filename}")
#     transcription = pipeline(filename, return_timestamps=True)
#     transcription = replace_wordly_with_wardley(transcription)
#     return transcription

def fn_distilled_transcribe(stream, new_chunk):
    text=""
    if not isinstance(new_chunk, str):    
        sr, y = new_chunk
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        if stream is not None:
            stream = np.concatenate([stream, y])
        else:
            stream = y
        text = distilled_transcriber({"sampling_rate": sr, "raw": stream})["text"]            
    else:
        stream = new_chunk      
        print(f"distilled_transcribe file: {new_chunk}")     
        text = distilled_transcriber(stream,return_timestamps=True)["text"] 
        text = text.strip() if text is not None else ""  
    return stream, text

# -------------------------------------
# faster-whisper
# -------------------------------------
FAST_WHISPER_MODEL_SIZE = "large-v3"
FAST_WHISPER_MODEL_DOWNLOAD_ROOT = "whisper_models"
FAST_WHISPER_MODEL_USE_LOCAL_ONLY = True
USE_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
faster_device = "cuda" if use_device=="cuda:0" else use_device
print(f"faster device: {faster_device}")    
print(f"faster compute_type: {USE_COMPUTE_TYPE}")    
faster_transcriber = WhisperModel(
    model_size_or_path=FAST_WHISPER_MODEL_SIZE,
    device=faster_device,
    compute_type=USE_COMPUTE_TYPE,
    cpu_threads=4,num_workers=1,
    download_root=FAST_WHISPER_MODEL_DOWNLOAD_ROOT,
    local_files_only=FAST_WHISPER_MODEL_USE_LOCAL_ONLY)

def fn_faster_transcribe(stream, new_chunk,return_timestamp:bool=False):
    global faster_transcriber
    if not isinstance(new_chunk, str):
        print(f"faster_transcribe numpy")            
        sr, y = new_chunk
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))  
        if (stream is not None) and (not isinstance(stream, str)):
            stream = np.concatenate([stream, y])
        else:
            stream = y
    else:
        stream = new_chunk      
        print(f"faster_transcribe file: {stream}")     
    segments, info = faster_transcriber.transcribe(audio=stream,
                                                   language=None,
                                                   task="transcribe",
                                                   beam_size=5,
                                                   vad_filter=True)
    text = ""
    for segment in segments:
        text = text+segment.text
        if return_timestamp:
            seg = []
            for word in segment.words:   
                txt="[{word.start}:.2fs -> {word.end}:.2fs] {word.word}" 
                seg.append(txt)
            text = ''.join(seg)
    return stream, text

# -------------------------------------
# UTILITIES
# -------------------------------------
def word_count(string):
    return(len(string.strip().split(" ")))

def youtube_download(url):
    try:
        link = YouTube(url)
        filename="downloads/youtube_audio_mp4"        
        source = link.streams.filter(only_audio=True)[0].download(filename)
        return source, None
    except Exception as e:
        gr.Error(f"youtube_download error: {e}")
        return None, None        
# -------------------------------------
# MAIN UI
# -------------------------------------

def fn_transcribe(stream, new_chunk,url):
    print("transcription started...")
    t0 = time.perf_counter()
    if url:
        new_chunk,stream = youtube_download(url)
        print("youtube_download completed.")
    stream, text = fn_faster_transcribe(stream, new_chunk)
    t1 = time.perf_counter()
    took_in_seconds = t1-t0
    words=word_count(text)
    status_msg=f"transcription completed. \nword count:[{words}] | took {took_in_seconds:.2f} seconds"    
    print(status_msg)
    gr.Info(status_msg)
    return stream, text

demo = gr.Interface(
    fn_transcribe,
    inputs=["state", 
            gr.Audio(sources=['microphone', 'upload'], format="mp3", type="filepath"),
            gr.Textbox(label="Paste YouTube link here", 
                       interactive=True)],
    outputs=["state", gr.Textbox(label="Whisper Transcript")],
    live=False,
)

# U.S. House votes to expel George Santos from Congress (4 minutes)
# https://www.youtube.com/watch?v=g0wTNOdcs5I
# Breaking down the George Santos expulsion vote and what happens next (11 minutes)
# https://www.youtube.com/watch?v=c1M4s5Uqr1s
# JUST IN: Dan Goldman Holds Press Briefing On Push To Expel George Santos (26 minutes)
# https://www.youtube.com/watch?v=qc4_H72sEqk

# 福建舰电磁弹射测试
# https://www.youtube.com/watch?v=ir5rA2TmTAk

# demo.launch()

## whisper transcribe   -- okay slower
# issue: incomplete
# jfk=[22] 0.18 s
# hp0=[355] 24.32 seconds
# word count:[619] | took 33.88 seconds
# youtube: https://www.youtube.com/watch?v=qc4_H72sEqk

## distilled transcribe  -- good better
# jfk=[22] 0.18 s
# hp0=[347] 4.5 to 5.7 seconds
# youtube: https://www.youtube.com/watch?v=qc4_H72sEqk
# word count:[3633] | took 31.69 seconds
# 福建舰电磁弹射测试
# word count:[935] | took 13.07 seconds

## faster transcribe  -- good better
# jfk=[22] 0.56 s
# hp0=[359] 21.05 seconds
# filetype: numpy
# youtube: https://www.youtube.com/watch?v=qc4_H72sEqk
# word count:[3886] | took 132.76 seconds
# word count:[3897] | took 139.87 seconds