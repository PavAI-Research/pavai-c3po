from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

# import os
# from dotenv import dotenv_values
# system_config = {
#     **dotenv_values("env.shared"),  # load shared development variables
#     **dotenv_values("env.secret"),  # load sensitive variables
#     **os.environ,  # override loaded values with environment variables
# }

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import pipeline
import transformers
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM
import time
import torch
import os
# import logging
# logging.basicConfig()
# logging.getLogger("faster_whisper").setLevel(logging.INFO)
# logger = logging.getLogger(__name__)
# distilled whisper
# pip install faster-whisper
# pip install optinum
# import optimum

logger.info("---WHISPER---")
logger.info(transformers.__version__)
logger.info(torch.__version__)

# # When running on CPU, make sure to set the same number of threads.
# cpus=str(int(os.cpu_count()/2))
# os.environ["OMP_NUM_THREADS"] = cpus
# device = "cuda" if torch.cuda.is_available() else "cpu"
# compute_type = "float16" if torch.cuda.is_available() else "int8"
# logger.info("CPUs:",cpus)
# logger.info("device:",device)
# logger.info("compute_type:",compute_type)

# model_size = "large-v3"
# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

g_faster_whisper_model = None

default_vad_parameters= {
    "threshold" : 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 2000,
    "window_size_samples": 1024,
    "speech_pad_ms":  400
}

def transcribe_faster(audio_file: str,
                           model_size: str = "large-v3",
                           device: str = None,
                           compute_type: str = "float16",
                           vad_filter: bool = True, vad_parameters: dict = default_vad_parameters,
                           task: str = "transcribe",
                           beam_size: int = 5,
                           best_of: int = 5,
                           patience: float = 1,
                           length_penalty: float = 1,
                           repetition_penalty: float = 1,
                           no_repeat_ngram_size: int = 0,
                           word_timestamps: bool = False,
                           language: str = None):
    """transcribe or translate multilinguage audio file"""
    logger.info("transcribe_faster_file: start")                    
    global g_faster_whisper_model
    t0 = time.perf_counter()
    # Run on GPU with FP16 if available - half-Precision
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
    g_faster_whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type) if g_faster_whisper_model is None else g_faster_whisper_model
    # VAD enabled
    # segments, _ = model.transcribe("audio.mp3", vad_filter=True)
    if vad_filter:
        segments, info = g_faster_whisper_model.transcribe(task=task, audio=audio_file,
                                                           vad_filter=vad_filter, vad_parameters=vad_parameters, beam_size=beam_size, best_of=best_of,
                                                           patience=patience, length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                                                           no_repeat_ngram_size=no_repeat_ngram_size, word_timestamps=word_timestamps)
    else:
        segments, info = g_faster_whisper_model.transcribe(task=task, audio=audio_file, beam_size=beam_size, best_of=best_of,
                                                           patience=patience, length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                                                           no_repeat_ngram_size=no_repeat_ngram_size, word_timestamps=word_timestamps)

    logger.info(f"detected language {info.language} with probability {info.language_probability:.2f}")                        
    # set language
    language = info.language if info else "en"
    transcribed_texts = []
    if word_timestamps:
        # Word-level timestamps
        transcribed_texts = [
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments]
    else:
        transcribed_texts=""
        for segment in segments:
            transcribed_texts = transcribed_texts+ f"{segment.text} "
    t1 = time.perf_counter()-t0
    logger.info(f"transcribe_faster_file: finished in {t1}")                
    return {task: transcribed_texts, "performance": t1, "detected_language": language}

def transcribe_distilled(audio_file: str,
                              model_id: str = "distil-whisper/distil-large-v2",
                              task: str = "transcribe",
                              device: str = None,
                              torch_dtype: str = None,
                              chunk_length_s: int = 30,
                              batch_size: int = 8,
                              return_timestamps: bool = True,
                              low_cpu_mem_usage: bool = True,
                              use_safetensors: bool = True,
                              max_new_tokens: int = 128,
                              use_flash_attention_2: bool = False,
                              to_bettertransformer: bool = False,
                              cache_dir="./models/whisper"):
    """transcribe audio file to english"""
    logger.info("transcribe_distilled_file: start")                
    t0 = time.perf_counter()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype,
                                                      low_cpu_mem_usage=low_cpu_mem_usage,
                                                      use_safetensors=use_safetensors,
                                                      use_flash_attention_2=use_flash_attention_2,
                                                      cache_dir=cache_dir)
    # print(model.config.model_type)
    if to_bettertransformer:
        model.to_bettertransformer()
    else:
        model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline("automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=max_new_tokens,
                    torch_dtype=torch_dtype,
                    device=device
                    )
    outputs = pipe(audio_file, chunk_length_s=chunk_length_s,
                   batch_size=batch_size, return_timestamps=True)
    t1 = time.perf_counter()-t0
    logger.info(f"transcribe_distilled_file: finished in {t1}")            
    return {task: outputs["text"], "performance": t1, "detected_language": "en"}

def text_to_file(out_text: str, file_name: str, output_dir: str = "./workspace/audio"):
    try:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        file_name = output_dir+"/"+file_name
        if isinstance(out_text, str):
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(out_text)
        elif isinstance(out_text, list):
            with open(file_name, 'w', encoding='utf-8') as f:
                for sentence in out_text:
                    f.write(sentence)
        else:
            file_name = None
            print("unrecognize output text")
        if file_name:
            file_stats = os.stat(file_name)
            print(file_stats)
            print(f'File Size in Bytes is {file_stats.st_size}')
            print(
                f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    except Exception as e:
        print(e)
        file_name = None
        print("error in writing text to file")
    return file_name

def transcribe_base(audio_file: str,
                         model_id: str = "openai/whisper-large-v3",
                         task: str = "transcribe",
                         device: str = None,
                         torch_dtype: str = None,
                         chunk_length_s: int = 30,
                         batch_size: int = 8,
                         return_timestamps: bool = False,
                         low_cpu_mem_usage: bool = True,
                         use_safetensors: bool = True,
                         max_new_tokens: int = 128,
                         use_flash_attention_2: bool = False,
                         to_bettertransformer: bool = False,
                         cache_dir="./models/whisper"):
    """transcribe audio file to english"""
    logger.info(f"transcribe_base_file: start")            
    t0 = time.perf_counter()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype,
                                                      low_cpu_mem_usage=low_cpu_mem_usage,
                                                      use_safetensors=use_safetensors,
                                                      use_flash_attention_2=use_flash_attention_2,
                                                      cache_dir=cache_dir)
    # print(model.config.model_type)
    if to_bettertransformer:
        model.to_bettertransformer()
    else:
        model.to(device)
    pipe = pipeline("automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=device
                    )
    outputs = pipe(audio_file, chunk_length_s=chunk_length_s,
                   batch_size=batch_size, return_timestamps=return_timestamps)
    t1 = time.perf_counter()-t0
    logger.info(f"transcribe_base_file: finished in {t1}")        
    return {task: outputs["text"], "performance": t1, "detected_language": "en"}

# def transcribe_speculative_decoding(audio_file: str,
#                                          teacher_model_id: str = "openai/whisper-large-v3",
#                                          student_model_id: str = "distil-whisper/distil-large-v2",
#                                          task: str = "transcribe",
#                                          device: str = None,
#                                          torch_dtype: str = None,
#                                          chunk_length_s: int = 30,
#                                          return_timestamps: bool = False,
#                                          low_cpu_mem_usage: bool = True,
#                                          use_safetensors: bool = True,
#                                          max_new_tokens: int = 128,
#                                          use_flash_attention_2: bool = False,
#                                          to_bettertransformer: bool = False,
#                                          cache_dir="./models/whisper"):
#     """
#     speculative decoding load both the teacher: openai/whisper-large-v2. 
#     As well as the assistant (a.k.a student) distil-whisper/distil-large-v2.
#     """
#     logger.info("transcribe_speculative_decoding_file: start")
#     t0 = time.perf_counter()
#     if device is None:
#         device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     if torch_dtype is None:
#         torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     # techer model
#     teacher_model = None
#     if use_flash_attention_2:
#         teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             teacher_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage,
#             use_safetensors=use_safetensors, attn_implementation="flash_attention_2", cache_dir=cache_dir)
#     else:
#         teacher_model = AutoModelForSpeechSeq2Seq.from_pretrained(
#             teacher_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage,
#             use_safetensors=use_safetensors, cache_dir=cache_dir)
#     # move to device
#     teacher_model.to(device)
#     processor = AutoProcessor.from_pretrained(teacher_model_id)
#     # student model
#     student_model = AutoModelForCausalLM.from_pretrained(
#         student_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, use_safetensors=use_safetensors)
#     student_model.to(device)
#     # pipeline
#     logger.info("working on transcribe_speculative_decoding_file:", audio_file)
#     pipe = pipeline("automatic-speech-recognition",
#                     model=teacher_model_id,
#                     tokenizer=processor.tokenizer,
#                     feature_extractor=processor.feature_extractor,
#                     max_new_tokens=max_new_tokens,
#                     generate_kwargs={"assistant_model": student_model},
#                     torch_dtype=torch_dtype, device=device)
#     outputs = pipe(book_audio_file, chunk_length_s=chunk_length_s,
#                    batch_size=1, return_timestamps=return_timestamps)
#     t1 = time.perf_counter()-t0
#     logger.info(f"transcribe_speculative_decoding_file: finished in {t1}")    
#     return {task: outputs["text"], "performance": t1, "detected_language": "en"}

# if __name__ == "__main__":
#     book_audio_file = "/home/pop/development/mclab/pavai/workspace/samples/audio_book_self_Improvement_101.wav"
#     book_audio_file = "/home/pop/development/mclab/pavai/workspace/samples/audio_chatgpt_usage_zh.wav"
#     book_audio_file = "/home/pop/development/mclab/pavai/workspace/samples/audio_file_phillips_screw_hp0.wav"
#     transcribed_texts=transcribe_faster_file(audio_file=book_audio_file,task= "transcribe")
#     print(transcribed_texts)
    # transcribed_text_to_file(out_text=transcribed_texts,file_name="transcribe_audio_chatgpt_usage_zh.txt",output_dir="./workspace/audio")
    # translated_texts=transcribe_faster_file(audio_file=book_audio_file,task= "translate")
    # print(translated_texts)
    # text_to_file(out_text=translated_texts,file_name="translate_audio_chatgpt_usage_zh.txt",output_dir="./workspace/audio")
    #transcribed_texts=transcribe_distilled_file(audio_file=book_audio_file,task= "transcribe",batch_size=16,cache_dir="./models/whisper")
    #print(transcribed_texts)
    # transcribed_texts = transcribe_base_file(audio_file=book_audio_file, batch_size=16,task="transcribe", cache_dir="./models/whisper")
    # print(transcribed_texts)
    #transcribed_texts = transcribe_speculative_decoding_file(audio_file=book_audio_file, task="transcribe", cache_dir="./models/whisper")
    #print(transcribed_texts)    
