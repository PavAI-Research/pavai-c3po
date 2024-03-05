from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

from abc import ABC, abstractmethod
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.audio_utils import ffmpeg_read
import numpy as np
import torch
from huggingface_hub import hf_hub_download,snapshot_download    
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model, format_timestamp, get_logger
from pytube import YouTube
import functools
import os
import time
import pavai.shared.system_types as system_types

# global
DEFAULT_WHISPER_MODEL_SIZE = "large"
DISTILLED_WHISPER_MODEL_SIZE = "distil-whisper/distil-large-v2"

AVAILABLE_WHISPER_MODELS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
}

DEFAULT_VAD_OPTIONS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 2000,
    "window_size_samples": 1024,
    "speech_pad_ms": 400
}


class AbstractTranscriberClass(ABC):
    """
    The Abstract Class defines a template method that contains a skeleton of
    audio transcribe algorithm, composed of calls to (usually) abstract primitive
    operations.
    Concrete subclasses should implement these operations, but leave the
    template method itself intact.
    """
    def __init__(self, model_id_or_path:str, use_device:str="cpu", 
                 use_torch_type:str="float16",
                 model_tokenizer_id_or_path:str=None,
                 use_flash_attention_2:bool=False,
                 use_download_root:str="resources/models/whisper",
                 use_local_model_only:bool=False) -> None:
        super().__init__()
        self._model_id_or_path = model_id_or_path
        self._use_device = use_device
        self._use_torch_type = use_torch_type
        self._use_flash_attention_2 = use_flash_attention_2
        self._model_tokenizer_id_or_path = model_tokenizer_id_or_path           
        self._use_task = "transcribe"                                    
        self._model = None
        self._tokenizer = None                      
        self._processor = None                                
        self._pipeline = None 
        self._download_root = use_download_root
        self._local_model_only = use_local_model_only

    @property
    def model_id_or_path(self):
        return self._model_id_or_path
    
    @model_id_or_path.setter
    def model_id_or_path(self, new_name):
        self._model_id_or_path = new_name

    @property
    def model_tokenizer_id_or_path(self):
        return self._model_tokenizer_id_or_path
    
    @model_id_or_path.setter
    def model_tokenizer_id_or_path(self, new_name):
        self._model_tokenizer_id_or_path = new_name

    @property
    def use_device(self):
        return self._use_device
    
    @use_device.setter
    def use_device(self, new_name):
        self._use_device = new_name

    @property
    def use_local_model_only(self):
        return self._local_model_only
    
    @use_local_model_only.setter
    def use_local_model_only(self, new_name):
        self._local_model_only = new_name

    @property
    def use_download_root(self):
        return self._download_root
    
    @use_download_root.setter
    def use_download_root(self, new_name):
        self._download_root = new_name

    @property
    def use_torchtype(self):
        return self._use_torchtype
    
    @use_torchtype.setter
    def use_torchtype(self, new_name):
        self._use_torchtype = new_name

    @property
    def use_input_source(self):
        return self._use_input_source
    
    @use_input_source.setter
    def use_input_source(self, new_name):
        self._use_input_source = new_name

    @property
    def use_input_format(self):
        return self._use_input_format
    
    @use_input_format.setter
    def use_input_format(self, new_name):
        self._use_input_format = new_name

    @property
    def use_output_format(self):
        return self._use_output_format
    
    @use_output_format.setter
    def use_output_format(self, new_name):
        self._use_output_format = new_name

    @property
    def use_task(self):
        return self._use_task
    
    @use_task.setter
    def use_task(self, new_name):
        self._use_task = new_name

    @property
    def use_flash_attention_2(self):
        return self._use_flash_attention_2
    
    @use_task.setter
    def use_flash_attention_2(self, new_name):
        self._use_flash_attention_2 = new_name

    def word_count(self,string):
        return(len(string.strip().split(" ")))


    def normalize_audio_chunks(self,wav_data):
        audio_chunk = np.frombuffer(wav_data, dtype=np.int16)
        audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        return audio_chunk

    def youtube_download(self,url:str,local_storage:str="workspace/downloads/youtube_audio"):
        t0 = time.perf_counter()     
        local_file = None   
        try:
            link = YouTube(url)
            local_file = link.streams.filter(only_audio=True)[0].download(local_storage)
        except Exception as e:
            logger.error(f"youtube_download error: {e}")    
        took_in_seconds = time.perf_counter() - t0  
        status_msg=f"youtube_download [{url}] took {took_in_seconds:.2f} seconds"   
        logger.debug(status_msg) 
        print(status_msg)                  
        return local_file       

    def transcribe(self) -> None:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        print(f"use_device: {self.use_device}")
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def transcribe_file(self,input_file:str,task:str="transcribe",return_timestamps:bool=True,language:str=None) -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def transcribe_numpy(self,input_audio,task:str="transcribe",return_timestamps:bool=True,language:str=None) -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    def transcribe_youtube(self,input_url:str,task:str="transcribe",return_timestamps:bool=True,language:str=None) -> str:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        self.prepare_output()
        self.hook1()

    # These operations already have implementations.
    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def load_tokenizer(self) -> None:
        pass

    @abstractmethod
    def create_pipeline(self) -> None:
        pass

    # These operations have to be implemented in subclasses.
    @abstractmethod
    def prepare_input(self) -> None:
        pass

    @abstractmethod
    def prepare_output(self) -> None:
        pass

    # These are "hooks." Subclasses may override them, but it's not mandatory
    # since the hooks already have default (but empty) implementation. Hooks
    # provide additional extension points in some crucial places of the algorithm.
    def hook1(self) -> None:
        pass

class WhisperTranscriber(AbstractTranscriberClass):
    """
    Concret Transcriber Class override only required class operations.
    """
    # "openai/whisper-large-v3"
    def load_model(self) -> None:
        logger.debug(f"WhisperTranscriber: load_model {self.model_id_or_path}") 
        if self._model is None: 
            logger.debug(f"use_flash_attention_2: {self.use_flash_attention_2}")             
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id_or_path, torch_dtype=self.use_torchtype,
                use_flash_attention_2=self.use_flash_attention_2,
                low_cpu_mem_usage=True, use_safetensors=True)
                
            if not is_flash_attn_2_available():
                # use flash attention from pytorch sdpa - required: pip install optimum=1.14.1
                self._model = self._model.to_bettertransformer()
                logger.debug("use_flash_attention_2: to_bettertransformer")                             

            self._model.to(self.use_device)
            logger.debug(f"moved model to device: {self.use_device}")                                         

    def load_tokenizer(self) -> None:
        logger.debug(f"WhisperTranscriber: load_tokenizer {self.model_id_or_path}")
        if self._processor is None:        
            self._processor = AutoProcessor.from_pretrained(self.model_id_or_path)        

    def create_pipeline(self) -> None:
        logger.debug(f"WhisperTranscriber: create_pipeline")        
        if self._pipeline is None:
            self._pipeline = pipeline("automatic-speech-recognition", 
                               model=self._model,
                               tokenizer=self._processor.tokenizer,
                               feature_extractor=self._processor.feature_extractor,                               
                               device=self.use_device,
                               torch_dtype=self.use_torchtype,
                               max_new_tokens=128,
                               chunk_length_s=15, #default 15    
                               batch_size=16,     #default 16                           
                               generate_kwargs={"task": self.use_task},
                               model_kwargs={"use_flash_attention_2": self.use_flash_attention_2},
                               return_timestamps=True)        

    def prepare_input(self) -> None:
        logger.debug(f"WhisperTranscriber: prepare_input")                

    def prepare_output(self) -> None:
        logger.debug(f"WhisperTranscriber: prepare_output")                        

    def transcribe_file(self, input_file,task:str="transcribe",return_timestamps:bool=True,language:str=None) -> dict:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        logger.debug(f"transcribe_file: use_device: {self.use_device}")                                
        t0 = time.perf_counter()                
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        text = self._pipeline(input_file,return_timestamps=return_timestamps)["text"] 
        text = text.strip() if text is not None else ""   
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_file completed took {took_in_seconds:.2f} seconds"    
        print(status_msg)        
        logger.debug(status_msg)                                        
        self.prepare_output()
        self.hook1()
        outputs = {
            "language": language,
            "transcription": text,
            "timestamps": [],            
        }          
        return outputs

    def transcribe_numpy(self,input_audio,task:str="transcribe",return_timestamps:bool=True,language:str=None) -> dict:
        """
        The template method defines the skeleton of transcribe transcribe_audio_data algorithm.
        """
        logger.debug(f"transcribe_numpy: use_device: {self.use_device}")                                
        t0 = time.perf_counter()                        
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        sr, y = input_audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        text = self._pipeline({"sampling_rate": sr, "raw": y})["text"]            
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_numpy completed took {took_in_seconds:.2f} seconds"    
        print(status_msg)        
        logger.debug(status_msg)                                                
        self.prepare_output()
        self.hook1()
        outputs = {
            "language": language,
            "transcription": text,
            "timestamps": [],            
        }          
        return outputs        

    def transcribe_youtube(self,input_url:str,task:str="transcribe",return_timestamps:bool=True,language:str=None) -> dict:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        logger.debug(f"transcribe_youtube: use_device: {self.use_device}")                                
        t0 = time.perf_counter()        
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        local_file = self.youtube_download(input_url)
        outputs = self.transcribe_file(local_file)
        words=self.word_count(outputs["transcription"])
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_youtube completed. \nword count:[{words}] | took {took_in_seconds:.2f} seconds"    
        print(status_msg)
        logger.debug(status_msg)                                                        
        self.prepare_output()
        self.hook1()
        return outputs        

    def hook1(self) -> None:
        logger.debug(f"WhisperTranscriber: hook1")                                

class DistrilledTranscriber(WhisperTranscriber):
    """
    DistrilledTranscriber Transcriber Class override only required class operations.
    """
    def load_tokenizer(self) -> None:
        logger.debug(f"DistrilledTranscriber: load_tokenizer {self.model_tokenizer_id_or_path}")                                        
        if self._processor is None:        
            self._processor = AutoProcessor.from_pretrained(self.model_tokenizer_id_or_path)        

class FasterTranscriber(AbstractTranscriberClass):
    """
    FasterTranscriber Class override only required class operations.
    """
    def load_model(self) -> None:
        logger.debug(f"FasterTranscriber: load_model: None")                                                

    def load_tokenizer(self) -> None:
        logger.debug(f"FasterTranscriber: load_tokenizer: None")                                                

    def create_pipeline(self) -> None:
        logger.debug(f"FasterTranscriber: load_tokenizer {self.model_id_or_path}")                                        
        if self._pipeline is None:
            self.use_torchtype = "float16" if torch.cuda.is_available() else "int8"
            self.use_device = "cuda" if self.use_device=="cuda:0" else self.use_device
            self.use_cpu_threads = int(os.cpu_count()/2)  #use half only
            logger.debug(f"faster device: {self.use_device}")                                                    
            logger.debug(f"faster compute_type: {self.use_torchtype}")                                                                
            self._pipeline = WhisperModel(
                model_size_or_path=self.model_id_or_path,
                device=self.use_device,
                compute_type=self.use_torchtype,
                cpu_threads=self.use_cpu_threads,
                num_workers=1,
                download_root=self.use_download_root,
                local_files_only=self.use_local_model_only)        

    def prepare_input(self) -> None:
        logger.debug(f"FasterTranscriber: prepare_input: None")                                                

    def prepare_output(self) -> None:
        logger.debug(f"FasterTranscriber: prepare_output: None")                                                

    def transcribe_file(self, input_file,task:str="transcribe",
                        return_timestamps:bool=False,language:str=None) -> dict:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        text = ""
        seg = []        
        logger.debug(f"FasterTranscriber: transcribe_file: {self.use_device}")                                                
        t0 = time.perf_counter()                
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        segments, info = self._pipeline.transcribe(audio=input_file,
                                                   language=language,
                                                   task=task,
                                                   beam_size=5,
                                                   vad_filter=True)
        
        logger.info(f"Detected language [{info.language}] with probability {info.language_probability}:%.2f")        
        language = info.language if language is None else language
        for segment in segments:
            text = text+segment.text
            if return_timestamps and segment.words:
                for word in segment.words:   
                    txt="[{word.start}:.2fs -> {word.end}:.2fs] {word.word}" 
                    seg.append(txt)
                text = ''.join(seg)
        text = text.strip() if text is not None else ""   
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_file completed took {took_in_seconds:.2f} seconds"    
        logger.info(status_msg)                                                
        self.prepare_output()
        self.hook1()
        outputs = {
            "language": language,
            "transcription": text,
            "timestamps": seg,            
        }        
        return outputs

    def transcribe_numpy(self,input_audio,task:str="transcribe",
                         return_timestamps:bool=True,language:str=None) -> dict:
        """
        The template method defines the skeleton of transcribe transcribe_audio_data algorithm.
        """
        text = ""
        seg = []        
        logger.debug(f"FasterTranscriber: transcribe_numpy: {self.use_device}")                                                
        t0 = time.perf_counter()                        
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        sr, y = input_audio
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))
        segments, info = self._pipeline.transcribe(audio=y,
                                                   language=None,
                                                   task=task,
                                                   beam_size=5,
                                                   vad_filter=True)
        language = info.language if language is None else language
        for segment in segments:
            text = text+segment.text
            if return_timestamps:
                for word in segment.words:   
                    txt="[{word.start}:.2fs -> {word.end}:.2fs] {word.word}" 
                    seg.append(txt)
            text = ''.join(seg)
        text = text.strip() if text is not None else ""   
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_file completed took {took_in_seconds:.2f} seconds"    
        logger.info(status_msg)         
        self.prepare_output()
        self.hook1()
        outputs = {
            "language": language,
            "transcription": text,
            "timestamps": seg,            
        }                
        return outputs

    def transcribe_youtube(self,input_url:str,task:str="transcribe",
                           return_timestamps:bool=True,language:str=None) -> dict:
        """
        The template method defines the skeleton of transcribe algorithm.
        """
        print(f"use_device: {self.use_device}")
        t0 = time.perf_counter()        
        self.load_model()
        self.load_tokenizer()
        self.create_pipeline()
        self.prepare_input()
        local_file = self.youtube_download(input_url)
        output = self.transcribe_file(local_file)
        took_in_seconds = time.perf_counter()-t0        
        status_msg=f"transcribe_youtube completed took {took_in_seconds:.2f} seconds"    
        logger.info(status_msg)          
        self.prepare_output()
        self.hook1()
        return output

## default transcriber
faster_transcriber = FasterTranscriber(model_id_or_path=DEFAULT_WHISPER_MODEL_SIZE)

@functools.lru_cache
def transcriber_client(abstract_transcriber: AbstractTranscriberClass, 
                       input_audio:None,input_type:str="audio",
                       task:str="transcribe",
                       return_timestamps:bool=False,
                       language:str=None) -> str:
    """
    The client code calls the template method to execute the algorithm. Client
    code does not have to know the concrete class of an object it works with, as
    long as it works with objects through the interface of their base class.
    """
    # ...
    # attach environment optimized settings
    use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    use_flash_attention_2 = is_flash_attn_2_available()    
    abstract_transcriber.use_device=use_device
    abstract_transcriber.use_torchtype=use_torch_dtype
    abstract_transcriber.use_flash_attention_2=use_flash_attention_2
    # perform operation 
    result = None
    if isinstance(input_audio,str):
        if input_audio.endswith(".mp3") or input_audio.endswith(".mp4") or input_audio.endswith(".wav"):
            result = abstract_transcriber.transcribe_file(
                input_file=input_audio,
                task=task,
                return_timestamps=return_timestamps,
                language=language
                )
        elif input_type =="youtube":
            result = abstract_transcriber.transcribe_youtube(
                input_url=input_audio,
                task=task,
                return_timestamps=return_timestamps,
                language=language                
                )
    elif isinstance(input_audio,np):
        result = abstract_transcriber.transcribe_numpy(
            input_audio=input_audio,
            task=task,
            return_timestamps=return_timestamps,
            language=language)        
    else:
        raise Exception("unsupported input audio type!")
    return result
    # ...

def get_transcriber(model_id_or_path: str = DEFAULT_WHISPER_MODEL_SIZE,transcriber_class:str="FasterTranscriber"):
    global faster_transcriber   
    logger.info(f"get_transcriber:{transcriber_class}")    
    # default to faster-whisper
    if faster_transcriber is None:
        faster_transcriber = FasterTranscriber(model_id_or_path=model_id_or_path)
    if transcriber_class=="DistrilledTranscriber":
        return DistrilledTranscriber(model_id_or_path=model_id_or_path)    
    return faster_transcriber

def speech_to_text(input_audio: Union[str, BinaryIO, np.ndarray],
                   task_mode="transcribe",
                   model_size: str = "large",
                   beam_size: int = 5,
                   vad_filter: bool = True,
                   language: str = None,
                   include_timestamp_seg=False) -> str:
    """
    # convert speech to text
    """
    logger.info(f"speech_to_text: mode({task_mode})")
    if isinstance(input_audio, str):
        logger.info(f"input audio file: {input_audio}")        
    else:
        logger.info(f"input audio type: {type(input_audio)}")                
    
    transcriber=get_transcriber()
    outputs = transcriber_client(abstract_transcriber=transcriber,
                                               input_audio=input_audio,
                                               input_type="audio",task="transcribe")

    return outputs["transcription"], outputs["language"]

def get_or_download_hf_model_file(cache_dir:str="resources/models/whisper"):
    logger.info("get_or_download_hf_model_file")        
    done=False
    hf_hub_download(repo_id="openai/whisper-large-v3",filename="config.json",cache_dir=cache_dir)

def get_or_download_whisper_model_snapshot(cache_dir:str="resources/models/whisper"):
    logger.info("get_or_download_hf_model_snapshot")    
    ##local_model_file=snapshot_download(repo_id="openai/whisper-large-v3", cache_dir="resources/models/whisper")               
    ##local_model_file=snapshot_download(repo_id="distil-whisper/distil-large-v2", cache_dir=cache_dir)           
    local_model_file=snapshot_download(repo_id="Systran/faster-whisper-large-v3", cache_dir=cache_dir)       
    local_model_file=snapshot_download(repo_id="Systran/faster-whisper-large-v2", cache_dir=cache_dir) 
    logger.info(local_model_file)
    if os.path.exists(local_model_file):
        logger.info(f"model file downloaded folder: {cache_dir} - Success!")
    else:
        logger.error(f"model file downloaded - Failed!")
    return local_model_file 

if __name__ == "__main__":
#     print("Same client code can work with different subclasses:")
    get_or_download_whisper_model_snapshot()

    input_audio_file="/home/pop/development/mclab/realtime/examples/jfk.wav"
    print("\nTEST-Whisper-1")
    #whisper_transcriber = WhisperTranscriber(model_id_or_path="openai/whisper-large-v3", use_local_model_only=True)
    whisper_transcriber=get_transcriber()
    # print(whisper_transcriber._download_root)
    # print(whisper_transcriber._model_id_or_path)
    whisper_result = transcriber_client(abstract_transcriber=whisper_transcriber,
                                input_audio=input_audio_file,
                                input_type="audio",
                                task="transcribe")
    print(whisper_result)

