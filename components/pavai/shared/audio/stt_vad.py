# This is a work in progress. There are still bugs. Once it is production-ready this will become a full repo.
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
from rich.logging import RichHandler
from rich import print,pretty,console
from rich.pretty import (Pretty,pprint)
from rich.panel import Panel
from rich.console import Console
logging.basicConfig(level=logging.WARN, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()

import queue
import numpy as np
import pyaudio
import webrtcvad
import collections
from halo import Halo
import sounddevice as sd
import torch
import torchaudio
import git 
import os 

CPUs = os.cpu_count()
torch.set_num_threads(int(CPUs/2))
DEFAULT_SAMPLE_RATE = 16000
INT16_MAX_ABS_VALUE = 32768.0

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS):
        def proxy_callback(in_data, frame_count, time_info, status):
            # pylint: disable=unused-argument
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None:
            def callback(in_data): return self.buffer_queue.put(in_data)
        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS /
                              float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(
            self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        # if not default device
        if self.device:
            kwargs['input_device_index'] = self.device

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(
        lambda self: 1000 * self.block_size // self.sample_rate)

class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None):
        super().__init__(device=device, input_rate=input_rate)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Generator that yields all audio frames from microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            raise Exception("Resampling required")

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

def init_vadaudio(webRTC_aggressiveness:int=3,device:int=None,rate:int=DEFAULT_SAMPLE_RATE):
    vad_audio = VADAudio(aggressiveness=webRTC_aggressiveness,
                         device=device,
                         input_rate=rate)
    
    return vad_audio     

def _get_vad_model(repo_or_dir:str='snakers4/silero-vad',silaro_model_name:str="silero_vad",
                   source:str="github", reload:bool=False,silero_use_onnx:bool=True):    
    model, utils = torch.hub.load(repo_or_dir=repo_or_dir,
                                  model=silaro_model_name,
                                  source=source,
                                  force_reload=reload,
                                  onnx=silero_use_onnx)
    logger.debug(f"_get_vad_model loaded!")    
    return model, utils

def init_vad_model(repo_or_dir:str='snakers4/silero-vad',silaro_model_name:str="silero_vad",
                   reload:bool=False,silero_use_onnx:bool=True,download_root:str="resources/models"):
    """Silero VAD - pre-trained enterprise-grade Voice Activity Detector"""
    logger.debug(f"init_vad_model {repo_or_dir} saved to {download_root}")
    torchaudio.set_audio_backend("soundfile")
    model=None 
    utils=None
    try:
        # get local copy first
        # credit original: https://github.com/snakers4/silero-vad.git
        repo_or_dir=download_root+"/silero-vad"
        model, utils = _get_vad_model(repo_or_dir=repo_or_dir,
                                      silaro_model_name=silaro_model_name,
                                      source="local",
                                      reload=reload,silero_use_onnx=silero_use_onnx)
        #(get_speech_timestamps, get_language, save_audio, read_audio,VADIterator, collect_chunks) = utils
    except Exception as e:
        try:
            # get files from github
            repo = git.Repo.clone_from(url='https://github.com/minyang-chen/silero-vad.git',
                                   to_path=download_root,branch='master')
            repo_or_dir=download_root+"/silero-vad"
            model, utils = _get_vad_model(repo_or_dir=repo_or_dir,
                                      silaro_model_name=silaro_model_name,
                                      source="local",
                                      reload=reload,silero_use_onnx=silero_use_onnx)
        except Exception as e:
            # otherwise, attempt download from pytorch.hub         
            repo_or_dir='snakers4/silero-vad'
            model, utils = _get_vad_model(repo_or_dir=repo_or_dir,silaro_model_name=silaro_model_name,
                                      source="github",
                                      reload=reload,silero_use_onnx=silero_use_onnx)
    return model,utils

def Int2Float(sound):
    _sound = np.copy(sound)
    abs_max = np.abs(_sound).max()
    _sound = _sound.astype('float32')
    if abs_max > 0:
        _sound *= 1/abs_max
    audio_float32 = torch.from_numpy(_sound.squeeze())
    return audio_float32

def has_speech_activity(vad_model,utils,audio_chunk):
    (get_speech_timestamps,save_audio,read_audio,VADIterator,collect_chunks)=utils    
    audio_float32 = Int2Float(audio_chunk)
    time_stamps = get_speech_timestamps(audio_float32, vad_model,
                                        threshold=0.5, sampling_rate=16000,
                                        min_speech_duration_ms=250,
                                        min_silence_duration_ms=100, # default 100
                                        window_size_samples=1024, # default=512
                                        max_speech_duration_s=60, # max speech duration 
                                        speech_pad_ms=30)
    # if (len(time_stamps) > 0):
    #     print("silero VAD has detected a possible speech")    
    return time_stamps

def normalize_audio_chunks(wav_data):
    audio_chunk = np.frombuffer(wav_data, dtype=np.int16)            
    audio_chunk = audio_chunk.astype(np.float32) / INT16_MAX_ABS_VALUE 
    return audio_chunk

def os_speaker(msg:str):
    import os
    os.system('spd-say "...okay. one moment please."')
    # https://stackoverflow.com/questions/16573051/sound-alarm-when-code-finishes
    #duration = 1  # seconds
    #freq = 440  # Hz
    #os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

if __name__ == '__main__':
    os_speaker("hello world!")

