# import os
# from dotenv import dotenv_values
# system_config = {
#     **dotenv_values("env.shared"),  # load shared development variables
#     **dotenv_values("env.secret"),  # load sensitive variables
#     **os.environ,  # override loaded values with environment variables
# }
import sys
import warnings
import torch
import os
from rich.console import Console
# from rich.logging import RichHandler
# import logging
# logging.basicConfig(level=logging.ERROR, format="%(message)s",
#                     datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
# logger = logging.getLogger(__name__)
# # pretty.install()
# warnings.filterwarnings("ignore")
from pavai.shared.handfree_aio import system_startup, system_initialization, activate_handfree_system

from pavai.setup import config 
from pavai.setup import logutil

console = Console()

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

_USE_VOICE_API = False

WAKEUP_KEYWORDS_WORDS = "ryan"
INIT_MODEL_TRANSCRIPTION = "large-v2"
INIT_MODEL_TRANSCRIPTION_REALTIME = "large-v2"

CPUs = os.cpu_count()
torch.set_num_threads(int(CPUs/2))
DEFAULT_SAMPLE_RATE = 16000
USE_ONNX = False
PIPER_AI_VOICES = ["Amy","Ryan"]
LIBRI_AI_VOICES = ["Ryan", "Jane", "Vinay", "Nima","Yinghao", "Keith", "May", "June"]

if __name__ == '__main__':
    try:
        system_startup(output_voice="jane")
        system_initialization()
        activate_handfree_system()
    except (KeyboardInterrupt, SystemExit):
        pass

    # import argparse
    # parser = argparse.ArgumentParser(
    #     description="Stream from microphone to webRTC and silero VAD")

    # parser.add_argument('-v', '--webRTC_aggressiveness', type=int, default=3,
    #                     help="Set aggressiveness of webRTC: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    # parser.add_argument('--nospinner', action='store_true',
    #                     help="Disable spinner")
    # parser.add_argument('-d', '--device', type=int, default=None,
    #                     help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")

    # parser.add_argument('-name', '--silaro_model_name', type=str, default="silero_vad",
    #                     help="select the name of the model. You can select between 'silero_vad',''silero_vad_micro','silero_vad_micro_8k','silero_vad_mini','silero_vad_mini_8k'")
    # parser.add_argument('--reload', action='store_true',
    #                     help="download the last version of the silero vad")

    # # parser.add_argument('-ts', '--trig_sum', type=float, default=0.25,
    # #                     help="overlapping windows are used for each audio chunk, trig sum defines average probability among those windows for switching into triggered state (speech state)")

    # # parser.add_argument('-nts', '--neg_trig_sum', type=float, default=0.07,
    # #                     help="same as trig_sum, but for switching from triggered to non-triggered state (non-speech)")

    # # parser.add_argument('-N', '--num_steps', type=int, default=8,
    # #                     help="nubmer of overlapping windows to split audio chunk into (we recommend 4 or 8)")

    # # parser.add_argument('-nspw', '--num_samples_per_window', type=int, default=4000,
    # #                     help="number of samples in each window, our models were trained using 4000 samples (250 ms) per window, so this is preferable value (lesser values reduce quality)")

    # # parser.add_argument('-msps', '--min_speech_samples', type=int, default=10000,
    # #                     help="minimum speech chunk duration in samples")

    # # parser.add_argument('-msis', '--min_silence_samples', type=int, default=500,
    # #                     help=" minimum silence duration in samples between to separate speech chunks")
    # ARGS = parser.parse_args()
    # ARGS.rate = DEFAULT_SAMPLE_RATE
