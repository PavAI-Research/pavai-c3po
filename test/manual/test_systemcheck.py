from __future__ import annotations
from dotenv import dotenv_values
import time
import warnings
import logging
from rich import print, pretty, console
from rich.logging import RichHandler
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True,tracebacks_show_locals=True)])
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
pretty.install()
import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
logger.info(os.getcwd())

from src.shared.system_checks import pavai_vocie_system_health_check,pavai_talkie_system_health_check


if __name__ == "__main__":
    pavai_vocie_system_health_check(output_voice="en_amy")
    #pavai_talkie_system_health_check(output_voice="en_ryan")
    #intro_message="hi, I am Ryan, your personal multilingual AI assistant for everyday tasks, how may I help you today?" 
    #speak_instruction(intro_message,output_voice="en_ryan")