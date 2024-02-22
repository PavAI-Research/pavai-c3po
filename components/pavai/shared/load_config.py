import json
from pathlib import Path
from rich import print
from rich.table import Table
from rich.console import Console
from rich.markdown import Markdown

_FILE = Path(__file__)
_DIR = _FILE.parent

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

_BOS = "^"
_EOS = "$"
_PAD = "_"

def show_shutdown_message():
    console = Console()    
    markdown = Markdown("# Talking Llama - shutdown \nhave a nice day!")
    console.print(markdown)

def show_startup_message():
    console = Console()    
    markdown = Markdown("# Welcome to Talking Llama - offline LLM voice assistant \n say a 'prompt' follow by 'action word' \nthen wait till confirmation text")
    console.print(markdown)
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Word")
    table.add_column("Description")
    table.add_row("1", "hello", "say 'hello' to start capture")
    table.add_row("2", "please", "say 'please' stop capture to send prompt to LLM")
    table.add_row("3", "over", "say 'over' action to send prompt to LLM")
    table.add_row("4", "say again", "say 'say again' action repeat last response")
    table.add_row("5", "new conversation", "say 'new conversation' action to start new chat")
    table.add_row("6", "open notes", "say 'open notes' action to record speech to note file")
    table.add_row("7", "close notes", "say 'close notes' action to save a note file")
    table.add_row("8", "search notes", "say 'search notes' action to save a note file")    
    table.add_row("9", "secretary", "say 'secretary' follow by 'question' then 'please' to schedule a lawyer appointment")
    console.print(table)

def load_config_file():
    # load configs
    with open('/home/pop/development/mclab/realtime/RealtimeSTT/tests/config.json', 'r') as FP:
        config_file = json.load(FP)
    return config_file

def load_llm_config():
    config_file=load_config_file()
    return config_file["llm"]["default"]

def load_speech_config():
    config_file=load_config_file()
    return config_file["speech"]["default"]

def load_voice_config(lang:str="default"):
    config_file=load_config_file()
    if lang=="en_us":
        return config_file["voice"]["en_us"]
    elif lang=="cn":
        return config_file["voice"]["cn"]         
    else:              
        return config_file["voice"]["default"]     

def load_speaker_config(lang:str="default"):
    config_file=load_config_file()
    if lang=="en_us":
        return config_file["speaker"]["en_us"]
    elif lang=="cn":
        return config_file["speaker"]["cn"]        
    else:              
        return config_file["speaker"]["default"]     

def load_provider_config():
    config_file=load_config_file()
    return config_file["providers"]

