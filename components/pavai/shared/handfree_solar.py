
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
from rich.logging import RichHandler
from rich import print,pretty,console
from rich.pretty import (Pretty,pprint)
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
#pretty.install()
import warnings 
warnings.filterwarnings("ignore")
import os
import time
from halo import Halo
import traceback
import random
from datetime import datetime
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
import faster_whisper
logging.getLogger("handfree_whisper").setLevel(logging.WARN)

import openai
from openai import OpenAI
import functools
import torch
import sounddevice as sd

from pavai.shared.audio.stt_vad import init_vad_model,init_vadaudio,has_speech_activity,normalize_audio_chunks
from pavai.shared.system_checks import PAVAI_APP_TALKIE,DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT,pavai_talkie_system_health_check

from pavai.shared.audio.tts_client import text_speaker_ai
from pavai.shared.aio.llmtokens import HistoryTokenBuffer
from pavai.shared.aio.llmchat import (system_prompt_assistant, llm_chat_completion,llm_chat)
from pavai.shared.solar.llmclassify import isvalid_command,match_talkie_codes
from pavai.shared.solar.llmclassify import text_to_domain_model_mapping
from pavai.shared.solar.llmprompt import lookup_expert_system_prompt
import pavai.shared.solar.llmchat as llmchat
import pavai.shared.solar.llmprompt as llmprompt 
import pavai.shared.solar.llmcognitive as llmcognitive 

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import random
import numpy as np

#import nltk
#nltk.download('punkt')
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

CPUs = os.cpu_count()
torch.set_num_threads(int(CPUs/2))

print("--GLOBAL SYSTEM MODE----")
print(system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=system_config["GLOBAL_TTS"]
_GLOBAL_TTS_LIBRETTS_VOICE=system_config["GLOBAL_TTS_LIBRETTS_VOICE"]

_GLOBAL_STT=system_config["GLOBAL_STT"]
_GLOBAL_TTS_API_ENABLE=system_config["GLOBAL_TTS_API_ENABLE"]
_GLOBAL_TTS_API_URL=system_config["GLOBAL_TTS_API_URL"]
_GLOBAL_TTS_API_LANGUAGE=system_config["GLOBAL_TTS_API_LANGUAGE"]
_GLOBAL_TTS_API_SPEAKER_MODEL=system_config["GLOBAL_TTS_API_SPEAKER_MODEL"]

console = Console()
USE_ONNX = False

DEFAULT_SAMPLE_RATE = 16000
_CONVERSATION_LANGUAGE="en"

def get_time_of_day(time):
    if time < 12:
        return "Morning"
    elif time < 16:
        return "Afternoon"
    elif time < 19:
        return "Evening"
    else:
        return "Night"

def fn_current_time():
    now = datetime.now()
    time_now = get_time_of_day(now.hour)
    current_time=(f"current time: {time_now} at {now.hour} {now.minute} .")
    return current_time   

def system_startup(output_voice:str="en_ryan"):
    SYSTEM_READY=False        
    try:
        SYSTEM_READY=pavai_talkie_system_health_check(output_voice=output_voice)        
        console.print("\n[blue]System Ready...")             
    except Exception as e:       
        traceback.print_exc()        
        print("An error occurred:",e)                
        logger.error(f"An exception occurred at system_startup!")
        raise SystemExit("program exited")
    if SYSTEM_READY:
        text_speaker_ai(sd,f"I am listening. how may I help you today?",output_voice="en_ryan")     
    else:
        text_speaker_ai(sd,"Oops, system startup check failed!. please check console log.",output_voice="en_ryan")   

## Text-to-Speech 
def speaker_announce(instruction:str,output_voice:str="en_ryan"):
    text_speaker_ai(sd,instruction,output_voice)    
    return instruction

## -----------------------------------------
## LLM
## -----------------------------------------
import pavai.functions.talkiellm as talkiellm

talkieapi=None
def chat_service(user_prompt:str=None, history:list=[], 
                   system_prompt: str = system_prompt_assistant,
                   stop_criterias=["</s>"],target_model_id:str=None,ask_expert:str=None):    
    global talkieapi

    if _GLOBAL_SYSTEM_MODE=="SOLAR":
        if talkieapi is None:
            default_url=system_config["SOLAR_LLM_DEFAULT_SERVER_URL"] 
            default_api_key=system_config["SOLAR_LLM_DEFAULT_API_KEY"]             
            domain_url=system_config["SOLAR_LLM_DOMAIN_SERVER_URL"] 
            domain_api_key=system_config["SOLAR_LLM_DOMAIN_API_KEY"]     
            skip_content_safety_check=system_config["SOLAR_SKIP_CONTENT_SAFETY_CHECK"]    
            skip_content_safety_check = True if skip_content_safety_check=="true" else False
            skip_data_security_check=system_config["SOLAR_SKIP_DATA_SECURITY_CHECK"] 
            skip_data_security_check = True if skip_data_security_check=="true" else False            
            skip_self_critique_check=system_config["SOLAR_SKIP_SELF_CRITIQUE_CHECK"]
            skip_self_critique_check = True if skip_self_critique_check=="true" else False                        
            logger.debug("---LLMSolarClient:Settings---")
            logger.debug(f"LLM_DEFAULT_SERVER_URL: {default_url}")
            logger.debug(f"LLM_DEFAULT_API_KEY: {default_api_key}")
            logger.debug(f"LLM_DOMAIN_SERVER_URL: {domain_url}")
            logger.debug(f"LLM_DOMAIN_API_KEY: {domain_api_key}")
            logger.debug(f"SKIP_CONTENT_SAFETY_CHECK: {skip_content_safety_check}")
            logger.debug(f"SKIP_DATA_SECURITY_CHECK: {skip_data_security_check}")
            logger.debug(f"SKIP_SELF_CRITIQUE_CHECK: {skip_self_critique_check}")
            talkieapi = llmcognitive.LLMSolarClient(default_url=default_url,
                                                   default_api_key=default_api_key,
                                                   domain_url=domain_url,
                                                   domain_api_key=domain_api_key,
                                                   skip_content_safety_check=skip_content_safety_check,
                                                   skip_data_security_check=skip_data_security_check,
                                                   skip_self_critique_check=skip_self_critique_check)            
            ## alternative using llm_api.json
            ##llmsolar=llmcognitive.LLMSolarClient.new_instance("llm_api.json")
        cleanup=False
        if target_model_id is not None:
            if ask_expert is None:
                input_data = {"input_query": user_prompt,"input_model_id": target_model_id}          
            else:
                input_data = {"input_query": user_prompt,"ask_expert":ask_expert ,"input_model_id": target_model_id}          
        else:
            input_data = {"input_query": user_prompt}
        output,history=talkieapi.chat(input_data,history,cleanup)
        logger.debug(f"[blue]{output['output_text']}[/blue]",extra=dict(markup=True))
        reply =output['output_text']
    else:
        ## REPLACEMENT
        if talkieapi is None:
            talkieapi = talkiellm.TalkieAPI(use_local=True,options=talkiellm.LLM_OPTIONS)
        reply, history = talkieapi.chat(user_prompt=user_prompt,system_prompt = system_prompt,history=history)
        ### messages, history, reply = llm_chat_completion(user_Prompt=user_prompt, history= history,system_prompt = system_prompt,stop_criterias=stop_criterias)
        logger.debug(f"[blue]{reply}[/blue]",extra=dict(markup=True))
    return reply,history

def process_conversation(new_query:str,history:list, context_size:int=4096*2, 
                       system_prompt: str = "You are an intelligent AI assistant who can help answer user query.",
                       stop_criterias=["</s>"],output_voice="en_ryan",target_model_id:str=None, ask_expert:str=None):
    history_buffer = HistoryTokenBuffer(history=history,max_tokens=context_size)
    history_buffer.update(new_query)
    console.print(f"[gray]context token counts available: {history_buffer.token_count} / max {history_buffer.max_tokens}[/gray]")
    if len(history_buffer.overflow_summary)>0:
        # reduce and compress history content 
        logger.info(f"creating overflow summary: {history_buffer.overflow_summary}")
        summary="provide an summary of following text:\n "+history_buffer.overflow_summary
        assistant_response, history = chat_service(user_prompt=new_query, history=history,
                                                     system_prompt=system_prompt,stop_criterias=stop_criterias,target_model_id=target_model_id,ask_expert=ask_expert)
        text_speaker_ai(sd,text=assistant_response,output_voice=output_voice)       
        history_buffer.update(summary)    
        history=history_buffer.history
        history_buffer.overflow_summary=""
    # perform new query now
    assistant_response,history = chat_service(user_prompt=new_query, history=history,
                                                 system_prompt=system_prompt,stop_criterias=stop_criterias)
    #console.log(f"\nAI: {assistant_response}\n")
    return assistant_response, history 

## -----------------------------------------
## Speech-to-Text
## -----------------------------------------
@functools.lru_cache
def _get_whisper_model(model_path:str="large-v2",
                      download_root="resources/models/whisper",
                      local_files_only=True,
                      device:str=None):
    global whisper_model
    if device is None:
        device='cuda' if torch.cuda.is_available() else 'cpu'
    whisper_model = faster_whisper.WhisperModel(model_size_or_path=model_path,
                                            download_root=download_root,
                                            local_files_only=local_files_only,
                                            device='cuda' if torch.cuda.is_available() else 'cpu')
    return whisper_model

def init_whisper(model_path:str="large-v2",download_root="resources/models/whisper",local_files_only=True):
    global whisper_model
    try:
        # get local copy first
        whisper_model = _get_whisper_model(model_path=model_path,local_files_only=True)
    except Exception as e:
        print("An error occurred:", str(e.args))
        print(traceback.format_exc())
        logger.error(f"An exception occurred at init_whisper!")        
        if "CUDA failed with error out of memory" in str(e.args):
            print("attempt load whisper model into CPU only")
            whisper_model = _get_whisper_model(model_path=model_path,local_files_only=True,device="cpu")
        else:
            # otherwise, attempt download from huggingface        
            whisper_model = _get_whisper_model(model_path=model_path,download_root=download_root,local_files_only=False)
    return whisper_model

def whisper_transcribe(model:WhisperModel,audio,language:str=None):
    try:
        ## skip setting language=language if language else None,
        segments, language = model.transcribe(audio, 
                                    task= "transcribe",
                                    vad_filter=True)
        transcription = " ".join(seg.text for seg in segments).strip()
        return transcription, language
    except Exception as e:
        traceback.print_exc()        
        print("An error occurred:", e)                
        logger.error(f"An exception occurred at whisper_transcribe!")
    return None, language

## -----------------------------------------
## Conversation Loop
## -----------------------------------------
# keep track of running status
_CONVERSATION_QUERY_WORD="please"
_CONVERSATION_END_WORD="reset"
_CONVERSATION_CLEAR_WORD="clear"
_CONVERSATION_LANGUAGE="en"
_CONVERSATION_BEGIN_WORD=DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT.lower() if DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT is not None else "ryan"

ask_expert_help_words=["i need advise","ask expert advise","i looking for expert option","i need some help","i need advise"]

start_conversation_words=["ryan","hello, ryan","hey buddy","ok, ryan","hey ryan",'hi ryan']

talkie_codes_okay_action=["copy","affirmative","roger","over","out","roger that","please",
                          "thank you","thanks","is that right",
                          "got it","okay",'yep','right',"understood",'definitely','absolutely',
                          "thinking","that is correct.","sure","not sure","okie dokie",
                          "maybe","that is nice","sounds good", "good, thanks"]
talkie_codes_redo_action=["repeat"]
talkie_codes_undo_action=["disregard"]
talkie_codes_bad_action=["negative"]

# command to record a conversation to file
_COMMAND_RECORDING_START="start recording"
_COMMAND_RECORDING_STOP="stop recording"

# keep track of running status
RUNNING_STATUS=None
# keep track question capture mode
IN_SPEECH_RECORDING=False
# keep track conversation mode
IN_CONVERSATION_MODE=False
WAIT_USER_RESPONSE=True

# keep track of running status
ASK_EXPERT=False
ASK_EXPERT_TOPIC=None

# conversation variables
spinner = None
speech_buffer=""
wav_data = bytearray() 
ai_voice_text=""
conversation_history:list=[]

# global objects
vad_model = None
utils = None
whisper_model=None
vad_audio=None

def wakeup_greeting(output_voice:str="en_ryan"):
    now = datetime.now()
    time_now = get_time_of_day(now.hour)
    greeting_messages = ['hi, I am listening...', 
                         'I am here.', 
                         'hi...there', 
                         'hey? what is up?', 
                         'hi! how can I help you?',
                         'hello, tell me what you need.',
                         'okie dokie. I am listening.', 
                         'howdy!. I am here.',                         
                         'hi, what is up?']    
    greeting = random.choice(greeting_messages)
    text_speaker_ai(sd,greeting,output_voice)    
    #os.system(f"spd-say {greeting} sir!")
    console.print("\n[red]Speak now...") 
    return greeting   

def system_initialization(webRTC_aggressiveness:int=3,
                          silaro_model_name:str="silero_vad",
                          whisper_model_path="large-v2",
                          device:int=None, rate:int=DEFAULT_SAMPLE_RATE, 
                          reload:bool=False,nospinner:bool=False):
    """
    system_initialization parameters:
    webRTC_aggressiveness=an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3
    device=input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().
    silaro_model_name= 'silero_vad',''silero_vad_micro','silero_vad_micro_8k','silero_vad_mini','silero_vad_mini_8k'"
    """
    # global
    global vad_model
    global utils
    global whisper_model
    global vad_audio

    # vad model    
    vad_model,utils = init_vad_model(silaro_model_name=silaro_model_name,silero_use_onnx=True)
    (get_speech_timestamps,save_audio,read_audio,VADIterator,collect_chunks)=utils
    # whisper model
    whisper_model = init_whisper(model_path=whisper_model_path)
    # wake up detector
    #porcupine = init_porcupine(wake_words="ryan",wake_words_sensitivity=0.5) 
    # start audio with VAD
    vad_audio=init_vadaudio(webRTC_aggressiveness,device,rate)
    # Stream from microphone to DeepSpeech using VAD
    text="[🎤] Listening... (ctrl-C to exit)..."
    #console.print("[green]"+text)   
    console.print(f"[green]{DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT} is ready :smiley:")    

def activate_handfree_system(reload:bool=False,nospinner:bool=False):
    """activate_handfree_system"""
    global spinner
    if not nospinner:
        spinner = Halo(spinner='line',text_color="blue")
    wav_data = bytearray()
    spinner.succeed(f"{PAVAI_APP_TALKIE} is listening...🎤")    
    while True:
        frames = vad_audio.vad_collector() 
        detected_audio_activity(frames)

def detected_audio_activity(frames:list=[]):
    """process_detected_audio_activity"""
    global spinner
    global speech_buffer    
    global RUNNING_STATUS
    global IN_SPEECH_RECORDING
    global WAIT_USER_RESPONSE

    WAIT_USER_RESPONSE=False
   #global wav_data
    wav_data = bytearray() 
    if spinner:
        spinner.start("detecting:")    
    #console.log("voice conversation enbaled")     
    if WAIT_USER_RESPONSE:
        logger.debug("waiting user response - stop detecting")
        return 
    for frame in frames:
        if frame is not None:
            wav_data.extend(frame)
        else:
            audio_chunk=normalize_audio_chunks(wav_data)
            time_stamps=has_speech_activity(vad_model,utils,audio_chunk) 
            if (len(time_stamps) > 0):
                with Progress(transient=False) as progress: 
                    task = progress.add_task("thinking...", total=1)
                    moderate_conversation(whisper_model,audio_chunk,language="en")
                    progress.advance(task)
            else:
                #console.print("silero VAD has detected a noise", end="\n")
                wav_data=bytearray() 
            # always discard unused text not in speech
            wav_data=bytearray()           
        if RUNNING_STATUS=="user_query_end" or RUNNING_STATUS=="reset":
            frames = None
            RUNNING_STATUS ="detecting"
            IN_CONVERSATION_MODE=False
            IN_SPEECH_RECORDING=False
            break

def moderate_conversation(whisper_model,audio_chunk,language:str=_CONVERSATION_LANGUAGE, output_voice:str="en_ryan"):
    """handle_detected_speech"""
    global spinner
    global speech_buffer
    global ai_voice_text    
    global conversation_history    
    global RUNNING_STATUS
    global IN_SPEECH_RECORDING
    global IN_CONVERSATION_MODE    
    global WAIT_USER_RESPONSE        
    global ASK_EXPERT    
    
    #@Halo(text='conversation begin:', spinner='dots',color='green')
    def conversation_begin():
        global speech_buffer
        global ai_voice_text    
        global RUNNING_STATUS
        global IN_SPEECH_RECORDING
        global IN_CONVERSATION_MODE    
        global WAIT_USER_RESPONSE 
        #console.log(f"conversarion_begin")
        spinner.enabled=False        
        ai_voice_text=wakeup_greeting()   
        transcribed_text=""
        speech_buffer=""       
        IN_SPEECH_RECORDING=True 
        RUNNING_STATUS="user_query_begin"
        WAIT_USER_RESPONSE=True
        IN_CONVERSATION_MODE=True        
        return transcribed_text       

    #@Halo(spinner='dots', color='red', animation='marquee')
    def conversation_query(user_prompt:str, system_prompt:str=None, target_model_id:str=None,ask_expert:str=None):
        global ai_voice_text   
        global conversation_history
        if user_prompt is None or len(user_prompt.strip())==0:
            return  "skip", 0
        # speaker_announce(instruction="working on it",output_voice=output_voice)   
        user_prompt=user_prompt.strip()            
        try:
            console.print(f"[bold]Query:[/bold]: [blue]{user_prompt}[/blue]")         
            if system_prompt is None:   
                assistant_response, conversation_history =process_conversation(user_prompt,conversation_history) 
            else:
                assistant_response, conversation_history =process_conversation(user_prompt,conversation_history,
                                                                               system_prompt=system_prompt,
                                                                               target_model_id=target_model_id,ask_expert=ask_expert)                 
            ai_voice_text=assistant_response
            status=0
            console.print(f"[bold]Answer:[/bold]: [magenta]{assistant_response}[/magenta]")            
            #time.sleep(1)
            if assistant_response is not None:
                assistant_response=assistant_response.replace("\n"," ") 
            speaker_announce(instruction=f"{assistant_response}") #is that good?                           
        except Exception as e:
            status=1
            print("An error occurred:", e)                            
            print(traceback.print_exc())        
            logger.error(f"An exception occurred at conversation_query!")
            speaker_announce(instruction="oops!, request failed due error. please check console log.") 
        return "ok", status

    #@Halo(text='conversation capturing:', spinner='dots',color='yellow',animation='bounce')
    def conversation_capture_speech(transcribed_text:str):
        global ai_voice_text   
        global speech_buffer
        if speech_buffer is None:
            speech_buffer=transcribed_text
        else: 
            speech_buffer=speech_buffer+" "+transcribed_text
        # discard any ai speaker text
        speech_buffer=speech_buffer.replace(ai_voice_text,"").strip()        
        speech_buffer=speech_buffer.strip()
        spinner.info(f"speech: {speech_buffer}") 

    #@Halo(spinner='dots',color='green')
    def conversation_mode(sentence_last_word:str,transcribed_text:str,speech_buffer:str):
        global ai_voice_text    
        global RUNNING_STATUS
        global IN_SPEECH_RECORDING
        global ASK_EXPERT    
        #global WAIT_USER_RESPONSE      
        RUNNING_STATUS="conversation"   
        console.print(f":loudspeaker: {transcribed_text}",style="bold")
        transcribed_text=transcribed_text.rsplit(' ', 1)[0]
        transcribed_text=transcribed_text.replace(sentence_last_word,"")
        # drop ai spoken words if exist
        if speech_buffer is None:
            speech_buffer=transcribed_text
        speech_buffer=speech_buffer.replace(ai_voice_text,"").strip()
        speech_buffer=speech_buffer.replace(sentence_last_word,"").strip()
        # identify the topic first 
        if ASK_EXPERT:
            model_id, id2domain = text_to_domain_model_mapping(speech_buffer)
            system_prompt=lookup_expert_system_prompt(id2domain)
            result, status=conversation_query(user_prompt=speech_buffer, system_prompt=system_prompt,model_id=model_id,ask_expert=id2domain)
        else:
            result, status=conversation_query(user_prompt=speech_buffer)
        if status==0:
            spinner.succeed(result)  
        else:
            spinner.fail(result)        
        speech_buffer=""
        ai_voice_text=result
        return speech_buffer            

    @Halo(text='reset:', spinner='dots',color='blue')
    def conversation_reset():
        global speech_buffer
        global ai_voice_text    
        global RUNNING_STATUS
        global IN_SPEECH_RECORDING
        global IN_CONVERSATION_MODE    
        global WAIT_USER_RESPONSE   
        global conversation_history        
        console.print(f":relieved: conversarion_reset",style="bold")
        speaker_announce(instruction="resetting success. start new question.")  
        RUNNING_STATUS='reset'
        transcribed_text=""
        speech_buffer=""
        ai_voice_text=""
        IN_SPEECH_RECORDING=False
        IN_CONVERSATION_MODE=False 
        WAIT_USER_RESPONSE=True
        conversation_history=[]
        return transcribed_text                         

    @Halo(spinner='dots',color='yellow')
    def show_recording_text(transcribed_text:str):
        console.print(f"recorded :pencil2: [blue]{transcribed_text}", style="bold")        
        #console.log(f"show_recording_text 🎤")
        #speaker_announce(instruction="okay, I am ready for next question.")                

    @Halo(spinner='dots',color='green')
    def show_detected_text(transcribed_text:str):
        console.print(f"detected: [green]{transcribed_text}",style="bold")        
        #console.log(f"show_detected_text 🎤 💡")
        #speaker_announce(instruction="okay, I am ready for next question.")                

    def get_sentence_last_word(transcribed_text:str)->str:
        res = transcribed_text.split(" ")
        sentence_last_word=res.pop()
        #console.print(f"\nlast word: {sentence_last_word}")
        logger.debug(f"last word: {sentence_last_word}")        
        sentence_last_word=sentence_last_word.replace('.',"").strip()
        sentence_last_word=sentence_last_word.replace('?',"").strip()        
        return sentence_last_word
    
    ## process speech    
    transcribed_text,transcribed_lang=whisper_transcribe(whisper_model,audio_chunk,language=_CONVERSATION_LANGUAGE)
    transcribed_text=transcribed_text.lower()
    sentence_last_word=get_sentence_last_word(transcribed_text)

    found_start_conversation_words=match_talkie_codes(transcribed_text,start_conversation_words)
    
    if len(found_start_conversation_words)>0 or _CONVERSATION_BEGIN_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(_CONVERSATION_BEGIN_WORD,"")                
        transcribed_text=conversation_begin()
        ASK_EXPERT=False        
        WAIT_USER_RESPONSE=True
        return 
    if isvalid_command(sentence_last_word,"appointment booking"):        
        WAIT_USER_RESPONSE=False
        spinner.warn("book appointment...")          
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")        
        speaker_announce(instruction="book appointment...",output_voice=output_voice)          
        #transcribed_text=conversation_begin()
        WAIT_USER_RESPONSE=True
        return     
    if _CONVERSATION_END_WORD in sentence_last_word \
        or _CONVERSATION_CLEAR_WORD in sentence_last_word \
            or isvalid_command(sentence_last_word,"reset") \
                or isvalid_command(sentence_last_word,"clear"):    
        spinner.warn("reset...")  
        WAIT_USER_RESPONSE=False        
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        transcribed_text=conversation_reset() 
        WAIT_USER_RESPONSE=True 
        IN_CONVERSATION_MODE=False 
        # restart the conversation
        WAIT_USER_RESPONSE=False
        transcribed_text=transcribed_text.replace(_CONVERSATION_BEGIN_WORD,"")                
        transcribed_text=conversation_begin()
        WAIT_USER_RESPONSE=True
        ASK_EXPERT=False
        return 
    if isvalid_command(input_command=sentence_last_word,target_command="correction"):
        spinner.warn("correction...")  
        WAIT_USER_RESPONSE=False        
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        transcribed_text=""
        speech_buffer="" 
        speaker_announce(instruction="correction success. please try ask question again.",output_voice=output_voice)  
        WAIT_USER_RESPONSE=True         
        return         
    if isvalid_command(sentence_last_word,"say it again") or isvalid_command(sentence_last_word,"repeat"):
        spinner.warn("say again...")  
        WAIT_USER_RESPONSE=False      
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        speaker_announce(instruction=ai_voice_text,output_voice=output_voice)    
        transcribed_text=""
        speech_buffer="" 
        WAIT_USER_RESPONSE=True 
        return         
    ask_experts_words=match_talkie_codes(transcribed_text,ask_expert_help_words,acceptable_ratio=0.6)
    if len(ask_experts_words)>0:
    #if isvalid_command(sentence_last_word,"say it again") or isvalid_command(sentence_last_word,"repeat"):
        spinner.warn("seeking expert help...")
        ai_voice_text="gotchat, please briefly describe the problem or require skill-sets or expertises."
        speaker_announce(instruction=ai_voice_text,output_voice=output_voice)              
        WAIT_USER_RESPONSE=False      
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        ASK_EXPERT=True
        #speaker_announce(instruction=ai_voice_text,output_voice=output_voice)    
        transcribed_text=""
        speech_buffer="" 
        WAIT_USER_RESPONSE=True 
        return         
    # keep speech recording
    if WAIT_USER_RESPONSE:
        speech_buffer=speech_buffer+" "+transcribed_text
        speech_buffer=speech_buffer.replace(ai_voice_text,"").strip()        
        speech_buffer=speech_buffer.strip()
        spinner.info(f"speech: {speech_buffer}")         
        show_recording_text(transcribed_text)
    else:
        show_detected_text(transcribed_text)   

    # running continue conversation mode
    if IN_CONVERSATION_MODE and WAIT_USER_RESPONSE:
        # convert word to english
        found_action_words=match_talkie_codes(sentence_last_word,talkie_codes_okay_action)
        ##found_action_words = set(talkie_codes_okay_action) & set({sentence_last_word})
        if len(found_action_words)>0:
            WAIT_USER_RESPONSE=False 
            # discard last word
            transcribed_text=transcribed_text.replace(sentence_last_word,"")
            speech_buffer=conversation_mode(sentence_last_word,transcribed_text,speech_buffer)
            WAIT_USER_RESPONSE=True 
            return 

def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')

# if __name__=="__main__":
#     print(system_config)
#     test_str="""
#     Basically it loops through the text in chunks of the length 1023 and finds the last occurence of "\n" using python .rfind(), which the algorithm then uses as the start_idx.
#     Just make sure you have "\n" appended to the end, else the loop will never end as it always searches for the next linebreak as long as the end_idx is smaller than the length of the string.
#     """    
#     #test_str="""hello""" 
#     slice_text_into_chunks(test_str)
