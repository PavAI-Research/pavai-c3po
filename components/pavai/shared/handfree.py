from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

from rich import print,pretty,console
from rich.console import Console
from rich.progress import Progress

import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
import random
random.seed(0)
import numpy as np
np.random.seed(0)
#import nltk
#nltk.download('punkt')
import os
CPUs = os.cpu_count()
torch.set_num_threads(int(CPUs/2))

from halo import Halo
import traceback
import random
from datetime import datetime
import sounddevice as sd
from faster_whisper import WhisperModel
import faster_whisper
import pavai.shared.system_checks as system_checks
import pavai.shared.audio.stt_vad as stt_vad
import pavai.shared.audio.tts_client as tts_client 
import pavai.llmone.llmtokens as llmtokens
import pavai.llmone.chatprompt as chatprompt 
import pavai.llmone.llmproxy as llmproxy

console = Console()

# user settings
TALKIER_SYS_VOICE=config.system_config["TALKIER_SYS_VOICE"]
TALKIER_SYS_WAKEUP_WORD=config.system_config["TALKIER_SYS_WAKEUP_WORD"]

TALKIER_USER_VOICE=config.system_config["TALKIER_USER_VOICE"]
TALKIER_USER_WAKEUP_WORD=config.system_config["TALKIER_USER_WAKEUP_WORD"]


TALKIER_ACTIVATE_VOICE = TALKIER_USER_VOICE
## WHOLE TEAM
# TALKIER_MARK_VOICE="mark_real"
# TALKIER_MARK_WAKEUP_WORD="mark"

TALKIER_ANTHONY_VOICE="anthony_real"
TALKIER_ANTHONY_WAKEUP_WORD="activate anthony"

TALKIER_C3PO_VOICE="c3po_01"
TALKIER_C3PO_WAKEUP_WORD="activate c3po"

TALKIER_LUKE_VOICE="luke_force"
TALKIER_LUKE_WAKEUP_WORD="activate luke"

TALKIER_YODA_VOICE="yoda_force"
TALKIER_YODA_WAKEUP_WORD="activate yoda"

TALKIER_LEIA_VOICE="leia_01"
TALKIER_LEIA_WAKEUP_WORD="activate leia"


_USE_VOICE_API=False
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

def system_startup(output_voice:str=None):
    global TALKIER_ACTIVATE_VOICE
    SYSTEM_READY=False        
    try:
        output_voice = TALKIER_SYS_VOICE if output_voice is None else output_voice
        SYSTEM_READY=system_checks.pavai_talkie_system_health_check(output_voice=output_voice)        
        console.print("\n[blue]System Ready...")             
    except Exception as e:       
        traceback.print_exc()        
        print("An error occurred:", repr(e))                
        logger.error(f"An exception occurred at system_startup!")
        raise SystemExit("program exited")

    if SYSTEM_READY:
        text_speaker_ai(f"system is ready. now turn to you {TALKIER_USER_VOICE}",output_voice=TALKIER_SYS_VOICE)     
        text_speaker_ai(f"thanks! my name is {TALKIER_USER_VOICE}. I am your voice assistant.",output_voice=TALKIER_USER_VOICE)             
        text_speaker_ai(f"to start a conversation, just call my name. ask your question then say please at the end.",output_voice=TALKIER_USER_VOICE)             
        text_speaker_ai(f"yep. you got it. so how may I help you today?",output_voice=TALKIER_USER_VOICE)                     
        console.print(f"[green]to start a conversation, just call my name:{TALKIER_USER_VOICE}. ask your question then say please at the end.[/green]")
        console.print(f"[green]you got it. so how may I help you today?[/green]")
    else:
        text_speaker_ai("Oops, system startup check failed!. please check console log.",output_voice=TALKIER_SYS_VOICE)   
        console.print(f"[red]Oops, system startup check failed!. please check console log.[/red]")        

def text_speaker_ai(text:str,output_voice:str=None):
    output_voice = TALKIER_USER_VOICE if output_voice is None else output_voice
    if TALKIER_ACTIVATE_VOICE != TALKIER_USER_VOICE:
        output_voice = TALKIER_ACTIVATE_VOICE
    logger.info(f"Active Voice: {output_voice}")
    tts_client.system_tts_local(text=text, output_voice=output_voice)

def text_speaker_human(sd,text:str,output_voice:str=None,vosk_params=None):
    output_voice = TALKIER_USER_VOICE if output_voice is None else output_voice    
    text_speaker_ai(text=text, output_voice=output_voice)

## -----------------------------------------
## LLM
## -----------------------------------------
# def chat_service(user_prompt:str=None, history:list=[], 
#                    system_prompt: str =chatprompt.system_prompt_assistant,
#                    stop_criterias=["</s>"]):    
#     messages, history, reply = llmproxy.chat_api_local(user_Prompt=user_prompt, history= history,system_prompt = system_prompt,stop_criterias=stop_criterias)
#     return reply,history

def process_conversation(new_query:str,history:list, context_size:int=4096*2, 
                       system_prompt: str = "You are an intelligent AI assistant who can help answer user query.",
                       stop_criterias=["</s>"],output_voice:str=None):

    history_buffer = llmtokens.HistoryTokenBuffer(history=history,max_tokens=context_size)
    history_buffer.update(new_query)
    console.print(f"[gray]context token counts available: {history_buffer.token_count} / max {history_buffer.max_tokens}[/gray]")

    conversation_system_prompt = chatprompt.safe_system_prompt
    new_query = new_query + chatprompt.short_response

    if len(history_buffer.overflow_summary)>0:
        # reduce and compress history content 
        logger.info(f"creating overflow summary: {history_buffer.overflow_summary}")
        summary="provide an summary of following text:\n "+history_buffer.overflow_summary
        assistant_response, history = llmproxy.chat_api(user_prompt=new_query, 
                                                        history=history,system_prompt=conversation_system_prompt,
                                                        stop_criterias=stop_criterias)
        
        output_voice = TALKIER_USER_VOICE if output_voice is None else output_voice
        text_speaker_ai(text=assistant_response,output_voice=output_voice)       
        history_buffer.update(summary) 
        history=history_buffer.history
        history_buffer.overflow_summary=""
    # perform new query now
    assistant_response,history = llmproxy.chat_api(user_prompt=new_query, history=history,
                                                 system_prompt=conversation_system_prompt,
                                                 stop_criterias=stop_criterias)
    #console.log(f"\nAI: {assistant_response}\n")
    return assistant_response, history 

## -----------------------------------------
## Speech-to-Text
## -----------------------------------------
def _get_whisper_model(model_path:str="large-v2",
                      download_root="resources/models/whisper",
                      local_files_only=True):
    model = faster_whisper.WhisperModel(model_size_or_path=model_path,
                                            download_root=download_root,
                                            local_files_only=local_files_only,
                                            device='cuda' if torch.cuda.is_available() else 'cpu')
    return model

def init_whisper(model_path:str="large-v2",download_root="resources/models/whisper",local_files_only=True):
    model=None
    try:
        model = _get_whisper_model(model_path=model_path,local_files_only=True)
    except Exception as e:
        traceback.print_exc()        
        print("An error occurred:", e)                
        logger.error(f"An exception occurred at init_whisper!")
        # otherwise, attempt download from huggingface        
        model = _get_whisper_model(model_path=model_path,download_root=download_root,local_files_only=False)
    return model

def whisper_transcribe(model:WhisperModel,audio,language:str=None):
    try:
        ## trasnscribe all into english only
        segments, language = model.transcribe(audio, 
                                    task= "translate",
                                    language=language if language else None,
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
##_CONVERSATION_BEGIN_WORD=system_checks.DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT.lower() if system_checks.DEFAULT_TTS_VOICE_MODEL_TALKIE_AGENT is not None else "ryan"
_CONVERSATION_BEGIN_WORD="jane"

talkie_codes_okay_action=["copy","affirmative","roger","over","out","roger that","please","thank you","thanks","got it","okay","continue"]
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


def wakeup_greeting(output_voice:str=None):
    now = datetime.now()
    time_now = get_time_of_day(now.hour)
    greeting_messages = ['hi, I am listening...', 
                         'hi, I am here.', 
                         'hi, there', 
                         'hey? what is up?', 
                         'hi! how can I help you?',
                         'hello, tell me what you need.', 
                         'aha yeah!',
                         'hello, hello, yes!',                         
                         'yes, listening!',                         
                         'hi, what is up?']
    greeting = str(random.choice(greeting_messages))
    output_voice = TALKIER_SYS_VOICE if output_voice is None else output_voice
    text_speaker_ai(text=greeting,output_voice=output_voice)    
    console.print("\n[red]Speak now...") 
    #os.system(f"spd-say {greeting} sir!")    
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
    vad_model,utils = stt_vad.init_vad_model(silaro_model_name=silaro_model_name,silero_use_onnx=True)
    (get_speech_timestamps,save_audio,read_audio,VADIterator,collect_chunks)=utils
    # whisper model
    whisper_model = init_whisper(model_path=whisper_model_path)
    # wake up detector
    #porcupine = init_porcupine(wake_words="ryan",wake_words_sensitivity=0.5) 
    # start audio with VAD
    vad_audio=stt_vad.init_vadaudio(webRTC_aggressiveness,device,rate)
    # Stream from microphone to DeepSpeech using VAD
    text="[🎤] Listening... (ctrl-C to exit)..."
    #console.print("[green]"+text)   
    console.print(f"[green]{TALKIER_USER_VOICE} is ready :smiley:")    

def activate_handfree_system(reload:bool=False,nospinner:bool=False):
    """activate_handfree_system"""
    global spinner
    if not nospinner:
        spinner = Halo(spinner='line',text_color="blue")
    wav_data = bytearray()
    spinner.succeed(f"{system_checks.PAVAI_APP_TALKIE} is listening...🎤")    
    #console.print("start conversation",end="\n") 
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

    wav_data = bytearray() 
    if spinner:
        spinner.start("detecting:")    
    if WAIT_USER_RESPONSE:
        logger.debug("waiting user response - stop detecting")
        return 
    for frame in frames:
        if frame is not None:
            wav_data.extend(frame)
        else:
            audio_chunk=stt_vad.normalize_audio_chunks(wav_data)
            time_stamps=stt_vad.has_speech_activity(vad_model,utils,audio_chunk) 
            if (len(time_stamps) > 0):
                with Progress(transient=False) as progress: 
                    task = progress.add_task("thinking...", total=1)
                    moderate_conversation(whisper_model,audio_chunk,language="en")
                    progress.advance(task)
            else:
                #console.print("silero VAD has detected a noise", end="\n")
                wav_data=bytearray() 
                # discard unused text not in speech
            wav_data=bytearray()           
        if RUNNING_STATUS=="user_query_end" or RUNNING_STATUS=="reset":
            frames = None
            RUNNING_STATUS ="detecting"
            IN_CONVERSATION_MODE=False
            IN_SPEECH_RECORDING=False
            break

def moderate_conversation(whisper_model,audio_chunk,language:str=_CONVERSATION_LANGUAGE, output_voice:str=None):
    """handle_detected_speech"""
    global spinner
    global speech_buffer
    global ai_voice_text    
    global conversation_history    
    global RUNNING_STATUS
    global IN_SPEECH_RECORDING
    global IN_CONVERSATION_MODE    
    global WAIT_USER_RESPONSE      
    global TALKIER_ACTIVATE_VOICE

    output_voice = TALKIER_USER_VOICE if output_voice is None else output_voice
    if TALKIER_ACTIVATE_VOICE!=TALKIER_USER_VOICE:
       output_voice= TALKIER_ACTIVATE_VOICE
    
    #@Halo(text='conversation begin:', spinner='dots',color='green')
    def conversation_begin(output_voice:str=None):
        global speech_buffer
        global ai_voice_text    
        global RUNNING_STATUS
        global IN_SPEECH_RECORDING
        global IN_CONVERSATION_MODE    
        global WAIT_USER_RESPONSE 
        #console.log(f"conversarion_begin")
        spinner.enabled=False        
        ai_voice_text=wakeup_greeting(output_voice)   
        transcribed_text=""
        speech_buffer=""       
        IN_SPEECH_RECORDING=True 
        RUNNING_STATUS="user_query_begin"
        WAIT_USER_RESPONSE=True
        IN_CONVERSATION_MODE=True        
        return transcribed_text       

    #@Halo(spinner='dots', color='red', animation='marquee')
    def conversation_query(user_prompt:str,output_voice:str=None):
        global ai_voice_text   
        global conversation_history
        if user_prompt is None or len(user_prompt.strip())==0:
            return  "skip", 0
        # speaker_announce(text="working on it",output_voice=output_voice)   
        user_prompt=user_prompt.strip()            
        try:
            console.print(f"[bold]Query:[/bold]: [blue]{user_prompt}[/blue]")            
            assistant_response, conversation_history =process_conversation(user_prompt,conversation_history) 
            ai_voice_text=assistant_response
            status=0
            console.print(f"[bold]Answer:[/bold]: [magenta]{assistant_response}[/magenta]")            
            #time.sleep(1)
            if assistant_response is not None:
                assistant_response=assistant_response.replace("\n"," ")             
            text_speaker_ai(text=f"{assistant_response}",output_voice=output_voice) #is that good?                           
        except Exception as e:
            status=1
            traceback.print_exc()        
            print("An error occurred:", e)                
            logger.error(f"An exception occurred at conversation_query!")
            text_speaker_ai(text="oops!, request failed due error. please check console log.",output_voice=TALKIER_SYS_VOICE) 
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
        global TALKIER_ACTIVATE_VOICE
        RUNNING_STATUS="conversation"   
        console.print(f":loudspeaker: {transcribed_text}",style="bold")
        transcribed_text=transcribed_text.rsplit(' ', 1)[0]
        transcribed_text=transcribed_text.replace(sentence_last_word,"")
        # drop ai spoken words if exist
        if speech_buffer is None:
            speech_buffer=transcribed_text
        speech_buffer=speech_buffer.replace(ai_voice_text,"").strip()
        speech_buffer=speech_buffer.replace(sentence_last_word,"").strip()
        result, status=conversation_query(user_prompt=speech_buffer, output_voice=TALKIER_ACTIVATE_VOICE)
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
        text_speaker_ai(text="resetting success. start new question.")  
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
    ##if _CONVERSATION_BEGIN_WORD in transcribed_text.lower():
    if TALKIER_SYS_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_SYS_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_SYS_VOICE)
        WAIT_USER_RESPONSE=True
        return     
    if TALKIER_USER_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_USER_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_USER_VOICE)
        WAIT_USER_RESPONSE=True
        TALKIER_ACTIVATE_VOICE=TALKIER_USER_VOICE
        return     
    if TALKIER_C3PO_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_C3PO_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_C3PO_VOICE)
        WAIT_USER_RESPONSE=True
        TALKIER_ACTIVATE_VOICE=TALKIER_C3PO_VOICE        
        return     
    if TALKIER_LUKE_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_LUKE_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_LUKE_VOICE)
        WAIT_USER_RESPONSE=True
        TALKIER_ACTIVATE_VOICE=TALKIER_LUKE_VOICE                
        return     
    if TALKIER_YODA_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_YODA_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_YODA_VOICE)
        WAIT_USER_RESPONSE=True
        TALKIER_ACTIVATE_VOICE=TALKIER_YODA_VOICE                        
        return     
    if TALKIER_ANTHONY_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_ANTHONY_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_ANTHONY_VOICE)
        WAIT_USER_RESPONSE=True
        TALKIER_ACTIVATE_VOICE=TALKIER_ANTHONY_VOICE                                
        return     
    if TALKIER_LEIA_WAKEUP_WORD in transcribed_text.lower():
        WAIT_USER_RESPONSE=False
        # discard last word
        transcribed_text=transcribed_text.replace(TALKIER_LEIA_WAKEUP_WORD,"")                
        transcribed_text=conversation_begin(output_voice=TALKIER_LEIA_VOICE)
        TALKIER_ACTIVATE_VOICE=TALKIER_LEIA_VOICE                                        
        WAIT_USER_RESPONSE=True
        return     

    if _CONVERSATION_END_WORD in sentence_last_word or _CONVERSATION_CLEAR_WORD in sentence_last_word:
        spinner.warn("reset...")  
        WAIT_USER_RESPONSE=False        
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        transcribed_text=conversation_reset() 
        WAIT_USER_RESPONSE=True 
        IN_CONVERSATION_MODE=False 
        return 
    if "correction" in sentence_last_word:
        spinner.warn("correction...")  
        WAIT_USER_RESPONSE=False        
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        transcribed_text=""
        speech_buffer="" 
        text_speaker_ai(text="correction success. please try ask question again.",output_voice=output_voice)  
        WAIT_USER_RESPONSE=True         
        return     
    if "say again" in sentence_last_word or "repeat" in sentence_last_word:
        spinner.warn("say again...")  
        WAIT_USER_RESPONSE=False      
        # discard last word
        transcribed_text=transcribed_text.replace(sentence_last_word,"")                
        text_speaker_ai(text=ai_voice_text,output_voice=output_voice)    
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
        found_action_words = set(talkie_codes_okay_action) & set({sentence_last_word})
        if len(found_action_words)>0:
            WAIT_USER_RESPONSE=False 
            # discard last word
            transcribed_text=transcribed_text.replace(sentence_last_word,"")
            speech_buffer=conversation_mode(sentence_last_word,transcribed_text,speech_buffer)
            WAIT_USER_RESPONSE=True 
            return 

def clear_console():
    os.system('clear' if os.name == 'posix' else 'cls')

def slice_text_into_chunks(input_text:str, chunk_size:int=100):
    if len(input_text)<chunk_size:
        return [input_text]
    else:
        K = int(len(input_text)/100)    
        logger.info(f"The original string is: {str(input_text)} size of {len(input_text)} in chunks: {K}")
        # compute chunk length 
        chnk_len = len(input_text) // K
        result = []
        for idx in range(0, len(input_text), chnk_len):
            result.append(input_text[idx : idx + chnk_len])
        logger.info(f"The K chunked list {str(result)}") 
        return result

def lpad_text(text:str, max_length:int=43, endingchar:str="c")->str:
    if len(text) < max_length:
        text=text.ljust(max_length, '…')
    #print("lpad_text size:", len(text))
    return text+"."

if __name__=="__main__":
    wakeup_greeting(output_voice="jane")
    #from pavai.shared.styletts2.text_utils import TextCleaner
    # Creating a string variable
    greeting = "hey, what is up?"
    #speechtext= lpad_text(text=greeting)

    output_voice="jane"
    # text_speaker_ai(text=greeting,output_voice=output_voice)   
    # greeting="""
    # Basically it loops through the s.
    # """
    #tokens = TextCleaner(greeting)
    #print("tokens: ",tokens)    
    # print("text length: ",len(speechtext))
    # tts_client.speaker_text(text=speechtext, output_voice=output_voice)

    greeting="""
    Basically it loops through the text in chunks of the length 1023 and finds the last occurence of "\n" using python .rfind(), which the algorithm then uses as the start_idx.
    Just make sure you have "\n" appended to the end, else the loop will never end as it always searches for the next linebreak as long as the end_idx is smaller than the length of the string.
    """    
    #tts_client.system_tts_local(text=greeting, output_voice=output_voice)

    # #test_str="""hello""" 
    # slice_text_into_chunks(test_str)