# pip install python-dotenv
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
from rich.logging import RichHandler
from rich import pretty
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()
import warnings 
warnings.filterwarnings("ignore")

import traceback
import os, sys
import gradio as gr
import torch
import pandas as pd
import numpy as np
#import pavai.shared.datasecurity as datasecurity
from typing import BinaryIO, Union
from transformers.utils import is_flash_attn_2_available
#from pavai.shared.audio.voices_piper import (text_to_speech, speak_acknowledge,speak_wait, speak_done, speak_instruction)
from pavai.shared.system_checks import (pavai_vocie_system_health_check, DEFAULT_SYSTEM_MODE, SYSTEM_THEME_SOFT,SYSTEM_THEME_GLASS,VOICE_PROMPT_INSTRUCTIONS_TEXT)
from pavai.shared.audio.transcribe import (speech_to_text, FasterTranscriber,DEFAULT_WHISPER_MODEL_SIZE)
#from pavai.shared.llmproxy import chatbot_ui_client,chat_count_tokens,multimodal_ui_client
from pavai.shared.llmproxy import chat_count_tokens,multimodal_ui_client
from pavai.shared.image.text2image import (StableDiffusionXL, image_generation_client,DEFAULT_TEXT_TO_IMAGE_MODEL)
from pavai.shared.fileutil import get_text_file_content
from pavai.shared.commands import filter_commmand_keywords
from pavai.shared.grammar import (fix_grammar_error)
from pavai.shared.audio.tts_client import (system_tts_local,speaker_file_v2,get_speaker_audio_file,speak_acknowledge,speak_wait, speak_done, speak_instruction)
## get_speaker_audio_file, 
## remove Iterable, List, NamedTuple, Optional, Tuple,
## removed from os.path import dirname, join, abspath
from pavai.shared.aio.llmchat import (system_prompt_assistant, DEFAULT_LLM_CONTEXT_SIZE)
from pavai.shared.llmcatalog import LLM_MODEL_KX_CATALOG_TEXT
from pavai.shared.solar.llmprompt import knowledge_experts_system_prompts
from pavai.shared.aio.llmchat import get_llm_library
from pavai.vocei_web.system_settings_ui import SystemSetting

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2024"
__version__ = "0.0.3"

print("--GLOBAL SYSTEM MODE----")
print(system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=system_config["GLOBAL_TTS"]

# tested version gradio version: 4.7.1
# pip install gradio==4.7.1
# whisper model
DEFAULT_COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
DEFAULT_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
INT16_MAX_ABS_VALUE = 32768.0
DEFAULT_MAX_MIC_RECORD_LENGTH_IN_SECONDS = 30*60*60  # 30 miniutes
DEFAULT_WORKING_DIR = "./workspace"

# Global variables
system_is_ready = True
use_device = "cuda:0" if torch.cuda.is_available() else "cpu"
use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()

faster_transcriber = FasterTranscriber(model_id_or_path=DEFAULT_WHISPER_MODEL_SIZE)
stablediffusion_model = StableDiffusionXL(model_id_or_path=DEFAULT_TEXT_TO_IMAGE_MODEL)

# Knowledge experts and Domain Models
knowledge_experts = list(knowledge_experts_system_prompts.keys())
domain_models = list(get_llm_library().keys())
# global settings
_QUERY_ASK_EXPERT_ID=None  # planner
_QUERY_ASK_EXPERT_PROMPT=None  # default
_QUERY_LLM_MODEL_ID=None #"zephyr-7b-beta.Q4_K_M.gguf"
_QUERY_LLM_MODEL_INFO=None # list of [id,id,ful_url] 
# data security check - PII
_QUERY_ENABLE_PII_ANALYSIS=False 
_QUERY_ENABLE_PII_ANONYMIZATION=False
# content safety check
_QUERY_CONTENT_SAFETY_GUARD=False

# turn-off chatbot voice response
_TURN_OFF_CHATBOT_RESPONSE_VOICE=False

# uploaded image file
_USER_UPLOADED_IMAGE_FILE=None
_USER_UPLOADED_IMAGE_DEFAULT_QUERY="describe what does the image say?"
_USER_UPLOADED_IMAGE_FILE_DESCRIPTION=None
_TURN_OFF_IMAGE_CHAT_MODE=True


"""UI-CHATBOT SPEAKER"""
class ChatbotSpeaker(SystemSetting):

    def cb_text_to_speech(self,text: str, output_voice: str = "jane", mute=False, autoplay=False):
        out_file=None
        if not _TURN_OFF_CHATBOT_RESPONSE_VOICE:
            out_file = speaker_file_v2(text=text, output_voice=output_voice, mute=mute, autoplay=autoplay)
        return out_file

    def cb_hide_outputs(self,enable_text_to_image: bool = False):
        if enable_text_to_image:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=False)

    def cb_filter_commmand_keywords(self,input_text, history):
        global _USER_UPLOADED_IMAGE_FILE
        global _USER_UPLOADED_IMAGE_DEFAULT_QUERY
        output_text, history, output_filename, task_command, language = filter_commmand_keywords(
            input_text, history)
        ### redirect image chat command to multimodal chat
        _image_chat_command = "/imagechat?"
        if task_command==_image_chat_command:
            result_text=output_text.split("???")
            _USER_UPLOADED_IMAGE_FILE = result_text[1]
            _USER_UPLOADED_IMAGE_DEFAULT_QUERY = result_text[0].replace('%20',' ')
            output_text=_USER_UPLOADED_IMAGE_DEFAULT_QUERY

        return output_text, history, output_filename, task_command, language

    def cb_add_file(self,history, file):
        filename = None
        if file is not None:
            if isinstance(file, str):
                filename = file
            else:
                filename = file.name
            history = history + [((filename), None)]
            logger.info(f"add_file: {file} | {history}")
            return history, filename
        else:
            return history, None
        
    def cb_process_upload_image(self,text: str, history:list, output_voice: str = "en", mute=False,
                                autoplay=False, prefix_text="multimodal image file:\n"):
        global _USER_UPLOADED_IMAGE_FILE
        logger.info(f"process_upload_image_file: {text}")
        gr.Info("processing upload image file.")        
        if mute or text is None or len(text.strip()) == 0:
            return None
        output_voice = "en" if output_voice is None else output_voice
        history = [] if history is None else history
        if text.lower().endswith(".png") or text.lower().endswith(".jpg") or text.lower().endswith(".jpeg"):
            gr.Info(f"working on upload file {text}")
            _USER_UPLOADED_IMAGE_FILE = "file:///"+text
            file_contents = "file upload success"
            history = history + [[file_contents, None]]
            return history, text, "process_upload_image"        
        else:
            return history, text, "process_upload_image"

    def cb_process_upload_audio(self,text: str, history, output_voice: str = "en", mute=False,
                                autoplay=False, prefix_text="transcribed audio file:\n"):
        logger.info(f"process_upload_file_audio: {text}")
        gr.Info("processing upload audio file.")        
        if mute or text is None or len(text.strip()) == 0:
            return None
        output_voice = "en" if output_voice is None else output_voice
        history = [] if history is None else history
        if text.endswith(".mp3") or text.endswith(".wav"):
            file_path = text
            file_contents, detected_language = speech_to_text(input_audio=text, task_mode="transcribe")
            file_contents = prefix_text+file_contents
            history = history + [[file_contents, None]]
            return history, "processed upload file, any question?", "process_upload_audio"
        else:
            return history, text, "process_upload_audio"

    def cb_process_upload_text(self,text: str, history, language: str = "en"):
        logger.info(f"process_upload_text: {text}")
        gr.Info("processing upload text file.")    
        if text is None or len(text.strip()) == 0:
            return None
        language = "en" if language is None else language
        history = [] if history is None else history
        text_file = text.lower()
        if text_file.endswith(".txt") or text_file.endswith(".md") \
                or text_file.endswith(".html") or text_file.endswith(".pdf"):
            file_contents, history, file_path = get_text_file_content(
                file_path=text_file, history=history, prefix_text="get file content:\n")
            return history, "processed upload file, any question?", "process_upload_text"
        else:
            return history, text, "process_upload_text"

    def cb_update_chat_session(self,chat_history: list = [],
                            llm_max_context: int = DEFAULT_LLM_CONTEXT_SIZE,
                            system_prompt: str = system_prompt_assistant):
        """
        history.append({"role": "assistant", "content": reply})
        """
        gr.Info("update_chat_session...")
        # global llm
        if chat_history is None or len(chat_history) == 0:
            logger.debug("empty chat history")
            return [], chat_history, 123
        llm_max_context = llm_max_context-(llm_max_context*0.10)  # keep 10% free
        chat_tokens_size = 0
        for row in chat_history:
            if isinstance(row["content"],str):
                chat_tokens_size += chat_count_tokens(row["content"])
        logger.info(f"session tokens: {chat_tokens_size}")
        if chat_tokens_size > llm_max_context:
            # reduce exceed tokens
            gr.Warning("summarized conversation due tokens exceed model max context size")
            while chat_tokens_size > llm_max_context:
                chat_row = chat_history.pop(0)
                row_tokens = chat_count_tokens(chat_row["content"])
                chat_tokens_size = chat_tokens_size-row_tokens
            # summarize history to avoid hitting LLM max content
            summary_Prompt = "summarize current conversations history: " 
            messages, chat_history, reply = self.chat_client(input_text=summary_Prompt, chat_history=chat_history)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary_Prompt},
                {"role": "assistant", "content": reply}
            ]
            chat_history = []
            chat_history = chat_history+messages
        messages = []
        for i in range(0, len(chat_history)-1):
            if chat_history[i]["role"] != "system":
                messages.append((chat_history[i]["content"], chat_history[i+1]["content"]))
        logger.debug(chat_history)
        logger.info(messages)
        return messages, chat_history, []

    def cb_reset_chat_outputs(self):
        return gr.update(value="", visible=False, interactive=True), gr.update(value="", visible=False, interactive=True)

    def turn_off_chatbot_voice_options(self,x):
        global _TURN_OFF_CHATBOT_RESPONSE_VOICE
        logger.warn(f"change turn_off_voice_response_options: {x}")
        _TURN_OFF_CHATBOT_RESPONSE_VOICE = x

    def select_content_safety_options(self,x):
        global _QUERY_CONTENT_SAFETY_GUARD
        logger.warn(f"change content_safety_options: {x}")
        _QUERY_CONTENT_SAFETY_GUARD = x

    def select_expert_option(self,x):
        global _QUERY_ASK_EXPERT_PROMPT
        global _QUERY_ASK_EXPERT_ID
        logger.warn(f"change knowledge expertise: {x}")
        expert_system_prompt = knowledge_experts_system_prompts[x]
        logger.debug("system prompts", expert_system_prompt)
        _QUERY_ASK_EXPERT_PROMPT = expert_system_prompt
        _QUERY_ASK_EXPERT_ID=x

    def select_model_option(self,x):
        global _QUERY_LLM_MODEL_INFO    
        global _QUERY_LLM_MODEL_ID    
        logger.warn(f"change domain model: {x}")
        domain_model = get_llm_library()[x]
        logger.debug(f"system prompts {domain_model}")    
        _QUERY_LLM_MODEL_INFO=domain_model    
        _QUERY_LLM_MODEL_ID=x

    def turn_off_image_chat_options(self,x):
        global _TURN_OFF_IMAGE_CHAT_MODE
        logger.warn(f"change turn_off_image_chat_options: {x}")
        _USER_UPLOADED_IMAGE_FILE = None
        _TURN_OFF_IMAGE_CHAT_MODE = x

    def _mute_voice_speaker_for_these_commands_output(self,task_command: str) -> bool:
        logger.debug(f"skip_command_output: {task_command}")
        transcribe_youtube_command = "/disable_transcribe_youtube:"
        translate_youtube_command = "/disable_translate_youtube:"
        load_text_file_command = "/disable_load_text:"
        # speak_wait()
        if transcribe_youtube_command in task_command \
            or translate_youtube_command in task_command \
                or load_text_file_command in task_command:
            speak_instruction("process command: "+task_command)
            return True
        else:
            return False

    def cb_botchat(self,bot_history, chat_history=[], task_command="fn_botchat"):
        global _USER_UPLOADED_IMAGE_FILE
        logger.info("cb_botchat")
        if bot_history is None or len(bot_history) == 0:
            return [], [], ""
        input_text = bot_history[-1][0]
        logger.info(f"cb_botchat: {input_text} task_command:{task_command}")
        if self._mute_voice_speaker_for_these_commands_output(task_command):
            return bot_history, chat_history, input_text
        if input_text is not None and len(input_text) > 0:        
            ## image chat
            if _USER_UPLOADED_IMAGE_FILE is not None:
                reply_messages, chat_history,reply_text = self.cb_imagechat(image_file_path=_USER_UPLOADED_IMAGE_FILE,
                                                        user_query=input_text,
                                                        history=chat_history)
                bot_messages=reply_messages  
                #logger.info(f"cb_botchat->image_chat_history:{bot_messages[-1][1]}")            
                return bot_messages, chat_history, reply_text
            else:    
                bot_messages, chat_history, reply_text = self.chat_client(input_text, chat_history)
                ## respond        
                if bot_messages is not None and len(bot_messages) > 0:
                    logger.info(f"cb_botchat->bot_history:{bot_messages[-1][1]}")
                    return bot_messages, chat_history, reply_text
                else:
                    logger.info(f"cb_botchat-> reply error: {reply_text}")
                    _USER_UPLOADED_IMAGE_FILE = None
        
        return bot_history, chat_history, ""

    def cb_imagechat(self,image_file_path:str=None, user_query:str=None, history:list=[]):
        # reject conditions
        if image_file_path is None:
            return history, "missing image file path" , "image chat"
        if user_query is not None:
            if image_file_path == ("file:///".join(user_query)):
                user_query = _USER_UPLOADED_IMAGE_DEFAULT_QUERY
            if user_query=="file upload success":
                user_query = _USER_UPLOADED_IMAGE_DEFAULT_QUERY            
        else:
            user_query = _USER_UPLOADED_IMAGE_DEFAULT_QUERY        
        # process request    :file upload success
        gr.Info(f"imagechat with file {image_file_path}")        
        if image_file_path.lower().startswith("http://") or image_file_path.lower().startswith("https://"):
            reply_messages, history, reply_text = multimodal_ui_client(input_text=image_file_path,
                                                        user_query=user_query,
                                                        chat_history=history)
            return reply_messages, history, reply_text        
        elif image_file_path.lower().endswith(".png") or image_file_path.lower().endswith(".jpg") or image_file_path.lower().endswith(".jpeg"):
            reply_messages, history, reply_text = multimodal_ui_client(input_text=image_file_path,
                                                        user_query=user_query,
                                                        chat_history=history)
            return reply_messages, history, reply_text
        else:
            return [], history, "imagechat url is not valid. the link should end with .png or .jpeg."

    def cb_user_chat(self,user_message, chatbot_history, task_command):
        if user_message is None or len(user_message) == 0:
            gr.Warning("Empty input text! enter text then try again.")
            return
        if self._mute_voice_speaker_for_these_commands_output(task_command):
            return "", chatbot_history
        chatbot_history = chatbot_history + [[user_message, None]]
        return "", chatbot_history

    def cb_sentence_word_splitter(self,num_of_words: int, sentence: str) -> list:
        pieces = sentence.split()
        return [" ".join(pieces[i:i+num_of_words]) for i in range(0, len(pieces), num_of_words)]

    def cb_chunk_sentences_to_max_tokens(self, sentences: list, max_length: int = 256):
        fixed_size_sentences = []
        tokens = sentences.lower().split()
        if len(tokens) > (max_length):
            chunks = self.cb_sentence_word_splitter(num_of_words=max_length, sentence=sentences)
            for achunk in chunks:
                fixed_size_sentences.append(achunk)
        else:
            fixed_size_sentences.append(sentences)
        return fixed_size_sentences

    def cb_convert_text_to_speech(self,input_text:str, output_voice:str, mute=False, autoplay=False, delay:int=33):
        import time
        gr.Info("processing text to speech, please wait!")
        cb_text_chunks = self.cb_chunk_sentences_to_max_tokens(sentences=input_text,max_length=90)
        send_full_audio=False
        print("convert_text_to_speech audio chunks: ",len(cb_text_chunks))
        if len(cb_text_chunks)==1:
            send_full_audio=True
        for chunk in cb_text_chunks:
            audio_file=self.cb_text_to_speech(text=chunk,output_voice=output_voice, mute=mute,autoplay=autoplay)
            if send_full_audio:
                yield  audio_file
            else:           
                time.sleep(delay)
                yield audio_file

    def build_chatbot_speaker_ui(self):
        self.blocks_chatbot_speaker = gr.Blocks(title="CHATBOT-SPEAKER-UI",analytics_enabled=False)
        with self.blocks_chatbot_speaker as llm_chatbot_ui:
            self.history_state = gr.State([])
            self.session_tokens = gr.State(0)

            ## hidden components            
            cb_textbox_detected_spoken_language=gr.Text(visible=False)                    
            cb_bot_output_txt=gr.Text(visible=False)
            available_ai_speaker_langs=["ar","cs","da","de","en","en_ryan","en_ryan_medium","en_ryan_low","en_amy","en_amy_low","en_kusal","en_lessac","en_lessac_low","ru","zh","fr","uk","es"]
            with gr.Accordion("AI Speaker Voice", open=False):            
                checkbox_turn_off_voice_response = gr.Checkbox(
                    label="Turn off Voice Response", interactive=True)
                checkbox_turn_off_voice_response.change(fn=self.turn_off_chatbot_voice_options,inputs=checkbox_turn_off_voice_response)               
                cb_bot_audio_output=gr.Audio(visible=True, autoplay=True)                  
                cb_text_to_speech_target_lang = gr.Dropdown(label="Select a target AI language speaker",choices=available_ai_speaker_langs, value="en")
                cb_text_to_speech_input=gr.TextArea(label="Convert Text to AI voice",max_lines=5,visible=True, placeholder="Type or Paste text here. click convert to AI voice")                                    
                with gr.Row():
                    cb_text_to_speech_convert=gr.Button(value="convert", size="sm", scale=3)
                    cb_text_to_speech_clear=gr.ClearButton(components=[cb_bot_audio_output,cb_text_to_speech_input],
                                                           value="clear", size="sm", scale=1)                
                ## handle voice conversion
                cb_text_to_speech_input.submit(fn=self.cb_convert_text_to_speech, inputs=[cb_text_to_speech_input,cb_text_to_speech_target_lang],outputs=[cb_bot_audio_output])                
                cb_text_to_speech_convert.click(fn=self.cb_convert_text_to_speech, inputs=[cb_text_to_speech_input,cb_text_to_speech_target_lang],outputs=[cb_bot_audio_output])

            with gr.Group():
                with gr.Accordion("Models & Parameters", open=False):
                    llm_textbox_system_prompt = gr.Textbox(system_prompt_assistant, lines=3, label="System Prompt", interactive=True)
                    llm_slider_temperature = gr.Slider(
                        0, 10, label="Temperature", interactive=True)
                    with gr.Row():
                        llm_max_tokens = gr.Textbox(
                            value=256, label="Max tokens", interactive=True)
                        llm_context_size = gr.Textbox(
                            value=6000, label="Context size", interactive=True)
                    with gr.Row():
                        """content safety options"""                                
                        cbcheckbox_content_safety = gr.Checkbox(
                                value=False, label="enable content safety guard",
                                info="enforce content safety check on input and output")    
                        cbcheckbox_content_safety.change(fn=self.select_content_safety_options,inputs=cbcheckbox_content_safety)                                                                           

                        """image chat options"""                                
                        cbcheckbox_turnoff_image_chat_option = gr.Checkbox(
                                value=False, label="turn-off image chat",
                                info="turn-off to reset image chat mode")    
                        cbcheckbox_turnoff_image_chat_option.change(fn=self.turn_off_image_chat_options,
                                                                    inputs=cbcheckbox_turnoff_image_chat_option)                                                                           
                    with gr.Row():
                        """chatbot knowledge and domain model experts"""
                        cb_expert_options = gr.Dropdown(knowledge_experts, label="(AI)Knowledge Experts", info="[optional] route question to subject with domain knowledge.")
                        cb_expert_options.change(fn=self.select_expert_option,inputs=cb_expert_options)        
                        cb_model_options = gr.Dropdown(domain_models, label="Domain Models",info="[optional] use specialized model")
                        cb_model_options.change(fn=self.select_model_option,inputs=cb_model_options)

                chatbot_ui = gr.Chatbot(
                    value=[], elem_id="chatbot",
                    height=490, bubble_full_width=False,
                    layout="bubble", show_copy_button=True, show_share_button=True,
                )
                with gr.Row():
                    textbox_user_prompt = gr.Textbox(scale=4, show_label=True,
                                                    label="User input",
                                                    placeholder="Enter text and press enter, or record an audio",
                                                    container=True, autofocus=True, interactive=True, show_copy_button=True
                                                    )
                with gr.Row():
                    textbox_local_filename = gr.Textbox(scale=5, show_label=True, visible=False,
                                                        label="local filename",
                                                        show_copy_button=True
                                                        )
                    textbox_chatbot_response = gr.Textbox(scale=5, show_label=False, visible=False,
                                                        label="chatbot response",
                                                        show_copy_button=True)
                with gr.Row():
                    btn_new_conversation = gr.ClearButton(components=[textbox_user_prompt, chatbot_ui, self.history_state], value="🗨 New Conversation", scale=2)
                    btn_upload_chat_file = gr.UploadButton(label="📁 Upload a File", size="sm", file_types=["image", "video", "audio"])
                    btn_upload_chat_file.click(lambda: gr.update(
                        visible=True), None, textbox_local_filename, queue=False)
                    # Connect upload button to chatbot
                    file_msg = btn_upload_chat_file.upload(
                        self.cb_add_file,
                        inputs=[chatbot_ui, btn_upload_chat_file],
                        outputs=[chatbot_ui, textbox_local_filename], queue=False
                    ).then(
                        speak_wait, None, None, queue=False
                    )
                    logger.info(f"process uploaded file.")                        
                    file_msg.then(
                        self.cb_process_upload_audio,
                        inputs=[textbox_local_filename, chatbot_ui],
                        outputs=[chatbot_ui, textbox_user_prompt,
                                textbox_chatbot_response], queue=False  
                    ).then(
                        self.cb_process_upload_text,
                        inputs=[textbox_local_filename, chatbot_ui],
                        outputs=[chatbot_ui, textbox_user_prompt,
                                textbox_chatbot_response], queue=False 
                    ).then(
                        self.cb_process_upload_image,
                        inputs=[textbox_local_filename, chatbot_ui],
                        outputs=[chatbot_ui, textbox_user_prompt,
                                textbox_chatbot_response], queue=False 
                    ).then(            
                        self.cb_botchat,
                        inputs=[chatbot_ui, self.history_state, textbox_chatbot_response],
                        outputs=[chatbot_ui, self.history_state,
                                textbox_chatbot_response], queue=False
                    ).then(
                        self.cb_update_chat_session,
                        inputs=[self.history_state],
                        outputs=[chatbot_ui, self.history_state,
                                self.session_tokens], queue=False
                    ).then(
                        speak_done, None, None, queue=False
                    ).then(
                        self.cb_reset_chat_outputs, outputs=[
                            textbox_local_filename, textbox_chatbot_response], queue=True
                    )

                    # Connect new conversation button
                    btn_new_conversation.click(self.cb_hide_outputs,inputs=None,
                                            outputs=[cb_bot_output_txt,cb_bot_audio_output]
                                            ).then(lambda: None, None, chatbot_ui, queue=False
                                                    ).then(lambda: None, [], self.history_state, queue=False
                                                            )
                    # 2.Connect user input to chatbot
                    cb_txt_msg = textbox_user_prompt.submit(self.cb_filter_commmand_keywords,
                                                        inputs=[
                                                            textbox_user_prompt, chatbot_ui],
                                                        outputs=[textbox_user_prompt, chatbot_ui,
                                                                textbox_local_filename, textbox_chatbot_response,
                                                                cb_textbox_detected_spoken_language], queue=False
                                                        ).then(
                    self.cb_add_file,
                    inputs=[chatbot_ui,textbox_local_filename],
                    outputs=[chatbot_ui, textbox_local_filename], queue=False
                )

                cb_txt_msg.then(self.cb_user_chat,
                                inputs=[textbox_user_prompt, chatbot_ui,
                                        textbox_chatbot_response],
                                outputs=[textbox_user_prompt, chatbot_ui], queue=False
                                ).then(self.cb_botchat,
                                    inputs=[chatbot_ui, self.history_state,
                                            textbox_chatbot_response],
                                    outputs=[chatbot_ui, self.history_state,
                                                textbox_chatbot_response], queue=False
                                ).then(self.cb_update_chat_session,
                                        inpusystem_speaker_v2ts=[self.history_state],
                                        outputs=[chatbot_ui, self.history_state,self.session_tokens], queue=False                                        
                                ).then(self.cb_convert_text_to_speech,
                                        inputs=[textbox_chatbot_response, cb_text_to_speech_target_lang, checkbox_turn_off_voice_response],
                                        outputs=[cb_bot_audio_output], queue=False
                                ).then(self.cb_reset_chat_outputs, 
                                    outputs=[textbox_local_filename, textbox_chatbot_response], queue=True
                                )
                #.then(lambda: gr.update(interactive=True), None, textbox_user_prompt)

                with gr.Accordion("Available Custom Commands", open=False):
                    # gr.Markdown(config.INSTRUCTIONS_TEXT)
                    with gr.Row():
                        with gr.Column(1):
                            gr.Markdown("#### Youtube Commands:")
                            gr.Markdown("""
                            | command            | parameters    |  example                                                                 |
                            |----------          |:-------------:|:------                                                                   |
                            | /ytranscribe video|  url          | ```/ytranscribe:https://www.youtube.com/watch?v=E9lAeMz1DaM (english)```|
                            | /ysummarize audio |  url          | ```/ysummarize:https://www.youtube.com/watch?v=E9lAeMz1DaM (english)```|
                            | /ytranslate audio to english |  url          | ```/ytranslate:https://www.youtube.com/watch?v=hz5xWgjSUlk (french)```|

                            **Random Examples: English** 
                            ``` 
                            15 Tips To Manage Your Time Better
                            /ytranscribe:https://www.youtube.com/watch?v=GBM2k2zp-MQ
                                                                        
                            Avril Lavigne - Breakaway
                            /ytranscribe:https://www.youtube.com/watch?v=oc7JLUvY9xI               
                            ```            
                            **Random Examples: Non-english** 
                            ``` 
                            > Case of Mixed languages
                            /ytranscribe:https://www.youtube.com/watch?v=zHGCJD64Djo                                
                            /ytranslate:https://www.youtube.com/watch?v=zHGCJD64Djo   
                                                
                            > Case of Non-english Talk show
                            (French talk-show: Raymond Devos-Parler pour ne rien dire)
                            /ytranslate:https://www.youtube.com/watch?v=hz5xWgjSUlk                                  
                            /ytranscribe:https://www.youtube.com/watch?v=hz5xWgjSUlk                                                   
                            /ysummarize:https://www.youtube.com/watch?v=hz5xWgjSUlk      
                            
                            > More Case of Non-english songs                                                                                                             
                            (Japanese: original song 彼岸花(マンジュシャカ) by Yamaguchi Momoe)
                            /ytranscribe:https://www.youtube.com/watch?v=JmS6zsR3UBs
                            /ytranslate:https://www.youtube.com/watch?v=JmS6zsR3UBs  
                            (Spanish song ASI ES LA VIDA by Enrique Iglesias,Maria Becerra)            
                            /ytranscribe:https://www.youtube.com/watch?v=XUoXE3bmDJY                        
                            /ytranslate:https://www.youtube.com/watch?v=XUoXE3bmDJY                                                          
                            (Cantonese song 蔓珠莎華 by various singers)
                            /ytranslate:https://www.youtube.com/watch?v=fxruu8cZams                                
                            /ytranslate:https://www.youtube.com/watch?v=f0Bn_wKLL5M                                        
                            ```            
                            `/ytranscribe` - extract audio from youtube video and transcribe audio to text

                            `/ysummarize` - extract audio from youtube video, pass to LLM to generate a summary of key points

                            `/ytranslate` - translate foreign language audio into English to english text.
                                                    
                            """)
                        with gr.Column(1):
                            gr.Markdown("#### Web Commands:")
                            gr.Markdown("""
                            | command            |      parameters                        |  example            |
                            |----------          |:-------------:                         |------:              |
                            | /ddgsearch? |  query=CN towe&max_results=5                       | ```/ddgsearch?query=CN towe&max_results=5``` |
                            | /ddgnews?   |  query=holiday&region=us&max_results=5&timelimit=d| ```/ddgnews?query=IBM stock price&max_results=5&backend=api``` |
                            | /ddgimage?  |  query,region="us-en",max_results=5 | ```/ddgimage?query=butterfly&max_results=5&backend=api&size=small&color=color&license_image=Public```    |
                            | /ddgvideo?  |  query,region="us-en",max_results=5 | ```/ddgvideo?query=butterfly&max_results=5&backend=api&resolution=standard&duration=short&license_videos=youtube``` |
                            | /ddganswer? |  query | ```/ddganswer?query=sun``` |                   
                            | /ddgsuggest? |  query | ```/ddgsuggest?query=butterfly``` |                                        
                            | /ddgtranslate? |  query | ```/ddgtranslate?query=butterfly&to_lang=fr``` |                                                            
                            """)
                            gr.Markdown("#### Summarize text file:")
                            gr.Markdown("""
                                        ```/tfsummarize``` - generate a summary of the text content
                                        **usage**
                                        ```/tfsummarize:https://cdn.serc.carleton.edu/files/teaching_computation/workshop_2018/activities/plain_text_version_declaration_inde.txt```                        
                            """)
        return self.blocks_chatbot_speaker
