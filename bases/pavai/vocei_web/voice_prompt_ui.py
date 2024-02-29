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

import os, sys
import gradio as gr
import torch
import pandas as pd
import numpy as np
#import pavai.shared.datasecurity as datasecurity
from typing import BinaryIO, Union
from transformers.utils import is_flash_attn_2_available
from pavai.shared.system_checks import (pavai_vocie_system_health_check, DEFAULT_SYSTEM_MODE, SYSTEM_THEME_SOFT,SYSTEM_THEME_GLASS,VOICE_PROMPT_INSTRUCTIONS_TEXT)
#from pavai.shared.llmproxy import chatbot_ui_client,chat_count_tokens,multimodal_ui_client
from pavai.shared.llmproxy import chat_count_tokens,multimodal_ui_client
from pavai.shared.image.text2image import (StableDiffusionXL, image_generation_client,DEFAULT_TEXT_TO_IMAGE_MODEL)
from pavai.shared.fileutil import get_text_file_content
from pavai.shared.commands import filter_commmand_keywords
from pavai.shared.grammar import (fix_grammar_error)
from pavai.shared.audio.transcribe import (speech_to_text, FasterTranscriber,DEFAULT_WHISPER_MODEL_SIZE)
from pavai.shared.audio.tts_client import (system_tts_local,speaker_file_v2,get_speaker_audio_file,speak_acknowledge,speak_wait, speak_done, speak_instruction)
#from pavai.shared.audio.voices_piper import (text_to_speech, speak_acknowledge,speak_wait, speak_done, speak_instruction)
#from pavai.shared.audio.voices_styletts2 import (text_to_speech, speak_acknowledge,speak_wait, speak_done, speak_instruction)

## remove Iterable, List, NamedTuple, Optional, Tuple,
## removed from os.path import dirname, join, abspath
from pavai.shared.aio.llmchat import get_llm_library
from pavai.shared.aio.llmchat import (system_prompt_assistant, DEFAULT_LLM_CONTEXT_SIZE)
from pavai.shared.llmcatalog import LLM_MODEL_KX_CATALOG_TEXT
from pavai.shared.solar.llmprompt import knowledge_experts_system_prompts
#from pavai.vocei_web.translator_ui import CommunicationTranslator,ScratchPad
from pavai.vocei_web.system_settings_ui import SystemSetting
import traceback
import sounddevice as sd
import cleantext

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

"""UI-VOICE-PROMPT"""
class VoicePrompt(SystemSetting):

    def vc_speech_to_text(self,input_audio: Union[str, BinaryIO, np.ndarray],
                        task_mode="transcribe",
                        model_size: str = "large",
                        beam_size: int = 5,
                        vad_filter: bool = True,
                        language: str = None,
                        include_timestamp_seg=False) -> str:
        transcription, language = speech_to_text(input_audio=input_audio, task_mode=task_mode, model_size=model_size,
                                                beam_size=beam_size, vad_filter=vad_filter, language=language,
                                                include_timestamp_seg=include_timestamp_seg)
        return transcription, language

    def vc_fix_grammar_error(self,raw_text: str,
                            enabled: bool = False,
                            max_length: int = 128,
                            repetition_penalty: float = 1.05,
                            num_beams: int = 4):
        corrected_text, raw_text = fix_grammar_error(raw_text=raw_text, enabled=enabled, max_length=max_length,
                                                    repetition_penalty=repetition_penalty, num_beams=num_beams)
        return corrected_text, raw_text

    def vc_text_to_image(self,user_prompt: str,
                        neg_prompt: str = "ugly, blurry, poor quality",
                        output_filename="new_text_to_image_1.png",
                        storage_path: str = "workspace/text-to-image") -> str:
        if user_prompt is None or len(user_prompt) == 0:
            gr.Warning("empty prompt text!")
            return None
        # load on-demand to reduce memory usage at cost of lower performance
        output_filename = image_generation_client(abstract_image_generator=stablediffusion_model,
                                                user_prompt=user_prompt, neg_prompt=neg_prompt,
                                                output_filename=output_filename, storage_path=storage_path)
        return output_filename

    def vc_hide_outputs(self,enable_text_to_image: bool = False):
        if enable_text_to_image:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        else:
            return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=False)

    def vc_text_to_speech(self,text: str, output_voice: str = "en", mute=False, autoplay=False):
        #out_file = text_to_speech(text=text, output_voice=output_voice, mute=mute, autoplay=autoplay)
        tts_voice=system_config["GLOBAL_TTS_LIBRETTS_VOICE"]
        out_file = speaker_file_v2(sd,text=text,output_voice="jane",autoplay=autoplay)
        return out_file

    def vc_voice_chat(self,user_message, chatbot_history):
        chatbot_history = chatbot_history + [[user_message, None]]
        return user_message, chatbot_history

    def vc_unhide_outputs(self,enable_text_to_image: bool = False):
        if enable_text_to_image:
            return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=True), gr.Button(visible=True), gr.Button(visible=True)
        else:
            return gr.Textbox(visible=True), gr.Audio(visible=True), gr.Image(visible=False), gr.Button(visible=False), gr.Button(visible=False)

    def vc_ask_assistant_to_image(self,enable_image_generation: bool, input_text: str, bot_history: list, chat_history: list):
        logger.info(f"ask_assistant_to_image: {input_text}")
        new_image_file = None
        reply = ""
        if input_text is not None and len(input_text) > 0:
            if enable_image_generation:
                new_image_file = self.vc_text_to_image(input_text)
                return bot_history, chat_history, "AI image generated.", new_image_file
            else:
                bot_history, chat_history, reply = self.chat_client(input_text, chat_history)
                return bot_history, chat_history, reply, new_image_file
        return bot_history, chat_history, reply, new_image_file

    def vc_add_query_log_entry(self,logdf: pd.DataFrame, query_text: str, response_text: str, duration: int = 0):
        if query_text is None or len(query_text) == 0:
            return
        idx = 1 if len(logdf) == 0 else len(logdf)+1
        if response_text is None:
            return 
        new_row = {'index': idx,
                'query_text': query_text, 'query_length': len(query_text),
                'response_text': response_text, 'response_length': len(response_text),
                "took": duration}
        logdf.loc[len(logdf)] = new_row
        return logdf

    def vc_set_task_mode(self,choice, task_mode_state):
        if choice == "translate":
            task_mode_state = choice
        else:
            task_mode_state = "transcribe"
        return task_mode_state

    def vc_update_log_entries(self,max_values):
        return gr.update(interactive=True, row_count=max_values)

    def vc_show_or_hide_input_textbox(self,checked):
        return gr.Textbox(visible=checked)

    def pii_data_analysis_options(self,x):
        global _QUERY_ENABLE_PII_ANALYSIS
        logger.warn(f"change pii_data_analysis_options: {x}")
        _QUERY_ENABLE_PII_ANALYSIS = x

    def pii_data_amomymization_options(self,x):
        global _QUERY_ENABLE_PII_ANONYMIZATION    
        logger.warn(f"change pii_data_amomymization_options: {x}")
        _QUERY_ENABLE_PII_ANONYMIZATION=x    

    def vc_sentence_word_splitter(self,num_of_words: int, sentence: str) -> list:
        pieces = sentence.split()
        return [" ".join(pieces[i:i+num_of_words]) for i in range(0, len(pieces), num_of_words)]

    def vc_chunk_sentences_to_max_tokens(self, sentences: list, max_length: int = 256):
        fixed_size_sentences = []
        tokens = sentences.lower().split()
        if len(tokens) > (max_length):
            chunks = self.sentence_word_splitter(num_of_words=max_length, sentence=sentences)
            for achunk in chunks:
                fixed_size_sentences.append(achunk)
        else:
            fixed_size_sentences.append(sentences)
        return fixed_size_sentences

    def vc_convert_text_to_speech(self,input_text:str, output_voice:str, mute=False, autoplay=False, delay:int=33):
        import time
        gr.Info("processing text to speech, please wait!")
        wav=self.vc_text_to_speech(text=input_text,output_voice="jane",mute=False, autoplay=False)
        return wav
    
        # cb_text_chunks = self.vc_chunk_sentences_to_max_tokens(sentences=input_text,max_length=90)
        # send_full_audio=False
        # print("convert_text_to_speech audio chunks: ",len(cb_text_chunks))
        # if len(cb_text_chunks)==1:
        #     send_full_audio=True
        # for chunk in cb_text_chunks:
        #     audio_file=self.vc_text_to_speech(text=chunk,output_voice=output_voice,mute=False, autoplay=False,)
        #     if send_full_audio:
        #         yield  audio_file

        #     else:           
        #         time.sleep(delay)
        #         yield audio_file

    def build_voice_prompt_ui(self):
        self.blocks_voice_prompt = gr.Blocks(title="VOICE-PROMPT-UI",analytics_enabled=False)
        with self.blocks_voice_prompt as llm_voice_query_ui:
            with gr.Group():    
                self.history_state = gr.State([])
                listening_status_state = gr.State(0)
                listening_sentence_state = gr.State([])
                activities_state = gr.State([])
                task_mode_state = gr.State("transcribe")
                """Configuration"""
                with gr.Accordion("Spoken Language", open=False):                
                    with gr.Row():
                        with gr.Column(1):
                            available_ai_speaker_langs=["ar","cs","da","de","en","en_ryan","en_ryan_medium","en_ryan_low","en_amy","en_amy_low","en_kusal","en_lessac","en_lessac_low","ru","zh","fr","uk","es"]
                            self.vc_text_to_speech_target_lang = gr.Dropdown(label="Set spoken language",choices=available_ai_speaker_langs, value="en", info="default attempt to auto detect spoken language.")
                        with gr.Column(2):
                            textbox_detected_spoken_language = gr.Textbox(label="Auto detect spoken language", value="en")

                with gr.Accordion("Additional Options", open=False):
                    with gr.Row():
                        with gr.Column(1):
                            radio_task_mode = gr.Radio(choices=["transcribe", "translate"], value="transcribe", interactive=True,
                                                    label="transcription options",
                                                    info="[transcribe] preserved spoken language while [translate] converts all spoken languages to english only.]")
                            radio_task_mode.change(fn=self.vc_set_task_mode, inputs=[
                                                radio_task_mode, task_mode_state], outputs=task_mode_state)
                        with gr.Column(1):
                            checkbox_enable_text_to_image = gr.Checkbox(
                                value=False, label="enable image generation",
                                info="speak a shorter prompts to generate creative AI image.")                            
                            checkbox_enable_grammar_fix = gr.Checkbox(value=False,
                                                                    label="enable single-shot grammar synthesis correction model",
                                                                    info="grammar synthesis model it does not semantically change text/information that is grammatically correct.")
                            # slider_max_query_log_entries = gr.Slider(
                            #     5, 30, step=1, label="Max logs", interactive=True)                                                        
                        with gr.Column(1):
                            checkbox_show_transcribed_text = gr.Checkbox(value=True,
                                                                        label="show transcribed voice text")
                            checkbox_show_response_text = gr.Checkbox(value=True,
                                                                    label="show response text")
                            slider_max_query_log_entries = gr.Slider(5, 30, step=1, label="Max logs", interactive=True)                            
                    with gr.Row():
                        """content safety options"""                                
                        checkbox_content_safety = gr.Checkbox(
                                value=False, label="enable content safety guard",
                                info="enforce content safety check on input and output")    
                        checkbox_content_safety.change(fn=self.select_content_safety_options,inputs=checkbox_content_safety)                                                                           
                        """data security options"""                
                        checkbox_enable_PII_analysis = gr.Checkbox(
                            value=False, label="enable PII data analysis",
                            info="analyze and report PII data on query input and outputs")
                        checkbox_enable_PII_analysis.change(fn=self.pii_data_analysis_options,
                                                            inputs=checkbox_enable_PII_analysis)
                        checkbox_enable_PII_anonymization = gr.Checkbox(
                            value=False, label="enable PII data anonymization",
                            info="apply anonymization of PII data on input and output")    
                        checkbox_enable_PII_anonymization.change(fn=self.pii_data_amomymization_options,
                                                            inputs=checkbox_enable_PII_anonymization)                                       
                    with gr.Row():                
                        """knowledge and domain model experts"""       
                        with gr.Column(2):         
                            expert_options = gr.Dropdown(knowledge_experts, label="Knowledge Experts (AI)", info="[optional] route question to subject with domain knowledge.")
                            expert_options.change(fn=self.select_expert_option,inputs=expert_options)        
                        with gr.Column(1):                            
                            model_options = gr.Dropdown(domain_models,label="Domain Models",info="[optional] use specialized model")                        
                            model_options.change(fn=self.select_model_option,inputs=model_options)        

                """LANGUAGE SETIING"""
                def overrider_auto_detect_language(input_lang:str):
                    textbox_detected_spoken_language=input_lang

                self.vc_text_to_speech_target_lang.change(fn=overrider_auto_detect_language,
                                                          inputs=[self.vc_text_to_speech_target_lang], 
                                                          outputs=[textbox_detected_spoken_language])               

                """AUDIO INPUT/OUTPUTS"""
                # with gr.Group():
                default_max_microphone_recording_length_30_min = DEFAULT_MAX_MIC_RECORD_LENGTH_IN_SECONDS        
                # Audio Input Box
                with gr.Row():
                    speaker_audio_input = gr.Audio(scale=2, sources=["microphone", "upload"], elem_id="user_speaker",
                                                    type="filepath", label="press [record] to start speaking",
                                                    show_download_button=True, show_share_button=True, visible=True,
                                                    format="mp3", max_length=default_max_microphone_recording_length_30_min,
                                                    waveform_options={"waveform_progress_color": "green", "waveform_progress_color": "green"})

                    speaker_transcribed_input = gr.Textbox(label="transcribed query",
                                                            info="press [shipt]+[enter] key to re-submit transcribe text",
                                                            scale=2, lines=3,
                                                            show_copy_button=True, interactive=True,
                                                            container=True, visible=True)

                    speaker_original_input = gr.Textbox(label="without grammar fix version",
                                                        scale=2, lines=3, show_copy_button=True, interactive=True,
                                                        container=True, visible=False)

                    # handle change configuration option checkbox
                    checkbox_enable_grammar_fix.change(self.vc_show_or_hide_input_textbox, checkbox_enable_grammar_fix, speaker_original_input, queue=False)

                    # clear transcribed textbox
                    speaker_audio_input.clear(lambda: gr.update(value=""), None, speaker_transcribed_input)

                    # handle audio recording as input
                    speaker_input_mic = speaker_audio_input.stop_recording(
                         speak_wait, None, None, queue=False
                    ).then(
                        self.vc_speech_to_text,
                        inputs=[speaker_audio_input, task_mode_state],
                        outputs=[speaker_transcribed_input,textbox_detected_spoken_language],queue=False
                    ).then(
                        self.vc_fix_grammar_error,
                        inputs=[speaker_transcribed_input,checkbox_enable_grammar_fix],
                        outputs=[speaker_transcribed_input,speaker_original_input], queue=False
                    )

                    # handle audio file upload as input
                    speaker_input_file = speaker_audio_input.upload(
                        speak_wait, None, None, queue=False
                    ).then(
                        self.vc_speech_to_text,
                        inputs=[speaker_audio_input, task_mode_state],
                        outputs=[speaker_transcribed_input,textbox_detected_spoken_language],
                        queue=False
                    ).then(
                        self.vc_fix_grammar_error,
                        inputs=[speaker_transcribed_input,checkbox_enable_grammar_fix],
                        outputs=[speaker_transcribed_input,speaker_original_input], queue=False
                    )
                with gr.Row():
                    bot_audio_output = gr.Audio(scale=1, label="Assistant response audio",
                                                format="mp3", elem_id="ai_speaker", autoplay=True, visible=False,
                                                waveform_options={"waveform_progress_color": "orange", "waveform_progress_color": "orange"})
                    bot_output_txt = gr.Textbox(
                        label="Assistant response text", scale=2, lines=3,
                        info="AI generated response text",
                        placeholder="", container=True, visible=False, show_copy_button=True
                    )
                with gr.Row():
                    bot_output_image = gr.Image(
                        label="generated image", type="filepath", interactive=False, visible=False)
                with gr.Row():
                    # --summarize response
                    btn_generate_request_image = gr.Button(
                        value="Generate transcribed text image", size="sm", visible=False)
                    btn_generate_request_image.click(fn=self.vc_text_to_image, inputs=[speaker_transcribed_input],
                                                        outputs=[bot_output_image], queue=False)

                    btn_generate_response_image = gr.Button(
                        value="Generate response text image", size="sm", visible=False)
                    btn_generate_response_image.click(fn=self.vc_text_to_image, inputs=[bot_output_txt],
                                                        outputs=[bot_output_image], queue=False)
                    bot_output_image.change(self.vc_unhide_outputs, inputs=[checkbox_enable_text_to_image], outputs=[
                                            bot_output_txt, bot_audio_output, bot_output_image], queue=False)

                    btn_clear = gr.ClearButton([speaker_audio_input, bot_audio_output, bot_output_image,
                                                speaker_transcribed_input, bot_output_txt, activities_state],
                                                size="sm",value="🎙️ New Query")
                    btn_clear.click(
                        self.vc_hide_outputs, inputs=[checkbox_enable_text_to_image],
                        outputs=[bot_output_txt, bot_audio_output, bot_output_image], queue=False)

                with gr.Accordion("User Instructions", open=False):
                    gr.Markdown(VOICE_PROMPT_INSTRUCTIONS_TEXT)

                """QUERY LOGS"""
                with gr.Accordion("Query Logs", open=False):
                    # Creating dataframe to log queries
                    df_query_log = pd.DataFrame(
                        {"index": [], "query_text": [], "query_length": [],"response_text": [], "response_length": [], "took": []}
                    )
                    styler = df_query_log.style.highlight_between(color='lightgreen', axis=0)
                    gr_query_log = gr.Dataframe(value=styler, interactive=True, wrap=True)

                    max_values = 10
                    slider_max_query_log_entries.change(self.vc_update_log_entries,
                                                        inputs=[slider_max_query_log_entries],
                                                        outputs=[gr_query_log], queue=False)

                # handle speaker input mic
                speaker_input_mic.then(
                    speak_acknowledge, None, None, queue=False
                ).then(
                    self.vc_voice_chat,
                    inputs=[speaker_transcribed_input, activities_state],
                    outputs=[speaker_transcribed_input, activities_state], queue=False
                ).then(
                    self.vc_unhide_outputs,
                    inputs=[checkbox_enable_text_to_image],
                    outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                            btn_generate_request_image, btn_generate_response_image], queue=False
                ).then(
                    self.vc_ask_assistant_to_image,
                    inputs=[checkbox_enable_text_to_image,
                            speaker_transcribed_input, activities_state, self.history_state],
                    outputs=[activities_state, self.history_state,
                            bot_output_txt, bot_output_image], queue=False
                ).then(
                    self.vc_add_query_log_entry,
                    inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
                    outputs=[gr_query_log], queue=True
                ).then(
                    lambda: gr.update(visible=True), None, textbox_detected_spoken_language, queue=True
                ).then(
                    self.vc_convert_text_to_speech,
                    inputs=[bot_output_txt, textbox_detected_spoken_language],
                    outputs=[bot_audio_output], queue=False
                )

                # handle speaker input file
                speaker_input_file.then(
                    speak_acknowledge, None, None, queue=False
                ).then(
                    self.vc_voice_chat,
                    inputs=[speaker_transcribed_input, activities_state],
                    outputs=[speaker_transcribed_input, activities_state], queue=False
                ).then(
                    self.vc_unhide_outputs,
                    inputs=[checkbox_enable_text_to_image],
                    outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                            btn_generate_request_image, btn_generate_response_image], queue=False
                ).then(
                    self.vc_ask_assistant_to_image,
                    inputs=[checkbox_enable_text_to_image,
                            speaker_transcribed_input, activities_state, self.history_state],
                    outputs=[activities_state, self.history_state,
                            bot_output_txt, bot_output_image], queue=False
                ).then(
                    self.vc_add_query_log_entry,
                    inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
                    outputs=[gr_query_log], queue=True
                ).then(
                    self.vc_convert_text_to_speech,
                    inputs=[bot_output_txt, textbox_detected_spoken_language],
                    outputs=[bot_audio_output], queue=False
                )

                # handle transcribed text submit event
                speaker_transcribed_input.submit(
                    speak_acknowledge, None, None, queue=False
                ).then(
                    self.vc_voice_chat,
                    inputs=[speaker_transcribed_input, activities_state],
                    outputs=[speaker_transcribed_input, activities_state], queue=False
                ).then(
                    self.vc_unhide_outputs,
                    inputs=None,
                    outputs=[bot_output_txt, bot_audio_output], queue=False
                ).then(
                    self.vc_ask_assistant_to_image,
                    inputs=[checkbox_enable_text_to_image,
                            speaker_transcribed_input, activities_state, self.history_state],
                    outputs=[activities_state, self.history_state,
                            bot_output_txt, bot_output_image], queue=False
                ).then(
                    self.vc_add_query_log_entry,
                    inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
                    outputs=[gr_query_log], queue=True
                ).then(
                    self.vc_convert_text_to_speech,
                    inputs=[bot_output_txt, textbox_detected_spoken_language],
                    outputs=[bot_audio_output], queue=False
                )

                # unhide buttons
                checkbox_enable_text_to_image.change(
                    self.vc_unhide_outputs,
                    inputs=[checkbox_enable_text_to_image],
                    outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                            btn_generate_request_image, btn_generate_response_image], queue=False
                )
                # "realtime/gradio_tutor/audio/hp0.wav"
                # "realtime/examples/jfk.wav"
                gr.Examples(
                examples=["workspace/samples/jfk.wav"],
                inputs=[speaker_audio_input],
                outputs=[speaker_transcribed_input, textbox_detected_spoken_language],
                fn=speech_to_text,
                cache_examples=False,
                label="example audio file: president JFK speech"
            )
        return self.blocks_voice_prompt
