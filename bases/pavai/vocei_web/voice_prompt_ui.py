from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)
import gradio as gr
import torch
import pandas as pd
import numpy as np
from typing import BinaryIO, Union
from transformers.utils import is_flash_attn_2_available
from pavai.shared.system_checks import (DEFAULT_SYSTEM_MODE, SYSTEM_THEME_SOFT,SYSTEM_THEME_GLASS,VOICE_PROMPT_INSTRUCTIONS_TEXT)
from pavai.shared.image.text2image import (StableDiffusionXL, image_generation_client,DEFAULT_TEXT_TO_IMAGE_MODEL)
from pavai.shared.grammar import (fix_grammar_error)
from pavai.shared.audio.transcribe import (speech_to_text, FasterTranscriber,DEFAULT_WHISPER_MODEL_SIZE)
from pavai.shared.audio.tts_client import (system_tts_local,speaker_file_v2,get_speaker_audio_file,speak_acknowledge,speak_wait, speak_done, speak_instruction)
from pavai.vocei_web.system_settings_ui import SystemSetting
import pavai.llmone.remote.chatmodels as chatmodels
import pavai.llmone.chatprompt as chatprompt
import sounddevice as sd
# from pavai.shared.fileutil import get_text_file_content
# from pavai.shared.commands import filter_commmand_keywords
#import cleantext
#import os, sys
import traceback
# pip install python-dotenv
# tested version gradio version: 4.7.1
# pip install gradio==4.7.1

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2024"
__version__ = "0.0.3"

logger.warn("--GLOBAL SYSTEM MODE----")
logger.warn(config.system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE=config.system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS=config.system_config["GLOBAL_TTS"]

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
knowledge_experts = list(chatprompt.knowledge_experts_system_prompts.keys())

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
                        task_mode="translate",#"transcribe",
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

    def vc_text_to_speech(self,text: str, output_voice: str = "jane",voice_emotion:str=None, mute=False, autoplay=False):
        out_file = speaker_file_v2(text=text,output_voice=output_voice,autoplay=autoplay)
        return out_file

    def vc_voice_chat(self,user_message, chatbot_history):
        chatbot_history = chatbot_history + [[user_message, None]]
        return user_message, chatbot_history

    def vc_unhide_outputs(self,enable_text_to_image: bool = False):
        if enable_text_to_image:
            return gr.Textbox(visible=False), gr.Audio(visible=False), gr.Image(visible=True), gr.Button(visible=True), gr.Button(visible=True)
        else:
            return gr.Textbox(visible=True), gr.Audio(visible=True), gr.Image(visible=False), gr.Button(visible=False), gr.Button(visible=False)

    def vc_query_parameters(self,domain_expert, response_style, api_base, api_key,activate_model_id,system_prompt,top_p,temperature,max_tokens,present_penalty,stop_words,frequency_penalty,gpu_offload_layers):
        self.user_settings["_SYSTEM_MODE"] = config.system_config["GLOBAL_SYSTEM_MODE"]    
        self.user_settings["_QUERY_API_BASE"] = str(api_base).strip()
        self.user_settings["_QUERY_API_KEY"] = str(api_key).strip()
        self.user_settings["_QUERY_MODEL_ID"] = str(activate_model_id).strip()
        self.user_settings["_QUERY_TOP_P"] = int(top_p)
        self.user_settings["_QUERY_TEMPERATURE"] = float(temperature)
        self.user_settings["_QUERY_MAX_TOKENS"] = int(max_tokens)
        self.user_settings["_QUERY_PRESENT_PENALTY"] = int(present_penalty)
        if stop_words is not None:
            stop_words=str(stop_words).split(",")
        self.user_settings["_QUERY_STOP_WORDS"] = stop_words
        self.user_settings["_QUERY_FREQUENCY_PENALTY"] = int(frequency_penalty)
        self.user_settings["_QUERY_SYSTEM_PROMPT"] = str(system_prompt).strip()
        self.user_settings["_QUERY_DOMAIN_EXPERT"] = str(domain_expert).strip()
        self.user_settings["_QUERY_RESPONSE_STYLE"] = str(response_style).strip()
        self.user_settings["_QUERY_GPU_OFFLOADING_LAYERS"] = int(gpu_offload_layers)

    def vc_ask_assistant_to_image(self,enable_image_generation: bool, input_text: str, bot_history: list, chat_history: list):
        logger.info(f"ask_assistant_to_image: {input_text}")
        new_image_file = None
        reply = ""
        status_line=""
        if input_text is not None and len(input_text) > 0:
            if enable_image_generation:
                new_image_file = self.vc_text_to_image(input_text)
                return bot_history, chat_history, "AI image generated.", new_image_file, status_line
            else:
                bot_history, chat_history, reply, status_line = self.chat_client(input_text, chat_history,self.user_settings)
                return bot_history, chat_history, reply, new_image_file, status_line
        return bot_history, chat_history, reply, new_image_file, status_line

    def vc_add_query_log_entry(self,logdf: pd.DataFrame, query_text: str, response_text: str, duration: int = 0):
        if query_text is None or len(query_text) == 0:
            return
        idx = 1 if len(logdf) == 0 else len(logdf)+1
        if response_text is None or len(response_text.strip())==0:
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

    def vc_convert_text_to_speech(self,selected_voice:str=None,voice_emotion:str=None,input_text:str=None, output_voice:str=None, mute=False, autoplay=False, delay:int=33):
        gr.Info(f"processing text to speech with voice:{selected_voice}") 
        logger.debug(f"convert_text_to_speech voice: {selected_voice} emotion: {voice_emotion}")
        if selected_voice is None or len(selected_voice)==0:
            wav_file = speaker_file_v2(text=input_text,output_voice="jane",autoplay=autoplay)
        else:
            if voice_emotion is not None and len(voice_emotion)>0:
                wav_file= speaker_file_v2(text=input_text,output_voice=selected_voice,output_emotion=voice_emotion, autoplay=autoplay)
            else:
                wav_file= speaker_file_v2(text=input_text,output_voice=selected_voice,autoplay=autoplay)
        return wav_file

    def list_voices(self)->list:
        import pavai.shared.styletts2.live_voices as live_voices
        voice_path = config.system_config["REFERENCE_VOICES"]
        voices = live_voices.get_voice_names(path=voice_path)
        return voices

    def list_speech_styles(self)->list:
        return chatprompt.speech_styles.keys()

    def get_api_base(self)->str:
        if config.system_config["GLOBAL_SYSTEM_MODE"]=="solar-openai":
            return config.system_config["SOLAR_LLM_DEFAULT_HOST"]            
        elif config.system_config["GLOBAL_SYSTEM_MODE"]=="ollama-openai":
            return config.system_config["SOLAR_LLM_OLLAMA_HOST"]  
        else:
             return "locally [all-in-one]"          

    def get_api_key(self)->str:
        if config.system_config["GLOBAL_SYSTEM_MODE"]=="solar-openai":
            return config.system_config["SOLAR_LLM_DEFAULT_API_KEY"]            
        elif config.system_config["GLOBAL_SYSTEM_MODE"]=="ollama-openai":
            return config.system_config["SOLAR_LLM_OLLAMA_API_KEY"]  
        else:
             return "Not Applicable"          

    def get_active_model(self)->str:
        if config.system_config["GLOBAL_SYSTEM_MODE"]=="solar-openai":
            return config.system_config["SOLAR_LLM_DEFAULT_MODEL_ID"]
        elif config.system_config["GLOBAL_SYSTEM_MODE"]=="ollama-openai":
            return config.system_config["SOLAR_LLM_OLLAMA_MODEL_ID"]
        else:
            return config.system_config["DEFAULT_LLM_MODEL_FILE"]          

    def get_gpu_offload_layers(self)->int:
        return int(config.system_config["DEFAULT_LLM_OFFLOAD_GPU_LAYERS"])

    def list_models(self)->list:
        if config.system_config["GLOBAL_SYSTEM_MODE"]=="solar-openai":
            return chatmodels.load_solar_models().keys()
        elif config.system_config["GLOBAL_SYSTEM_MODE"]=="ollama-openai":
            return chatmodels.load_ollama_models().keys()
        else:
            return chatmodels.load_local_models().keys()            
    
    def list_domain_experts(self)->list:
        return chatprompt.domain_experts.keys()

    def list_emotions(self)->list:
        return ["happy","sad","angry","surprised"]

    def vc_update_llm_mode(self,llm_mode):
        if llm_mode=="solar-openai":
            api_host = config.system_config["SOLAR_LLM_DEFAULT_SERVER_URL"]  
            api_key = config.system_config["SOLAR_LLM_DEFAULT_API_KEY"] 
            active_model = config.system_config["SOLAR_LLM_DEFAULT_MODEL_ID"]                                              
            model_dropdown = chatmodels.load_solar_models().keys()                                                                     
        elif llm_mode=="ollama-openai":
            api_host = config.system_config["SOLAR_LLM_OLLAMA_SERVER_URL"]  
            api_key = config.system_config["SOLAR_LLM_OLLAMA_API_KEY"] 
            active_model = config.system_config["SOLAR_LLM_OLLAMA_MODEL_ID"]                                              
            model_dropdown = chatmodels.load_ollama_models().keys()                                                                     
        else:
            api_host = "locally-aio"
            api_key = "not applicable"
            active_model = config.system_config["DEFAULT_LLM_MODEL_FILE"]                                              
            model_dropdown = chatmodels.load_local_models().keys()                                                                                             
        return [api_host,api_key,active_model,model_dropdown]

    def build_voice_prompt_ui(self):
        self.set_user_settings()        
        self.blocks_voice_prompt = gr.Blocks(title="VOICE-PROMPT-UI",analytics_enabled=False,
                                             css=".gradio-container {background: url('file=pavai_logo_large.png')}"
                                             )
        with self.blocks_voice_prompt as llm_voice_query_ui:
            with gr.Group():    
                self.history_state = gr.State([])
                listening_status_state = gr.State(0)
                listening_sentence_state = gr.State([])
                activities_state = gr.State([])
                task_mode_state = gr.State("transcribe")
                """Configuration"""
                with gr.Accordion("Spoken Language", open=False, visible=False):                
                    with gr.Row():
                        with gr.Column(1):
                            available_ai_speaker_langs=["ar","cs","da","de","en","ru","zh","fr","uk","es"]
                            self.vc_text_to_speech_target_lang = gr.Dropdown(label="Set spoken language",choices=available_ai_speaker_langs, value="en", info="default attempt to auto detect spoken language.")
                        with gr.Column(2):
                            textbox_detected_spoken_language = gr.Textbox(label="Auto detect spoken language", value="en")
                with gr.Row():
                    with gr.Column(1):         
                        """Human Voices"""       
                        vc_human_voices_options = gr.Dropdown(label="Voice",choices=self.list_voices(), value="anthony_real")
                        vc_selected_voice = gr.Text(visible=False, value="anthony_real")
                    with gr.Column(1):                                 
                        """Voice Emotions"""       
                        vc_voice_emotions = gr.Dropdown(label="Emotion",choices=self.list_emotions())
                        vc_selected_emotion = gr.Text(visible=False)
                    with gr.Column(1):         
                        """knowledge and domain model experts"""                               
                        vc_domain_expert_options = gr.Dropdown(choices=self.list_domain_experts(), label="Persona(AI)")
                        vc_selected_domain_expert = gr.Text(visible=False)
                    with gr.Column(1):         
                        """Response style"""                               
                        vc_response_style_options = gr.Dropdown(label="Tone", choices=self.list_speech_styles())
                        vc_selected_response_tyle = gr.Text(visible=False)
                    def change_domain_export(new_expert:str):
                        gr.Info(f"set new domain expert to {new_expert}")    
                        self.new_domain_expert=new_expert
                        self.new_system_prompt=chatprompt.domain_experts[new_expert]
                        return self.new_domain_expert, self.new_system_prompt

                    def change_response_tyle(new_style:str):
                        gr.Info(f"set new response style to {new_style}")    
                        self.new_response_style=new_style
                        return new_style

                    def change_voice(new_voice:str):
                        gr.Info(f"set voice to {new_voice}") 
                        self.new_voice =  new_voice 
                        return new_voice

                    def change_emotion(new_emotion:str):
                        gr.Info(f"set voice emotion to {new_emotion}") 
                        self.new_emotion =  new_emotion 
                        return new_emotion

                with gr.Accordion("Additional Options", open=False):
                    with gr.Row():
                        with gr.Column(1):
                            radio_task_mode = gr.Radio(choices=["transcribe", "translate"], value="transcribe",visible=False,interactive=True,
                                                    label="transcription options",
                                                    info="[transcribe] preserved spoken language while [translate] converts all spoken languages to english only.]")
                            radio_task_mode.change(fn=self.vc_set_task_mode, inputs=[radio_task_mode, task_mode_state], outputs=task_mode_state)
                            checkbox_enable_text_to_image = gr.Checkbox(value=False, label="enable image generation",info="speak a shorter prompts to generate creative AI image.")                                                        
                            checkbox_enable_grammar_fix = gr.Checkbox(value=False, label="enable single-shot grammar synthesis correction model", info="It does not semantically change that is grammatically correct.")
                        with gr.Column(1):
                            """content safety options"""                                
                            # checkbox_content_safety = gr.Checkbox(
                            #         value=False, label="enable content safety guard",
                            #         info="enforce content safety check on input and output")    
                            # checkbox_content_safety.change(fn=self.select_content_safety_options,inputs=checkbox_content_safety)                                                                           
                            """data security options"""                
                            checkbox_enable_PII_analysis = gr.Checkbox(
                                value=eval(config.system_config["_QUERY_ENABLE_PII_ANALYSIS"]), label="enable PII data analysis",
                                info="analyze and report PII data on query input and outputs")
                            checkbox_enable_PII_analysis.change(fn=self.pii_data_analysis_options,
                                                                inputs=checkbox_enable_PII_analysis)
                            checkbox_enable_PII_anonymization = gr.Checkbox(
                                value=eval(config.system_config["_QUERY_ENABLE_PII_ANONYMIZATION"]), label="enable PII data anonymization",
                                info="apply anonymization of PII data on input and output")    
                            checkbox_enable_PII_anonymization.change(fn=self.pii_data_amomymization_options,
                                                            inputs=checkbox_enable_PII_anonymization)                                      
                        with gr.Column(1):
                            checkbox_show_transcribed_text = gr.Checkbox(value=True,
                                                                        label="show transcribed voice text")
                            checkbox_show_response_text = gr.Checkbox(value=True,
                                                                    label="show response text")
                            slider_max_query_log_entries = gr.Slider(5, 30, step=1, label="Max logs", interactive=True)                            
 
                """LANGUAGE SETIING"""
                # def overrider_auto_detect_language(input_lang:str):
                #     textbox_detected_spoken_language=input_lang

                # self.vc_text_to_speech_target_lang.change(fn=overrider_auto_detect_language,
                #                                           inputs=[self.vc_text_to_speech_target_lang], 
                #                                           outputs=[textbox_detected_spoken_language])               

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
                    bot_audio_output = gr.Audio(scale=1, label="assistant response audio",
                                                format="mp3", elem_id="ai_speaker", autoplay=True, visible=False,
                                                waveform_options={"waveform_progress_color": "orange", "waveform_progress_color": "orange"})
                    bot_output_txt = gr.Textbox(
                        label="assistant response text", scale=2, lines=3,
                        info="AI generated response text",
                        placeholder="", container=True, visible=False, show_copy_button=True
                    )
                with gr.Row():
                    bot_output_image = gr.Image(
                        label="generated image", type="filepath", interactive=False, visible=False)
                with gr.Row():
                     bot_status_line = gr.HTML()            
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
                                                size="sm",value="🎙️ New")
                    btn_clear.click(
                        self.vc_hide_outputs, inputs=[checkbox_enable_text_to_image],
                        outputs=[bot_output_txt, bot_audio_output, bot_output_image], queue=False)
                
                with gr.Accordion("Model Parameters", open=False):
                    with gr.Row():
                        radio_llm_mode = gr.Radio(choices=["locally-aio", "solar-openai", "ollama-openai"], 
                                                  value=config.system_config["GLOBAL_SYSTEM_MODE"],
                                                  visible=True,interactive=True,
                                                    label="LLM modes", info="currently running mode")
                    with gr.Row():
                        with gr.Column():
                            bot_api_host = gr.Textbox(
                                label="API base",
                                info="api endpoint url",
                                value=self.get_api_base(),
                            )
                        with gr.Column():
                            bot_api_key = gr.Textbox(
                                label="API Key",
                                value=self.get_api_key(),
                                type="password",
                                placeholder="sk..",
                                info="API keys for the url",
                            )                            
                        with gr.Column():
                            bot_active_model = gr.Textbox(
                                label="Active model",
                                value=self.get_active_model(),
                                info="current model",
                            )
                        with gr.Column():
                            bot_model_dropdown = gr.Dropdown(
                                label="Available models",
                                value=self.get_active_model(),
                                choices=self.list_models(),
                                info="select a model",
                                interactive=True,
                            )

                    system_msg_info = """System message helps set the behavior of the AI Assistant. For example, the assistant could be instructed with 'You are a helpful assistant.'"""
                    bot_system_prompt = gr.Textbox(
                        label="Instruct the AI Assistant to set its beaviour",
                        info=system_msg_info,
                        value="You are helpful AI assistant on helping answer user question and research.",
                        placeholder="Type here..",
                        lines=2,
                    )
                    accordion_msg = gr.HTML(
                        value="🚧 To set System message you will have to refresh the app",
                        visible=False,
                    )
                    # top_p, temperature
                    with gr.Row():
                        with gr.Column():
                            bot_top_p = gr.Slider(
                                minimum=-0,
                                maximum=40.0,
                                value=1.0,
                                step=0.05,
                                interactive=True,
                                label="Top-p (nucleus sampling)",
                            )
                        with gr.Column():
                            bot_temperature = gr.Slider(
                                minimum=0,
                                maximum=5.0,
                                value=1.0,
                                step=0.1,
                                interactive=True,
                                label="Temperature",
                            )
                    with gr.Row():
                        with gr.Column():
                            bot_max_tokens = gr.Slider(info="number of new tokens",
                                minimum=1,
                                maximum=16384,
                                value=256,
                                step=1,
                                interactive=True,
                                label="Max Tokens",
                            )
                        with gr.Column():
                            bot_presence_penalty = gr.Number(
                                label="presence_penalty", value=0, precision=0
                            )
                        with gr.Column():
                            bot_stop_words = gr.Textbox(info="separate by commas",
                                label="stop words", value="<"
                            )
                        with gr.Column():
                            bot_frequency_penalty = gr.Number(info="cost of repeat words",
                                label="frequency_penalty", value=0, precision=0
                            )
                        with gr.Column():
                            bot_gpu_offload_layers = gr.Number(label="gpu offloading layers",value=self.get_gpu_offload_layers(), visible=True, precision=0)

                with gr.Accordion("Instructions", open=False):
                    gr.Markdown(VOICE_PROMPT_INSTRUCTIONS_TEXT)

                """QUERY LOGS"""
                with gr.Accordion("Logs", open=False):
                    # Creating dataframe to log queries
                    df_query_log = pd.DataFrame(
                        {"index": [], "query_text": [], "query_length": [],"response_text": [], "response_length": [], "took": []}
                    )
                    styler = df_query_log.style.highlight_between(color='lightgreen', axis=0)
                    gr_query_log = gr.Dataframe(value=styler, interactive=False, wrap=True)

                    max_values = 10
                    slider_max_query_log_entries.change(self.vc_update_log_entries,inputs=[slider_max_query_log_entries],outputs=[gr_query_log], queue=False)

                # handle speaker input mic
                speaker_input_mic.then(
                    self.vc_voice_chat,
                    inputs=[speaker_transcribed_input, activities_state],
                    outputs=[speaker_transcribed_input, activities_state], queue=False
                ).then(
                    self.vc_unhide_outputs,
                    inputs=[checkbox_enable_text_to_image],
                    outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                            btn_generate_request_image, btn_generate_response_image], queue=False
                ).then(fn=self.vc_query_parameters, 
                       inputs=[vc_selected_domain_expert,vc_selected_response_tyle,bot_api_host,bot_api_key,
                               bot_active_model,bot_system_prompt,bot_top_p,bot_temperature,
                               bot_max_tokens,bot_presence_penalty,
                               bot_stop_words,bot_frequency_penalty,
                               bot_gpu_offload_layers],queue=False
                ).then(
                    self.vc_ask_assistant_to_image,
                    inputs=[checkbox_enable_text_to_image,
                            speaker_transcribed_input, activities_state, self.history_state],
                    outputs=[activities_state, self.history_state,
                            bot_output_txt, bot_output_image, bot_status_line], queue=False
                ).then(
                    self.vc_add_query_log_entry,
                    inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
                    outputs=[gr_query_log], queue=True
                ).then(
                    lambda: gr.update(visible=True), None, textbox_detected_spoken_language, queue=True
                ).then(
                    self.vc_convert_text_to_speech,
                    inputs=[vc_selected_voice,vc_voice_emotions,bot_output_txt, textbox_detected_spoken_language],
                    outputs=[bot_audio_output], queue=False
                )

                # handle speaker input file
                speaker_input_file.then(
                    self.vc_voice_chat,
                    inputs=[speaker_transcribed_input, activities_state],
                    outputs=[speaker_transcribed_input, activities_state], queue=False
                ).then(
                    self.vc_unhide_outputs,
                    inputs=[checkbox_enable_text_to_image],
                    outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                            btn_generate_request_image, btn_generate_response_image], queue=False
                ).then(fn=self.vc_query_parameters, 
                       inputs=[vc_selected_domain_expert,vc_selected_response_tyle,bot_api_host,bot_api_key,
                               bot_active_model,bot_system_prompt,bot_top_p,bot_temperature,
                               bot_max_tokens,bot_presence_penalty,
                               bot_stop_words,bot_frequency_penalty,
                               bot_gpu_offload_layers],queue=False
                ).then(
                    self.vc_ask_assistant_to_image,
                    inputs=[checkbox_enable_text_to_image,
                            speaker_transcribed_input, activities_state, self.history_state],
                    outputs=[activities_state, self.history_state,
                            bot_output_txt, bot_output_image,bot_status_line], queue=False
                ).then(
                    self.vc_add_query_log_entry,
                    inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
                    outputs=[gr_query_log], queue=True
                ).then(
                    self.vc_convert_text_to_speech,
                    inputs=[vc_selected_voice,vc_voice_emotions,bot_output_txt, textbox_detected_spoken_language],
                    outputs=[bot_audio_output], queue=False
                )

                # handle transcribed text submit event
                speaker_transcribed_input.submit(
                    self.vc_voice_chat,
                    inputs=[speaker_transcribed_input, activities_state],
                    outputs=[speaker_transcribed_input, activities_state], queue=False
                ).then(
                    self.vc_unhide_outputs,
                    inputs=None,
                    outputs=[bot_output_txt, bot_audio_output], queue=False
                ).then(fn=self.vc_query_parameters, 
                       inputs=[vc_selected_domain_expert,vc_selected_response_tyle,bot_api_host,bot_api_key,
                               bot_active_model,bot_system_prompt,bot_top_p,bot_temperature,
                               bot_max_tokens,bot_presence_penalty,
                               bot_stop_words,bot_frequency_penalty,
                               bot_gpu_offload_layers],queue=False
                ).then(
                    self.vc_ask_assistant_to_image,
                    inputs=[checkbox_enable_text_to_image,
                            speaker_transcribed_input, activities_state, self.history_state],
                    outputs=[activities_state, self.history_state,
                            bot_output_txt, bot_output_image,bot_status_line], queue=False
                ).then(
                    self.vc_add_query_log_entry,
                    inputs=[gr_query_log, speaker_transcribed_input, bot_output_txt],
                    outputs=[gr_query_log], queue=True
                ).then(
                    self.vc_convert_text_to_speech,
                    inputs=[vc_selected_voice,vc_voice_emotions,bot_output_txt, textbox_detected_spoken_language],
                    outputs=[bot_audio_output], queue=False
                )

                # unhide buttons
                checkbox_enable_text_to_image.change(
                    self.vc_unhide_outputs,
                    inputs=[checkbox_enable_text_to_image],
                    outputs=[bot_output_txt, bot_audio_output, bot_output_image,
                            btn_generate_request_image, btn_generate_response_image], queue=False
                )

                ## User Options
                vc_human_voices_options.change(fn=change_voice,inputs=[vc_human_voices_options], outputs=[vc_selected_voice])               
                vc_voice_emotions.change(fn=change_emotion, inputs=[vc_voice_emotions], outputs=[vc_selected_emotion])
                vc_domain_expert_options.change(fn=change_domain_export, inputs=[vc_domain_expert_options], outputs=[vc_selected_domain_expert,bot_system_prompt])
                vc_response_style_options.change(fn=change_response_tyle, inputs=[vc_response_style_options], outputs=[vc_selected_response_tyle])

                ## LLM mode
                ## "locally-aio", "solar-openai", "ollama-openai"
                                # in-line update event handler
                def update_active_model(new_model_id):
                    gr.Info(f"selected new model {new_model_id}")
                    return new_model_id
                
                bot_model_dropdown.change(fn=update_active_model,
                    inputs=[bot_model_dropdown],
                    outputs=[bot_active_model]
                )
                
                # radio_llm_mode.change(fn=self.vc_update_llm_mode, 
                #                         inputs=[radio_llm_mode], 
                #                         outputs=[bot_api_host,bot_api_key,bot_active_model,bot_model_dropdown])
                # "realtime/gradio_tutor/audio/hp0.wav"
                # "realtime/examples/jfk.wav"
                # ------------
                # gr.Examples(
                # examples=["workspace/samples/jfk.wav"],
                # inputs=[speaker_audio_input],
                # outputs=[speaker_transcribed_input, textbox_detected_spoken_language],
                # fn=speech_to_text,
                # cache_examples=False,
                # label="example audio file: president JFK speech"
                # )
        return self.blocks_voice_prompt
