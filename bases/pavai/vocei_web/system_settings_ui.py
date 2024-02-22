## SystemSetting.py
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

import gradio as gr
import torch

from transformers.utils import is_flash_attn_2_available
import pavai.shared.datasecurity as datasecurity
from pavai.shared.llmproxy import chatbot_ui_client


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

"""UI-SYSTEM SETTINGS"""
class SystemSetting:

    def __init__(self) -> None:
        self.history_state = gr.State([])
        self.session_tokens = gr.State(0)
        self.detected_lang_state = gr.State("en")

    def chunk_text_to_fixed_length(self,text: str, length: int):
        text = text.strip()
        result = [text[0+i:length+i] for i in range(0, len(text), length)]
        return result

    def chunk_sentences_to_fixed_length(self, sentences: str, max_length: int = 768):
        if len(sentences)>max_length:
            fixed_size_sentences = []
            chunks = self.chunk_text_to_fixed_length(text=sentences, length=max_length)        
            for chunk in chunks:
                fixed_size_sentences.append(chunk)
            return fixed_size_sentences
        else:
            return sentences

    def chat_client(self,input_text:str, chat_history:list):
        #_GLOBAL
        global _QUERY_ASK_EXPERT_ID
        global _QUERY_ASK_EXPERT_PROMPT
        global _QUERY_LLM_MODEL_ID
        global _QUERY_LLM_MODEL_INFO
        global _QUERY_ENABLE_PII_ANALYSIS
        global _QUERY_ENABLE_PII_ANONYMIZATION
        # content handling flags
        user_override=False
        skip_content_safety_check=True
        skip_data_security_check=True            
        skip_self_critique_check=True
        gr.Info("working on it...")    
        if _QUERY_ENABLE_PII_ANALYSIS or _QUERY_ENABLE_PII_ANONYMIZATION:                
            skip_data_security_check=False 
            user_override=True  
        if _QUERY_CONTENT_SAFETY_GUARD:
            skip_content_safety_check=False
            user_override=True               
        ## data security check on input
        if _QUERY_ENABLE_PII_ANALYSIS:
            pii_results = datasecurity.analyze_text(input_text=input_text)
            logger.info(pii_results,extra=dict(markup=True))  
            gr.Warning('PII detected in INPUT text.')        
        if _QUERY_ASK_EXPERT_PROMPT is None:
            chatbot_ui_messages, chat_history, reply_text = chatbot_ui_client(input_text=input_text, 
                                                                        chat_history=chat_history,
                                                                        ask_expert=_QUERY_ASK_EXPERT_ID,
                                                                        target_model_info=_QUERY_LLM_MODEL_INFO,
                                                                        user_override=user_override,
                                                                        skip_content_safety_check=skip_content_safety_check,
                                                                        skip_data_security_check=skip_data_security_check,
                                                                        skip_self_critique_check=skip_self_critique_check)
        else:
            gr.Info("routing request to domain experts")    
            chatbot_ui_messages, chat_history, reply_text = chatbot_ui_client(input_text=input_text, 
                                                                        chat_history=chat_history,
                                                                        system_prompt=_QUERY_ASK_EXPERT_PROMPT,
                                                                        ask_expert=_QUERY_ASK_EXPERT_ID,
                                                                        target_model_info=_QUERY_LLM_MODEL_INFO,
                                                                        user_override=user_override,
                                                                        skip_content_safety_check=skip_content_safety_check,
                                                                        skip_data_security_check=skip_data_security_check,
                                                                        skip_self_critique_check=skip_self_critique_check)                        
        ## data security check on output
        pii_results=None
        if _QUERY_ENABLE_PII_ANALYSIS:
            pii_results = datasecurity.analyze_text(input_text=reply_text)
            logger.warn(pii_results,extra=dict(markup=True))
            gr.Warning('PII detected in OUTPUT text.')                           

        if _QUERY_ENABLE_PII_ANONYMIZATION:
            if pii_results:
                reply_text = datasecurity.anonymize_text(input_text=reply_text,analyzer_results=pii_results)                
            else:
                reply_text = datasecurity.anonymize_text(input_text=reply_text)
            logger.info(pii_results,extra=dict(markup=True))
            gr.Warning('OUTPUT text. anonymized')

        return chatbot_ui_messages, chat_history, reply_text

    def save_settings(self,settings):
        pass

    def build_system_setting_ui(self):
        self.blocks_system_settings = gr.Blocks(title="SYSTEM SETTINGS-UI",analytics_enabled=False)
        with self.blocks_system_settings as llm_setting_ui:
            with gr.Group():        
                textbox_workspace = gr.Text(label="workspace", value="workspace",placeholder="data storage location",interactive=True)
                gr.Markdown("#### LLM Model and hyper-parameters")
                with gr.Row():
                    llm_model_path = gr.Text(
                        label="model_path", value="resources/models/llm", interactive=True)
                    llm_model_file = gr.Text(
                        label="model_file", value="zephyr-7b-beta.Q5_K_M.gguf", interactive=True)
                with gr.Row():
                    llm_model_chat_format = gr.Text(
                        label="model_chat_format", value="chatml", interactive=True)
                    llm_offload_gpu_layers = gr.Textbox(
                        value=35, label="offload gpu layers", interactive=True)

                gr.Markdown("#### Whisper Model")
                with gr.Row():
                    stt_model_size = gr.Text(
                        label="stt model size", value="large", interactive=True)
                    stt_model_name = gr.Text(
                        label="stt model name", value="Systran/faster-whisper-large-v3", interactive=True)

                gr.Markdown("#### Grammar synthesis model")
                with gr.Row():
                    grammar_model_size = gr.Text(
                        label="grammar model size", value="pszemraj/grammar-synthesis-small", interactive=True)

                gr.Markdown("#### AI Voice model")
                with gr.Row():
                    voice_model_language = gr.Dropdown(
                        choices=["en", "fr", "zh", "es"], value="en", label="Language", interactive=True)
                    voice_model_gender = gr.Dropdown(
                        choices=["Amy"], value="Amy", label="model gender", interactive=True)
                with gr.Row():
                    voice_model_onnx_file = gr.Textbox(
                        label="model onnx file", value="./models/voices/en_US-amy-medium.onnx", interactive=True)
                    voice_model_json_file = gr.Textbox(
                        label="model json file", value="./models/voices/en_US-amy-medium.onnx.json", interactive=True)
            btn_save_settings = gr.Button(value="Save Changes")
            btn_save_settings.click(self.save_settings,
                                    inputs=[textbox_workspace, llm_model_path,
                                            llm_model_file, llm_model_chat_format],
                                    outputs=None, queue=False)
        return self.blocks_system_settings


