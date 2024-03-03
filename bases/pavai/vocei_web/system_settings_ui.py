from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import time
import gradio as gr
import torch
from transformers.utils import is_flash_attn_2_available
import pavai.llmone.datasecurity as datasecurity
import pavai.llmone.llmproxy as llmproxy
import pavai.llmone.chatprompt as chatprompt

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
        self.set_user_settings()
        self.new_domain_expert=None
        self.new_response_style=None
        self.new_voice =None
        self.new_emotion =None       
        self.chatbot=[] 
        self.new_system_prompt=None

    def set_user_settings(self):
        self.user_settings={}
        self.user_settings["_QUERY_ENABLE_PII_ANALYSIS"]=eval(config.system_config["_QUERY_ENABLE_PII_ANALYSIS"])
        self.user_settings["_QUERY_ENABLE_PII_ANONYMIZATION"]=eval(config.system_config["_QUERY_ENABLE_PII_ANONYMIZATION"])
        self.user_settings["_QUERY_CONTENT_SAFETY_GUARD"]=eval(config.system_config["_QUERY_CONTENT_SAFETY_GUARD"])
        self.user_settings["_QUERY_ASK_EXPERT_ID"]=None
        self.user_settings["_QUERY_ASK_EXPERT_PROMPT"]=None        
        self.user_settings["_QUERY_LLM_MODEL_ID"]=None
        self.user_settings["_QUERY_LLM_MODEL_INFO"]=None     
        ## LLM
        self.user_settings["_QUERY_DOMAIN_EXPERT"]=None
        self.user_settings["_QUERY_RESPONSE_STYLE"]=None     
        self.user_settings["_QUERY_SYSTEM_PROMPT"]=None     

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

    def chat_client(self,input_text:str, chat_history:list,user_settings:dict):
        #_GLOBAL
        # _QUERY_ASK_EXPERT_ID=user_settings["_QUERY_ASK_EXPERT_ID"]
        # _QUERY_ASK_EXPERT_PROMPT=user_settings["_QUERY_ASK_EXPERT_PROMPT"]
        # if _QUERY_ASK_EXPERT_ID is not None and len(_QUERY_ASK_EXPERT_ID)==0:
        #     _QUERY_ASK_EXPERT_ID = None                
        _QUERY_LLM_MODEL_ID=user_settings["_QUERY_MODEL_ID"]
        _QUERY_LLM_MODEL_BASE=user_settings["_QUERY_API_BASE"]
        _QUERY_LLM_MODEL_KEY=user_settings["_QUERY_API_KEY"]     
        target_model_info=[_QUERY_LLM_MODEL_ID,_QUERY_LLM_MODEL_BASE,_QUERY_LLM_MODEL_KEY]   

        _QUERY_ENABLE_PII_ANALYSIS=user_settings["_QUERY_ENABLE_PII_ANALYSIS"]
        _QUERY_ENABLE_PII_ANONYMIZATION=user_settings["_QUERY_ENABLE_PII_ANONYMIZATION"]
        _QUERY_CONTENT_SAFETY_GUARD=user_settings["_QUERY_CONTENT_SAFETY_GUARD"]
        ### LLM
        _QUERY_DOMAIN_EXPERT=user_settings["_QUERY_DOMAIN_EXPERT"]
        _QUERY_RESPONSE_STYLE=user_settings["_QUERY_RESPONSE_STYLE"]
        ### pick up from UI    
        _QUERY_SYSTEM_EXPERT_PROMPT = user_settings["_QUERY_SYSTEM_PROMPT"]        
        if _QUERY_DOMAIN_EXPERT is not None and len(_QUERY_DOMAIN_EXPERT)>0:
            _QUERY_SYSTEM_EXPERT_PROMPT=chatprompt.domain_experts[_QUERY_DOMAIN_EXPERT]
        if _QUERY_DOMAIN_EXPERT is not None and len(_QUERY_DOMAIN_EXPERT)==0:
            _QUERY_DOMAIN_EXPERT = None        
        if _QUERY_RESPONSE_STYLE is not None and len(_QUERY_RESPONSE_STYLE)==0:
            _QUERY_RESPONSE_STYLE = None        
        logger.info("<chat>")
        logger.info(f"target llm: {target_model_info}")
        logger.info(f"model id         : {_QUERY_LLM_MODEL_ID}")
        logger.info(f"persona(expert)  : {_QUERY_DOMAIN_EXPERT}")
        logger.info(f"query response   : {_QUERY_RESPONSE_STYLE}") 
        logger.info(f"system prompt    : {_QUERY_SYSTEM_EXPERT_PROMPT}")
        logger.info(f"user prompt      : {input_text}")        
        logger.info("</chat>")
        ## Model Parameters
        t0=time.perf_counter()
        warnings=[]
        # content handling flags
        user_override=False
        skip_content_safety_check=True
        skip_data_security_check=True            
        skip_self_critique_check=True
        #gr.Info("working on it...")    
        """data security pre-check"""
        if _QUERY_ENABLE_PII_ANALYSIS or _QUERY_ENABLE_PII_ANONYMIZATION:                
            skip_data_security_check=False 
            user_override=True  
        if _QUERY_CONTENT_SAFETY_GUARD:
            skip_content_safety_check=False
            user_override=True               
        ## data security check on input
        if _QUERY_ENABLE_PII_ANALYSIS:
            if isinstance(input_text, str):
                if "/tmp/gradio" not in input_text:
                    pii_results = datasecurity.analyze_text(input_text=input_text)
                    logger.info(pii_results,extra=dict(markup=True))  
                    if len(pii_results)>0:
                        gr.Warning('PII detected in INPUT text.')  
                        warnings.append(f'[Warning]: PII detected in INPUT {str(pii_results)}| ')      
    
        gr.Info("routing UI request to LLM")    
        chatbot_ui_messages, chat_history, reply_text = llmproxy.chat_api_ui(input_text=input_text, 
                                                                    chatbot=[],
                                                                    chat_history=chat_history,
                                                                    system_prompt=_QUERY_SYSTEM_EXPERT_PROMPT,
                                                                    ask_expert=_QUERY_DOMAIN_EXPERT,
                                                                    model_id=_QUERY_LLM_MODEL_ID,
                                                                    response_style = _QUERY_RESPONSE_STYLE,
                                                                    target_model_info=target_model_info,
                                                                    user_override=user_override,
                                                                    skip_content_safety_check=skip_content_safety_check,
                                                                    skip_data_security_check=skip_data_security_check,
                                                                    skip_self_critique_check=skip_self_critique_check,
                                                                    user_settings=user_settings)                        
        """data security post-check"""        
        pii_results=None
        if _QUERY_ENABLE_PII_ANALYSIS:
            if isinstance(reply_text, str):            
                pii_results = datasecurity.analyze_text(input_text=reply_text)
                logger.warn(pii_results,extra=dict(markup=True))
                if len(pii_results)>0:
                    gr.Warning('PII detected in OUTPUT text.')           
                    warnings.append(f'[Warning]: PII detected in OUTPUT {str(pii_results)}. |')                                  

        if _QUERY_ENABLE_PII_ANONYMIZATION:
            if isinstance(reply_text, str):                        
                if pii_results:
                    reply_text = datasecurity.anonymize_text(input_text=reply_text,analyzer_results=pii_results)                
                else:
                    reply_text = datasecurity.anonymize_text(input_text=reply_text)
                logger.info(pii_results,extra=dict(markup=True))
                gr.Warning('OUTPUT text has been anonymized')
                warnings.append('> OUTPUT text has been anonymized.')                                  

        warning_text=""
        warnings.append(f" it took {(time.perf_counter()-t0):.2f} seconds")        
        if len(warnings)>0:
            warning_text="<p style='color:blue;'>"
            warning_text=warning_text+" ".join(warnings)
            warning_text=warning_text+"</p>"
        return chatbot_ui_messages, chat_history, reply_text,warning_text


