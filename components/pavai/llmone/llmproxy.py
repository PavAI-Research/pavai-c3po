import time
from pathlib import Path
import traceback
from openai import OpenAI
import sys
import pavai.llmone.local.localllm as localllm
import pavai.llmone.chatprompt as chatprompt
import pavai.llmone.remote.chatbotllm as chatbotllm
from pavai.setup import config
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)
sys.path.append(str(Path(__file__).parent.parent))

logger.warn("--GLOBAL SYSTEM MODE----")
logger.warn(config.system_config["GLOBAL_SYSTEM_MODE"])
_GLOBAL_SYSTEM_MODE = config.system_config["GLOBAL_SYSTEM_MODE"]
_GLOBAL_TTS = config.system_config["GLOBAL_TTS"]
_GLOBAL_TTS_LIBRETTS_VOICE = config.system_config["GLOBAL_TTS_LIBRETTS_VOICE"]
_GLOBAL_STT = config.system_config["GLOBAL_STT"]
_GLOBAL_DEFAULT_LLM_MODEL_FILE = config.system_config["DEFAULT_LLM_MODEL_FILE"]

_GLOBAL_DEFAULT_LLM_MODEL_INFO = [config.system_config["DEFAULT_LLM_MODEL_FILE"],
                                  config.system_config["DEFAULT_LLM_MODEL_WEB_URL"],
                                  config.system_config["DEFAULT_LLM_MODEL_CHAT_FORMAT"]
                                  ]
# global
llmsolar = None

def chat_api_ui(input_text: str,
                chatbot: list, chat_history: list,
                system_prompt: str = None,
                ask_expert: str = None,
                model_id:str=None,
                response_style:str=None,                
                target_model_info: list = None,
                user_override: bool = False,
                skip_content_safety_check: bool = True,
                skip_data_security_check: bool = True,
                skip_self_critique_check: bool = True,
                user_settings: dict = None,):
    t0 = time.perf_counter()
    if input_text is None:
        logger.error("Empty input query is not valid!")
        return 
    
    ## prepare system prompt 
    if system_prompt is None or len(system_prompt.strip())==0:
        query_system_prompt=chatprompt.safe_system_prompt
    else:
        ## guard content safety in response 
        query_system_prompt = system_prompt +".\n"+chatprompt.guard_system_prompt

    ## user query
    if response_style is not None:
        query_user_prompt = input_text+f".\ Please respond in this writing style {response_style}"
    else:
        query_user_prompt = input_text

    ## context routing
    ## persona / expert (model)
    query_ask_expert = ask_expert if ask_expert is not None else None
    ## model id and model info
    # model_id = # target_model_info = 

    logger.info(f"LLM User Query: {input_text}")
    if isinstance(input_text, list):
        logger.info(f"[blue]{str(input_text)}[/blue]", extra=dict(markup=True))
        return chatbot, chat_history, input_text[0]
    if "/tmp/gradio" in input_text:
        return chatbot, chat_history, input_text

    # mode: "solar-openai", "ollama-openai"
    if _GLOBAL_SYSTEM_MODE == "solar-openai" or _GLOBAL_SYSTEM_MODE == "ollama-openai":
        reply_text, chat_history = chat_api_remote(user_prompt=query_user_prompt,
                                                   history=chat_history,
                                                   ask_expert=query_ask_expert,
                                                   system_prompt=query_system_prompt,
                                                   user_override=user_override,
                                                   skip_content_safety_check=skip_content_safety_check,
                                                   skip_data_security_check=skip_data_security_check,
                                                   skip_self_critique_check=skip_self_critique_check,
                                                   user_settings=user_settings,
                                                   model_id=model_id                                                   
                                                   )
    else:
        # mode: locally-aio
        chatbot_ui_messages, chat_history, reply_text = localllm.chat_completion(user_Prompt=query_user_prompt,
                                                                                 history=chat_history,
                                                                                 system_prompt=query_system_prompt,
                                                                                 ask_expert=query_ask_expert,
                                                                                 target_model_info=target_model_info,
                                                                                 user_settings=user_settings,
                                                                                 model_id=model_id
                                                                                 )
        logger.debug(f"[blue]{reply_text}[/blue]", extra=dict(markup=True))

        # update chatbot messages
        if chatbot is None:
            chatbot = []
        chatbot.append((input_text, reply_text))

    t1 = time.perf_counter()-t0
    logger.info(f"llmproxy.chat_api_ui: took {t1:.6f} seconds")
    return chatbot, chat_history, reply_text

def chat_api_remote(user_prompt: str = None, history: list = [],
                    system_prompt: str = chatprompt.system_prompt_assistant,
                    stop_criterias=["</s>"],
                    ask_expert: str = None,
                    target_model_info: list = None,
                    user_override: bool = False,
                    skip_content_safety_check: bool = True,
                    skip_data_security_check: bool = True,
                    skip_self_critique_check: bool = True,
                    user_settings: dict = None,
                    model_id:str=None   
                    ):

    t0 = time.perf_counter()
    reply_text = ""
    reply_messages=[]
    try:
        if _GLOBAL_SYSTEM_MODE == "ollama-openai":
            default_url = str(
                config.system_config["SOLAR_LLM_OLLAMA_HOST"]).strip()
            default_api_key = str(
                config.system_config["SOLAR_LLM_OLLAMA_API_KEY"]).strip()
            default_model_id = str(
                config.system_config["SOLAR_LLM_OLLAMA_MODEL_ID"]).strip()
            reply_text, reply_messages, reply_status = chatbotllm.api_calling_v2(
                provider="ollama-openai",
                api_host=default_url,
                api_key=default_api_key,
                active_model=default_model_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                history=history,
                stop_words=stop_criterias
            )
        else:
            default_url = str(
                config.system_config["SOLAR_LLM_DEFAULT_HOST"]).strip()
            default_api_key = str(
                config.system_config["SOLAR_LLM_DEFAULT_API_KEY"]).strip()
            default_model_id = str(
                config.system_config["SOLAR_LLM_DEFAULT_MODEL_ID"]).strip()
            reply_text, reply_messages, reply_status = chatbotllm.api_calling_v2(
                provider="solar-openai",
                api_host=default_url,
                api_key=default_api_key,
                active_model=default_model_id,
                prompt=user_prompt,
                system_prompt=system_prompt,
                history=history,
                stop_words=stop_criterias
            )
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        logger.error(f"llmproxy.chat_api_remote has error {str(e.args)}")
        localllm._free_gpu_resources()

    logger.debug(f"[blue]{reply_text}[/blue]", extra=dict(markup=True))
    t1 = time.perf_counter()-t0
    logger.debug(f"llmproxy.chat_api_remote: took {t1} seconds")
    return reply_text, reply_messages

def chat_api_local(user_Prompt: str, history: list = [],
                   system_prompt: str = localllm.safe_system_prompt,
                   stop_criterias=["</s>"],
                   ask_expert: str = None,
                   target_model_info: list = None,
                   user_settings: dict = None,
                   model_id:str=None   
                   ):
    global llm_client
    reply = None
    try:
        system_prompt = localllm.safe_system_prompt if system_prompt is None else system_prompt
        if target_model_info is not None and len(target_model_info) > 0:
            """load new model"""
            if llm_client is not None:
                llm_client._llm._pipeline = None
                llm_client._llm = None
                del llm_client
             # release previous model gpu resources
            localllm._free_gpu_resources()
            llm_client = localllm.new_llm_instance(target_model_info)
        # make call
        llm_client = create_llm_local()
        #messages, xhistory, reply = llmproxy.chat_api_local("hello", history=[])            
        # llm_client = localllm.get_llm_instance()
        history = [] if history is None else history

        # model parameters
        if user_settings is not None:
            activate_model_id = user_settings["_QUERY_MODEL_ID"]
            top_p = user_settings["_QUERY_TOP_P"]
            temperature = user_settings["_QUERY_TEMPERATURE"]
            max_tokens = user_settings["_QUERY_MAX_TOKENS"]
            present_penalty = user_settings["_QUERY_PRESENT_PENALTY"]
            stop_criterias = user_settings["_QUERY_STOP_WORDS"]
            frequency_penalty = user_settings["_QUERY_FREQUENCY_PENALTY"]
            # system_prompt = user_settings["_QUERY_SYSTEM_PROMPT"] = str(system_prompt).strip()
            # self.user_settings["_QUERY_DOMAIN_EXPERT"] = str(domain_expert).strip()
            # self.user_settings["_QUERY_RESPONSE_STYLE"] = str(response_style).strip()
            gpu_offload_layers = user_settings["_QUERY_GPU_OFFLOADING_LAYERS"]
            messages, history, reply = llm_client.simple_chat(prompt=user_Prompt,
                                                              history=history,
                                                              system_prompt=system_prompt,
                                                              stop_criterias=stop_criterias,
                                                              temperature=temperature,
                                                              top_p=top_p,
                                                              max_tokens=max_tokens)
        else:
            messages, history, reply = llm_client.simple_chat(prompt=user_Prompt,
                                                              history=history,
                                                              system_prompt=system_prompt,
                                                              stop_criterias=stop_criterias)
        return messages, history, reply
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        logger.error(f"llmproxy.chat_api_local has error {str(e.args)}")
        localllm._free_gpu_resources()
    return [], history, reply

def chat_api(user_prompt: str = None, history: list = [],
                    system_prompt: str = chatprompt.system_prompt_default,
                    stop_criterias=["</s>"],
                    ask_expert: str = None,
                    target_model_info: list = None,
                    user_override: bool = False,
                    skip_content_safety_check: bool = True,
                    skip_data_security_check: bool = True,
                    skip_self_critique_check: bool = True,
                    user_settings: dict = None
                    ):

    t0 = time.perf_counter()
    reply_text = ""
    reply_messages=[]
    if _GLOBAL_SYSTEM_MODE == "ollama-openai" or _GLOBAL_SYSTEM_MODE == "solar-openai":
        reply_text, reply_messages = chat_api_remote(
            user_prompt=user_prompt,
            history=history,
            system_prompt=system_prompt,
            stop_criterias=stop_criterias,
            ask_expert=ask_expert,
            target_model_info=target_model_info,
            user_override=user_override,
            user_settings=user_settings
        )     
    else:
        messages, reply_messages, reply_text = chat_api_local(
            user_Prompt=user_prompt, 
            history= history,
                   system_prompt = system_prompt,
                   stop_criterias=stop_criterias,
                   ask_expert = ask_expert,
                   target_model_info = target_model_info,
                   user_settings = user_settings
        )

    logger.debug(f"[blue]{reply_text}[/blue]", extra=dict(markup=True))
    t1 = time.perf_counter()-t0
    logger.debug(f"llmproxy.chat_api: took {t1} seconds")
    return reply_text, reply_messages

def chat_models_local(
        model_name_or_path: str = None,
        model_file: str = None,
        model_download: str = None,
        model_path: str = localllm.DEFAULT_LLM_MODEL_PATH,
        model_chat_format: str = localllm.DEFAULT_LLM_CHAT_FORMAT
):
    return localllm.AbstractLLMClass.get_llm_model(
        model_name_or_path=model_name_or_path,
        model_file=model_file,
        model_download=model_download,
        model_path=model_path
    )

def create_llm_local(runtime_file:str="resources/config/llm_defaults.json"):
    return localllm.get_llm_instance(runtime_file)

def normalize_imagechat_history(history: list):
    clean_history = history
    for i in range(0, len(history)-1):
        if history[i]["role"] != "system":
            if history[i]["role"] == "user":
                if isinstance(history[i]["content"], list):
                    image_question = history[i]["content"][1]['text']
                    image_url = history[i]["content"][0]['image_url']['url']
                    # image_response = history[i+1]["content"]
                    # clean_history.append((image_question,image_response))
                    image_question = "<p><image src='"+image_url + \
                        "' alt='show uploaded image'/>"+image_question+"</p>"
                    history[i]["content"] = image_question
    return clean_history

def word_count(text):
    return (len(text.strip().split(" ")))

def chat_count_tokens(text):
    llm_client = localllm.get_llm_instance()
    if llm_client._llm is not None and llm_client._llm._pipeline is not None:
        model = llm_client._llm._pipeline
        return len(model.tokenize(text.encode() if isinstance(text, str) else text))
    else:
        return word_count(text)
