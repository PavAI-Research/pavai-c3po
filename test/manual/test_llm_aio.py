import logging
import warnings
from dotenv import dotenv_values
from rich import pretty
from rich.logging import RichHandler
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))
logger.info(os.getcwd())

import src.shared.solar.llmprompt as llmprompt
import src.shared.llmproxy as llmproxy
import src.shared.aio.llmchat as llmchat

llm_local = {
    "model_runtime": "llamacpp",
    "model_id": None,
    "model_file_name": "zephyr-7b-beta.Q4_K_M.gguf",
    "model_file_url": "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
    "model_source": "filesystem",
    "model_architecure": "llama-2",
    "api_base": None,
    "api_key": None,
    "api_domain": None,
    "api_organization": None,
    "use_device": "cuda",
    "use_torch_type": "float16",
    "model_tokenizer_id_or_path": None,
    "use_flash_attention_2": "",
    "use_download_root": "/home/pop/development/mclab/talking-llama/models/llm/",
    "use_local_model_only": True,
    "use_max_context_size": 4096*2,
    "use_n_gpu_layers": 38,
    "verbose": True,
    "use_seed": 123
}

def test_local_embedding():
    logger.info("***TEST EMBEDDING***")
    llm_client = llmchat.LLMClient(llmchat.LLMllamaLocal(**llm_local))
    reply = llm_client.simple_embedding(
        input="hello", model="text-embedding-ada-002")
    logger.info(reply)
    logger.info("\n")


def test_llmproxy_basic():
    logger.info("***TEST LLM Proxy***")
    prompt = "tell me about Toronto CN-Tower restaurant meals"
    chatbot_ui_messages, chat_history, reply_text = llmproxy.chatbot_ui_client(
        input_text=prompt, chat_history=[])
    logger.info(reply_text)
    logger.info("\n")
    # prompt="planets in the solar system are "
    # chatbot_ui_messages, chat_history, reply_text  = llmproxy.chatbot_ui_client(input_text=prompt,chat_history=[])
    # logger.info(reply_text)
    # logger.info("\n")


def test_llmproxy_change_model():
    # mistral-7b-instruct-v0.2.Q4_K_M.gguf
    # expert_prompt=llmprompt.knowledge_experts_system_prompts["trivia_master"]
    expert_model = llmchat.gguf_map["mistral-7b-instruct-v0.2.Q4_K_M.gguf"]
    prompt = "who live longer human or dinasour?"
    chatbot_ui_messages, chat_history, reply_text = llmproxy.chatbot_ui_client(
        input_text=prompt, chat_history=[],
        target_model_info=expert_model)
    logger.info(reply_text)
    logger.info("\n")


def test_llmproxy_ask_experts():
    # zephyr-7b-beta.Q4_K_M.gguf
    expert_prompt = llmprompt.knowledge_experts_system_prompts["personal_finance_advisor"]
    expert_model = llmchat.gguf_map["zephyr-7b-beta.Q4_K_M.gguf"]
    prompt = "which investment is better common stock or options?"
    print("expert prompt: ", expert_prompt)
    chatbot_ui_messages, chat_history, reply_text = llmproxy.chatbot_ui_client(
        input_text=prompt, chat_history=[],
        system_prompt=expert_prompt,
        ask_expert="personal_finance_advisor",
        target_model_info=expert_model)
    logger.info(reply_text)
    logger.info("\n")

def test_llmproxy_chatservice():
    # zephyr-7b-beta.Q5_K_M.gguf
    expert_prompt = llmprompt.knowledge_experts_system_prompts["personal_finance_advisor"]
    expert_model = llmchat.gguf_map["zephyr-7b-beta.Q5_K_M.gguf"]
    prompt = "which investment is better common stock or options?"
    print("expert prompt: ", expert_prompt)    
    chatbot_ui_messages, chat_history, reply_text = llmproxy.chat_service(
        input_text=prompt, chat_history=[],
        system_prompt=expert_prompt,
        ask_expert="personal_finance_advisor",        
        target_model_info=expert_model)
    logger.info(reply_text)
    logger.info("\n")

def test_local_completion():
    logger.info("***TEST COMPLETION***")
    llm_client = llmchat.LLMClient(llmchat.LLMllamaLocal(**llm_local))
    reply, status_code = llm_client.simple_completion(
        prompt="planets in the solar system are ")
    logger.info(reply)
    logger.info("\n")
    reply, status_code = llm_client.simple_completion(
        prompt="write a quick sort in python")
    logger.info(reply)
    logger.info("\n")


def test_local_multi_turn_conversation():
    logger.info("***TEST CONVERSATION MODE***")
    llm_client = llmchat.LLMClient(llmchat.LLMllamaLocal(**llm_local))

    # test-1 conversation start
    messages, history, reply = llm_client.simple_chat(prompt="hey", history=[],
                                                      stop_criterias=["</s>"],
                                                      system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # ------------------------------
    # sample conversation
    # ------------------------------
    logger.info("---"*20)

    # test ask question-1
    messages, history, reply = llm_client.simple_chat(prompt="I need some help on writing a formal resume. any suggestion?", history=[],
                                                      stop_criterias=["</s>"],
                                                      system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")
    # test ask question-2
    messages, history, reply = llm_client.simple_chat(prompt="can you give me good sample of product manager profile summary?", history=history,
                                                      stop_criterias=["</s>"],
                                                      system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # test ask question-3
    messages, history, reply = llm_client.simple_chat(prompt="Is adding references necessary?, if Yes, then how many references works the best",
                                                      history=history,
                                                      stop_criterias=["</s>"],
                                                      system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # test ask question-4
    messages, history, reply = llm_client.simple_chat(prompt="how many pages a good resume should be?",
                                                      history=history,
                                                      stop_criterias=["</s>"],
                                                      system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # test ask question-5
    messages, history, reply = llm_client.simple_chat(prompt="thanks for your help!",
                                                      history=history,
                                                      stop_criterias=["</s>"],
                                                      system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n\n")


def test_local_ask_standalone_questions():
    logger.info("***TEST ASK QUESTIONS including domain expert mode***")
    # llm_settings = LLM_Setting("llm_local.json")
    # llm_client = LLMClient(LLMllamaLocal(**llm_settings.config))
    # logger.info(llm_settings.config)
    llm_client = llm_client.LLMClient(llm_client.LLMllamaLocal(**llm_local))
    # ------------------------------
    # chatbot individual questions
    # ------------------------------
    logger.info("---"*20)

    # general question-1
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me about Toronto CN-Tower", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")
    # domain expert: question-1
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me about Toronto CN-Tower", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_trivia_master)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-2
    messages, history, reply = llm_client.simple_chat(
        prompt="can you remind me, who was the 38th president of united states?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")
    # domain expert: question-2
    messages, history, reply = llm_client.simple_chat(
        prompt="can you remind me, who was the 38th president of united states?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_historical_expert)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-3
    messages, history, reply = llm_client.simple_chat(
        prompt="why pi has infinite value?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # domain expert: question-3
    messages, history, reply = llm_client.simple_chat(
        prompt="why pi has infinite value?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_math_tutor)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-4
    messages, history, reply = llm_client.simple_chat(
        prompt="what is 4096 times 4?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_math_tutor)
    logger.info(messages[-1])
    logger.info("\n")

    # domain expert: question-4
    messages, history, reply = llm_client.simple_chat(
        prompt="what is 4096 times 4?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_math_tutor)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-5
    messages, history, reply = llm_client.simple_chat(
        prompt="who live longer human or dinosaur?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # domain expert: question-5
    messages, history, reply = llm_client.simple_chat(
        prompt="who live longer human or dinosaur?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_trivia_master)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-6
    messages, history, reply = llm_client.simple_chat(
        prompt="how to stay healhy?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # domain expert: question-6
    messages, history, reply = llm_client.simple_chat(
        prompt="how to stay healhy?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_nutritionist)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-7
    messages, history, reply = llm_client.simple_chat(
        prompt="any suggesttion on a fun birthday party on weekend?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # domain expert: question-7
    messages, history, reply = llm_client.simple_chat(
        prompt="any suggesttion on a fun birthday party on weekend?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_party_planner)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-8
    messages, history, reply = llm_client.simple_chat(
        prompt="can you provide step-by-step instruction on how to remove virus from my computer?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")

    # domain expert: question-8
    messages, history, reply = llm_client.simple_chat(
        prompt="can you provide step-by-step instruction on how to remove virus from my computer?", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_cyber_security_specialist)
    logger.info(messages[-1])
    logger.info("\n")

    logger.info("---"*20)

    # general question-9
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me planets in the solar system", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_assistant)
    logger.info(messages[-1])
    logger.info("\n")
    # domain expert: question-9
    messages, history, reply = llm_client.simple_chat(
        prompt="tell me planets in the solar system", history=[],
        stop_criterias=["</s>"], system_prompt=llm_client.system_prompt_science_explainer)
    logger.info(messages[-1])
    logger.info("\n")


def test_api_calls():
    llm_settings = llm_client.LLM_Setting("llm_api.json")
    llm_client = llm_client.LLMClient(
        llm_client.LLMllamaOpenAI(**llm_settings.config))
    logger.info(llm_settings.config)
    messages, history, reply = llm_client.simple_chat(
        prompt="hello", history=[])
    logger.info(reply)
    logger.info("\n")
    reply, status_code = llm_client.simple_completion(
        prompt="write a quick sort in python")
    logger.info(reply)
    logger.info("api done!")


if __name__ == "__main__":
    test_llmproxy_basic()
    test_llmproxy_change_model()
    test_llmproxy_ask_experts()
    # from psutil import virtual_memory
    # ram_gb = virtual_memory().total / 1e9
    # logger.info(f"available RAM: {ram_gb}")
    # test_local_embedding()
    # test_local_completion()
    # test_local_multi_turn_conversation()
    # test_local_ask_standalone_questions()
    logger.info("local done!\n\n")
