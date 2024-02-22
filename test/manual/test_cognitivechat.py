from __future__ import annotations
from dotenv import dotenv_values
import time
import warnings
import logging
from rich import print, pretty, console
from rich.logging import RichHandler
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True,tracebacks_show_locals=True)])
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
pretty.install()
import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
logger.info(os.getcwd())

from openai import OpenAI
import openai
#import src.shared.llm.llm_chat as llm_chat
import src.shared.solar.llmcognitive as llmcognitive 
import src.shared.solar.llmprompt as llmprompt 

def test_solarnetwork_chat1():
    llmsc=llmcognitive.LLMSolarClient.new_instance("llm_api.json")
    # default_url="http://192.168.0.29:8004/v1"
    # domain_url="http://192.168.0.29:8004/v1"
    history=[]
    cleanup=False
    input_data = {
        "input_query": "which is taller Toronto CN-Tower or New York City Empire State Building?",
    }
    output,history=llmsc.chat(input_data,history,cleanup)
    logger.debug(f"[blue]{output['output_text']}[/blue]",extra=dict(markup=True))

    input_data = {
        "input_query": "which one is older Toronto CN-Tower or Empire State Building?",
    }
    output,history=llmsc.chat(input_data,history,cleanup)    
    logger.debug(f"[cyan]{output['output_text']}[/cyan]",extra=dict(markup=True))

    input_data = {
        "input_query": "nice, who build Empire State Building?",
    }
    output,history=llmsc.chat(input_data,history,cleanup)    
    logger.debug(f"[cyan]{output['output_text']}[/cyan]",extra=dict(markup=True))

def test_solarnetwork_chat2():
    t0=time.perf_counter()
    default_url="http://192.168.0.29:8004/v1"
    domain_url="http://192.168.0.29:8004/v1"
    solarclient = llmcognitive.LLMSolarClient(default_url=default_url,domain_url=domain_url)
    # input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
    # chat message basic one
    logger.info("chat message basic")
    user_query = {
        "input_query": "which is taller Toronto CN-Tower or New York City Empire State Building?",
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")
    t2=time.perf_counter() - t0
    logger.info("test-1 done in ", f"{t2:.6f}", " seconds")

def test_solarnetwork_multiple():
    """
    The client code parameterize invoker with any commands.
    """
    import time
    t0=time.perf_counter()
    default_url="http://192.168.0.29:8004/v1"
    domain_url="http://192.168.0.29:8004/v1"
    solarclient = llmcognitive.LLMSolarClient(default_url=default_url,domain_url=domain_url)
    # input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
    # chat message basic one
    logger.info("chat message basic")
    user_query = {
        "input_query": "list of the planet names in the solar system",
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")
    t2=time.perf_counter() - t0
    print("test-1 done in ", f"{t2:.6f}", " seconds")

    t3=time.perf_counter()
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")
    t4=time.perf_counter() - t3
    print("test-2 done in ", f"{t4:.6f}", " seconds")
    print("repeat call time saved ", f"{t2-t4:.6f}", " seconds")

    # chat message using ask-expert
    logger.info("chat message using ask-expert")
    user_query = {
        "input_query": "list of the planet names in the solar system",
        "ask_expert": "science_explainer",
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")

    # # chat message with specify model
    logger.info("chat message use specific model")
    user_query = {
        "input_type": "chat",
        "input_query": "list of the planet names in the solar system",
        "input_model_id": "zerphyr-7b-beta.Q4",
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")

    # chat message use different model
    logger.info("chat message use specific model")
    user_query = {
        "input_type": "chat",
        "input_query": "list of the planet names in the solar system",
        "input_model_id": "mistral-7b-instruct-v0.2",
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")

    # chat message include image
    logger.info("chat message include image")
    user_query = {
        "input_type": "chat",
        "input_query": "describe the image details",
        "input_image": "https://hips.hearstapps.com/hmg-prod/images/how-to-make-bath-bombs-1675185865.jpg",
        "input_model_id": "llava-v1.5-7b",
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")
    t1=time.perf_counter() - t0
    print("done in ", f"{t1:.6f}", " seconds")

def test_solarnetwork_self_critique():
    t0=time.perf_counter()
    default_url="http://192.168.0.29:8004/v1"
    domain_url="http://192.168.0.29:8004/v1"
    ##solarclient=llm_cognitive.LLMSolarClient.new_instance("llm_api.json")
    solarclient = llmcognitive.LLMSolarClient(default_url=default_url,domain_url=domain_url)
    # chat message include self-critique (add additional time)
    logger.info("chat message include self-critique")
    user_query = {
        "input_type": "chat",
        "input_query": "what is the distance between earth and the moon?",
        "input_second_option_source": "llm",
        "input_second_option_model_id": "mistral-7b-instruct-v0.2"
    }
    output, history = solarclient.chat(input_data=user_query)
    logger.warn(f"Question:\n{output['input_text']}")
    logger.warn(f"Final Answer:\n{output['output_text']}")
    t1=time.perf_counter() - t0
    logger.info("done in ", f"{t1:.6f}", " seconds")
    #print(history)

if __name__ == "__main__":
    print(llmprompt.guard_system_prompt_assistant)
    test_solarnetwork_chat1()
    # test_solarnetwork_chat2()
    # test_solarnetwork_self_critique()
    # test_solarnetwork_multiple()

