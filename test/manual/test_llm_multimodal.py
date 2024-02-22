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
print(os.getcwd())
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
import openai
import src.shared.solar.llmchat as llmchat
import src.shared.solar.llmcognitive as llmcognitive 
import src.shared.solar.llmprompt as llmprompt 

def test_multimodalchat_llava_1_5(domain_client):
    print("-----multimodalchat:llava_1_5-----")
    # include image
    print("\nllava TEST-1\n")
    user_image_url = "https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/02/Left-Handed-Male-Writing-1296x728-Header.jpg?w=1155&h=1528"
    user_query = "What does the image say about the writing style?"
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            history=[],
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "text"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")

    # no image
    print("\nllava TEST-2 - no image\n")
    user_query = "Anything different between left-hand and right-hand writing styles?"
    history, reply_object= llmchat.multimodalchat(client=domain_client, query=user_query, history=[],
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "text"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")

    # include image
    print("llava TEST-3")
    user_image_url = "https://cdn11.bigcommerce.com/s-a1x7hg2jgk/images/stencil/1280x1280/products/28260/152885/epson-tm-u300pc-receipt-printer-pos-broken-hinge-on-cover-3.24__38898.1490224087.jpg?c=2?imbypass=on"
    user_query = "What does the image say about ports available? Format your response as a json object with a single 'text' key and 'specification' key"
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            history=history,
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "json_object"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")

    # include image
    print("llava TEST-4")
    user_image_url = "https://forums.flightsimulator.com/uploads/default/original/4X/6/f/1/6f1082be7b0331719aa121824c56b6950a4c303e.jpeg"
    user_query = "Extract all visible text from the image. Format your response as a json object."
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            history=history,
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "json_object"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")

    # no image
    print("\nllava TEST-5 - no image\n")
    user_query = "please summary the conversation in key points."
    history, reply_object = llmchat.multimodalchat(client=domain_client, query=user_query, history=history,
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "text"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")


def test_multimodalchat_bakllava1(domain_client):
    print("-----multimodalchat:bakllava1-----")
    print("bakllava TEST-0 - no image")
    user_query = "please list top 5 largest cities in canada in term of population, economic mertrics?"
    history, reply_object = llmchat.multimodalchat(client=domain_client, query=user_query, history=[], 
                                                    system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "text"}, model_id="bakllava1")
    print(reply_object["response"], end="\n")

    # include image
    print("bakllava TEST-1")
    user_image_url = "https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/02/Left-Handed-Male-Writing-1296x728-Header.jpg?w=1155&h=1528"
    user_query = "What does the image say about the writing style? Format your response as a json object with a single 'text' key and 'rating' key in scale of 1 to 10."
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "json_object"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")
    print("bakllava TEST-2")
    # include image
    user_image_url = "https://cdn11.bigcommerce.com/s-a1x7hg2jgk/images/stencil/1280x1280/products/28260/152885/epson-tm-u300pc-receipt-printer-pos-broken-hinge-on-cover-3.24__38898.1490224087.jpg?c=2?imbypass=on"
    user_query = "What does the image say about ports available? Format your response as a json object with a single 'text' key and 'specification' key"
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "json_object"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")

    # include image
    print("bakllava TEST-3")
    user_image_url = "https://forums.flightsimulator.com/uploads/default/original/4X/6/f/1/6f1082be7b0331719aa121824c56b6950a4c303e.jpeg"
    user_query = "Extract all visible text from the image. Format your response as a json object."
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "json_object"}, model_id="llava-v1.5-7b")
    print(reply_object["response"], end="\n")


def normalize_imagechat_history(history:list):
    clean_history=history
    for i in range(0, len(history)-1):
        if history[i]["role"] != "system":
            if history[i]["role"] == "user":
                if isinstance(history[i]["content"],list):   
                    image_question= history[i]["content"][1]['text']   
                    image_url = history[i]["content"][0]['image_url']['url'] 
                    #image_response = history[i+1]["content"]   
                    #clean_history.append((image_question,image_response))
                    image_question=image_question+"<p><img src='"+image_url+"' alt='uploaded image'>"+image_question+"</p>"
                    history[i]["content"]=image_question
    return clean_history

if __name__ == "__main__":
    guard_client = OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        base_url="http://192.168.0.29:8004/v1"
    )
    domain_client = OpenAI(
        # This is the default and can be omitted
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        base_url="http://192.168.0.29:8004/v1"
    )
    # test_multimodalchat_llava_1_5(domain_client)

    # include image
    print("\nllava TEST-1\n")
    local_image_file="/tmp/gradio/26f63cc5e677d6e4b106a13413ac942c021ecb67/invoice1.png"
    user_image_url = "file:///"+local_image_file
    user_query = "describe what does the image say?"
    history, reply_object = llmchat.multimodalchat(client=domain_client,
                                            query=user_query,
                                            image_url=user_image_url,
                                            history=[],
                                            system_prompt=llmprompt.system_prompt_assistant,
                                            response_format={"type": "text"}, model_id="llava-v1.5-7b")
    
    ## normalize history to text content only
    clean_history=normalize_imagechat_history(history)
    #print(reply_object["response"], end="\n")
    print(clean_history, end="\n")    

    # #self-critique        # Enables `response_model`
    # #default_client = instructor.patch(client=guard_client)
    # test_gather_second_options(guard_client,domain_client)
    # test_moderate_and_query_skip_safety_check(guard_client,domain_client)
    # 写一首诗庆典中国新年 2024
    
