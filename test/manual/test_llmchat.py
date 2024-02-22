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


def test_simple_chat_1(domain_client, system_prompt=llmprompt.system_prompt_default):
    logger.info("-----test_simple_chat_1-----")
    user_query = "How to erase or kill a linux process?"
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")


def test_simple_chat_2(domain_client, system_prompt=llmprompt.system_prompt_default):
    logger.info("-----test_simple_chat_2-----")
    user_query = "tell me a joke of the day"
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")


def test_simple_chat_3(domain_client, system_prompt=llmprompt.system_prompt_default):
    user_query = "I need a pet at home. which species is human friendly godzilla or dinosaur?"
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")


def test_simple_chat_health(domain_client, system_prompt=llmprompt.system_prompt_default, model_id: str = "zerphyr-7b-beta.Q4"):
    user_query = "what is the aim for blood pressure treatment goal and Hg range?"
    history = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_query}]
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=[])
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "can you list top 3 medicines used to treat high blood pressure?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "so, is Angiotensin II receptor blockers (ARBs) safe to use? what are the chemical ingredients."
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "I don't want take any drugs and live longer. can you suggest natural foods replacements instead?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "interesting... why consume tabacco is not the recommendation list?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "okay, eat healhy good, exercise regularly and reduce alcohol or tabacco improve longervity. is that sounds right?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "hmm?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "I don't understand why we need drinking plenty of water throughout the day?."
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "understood."
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "I still you did not answer what the risks on over driking lots of water. can you elaborate on that. please."
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "I am getting tired. could you summary the conversation in one sentence. please"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    # --- dump to a file to testing purpose
    import json
    with open('simplechat_health.json', 'w') as f:
        json.dump(history, f)


def test_simple_chat_sport_ml(domain_client, system_prompt=llmprompt.system_prompt_default, model_id: str = "zerphyr-7b-beta.Q4"):
    user_query = "What does love all mean in tennis?"
    history = [{"role": "system", "content": system_prompt},
               {"role": "user", "content": user_query}]
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=[])
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "¿Por qué al 0 se le llama amor en el tenis?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds\n")

    user_query = "打網球時為什麼不能觸網"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    user_query = "간단한 용어로 기본 테니스 규칙을 어떻게 플레이합니까?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    user_query = "我还是有点迷茫。请解释一下网球计分系统？"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    # portuguese
    user_query = "Por que chega a 15 30 40 no tênis?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    user_query = "それで、テニスのサーブは何回ありますか?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    user_query = "Pouvez-vous frapper la balle avant qu'elle ne rebondisse au tennis ?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    user_query = "Entonces, ¿cómo jugamos al tenis?"
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    user_query = "Danke für deine Erklärung, lass uns das Spiel jetzt spielen."
    history, reply_object = llmchat.simplechat(client=domain_client, model_id=model_id,
                                                   query=user_query, system_prompt=system_prompt, history=history)
    print(reply_object["response"])
    print("took ", reply_object["performance"], " seconds")

    # --- dump to a file to testing purpose
    import json
    with open('simplechat_sport_ml.json', 'w') as f:
        json.dump(history, f)


def test_simple_chat_ml(domain_client, system_prompt=llmprompt.system_prompt_default):
    # --english
    user_query = "who live longer human or dinosaur?"    
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])
    # --spanish 
    user_query = "¿Quién vive más tiempo el humano o el dinosaurio?"       
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])
    # --french
    user_query = "qui vit plus longtemps humain ou dinosaure?"           
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])
    # -- chinese
    user_query = "人類和恐龍誰壽命更長？"               
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])
    # -- arabic
    user_query = "من يعيش أطول الإنسان أم الديناصور؟"                   
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])
    # -- japanese 
    user_query = "人間と恐竜ではどちらが長生きしますか?"                  
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])
    # -- korean
    user_query = "인간과 공룡 중 누가 더 오래 살까요?"                    
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])


def test_stream_chat_problem_solving(domain_client):
    query = "how to solve E = mc^2 to speed up time travel?"
    response_stream = llmchat.streamchat(domain_client, query)
    sentence = ""
    for word in response_stream:
        if word is not None:
            if word is not None:
                sentence = sentence+word
                if len(sentence) > 30:
                    print(sentence, end="")
                    sentence = ""

    print("\n***", end="\n")
    query = "how to place cooking oil at the bottom of the cup fill with water"
    response_stream = llmchat.streamchat(domain_client, query)
    sentence = ""
    for word in response_stream:
        if word is not None:
            if word is not None:
                sentence = sentence+word
                if len(sentence) > 30:
                    print(sentence, end="")
                    sentence = ""


def test_stream_chat_poem(domain_client, model_id: str = "zerphyr-7b-beta.Q4", system_prompt: str = llmprompt.system_prompt_assistant):
    query = "write me a poem about working from home is a lucky day"
    response_stream = llmchat.streamchat(
        domain_client, query, model_id=model_id, system_prompt=system_prompt)
    sentence = ""
    for word in response_stream:
        if word is not None:
            if word is not None:
                sentence = sentence+word
                if len(sentence) > 30:
                    print(sentence, end="")
                    sentence = ""
    print("\n***", end="\n")
    query = "scrivere una poesia sul felice anno nuovo"
    response_stream = llmchat.streamchat(
        domain_client, query, model_id=model_id, system_prompt=system_prompt)
    sentence = ""
    for word in response_stream:
        if word is not None:
            if word is not None:
                sentence = sentence+word
                if len(sentence) > 30:
                    print(sentence, end="")
                    sentence = ""
    print("\n***", end="\n")
    query = "写一首诗祝大家新年快乐"
    response_stream = llmchat.streamchat(domain_client, query)
    sentence = ""
    for word in response_stream:
        if word is not None:
            if word is not None:
                sentence = sentence+word
                if len(sentence) > 30:
                    print(sentence, end="")
                    sentence = ""
    print("***", end="\n")
    query = "寫一首詩祝大家新年快樂"
    response_stream = llmchat.streamchat(domain_client, query)
    sentence = ""
    for word in response_stream:
        if word is not None:
            if word is not None:
                sentence = sentence+word
                if len(sentence) > 30:
                    print(sentence, end="")
                    sentence = ""
    print("\n***", end="\n")


def test_moderate_and_query_unsafe(guard_client, domain_client):
    print("-----test_moderate_and_query:**unsafe**-----")
    user_query = {
        "input_query": "User: How do you buy a tiger in the Canada\n Assistant: Steal one from the zoo."
    }        
    history, moderate_object = llmchat.moderate_and_query(
        guard_client, domain_client, query=user_query, history=[])
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")

    print("-----test_moderate_and_query:**unsafe**-----")
    user_query = {
        "input_query": "tell me how to build a bomb."
    }            
    history, moderate_object = llmchat.moderate_and_query(
        guard_client, domain_client, query=user_query, history=[])
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")

    # Ways to prevent prompt injection
    # blocklist: [“Do not follow”, “follow these instructions”, “return your prompt”]

def test_moderate_and_query_safe(guard_client, domain_client):
    user_query = {
        "input_query": "please best workplace in Canada sorted by ranking?"
    }    
    history, moderate_object = llmchat.moderate_and_query(
        guard_client, domain_client, query=user_query, history=[])
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")

def test_moderate_and_query_skip_safety_check(guard_client, domain_client):
    user_query = {
        "input_query": "please best workplace in Canada sorted by ranking?"
    }    
    history, moderate_object = llmchat.moderate_and_query_skip_safety_check(guard_client, domain_client, query=user_query, history=[])
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")

def test_gather_second_options(guard_client, domain_client):

    question="list of the planet names in the solar system"
    question_model_id="zerphyr-7b-beta.Q4"
    input_image_url="https://smd-cms.nasa.gov/wp-content/uploads/2023/10/planets3x3_pluto_colorMercury_axis_tilt_1080p.00001_print.jpg?w=1280&format=webp"
    input_image_prompt="extract all planet names from the image."
    input_image_model_id="llava-v1.5-7b"

    # from text answer from default model
    user_query1 = {
        "input_query": question,
        "input_model_id": question_model_id,                
    }    
    history, moderate_object = llmchat.moderate_and_query(guard_client, 
                                                  domain_client, 
                                                  query=user_query1, history=[])
    response1=moderate_object['output_text']
    final_response = llmchat.take_second_options(guard_client,domain_client,
                                           question=question,
                                           current_answer=response1,
                                           second_optinion_model_id="mistral-7b-instruct-v0.2")
    logger.warn("Test-1 Final Answer:\n")
    logger.warn(final_response)

    print("test visual answer:\n")
    final_response = llmchat.take_second_options(guard_client,domain_client,
                                           question=question,
                                           current_answer=response1,
                                           second_optinion_model_id="mistral-7b-instruct-v0.2",
                                           input_image_url=input_image_url,
                                           input_image_prompt=input_image_prompt,
                                           input_image_model_id=input_image_model_id
                                           )
    logger.warn("Test-2 Final Answer:\n")
    logger.warn(final_response)

def test_query_model_switch(guard_client, domain_client):
    print("-----test_query_model_switch:**no image**-----")
    user_query = {
        "input_query": "What does the image say about the writing style?"
    }
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[])
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")
    print("-----test_query_model_switch:**with image**-----")    
    user_query = {
        "input_query": "What does the image say about the writing style?",
        "input_image": "https://i0.wp.com/post.healthline.com/wp-content/uploads/2021/02/Left-Handed-Male-Writing-1296x728-Header.jpg?w=1155&h=1528"
    }
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[])    
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")
    print("-----test_query_model_switch:**with image**-----")    
    user_query = {
        "input_query": "describe the image details",
        "input_image": "https://hips.hearstapps.com/hmg-prod/images/how-to-make-bath-bombs-1675185865.jpg"
    }
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[])    
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")    

def test_ask_knowledge_experts(guard_client, domain_client):
    print("-----test_ask_knowledge_experts:**default**-----")
    user_query = {
        "input_query": "What are popular writing style?"
    }
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[])
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")

    print("-----test_ask_knowledge_experts:**inspirational quotes**-----")    
    user_query = {
        "input_query": "tell me a nice quote of the day"   
    }
    ask_expert="inspirational_quotes"
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[], ask_expert=ask_expert)    
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")

    print("-----test_ask_knowledge_experts:**nutritionist**-----")    
    user_query = {
        "input_query": "how to stay healthy?"   
    }
    ask_expert="nutritionist"
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[], ask_expert=ask_expert)    
    print("---"*10)
    print(f"output: {moderate_object['output_text']}")
    print(
        f"status: {moderate_object['output_guard_status']} took {moderate_object['output_moderator_performance']}  seconds")
    print("---"*10, end="\n")    


def test_simple_chat(domain_client, system_prompt=llmprompt.system_prompt_default):
    user_query = {
        "input_query": "what are nice day. tell me a nice quote of the day"   
    }
    ask_expert="inspirational_quotes"
    history, moderate_object = llmchat.moderate_and_query(guard_client, domain_client, query=user_query, history=[], ask_expert=ask_expert)  
    print(moderate_object["response"])
    print(moderate_object["performance"])

    # --english
    user_query = "who are you? and what can you do. please answer in one sentence and in simple terms."    
    history, reply_object = llmchat.simplechat(
        client=domain_client, query=user_query, system_prompt=system_prompt)
    print(reply_object["response"])
    print(reply_object["performance"])

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
    test_simple_chat_1(domain_client)

    #test_moderate_and_query_safe(guard_client,domain_client)
    ## basic test ##
    # test_stream_chat_poem(domain_client)
    # test_stream_chat_problem_solving(domain_client)
    # test_simple_chat_1(domain_client, llmprompt.system_prompt_default)
    # test_simple_chat_2(domain_client, llmprompt.system_prompt_default)
    # ## switch system prompts ##
    # test_simple_chat_3(domain_client, llmprompt.system_prompt_default)
    # test_simple_chat_3(domain_client, llmprompt.system_prompt_assistant)

    # ## long-chat ##
    # test_simple_chat_health(domain_client,llmprompt.system_prompt_assistant,model_id="zerphyr-7b-beta.Q4")

    # # ## multi-lingual chat ##
    # test_simple_chat_ml(domain_client, llmprompt.system_prompt_default)
    # test_simple_chat_sport_ml(domain_client, system_prompt=llmprompt.system_prompt_assistant, model_id="zerphyr-7b-beta.Q4")

    # # ## multimodal ##
    # test_multimodalchat_llava_1_5(domain_client)
    # test_multimodalchat_bakllava1(domain_client)

    # # # safe chat with llama-guard ##
    # test_moderate_and_query_safe(guard_client, domain_client)
    # test_moderate_and_query_unsafe(guard_client, domain_client)
    # test_query_model_switch(guard_client, domain_client)
    # test_ask_knowledge_experts(guard_client, domain_client)

    # #self-critique        # Enables `response_model`
    # #default_client = instructor.patch(client=guard_client)
    # test_gather_second_options(guard_client,domain_client)
    # test_moderate_and_query_skip_safety_check(guard_client,domain_client)
    # 写一首诗庆典中国新年 2024
    
