from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

# from __future__ import annotations
from scipy.special import expit
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
import torch
# from rich import print, pretty, console
# import warnings
# from rich.logging import RichHandler
# import time
# import logging
# from dotenv import dotenv_values
# system_config = dotenv_values("env_config")
# logging.basicConfig(level=logging.DEBUG, format="%(message)s",
#                     datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
# logger = logging.getLogger(__name__)
# pretty.install()
# warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer, util
import functools

# pip install sentence_transformers

# global instance 
g_classify_topic_model = None
g_classify_topic_tokenizer = None

# def classify_text_emotion(query: str, model_id: str = "SamLowe/roberta-base-go_emotions", top_k=3, cache_dir: str = "./models", low_cpu_mem_usage=True, use_safetensors=True):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_id, cache_dir=cache_dir, low_cpu_mem_usage=low_cpu_mem_usage, use_safetensors=use_safetensors)
#     classifier = pipeline("text-classification", model=model,
#                           tokenizer=tokenizer, torch_dtype=torch_dtype, device=device)
#     predictions = classifier([query], top_k=top_k)
#     # print(predictions[0])
#     return predictions[0]


# def hallucination_detection(text1: str, text2: str, model_id: str = 'vectara/hallucination_evaluation_model'):
#     """
#     The model outputs a probabilitity from 0 to 1, 0 being a hallucination and 1 being factually consistent. The predictions can be thresholded at 0.5 to predict whether a document is consistent with its source.
#     """
#     model = CrossEncoder('vectara/hallucination_evaluation_model')
#     scores = model.predict([[text1, text2]])
#     return scores

# def classify_topic_model(query: str, model_id: str = "scroobiustrip/topic-model-v3", cache_dir: str = "./models"):
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#     tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
#     model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir,
#                                                                low_cpu_mem_usage=True, use_safetensors=True)
#     pipe = pipeline("text-classification", model=model,
#                     tokenizer=tokenizer, torch_dtype=torch_dtype, device=device)
#     result = pipe(query)
#     # print(result[0]["label"],result[0]["score"])
#     print("classify_topic_model:", result, end="\n")
#     return result[0]

def classify_intent(query: str, model_id: str = "Falconsai/intent_classification", cache_dir: str = "resources/models/classify"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir,
                                                               low_cpu_mem_usage=True, use_safetensors=True)
    pipe = pipeline("text-classification", model=model,
                    tokenizer=tokenizer, torch_dtype=torch.float16, device=device)
    result = pipe(query)
    print("classify_intent:", result, end="\n")
    return result[0]

def classify_single_topic(query: str, model_id: str = "cardiffnlp/tweet-topic-21-multi", cache_dir: str = "resources/models/classify"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir,
                                                               low_cpu_mem_usage=True, use_safetensors=True)
    class_mapping = model.config.id2label
    pipe = pipeline("text-classification", model=model,
                    tokenizer=tokenizer, torch_dtype=torch_dtype, device=device)
    result = pipe(query)
    # print(result[0]["label"],result[0]["score"])
    print("classify_topic_class:", result, end="\n")
    return result[0]


def classify_multiple_topics(query: str, model_id: str = "cardiffnlp/tweet-topic-21-multi", cache_dir: str = "resources/models/classify", low_cpu_mem_usage: bool = True, use_safetensors: bool = True):
    global g_classify_topic_model
    global g_classify_topic_tokenizer
    g_classify_topic_tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir) if g_classify_topic_tokenizer is None else g_classify_topic_tokenizer
    g_classify_topic_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, cache_dir=cache_dir, low_cpu_mem_usage=low_cpu_mem_usage, use_safetensors=use_safetensors) if g_classify_topic_model is None else g_classify_topic_model
    class_mapping = g_classify_topic_model.config.id2label
    tokens = g_classify_topic_tokenizer(query, return_tensors='pt')
    output = g_classify_topic_model(**tokens)
    scores = output[0][0].detach().numpy()
    scores = expit(scores)
    predictions = (scores >= 0.5) * 1
    # Map to classes
    topics = []
    for i in range(len(predictions)):
        if predictions[i]:
            topics.append(class_mapping[i])
    return topics


# def test_classify_topic_model():
#     t0 = time.perf_counter()
#     query = "It is great to see athletes promoting awareness for climate change."
#     result = classify_topic_model(query)
#     print("test_classify_topic_model:",
#           result['label'], result['score'], end="\n")
#     print("test_classify_topic_model tooked ", time.perf_counter()-t0)


# def test_classify_single_topic():
#     t0 = time.perf_counter()
#     query = "It is great to see athletes promoting awareness for climate change."
#     result = classify_single_topic(query)
#     print("test_classify_single_topic:",
#           result['label'], result['score'], end="\n")
#     print("test_classify_single_topic tooked ", time.perf_counter()-t0)


# def test_classify_multiple_topics(query):
#     t0 = time.perf_counter()
#     topics = classify_multiple_topics(query)
#     print("test_classify_multiple_topics:", topics, end="\n")
#     print("test_classify_multiple_topics tooked ", time.perf_counter()-t0)


# def test_hallucination_detection():
#     scores = hallucination_detection(
#         [["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],])
#     print(scores)

# def test_text_emotion_detection():
#     query = "I am not having a great day"
#     scores = classify_text_emotion(query)
#     print(scores)

domain_models_map = {
    "default": "zerphyr-7b-beta.Q4",
    "general": "mistral-7b-instruct-v0.2",
    "multimodal1": "llava-1-5",
    "multimodal2": "bakllava1",
    "mixtral": "mixtral-8x7b-instruct-v0.1",
    "medicine": "medicine-llm-13b.Q3",
    "law": "law-llm-13b.Q3",
    "finance": "finance-llm.Q4",
    "llamaguard": "llamaguard",
    "safe": "zerphyr-7b-beta.Q4"
}

class_topic_map = {
    "arts_&_culture": ['default'],
    "business_&_entrepreneurs": ['finance'],
    "celebrity_&_pop_culture": ['default'],
    "diaries_&_daily_life": ['default'],
    "family": ['default'],
    "fashion_&_style": ['default'],
    "film_tv_&_video": ['multimodal1'],
    "fitness_&_health": ['medicine'],
    "food_&_dining": ['default'],
    "gaming": ['default'],
    "learning_&_educational": ['default'],
    "news_&_social_concern": ['default'],
    "other_hobbies": ['default'],
    "relationships": ['default'],
    "science_&_technology": ['default'],
    "sports": ['default'],
    "travel_&_adventure": ['default'],
    "youth_&_student_life": ['default']
}

@functools.lru_cache
def text_to_domain_model_mapping(text: str):
    logger.info(f"text_to_domain_model:\n[{text}]")
    topics = classify_multiple_topics(query=text)
    logger.info(f"test_classify_multiple_topics: {topics}")
    model_id = domain_models_map["default"]
    id2domain = ""
    if len(topics)>0: 
        if topics[0] in class_topic_map.keys():
            id2domain = class_topic_map[f"{topics[0]}"]
            logger.info(f"found topic:{id2domain}")
            if id2domain[0] in domain_models_map.keys():
                model_id = domain_models_map[id2domain[0]]
                logger.info(f"found model: {model_id}")
    logger.info(f"model_id: {model_id}")
    return model_id,id2domain

# global
g_mlmodel=None

@functools.lru_cache
def get_compare_command_model(model_id_or_path:str="distiluse-base-multilingual-cased-v1"):
    global g_mlmodel    
    g_mlmodel = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    return g_mlmodel

@functools.lru_cache
def cosine_command(input_command:str="请", target_command:str="please"):
    global g_mlmodel
    logger.info(f"cosine_command: {input_command} vs {target_command}")
    g_mlmodel = get_compare_command_model() if g_mlmodel is None else g_mlmodel
    emb1 = g_mlmodel.encode(input_command)
    emb2 = g_mlmodel.encode(target_command)
    cos_sim = util.cos_sim(emb1, emb2)
    cos_sim_ratio = float(cos_sim[0][0])
    logger.info(f"cosine_command: {input_command} | {target_command} = {cos_sim_ratio:.2f}")
    return cos_sim_ratio

##@functools.lru_cache
def isvalid_command(input_command:str="请", target_command:str="please wait", acceptable_ratio:float=0.8):
    if input_command is None or len(input_command)==0:
        return
    if target_command is None or len(target_command)==0:
        return    
    input_command=str(input_command).strip()
    target_command=str(target_command).strip()
    result=cosine_command(input_command=input_command, target_command=target_command)
    if result>=acceptable_ratio:
        return True
    else:
        return False

##@functools.lru_cache
def match_talkie_codes(input_command:str="请",talkie_codes_list:list=["copy","affirmative","roger","over","out","roger that","please","thank you","thanks"],
                       acceptable_ratio:int=0.8):
    founds=[]
    for action in talkie_codes_list:
        if isvalid_command(input_command,action,acceptable_ratio=acceptable_ratio):
            founds.append(action)
    logger.debug("result commands: {founds}")
    return founds

if __name__ == "__main__":
    sameCommand=cosine_command(input_command="请", target_command="please wait")
    if sameCommand>0.6:
        print("SAME")

    sameCommand=cosine_command(input_command="謝謝", target_command="thanks")
    if sameCommand>0.6:
        print("SAME")
        
    sameCommand=cosine_command(input_command="gracias", target_command="thanks")
    if sameCommand>0.6:
        print("SAME")

    sameCommand=cosine_command(input_command="ok ryan", target_command="ryan")
    if sameCommand>0.6:
        print("SAME")            

    #test_text_emotion_detection()
    # test_hallucination_detection()
    # test_classify_topic_model()
    # test_classify_single_topic()
    # sentences = [
    #     "A man walks into a bar and buys a drink",
    #     "A boy is jumping on skateboard in the middle of a red bridge.",
    #     "It is great to see athletes promoting awareness for climate change.",
    #     "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 representing 7.7 % of net sales.",
    #     "Bids or offers include at least 1,000 shares and the value of the shares must correspond to at least EUR 4,000.",
    #     "Raute reported a loss per share of EUR 0.86 for the first half of 2009 , against EPS of EUR 0.74 in the corresponding period of 2008.",
    #     "hello world"
    # ]
    # for text in sentences:
    #     topics=classify_multiple_topics(query=text)
    #     print("test_classify_multiple_topics:",topics, end="\n")
    #    
    # text = "Replace with IP address and port of your llama-cpp-python server"
    # model_id = text_to_domain_model_mapping(text)
    # print("model_id:", model_id)
