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

import src.shared.solar.llmclassify as classification

def test_classify_topic_model():
    t0 = time.perf_counter()
    query = "It is great to see athletes promoting awareness for climate change."
    result = classification.classify_topic_model(query)
    print("test_classify_topic_model:",
          result['label'], result['score'], end="\n")
    print("test_classify_topic_model tooked ", time.perf_counter()-t0)


def test_classify_single_topic():
    t0 = time.perf_counter()
    query = "It is great to see athletes promoting awareness for climate change."
    result = classification.classify_single_topic(query)
    print("test_classify_single_topic:",
          result['label'], result['score'], end="\n")
    print("test_classify_single_topic tooked ", time.perf_counter()-t0)


def test_classify_multiple_topics(query):
    t0 = time.perf_counter()
    topics = classification.classify_multiple_topics(query)
    print("test_classify_multiple_topics:", topics, end="\n")
    print("test_classify_multiple_topics tooked ", time.perf_counter()-t0)


def test_hallucination_detection():
    scores = classification.hallucination_detection(
        [["A man walks into a bar and buys a drink", "A bloke swigs alcohol at a pub"],])
    print(scores)

def test_text_emotion_detection():
    query = "I am not having a great day"
    scores = classification.classify_text_emotion(query)
    print(scores)


if __name__ == "__main__":
    # test_text_emotion_detection()
    # test_hallucination_detection()
    # test_classify_topic_model()
    # test_classify_single_topic()
    sentences = [
        "A man walks into a bar and buys a drink",
        "A boy is jumping on skateboard in the middle of a red bridge.",
        "It is great to see athletes promoting awareness for climate change.",
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 representing 7.7 % of net sales.",
        "Bids or offers include at least 1,000 shares and the value of the shares must correspond to at least EUR 4,000.",
        "Raute reported a loss per share of EUR 0.86 for the first half of 2009 , against EPS of EUR 0.74 in the corresponding period of 2008.",
        "hello world"
    ]
    for text in sentences:
        topics=classification.classify_multiple_topics(query=text)
        print("test_classify_multiple_topics:",topics, end="\n")
       
    text = "Replace with IP address and port of your llama-cpp-python server"
    model_id = classification.text_to_domain_model_mapping(text)
    print("model_id:", model_id)

    sameCommand=classification.semantic_same_command(input_command="请", target_command="please wait")
    if sameCommand>0.6:
        print("SAME")
    sameCommand=classification.semantic_same_command(input_command="hola", target_command="hello")
    if sameCommand>0.6:
        print("SAME")        
