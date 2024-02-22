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
import src.shared.solar.llmchat as llmchat
import src.shared.solar.llmcognitive as llmcognitive 
import src.shared.solar.llmprompt as llmprompt 
import src.shared.solar.llmcontentguard as llmcontentguard
import src.shared.solar.llmfunctionary as functionary
import instructor

def test_self_critique(client: OpenAI):
    question = "Question: Medications similar to albuterol?"
    context = """
        I am sorry, but as an AI assistant, I cannot provide any medical advice or prescribe medicines. It is important that you consult with a healthcare professional for any questions related to your health. However, I can help you find information about other options that may be similar to albuterol.

        Albuterol is a beta-agonist medication used to treat bronchospasm and asthma. If you are looking for alternatives, there are several medications that work in a similar way. Some examples include levalbuterol, salbutamol, terbutaline, and formoterol. These medications can be prescribed by your doctor if needed.

        It's important to note that while these medications may have similar properties to albuterol, they may not necessarily be a suitable replacement in all cases. Your doctor will need to consider your medical history, symptoms, and other factors before determining the best course of treatment for you.


        Sure! Albuterol is a type of medication called a bronchodilator. Bronchodilators help open up the airways in your lungs by relaxing the muscles that surround them, making it easier to breathe. 

        Here are some other commonly used bronchodilators that work similarly to albuterol:

        1. Salmeterol (Serevent): This medication is also a long-acting bronchodilator and helps relieve shortness of breath caused by asthma or chronic obstructive pulmonary disease (COPD).
        2. Formoterol (Foradil, Perforom): Like albuterol, formoterol is a quick-acting bronchodilator that can be taken as an inhaler or through a nebulizer. It helps relieve symptoms of asthma and COPD.
        3. Leukotriene modifiers: These medications work by reducing the production of leukotrienes, which are inflammatory molecules that narrow the airways and make breathing difficult. Popular examples include montelukast (Singulair) and zafirlukast (Accolate).
        4. Methacholine: While

        """
    qa = llmcognitive.self_critique(client=client, question=question,
                       context=context, model_id="functionary")
    print(qa)


def test_extract_object(client: OpenAI):
    message = "Jason is 25 years old live in city of toronto"
    response_object = functionary.extract_response_object(client=client,
                                              message=message, object_model=functionary.UserDetail)
    print("schema object:", response_object, end="\n")


def test_classify_topic(client: OpenAI):
    query = "Hello there I'm a Nigerian prince and I want to give you money"
    query = "where should I invest my money in 2024"
    prediction = functionary.classify_text_topic(client, query)
    # assert prediction.class_label == Labels.SPAM
    print(prediction.topic_label)


if __name__ == "__main__":
    default_client = openai.OpenAI(
        api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # can be anything
        # NOTE: Replace with IP address and port of your llama-cpp-python server
        base_url="http://192.168.0.29:8004/v1"
    )
    domain_client = openai.OpenAI(
        api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # can be anything
        base_url = "http://192.168.0.29:8004/v1" # NOTE: Replace with IP address and port of your llama-cpp-python server
    )

    # Enables `response_model`
    default_client = instructor.patch(client=default_client)
    test_self_critique(client=default_client)
    # ## extract text to schema object
    test_extract_object(client=default_client)
    # test_classify_topic(client=default_client)
    topic = functionary.classify_text_topic(
        default_client, data="Replace with IP address and port of your llama-cpp-python server")
    print("Topics:",topic.topic_label)

