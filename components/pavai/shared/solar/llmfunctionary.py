import enum
import os
import asyncio
import openai
import json
from openai import OpenAI
from pydantic import BaseModel
import instructor
from typing_extensions import Annotated
from pydantic import BaseModel, BeforeValidator
from .llmprompt import guard_system_prompt, system_prompt_assistant
#!pip install -U instructor

# Function Calling With OpenAI Python Client

# Enables `response_model`
# default_client = instructor.patch(client=default_client)

def get_current_weather(location: str):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
    return json.dumps({"location": location, "temperature": "unknown due error"})


def conversation_function_call_(client: OpenAI):
    # Step 1: send the conversation and available functions to the model
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="functionary",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # print(response_message,"\n**")

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        # extend conversation with assistant's reply
        messages.append(response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response

        print(messages, "**\n**")
        second_response = client.chat.completions.create(
            model="functionary",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response

# use function calling
# print(run_conversation(default_client))

class UserDetail(BaseModel):
    name: str
    age: int
    city: str


def extract_response_object(client: OpenAI, message: str, object_model: BaseModel):
    response_object = client.chat.completions.create(
        model="functionary",
        response_model=object_model,
        messages=[
            {"role": "user", "content": f"Extract {message}"},
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"}
    )
    return response_object


class TopicLabels(str, enum.Enum):
    """Enumeration for single-label text classification."""
    SPAM = "spam"
    LAW = "law"
    FINANCE = "finance"
    INVESTMENT = "investiment"
    MEDICINE = "medicine"
    DEFAULT = "default"


class TopicPrediction(BaseModel):
    """
    Class for a single class label prediction.
    """
    topic_label: TopicLabels


class QuestionAnswer(BaseModel):
    question: str
    answer: str


def self_critique(client: OpenAI, question: str,
                  context: str, model_id: str = "functionary", system_prompt: str = guard_system_prompt,
                  history: list = [], max_tokens: int = 256, seed: int = 222, temperature: float = 1, top_p: float = 1,
                  user: str = "user", stop: list = ["</s>"], response_format: dict = {"type": "text"}) -> str:
    """self critique response text"""
    # Enables `response_model`
    default_client = instructor.patch(client=client)

    answer: QuestionAnswer = client.chat.completions.create(
        model="functionary", response_model=QuestionAnswer,
        messages=[
            {
                "role": "system",
                "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
            },
            {
                "role": "user",
                "content": f"using the context: {context}\n\nAnswer the following question: {question}",
            },
        ],
    )
    return answer


def classify_text_topic(client: OpenAI, data: str, model_id: str = "functionary") -> TopicPrediction:
    """Perform single topic classification on the input text."""
    result = client.chat.completions.create(
        model=model_id,
        response_model=TopicPrediction,
        messages=[{
            "role": "user",
            "content": f"Classify the following text: {data}", },],
        temperature=1,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "text"})
    return result


# def test_self_critique(client: OpenAI):
#     question = "Question: Medications similar to albuterol?"
#     context = """
#         I am sorry, but as an AI assistant, I cannot provide any medical advice or prescribe medicines. It is important that you consult with a healthcare professional for any questions related to your health. However, I can help you find information about other options that may be similar to albuterol.

#         Albuterol is a beta-agonist medication used to treat bronchospasm and asthma. If you are looking for alternatives, there are several medications that work in a similar way. Some examples include levalbuterol, salbutamol, terbutaline, and formoterol. These medications can be prescribed by your doctor if needed.

#         It's important to note that while these medications may have similar properties to albuterol, they may not necessarily be a suitable replacement in all cases. Your doctor will need to consider your medical history, symptoms, and other factors before determining the best course of treatment for you.


#         Sure! Albuterol is a type of medication called a bronchodilator. Bronchodilators help open up the airways in your lungs by relaxing the muscles that surround them, making it easier to breathe. 

#         Here are some other commonly used bronchodilators that work similarly to albuterol:

#         1. Salmeterol (Serevent): This medication is also a long-acting bronchodilator and helps relieve shortness of breath caused by asthma or chronic obstructive pulmonary disease (COPD).
#         2. Formoterol (Foradil, Perforom): Like albuterol, formoterol is a quick-acting bronchodilator that can be taken as an inhaler or through a nebulizer. It helps relieve symptoms of asthma and COPD.
#         3. Leukotriene modifiers: These medications work by reducing the production of leukotrienes, which are inflammatory molecules that narrow the airways and make breathing difficult. Popular examples include montelukast (Singulair) and zafirlukast (Accolate).
#         4. Methacholine: While

#         """
#     qa = self_critique(client=client, question=question,
#                        context=context, model_id="functionary")
#     print(qa)


# def test_extract_object(client: OpenAI):
#     message = "Jason is 25 years old live in city of toronto"
#     response_object = extract_response_object(client=client,
#                                               message=message, object_model=UserDetail)
#     print("schema object:", response_object, end="\n")


# def test_classify_topic(client: OpenAI):
#     query = "Hello there I'm a Nigerian prince and I want to give you money"
#     query = "where should I invest my money in 2024"
#     prediction = classify_text_topic(client, query)
#     # assert prediction.class_label == Labels.SPAM
#     print(prediction.topic_label)


# if __name__ == "__main__":
#     default_client = openai.OpenAI(
#         api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # can be anything
#         # NOTE: Replace with IP address and port of your llama-cpp-python server
#         base_url="http://192.168.0.29:8004/v1"
#     )
#     domain_client = openai.OpenAI(
#         api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # can be anything
#         base_url = "http://192.168.0.29:8004/v1" # NOTE: Replace with IP address and port of your llama-cpp-python server
#     )

#     # Enables `response_model`
#     default_client = instructor.patch(client=default_client)
#     # test_self_critique(client=default_client)
#     # ## extract text to schema object
#     # test_extract_object(client=default_client)
#     # test_classify_topic(client=default_client)
#     topic = classify_text_topic(
#         default_client, data="Replace with IP address and port of your llama-cpp-python server")
#     print(topic)
