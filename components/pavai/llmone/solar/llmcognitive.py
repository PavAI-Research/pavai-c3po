from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

#from __future__ import annotations
from abc import ABC, abstractmethod
from openai import OpenAI
import openai
from urllib.request import urlopen
#import functools
import json
#from dotenv import dotenv_values
import time
# import warnings
# import logging
# from rich import print, pretty, console
# from rich.logging import RichHandler
# system_config = dotenv_values("env_config")
# logging.basicConfig(level=logging.INFO, format="%(message)s",
#                     datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
# logger = logging.getLogger(__name__)
# warnings.filterwarnings("ignore")
# pretty.install()

from pavai.llmone.solar.llmchat import moderate_and_query, take_second_options, moderate_and_query_skip_safety_check
from pavai.llmone.solar.llmdatasecurity import DataSecurityEngine
from pavai.llmone.solar.llmclassify import text_to_domain_model_mapping,domain_models_map
from pavai.llmone.solar.llmfunctionary import self_critique
from pavai.llmone.solar.llmcontentguard import safety_check
from pavai.llmone.solar.llmprompt import lookup_expert_system_prompt, system_prompt_assistant, system_prompt_default
import textwrap
import pavai.llmone.solar.llmtype as llmtype

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"


# class Action(ABC):
#     """
#     The Command interface declares a method for executing an action.
#     """
#     @abstractmethod
#     def execute(self) -> any:
#         pass


class StartAction(llmtype.Action):
    """
    StartAction implementation
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        self._network = network

    def execute(self) -> any:
        logger.debug(f"StartAction: execute.")
        # ensure dependencies are up-running
        return self._network._memory


class InputAction(llmtype.Action):
    """
    InputAction implement checks on input.
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        self._network = network

    def execute(self) -> any:
        logger.debug(f"InputAction: execute.")
        self._network._memory["input_activity"] = []
        self._network._memory["input_activity"].append("InputAction: started")
        # Data Security Check
        if self._network._skip_data_security_check:
            logger.info(f"InputAction: skip data security check.")
        else:
            if self._network._llm_data_securiy is None:
                self._network._llm_data_securiy = DataSecurityEngine()
            if "input_query" in self._network._input_data.keys():
                self._network._memory["input_activity"].append(
                    "data security check: started")
                input_text = self._network._input_data["input_query"]
                logger.info(f"InputAction: _llm_data_securiy.check_text_pii")
                pii_results = self._network._llm_data_securiy.check_text_pii(
                    input_text)
                if pii_results:
                    logger.warn("PII detected in input text")
                    logger.warn(pii_results)
                    self._network._input_data["input_warning"] = "PII detected in input text"
                # @TODO apply anonymization
        self._network._memory["input_activity"].append("InputAction: finished")
        return self._network._memory


class OutputAction(llmtype.Action):
    """
    OutputAction implement checks on output.
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        self._network = network

    def execute(self) -> any:
        logger.debug(f"OutputAction: executed")
        self._network._memory["output_activity"] = []
        self._network._memory["output_activity"].append(
            "OutputAction: started")
        # Data Security Check
        if self._network._skip_data_security_check:
            logger.info(f"InputAction: skip data security check.")
        else:
            if self._network._llm_data_securiy is None:
                self._network._llm_data_securiy = DataSecurityEngine()
            if "output_text" in self._network._output_data.keys():
                self._network._memory["output_activity"].append(
                    "data security check: started")
                output_text = self._network._output_data["output_text"]
                logger.info(f"OutputAction: _llm_data_securiy.check_text_pii")
                pii_results = self._network._llm_data_securiy.check_text_pii(
                    output_text)
                if pii_results:
                    logger.warn("PII detected in output text")
                    logger.warn(pii_results)
                    self._network._output_data["output_warning"] = "PII detected in output text"
                # @TODO apply anonymization
        self._network._memory["output_activity"].append(
            "OutputAction: finished")
        return self._network._memory


class RoutingAction(llmtype.Action):
    """
    RoutingAction forward request to intent domain model with the topic expertise 
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        self._network = network

    def execute(self) -> any:
        logger.debug(f"RoutingAction: execute.")
        self._network._memory["routing_activity"] = []
        self._network._memory["routing_activity"].append(
            "RoutingAction: started")
        input_text = self._network._input_data["input_query"]
        # default
        input_type = "chat"
        id2domain = "default"
        # override default model
        if "input_type" in self._network._input_data.keys():
            input_type = self._network._input_data["input_type"]
        if "input_model_id" in self._network._input_data.keys():
            model_id = self._network._input_data["input_model_id"]
        else:
            try:
                model_id = domain_models_map["default"]
                # reduce text size to fit classifier model tensor size 512
                ## @TODO find better approach
                classify_model_tensor_size=512
                if (input_text)>classify_model_tensor_size:                
                    input_text_short=textwrap.shorten(input_text, classify_model_tensor_size)
                model_id, id2domain = text_to_domain_model_mapping(input_text_short)
            except:
                # set a default value
                id2domain="learning_&_educational" 
        self._network._memory["input_routing_text"] = input_text
        self._network._memory["input_routing_type"] = input_type
        self._network._memory["input_routing_domain"] = id2domain
        self._network._memory["input_routing_model_id"] = model_id
        # set query expert
        ask_expert = "default"
        if "ask_expert" in self._network._input_data.keys():
            ask_expert = self._network._input_data["ask_expert"]
            self._network._memory["ask_expert"] = ask_expert
        logger.info(
            f"RoutingAction: ask expert: [{ask_expert}] and model_id: [{model_id}]")
        self._network._memory["routing_activity"].append(
            "RoutingAction: finished")
        return self._network._memory


class FinishAction(llmtype.Action):
    """
    FinishAction implement checks on input.
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        self._network = network

    def execute(self) -> any:
        logger.info(f"FinishAction: executed")
        logger.debug(f"history: {self._network._history}")
        # logger.info(f"free up resources: input, output, history and memory")
        # self._network._history = []
        # self._input_data = {}
        # self._output_data = {}
        # self._memory = {"type": "memory", "activity": [], "history": []}
        return self._network._memory


class LLMReceiver:
    """
    The LLMReceiver classes contain some important business logic. They know how to
    perform all kinds of operations, associated with carrying out a request. In
    fact, any class may serve as a Receiver.
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        self._network = network,

    def do_input_intent_classification(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on classify input intent )")

    def do_input_voice_emotion_classification(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on classify input intent )")

    def do_input_text_emotion_classification(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on classify input intent )")

    def do_input_enrichment(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on input enrichment (RAG) ")

    def do_chat(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on chat.)")

    def do_code(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on code.)")

    def do_completion(self) -> None:
        logger.debug(f"LLMCommandReceiver: working on completion )")

    def do_output_self_critique(self) -> None:
        logger.debug(f"LLMCommandReceiver: Working on self-critique ")


class LLMReceiverLocal(LLMReceiver):
    """
    The Receiver classes contain some important business logic. They know how to
    perform all kinds of operations, associated with carrying out a request. In
    fact, any class may serve as a Receiver.
    """

    def do_input_intent_classification(self) -> None:
        logger.info(f"LLMReceiverLocal: working on classify input intent.)")

    def do_input_voice_emotion_classification(self) -> None:
        logger.info(f"LLMReceiverLocal: working on classify input intent.)")

    def do_input_text_emotion_classification(self) -> None:
        logger.info(f"LLMReceiverLocal: working on classify input intent.)")

    def do_input_enrichment(self) -> None:
        logger.info(f"LLMReceiverLocal: working on input enrichment (RAG).)")

    def do_chat(self) -> None:
        logger.info(f"LLMReceiverLocal: working on chat.)")

    def do_code(self) -> None:
        logger.info(f"LLMReceiverLocal: working on code.)")

    def do_completion(self) -> None:
        logger.info(f"LLMReceiverLocal: working on completion.)")

    def do_output_self_critique(self) -> None:
        logger.info(f"LLMReceiverLocal: Working on self-critique.)")


class LLMReceiverAPI(LLMReceiver):
    """
    The Receiver classes contain some important business logic. They know how to
    perform all kinds of operations, associated with carrying out a request. In
    fact, any class may serve as a Receiver.
    """

    def __init__(self, network: llmtype.LLMSolarNetwork) -> None:
        """
        LLMReceiverAPI
        """
        self._network = network

    def do_input_intent_classification(self) -> None:
        logger.debug(f"LLMReceiverAPI: do_input_intent_classification. - TODO")

    def do_input_voice_emotion_classification(self) -> None:
        logger.debug(
            f"LLMReceiverAPI: do_input_voice_emotion_classification. - TODO")

    def do_input_text_emotion_classification(self) -> None:
        logger.debug(
            f"LLMReceiverAPI: do_input_text_emotion_classification - TODO")

    def do_input_enrichment(self) -> None:
        logger.debug(f"LLMReceiverAPI: do_input_enrichment. - TODO")

    def do_chat(self) -> None:
        logger.debug(f"LLMReceiverAPI: do_chat() started")
        input_type = self._network._memory["input_routing_type"]
        if "chat" not in input_type:
            logger.info(f"LLMReceiverAPI: do_chat() skipped due {input_type}")
            return
        self._network._memory["do_chat_activity"] = []
        self._network._memory["do_chat_activity"].append(
            "LLMReceiverAPI.do_chat(): started")
        logger.info(
            f"LLMReceiverAPI.do_chat_activity: running moderation and query")
        ask_expert = None
        if "ask_expert" in self._network._input_data.keys():
            ask_expert = self._network._memory["ask_expert"]
        # skip content safety check if enable
        if self._network._skip_content_safety_check:
            history, moderate_object = moderate_and_query_skip_safety_check(self._network._llm_default,
                                                                            self._network._llm_domain,
                                                                            query=self._network._input_data,
                                                                            history=self._network._history,
                                                                            ask_expert=ask_expert)
        else:
            history, moderate_object = moderate_and_query(self._network._llm_default,
                                                          self._network._llm_domain,
                                                          query=self._network._input_data,
                                                          history=self._network._history,
                                                          ask_expert=ask_expert)
        # domain_client = self._network._llm_domain_client
        self._network._history = history
        self._network._output_data = moderate_object
        logger.info("---"*10)
        # logger.info(f"output: {moderate_object['output_text']}")
        logger.info(
            f"content safety status: **[{moderate_object['output_guard_status']}]** took {moderate_object['output_moderator_performance']}  seconds")
        logger.info("---\n")
        # self._network._output_data["response"]=response_text
        self._network._memory["output_data"] = moderate_object
        # save a copy in memory
        self._network._memory["history"].append(
            (moderate_object['input_text'], moderate_object['output_text']))
        logger.debug(f"LLMReceiverAPI: do_chat() finished")
        self._network._memory["do_chat_activity"].append(
            "LLMReceiverAPI.do_chat(): finished")

    def do_code(self) -> None:
        # logger.info(f"LLMReceiverAPI: do_code()")
        input_type = self._network._memory["input_routing_type"]
        if "code" not in input_type:
            logger.info(f"LLMReceiverAPI: do_code() skipped due {input_type}")
            return

    def do_completion(self) -> None:
        # logger.debug(f"LLMReceiverAPI: do_completion() started")
        input_type = self._network._memory["input_routing_type"]
        if "completion" not in input_type:
            logger.info(
                f"LLMReceiverAPI: do_completion() skipped : {input_type}")
            return

    def do_output_self_critique(self) -> None:
        if self._network._skip_self_critique_check:
            logger.info(f"LLMReceiverAPI: do_output_self_critique() - skipped")
            return
        if "input_second_option_source" in self._network._input_data.keys():
            logger.info(f"LLMReceiverAPI: do_output_self_critique() started")
            input_second_option_source = self._network._input_data["input_second_option_source"]
            if "llm" == input_second_option_source:
                if "input_second_option_model_id" in self._network._input_data.keys():
                    logger.info("enable second option check")
                    input_second_option_model_id = self._network._input_data[
                        "input_second_option_model_id"]
                    if "output_text" not in self._network._output_data.keys():
                        logger.warn(
                            f"LLMReceiverAPI: do_output_self_critique(). Missing Previous answer.")
                        return
                    final_answer = take_second_options(guard_client=self._network._llm_default,
                                                       domain_client=self._network._llm_domain,
                                                       second_optinion_model_id=input_second_option_model_id,
                                                       question=self._network._input_data["input_query"],
                                                       current_answer=self._network._output_data["output_text"])
                    self._network._output_data["output_text"] = final_answer
                    logger.info(
                        f"LLMReceiverAPI: do_output_self_critique() processed")
                    self._network._memory["output_data"] = self._network._output_data
        else:
            logger.debug(
                f"LLMReceiverAPI: do_output_self_critique() - skipped")
        logger.debug(f"LLMReceiverAPI: do_output_self_critique() - finished")


class ThinkAction(llmtype.Action):
    """
    handle thinking and actions to take 
    """

    def __init__(self, network: llmtype.LLMSolarNetwork, receiver: LLMReceiver) -> None:
        """
        ThinkAction accept one or several receiver objects along with any context data via the constructor.
        """
        self._network = network
        self._receiver = receiver

    def execute(self) -> None:
        """
        delegate to any methods of a receiver.
        """
        logger.debug("ThinkAction: execute start")
        self._receiver.do_input_intent_classification()
        self._receiver.do_input_voice_emotion_classification()
        self._receiver.do_input_text_emotion_classification()
        self._receiver.do_input_enrichment()
        self._receiver.do_chat()
        self._receiver.do_code()
        self._receiver.do_completion()
        self._receiver.do_output_self_critique()
        logger.debug("ThinkAction: execute finish")


# class LLMSolarNetwork:
#     """
#     The Solar Network (SN) is a data network that makes use of several technologies (i.e. machine learning, knowledge representation, computer network, 
#     network management) to solve problems.
#     """
#     # actions
#     _on_start = None
#     _on_input = None
#     _on_routing = None
#     _on_thinking = None
#     _on_storage = None
#     _on_output = None
#     _on_finish = None

#     # messy information keep in the brain
#     # such as short-term and long-term memory
#     _memory = {"type": "memory", "activity": [], "history": []}
#     _input_data = {}
#     _output_data = {}
#     _history = []

#     # LLM clients
#     _llm_default = None
#     _llm_domain = None

#     # Data Security
#     _llm_data_securiy = None
#     """
#     Initialize commands.
#     """

#     def __init__(self, default_client: OpenAI = None,
#                  domain_client: OpenAI = None,
#                  data_security: DataSecurityEngine = None,
#                  skip_content_safety_check: bool = True,
#                  skip_data_security_check: bool = False,
#                  skip_self_critique_check: bool = False) -> None:
#         """constructor"""
#         self._llm_default = default_client
#         self._llm_domain = domain_client
#         self._llm_data_securiy = data_security
#         self._skip_content_safety_check = skip_content_safety_check
#         self._skip_data_security_check = skip_data_security_check
#         self._skip_self_critique_check = skip_self_critique_check

#     def set_on_start(self, command: Action):
#         self._on_start = command

#     def set_on_input(self, command: Action):
#         self._on_input = command

#     def set_on_routing(self, command: Action):
#         self._on_routing = command

#     def set_on_thinking(self, command: Action):
#         self._on_thinking = command

#     def set_on_output(self, command: Action):
#         self._on_output = command

#     def set_on_finish(self, command: Action):
#         self._on_finish = command

#     def do_actions(self, input_data: dict) -> None:
#         """
#         Invoker passes a request to a receiver indirectly, by executing a command.
#         """
#         self._input_data = input_data
#         self._memory["input_data"] = input_data

#         logger.debug(
#             "LLMSolarNetwork: Does anybody want something done before I begin?")
#         if isinstance(self._on_start, Action):
#             out = self._on_start.execute()
#             if out is not None:
#                 self._memory = out

#         logger.debug("LLMSolarNetwork: what input")
#         if isinstance(self._on_input, Action):
#             out = self._on_input.execute()
#             if out is not None:
#                 self._memory = out

#         logger.debug("LLMSolarNetwork: routing request to domain experts")
#         if isinstance(self._on_routing, Action):
#             out = self._on_routing.execute()
#             if out is not None:
#                 self._memory = out

#         logger.debug("LLMSolarNetwork: thinking")
#         if isinstance(self._on_thinking, Action):
#             out = self._on_thinking.execute()
#             if out is not None:
#                 self._memory = out

#         logger.debug(
#             "LLMSolarNetwork: now...doing something really important...")

#         logger.debug("LLMSolarNetwork: prepare output")
#         if isinstance(self._on_output, Action):
#             out = self._on_output.execute()
#             if out is not None:
#                 self._memory = out

#         logger.debug(
#             "LLMSolarNetwork: Does anybody want something done after I finish?")
#         if isinstance(self._on_finish, Action):
#             out = self._on_finish.execute()
#             if out is not None:
#                 self._memory = out

#     def do_cleanup(self):
#         logger.info(f"free up resources: input, output, history and memory")
#         self._history = []
#         self._input_data = {}
#         self._output_data = {}
#         self._memory = {"type": "memory", "activity": [], "history": []}


# class LLMSolarSetting(object):
#     def __init__(self, url):
#         if "https" in url or "http" in url:
#             with urlopen(url) as content:
#                 self.config = json.load(content)
#         else:
#             self.config = json.load(open(url))

#     def __call__(self, value):
#         return self.config[value]


# class LLMSolarClient():
#     """settings"""
#     _skip_content_safety_check = True
#     _skip_data_security_check = False
#     _skip_self_critique_check = False
#     """objects"""
#     _cn_invoker = None
#     _datasecurity = None
#     _default_client = None
#     _domain_client = None

#     def __init__(self,
#                  default_url: str = None,
#                  default_api_key: str = "EMPTY",
#                  domain_url: str = None,
#                  domain_api_key: str = "EMPTY",
#                  skip_content_safety_check: bool = True,
#                  skip_data_security_check: bool = False,
#                  skip_self_critique_check: bool = False) -> None:
#         # api client
#         self._default_client = openai.OpenAI(
#             api_key=default_api_key, base_url=default_url)
#         self._domain_client = openai.OpenAI(
#             api_key=domain_api_key, base_url=domain_url)
#         # security engine
#         self._datasecurity = DataSecurityEngine()
#         self._skip_content_safety_check = skip_content_safety_check
#         self._skip_data_security_check = skip_data_security_check
#         self._skip_self_critique_check = skip_self_critique_check
#         # solar network
#         self._cn_invoker = LLMSolarNetwork(default_client=self._default_client,
#                                            domain_client=self._domain_client,
#                                            data_security=self._datasecurity,
#                                            skip_content_safety_check=self._skip_content_safety_check,
#                                            skip_data_security_check=self._skip_data_security_check,
#                                            skip_self_critique_check=self._skip_self_critique_check)
#         self._cn_invoker.set_on_start(StartAction(self._cn_invoker))
#         self._cn_invoker.set_on_input(InputAction(self._cn_invoker))
#         self._cn_invoker.set_on_routing(RoutingAction(self._cn_invoker))
#         self._cn_invoker.set_on_thinking(ThinkAction(
#             network=self._cn_invoker, receiver=LLMReceiverAPI(self._cn_invoker)))
#         self._cn_invoker.set_on_output(OutputAction(self._cn_invoker))
#         self._cn_invoker.set_on_finish(FinishAction(self._cn_invoker))

#     def chat(self, input_data: dict, history: list = [], cleanup: bool = True):
#         logger.info("LLMSolarClient.run_query()")
#         t0 = time.perf_counter()
#         try:
#             self._cn_invoker._history = history
#             self._cn_invoker.do_actions(input_data=input_data)
#             logger.info(
#                 f"\nQuestion:\n[green]{self._cn_invoker._input_data['input_query']}[/green]", extra=dict(markup=True))
#             logger.info(
#                 f"\nAnswer:\n[blue]{self._cn_invoker._output_data['output_text']}[/blue]", extra=dict(markup=True))
#             output = self._cn_invoker._output_data
#             history = self._cn_invoker._history
#             if cleanup:
#                 self._cn_invoker.do_cleanup()
#         except Exception as e:
#             print(e)
#             logger.exception("LLMSolarClient.chat() error.")
#             output={}
#             output["output_text"] = "oops, found unexpected system error."
#         t1 = time.perf_counter() - t0
#         logger.info(f"LLMSolarClient.chat() took {t1:.6f} seconds")
#         return output, history

#     @staticmethod
#     def new_instance(runtime_file: str = "llm_api.json"):
#         llm_settings = LLMSolarSetting(runtime_file)
#         default_api_base = llm_settings.config["default_api_base"]
#         default_api_key = llm_settings.config["default_api_key"]
#         domain_api_base = llm_settings.config["domain_api_base"]
#         domain_api_key = llm_settings.config["domain_api_key"]
#         skip_content_safety_check = llm_settings.config["skip_content_safety_check"]
#         skip_data_security_check = llm_settings.config["skip_data_security_check"]
#         skip_self_critique_check = llm_settings.config["skip_self_critique_check"]
#         client = LLMSolarClient(default_url=default_api_base,
#                                 default_api_key=default_api_key,
#                                 domain_url=domain_api_base,
#                                 domain_api_key=domain_api_key,
#                                 skip_content_safety_check=skip_content_safety_check,
#                                 skip_data_security_check=skip_data_security_check,
#                                 skip_self_critique_check=skip_self_critique_check)
#         return client


# def test_solarnetwork():
#     """
#     The client code parameterize invoker with any commands.
#     """
#     import time
#     t0 = time.perf_counter()
#     default_url = "http://192.168.0.29:8004/v1"
#     domain_url = "http://192.168.0.29:8004/v1"
#     solarclient = LLMSolarClient(
#         default_url=default_url, domain_url=domain_url)
#     # input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
#     # chat message basic one
#     logger.info("chat message basic")
#     user_query = {
#         "input_query": "list of the planet names in the solar system",
#     }
#     output, history = solarclient.chat(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")
#     t2 = time.perf_counter() - t0
#     print("test-1 done in ", f"{t2:.6f}", " seconds")

#     t3 = time.perf_counter()
#     output, history = solarclient.chat(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")
#     t4 = time.perf_counter() - t3
#     print("test-2 done in ", f"{t4:.6f}", " seconds")
#     print("repeat call time saved ", f"{t2-t4:.6f}", " seconds")

#     # chat message using ask-expert
#     logger.info("chat message using ask-expert")
#     user_query = {
#         "input_query": "list of the planet names in the solar system",
#         "ask_expert": "science_explainer",
#     }
#     output, history = solarclient.chat(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")

#     # # chat message with specify model
#     logger.info("chat message use specific model")
#     user_query = {
#         "input_type": "chat",
#         "input_query": "list of the planet names in the solar system",
#         "input_model_id": "zerphyr-7b-beta.Q4",
#     }
#     output, history = solarclient.chat(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")

#     # chat message use different model
#     logger.info("chat message use specific model")
#     user_query = {
#         "input_type": "chat",
#         "input_query": "list of the planet names in the solar system",
#         "input_model_id": "mistral-7b-instruct-v0.2",
#     }
#     output, history = solarclient.chat(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")

#     # chat message include image
#     logger.info("chat message include image")
#     user_query = {
#         "input_type": "chat",
#         "input_query": "describe the image details",
#         "input_image": "https://hips.hearstapps.com/hmg-prod/images/how-to-make-bath-bombs-1675185865.jpg",
#         "input_model_id": "llava-v1.5-7b",
#     }
#     output, history = solarclient.chat(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")
#     t1 = time.perf_counter() - t0
#     print("done in ", f"{t1:.6f}", " seconds")


# def test_solarclient_simple():
#     t0 = time.perf_counter()
#     default_url = "http://192.168.0.29:8004/v1"
#     domain_url = "http://192.168.0.29:8004/v1"
#     solarclient = LLMSolarClient(
#         default_url=default_url, domain_url=domain_url)
#     # input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
#     # chat message basic one
#     logger.info("chat message basic")
#     user_query = {
#         "input_query": "which is taller Toronto CN-Tower or New York City Empire State Building?",
#     }
#     output, history = solarclient.run_query(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")
#     t2 = time.perf_counter() - t0
#     print("test-1 done in ", f"{t2:.6f}", " seconds")


# def test_self_critique():
#     t0 = time.perf_counter()
#     default_url = "http://192.168.0.29:8004/v1"
#     domain_url = "http://192.168.0.29:8004/v1"
#     solarclient = LLMSolarClient(
#         default_url=default_url, domain_url=domain_url)
#     # chat message include self-critique (add additional time)
#     logger.info("chat message include self-critique")
#     user_query = {
#         "input_type": "chat",
#         "input_query": "what is the distance between earth and the moon?",
#         "input_second_option_source": "llm",
#         "input_second_option_model_id": "mistral-7b-instruct-v0.2"
#     }
#     output, history = solarclient.run_query(input_data=user_query)
#     logger.warn(f"Question:\n{output['input_text']}")
#     logger.warn(f"Final Answer:\n{output['output_text']}")
#     t1 = time.perf_counter() - t0
#     print("done in ", f"{t1:.6f}", " seconds")
#     # print(history)


# if __name__ == "__main__":
#     llmsc = LLMSolarClient(
#         default_url="http://192.168.0.29:8004/v1",
#         default_api_key="EMPTY",
#         domain_url="http://192.168.0.29:8004/v1",
#         domain_api_key="EMPTY")
#     # llmsc=LLMSolarClient.new_instance("../llm_api.json")
#     user_query = {
#         "input_query": "which is taller Toronto CN-Tower or New York City Empire State Building?",
#     }
#     llmsc.chat(user_query)
#     # test_solarclient_simple()
#     # test_self_critique()
#     # test_solarnetwork()
