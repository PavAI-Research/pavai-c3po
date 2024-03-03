from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

#from abc import ABC, abstractmethod
#from openai import OpenAI
#import openai
from urllib.request import urlopen
##import json
#import time

from pavai.llmone.solar.llmchat import moderate_and_query, take_second_options, moderate_and_query_skip_safety_check
#from pavai.llmone.datasecurity import DataSecurityEngine
from pavai.llmone.solar.llmclassify import text_to_domain_model_mapping,domain_models_map
from pavai.llmone.solar.llmfunctionary import self_critique
from pavai.llmone.solar.llmcontentguard import safety_check
from pavai.llmone.solar.llmprompt import lookup_expert_system_prompt, system_prompt_assistant, system_prompt_default
import textwrap
import pavai.llmone.solar.llmtype as llmtype

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"


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
