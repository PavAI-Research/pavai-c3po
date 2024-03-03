from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)
from abc import ABC, abstractmethod
from openai import OpenAI
import openai
from urllib.request import urlopen
import json
import time

class LLMSolarNetwork:
    """
    The Solar Network (SN) is a data network that makes use of several technologies (i.e. machine learning, knowledge representation, computer network, 
    network management) to solve problems.
    """
    # actions
    _on_start = None
    _on_input = None
    _on_routing = None
    _on_thinking = None
    _on_storage = None
    _on_output = None
    _on_finish = None

    # messy information keep in the brain
    # such as short-term and long-term memory
    _memory = {"type": "memory", "activity": [], "history": []}
    _input_data = {}
    _output_data = {}
    _history = []

    # LLM clients
    _llm_default = None
    _llm_domain = None

    # Data Security
    _llm_data_securiy = None
    """
    Initialize commands.
    """

    def __init__(self, default_client: OpenAI = None,
                 domain_client: OpenAI = None,
                 data_security: DataSecurityEngine = None,
                 skip_content_safety_check: bool = True,
                 skip_data_security_check: bool = False,
                 skip_self_critique_check: bool = False) -> None:
        """constructor"""
        self._llm_default = default_client
        self._llm_domain = domain_client
        self._llm_data_securiy = data_security
        self._skip_content_safety_check = skip_content_safety_check
        self._skip_data_security_check = skip_data_security_check
        self._skip_self_critique_check = skip_self_critique_check

    def set_on_start(self, command: Action):
        self._on_start = command

    def set_on_input(self, command: Action):
        self._on_input = command

    def set_on_routing(self, command: Action):
        self._on_routing = command

    def set_on_thinking(self, command: Action):
        self._on_thinking = command

    def set_on_output(self, command: Action):
        self._on_output = command

    def set_on_finish(self, command: Action):
        self._on_finish = command

    def do_actions(self, input_data: dict) -> None:
        """
        Invoker passes a request to a receiver indirectly, by executing a command.
        """
        self._input_data = input_data
        self._memory["input_data"] = input_data

        logger.debug(
            "LLMSolarNetwork: Does anybody want something done before I begin?")
        if isinstance(self._on_start, Action):
            out = self._on_start.execute()
            if out is not None:
                self._memory = out

        logger.debug("LLMSolarNetwork: what input")
        if isinstance(self._on_input, Action):
            out = self._on_input.execute()
            if out is not None:
                self._memory = out

        logger.debug("LLMSolarNetwork: routing request to domain experts")
        if isinstance(self._on_routing, Action):
            out = self._on_routing.execute()
            if out is not None:
                self._memory = out

        logger.debug("LLMSolarNetwork: thinking")
        if isinstance(self._on_thinking, Action):
            out = self._on_thinking.execute()
            if out is not None:
                self._memory = out

        logger.debug(
            "LLMSolarNetwork: now...doing something really important...")

        logger.debug("LLMSolarNetwork: prepare output")
        if isinstance(self._on_output, Action):
            out = self._on_output.execute()
            if out is not None:
                self._memory = out

        logger.debug(
            "LLMSolarNetwork: Does anybody want something done after I finish?")
        if isinstance(self._on_finish, Action):
            out = self._on_finish.execute()
            if out is not None:
                self._memory = out

    def do_cleanup(self):
        logger.info(f"free up resources: input, output, history and memory")
        self._history = []
        self._input_data = {}
        self._output_data = {}
        self._memory = {"type": "memory", "activity": [], "history": []}

class LLMSolarSetting(object):
    def __init__(self, url):
        if "https" in url or "http" in url:
            with urlopen(url) as content:
                self.config = json.load(content)
        else:
            self.config = json.load(open(url))

    def __call__(self, value):
        return self.config[value]

class LLMSolarClient():
    """settings"""
    _skip_content_safety_check = True
    _skip_data_security_check = False
    _skip_self_critique_check = False
    """objects"""
    _cn_invoker = None
    _datasecurity = None
    _default_client = None
    _domain_client = None

    def __init__(self,
                 default_url: str = None,
                 default_api_key: str = "EMPTY",
                 domain_url: str = None,
                 domain_api_key: str = "EMPTY",
                 skip_content_safety_check: bool = True,
                 skip_data_security_check: bool = False,
                 skip_self_critique_check: bool = False) -> None:
        # api client
        self._default_client = openai.OpenAI(
            api_key=default_api_key, base_url=default_url)
        self._domain_client = openai.OpenAI(
            api_key=domain_api_key, base_url=domain_url)
        # security engine
        self._datasecurity = DataSecurityEngine()
        self._skip_content_safety_check = skip_content_safety_check
        self._skip_data_security_check = skip_data_security_check
        self._skip_self_critique_check = skip_self_critique_check
        # solar network
        self._cn_invoker = LLMSolarNetwork(default_client=self._default_client,
                                           domain_client=self._domain_client,
                                           data_security=self._datasecurity,
                                           skip_content_safety_check=self._skip_content_safety_check,
                                           skip_data_security_check=self._skip_data_security_check,
                                           skip_self_critique_check=self._skip_self_critique_check)
        self._cn_invoker.set_on_start(StartAction(self._cn_invoker))
        self._cn_invoker.set_on_input(InputAction(self._cn_invoker))
        self._cn_invoker.set_on_routing(RoutingAction(self._cn_invoker))
        self._cn_invoker.set_on_thinking(ThinkAction(
            network=self._cn_invoker, receiver=LLMReceiverAPI(self._cn_invoker)))
        self._cn_invoker.set_on_output(OutputAction(self._cn_invoker))
        self._cn_invoker.set_on_finish(FinishAction(self._cn_invoker))

    def chat(self, input_data: dict, history: list = [], cleanup: bool = True):
        logger.info("LLMSolarClient.run_query()")
        t0 = time.perf_counter()
        try:
            self._cn_invoker._history = history
            self._cn_invoker.do_actions(input_data=input_data)
            logger.info(
                f"\nQuestion:\n[green]{self._cn_invoker._input_data['input_query']}[/green]", extra=dict(markup=True))
            logger.info(
                f"\nAnswer:\n[blue]{self._cn_invoker._output_data['output_text']}[/blue]", extra=dict(markup=True))
            output = self._cn_invoker._output_data
            history = self._cn_invoker._history
            if cleanup:
                self._cn_invoker.do_cleanup()
        except Exception as e:
            print(e)
            logger.exception("LLMSolarClient.chat() error.")
            output={}
            output["output_text"] = "oops, found unexpected system error."
        t1 = time.perf_counter() - t0
        logger.info(f"LLMSolarClient.chat() took {t1:.6f} seconds")
        return output, history

    @staticmethod
    def new_instance(runtime_file: str = "llm_api.json"):
        llm_settings = LLMSolarSetting(runtime_file)
        default_api_base = llm_settings.config["default_api_base"]
        default_api_key = llm_settings.config["default_api_key"]
        domain_api_base = llm_settings.config["domain_api_base"]
        domain_api_key = llm_settings.config["domain_api_key"]
        skip_content_safety_check = llm_settings.config["skip_content_safety_check"]
        skip_data_security_check = llm_settings.config["skip_data_security_check"]
        skip_self_critique_check = llm_settings.config["skip_self_critique_check"]
        client = LLMSolarClient(default_url=default_api_base,
                                default_api_key=default_api_key,
                                domain_url=domain_api_base,
                                domain_api_key=domain_api_key,
                                skip_content_safety_check=skip_content_safety_check,
                                skip_data_security_check=skip_data_security_check,
                                skip_self_critique_check=skip_self_critique_check)
        return client

class Action(ABC):
    """
    The Command interface declares a method for executing an action.
    """
    @abstractmethod
    def execute(self) -> any:
        pass

