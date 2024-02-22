import time
from dotenv import dotenv_values
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import (Pretty, pprint)
from rich import print, pretty, console
import logging
import warnings
from pathlib import Path
import sys
import os
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",
                    datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()
warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent))
print(os.getcwd())
import src.shared.solar.llmdatasecurity as llmdatasecurity
import src.shared.datasecurity as datasecurity

def test_data_security_1():
    datasecurity = llmdatasecurity.DataSecurityEngine()
    input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
    # input_text = "Hello, how are you?"
    pii_results = datasecurity.check_text_pii(input_text)
    deid_text = datasecurity.anonymize_text(input_text=input_text,
                                            analyzer_results=pii_results,
                                            operators=datasecurity._custom_operators)
    logger.info(
        f"Original Text:\n[blue]{input_text}[/blue]", extra=dict(markup=True))
    logger.info(
        f"Anonymized Text:\n [green]{deid_text}[/green]", extra=dict(markup=True))

def test_data_security_2():
    input_text = "Hello, my name is John and I live in New York. My credit card number is 3782-8224-6310-005 and my phone number is (212) 688-5500."
    # input_text = "Hello, how are you?"
    pii_results = datasecurity.analyze_text(input_text=input_text)
    deid_text = datasecurity.anonymize_text(input_text=input_text,
                                            analyzer_results=pii_results)
    logger.info(pii_results,extra=dict(markup=True))
    logger.info(f"Original Text:\n[blue]{input_text}[/blue]", extra=dict(markup=True))
    logger.info(f"Anonymized Text:\n [green]{deid_text}[/green]", extra=dict(markup=True))

if __name__ == "__main__":
    test_data_security_1()
    test_data_security_2()
