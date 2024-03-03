from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import time
import os
import pavai.llmone.llmdatasecurity as llmdatasecurity

#global
datasecurity = None
def analyze_text(input_text:str, check_pii:bool=True):
    global datasecurity
    datasecurity = llmdatasecurity.DataSecurityEngine() if datasecurity is None else datasecurity
    pii_results = datasecurity.check_text_pii(input_text)
    if pii_results:
        logger.warn("PII dected in the text.")
        logger.debug(pii_results,extra=dict(markup=True))
    return pii_results
 
def anonymize_text(input_text:str, analyzer_results:list=None):
    global datasecurity
    datasecurity = llmdatasecurity.DataSecurityEngine() if datasecurity is None else datasecurity
    if analyzer_results is None:
       analyzer_results = datasecurity.check_text_pii(input_text) 
    deid_text = datasecurity.anonymize_text(input_text=input_text,
                                            analyzer_results=analyzer_results)
    logger.debug(f"Original Text:\n[blue]{input_text}[/blue]", extra=dict(markup=True))
    logger.debug(f"Anonymized Text:\n [green]{deid_text}[/green]", extra=dict(markup=True))
    return deid_text
