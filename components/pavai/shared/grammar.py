from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

# import os
# from dotenv import dotenv_values
# system_config = {
#     **dotenv_values("env.shared"),  # load shared development variables
#     **dotenv_values("env.secret"),  # load sensitive variables
#     **os.environ,  # override loaded values with environment variables
# }
# from dotenv import dotenv_values
# system_config = dotenv_values("env_config")
# import logging
# from rich.logging import RichHandler
# from rich import print,pretty,console
# from rich.pretty import (Pretty,pprint)
# from rich.panel import Panel
# logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
# logger = logging.getLogger(__name__)
# pretty.install()

# import warnings 
# warnings.filterwarnings("ignore")
import os 
import functools
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from huggingface_hub import hf_hub_download,snapshot_download    
#import torch 
# use_device = "cuda" if torch.cuda.is_available() else "cpu"
# use_torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

DEFAULT_GRAMMAR_MODEL_SIZE = "pszemraj/grammar-synthesis-small"
grammar_corrector=None

@functools.lru_cache
def init_grammar_correction_model(model_name_or_path: str = DEFAULT_GRAMMAR_MODEL_SIZE, device_map='auto',cache_dir:str="resources/models/grammar", local_files_only:bool=False):
    global grammar_corrector
    """
    -pszemraj/grammar-synthesis-small
    -pszemraj/flan-t5-xl-grammar-synthesis
    The intent is to create a text2text language model that successfully completes 
    "single-shot grammar correction" on a potentially grammatically incorrect text that could have a lot of mistakes with the important qualifier of it does not semantically change text/information that IS grammatically correct.    
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,cache_dir=cache_dir, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,cache_dir=cache_dir, local_files_only=local_files_only)
    grammar_corrector = pipeline("text2text-generation", model=model, tokenizer=tokenizer,device_map=device_map)
    ## alternative using pipeline only
    #grammar_corrector = pipeline("text2text-generation", model_name_or_path, device_map=device_map)
    return grammar_corrector

def fix_grammar_error(raw_text: str,
                      enabled: bool = False,
                      max_length: int = 128,
                      repetition_penalty: float = 1.05,
                      num_beams: int = 4):
    # raw_text="Iwen 2the store yesturday to bye some food. I needd milk, bread, andafew otter things. The $$tore was reely crowed and I had a hard time finding everyting I needed. I finaly madeit t0 dacheck 0ut line and payed for my stuff."
    if not enabled:
        return raw_text, None
    global grammar_corrector
    params = {
        'max_length': max_length,
        'repetition_penalty': repetition_penalty,
        'num_beams': num_beams
    }
    if grammar_corrector is None:
        grammar_corrector = init_grammar_correction_model()
    results = grammar_corrector(raw_text, **params)
    corrected_text = results[0]['generated_text']
    response = "---original version--\n"+raw_text + \
        "\n---revised version---\n"+corrected_text
    logger.info(f"fix_grammar_error:{response}")
    return corrected_text, raw_text

def get_or_download_grammar_model_snapshot(cache_dir:str="resources/models/grammar"):
    logger.info("get_or_download_grammar_model_snapshot")    
    #local_model_file=snapshot_download(repo_id="pszemraj/bart-base-grammar-synthesis", cache_dir=cache_dir)           
    local_model_file=snapshot_download(repo_id="pszemraj/grammar-synthesis-small", cache_dir=cache_dir)               
    logger.info(local_model_file)
    if os.path.exists(local_model_file):
        logger.info(f"model file downloaded folder: {cache_dir} - Success!")
    else:
        logger.error(f"model file downloaded - Failed!")
    return local_model_file 

# if __name__ == "__main__":
#     get_or_download_grammar_model_snapshot()
#     init_grammar_correction_model()
#     raw_text = "Iwen 2the store yesturday to bye some food. I needd milk, bread, andafew otter things. The $$tore was reely crowed and I had a hard time finding everyting I needed. I finaly madeit t0 dacheck 0ut line and payed for my stuff."
#     result=fix_grammar_error(raw_text)
#     print(f"[test#4] grammar model {result}: Passed")                   
