from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import json


def load_model_config(path: str) -> dict:
    models = None
    try:
        with open(path) as handle:
            models = json.loads(handle.read())
            logger.info(models)
    except Exception as e:
        logger.error(str(e.args))
        raise ValueError("Missing local models config file. please check!")
    return models

def load_local_models(path: str='resources/config/llm_locally_aio.json') -> dict:
    return load_model_config(path)

def load_solar_models(path: str='resources/config/llm_solar_openai.json') -> dict:
    return load_model_config(path)

def load_ollama_models(path: str='resources/config/llm_ollama_openai.json') -> dict:
    return load_model_config(path)

if __name__ == "__main__":
    models = load_local_models().keys()
    print(models)