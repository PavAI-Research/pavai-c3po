from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import json

def load_voices(path: str) -> dict:
    reference_voices = None
    try:
        with open('resources/config/reference_voices.json') as handle:
            reference_voices = json.loads(handle.read())
            logger.info(reference_voices)
    except Exception as e:
        logger.error(str(e.args))
        raise ValueError("Missing reference voices config file. please check!")
    return reference_voices


def get_voice_names(path: str) -> list:
    reference_voices = load_voices(path)
    return reference_voices.keys()


def get_voice_files(path: str) -> list:
    reference_voices = load_voices(path)
    return reference_voices.values()


if __name__ == "__main__":
    voicefile = "resources/config/reference_voices.json"
    voices = get_voice_names(voicefile)
    print(voices)
