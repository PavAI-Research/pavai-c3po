from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import os
import traceback
import requests
import shutil

def download_file(url, local_path: str = None):
    local_filename = url.split('/')[-1]
    if local_path is not None:
        local_filename = local_path+local_filename
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename


def get_styletts2_model_files(local_voice_path: str="resources/models/styletts2/Models", remote_folder: str = "https://huggingface.co/mychen76") -> str:
    """download styletts model file from remote location"""
    try:
        LibriTTS=local_voice_path+"/LibriTTS/"
        if not os.path.exists(LibriTTS):
            os.mkdir(LibriTTS)
            logger.warn(f"downloading {LibriTTS}")
            LibriTTS_model_config_url=f"{remote_folder}/styletts2/resolve/main/Models/LibriTTS/config.yml"
            local_filename = download_file(url=LibriTTS_model_config_url,local_path=LibriTTS)
            logger.warn(f"styletts2_model downloaded {local_filename}")
            LibriTTS_model_bin=f"{remote_folder}/styletts2/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth"
            local_filename = download_file(url=LibriTTS_model_bin,local_path=LibriTTS)
            logger.warn(f"styletts2_model downloaded {local_filename}")
        else:
            logger.info(f"styletts2_model already exist: {LibriTTS}")            

        LJSpeech=local_voice_path+"/LJSpeech/"
        if not os.path.exists(LJSpeech):
            os.mkdir(LJSpeech)
            logger.warn(f"downloading {LJSpeech}")            
            LJSpeech_model_config_url=f"{remote_folder}/styletts2/resolve/main/Models/LJSpeech/config.yml"
            local_filename = download_file(url=LJSpeech_model_config_url,local_path=LJSpeech)
            logger.warn(f"styletts2_model downloaded {local_filename}")                      
            LJSpeech_model_bin=f"{remote_folder}/styletts2/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth"
            local_filename = download_file(url=LJSpeech_model_bin,local_path=LJSpeech)
            logger.info(f"styletts2_model downloaded {local_filename}")          
        else:
            logger.info(f"styletts2_model already exist: {LJSpeech}")            
    except Exception as e:
        logger.error(f"Exception occured {e.args}")
        print(traceback.format_exc())
        logger.error(str(traceback.format_exc()))
        raise Exception("Failed to download styletts2 model files!")
    