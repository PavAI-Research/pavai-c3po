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
            print(f"downloading {LibriTTS}")
            LibriTTS_model_config_url=f"{remote_folder}/styletts2/resolve/main/Models/LibriTTS/config.yml"
            local_filename = download_file(url=LibriTTS_model_config_url,local_path=LibriTTS)
            print(f"styletts2_model downloaded {local_filename}")
            LibriTTS_model_bin=f"{remote_folder}/styletts2/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth"
            local_filename = download_file(url=LibriTTS_model_bin,local_path=LibriTTS)
            print(f"styletts2_model downloaded {local_filename}")
        else:
            print(f"styletts2_model already exist: {LibriTTS}")            

        LJSpeech=local_voice_path+"/LJSpeech/"
        if not os.path.exists(LJSpeech):
            os.mkdir(LJSpeech)
            print(f"downloading {LJSpeech}")            
            LJSpeech_model_config_url=f"{remote_folder}/styletts2/resolve/main/Models/LJSpeech/config.yml"
            local_filename = download_file(url=LJSpeech_model_config_url,local_path=LJSpeech)
            print(f"styletts2_model downloaded {local_filename}")                      
            LJSpeech_model_bin=f"{remote_folder}/styletts2/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth"
            local_filename = download_file(url=LJSpeech_model_bin,local_path=LJSpeech)
            print(f"styletts2_model downloaded {local_filename}")          
        else:
            print(f"styletts2_model already exist: {LJSpeech}")            
    except Exception as e:
        print("Exception occured ",e.args)
        print(traceback.format_exc())
        raise Exception("Failed to download styletts2 model files!")
    