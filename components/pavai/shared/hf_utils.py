# # For download the models
# !pip -q install huggingface_hub

import os
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['HF_TOKEN']='1'

from datetime import datetime
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
import json
import os

cache_dir="./models"

def hf_model_snapshot_download(model_name_or_path:str=None,
                               snapshot_file:str="local_hf_snapshot.json",
                               cache_dir:str="./models", 
                               append_file:bool=True,notes:str=None):
    if os.path.exists(snapshot_file):
        with open(snapshot_file, 'r') as openfile:
            file_json_object = json.load(openfile)
    else:
        file_json_object=[]
    if not append_file:
        file_json_object=[]
        try:
            os.remove(snapshot_file)
        except OSError as e:
            pass
    # download model file
    model_local_path=snapshot_download(repo_id=model_name_or_path,cache_dir=cache_dir)
    absolute_path = os.path.abspath(model_local_path)
    model_json = {
        "model_name_or_path": model_name_or_path,
        "model_path": model_local_path,
        "model_absolute_path": absolute_path,                     
        "cache_dir": cache_dir,     
        "download_date":  str(datetime.now()),
        "notes":notes
    }
    file_json_object.append(model_json)
    json_object = json.dumps(file_json_object, indent=4)
    with open(snapshot_file, "w+") as outfile:
        outfile.write(json_object)
    return model_local_path

def hf_model_file_download(model_name_or_path:str=None,
                           model_file:str=None,
                           model_alias:str=None,
                           chat_format:str=None,
                           snapshot_file:str="hf_models_gguf.json",
                           cache_dir:str="./models", 
                            append_file:bool=True,notes:str=None):
    if os.path.exists(snapshot_file):
        with open(snapshot_file, 'r') as openfile:
            file_json_object = json.load(openfile)
    else:
        file_json_object=[]
    if not append_file:
        file_json_object=[]
        try:
            os.remove(snapshot_file)
        except OSError as e:
            pass
    # download model file  model_basename="llamaguard-7b.Q2_K.gguf"
    model_local_path = hf_hub_download(repo_id=model_name_or_path, filename=model_file, cache_dir=cache_dir)
    absolute_path = os.path.abspath(model_local_path)
    model_json = {
        "model": model_local_path,
        "model_alias": model_alias,
        "chat_format": chat_format,
        "n_gpu_layers": -1,
        "offload_kqv": True,
        "n_threads": 4,
        "n_batch": 512,
        "n_ctx": 2048,
        "verbose":True 
    }
    file_json_object.append(model_json)
    json_object = json.dumps(file_json_object, indent=4)
    with open(snapshot_file, "w+") as outfile:
        outfile.write(json_object)
    return model_local_path

def hf_multimodal_files_download(
                           model_name_or_path:str=None,
                           model_file:str=None,
                           model_alias:str=None,
                           clip_model_path:str=None,
                           chat_format:str="llava",
                           snapshot_file:str="hf_domain_models.json",
                           cache_dir:str="./models", 
                            append_file:bool=True,notes:str=None):
    if os.path.exists(snapshot_file):
        with open(snapshot_file, 'r') as openfile:
            file_json_object = json.load(openfile)
    else:
        file_json_object=[]
    if not append_file:
        file_json_object=[]
        try:
            os.remove(snapshot_file)
        except OSError as e:
            pass
            
    # model file 
    model_local_path = hf_hub_download(repo_id=model_name_or_path, filename=model_file, cache_dir=cache_dir)
    # clip file
    clip_model_path = hf_hub_download(repo_id=model_name_or_path, filename=clip_model_path, cache_dir="./models")
    absolute_path = os.path.abspath(model_local_path)
    model_json = {
        "model": model_local_path,
        "model_alias": model_alias,
        "chat_format": chat_format,
        "clip_model_path":clip_model_path,
        "n_gpu_layers": -1,
        "offload_kqv": True,
        "n_threads": 4,
        "n_batch": 512,
        "n_ctx": 2048,
        "verbose":True 
    }
    file_json_object.append(model_json)
    json_object = json.dumps(file_json_object, indent=4)
    with open(snapshot_file, "w+") as outfile:
        outfile.write(json_object)
    return model_local_path, clip_model_path