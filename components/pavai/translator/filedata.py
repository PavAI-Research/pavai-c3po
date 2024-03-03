from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

import os 
import re
from pathlib import Path
import pathlib
import shutil
import requests
import multiprocessing
import mmap
import aiofiles
import json    
from bs4 import BeautifulSoup  
from markdown import markdown  
import html2text  
import fitz  
import asyncio
import docx2txt
import datetime 
import time

_DEFAULT_SYSTEM_DOWNLOADS_PATH="./downloads"

def load_text_file(command_name: str,
                   input_text: str,
                   prefix_text: str):
    """/summarize_text:https://cdn.serc.carleton.edu/files/teaching_computation/workshop_2018/activities/plain_text_version_declaration_inde.txt"""
    file_url = input_text.split(command_name)[1].strip()
    file_path = download_file(file_url)
    # file_contents = Path(file_path).read_text()
    with open(file_path) as f:
        file_contents = " ".join(line.rstrip() for line in f)
    if file_path.endswith(".html"):
        file_contents = html_to_text(file_path)
    if file_path.endswith(".md"):
        file_contents = markdown_to_text(file_contents)
    file_contents = prefix_text+file_contents
    history = history + [[file_contents, None]]
    history = history + [[file_path, None]]
    return file_contents, history, file_path, command_name

def convert_pdf_to_text(pdf_file_path: str) -> str:
    """convert pdf file to text"""
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print(text)
    return text

def get_text_file(filename:str):
    return pathlib.Path(filename).read_text()

def save_text_file(filename:str, filecontent:str):
    pathlib.Path(filename).write_text(filecontent)

def download_file(url:str, local_path: str = None):
    local_filename = url.split('/')[-1]
    if local_path is not None:
        local_filename = local_path+local_filename
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return local_filename

def markdown_to_text(markdown_string:str)->str:
    """ Converts a markdown string to plaintext """
    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)
    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)
    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))
    return text

def html_to_text(html_file:str)->str:
    html = open(html_file).read()
    #betterHTML = html.decode(errors='ignore')
    return html2text.html2text(html)

def docx_to_text(filename:str)->str:
    import docx2txt
    # extract text
    text = docx2txt.process(filename)
    return text

def make_dir_if_not_exist(storage_path:str):
    if storage_path is None:
        storage_path="./downloads"    
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)

def get_system_download_path():
    storage_path = _DEFAULT_SYSTEM_DOWNLOADS_PATH
    make_dir_if_not_exist(storage_path)
    return storage_path

def get_system_working_path():
    storage_path = _DEFAULT_SYSTEM_DOWNLOADS_PATH
    storage_path = storage_path+"/working"
    make_dir_if_not_exist(storage_path)    
    return storage_path

def get_system_youtube_path():
    storage_path = _DEFAULT_SYSTEM_DOWNLOADS_PATH
    storage_path = storage_path+"/youtube"
    make_dir_if_not_exist(storage_path)    
    return storage_path

def write_chunk_to_file(args):
    filename, chunk = args
    with open(filename, "ab") as file:
        file.write(chunk)
 
def write_large_data_to_file_parallel(filename, data, num_processes=4):
    pool = multiprocessing.Pool(processes=num_processes)
    chunk_size = len(data) // num_processes
    chunked_data = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    args_list = [(filename, chunk) for chunk in chunked_data]
    pool.map(write_chunk_to_file, args_list)
    pool.close()
    pool.join()

def write_large_data_to_file(filename:str, data:str):
    with open(filename, "wb+") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE) as mapped_file:
            mapped_file.write(data)

def write_large_non_textual_data_to_file(filename, data, chunk_size=8192):
    with open(filename, "wb", buffering=chunk_size) as file:
        for chunk in data:
            file.write(chunk)

async def async_get_json_file(input_filename:str)->str:
    async with aiofiles.open(input_filename, mode='r') as f:
        contents = await f.read()
    json_data = json.loads(contents)
    return json_data

async def async_read_file(input_filename:str):
    async with aiofiles.open(input_filename, mode='r') as f:
        file_data = await f.read()
    return file_data

async def async_write_file(input_filename:str, file_data:str):
    async with aiofiles.open(input_filename, mode='w+') as f:
        await f.write(file_data)

async def async_append_file(input_filename:str, file_data:str):
    # Read the contents of the json file.
    async with aiofiles.open('rhydon.json', mode='r') as f:
        contents = await f.read()
    # Open a new file to write the list of moves into.
    async with aiofiles.open(input_filename, mode='w') as f:
        await f.write('\n'.join(file_data))

def load_file_content(filepath:str,chatbot:list=[],history:list=[]):
    import time
    t0=time.perf_counter()    
    # Get the status
    status = os.stat(filepath)
    print(status)    
    #filepath=filepath.lower()    
    #with open(filepath) as f:
    #    file_contents = " ".join(line.rstrip() for line in f)
    if filepath.lower().endswith(".html"):
        file_contents = html_to_text(filepath)
        file_contents = markdown_to_text(file_contents)
    elif filepath.lower().endswith(".md"):
        file_contents = Path(filepath).read_text()                
        file_contents = markdown_to_text(file_contents)
    elif filepath.lower().endswith(".pdf"):
        file_contents = convert_pdf_to_text(filepath)
    elif filepath.lower().endswith(".docx"):
        file_contents = docx_to_text(filepath)
    elif filepath.lower().endswith(".txt"):
        file_contents = Path(filepath).read_text()        
        print(f"loaded content size {len(file_contents)}")       
    else:
        raise ValueError(f"Unsupported file type {filepath}")
    t1=time.perf_counter()            
    took=(t1-t0)    
    ## update chatbot 
    chatbot=[] if chatbot is None else chatbot  
    chatbot.append((f"loaded file {filepath}", file_contents))
    ## update history
    history=[] if history is None else history    
    history.append({"role": "user", "content": f"loaded file {filepath}\n{file_contents}"})        
    print(f"load_text_file_content took {took}s")
    return chatbot, history 

def list_session_files(storage_path:str="./session_logs", extensions=".log"):
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)    
    # list files in current directory only
    res = []
    for file in os.listdir(storage_path):
        if file.endswith(extensions):
            res.append(file)
    return res

def append_text(new_line, filename="data.txt"):
    if os.path.exists(filename):    
        with open(filename, 'a') as fp:
            fp.writelines(str(new_line)+"\n")
    else:
        with open(filename, 'w+') as fp:
            fp.writelines(str(new_line)+"\n")

def append_json(new_data, filename='data.json'):
    if os.path.exists(filename):
        with open(filename,'r+') as file:
            file_data = json.load(file)
        file_data.append(new_data)            
        with open(filename, 'w') as outfile:
            json.dump(file_data, outfile, indent = 4)
    else:
        lst=json.loads('[]')
        lst.append(new_data)
        with open(filename,'w+') as file:
            json.dump(lst, file, indent = 4)

def save_session_files(chatbot:list, history:list, filename:str=None, storage_path:str="./session_logs"):
    if not os.path.exists(storage_path):
        os.mkdir(storage_path)    
    if filename is None:
        filename = "session_"+datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")

    save_chatbot_file = storage_path +"/"+filename+".log"
    for line in chatbot:
        append_text(new_line=line,filename=save_chatbot_file)
    print("saved text file")

    save_state_file = storage_path +"/"+filename+".json"
    for record in history:
        append_json(new_data=record,filename=save_state_file)
    print("saved json file")
    return save_chatbot_file

def delete_session_files(filename:str,storage_path:str="./session_logs" ):
    if filename is None:
        raise ValueError("nothing to delete, empty filename!")
    filename_txt=storage_path+"/"+filename
    filename_json=filename_txt.replace(".txt",".json") 
    try:      
        if os.path.exists(filename_json):
            os.remove(filename_json)
    except:
        pass    
    try:  
        if os.path.exists(filename_txt):
            os.remove(filename_txt)
    except:
        pass

def load_session_files(filename:str,chatbot:list=[],history:list=[], storage_path:str="./session_logs"):
    history=[] if history is None else history
    chatbot=[] if chatbot is None else chatbot
    filename=storage_path+"/"+filename
    if os.path.exists(filename):
        # read text file
        with open(filename,'r+') as file:
            Lines = file.readlines()
            for line in Lines:
                chatbot.append(eval(line.strip()))         
        # read json file
        jsonfilename=filename.replace(".log",".json")
        with open(jsonfilename,'r') as jsonfile:
            jsondata = json.load(jsonfile)
            history=history+jsondata

    return chatbot, history


if __name__=="__main__":
    # pip install PyMuPDF
    # pip install html2text
    # pip install aiofiles==0.6.0
    # pip install bs4
    # pip install markdown
    # pip install docx2txt
    # --pip install pypandoc

    #filepath="/mnt/win11/shared/multimodal/samples/LLM_VUI.docx"
    #chatbot, history = load_file_content(filepath)   
    #chatbot=[["hello","hi how are you"],["good","yesy"],["what a good day","yesy"]]
    #history=[{"role":"system", "content":"hellow"},{"role":"user", "content":"hi hi"},{"role":"assistant", "content":"yes"}]
    # save_session_files(chatbot,history)
    #print(chatbot)
    #print(history)
    print(list_session_files())
    text_data, json_data=load_session_files(
        filename='session_2024-02-19_11-18-28_PM.log',
        chatbot=[["akak","wowow"]],
        history=[{"role":"user", "content":"new research"}])
    print(text_data, json_data)
