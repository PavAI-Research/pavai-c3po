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

_DEFAULT_SYSTEM_DOWNLOADS_PATH=config.system_config["DEFAULT_SYSTEM_DOWNLOADS_PATH"]

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
    doc = fitz.open(pdf_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print(text)
    return text

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

def get_text_file_content(file_path: str, history, prefix_text: str):
    file_path=file_path.lower()
    file_contents = Path(file_path).read_text()
    with open(file_path) as f:
        file_contents = " ".join(line.rstrip() for line in f)
    if file_path.endswith(".html"):
        file_contents = html_to_text(file_path)
    elif file_path.endswith(".md"):
        file_contents = markdown_to_text(file_contents)
    elif file_path.endswith(".pdf"):
        file_contents = convert_pdf_to_text(file_path)
    file_contents = prefix_text+file_contents
    history = history + [[file_contents, None]]
    return file_contents, history, file_path

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
    betterHTML = html.decode(errors='ignore')
    return html2text.html2text(betterHTML)

def make_dir_if_not_exist(storage_path:str):
    if storage_path is None:
        storage_path="workspace/downloads"    
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
    storage_path = storage_path+"/youtube_audio"
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
