from __future__ import annotations
from dotenv import dotenv_values
system_config = dotenv_values("env_config")
import logging
import warnings 
from rich.logging import RichHandler
from rich import print,pretty
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger(__name__)
pretty.install()
warnings.filterwarnings("ignore")
import sys
import os
import time
import json
from abc import ABC, abstractmethod
from pytube import (YouTube, Search)
from duckduckgo_search import (DDGS, AsyncDDGS)
import duckduckgo_search
from retrying import retry
from pathlib import Path
import asyncio
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from urllib.parse import (urlparse, parse_qs, quote)
from shared.audio.transcribe import (transcriber_client, get_transcriber,
                             DISTILLED_WHISPER_MODEL_SIZE, DEFAULT_WHISPER_MODEL_SIZE)
from .fileutil import (get_system_working_path, get_system_download_path, get_system_youtube_path,
                           load_text_file, save_text_file, save_text_file)

__author__ = "mychen76@gmail.com"
__copyright__ = "Copyright 2023, "
__version__ = "0.0.3"

# duckduckgo setup
# ----------------
# pip install duckduckgo-search==4.1.1
# pip install retrying==1.3.4
# limitation:
# Thcommandse default rate limit for DuckDuckGo is 20 requests per second.

# Youtube commands
_youtube_transcribe_command = "/ytranscribe:"
_youtube_summarize_command = "/ysummarize:"
_youtube_translate_command = "/ytranslate:"
# Web commands
_web_search_command = "/ddgsearch?"
_web_news_command = "/ddgnews?"
_web_image_command = "/ddgimage?"
_web_video_command = "/ddgvideo?"
_web_answer_command = "/ddganswer?"
_web_translate_command = "/ddgtranslate?"
_web_suggest_command = "/ddgsuggest?"
# file commands
_summarize_text_file_command = "/tfsummarize:"
_load_text_file_command = "/tfload:"
_dummy_command = "/dummy"
# image
_image_chat_command = "/imagechat?"

# bypass curl-cffi NotImplementedError in windows https://curl-cffi.readthedocs.io/en/latest/faq/
if sys.platform.lower().startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class Command(ABC):
    """
    The Command interface declares a method for executing a command.
    """
    @abstractmethod
    def execute(self) -> any:
        pass

class SimpleCommand(Command):
    """
    commands implement simple operations.
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> list:
        print(f"Execute simpleCommand: ({self._payload})")

class DoNothingDummyCommand(Command):
    """
    commands implement no nothing command.
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> any:
        pass

class WebTextSearchCommand(SimpleCommand):
    """
    text search: ddgs text -k 'ayrton senna'    

    Duckduckgo search operators
    Keywords example 	Result
    cats dogs 	Results about cats or dogs
    "cats and dogs" 	Results for exact term "cats and dogs". If no results are found, related results are shown.
    cats -dogs 	Fewer dogs in results
    cats +dogs 	More dogs in results
    cats filetype:pdf 	PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html
    dogs site:example.com 	Pages about dogs from example.com
    cats -site:example.com 	Pages about cats, excluding example.com
    intitle:dogs 	Page title includes the word "dogs"
    inurl:cats 	Page url includes the word "cats"
    """

    def __init__(self, payload: str) -> None:
        """paramters: (keywords, max_results=5) """
        self._payload = payload

    def execute(self) -> str:
        """
        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m, y. Defaults to None.
            backend: api, html, lite. Defaults to api.
                api - collect data from https://duckduckgo.com,
                html - collect data from https://html.duckduckgo.com,
                lite - collect data from https://lite.duckduckgo.com.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.
        Yields:
            dict with search results.
        """
        logger.debug(f"WebTextSearchCommand input: ({self._payload})")
        results = []
        with DDGS() as ddgs:
            ddgs_results = [r for r in ddgs.text(keywords=self._payload["keywords"],
                                                 safesearch=self._payload["safesearch"],
                                                 backend=self._payload["backend"],
                                                 region=self._payload["region"],
                                                 timelimit=self._payload["timelimit"],
                                                 max_results=self._payload["max_results"])]
            for r in ddgs_results:
                results.append(r)

        formatted_output=""
        if isinstance(results,list):
            for r in results:
                formatted_output=formatted_output+"<b>"+r["title"]+"</b>\n"+r["href"]+"\n"+r["body"]+"<hr/>"
        return formatted_output

class WebImageSearchCommand(SimpleCommand):
    def __init__(self, payload: str) -> None:
        """
        Parameters: (keywords,region="us-en",max_results=5)
        """
        self._payload = payload

    def execute(self) -> any:
        """DuckDuckGo images search. Query params: https://duckduckgo.com/params
        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: Day, Week, Month, Year. Defaults to None.
            size: Small, Medium, Large, Wallpaper. Defaults to None.
            color: color, Monochrome, Red, Orange, Yellow, Green, Blue,
                Purple, Pink, Brown, Black, Gray, Teal, White. Defaults to None.
            type_image: photo, clipart, gif, transparent, line.
                Defaults to None.
            layout: Square, Tall, Wide. Defaults to None.
            license_image: any (All Creative Commons), Public (PublicDomain),
                Share (Free to Share and Use), ShareCommercially (Free to Share and Use Commercially),
                Modify (Free to Modify, Share, and Use), ModifyCommercially (Free to Modify, Share, and
                Use Commercially). Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Yields:
            dict with image search results.
        """
        logger.debug(f"Runnning WebImageSearchCommand: ({self._payload})")
        results = []
        with DDGS() as ddgs:
            ddgs_results = ddgs.images(
                keywords=self._payload["keywords"],
                region=self._payload["region"],
                safesearch=self._payload["safesearch"],
                size="small",
                color=None,
                type_image=None,
                layout=None,
                license_image=None,
                timelimit=self._payload["timelimit"],
                max_results=self._payload["max_results"],
            )
            for r in ddgs_results:
                results.append(r)

        formatted_output=""
        records_to_show=10
        max_count=0
        if isinstance(results,list):
            for r in results:
                formatted_output=formatted_output+"<b>"+r["title"]+"</b>\n<image src='"+r["image"]+"'/>\n"+r["source"]+"<hr/>"            
                max_count+=1
                if max_count>records_to_show:
                    break
        return formatted_output                

class WebVideoSearchCommand(SimpleCommand):
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> any:
        """DuckDuckGo videos search. Query params: https://duckduckgo.com/params

        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m. Defaults to None.
            resolution: high, standart. Defaults to None.
            duration: short, medium, long. Defaults to None.
            license_videos: creativeCommon, youtube. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.

        Yields:
            dict with videos search results

        """
        logger.debug(f"Runnning WebVideoSearchCommand: ({self._payload})")
        results = []
        with DDGS() as ddgs:
            ddgs_results = ddgs.videos(
                keywords=self._payload["keywords"],
                region=self._payload["region"],
                timelimit=self._payload["timelimit"],
                max_results=self._payload["max_results"],
                safesearch=self._payload["safesearch"],
                license_videos=self._payload["license_videos"],
                resolution=self._payload["resolution"],
                duration=self._payload["duration"],
            )
            for r in ddgs_results:
                results.append(r)

        formatted_output=""
        records_to_show=5
        max_count=0
        if isinstance(results,list):
            for r in results:
                formatted_output=formatted_output+"<b>"+r["title"]+"</b>\n"+r["description"]+" | "+r["duration"]+"\n"            
                formatted_output=formatted_output+"<div align='left'><a href='"+r["content"]+"'><image src='"+r["images"]['small']+"'/></a><hr/>"                
                max_count+=1
                if max_count>records_to_show:
                    break
        return formatted_output                

class WebAnswerCommand(SimpleCommand):
    """DuckDuckGo instant answers. Query params: https://duckduckgo.com/params"""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> any:
        """
         keywords: keywords for query.
        """
        logger.debug(f"Runnning WebAnswerCommand: ({self._payload})")
        results = []
        with DDGS() as ddgs:
            ddgs_results = ddgs.answers(keywords=self._payload["keywords"])
            for r in ddgs_results:
                results.append(r)

        formatted_output=""
        records_to_show=10
        max_count=0
        if isinstance(results,list):
            for r in results:
                if r["topic"] is not None:
                    formatted_output=formatted_output+"<b>"+r["text"]+"</b>\n"+r["topic"]+"<hr/>"
                else:
                    formatted_output=formatted_output+"<b>"+r["text"]+"</b>"+"<hr/>"            
                max_count+=1
                if max_count>records_to_show:
                    break
        return formatted_output                

class WebSuggestCommand(SimpleCommand):
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> any:
        """DuckDuckGo suggestions. Query params: https://duckduckgo.com/params
        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".

        Yields:
            dict with suggestions results.
        """
        logger.debug(f"Runnning WebSuggestCommand: ({self._payload})")
        if self._payload["region"] is None:
            self._payload["region"] = "wt-wt"
        results = []
        with DDGS() as ddgs:
            ddgs_results = ddgs.suggestions(
                keywords=self._payload["keywords"], region=self._payload["region"])
            for r in ddgs_results:
                results.append(r)

        formatted_output=""
        records_to_show=10
        max_count=0
        if isinstance(results,list):
            for r in results:
                formatted_output=formatted_output+"*<b>"+r["phrase"]+"</b>"+"<hr/>"            
                max_count+=1
                if max_count>records_to_show:
                    break
        return formatted_output                

class WebNewsCommand(SimpleCommand):
    """# get latest news: ddgs news -k "ukraine war" -s off -t d -m 10 """
    """# last day's and save it to a csv file: ddgs news -k "hubble telescope" -t d -m 50 -o csv """

    def __init__(self, payload: str) -> None:
        """
        Parameters: (keywords,region="us-en",timelimit="d", max_results=5)
        Args:
            keywords: keywords for query.
            region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
            safesearch: on, moderate, off. Defaults to "moderate".
            timelimit: d, w, m. Defaults to None.
            max_results: max number of results. If None, returns results only from the first response. Defaults to None.        
        """
        self._payload = payload

    def execute(self) -> any:
        # get latest news parameters:
        # self._payload["safesearch"]="moderate"
        # self._payload["timelimit"]="d"
        print(f"Runnning WebNewsSearchCommand: ({self._payload})")
        results = []
        with DDGS() as ddgs:
            ddgs_results = ddgs.news(
                keywords=self._payload["keywords"],
                region=self._payload["region"],
                safesearch=self._payload["safesearch"],
                timelimit=self._payload["timelimit"],
                max_results=self._payload["max_results"])
            for r in ddgs_results:
                results.append(r)

        formatted_output=""
        if isinstance(results,list):
            for r in results:
                if r["image"] is None:
                    formatted_output=formatted_output+"<b>"+r["date"]+"|"+r["title"]+"</b>\n"+r["url"]+"\n"+r["body"]+"\n"+r["source"]+"<hr/>"
                else:
                    formatted_output=formatted_output+"<b>"+r["date"]+"|"+r["title"]+"</b>\n"+r["url"]+"\n"+r["body"]+"\n<image src='"+r["image"]+"'/>\n"+r["source"]+"<hr/>"                                             
        return formatted_output

class WebTranslateCommand(SimpleCommand):
    def __init__(self, payload: str) -> None:
        """paramters: (keywords, to_lang="fr") """
        self._payload = payload

    def execute(self) -> any:
        """DuckDuckGo translate
        Args:
            keywords: string or a list of strings to translate
            from_: translate from (defaults automatically). Defaults to None.
            to: what language to translate. Defaults to "en".

        Returns:
            dict with translated keywords.
            example:
            {'detected_language': 'en', 'translated': 'papillon', 'original': 'butterfly'}
        """
        logger.debug(f"Runnning WebTranslateCommand: ({self._payload})")
        with DDGS() as ddgs:
            result = ddgs.translate(keywords=self._payload["keywords"],
                                    from_=self._payload["from_lang"],
                                    to=self._payload["to_lang"])
        output=f'from [{result["detected_language"]}] to [{self._payload["to_lang"]}]\n'
        output=output+f'translated: {result["translated"]} \n'
        output=output+f'original: {result["original"]} <hr/>'        
        return output

class ImageChatCommand(SimpleCommand):
    def __init__(self, payload: str) -> None:
        """paramters: (keywords, to_lang="fr") """
        self._payload = payload

    def execute(self) -> any:
        """Image Chat URL
        Args:
            keywords: string or a list of strings to translate
            question: translate from (defaults automatically). Defaults to None.
            url: what language to translate. Defaults to "en".

        Returns:
            dict on image description.
            example:
            {'image': 'https://', 'query': 'user question', 'response': 'AI resply'}
        """
        logger.debug(f"Runnning ImageChatCommand: ({self._payload})")
        output = {'image_url': self._payload["image"], 'query': self._payload["keywords"], 'response': 'pending'}
        return output["query"]+"???"+output["image_url"]


_available_web_commands_map = {
    _web_search_command: WebTextSearchCommand,
    _web_image_command: WebImageSearchCommand,
    _web_video_command: WebVideoSearchCommand,
    _web_news_command: WebNewsCommand,
    _web_answer_command: WebAnswerCommand,
    _web_translate_command: WebTranslateCommand,
    _web_suggest_command: WebSuggestCommand,
    _image_chat_command: ImageChatCommand
}

class WebCommandRunner():

    def retry_if_api_rate_limit_error(exception):
        """Return True if we should retry (in this case when it's an RateLimitException), False otherwise"""
        return isinstance(exception, duckduckgo_search.exceptions.DuckDuckGoSearchException)

    @retry(retry_on_exception=retry_if_api_rate_limit_error, stop_max_attempt_number=3, wait_fixed=2000)
    def process(self, input_text):
        """Method to process input command """
        output = ""
        command_to_run = self.extract_input_commands_and_parameters(
            input_text=input_text)
        if command_to_run:
            logger.debug(f"WebCommandBuilder.executing command:{input_text}")
            results = command_to_run.execute()
        if isinstance(results,list):
            for r in results:
                output=output+"<b>"+r["title"]+"</b>\n"+r["href"]+"\n"+r["body"]+"<hr/>"
        else:
           output= results
        return output

    def parse_input_parameters(self, input_text: str):
        """
        region: wt-wt, us-en, uk-en, ru-ru, etc. Defaults to "wt-wt".
        safesearch: on, moderate, off. Defaults to "moderate".
        timelimit: d, w, m. Defaults to None.
        backend: api, html, lite. Defaults to api.
           api - collect data from https://duckduckgo.com,
           html - collect data from https://html.duckduckgo.com,
           lite - collect data from https://lite.duckduckgo.com.
        """
        payload = {
            "keywords": input_text,
            "region": None,
            "timelimit": None,
            "max_results": 5,
            "safesearch": "moderate",
            "backend": "lite",
            "license_videos": None,
            "resolution": None,
            "duration": None,
            "to_lang": None,
            "from_lang": None,
            "size": None,
            "type_image": None,
            "color": None,
        }
        try:
            parsed_url = urlparse(input_text)
            if "query" in parse_qs(parsed_url.query):
                payload['keywords'] = parse_qs(parsed_url.query)['query'][0]
            if "region" in parse_qs(parsed_url.query):
                payload['region'] = parse_qs(parsed_url.query)['region'][0]
            if "timelimit" in parse_qs(parsed_url.query):
                payload['timelimit'] = parse_qs(parsed_url.query)[
                    'timelimit'][0]
            if "max_results" in parse_qs(parsed_url.query):
                payload['max_results'] = parse_qs(parsed_url.query)[
                    'max_results'][0]
                payload['max_results'] = int(payload['max_results'])
                if int(payload['max_results']) == 0:
                    payload['max_results'] = 5
            if "to_lang" in parse_qs(parsed_url.query):
                payload['to_lang'] = parse_qs(parsed_url.query)['to_lang'][0]
            if "from_lang" in parse_qs(parsed_url.query):
                payload['from_lang'] = parse_qs(parsed_url.query)[
                    'from_lang'][0]
            if "safesearch" in parse_qs(parsed_url.query):
                payload['safesearch'] = parse_qs(parsed_url.query)[
                    'safesearch'][0]
            if "backend" in parse_qs(parsed_url.query):
                payload['backend'] = parse_qs(parsed_url.query)['backend'][0]
            if "size" in parse_qs(parsed_url.query):
                payload['size'] = parse_qs(parsed_url.query)['size'][0]
            if "license_image" in parse_qs(parsed_url.query):
                payload['license_image'] = parse_qs(parsed_url.query)[
                    'license_image'][0]
            if "type_image" in parse_qs(parsed_url.query):
                payload['type_image'] = parse_qs(parsed_url.query)[
                    'type_image'][0]
            if "color" in parse_qs(parsed_url.query):
                payload['color'] = parse_qs(parsed_url.query)['color'][0]
            if "image" in parse_qs(parsed_url.query):
                payload['image'] = parse_qs(parsed_url.query)['image'][0]                
        except:
            pass
        return payload

    def extract_input_commands_and_parameters(self, input_text) -> SimpleCommand:
        """Method to determine command object."""
        command_to_run = None
        if input_text is None or len(input_text.strip()) == 0:
            return command_to_run
        try:
            for command in _available_web_commands_map.keys():
                if self.input_has_supported_command(input_text=input_text, command=command):
                    # extract input parameters
                    command_params = input_text.split(command)[1].strip()
                    return self.create_command(input_text=input_text, command=command,
                                               command_params=command_params)
        except Exception as e:
            logger.error(
                'Error extract_input_commands_and_parameters: %s', e, exc_info=True)
            # print("extract_input_commands_and_parameters has error:", e)
        return command_to_run

    def input_has_supported_command(self, input_text: str, command: str) -> bool:
        if input_text is None:
            return False
        if command in input_text:
            return True
        else:
            return False

    def create_command(self, input_text: str, command: str, command_params: str) -> SimpleCommand:
        """Method: Create a command instance."""
        input_payload = self.parse_input_parameters(input_text=input_text)
        input_payload["keywords"] = quote(input_payload["keywords"])
        return _available_web_commands_map[command](input_payload)

class YoutubeAudioDownloadCommand(SimpleCommand):

    def __init__(self, payload: str) -> None:
        """parameters: (keywords, max_results=5) """
        self._payload = payload

    def get_downloaded_file(self, local_filename_meta: str):
        output = None
        if os.path.exists(local_filename_meta):
            with open(local_filename_meta, "r") as fp:
                output = json.load(fp)
                output["status"] = "local_copy"
        return output

    def download_youtube(self, video_url: str, video_id: str, local_storage: str, local_filename_meta: str, local_filename_audio: str):
        output = {
            "video_url": video_url,
            "video_id": video_id,
            "audio_file": "",
            "title": "",
            "description": "",
            "status": "download",
            "performance": 0,
        }
        video = YouTube(video_url)
        audio = video.streams.filter(only_audio=True).first()
        output_file = audio.download(output_path=local_storage)
        output["title"] = video._title
        output["description"] = video.description
        output["age_restricted"] = video._age_restricted
        output["author"] = video._author
        output["channel"] = video.channel_url
        output["length"] = video.length
        output["publish_date"] = video._publish_date
        # save video audio file
        output["audio_file_meta"] = local_filename_meta
        output["audio_file"] = local_filename_audio
        # save video meta into JSON
        os.rename(output_file, local_filename_audio)
        with open(local_filename_meta, "w") as fp:
            json.dump(output, fp)
        return output

    def execute(self) -> any:
        t0 = time.perf_counter()
        video_id = self._payload["video_id"]
        video_url = self._payload["video_url"]
        local_storage = self._payload["local_storage"]
        print(f"YoutubeAudioDownloadCommand {video_url} started...")
        try:
            # local file names
            local_filename_meta = local_storage+"/"+video_id+".meta"
            local_filename_audio = local_storage+"/"+video_id+".mp4"
            output = self.get_downloaded_file(local_filename_meta)
            if output is None:
                output = self.download_youtube(video_url=video_url,
                                               video_id=video_id,
                                               local_storage=local_storage,
                                               local_filename_meta=local_filename_meta,
                                               local_filename_audio=local_filename_audio)
        except Exception as e:
            print(e)
            output["status"] = f"error: Youtube download failed due {e}"
        t1 = time.perf_counter()-t0
        output["performance"] = t1
        print(
            f"YoutubeAudioDownloadCommand {local_filename_audio} completed in {t1}s")
        return output

class YoutubeVideoDownloadCommand(SimpleCommand):

    def __init__(self, payload: str) -> None:
        """parameters: (keywords, max_results=5) """
        self._payload = payload

    def execute(self) -> any:
        t0 = time.perf_counter
        local_filename = self._payload["local_filename"]
        video_url = self._payload["video_url"]
        local_storage = self._payload["local_storage"]
        local_filename = local_storage+"/"+local_filename
        print(f"YoutubeVideoDownloadCommand {video_url} started...")
        if os.path.exists(local_filename):
            return local_filename
        video = YouTube(video_url)
        output_file = video.streams.first().download(output_path=local_storage)
        t1 = time.perf_counter-t0
        print(f"YoutubeVideoDownloadCommand {video_url} completed in {t1}s")
        return output_file

class YoutubeSearchCommand(SimpleCommand):

    def __init__(self, payload: str) -> None:
        """parameters: (query, max_results=5) """
        self._payload = payload

    def execute(self) -> any:
        t0 = time.perf_counter
        query = self._payload["query"]
        print(f"YoutubeSearchCommand {query} started...")
        ys = Search(query)
        if len(ys.results) == 0:
            return "No found"
        result = f"found {len(ys.results)} videos\n"
        for video in ys.results:
            result = result+"title: "+video._title+" ("+video.watch_url+")\n"
        t1 = time.perf_counter-t0
        print(f"YoutubeSearchCommand completed in {t1}s")
        return result

def transcribe_audio_to_text(input_audio: Union[str, BinaryIO, np.ndarray],
                             task_mode="transcribe",
                             model_size: str = "large",
                             beam_size: int = 5,
                             vad_filter: bool = True,
                             language: str = None,
                             include_timestamp_seg: bool = False,
                             transcriber_model_id: str = DISTILLED_WHISPER_MODEL_SIZE,
                             transcriber_class: str = "DistrilledTranscriber") -> str:
    """
    # convert speech to text
    """
    print("speech_to_text: mode=", task_mode)
    if isinstance(input_audio, str):
        print("input audio file: ", input_audio)
    else:
        print("input audio: ", type(input_audio))
    t0 = time.perf_counter()
    faster_transcriber = get_transcriber(model_id_or_path=transcriber_model_id,
                                         transcriber_class=transcriber_class)
    outputs = transcriber_client(abstract_transcriber=faster_transcriber,
                                 input_audio=input_audio,
                                 input_type="audio", task=task_mode)
    t1 = time.perf_counter()-t0
    print(f"transcribe_audio_to_text took {t1}s")
    return outputs["transcription"], outputs["language"]

def transcribe_parse_input(input_text: str, youtube_command: str):
    video_url = input_text.split(youtube_command)[1].strip()
    print(f"transcribe_youtube_audio: {video_url}")
    video_id = video_url.split("watch?v=")[1].strip()
    local_storage = get_system_youtube_path()
    payload = {
        "video_url": video_url,
        "video_id": video_id,
        "local_storage": local_storage
    }
    return payload

def transcribe_prepare_output(youtube_command:str,task_mode:str,language: str, input_text: str, audio_text: str, video_title: str, audio_filename: str, history: list):
    if youtube_command==_youtube_summarize_command:
        task_mode = "summarize" 
    local_filename = f"{audio_filename}_{task_mode}.txt"    
    save_text_file(filename=local_filename, filecontent=audio_text)
    print("saved transcribed text file: ",local_filename)
    if language is None or task_mode=="translate":
        language = "en"
    # format display text
    transcribed_output_text = f"""
    `{input_text}` 
    Title: {video_title} 
    """
    transcribed_output_text = transcribed_output_text+"\n"+audio_text
    history = history + [[transcribed_output_text, None]]
    history = history + [[local_filename, None]]
    output = {
        "transcribed_output_text": transcribed_output_text,
        "history": history,
        "audio_filename": local_filename,
        "language": language
    }
    local_meta_filename = f"{audio_filename}_{task_mode}.meta"    
    with open(local_meta_filename, "w") as fp:
        json.dump(output, fp)    
    return output

def get_transcribed_text_file(local_filename: str):
    data = None
    if os.path.exists(local_filename):
        data = Path(local_filename).read_text()
    return data

def get_transcribed_meta_file(local_filename_meta: str):
    output = None
    if os.path.exists(local_filename_meta):
        with open(local_filename_meta, "r") as fp:
            output = json.load(fp)
            output["status"] = "local_copy"
    return output

def transcribe_youtube(youtube_command: str,
                       input_text: str,
                       history: list,
                       prefix_text: str = "transcribed youtube audio:",
                       task_mode: str = "transcribe",
                       transcriber_class: str = "DistrilledTranscriber"):
    t0 = time.perf_counter()
    audio_filename = ""
    audio_text = input_text
    payload = transcribe_parse_input(input_text=input_text, youtube_command=youtube_command)
    output = YoutubeAudioDownloadCommand(payload).execute()
    audio_filename = output["audio_file"]
    if len(audio_filename.strip()) == 0:
        # something wrong if not audio file produced
        print("transcribe_youtube error: missing locally saved audio file!")
        return output["status"], history, audio_filename, "en"
    video_title = output["title"]
    history = history + [[input_text, None]]
    # use previous transcribed file if exist
    if youtube_command==_youtube_summarize_command:
        transcribe_meta_file = f"{audio_filename}_summarize.meta"
    else:
        transcribe_meta_file = f"{audio_filename}_{task_mode}.meta"            
    output=get_transcribed_meta_file(transcribe_meta_file)
    if output is None:
        audio_text, detected_language = transcribe_audio_to_text(audio_filename, task_mode=task_mode, transcriber_class=transcriber_class)        
        output = transcribe_prepare_output(youtube_command=youtube_command,task_mode=task_mode,language=detected_language, input_text=input_text,
                                       audio_text=audio_text,
                                       video_title=video_title,
                                       audio_filename=audio_filename, history=history)
    t1 = time.perf_counter()-t0
    output["performance"] = t1
    # add prefix text to output
    output["transcribed_output_text"] = prefix_text + output["transcribed_output_text"]        
    # ensure translate output language is always english
    if task_mode.lower()=="translate":
        output["language"] = "en"        
    print(f"transcribe_youtube_audio {payload['video_url']} took {t1}s")
    return output
    # return transcribed_output_text, history, audio_filename, language

def transcribe_youtube_audio(youtube_command: str,
                             input_text: str,
                             history: list = [],
                             prefix_text: str = "transcribed youtube audio:",
                             task_mode: str = "transcribe",
                             transcriber_class: str = "FasterTranscriber"):
    output = transcribe_youtube(youtube_command=youtube_command,
                                input_text=input_text, history=history,
                                prefix_text=prefix_text, task_mode=task_mode, transcriber_class=transcriber_class)
    return output["transcribed_output_text"], output["history"], output["audio_filename"], output["language"]

def filter_commmand_keywords(input_text, history):
    # task_commnad = "filter_commmand_keywords"
    print("filter_commmand_keywords: ", input_text, " | ", history)
    audio_filename = ""
    audio_text = input_text
    task_commmand = ""
    language = "en"
    file_path= get_system_youtube_path()+"/command.txt"
    if input_text is not None:
        # notes:avoid convert youtube link to lowercase
        if _youtube_transcribe_command in input_text:
            audio_text, history, audio_filename, language = transcribe_youtube_audio(_youtube_transcribe_command,
                                                                                     input_text, history,
                                                                                     prefix_text="transcribed text:\n",
                                                                                     task_mode="transcribe"
                                                                                     )
            task_commmand = _youtube_transcribe_command
        elif _youtube_summarize_command in input_text:
            audio_text, history, audio_filename, language = transcribe_youtube_audio(_youtube_summarize_command,
                                                                                     input_text, history,
                                                                                     prefix_text="summarize following in key points:\n",
                                                                                     task_mode="transcribe"
                                                                                     )
            task_commmand = _youtube_summarize_command
        elif _youtube_translate_command in input_text:
            audio_text, history, audio_filename, language = transcribe_youtube_audio(_youtube_translate_command,
                                                                                     input_text, history,
                                                                                     prefix_text="transcribed text in english:\n",
                                                                                     task_mode="translate"
                                                                                     )
            task_commmand = _youtube_translate_command
        elif _summarize_text_file_command in input_text:
            """/summarize_text:https://cdn.serc.carleton.edu/files/teaching_computation/workshop_2018/activities/plain_text_version_declaration_inde.txt"""
            task_commmand = _summarize_text_file_command
            prefix_text = "summarize following text in key points:\n"
            file_contents, history, file_path, task_commmand = load_text_file(
                task_commmand, input_text, prefix_text)
            return file_contents, history, file_path, task_commmand, language
        elif _load_text_file_command in input_text:
            task_commmand = _load_text_file_command
            prefix_text = "learn content below:\n"
            file_contents, history, file_path, task_commmand = load_text_file(
                task_commmand, input_text, prefix_text)
            return file_contents, history, file_path, task_commmand, language
        elif _web_search_command in input_text:
            """/ddgsearch?query=toronto closing in new year eve&max_results=5&backend=lite"""
            task_commmand = _web_search_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _web_news_command in input_text:
            """/ddgnews?query=toronto"""
            task_commmand = _web_news_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _web_video_command in input_text:
            """/ddgvideo?query=toronto"""
            task_commmand = _web_video_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _web_image_command in input_text:
            """/ddgimage?query=toronto"""
            task_commmand = _web_image_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _web_suggest_command in input_text:
            """/ddgsuggest?query=toronto"""
            task_commmand = _web_suggest_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _web_answer_command in input_text:
            """/ddganswer?query=toronto"""
            task_commmand = _web_answer_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _web_translate_command in input_text:
            """/ddgtranslate?query=toronto"""
            task_commmand = _web_translate_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, file_path, task_commmand, language
        elif _image_chat_command in input_text:
            """/imagechat?query=what is this&image='https://cdn11.bigcommerce.com/s-a1x7hg2jgk/images/stencil/1280x1280/products/28260/152885/epson-tm-u300pc-receipt-printer-pos-broken-hinge-on-cover-3.24__38898.1490224087.jpg?c=2?imbypass=on'"""
            task_commmand = _image_chat_command
            result = WebCommandRunner().process(input_text)
            history = history + [[result, None]]
            return result, history, result, task_commmand, language

    return audio_text, history, audio_filename, task_commmand, language

def cleanup_file(localfile: str):
    if localfile is None:
        print(f"cleanup_file: empty local file")
        return
    if os.path.isfile(localfile):
        os.remove(localfile)
        print(f"cleanup_file: success {localfile} file deleted")
    else:
        print("cleanup_file: error: %s file not found" % localfile)

