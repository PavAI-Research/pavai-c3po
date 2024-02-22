from __future__ import annotations
from dotenv import dotenv_values
import time
import warnings
import logging
from rich import print, pretty, console
from rich.logging import RichHandler
system_config = dotenv_values("env_config")
logging.basicConfig(level=logging.INFO, format="%(message)s",datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True,tracebacks_show_locals=True)])
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
pretty.install()
import os 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
logger.info(os.getcwd())

import src.shared.commands as commands


def test_web_commands():
    #
    # Web Search Command Test
    #
    input_text = "/ddgsearch?query=toronto closing in new year eve&max_results=5&backend=lite"
    results = commands.WebCommandRunner().process(input_text)
    print(results)
    print(type(results))

    time.sleep(3)
    input_text = "/ddgnews?query=IBM&max_results=5&backend=api"
    query_result = commands.WebCommandRunner().process(input_text=input_text)
    print(query_result)

    #time.sleep(3)
    input_text = "/ddganswer?query=sun"
    query_result = commands.WebCommandRunner().process(input_text=input_text)
    print(query_result)

    #time.sleep(3)
    input_text = "/ddgimage?query=butterfly&max_results=5&backend=api&size=small&color=color&license_image=Public"
    query_result = commands.WebCommandRunner().process(input_text=input_text)
    print(query_result)

    time.sleep(3)
    input_text = "/ddgvideo?query=butterfly&max_results=5&backend=api&resolution=standard&duration=short&license_videos=youtube"
    query_result = commands.WebCommandRunner().process(input_text=input_text)
    print(query_result)

    time.sleep(3)
    input_text = "/ddgtranslate?query=butterfly&to_lang=fr"
    query_result = commands.WebCommandRunner().process(input_text=input_text)
    print(query_result)

    time.sleep(3)
    input_text = "/ddgsuggest?query=butterfly"
    query_result = commands.WebCommandRunner().process(input_text=input_text)
    print(query_result)


def test_youtube_commands():
    # transcribe youtube video
    # restricted_input_text="/ytranscribe:https://www.youtube.com/watch?v=p9YdklNnzHI"
    # 2.10 hours - AMD Presents: Advancing AI  (137.61 seconds)
    # input_text ="/ytranscribe:https://www.youtube.com/watch?v=tfSZqjxsr0M&t=1551s"
    input_text = "/ytranscribe:https://www.youtube.com/watch?v=JmS6zsR3UBs"

    # test-1 download only
    audio_text = ""
    video_url = input_text.split(commands._youtube_transcribe_command)[1].strip()
    print(f"youtube_video: {video_url}")
    video_id = video_url.split("watch?v=")[1].strip()
    local_storage = commands.get_system_youtube_path()
    payload = {
        "video_url": video_url,
        "video_id": video_id,
        "local_storage": local_storage
    }
    output = commands.YoutubeAudioDownloadCommand(payload).execute()
    print(output["audio_file"])
    print(output["title"])
    print(output["status"])
    # cleanup_file(output["audio_file_meta"])
    # cleanup_file(output["audio_file"])

    # test-2 download and transcribe
    transcriber_class = "FasterTranscriber"
    transcribed_output_text, history, audio_filename, language = commands.transcribe_youtube_audio(
        youtube_command=commands._youtube_transcribe_command,
        history=[],
        task_mode="transcribe",
        input_text=input_text, transcriber_class=transcriber_class)
    print(transcribed_output_text, audio_filename, language)
    #print(transcribed_output_text, history, audio_filename, language)    

    # test-3 repeat - get local copy 
    transcribed_output_text, history, audio_filename, language = commands.transcribe_youtube_audio(
        youtube_command=commands._youtube_transcribe_command,
        history=[],
        task_mode="transcribe",        
        input_text=input_text, transcriber_class=transcriber_class)
    print(transcribed_output_text, audio_filename, language)
    #print(transcribed_output_text, history, audio_filename, language)

    # test-4 download and translate
    input_text = "/ytranslate:https://www.youtube.com/watch?v=JmS6zsR3UBs"
    transcriber_class="FasterTranscriber" ##"DistrilledTranscriber"
    translated_output_text, history, audio_filename, language=commands.transcribe_youtube_audio(
        youtube_command=commands._youtube_translate_command,
        history=[],
        input_text=input_text,
        task_mode="translate",
        transcriber_class=transcriber_class)
    print(transcribed_output_text, audio_filename, language)    
    #print(translated_output_text, history, audio_filename, language)
    
    # test-5 repeat - get local copy
    input_text = "/ytranslate:https://www.youtube.com/watch?v=JmS6zsR3UBs"
    transcriber_class="FasterTranscriber" ##"DistrilledTranscriber"
    translated_output_text, history, audio_filename, language=commands.transcribe_youtube_audio(
        youtube_command=commands._youtube_translate_command,
        history=[],
        input_text=input_text,
        task_mode="translate",
        transcriber_class=transcriber_class)
    print(transcribed_output_text, audio_filename, language)
    #print(translated_output_text, history, audio_filename, language)

    # test-6 summarize
    input_text = "/ysummarize:https://www.youtube.com/watch?v=JmS6zsR3UBs"
    transcriber_class="FasterTranscriber" ##"DistrilledTranscriber"
    translated_output_text, history, audio_filename, language=commands.transcribe_youtube_audio(
        youtube_command=commands._youtube_summarize_command,
        history=[],
        input_text=input_text,
        task_mode="translate",
        transcriber_class=transcriber_class)
    print(transcribed_output_text, audio_filename, language)
    #print(translated_output_text, history, audio_filename, language)
    #cleanup_file(audio_filename)

if __name__ == "__main__":
    test_web_commands()
    # test_youtube_commands()