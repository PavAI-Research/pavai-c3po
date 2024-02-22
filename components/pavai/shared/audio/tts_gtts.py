#!pip install gTTS

from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
from io import BytesIO
from pydub.playback import play
from pydub import AudioSegment
from gtts import gTTS
import os

def extract_text_from_subtitles(subtitle_file: str = 'subtitles.srt'):
    # Open and read the SRT file
    with open(subtitle_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    # Extract the text from the subtitle file
    text = ''
    for i, line in enumerate(lines):
        if i % 4 == 2:  # The text is on the third line of each block
            text += line.strip() + ' '  # Add a space between each subtitle block
            print(line.strip() + ' ')
    return text

def convert_to_audio(text: str, output_file: str = "example.mp3", output_dir: str = "./workspace/audio", language: str = "en"):
    # Create a TTS object and save the audio to a file
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_path = output_dir+"/"+output_file
    tts = gTTS(text=text, lang='en')
    tts.save(file_path)
    return file_path

def slice_text_into_chunks(input_text: str, chunk_size: int = 150):
    if len(input_text) < chunk_size:
        return [input_text]
    else:
        K = int(len(input_text)/chunk_size)
        #print(f"The original string is: {str(input_text)} size of {len(input_text)} in chunks: {K}")
        chnk_len = len(input_text) // K
        result = []
        for idx in range(0, len(input_text), chnk_len):
            result.append(input_text[idx: idx + chnk_len])
        #print(f"The K chunked list {str(result)}")
        return result

def text_to_speech_gtts(text, autoplay=False):
    #print("text_to_speech: ", text)
    if (len(str(text).strip())==0):
        print("text_to_speech_gtts: received empty text.")
        return    
    text_chunks = slice_text_into_chunks(text)    
    for chunk in text_chunks:        
        if (len(str(chunk).strip())==0):
            return
        out_file = 'gtts_text_to_speech.mp3'
        if text is None or len(chunk) == 0:
            return None
        # Initialize gTTS with the text to convert removed: lang='en',
        tts = gTTS(chunk, slow=False, tld='com')
        tts.save(out_file)
        if autoplay:
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            audio_data = AudioSegment.from_file(audio_bytes, format="mp3")
            play(audio_data)
    return out_file
    # linux Play the audio file
    # os.system('afplay ' + speech_file)