#!/bin/bash 
## pip install piper-tts vosk
echo "start speech synthesizer API server"
python src/shared/audio/vosk_server.py/vosk_server.py
