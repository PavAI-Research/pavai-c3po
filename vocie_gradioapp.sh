#!/bin/bash

## gradio mode
gradio bases/pavai/vocei_web/vocei_app.py

## fastapi 
## uvicorn "vocei:app" --host "0.0.0.0" --port 7860 --reload

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--name", type=str, default="User")
# args, unknown = parser.parse_known_args()

