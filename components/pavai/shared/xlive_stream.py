import gradio as gr
from transformers import pipeline
import numpy as np

## tested version gradio version: 4.7.1
print(f"gradio version: {gr.__version__}")
## gradio version: 4.7.1

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=['microphone','upload'])],
    ["state", "text"],
    live=True,
)

demo.launch()
