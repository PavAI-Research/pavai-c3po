# Backstory
In the Star Wars universe, C-3PO was designed as a protocol droid, equipped to aid in matters of etiquette, cultural norms, and language translation. With the ability to communicate in over six million forms of language, C-3PO serves as a robotic diplomat and translator across the vast and varied cultures of Lucas' imagined galaxy. The objective of the project is to refresh the implementation of C-3PO technological capaibilities in 2024 using latest AI technologies.

# Project Objective 
The goal of this project is to update and enhance C-3PO's technological capabilities by incorporating the most recent advancements in artificial intelligence (AI) technology in the year 2024. 

This will involve modernizing his existing abilities and potentially adding new ones, all with the aim of making him even more versatile and effective in his roles as a protocol droid, robotic diplomat, and translator. By utilizing cutting-edge AI technology, we hope to ensure that C-3PO remains a relevant and valuable asset in the ever-evolving landscape of Lucas' imagined galaxy.

### Pre-requisites
1. Python >= 3.10
2. [Poetry 1.8](https://python-poetry.org/docs/#installation)
   see your system installation
3. Clone this repository:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```
3. Install python requirements: 
```bash
pip install -r requirements.txt

On Windows add:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
```
Also install phonemizer and espeak if you want to run the demo:
```bash
pip install phonemizer
sudo apt-get install espeak-ng

### Important Configurations
In [config.yml](https://github.com/yl4579/StyleTTS2/blob/main/Configs/config.yml), there are a few important configurations to take care of:
- `OOD_data`: The path for out-of-distribution texts for SLM adversarial training. The format should be `text|anything`.
- `min_length`: Minimum length of OOD texts for training. This is to make sure the synthesized speech has a minimum length.
- 
## llamacpp
CMAKE_ARGS="-DLLAMA_CUBLAS=on" poetry run pip install llama-cpp-python==0.2.27 --force-reinstall --no-cache-dir

### Common Issues
[@Kreevoz](https://github.com/Kreevoz) has made detailed notes on common issues in finetuning, with suggestions in maximizing audio quality: [#81](https://github.com/yl4579/StyleTTS2/discussions/81). Some of these also apply to training from scratch. [@IIEleven11](https://github.com/IIEleven11) has also made 

## TODO
- [x] Training and inference demo code for single-speaker models (LJSpeech)
- [x] Test training code for multi-speaker models (VCTK and LibriTTS)
- [x] Finish demo code for multispeaker model and upload pre-trained models
- [x] Add a finetuning script for new speakers with base pre-trained multispeaker


## References
- [archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [rishikksh20/iSTFTNet-pytorch](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [nii-yamagishilab/project-NN-Pytorch-scripts/project/01-nsf](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf)

## License

Code: MIT License

Pre-Trained Models: Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.

# -------------------------------------
# Environment
# -------------------------------------
# tested version gradio version: 4.7.1
# print(f"use gradio version: {gr.__version__}")
# print(f"use torch version: {torch.__version__}")
# # pip install gradio==4.7.1

# pip install faster-whisper==0.10.0
