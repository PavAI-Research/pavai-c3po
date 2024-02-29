## Install packages and download models
## -------------------------------------
# %%shell
# git clone https://github.com/yl4579/StyleTTS2.git
# cd StyleTTS2
# pip install SoundFile torchaudio munch torch pydub pyyaml librosa nltk matplotlib accelerate transformers phonemizer einops einops-exts tqdm typing-extensions git+https://github.com/resemble-ai/monotonic_align.git
# sudo apt-get install espeak-ng
# git-lfs clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS
# mv StyleTTS2-LibriTTS/Models .
# mv StyleTTS2-LibriTTS/reference_audio.zip .
# unzip reference_audio.zip
# mv reference_audio Demo/reference_audio

# convert input audio to wav
# ffmpeg -i inputfile.flac output.wav

## Load models
import torch
import random
import numpy as np
import nltk
#nltk.download('punkt')

torch.manual_seed(200)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(200)
np.random.seed(200)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import sounddevice as sd
import phonemizer

import sys, os
# sys.path.append("./styletts2") 
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .models import load_F0_models,load_ASR_models,build_model,load_checkpoint
#from models import *
from .utils import maximum_path,get_data_path_list,length_to_mask,log_norm,get_image,recursive_munch,log_print
from .text_utils import TextCleaner
from .Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from .Utils.PLBERT.util import load_plbert
from pavai.shared.styletts2.download_models import get_styletts2_model_files

## System Configuration
StyleTTS2_LANGUAGE="en-us"
# StyleTTS2_CONFIG_FILE="StyleTTS2/Models/LibriTTS/config.yml"
# StyleTTS2_MODEL_FILE="StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth"

StyleTTS2_CONFIG_FILE="resources/models/styletts2/Models/LibriTTS/config.yml"
StyleTTS2_MODEL_FILE="resources/models/styletts2/Models/LibriTTS/epochs_2nd_00020.pth"


textclenaer = TextCleaner()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

# load phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language=StyleTTS2_LANGUAGE, preserve_punctuation=True,  with_stress=True)

try:
    config = yaml.safe_load(open(StyleTTS2_CONFIG_FILE))
except:
    print("StyleTTS2_CONFIG_FILE Not Found!")
    get_styletts2_model_files()
    ## read file downloaded
    config = yaml.safe_load(open(StyleTTS2_CONFIG_FILE))

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

#  Load models
params_whole = torch.load(StyleTTS2_MODEL_FILE, map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        ##print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)

def inference(text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

def LFinference(text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
  text = text.strip()
  ps = global_phonemizer.phonemize([text])
  ps = word_tokenize(ps[0])
  ps = ' '.join(ps)
  ps = ps.replace('``', '"')
  ps = ps.replace("''", '"')

  tokens = textclenaer(ps)
  tokens.insert(0, 0)
  tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

  with torch.no_grad():
      input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
      text_mask = length_to_mask(input_lengths).to(device)

      t_en = model.text_encoder(tokens, input_lengths, text_mask)
      bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
      d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

      s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                        embedding=bert_dur,
                                        embedding_scale=embedding_scale,
                                          features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)

      if s_prev is not None:
          # convex combination of previous and current style
          s_pred = t * s_prev + (1 - t) * s_pred

      s = s_pred[:, 128:]
      ref = s_pred[:, :128]

      ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
      s = beta * s + (1 - beta)  * ref_s[:, 128:]

      s_pred = torch.cat([ref, s], dim=-1)

      d = model.predictor.text_encoder(d_en,
                                        s, input_lengths, text_mask)

      x, _ = model.predictor.lstm(d)
      duration = model.predictor.duration_proj(x)

      duration = torch.sigmoid(duration).sum(axis=-1)
      pred_dur = torch.round(duration.squeeze()).clamp(min=1)


      pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
      c_frame = 0
      for i in range(pred_aln_trg.size(0)):
          pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
          c_frame += int(pred_dur[i].data)

      # encode prosody
      en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
      if model_params.decoder.type == "hifigan":
          asr_new = torch.zeros_like(en)
          asr_new[:, :, 0] = en[:, :, 0]
          asr_new[:, :, 1:] = en[:, :, 0:-1]
          en = asr_new

      F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

      asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
      if model_params.decoder.type == "hifigan":
          asr_new = torch.zeros_like(asr)
          asr_new[:, :, 0] = asr[:, :, 0]
          asr_new[:, :, 1:] = asr[:, :, 0:-1]
          asr = asr_new

      out = model.decoder(asr,
                              F0_pred, N_pred, ref.squeeze().unsqueeze(0))


  return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

def STinference(text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_text = ref_text.strip()
    ps = global_phonemizer.phonemize([ref_text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)

    ref_tokens = textclenaer(ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)


    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                                          embedding=bert_dur,
                                          embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                             num_steps=diffusion_steps).squeeze(1)


        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en,
                                         s, input_lengths, text_mask)

        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)

        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr,
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))


    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

## ------------------
## Synthesize speech
## ------------------

def librispeech(text:str, compute_style:any,voice:str='',
                alpha=0.0, beta=0.5,diffusion_steps:int=5, 
                embedding_scale:int=1, blocking_flag:bool=True,
                samplerate:int=24000,autoplay:bool=True):
    if isinstance(compute_style, str):
        # input wav file
        ref_s = compute_style(compute_style)
    else:
        # input torch.Tensor
        ref_s=compute_style
    start = time.time()
    wav = inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
    rtf = (time.time() - start) / (len(wav) / 24000)
    ##print(f"librispeech {voice} rtf took {rtf:5f}")
    if autoplay:
        sd.play(wav,samplerate=samplerate,blocking=blocking_flag)
    return wav

def test_libris_speech():
    text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
    #ref_s1 = compute_style("resources/models/styletts2/reference_audio/Gavin.wav")
    ref_s2 = compute_style("resources/models/styletts2/reference_audio/Jane.wav")
    ref_s3 = compute_style("resources/models/styletts2/reference_audio/Me1.wav")
    librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0.3, beta=0.7, diffusion_steps=10    )
    librispeech(text=text,compute_style=ref_s3, voice='Me1',alpha=0.3, beta=0.7, diffusion_steps=10)

"""MAIN"""
if __name__ == "__main__":
    ## Basic synthesis (5 diffusion steps)
    text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
    #ref_s1 = compute_style("resources/models/styletts2/reference_audio/Gavin.wav")
    ref_s2 = compute_style("resources/models/styletts2/reference_audio/Jane.wav")
    ref_s3 = compute_style("resources/models/styletts2/reference_audio/Me1.wav")
    # ref_s4 = compute_style("resources/models/styletts2/reference_audio/Me2.wav")
    # ref_s5 = compute_style("resources/models/styletts2/reference_audio/Me3.wav")
    # ref_s6 = compute_style("resources/models/styletts2/reference_audio/Vinay.wav")
    # ref_s7 = compute_style("resources/models/styletts2/reference_audio/Nima.wav")
    # ref_s8 = compute_style("resources/models/styletts2/reference_audio/Yinghao.wav")
    # ref_s9 = compute_style("resources/models/styletts2/reference_audio/Keith.wav")
    # ref_s10 = compute_style("resources/models/styletts2/reference_audio/May.wav")
    # ref_s11 = compute_style("resources/models/styletts2/reference_audio/June.wav")
    # ## Default setting (alpha = 0.3, beta=0.7)

    # librispeech(text=text,compute_style=ref_s1, voice='Gavin',alpha=0.3, beta=0.7, diffusion_steps=10)
    librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0.3, beta=0.7, diffusion_steps=10)
    librispeech(text=text,compute_style=ref_s3, voice='Me1',alpha=0.3, beta=0.7, diffusion_steps=10)
    #librispeech(text=text,compute_style=ref_s4, voice='Me2',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s5, voice='Me3',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s6, voice='Vinay',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s7, voice='Nima',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s8, voice='Yinghao',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s9, voice='Keith',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s10, voice='May',alpha=0.3, beta=0.7, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s11, voice='June',alpha=0.3, beta=0.7, diffusion_steps=10)

    # # Less diverse setting (alpha = 0.1, beta=0.3)
    # # alpha = 0 and beta = 0 the synthesized speech sounds the most siimlar to the reference speaker, 
    # librispeech(text=text,compute_style=ref_s1, voice='Gavin',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s3, voice='Me1',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s4, voice='Me2',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s5, voice='Me3',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s6, voice='Vinay',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s7, voice='Nima',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s8, voice='Yinghao',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s9, voice='Keith',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s10, voice='May',alpha=0, beta=0, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s11, voice='June',alpha=0, beta=0, diffusion_steps=10)

    # # different
    # # More diverse setting (alpha = 0.5, beta=0.95)
    # # Extreme setting (alpha = 1, beta=1)
    # librispeech(text=text,compute_style=ref_s1, voice='Gavin',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s3, voice='Me1',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s4, voice='Me2',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s5, voice='Me3',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s6, voice='Vinay',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s7, voice='Nima',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s8, voice='Yinghao',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s9, voice='Keith',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s10, voice='May',alpha=1, beta=1, diffusion_steps=10)
    # librispeech(text=text,compute_style=ref_s11, voice='June',alpha=1, beta=1, diffusion_steps=10)


    # reference_dicts = {}
    # reference_dicts['696_92939'] = "/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/696_92939_000016_000006.wav"
    # reference_dicts['1789_142896'] = "/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/1789_142896_000022_000005.wav"
        
    #sd.play("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/696_92939_000016_000006.wav")
    # noise = torch.randn(1,1,256).to(device)
    # for k, path in reference_dicts.items():
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #     #import IPython.display as ipd
    #     print(k + ' Synthesized:')
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True)    
    #     print('Reference:', path)
    #     #display(ipd.Audio(path, rate=24000, normalize=False))
    #     #sd.play(path,samplerate=24000,blocking=True)

    # noise = torch.randn(1,1,256).to(device)
    # for k, path in reference_dicts.items():
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #     #import IPython.display as ipd
    #     print(k + ' Synthesized:')
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True)        
    #     print(k + ' Reference:')
        #display(ipd.Audio(path, rate=24000, normalize=False))

    ## Basic synthesis (5 diffusion steps, unseen speakers)
    # reference_dicts = {}
    # # format: (path, text)
    # reference_dicts['1221-135767'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/1221-135767-0014.wav", "Yea, his honourable worship is within, but he hath a godly minister or two with him, and likewise a leech.")
    # reference_dicts['5639-40744'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/5639-40744-0020.wav", "Thus did this humane and right minded father comfort his unhappy daughter, and her mother embracing her again, did all she could to soothe her feelings.")
    # reference_dicts['908-157963'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/908-157963-0027.wav", "And lay me down in my cold bed and leave my shining lot.")
    # reference_dicts['4077-13754'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/4077-13754-0000.wav", "The army found the people in poverty and left them in comparative wealth.")

    # noise = torch.randn(1,1,256).to(device)
    # for k, v in reference_dicts.items():
    #     path, text = v
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #     import IPython.display as ipd
    #     print(k + ' Synthesized: ' + text)
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True)    
    #     #print(k + ' Reference:')
    #     #display(ipd.Audio(path, rate=24000, normalize=False))

    ## Speech expressiveness
    ## ---------------------
    ## The following section recreates the samples shown in Section 6 of the demo page. The speaker reference used is 1221-135767-0014.wav, which is unseen during training.
    #ref_s = compute_style("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/1221-135767-0014.wav")
    #ref_s = compute_style("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/mc_voice2.wav")

    # texts = {}
    # texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    # texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    # texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    # texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"
    # texts['okay']="Addressing your concerns about keeping the resulting audio intact, FLAC is a lossless format and decoding it to raw PCM stored in a WAV file will keep perfect fidelity."

    # for k,v in texts.items():
    #     wav = inference(v, ref_s, diffusion_steps=10, alpha=0.3, beta=0.7, embedding_scale=2)
    #     print(k + ": ")
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True)  

    # #ref_s = compute_style("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/Gavin.wav")
    # ref_s = compute_style("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/mc_voice3.wav")
    # for k,v in texts.items():
    #     wav = inference(v, ref_s, diffusion_steps=10, alpha=0.3, beta=0.7, embedding_scale=2)
    #     print(k + ": ")
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True) 

    # # unseen speaker
    # path = "/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/1221-135767-0014.wav"
    # ref_s = compute_style(path)

    # text = "How much variation is there?"
    # for _ in range(5):
    #     wav = inference(text, ref_s, diffusion_steps=10, alpha=0.3, beta=0.7, embedding_scale=1)
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True)

    # ## Zero-shot speaker adaptation
    # ## Synthesize in your own voice
    # ## Run the following cell to record your voice for 5 seconds. Please keep speaking to have the best effect.

    # reference_dicts = {}
    # # # format: (path, text)
    # reference_dicts['3'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/1221-135767-0014.wav", "As friends thing I definitely I've got more male friends.")
    # reference_dicts['4'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/Gavin.wav", "Everything is run by computer but you got to know how to think before you can do a computer.")
    # reference_dicts['5'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/mc_voice3.wav", "Then out in LA you guys got a whole another ball game within California to worry about.")
        
    # noise = torch.randn(1,1,256).to(device)
    # for k, v in reference_dicts.items():
    #     path, text = v
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.0, beta=0.5, diffusion_steps=5, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #    # import IPython.display as ipd
    #     print(f'Synthesized: {k}' + text)
    #     sd.play(wav,samplerate=24000,blocking=True)
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     #print('Reference:')
    #     #display(ipd.Audio(path, rate=24000, normalize=False))

    # ## Speaker’s Emotion Maintenance
    # # Since we want to maintain the emotion in the speaker (prosody), we set beta = 0.1 to make the speaker as closer to the reference as possible while having some diversity thruogh the slight timbre change.

    # reference_dicts = {}
    # # format: (path, text)
    # reference_dicts['Anger'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/anger.wav", "We have to reduce the number of plastic bags.")
    # reference_dicts['Sleepy'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/sleepy.wav", "We have to reduce the number of plastic bags.")
    # reference_dicts['Amused'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/amused.wav", "We have to reduce the number of plastic bags.")
    # reference_dicts['Disgusted'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/disgusted.wav", "We have to reduce the number of plastic bags.")
    
    # noise = torch.randn(1,1,256).to(device)
    # for k, v in reference_dicts.items():
    #     path, text = v
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.3, beta=0.1, diffusion_steps=10, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #     import IPython.display as ipd
    #     print(k + ' Synthesized: ' + text)
    #     sd.play(wav,samplerate=24000,blocking=True)    
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     #print(k + ' Reference:')
    #     #display(ipd.Audio(path, rate=24000, normalize=False))



    # start = time.time()
    # noise = torch.randn(1,1,256).to(device)
    # wav = inference(text, noise, diffusion_steps=5, embedding_scale=1)
    # rtf = (time.time() - start) / (len(wav) / 24000)
    # print(f"RTF = {rtf:5f}")
    # #import IPython.display as ipd
    # #display(ipd.Audio(wav, rate=24000))
    # sd.play(wav,samplerate=24000,blocking=True)

    # ## With higher diffusion steps (more diverse)
    # ## Since the sampler is ancestral, the higher the stpes, the more diverse the samples are, with the cost of slower synthesis speed.

    # start = time.time()
    # noise = torch.randn(1,1,256).to(
    # reference_dicts = {}
    # # format: (path, text)
    # reference_dicts['3'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/3.wav", "As friends thing I definitely I've got more male friends.")
    # reference_dicts['4'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/4.wav", "Everything is run by computer but you got to know how to think before you can do a computer.")
    # reference_dicts['5'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/5.wav", "Then out in LA you guys got a whole another ball game within California to worry about.")
        
    # noise = torch.randn(1,1,256).to(device)
    # for k, v in reference_dicts.items():
    #     path, text = v
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.0, beta=0.5, diffusion_steps=5, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #    # import IPython.display as ipd
    #     print('Synthesized: ' + text)
    #     sd.play(wav,samplerate=24000,blocking=True)
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     #print('Reference:')
    #     #display(ipd.Audio(path, rate=24000, normalize=False))

    ## Style Transfer
    # path = "/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/1221-135767-0014.wav"
    # s_ref = compute_style(path)

    # ref_texts = {}
    # ref_texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    # ref_texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    # ref_texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    # ref_texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"
    
    # text = "Yea, his honourable worship is within, but he hath a godly minister or two with him, and likewise a leech."
    # for k,v in ref_texts.items():
    #     wav = STinference(text, s_ref, v, diffusion_steps=10, alpha=0.5, beta=0.9, embedding_scale=1.5)
    #     print(k + ": Style Transfer")
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True)
        

    ## Speaker’s Emotion Maintenance
    # Since we want to maintain the emotion in the speaker (prosody), we set beta = 0.1 to make the speaker as closer to the reference as possible while having some diversity thruogh the slight timbre change.

    # reference_dicts = {}
    # # format: (path, text)
    # reference_dicts['Anger'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/anger.wav", "We have to reduce the number of plastic bags.")
    # reference_dicts['Sleepy'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/sleepy.wav", "We have to reduce the number of plastic bags.")
    # reference_dicts['Amused'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/amused.wav", "We have to reduce the number of plastic bags.")
    # reference_dicts['Disgusted'] = ("/home/pop/development/mclab/realtime/StyleTTS2-LibriTTS/reference_audio/disgusted.wav", "We have to reduce the number of plastic bags.")
    
    # noise = torch.randn(1,1,256).to(device)
    # for k, v in reference_dicts.items():
    #     path, text = v
    #     ref_s = compute_style(path)
    #     start = time.time()
    #     wav = inference(text, ref_s, alpha=0.3, beta=0.1, diffusion_steps=10, embedding_scale=1)
    #     rtf = (time.time() - start) / (len(wav) / 24000)
    #     print(f"RTF = {rtf:5f}")
    #     import IPython.display as ipd
    #     print(k + ' Synthesized: ' + text)
    #     sd.play(wav,samplerate=24000,blocking=True)    
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     #print(k + ' Reference:')
    #     #display(ipd.Audio(path, rate=24000, normalize=False))


    # wav = inference(text, noise, diffusion_steps=10, embedding_scale=1)
    # rtf = (time.time() - start) / (len(wav) / 24000)
    # print(f"RTF = {rtf:5f}")
    # #import IPython.display as ipd
    # #display(ipd.Audio(wav, rate=24000))
    # sd.play(wav)

    # texts = {}
    # texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    # texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    # texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    # texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"

    # for k,v in texts.items():
    #     noise = torch.randn(1,1,256).to(device)
    #     wav = inference(v, noise, diffusion_steps=10, embedding_scale=2) # embedding_scale=2 for more pronounced emotion
    #     print(k + ": ")
    #     #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     sd.play(wav,samplerate=24000,blocking=True) 

    # ## Long-form generation
    # ## --------------------
    # passage = """
    # If the supply of fruit is greater than the family needs, it may be made a source of income by sending the fresh fruit to the market if there is one near enough, or by preserving, canning, and making jelly for sale. To make such an enterprise a success the fruit and work must be first class. 
    # There is magic in the word "Homemade," when the product appeals to the eye and the palate; but many careless and incompetent people have found to their sorrow that this word has not magic enough to float inferior goods on the market. 
    # As a rule large canning and preserving establishments are clean and have the best appliances, and they employ chemists and skilled labor. The home product must be very good to compete with the attractive goods that are sent out from such establishments. 
    # Yet for first-class homemade products there is a market in all large cities. 
    # All first-class grocers have customers who purchase such goods.
    # """# @param {type:"string"}
    # sentences = passage.split('.') # simple split by comma
    # wavs = []
    # s_prev = None
    # for text in sentences:
    #     if text.strip() == "": continue
    #     text += '.' # add it back
    #     noise = torch.randn(1,1,256).to(device)
    #     wav, s_prev = LFinference(text, s_prev, noise, alpha=0.7, diffusion_steps=10, embedding_scale=1.5)
    #     wavs.append(wav)
    # #display(ipd.Audio(np.concatenate(wavs), rate=24000, normalize=False))
    # for wav in wavs:
    #     sd.play(wav,samplerate=24000,blocking=True)

