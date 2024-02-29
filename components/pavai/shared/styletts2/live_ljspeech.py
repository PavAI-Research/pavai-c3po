try: 
    import nltk
except:
    nltk.download('punkt')
    import nltk
# load packages
import gc
import time
import yaml
import random
import torch
import torchaudio
import numpy as np
from nltk.tokenize import word_tokenize
import sounddevice as sd
import phonemizer
from scipy.io.wavfile import write, read
#from utils import *
from pavai.shared.styletts2.models import load_F0_models,load_ASR_models,build_model
#,load_checkpoint
from pavai.shared.styletts2.utils import length_to_mask,recursive_munch
#,maximum_path,get_data_path_list,log_norm,get_image,log_print
from pavai.shared.styletts2.text_utils import TextCleaner
from pavai.shared.styletts2.Utils.PLBERT.util import load_plbert
from pavai.shared.styletts2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from typing import Any, Dict
import traceback
# import torch.nn.functional as F
# from munch import Munch
# from torch import nn
# import librosa
#  List,Union, Optional, Sequence, Mapping, Literal,
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(
                MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()

class LJSpeech(Singleton):

    def __init__(self,device:str=None, style_config:str=None, model_config:str=None):
        torch.manual_seed(100)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        random.seed(100)
        np.random.seed(100)

        ## System Configuration
        self.StyleTTS2_LANGUAGE="en-us"
        self.StyleTTS2_CONFIG_FILE="resources/models/styletts2/Models/LJSpeech/config.yml"
        self.StyleTTS2_MODEL_FILE="resources/models/styletts2/Models/LJSpeech/epoch_2nd_00100.pth"

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device = 'cpu' 
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean = -4 
        self.std = 4

        ## Load config
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language=self.StyleTTS2_LANGUAGE, preserve_punctuation=True,  with_stress=True)
        self.config = yaml.safe_load(open(self.StyleTTS2_CONFIG_FILE))

        # load pretrained ASR model
        self.ASR_config = self.config.get('ASR_config', False)
        self.ASR_path = self.config.get('ASR_path', False)
        self.text_aligner = load_ASR_models(self.ASR_path, self.ASR_config)

        # load pretrained F0 model
        self.F0_path = self.config.get('F0_path', False)
        self.pitch_extractor = load_F0_models(self.F0_path)

        # load BERT model
        self.BERT_path = self.config.get('PLBERT_dir', False)
        self.plbert = load_plbert(self.BERT_path)

        self.model = build_model(recursive_munch(self.config['model_params']), self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        ## Load models
        self.params_whole = torch.load(self.StyleTTS2_MODEL_FILE, map_location='cpu')
        self.params = self.params_whole['net']

        for key in self.model:
            if key in self.params:
                try:
                    self.model[key].load_state_dict(self.params[key])
                except:
                    from collections import OrderedDict
                    state_dict = self.params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

        self.textclenaer = TextCleaner()

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )

        # def preprocess(self,wave):
        #     wave_tensor = torch.from_numpy(wave).float()
        #     mel_tensor = self.to_mel(wave_tensor)
        #     mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        #     return mel_tensor

        # def compute_style(ref_dicts):
        #     reference_embeddings = {}
        #     for key, path in ref_dicts.items():
        #         wave, sr = librosa.load(path, sr=24000)
        #         audio, index = librosa.effects.trim(wave, top_db=30)
        #         if sr != 24000:
        #             audio = librosa.resample(audio, sr, 24000)
        #         mel_tensor = self.preprocess(audio).to(self.evice)

        #         with torch.no_grad():
        #             ref = self.model.style_encoder(mel_tensor.unsqueeze(1))
        #         reference_embeddings[key] = (ref.squeeze(1), audio)
            
        #     return reference_embeddings

    ## Synthesize speech
    def inference(self,text, noise, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise,
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()

    def LFinference(self,text, s_prev, noise, alpha=0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
            s_pred = self.sampler(noise,embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,embedding_scale=embedding_scale).squeeze(0)
            if s_prev is not None:
                # convex combination of previous and current style
                s_pred = alpha * s_prev + (1 - alpha) * s_pred

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy(), s_pred

    ## synthesize a text
    ## Basic synthesis (5 diffusion steps)
    def ljspeech(self,text:str, 
                diffusion_steps:int=random.randint(1, 10), 
                embedding_scale:int=random.randint(1, 2), 
                blocking_flag:bool=True, autoplay:bool=True):
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t0=time.perf_counter()
        start = time.time()
        noise = torch.randn(1,1,256).to(self.device)
        wav = self.inference(text, noise, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
        rtf = (time.time() - start) / (len(wav) / 24000)
        t1=time.perf_counter()    
        print(f"rtf inference took = {rtf:5f} in {t1-t0:.2f} seconds")
        if autoplay:
            sd.play(wav,samplerate=24000,blocking=blocking_flag)
        return wav

    def wipe_memory(self,objects:list=[]): # DOES WORK
        try:
            for obj in objects:
                del obj
            collected = gc.collect()
            print("Garbage collector: collected","%d objects." % collected)
            torch.cuda.empty_cache()
        except:
            pass

    def ljspeech_v2(self,text:str, alpha:int=0.7,
                    diffusion_steps:int=random.randint(1, 10), 
                    embedding_scale:int=random.randint(1, 2), 
                    samplerate=24000,
                    output_audiofile="workspace/temp/ljspeech_v2.wav",
                    blocking_flag:bool=True, autoplay:bool=True):
        """
        ## With higher diffusion steps (more diverse) with the cost of slower synthesis speed.
        ## choose a random generated number to emulate human like expressiveness    
        """
        try:
            t0=time.perf_counter()
            #device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sentences = text.split('.') # simple split by comma
            wavs = []
            s_prev = None
            for text in sentences:
                start = time.time()                
                if text.strip() == "": continue
                text += '.' # add it back
                noise = torch.randn(1,1,256).to(self.device)
                wav, s_prev = self.LFinference(text, s_prev, noise, alpha=alpha, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
                wavs.append(wav)
                rtf = (time.time() - start) / (len(wav) / samplerate)
                t1=time.perf_counter()    
                print(f"rtf inference took = {rtf:5f} elapsed {t1-t0:.2f} seconds")            
            #if autoplay:
            #    for wav in wavs:
            #        sd.play(wav,samplerate=samplerate,blocking=blocking_flag)    
            ## combine approach-1
            # combined=[]
            # for wav in wavs:
            #     combined=np.append(combined,wav)
            ## combine approach-2
            combined= np.concatenate(wavs) 
            scaled = np.int16(combined / np.max(np.abs(combined)) * 32767)
            write(output_audiofile, samplerate, scaled)
            if autoplay:
                sd.play(scaled,samplerate=samplerate,blocking=blocking_flag)       
            ## clean up
            self.wipe_memory(objects=[combined,text,wavs,sentences])            
        except Exception as e:
            print("Exeption occurred ", e.args)
            print(traceback.format_exc())
        finally:
            self.wipe_memory()  
        return output_audiofile

    def test_lj_speech(self):
        text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
        self.ljspeech(text=text,diffusion_steps=random.randint(3, 10), embedding_scale=random.randint(1, 2))
        self.ljspeech(text=text,diffusion_steps=random.randint(3, 10), embedding_scale=random.randint(1, 2))    

    def test_lj_speech_v2(self):
        passage = """
        If the supply of fruit is greater than the family needs, it may be made a source of income by sending the fresh fruit to the market if there is one near enough, or by preserving, canning, and making jelly for sale. To make such an enterprise a success the fruit and work must be first class. 
        There is magic in the word "Homemade," when the product appeals to the eye and the palate; but many careless and incompetent people have found to their sorrow that this word has not magic enough to float inferior goods on the market. 
        As a rule large canning and preserving establishments are clean and have the best appliances, and they employ chemists and skilled labor. The home product must be very good to compete with the attractive goods that are sent out from such establishments. 
        Yet for first-class homemade products there is a market in all large cities. 
        All first-class grocers have customers who purchase such goods."""
        audiofile = self.ljspeech_v2(text=passage,diffusion_steps=random.randint(3, 10), embedding_scale=random.randint(1, 2))
        #ljspeech_v2(text=passage,diffusion_steps=random.randint(3, 10), embedding_scale=random.randint(1, 2))    

## demo--1
# text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
# ljspeech(text=text,diffusion_steps=5, embedding_scale=1)  

# ## With higher diffusion steps (more diverse)
# ## Since the sampler is ancestral, the higher the stpes, the more diverse the samples are, with the cost of slower synthesis speed.

# start = time.time()
# noise = torch.randn(1,1,256).to(device)
# wav = inference(text, noise, diffusion_steps=10, embedding_scale=1)
# rtf = (time.time() - start) / (len(wav) / 24000)
# print(f"RTF = {rtf:5f}")
# #import IPython.display as ipd
# #display(ipd.Audio(wav, rate=24000))
# sd.play(wav)
if __name__=="__main__":
    # test_lj_speech_v2()
    # Long-form generation
    # --------------------
    #from pydub import AudioSegment
    #import numpy as np
    passage = """
    If the supply of fruit is greater than the family needs, it may be made a source of income by sending the fresh fruit to the market if there is one near enough, or by preserving, canning, and making jelly for sale. To make such an enterprise a success the fruit and work must be first class. 
    There is magic in the word "Homemade," when the product appeals to the eye and the palate; but many careless and incompetent people have found to their sorrow that this word has not magic enough to float inferior goods on the market. 
    As a rule large canning and preserving establishments are clean and have the best appliances, and they employ chemists and skilled labor. The home product must be very good to compete with the attractive goods that are sent out from such establishments. 
    Yet for first-class homemade products there is a market in all large cities. 
    All first-class grocers have customers who purchase such goods.
    """# @param {type:"string"}
    output_audio_file= LJSpeech().ljspeech_v2(text=passage, autoplay=False)

    import soundfile as sf
    data, fs = sf.read(output_audio_file)
    sd.play(data,samplerate=24000,blocking=True)
    
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
    # ## combine approach-1
    # # combined=[]
    # # for wav in wavs:
    # #     combined=np.append(combined,wav)
    # ## combine approach-2
    # combined= np.concatenate(wavs)

    # from scipy.io.wavfile import write
    # scaled = np.int16(combined / np.max(np.abs(combined)) * 32767)
    # rate=24000
    # write('/home/pop/combined_test.wav', rate, scaled)

    # sd.play(scaled,samplerate=24000,blocking=True)

    #combined.export(headingsNewsDir + generatedFile, format="wav")
    # import scikits.audiolab
    # # write array to file:
    # scikits.audiolab.wavwrite(combined, 'combined_audiolab_test.wav', fs=44100, enc='pcm16')
    # # play the array:
    # scikits.audiolab.play(vars, fs=44100)

    # texts = {}
    # texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
    # texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
    # texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
    # texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"

    # for k,v in texts.items():
    #     ljspeech(text=v,diffusion_steps=10, embedding_scale=2)        
    #     ljspeech(text=v,diffusion_steps=5, embedding_scale=1)    
    #     # noise = torch.randn(1,1,256).to(device)
    #     # wav = inference(v, noise, diffusion_steps=10, embedding_scale=2) # embedding_scale=2 for more pronounced emotion
    #     # print(k + ": ")
    #     # #display(ipd.Audio(wav, rate=24000, normalize=False))
    #     # sd.play(wav,samplerate=24000,blocking=True) 



