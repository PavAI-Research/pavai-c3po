from pavai.setup import config 
from pavai.setup import logutil
logger = logutil.logging.getLogger(__name__)

try: 
    import nltk
except:
    nltk.download('punkt')
    import nltk

# load packages
import time
import yaml
import torch
import random
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import sounddevice as sd
import soundfile as sf
import phonemizer
import sys, os
import numpy as np
from scipy.io.wavfile import write
from pavai.shared.styletts2.utils import length_to_mask,recursive_munch
from pavai.shared.styletts2.models import load_F0_models,load_ASR_models,build_model
## maximum_path,get_data_path_list,log_norm,get_image,log_print
from pavai.shared.styletts2.text_utils import TextCleaner
from pavai.shared.styletts2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from pavai.shared.styletts2.Utils.PLBERT.util import load_plbert
from pavai.shared.styletts2.download_models import get_styletts2_model_files
from typing import Any, Dict
from collections import OrderedDict
import traceback
import gc
import json
import pavai.shared.styletts2.speech_type as speech_type
import pavai.shared.styletts2.live_voices as live_voices 

class LibriSpeech(speech_type.Singleton):

    def __init__(self, device:str=None, style_config:str=None, model_config:str=None):
        torch.manual_seed(200)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        random.seed(200)
        np.random.seed(200)

        ## System Configuration
        self.StyleTTS2_LANGUAGE="en-us"
        #StyleTTS2_LANGUAGE="en" #"cmn" #"yue"-- cantonese
        self.StyleTTS2_CONFIG_FILE="resources/models/styletts2/Models/LibriTTS/config.yml"
        self.StyleTTS2_MODEL_FILE="resources/models/styletts2/Models/LibriTTS/epochs_2nd_00020.pth"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #self.device="cpu"
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.mean =-4 
        self.std = 4
        #languages=https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
        # load phonemizer
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language=self.StyleTTS2_LANGUAGE, preserve_punctuation=True,  with_stress=True)
        self.textclenaer = TextCleaner()
        try:
            self.config = yaml.safe_load(open(self.StyleTTS2_CONFIG_FILE))
        except:
            logger.error("StyleTTS2_CONFIG_FILE Not Found!")
            get_styletts2_model_files()
            ## re-read file downloaded
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

        self.model_params = recursive_munch(self.config['model_params'])
        self.model = build_model(self.model_params, self.text_aligner, self.pitch_extractor, self.plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        #  Load models
        self.params_whole = torch.load(self.StyleTTS2_MODEL_FILE, map_location='cpu')
        self.params = self.params_whole['net']

        for key in self.model:
            if key in self.params:
                try:
                    self.model[key].load_state_dict(self.params[key])
                except:
                    state_dict = self.params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
            clamp=False
        )
        ## Optimize inference
        ## self.model = torch.compile(self.model)

        self.ref_texts = {}
        self.ref_texts['happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
        self.ref_texts['sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
        self.ref_texts['angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
        self.ref_texts['surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"

        self.cached_voice={}
        # reference voices
        voice_config = config.system_config["REFERENCE_VOICES"]
        self.reference_voices = live_voices.load_voices(voice_config)

    def lookup_voice(self,name: str = "jane"):
        name=name.lower()
        if name in self.cached_voice.keys():
            logger.info(f"get compute style from cached: {name}")
            return self.cached_voice[name]
        ## lookup
        if self.reference_voices is None:
            reference_voices = {
                "ryan": "resources/models/styletts2/reference_audio/Ryan.wav",
                "jane": "resources/models/styletts2/reference_audio/Jane.wav",
                "me1": "resources/models/styletts2/reference_audio/Me1.wav",
                "me2": "resources/models/styletts2/reference_audio/Me2.wav",
                "me3": "resources/models/styletts2/reference_audio/Me3.wav",
                "vinay": "resources/models/styletts2/reference_audio/Vinay.wav",
                "nima": "resources/models/styletts2/reference_audio/Nima.wav",
                "yinghao": "resources/models/styletts2/reference_audio/Yinghao.wav",
                "keith": "resources/models/styletts2/reference_audio/Keith.wav",
                "may": "resources/models/styletts2/reference_audio/May.wav",
                "anthony": "resources/models/styletts2/reference_audio/anthony.wav",
                "c3p013": "resources/models/styletts2/reference_audio/c3p013.wav",
                "c3p0voice8": "resources/models/styletts2/reference_audio/c3p0_voice8.wav",
                "c3p0voice13": "resources/models/styletts2/reference_audio/c3p0_voice13.wav",
                "c3p0voice1": "resources/models/styletts2/reference_audio/c3p0_voice1.wav"
            }
        if name in self.reference_voices.keys():
            voice_path = self.reference_voices[name.lower()]
            voice = self.compute_style(voice_path)
        else:
            logger.error(f" Error Missing voice file {name}, fallback to default")
            name = "jane"
            voice_path = self.reference_voices[name.lower()]
            voice = self.compute_style(voice_path)
        ## cached it
        self.cached_voice[name] = voice
        return voice

    def preprocess(self,wave):
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def lpad_text(self,text:str, max_length:int=43, endingchar:str="c")->str:
        if len(text) < max_length:
            text=text.ljust(max_length, '…')
        return text+"."

    def compute_style(self,path:str, samplerate:int=24000, top_db:int=30):
        wave, sr = librosa.load(path, sr=samplerate)
        audio, index = librosa.effects.trim(wave, top_db=top_db)
        if sr != samplerate:
            audio = librosa.resample(audio, sr, samplerate)
        mel_tensor = self.preprocess(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))

        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self,text, ref_s, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,
                                            embedding_scale=embedding_scale,
                                                features=ref_s, # reference from the same speaker as the embedding
                                                num_steps=diffusion_steps).squeeze(1)
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,
                                            s, input_lengths, text_mask)

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
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

    def LFinference(self,text, s_prev, ref_s, alpha = 0.3, beta = 0.7, t = 0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        ps = ps.replace('``', '"')
        ps = ps.replace("''", '"')

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
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

            d = self.model.predictor.text_encoder(d_en,
                                                s, input_lengths, text_mask)

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
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        return out.squeeze().cpu().numpy()[..., :-100], s_pred # weird pulse at the end of the model, need to be fixed later

    def STinference(self,text, ref_s, ref_text, alpha = 0.3, beta = 0.7, diffusion_steps=5, embedding_scale=1):
        text = text.strip()
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        ref_text = ref_text.strip()
        ps = self.global_phonemizer.phonemize([ref_text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        ref_tokens = self.textclenaer(ps)
        ref_tokens.insert(0, 0)
        ref_tokens = torch.LongTensor(ref_tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(self.device)
            ref_text_mask = length_to_mask(ref_input_lengths).to(self.device)
            ref_bert_dur = self.model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())
            s_pred = self.sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                                            embedding=bert_dur,embedding_scale=embedding_scale,
                                            features=ref_s, # reference from the same speaker as the embedding
                                            num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
            s = beta * s + (1 - beta)  * ref_s[:, 128:]

            d = self.model.predictor.text_encoder(d_en,s, input_lengths, text_mask)

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
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = self.model.decoder(asr,F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later

    ## ------------------
    ## Synthesize speech
    ## ------------------
    def wipe_memory(self,objects:list=[]): # DOES WORK
        try:
            for obj in objects:
                del obj
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects.")
            torch.cuda.empty_cache()
        except:
            pass

    def librispeech(self,text:str, 
                    compute_style:any,
                    alpha=0.3, 
                    beta=0.7,
                    diffusion_steps:int=random.randint(3, 10), 
                    embedding_scale:int=random.randint(1, 2), 
                    blocking_flag:bool=True,
                    samplerate:int=24000,
                    autoplay:bool=True):
        t0=time.perf_counter()    
        if isinstance(compute_style, str):
            ref_s = self.lookup_voice(name=compute_style)                
        else:
            ref_s=compute_style # input torch.Tensor
        start = time.time()
        text=self.lpad_text(text)
        wav = self.inference(text, ref_s, 
                        alpha=alpha, beta=beta, 
                        diffusion_steps=diffusion_steps, 
                        embedding_scale=embedding_scale)
        rtf = (time.time() - start) / (len(wav) / 24000)
        t1=time.perf_counter()
        logger.info(f"librispeech rtf took {rtf:5f} in {t1-t0:.2f} seconds")
        if autoplay:
            sd.play(wav,samplerate=samplerate,blocking=blocking_flag)
        return wav

    def librispeech_v2(self,text:str, 
                    compute_style:any,
                    emotion:str=None,
                    alpha:int=0.5, 
                    beta:int=0.9,
                    diffusion_steps:int=random.randint(3, 10), 
                    embedding_scale:int=random.randint(1, 2), 
                    blocking_flag:bool=True,
                    samplerate:int=24000,
                    output_audiofile="workspace/temp/librispeech_v2.wav",
                    return_file:bool=True,
                    autoplay:bool=True)->str:
        t0=time.perf_counter()    
        if emotion is None:
            emotion="happy"
        try:              
            v = self.ref_texts[emotion.lower()]
            if isinstance(compute_style, str):
                ref_s = self.lookup_voice(name=compute_style)                
            else:
                ref_s=compute_style # input torch.Tensor

            sentences = self.sentence_word_splitter(text=text.strip(),num_of_words=43) 
            wavs = []        
            for text in sentences:        
                start = time.time()
                if text.strip() == "": continue
                text += '.' # add it back
                text=self.lpad_text(text)                
                wav = self.STinference(text, ref_s, v, 
                                diffusion_steps=diffusion_steps, 
                                alpha=alpha, beta=beta,embedding_scale=embedding_scale)
                rtf = (time.time() - start) / (len(wav) / samplerate)
                t1=time.perf_counter()
                wavs.append(wav)
                logger.info(f"librispeech_v2 rtf took {rtf:5f} in {t1-t0:.2f} seconds")
            ## putting them rogether    
            combined= np.concatenate(wavs) 
            scaled = np.int16(combined / np.max(np.abs(combined)) * 32767)
            write(output_audiofile, samplerate, scaled)
            if autoplay:
                sd.play(scaled,samplerate=samplerate,blocking=blocking_flag)       
            self.wipe_memory(objects=[combined,text,wavs,sentences,ref_s])                               
            if not return_file:
                return scaled
            self.wipe_memory(objects=[scaled])                               
        except Exception as e:
            logger.error(f"Exeption occurred {str(e.args)}")
            logger.error(traceback.format_exc())
        finally:
            self.wipe_memory()          
        return output_audiofile

    def chunk_text_to_fixed_length(self,text: str, length: int):
        text = text.strip()
        result = [text[0+i:length+i] for i in range(0, len(text), length)]
        return result

    def sentence_word_splitter(self,text: str,num_of_words: int) -> list:
        pieces = text.split()
        return [" ".join(pieces[i:i+num_of_words]) for i in range(0, len(pieces), num_of_words)]

    def librispeech_v3(self,text:str, 
                    compute_style:any,
                    alpha=0.3, 
                    beta=0.7,
                    diffusion_steps:int=random.randint(3, 10), 
                    embedding_scale:int=random.randint(1, 2), 
                    blocking_flag:bool=True,
                    samplerate:int=24000,
                    output_audiofile="workspace/temp/librispeech_v3.wav",
                    return_file:bool=True,                    
                    autoplay:bool=True)->str:
        t0=time.perf_counter()    
        try:        
            if isinstance(compute_style, str):
                ref_s = self.lookup_voice(name=compute_style)
            else:
                ref_s = compute_style # input torch.Tensor
            #sentences = text.split('.') # simple split by comma
            sentences = self.sentence_word_splitter(text=text.strip(),num_of_words=43)            
            wavs = []
            s_prev=None
            for text in sentences:
                start = time.time()                
                if text.strip() == "": continue
                text += '.' # add it back
                text=self.lpad_text(text)
                logger.debug(f"text length={len(text)}")
                ## disable due higher memory usage
                ## wav,s_prev = self.LFinference(text, s_prev, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
                wav = self.inference(text, ref_s, alpha=alpha, beta=beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)                
                wavs.append(wav)
                rtf = (time.time() - start) / (len(wav) / samplerate)
                t1=time.perf_counter()
                logger.info(f"librispeech_v3 rtf took {rtf:5f} in {t1-t0:.2f} seconds")
            ## putting them rogether    
            combined= np.concatenate(wavs) 
            scaled = np.int16(combined / np.max(np.abs(combined)) * 32767)
            write(output_audiofile, samplerate, scaled)
            if autoplay:
                sd.play(scaled,samplerate=samplerate,blocking=blocking_flag)       
            self.wipe_memory(objects=[combined,text,wavs,sentences,ref_s,s_prev])                    
            if not return_file:
                return scaled                
            ## clean up
            self.wipe_memory(objects=[scaled])                    
        except Exception as e:
            logger.error(f"Exeption occurred {str(e.args)}")
            logger.error(traceback.format_exc())
        finally:
            self.wipe_memory()          
        return output_audiofile

    def test_libris_speech(self):
        #text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
        sample_text="""
        Thank you for this! I have one more question if anyone can bite: I am trying to take the average of the first elements in these datapoints(i.e. datapoints[0][0]). Just to list them, I tried doing datapoints[0:5][0] but all I get is the first datapoint with both elements as opposed to wanting to get the first 5 datapoints containing only the first element. Is there a way to do this?
        """
        self.librispeech(text=sample_text,compute_style="jane", alpha=0.3, beta=0.7, diffusion_steps=10    )
        self.librispeech(text=sample_text,compute_style="me1", alpha=0.3, beta=0.7, diffusion_steps=10)

    def test_libris_speech_emotions(self):
        #ref_s2 = self.compute_style("resources/models/styletts2/reference_audio/Jane.wav")
        ref_texts = {}
        ref_texts['Happy'] = "We are happy to invite you to join us on a journey to the past, where we will visit the most amazing monuments ever built by human hands."
        ref_texts['Sad'] = "I am sorry to say that we have suffered a severe setback in our efforts to restore prosperity and confidence."
        ref_texts['Angry'] = "The field of astronomy is a joke! Its theories are based on flawed observations and biased interpretations!"
        ref_texts['Surprised'] = "I can't believe it! You mean to tell me that you have discovered a new species of bacteria in this pond?"    
        text = "Yea, his honourable worship is within, but he hath a godly minister or two with him, and likewise a leech."
        for k,v in ref_texts.items():
            print(k + ": Style Transfer")
            self.librispeech_v2(text=text,compute_style="jane",emotion=k)        

    def test_libris_speech_longspeech(self,voice_id:int=1):
        ## Long-form generation
        passage = """
        If the supply of fruit is greater than the family needs, it may be made a source of income by sending the fresh fruit to the market if there is one near enough, or by preserving, canning, and making jelly for sale. To make such an enterprise a success the fruit and work must be first class. 
        There is magic in the word "Homemade," when the product appeals to the eye and the palate; but many careless and incompetent people have found to their sorrow that this word has not magic enough to float inferior goods on the market. 
        As a rule large canning and preserving establishments are clean and have the best appliances, and they employ chemists and skilled labor. The home product must be very good to compete with the attractive goods that are sent out from such establishments. 
        Yet for first-class homemade products there is a market in all large cities. 
        All first-class grocers have customers who purchase such goods.
        """
        self.librispeech_v3(text=passage, compute_style="ryan")

"""MAIN"""
if __name__ == "__main__":

    #LibriSpeech().test_libris_speech()
    #LibriSpeech().test_libris_speech_emotions()
    #LibriSpeech().test_libris_speech_longspeech()
    ## Basic synthesis (5 diffusion steps)
    #text = "StyleTTS 2 is a text-to-speech model that leverages style diffusion and adversarial training with large speech language models to achieve human-level text-to-speech synthesis."
    passage = """
Canada is indeed a vast country. Here's some information on the size of Canada:

1. Land Area: Canada covers an area of approximately 9.9 million square kilometers (3.8 million square miles). This makes it the second-largest country in the world by total land area after Russia.

2. Water Area: Canada also has a large coastline that includes several lakes and islands. The total surface area of Canada's lakes (including Great Bear Lake, Great Slave Lake, Lake Winnipegosee, Lake Superior, and Lake Huron) is approximately 6.5 million square kilometers (2.5 million square miles).

3. Total Area: When we combine the land area and water area, Canada's total area is approximately 10 million square kilometers (3.8 million square miles). This is roughly one-third of the total land area of our planet!

I hope that helps answer your question. Let me know if there's anything else I can assist you with today!
    """
    #ref_s1 = LibriSpeech().compute_style("resources/models/styletts2/reference_audio/Ryan.wav")
    #ref_s2 = LibriSpeech().compute_style("resources/models/styletts2/reference_audio/Jane.wav")
    # ref_s3 = LibriSpeech().compute_style("resources/models/styletts2/reference_audio/Me1.wav")
    # ref_s4 = compute_style("resources/models/styletts2/reference_audio/Me2.wav")
    # ref_s5 = compute_style("resources/models/styletts2/reference_audio/Me3.wav")
    # ref_s6 = compute_style("resources/models/styletts2/reference_audio/Vinay.wav")
    # ref_s7 = compute_style("resources/models/styletts2/reference_audio/Nima.wav")
    # ref_s8 = compute_style("resources/models/styletts2/reference_audio/Yinghao.wav")
    # ref_s9 = compute_style("resources/models/styletts2/reference_audio/Keith.wav")
    # ref_s10 = compute_style("resources/models/styletts2/reference_audio/May.wav")
    # ref_s11 = compute_style("resources/models/styletts2/reference_audio/June.wav")
    # ## Default setting (alpha = 0.3, beta=0.7)

    output_audio_file = LibriSpeech().librispeech_v3(text=passage,compute_style="jane",autoplay=False)
    data, fs = sf.read(output_audio_file)
    sd.play(data,samplerate=24000,blocking=True)

    ## output_audio_file = LibriSpeech().librispeech_v2(text=passage,compute_style=ref_s2,emotion="sad",autoplay=False)
    ## data, fs = sf.read(output_audio_file)
    ## sd.play(data,samplerate=24000,blocking=True)

    #librispeech(text=text,compute_style=ref_s2, voice='Jane',alpha=0.3, beta=0.7, diffusion_steps=10)
    #librispeech(text=text,compute_style=ref_s3, voice='Me1',alpha=0.3, beta=0.7, diffusion_steps=10)
    
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
    #     wav = STinference(text, ref_s2, v, diffusion_steps=10, alpha=0.5, beta=0.9, embedding_scale=1.5)
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

