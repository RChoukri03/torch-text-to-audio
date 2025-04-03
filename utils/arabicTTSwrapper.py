# tts_wrapper.py

import os
import torch
import torchaudio

from models.tacotron2 import Tacotron2Wave
from text import arabic_to_buckwalter, buckwalter_to_phonemes, simplify_phonemes

class ArabicTTSWrapper:
    def __init__(self):
        self.models = {
            "custom_model": "pretrained/exp_tc2_adv/states_5000.pth",
            "pretrained_model": "pretrained/tacotron2_ar_adv.pth"
        }
        self.instances = {}

    def get_model(self, model_key):
        if model_key not in self.models:
            raise ValueError("Unknown model key")

        if model_key not in self.instances:
            model = Tacotron2Wave(self.models[model_key])
            model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.instances[model_key] = model

        return self.instances[model_key]

    def synthesize(self, text, model_key="model1", denoise=0.005):
        model = self.get_model(model_key)
        buck = arabic_to_buckwalter(text)
        phonemes = simplify_phonemes(buckwalter_to_phonemes(buck).replace(' ', '').replace('+', ' '))

        wavs = model.tts([text], batch_size=1, denoise=denoise)
        return wavs[0], phonemes
