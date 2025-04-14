# tts_wrapper.py

# import os
# import torch
# import torchaudio

# from models.tacotron2 import Tacotron2Wave
# from text import arabic_to_buckwalter, buckwalter_to_phonemes, simplify_phonemes

# class ArabicTTSWrapper:
#     def __init__(self):
#         self.models = {
#             "custom_model": "pretrained/exp_tc2_adv/states_7232.pth",
#             "pretrained_model": "pretrained/tacotron2_ar_adv.pth"
#         }
#         self.instances = {}

#     def get_model(self, model_key):
#         if model_key not in self.models:
#             raise ValueError("Unknown model key")

#         if model_key not in self.instances:
#             model = Tacotron2Wave(self.models[model_key])
#             model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#             self.instances[model_key] = model

#         return self.instances[model_key]

#     def synthesize(self, text, model_key="model1", denoise=0.005):
#         model = self.get_model(model_key)
#         buck = arabic_to_buckwalter(text)
#         phonemes = simplify_phonemes(buckwalter_to_phonemes(buck).replace(' ', '').replace('+', ' '))

#         wavs = model.tts([text], batch_size=1, denoise=denoise)
#         return wavs[0], phonemes
import os
import torch
import torchaudio
import zipfile
import gdown
import streamlit as st

from models.tacotron2 import Tacotron2Wave
from text import arabic_to_buckwalter, buckwalter_to_phonemes, simplify_phonemes


class ArabicTTSWrapper:
    def __init__(self):
        self.models = {
            "custom_model": "pretrained/states_7232.pth",
            "pretrained_model": "pretrained/tacotron2_ar_adv.pth"
        }

        self.gdrive_file_id = "1IMqtVOE6O_brqgP68CpG4rs1wrgIsCCp"  # 🔁 Remplace par le vrai file ID
        self.zip_path = "pretrained.zip"
        self.extract_dir = ""
        self.instances = {}

        self.ensure_models_exist()

    def ensure_models_exist(self):
        missing = [path for path in self.models.values() if not os.path.exists(path)]
        if missing:
            print("🔽 Téléchargement des modèles depuis Google Drive...")
            self.download_and_extract_models()

    def download_and_extract_models(self):
        url = f"https://drive.google.com/uc?id={self.gdrive_file_id}"

        with st.spinner("📥 Téléchargement du modèle..."):
            gdown.download(url, self.zip_path, quiet=False)
            print("✅ Téléchargement terminé")

        with st.spinner("📦 Extraction du modèle... veuillez patienter..."):
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            print("✅ Extraction terminée")

        os.remove(self.zip_path)
        st.success("✅ Modèle prêt à l’utilisation !")

    def get_model(self, model_key):
        if model_key not in self.models:
            raise ValueError("Unknown model key")

        if model_key not in self.instances:
            model = Tacotron2Wave(self.models[model_key])
            model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.instances[model_key] = model

        return self.instances[model_key]

    def synthesize(self, text, model_key="custom_model", denoise=0.005):
        model = self.get_model(model_key)
        buck = arabic_to_buckwalter(text)
        phonemes = simplify_phonemes(
            buckwalter_to_phonemes(buck).replace(' ', '').replace('+', ' ')
        )
        wavs = model.tts([text], batch_size=1, denoise=denoise)
        return wavs[0], phonemes
