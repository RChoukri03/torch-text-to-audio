# 🗣️ Arabic TTS - Synthèse vocale pour la langue arabe

Ce projet propose une chaîne complète de synthèse vocale pour la langue arabe, basée sur le modèle **Tacotron2** et un vocodeur neuronal. Il intègre un pipeline phonétique, un modèle entraîné, une interface utilisateur via Streamlit et un backend d’inférence avec Flask.
---
## 📌 Fonctionnalités

- 🔡 Entrée de texte en arabe (avec ou sans voyelles)  
- 📜 Génération de la transcription phonétique  
- 🔊 Synthèse audio (.wav) à partir du texte  
- 🧠 Sélection entre **modèle personnalisé** et **modèle préentraîné**  
- 💬 Interface utilisateur interactive via **Streamlit**  
- 🔁 Backend REST pour l'inférence (Flask)  
- ☁️ Compatible avec **déploiement local ou sur Streamlit Cloud**



## 🧪 Entraîner un modèle

Configurez votre fichier `configs/base_config.json`, puis lancez :

python train.py --config configs/base_config.json

## 🔊 Inférence (Texte → Audio)

Générez un fichier `.wav` à partir d’un texte arabe :

python inference.py --text "مرحبا بك" --checkpoint checkpoints/model.pt


## 💻 Interface utilisateur avec Streamlit

Lancer l'interface web localement :

streamlit run app/streamlit_app.py

ou à distance à travers:
https://torch-text-to-audio-h8i6aap73eogkxver6uffn.streamlit.app/


## 🧪 Entrainer sur Google Colab

Essayez directement sur Google Colab ici :  
👉 [🔗 Notebook Colab](https://colab.research.google.com/drive/1mHj_qb91Sc-kWnShjb6w_a5jj4hMUiDB#scrollTo=P1arcYIkUsXu&uniqifier=1)
