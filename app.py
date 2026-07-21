import os
import gdown
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Chat ou Chien ?", page_icon="🐾")

MODEL_PATH = "modele_chat_chien.keras"
GDRIVE_FILE_ID = "1MSKhZDycozlcG3jm-KH0MHMsqXlls64_"


@st.cache_resource
def charger_modele():
  if not os.path.exists(MODEL_PATH):
    with st.spinner("Téléchargement du modèle en cours..."):
      url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
      gdown.download(url, MODEL_PATH, quiet=False)
  return tf.keras.models.load_model(MODEL_PATH)


model = charger_modele()

st.title("Classification Chat vs Chien")
st.write(
    "Chargez une image et le modèle prédira s'il s'agit d'un chat ou d'un"
    " chien."
)

fichier = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if fichier is not None:
  img = Image.open(fichier).convert("RGB").resize((150, 150))
  arr = np.array(img) / 255.0
  arr = np.expand_dims(arr, axis=0)

  proba = model.predict(arr)[0][0]
  classe = "Chien" if proba > 0.5 else "Chat"
  confiance = proba if proba > 0.5 else 1 - proba

  st.image(img, caption="Image chargée", use_container_width=True)

  if classe == "Chien":
    st.success("🐶 **Résultat :** C'est un **Chien** !")
  else:
    st.info("🐱 **Résultat :** C'est un **Chat** !")

  st.progress(float(confiance))
  st.write(f"**Niveau de confiance :** {confiance:.1%}")
