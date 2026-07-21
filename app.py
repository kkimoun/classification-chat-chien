import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title='Chat ou Chien ?', page_icon='🐾')

@st.cache_resource
def charger_modele():
    return tf.keras.models.load_model('modele_chat_chien.keras')

model = charger_modele()

st.title('Classification Chat vs Chien')
st.write("Chargez une image et le modèle prédira s'il s'agit d'un chat ou d'un chien.")

fichier = st.file_uploader('Choisissez une image', type=['jpg', 'jpeg', 'png'])

if fichier is not None:
    img = Image.open(fichier).convert('RGB').resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    proba = model.predict(arr)[0][0]
    classe = 'Chien' if proba > 0.5 else 'Chat'
    confiance = proba if proba > 0.5 else 1 - proba
    
    # Affichage de l'image
    st.image(img, caption='Image chargée', use_container_width=True)
    
    # Affichage du résultat dynamique
    if classe == 'Chien':
        st.success(f"🐶 **Résultat :** C'est un **Chien** !")
    else:
        st.info(f"🐱 **Résultat :** C'est un **Chat** !")

    # Barre de certitude
    st.progress(float(confiance))
    st.write(f"**Niveau de confiance :** {confiance:.1%}")
"""

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code_app.strip())

print("✅ Le fichier app.py a été généré avec succès dans votre environnement Colab !")
