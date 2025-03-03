import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Charger le modèle
@st.cache_resource
def load_model():
    return load_learner("model.pkl")

learn = load_model()

# Fonction de prédiction
def predict(image):
    pred, pred_idx, probs = learn.predict(image)
    return pred, probs[pred_idx].item()

# Interface utilisateur
st.title("Détection de la Pneumonie avec IA")
st.write("Téléchargez une radiographie pulmonaire pour obtenir une prédiction.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    if st.button("Prédire"):
        prediction, confidence = predict(image)
        st.write(f"### Résultat : {prediction}")
        st.write(f"### Confiance : {confidence:.2f}")

