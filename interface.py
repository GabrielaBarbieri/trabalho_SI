import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.title("Sistema de Detecção de EPI")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    results = model.predict(img)
    st.image(results[0].plot(), caption="Imagem processada")
