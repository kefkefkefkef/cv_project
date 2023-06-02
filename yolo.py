import streamlit as st
from ultralytics import YOLO
import torch

model = YOLO("8n_25epochs.pt")
input_file = st.file_uploader("Загрузите картинку",type=["png", "jpg", "jpeg"])
results = model.predict(source=input_file)
st.image()