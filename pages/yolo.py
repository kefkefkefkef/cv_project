import streamlit as st
from ultralytics import YOLO
import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def main():
    st.title("Yolo turbines Detector")
    st.write("Let's find your turbines using computer vision") 
    model = YOLO("8n_25epochs.pt")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image)
        results = model.predict(source=image)
        st.image(results[0].plot())


if __name__ == '__main__':
    main()

