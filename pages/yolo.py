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
    option = st.selectbox(
    'How many epochs should I train?',
    ('25', '50', '100', '200'))
    
    if option == '25':
        model = YOLO("8n_25epochs.pt")
    elif option == '50':
        model = YOLO("8n_50epochs.pt")
    elif option == '100':
        model = YOLO("8n_100epochs.pt")
    elif option == '200':
        model = YOLO("8n_200epochs.pt")


    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        results = model.predict(source=image)
        st.image(results[0].plot())


if __name__ == '__main__':
    main()

