import streamlit as st
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import matplotlib as plt

model = YOLO("8n_25epochs.pt")
#input_file = st.file_uploader("Загрузите картинку",type=["png", "jpg", "jpeg"])



def main():
    st.title("Yolo turbines Detector")
    st.write("Let's find your turbines using computer vision") 

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        results = model.predict(source=image)

        #st.image(image, caption="Uploaded Image", use_column_width=True)
        
        #img_tensor = preprocess_image(image)
        fig, ax = plt.subplots()
        plt.axis('off')
        ax.imshow(results.render()[0])
        st.pyplot(fig)




