import streamlit as st
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


#input_file = st.file_uploader("Загрузите картинку",type=["png", "jpg", "jpeg"])



def main():
    st.title("Yolo turbines Detector")
    st.write("Let's find your turbines using computer vision") 
    model = YOLO("8n_25epochs.pt")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image)
        results = model.predict(source=image)
        #results = model.predict(source='pexels-kindel-media-9889060_jpg.rf.8f4dc6d892c47f44f650f25d56b97880.jpg')

        #st.image(image, caption="Uploaded Image", use_column_width=True)
        
        #img_tensor = preprocess_image(image)
        fig, ax = plt.subplots(1)
        plt.axis('off')
        ax.imshow(results.render()[0])
        st.pyplot(fig)


if __name__ == '__main__':
    main()

