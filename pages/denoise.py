import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
import streamlit as st
from PIL import Image
from PIL import ImageEnhance

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

path = '/home/nika/ds-phase-2/09-cv/denoise.pth' #local

def denoise_image2(image):
    # Загрузка модели и весов
    model = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
        nn.SELU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.SELU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
        nn.SELU(),
        nn.Dropout(0.5),
        nn.Conv2d(64, 32, kernel_size=2, bias=False),
        nn.SELU(),
        nn.Conv2d(32, 64, kernel_size=2, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=2, bias=False),
        nn.SELU(),
        nn.ConvTranspose2d(32, 64, kernel_size=2, bias=False),
        nn.SELU(),
        nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
        nn.BatchNorm2d(1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()

    # Преобразование исходного изображения в тензор
    preprocessing = T.Compose(
    [
        # T.Resize((420, 540)), # <-- при необходимости можно включить корректировку размера изображения
        T.ToTensor()
    ])
    image = preprocessing(image.convert('L')).unsqueeze(0).to(device) # <-- делаем изображение черно-белым

    # Получаем обработанное изображение
    with torch.no_grad():
        denoised_image = model(image)
        
    denoised_image = denoised_image.squeeze().cpu().numpy()
    denoised_image = (denoised_image * 255).astype(np.uint8)
    denoised_image = Image.fromarray(denoised_image)

    return denoised_image

def main():

    st.title('Denoising Dirty Documents')
    st.write('Very often, important documents lose their appearance due to an unfortunate accident.')
    st.write('Our application is designed to ensure that you no longer worry about such trivialities.')

    # Загружаем наше изображение
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Задаем параметры для настройки количества проходов через энкодер, контрастности, яркости и резкости выходного изображения
    passes = st.slider("Number of Passes", min_value=1, max_value=20, value=1,key='passes_slider')
    contrast_factor = st.slider("Сontrast factor", min_value=-10.0, max_value=10.0, value=1.0,key='contrast_slider')
    brightness_factor = st.slider("Brightness Factor", min_value=0.1, max_value=2.0, value=1.0, key='brightness_slider')
    sharpness_factor = st.slider("Sharpness Factor", min_value=-2.0, max_value=2.0, value=0.0, key='sharpness_slider')

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        image1=image 

        for pas in range(passes):
            denoised_image = denoise_image2(image)
            image=denoised_image
  
        contrast_enhancer = ImageEnhance.Contrast(denoised_image)
        contrast_adjusted_image = contrast_enhancer.enhance(contrast_factor)
        brightness_enhancer = ImageEnhance.Brightness(contrast_adjusted_image)
        brightness_adjusted_image = brightness_enhancer.enhance(brightness_factor)
        sharpness_enhancer = ImageEnhance.Sharpness(brightness_adjusted_image)
        sharpness_adjusted_image = sharpness_enhancer.enhance(sharpness_factor)
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(image1, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.image(sharpness_adjusted_image, caption="Denoised Image", use_column_width=True)
        

if __name__ == '__main__':
    main()

