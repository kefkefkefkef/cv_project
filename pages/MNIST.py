
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from tqdm.notebook import tqdm
import torch.nn.functional as F
# Заголовок и выбор цифры пользователем
st.title("MNIST Conditional GAN")

latent_size=100
image_size = (28, 28)
flatten_size = 784  # = 28 x 28 x 1
batch_size = 128
num_classes = 10
# Словарь с соответствием меток классов и цифр
label_dict = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Sicdx", 7: "Seven", 8: "Eight", 9: "Nine"}

class Generator(nn.Module):
    def __init__(self, latent_size_linear, num_classes, image_size):
        super().__init__()
        self.image_size = image_size
        self.layer1  = nn.Sequential(
            nn.Linear(latent_size_linear + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size[0] * image_size[1]),
            nn.Tanh(),
        )
    def forward(self, latent, labels):
        one_hot_labels = F.one_hot(labels, num_classes = num_classes)
        latent_labels = torch.cat((latent, one_hot_labels), dim = 1)
        out = self.layer1(latent_labels)
        # modify output flat -> to image_size => batch_size x channel x w x h
        return out.view(out.size(0), 1, self.image_size[0], self.image_size[1])





generator=Generator(latent_size, num_classes,image_size)  
#Загрузка весов

generator.load_state_dict(torch.load('cgan-MNIST-Gen-Dis-50epoch-0.0001-0.001-0.5-0.5-128-100.pth',map_location=torch.device('cpu')))

generator.eval()
selected_number = st.slider("Выберите число", min_value=0, max_value=9, step=1, value=3)

latent_vector = torch.randn(1, latent_size)
label_tensor = torch.tensor([selected_number])
with torch.no_grad():
    generated_image = generator(latent_vector, label_tensor).squeeze().cpu().numpy()

# Преобразование изображения в объект Image и вывод
generated_image = (generated_image*255).astype("uint8")
generated_image = Image.fromarray(generated_image)
# Вывод картинки в оттенках серого
st.image(generated_image, caption=f"Generated {label_dict[selected_number]}")



