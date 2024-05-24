# import streamlit as st
# from PIL import Image
# import torch
# import json
# import sys
# from pathlib import Path
# import requests
# from io import BytesIO
# import time
# import cv2
# import numpy as np

# st.write("# Локализация объектов")
# st.write("Здесь вы можете загрузить картинку со своего устройства, либо при помощи ссылки")

# project_root = Path(__file__).resolve().parents[1]
# sys.path.append(str(project_root))

# from models.model1.model import LocModel
# from models.model1.preprocessing import preprocess

# device = 'cpu'

# # Загрузка модели и словаря
# @st.cache_resource
# def load_model():
#     device = torch.device('cpu')
#     model = LocModel()
#     weights_path = 'models/model1/model_weights_2.pth'
#     state_dict = torch.load(weights_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.to(device)
#     model.eval()
#     return model

# model = load_model()

# ix2cls = {0: 'cucumber', 1: 'eggplant', 2: 'mushroom'}

# # Функция для предсказания класса изображения и рисования bounding box
# def predict(image):
#     img = preprocess(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         start_time = time.time()
#         preds_cls, preds_reg = model(img)
#         end_time = time.time()

#     pred_class = preds_cls.argmax(dim=1).item()
#     img = img.squeeze().permute(1, 2, 0).cpu().numpy()
#     img = (img * 255).astype(np.uint8)  # Преобразование значений пикселей в диапазон [0, 255]
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Переключаем обратно в BGR для OpenCV

#     pred_box_coords = (preds_reg[0] * 227).cpu().detach().numpy().astype('int')
#     pred_box = cv2.rectangle(
#         img.copy(), 
#         (pred_box_coords[0], pred_box_coords[1]),  # top left
#         (pred_box_coords[2], pred_box_coords[3]),  # bottom right
#         color=(255, 0, 0), thickness=2
#     )

#     # Преобразование изображения обратно в RGB для правильного отображения в Streamlit
#     pred_box = cv2.cvtColor(pred_box, cv2.COLOR_BGR2RGB)
#     pred_box = pred_box / 255.0  # Нормализация значений пикселей в диапазон [0, 1]

#     return pred_box, ix2cls[pred_class], end_time - start_time

# # Загрузка изображения по ссылке
# def load_image_from_url(url):
#     response = requests.get(url)
#     image = Image.open(BytesIO(response.content))
#     return image

# # Загрузка изображения через загрузку файла или по ссылке
# def load_image(image):
#     if isinstance(image, BytesIO):
#         return Image.open(image)
#     else:
#         return load_image_from_url(image)

# # Отображение изображения и результатов предсказания
# def display_results(image, pred_label, inference_time):
#     st.title(pred_label)
#     st.image(image, use_column_width=True)
#     st.write(f"Inference Time: {inference_time:.4f} seconds")


# uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# if uploaded_files is not None:
#     for uploaded_file in uploaded_files:
#         image = Image.open(uploaded_file)
#         predicted_img, predicted_class, inference_time = predict(image)
#         display_results(predicted_img, predicted_class, inference_time)

# image_urls = st.text_area('Enter image URLs (one URL per line)', height=100).strip().split('\n')
# image_urls = [url.strip() for url in image_urls if url.strip()]

# if image_urls:
#     for url in image_urls:
#         image = load_image_from_url(url)
#         predicted_img, predicted_class, inference_time = predict(image)
#         display_results(predicted_img, predicted_class, inference_time)


import streamlit as st
from PIL import Image
import torch
import json
import sys
from pathlib import Path
import requests
from io import BytesIO
import time
import cv2
import numpy as np

st.write("# Локализация объектов")
st.write("Здесь вы можете загрузить картинку со своего устройства, либо при помощи ссылки")

# Добавление пути к проекту и моделям
project_root = Path(__file__).resolve().parents[1]
models_path = project_root / 'models'
sys.path.append(str(models_path))

from model1.model import LocModel
from model1.preprocessing import preprocess

device = 'cpu'

# Загрузка модели и словаря
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = LocModel()
    weights_path = models_path / 'model1' / 'model_weights_2.pth'
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

ix2cls = {0: 'cucumber', 1: 'eggplant', 2: 'mushroom'}

# Функция для предсказания класса изображения и рисования bounding box
def predict(image):
    img = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        start_time = time.time()
        preds_cls, preds_reg = model(img)
        end_time = time.time()

    pred_class = preds_cls.argmax(dim=1).item()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)  # Преобразование значений пикселей в диапазон [0, 255]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Переключаем обратно в BGR для OpenCV

    pred_box_coords = (preds_reg[0] * 227).cpu().detach().numpy().astype('int')
    pred_box = cv2.rectangle(
        img.copy(), 
        (pred_box_coords[0], pred_box_coords[1]),  # top left
        (pred_box_coords[2], pred_box_coords[3]),  # bottom right
        color=(255, 0, 0), thickness=2
    )

    # Преобразование изображения обратно в RGB для правильного отображения в Streamlit
    pred_box = cv2.cvtColor(pred_box, cv2.COLOR_BGR2RGB)
    pred_box = pred_box / 255.0  # Нормализация значений пикселей в диапазон [0, 1]

    return pred_box, ix2cls[pred_class], end_time - start_time

# Загрузка изображения по ссылке
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Загрузка изображения через загрузку файла или по ссылке
def load_image(image):
    if isinstance(image, BytesIO):
        return Image.open(image)
    else:
        return load_image_from_url(image)

# Отображение изображения и результатов предсказания
def display_results(image, pred_label, inference_time):
    st.title(pred_label)
    st.image(image, use_column_width=True)
    st.write(f"Inference Time: {inference_time:.4f} seconds")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        predicted_img, predicted_class, inference_time = predict(image)
        display_results(predicted_img, predicted_class, inference_time)

image_urls = st.text_area('Enter image URLs (one URL per line)', height=100).strip().split('\n')
image_urls = [url.strip() for url in image_urls if url.strip()]

if image_urls:
    for url in image_urls:
        image = load_image_from_url(url)
        predicted_img, predicted_class, inference_time = predict(image)
        display_results(predicted_img, predicted_class, inference_time)

