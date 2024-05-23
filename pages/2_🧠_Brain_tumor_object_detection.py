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

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

weights = {
    'Axial': 'models/model2/besaxial_40epoch.pt',
    'Coronal': 'models/model2/best_brain_coronal_50+50.pt',
    'Sagittal': 'models/model2/best_brain_sagittal_70_epoch_yolov5l.pt'
}


@st.cache_resource
def load_all_models():
    models = {}

    model_ax = torch.hub.load('ultralytics/yolov5', 'custom', path=weights['Axial'], force_reload=True, device='cpu')
    # model_ax.load_state_dict(torch.load(weights['Axial'], map_location='cpu'))
    model_ax.eval()
    models['Axial'] = model_ax

    model_cor = torch.hub.load('ultralytics/yolov5', 'custom', path=weights['Coronal'], force_reload=True, device='cpu')
    # model_cor.load_state_dict(torch.load(weights['Coronal'], map_location='cpu'))
    model_cor.eval()
    models['Coronal'] = model_cor

    model_sag = torch.hub.load('ultralytics/yolov5', 'custom', path=weights['Sagittal'], force_reload=True, device='cpu')
    # model_sag.load_state_dict(torch.load(weights['Sagittal'], map_location='cpu'))
    model_sag.eval()
    models['Sagittal'] = model_sag

    return models

models = load_all_models()

st.title('Определение опухолей с помощью моделей YOLA')


option = st.selectbox('Выбери тип среза:', ('Axial', 'Coronal', 'Sagittal'))
model = models[option]


def predict(image, model):
    img = np.array(image)
    results = model(img)
    return results

# Функция для отображения результатов
def display_results(image, results):
    img = np.array(image)
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        label = f'{results.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

# Функция для загрузки изображения по URL
def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

# Функция для загрузки изображения
def load_image(image):
    if isinstance(image, BytesIO):
        return Image.open(image)
    else:
        return load_image_from_url(image)

# Загрузка изображений через загрузку файлов
uploaded_files = st.file_uploader("Выберите изображения...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        results = predict(image, model)
        result_img = display_results(image, results)
        
        st.image(result_img, caption=f"Результат предсказания ({option})", use_column_width=True)

# Загрузка изображений через URL
image_urls = st.text_area('Введите URL изображений (один URL на строку)', height=100).strip().split('\n')
image_urls = [url.strip() for url in image_urls if url.strip()]

if image_urls:
    for url in image_urls:
        image = load_image_from_url(url).convert("RGB")
        # st.image(image, caption="Загруженное изображение", use_column_width=True)
        
        results = predict(image, model)
        result_img = display_results(image, results)
        
        st.image(result_img, caption=f"Результат предсказания ({option})", use_column_width=True)