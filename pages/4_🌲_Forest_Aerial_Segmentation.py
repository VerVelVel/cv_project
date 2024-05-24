import streamlit as st
from tensorflow.keras.utils import img_to_array, load_img
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
import sys
import cv2
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from models.model3.model_unet import load_custom_unet_model

model_path = 'my_unet_model.h5'
model = load_custom_unet_model(model_path)

def load_and_prepare_image(image, target_size=(128, 128)):
    """ Загружает и подготавливает изображение для модели. """
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image = img_to_array(image)
    image = image / 255.0  #new
    image = np.expand_dims(image, axis=0)
    return image



def predict(image):
    """ Прогнозирует маску сегментации из изображения. """
    processed_image = load_and_prepare_image(image)
    prediction = model.predict(processed_image)
    return prediction[0]

# def display_prediction(prediction):
#     prediction = prediction.squeeze()  # Удаление лишних размерностей
#     mask = (prediction > 0.5).astype(np.uint8)  # Применение порога для создания бинарной маски
#     prediction_image = (mask * 255).astype(np.uint8)  # Масштабирование для отображения
#     return Image.fromarray(prediction_image)

def display_prediction(image, prediction):
    prediction = prediction.squeeze()
    mask = (prediction > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (image.width, image.height))  # Resize mask to the original image size
    color_mask = np.zeros_like(image)
    color_mask[mask == 1] = [255, 0, 0]  # Red color for mask
    combined_image = cv2.addWeighted(np.array(image), 1, color_mask, 0.5, 0)  # Overlay mask on image
    return Image.fromarray(combined_image)


def load_image_from_url(url):
    """ Загружает изображение по URL. """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

st.title('U-Net Model Deployment for Image Processing')

uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Обработка изображения и предсказание
    processed_image = load_and_prepare_image(image)
    prediction = model.predict(processed_image)
    display_image = display_prediction(image, prediction[0])
    st.image(display_image, caption='Segmented Image', use_column_width=True)

url = st.text_input("Enter the URL of an image...")
if url:
    try:
        image = load_image_from_url(url)
        st.image(image, caption='Image from URL', use_column_width=True)
        prediction = predict(image)
        display_image = display_prediction(prediction)
        st.image(display_image, caption='Processed Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
