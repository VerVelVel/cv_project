import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import requests
from PIL import Image
from io import BytesIO

from models.model3.model_unet import load_pretrained_model

model_path = 'my_unet_model.h5'  
model = load_pretrained_model(model_path)
# Загрузка модели
# model = load_model('my_unet_model.h5')

def load_and_prepare_image(image, target_size=(256, 256)):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Модель ожидает размерность (1, 256, 256, 3)
    return image

def predict(image):
    image = load_and_prepare_image(image)
    prediction = model.predict(image)
    return prediction[0]

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

st.title('U-Net Model Deployment for Image Processing')

# Загрузка файла с компьютера
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    prediction = predict(image)
    st.image(prediction, caption='Processed Image', use_column_width=True)

# Загрузка изображения по URL
url = st.text_input("Enter the URL of an image...")
if url:
    try:
        image = load_image_from_url(url)
        st.image(image, caption='Image from URL', use_column_width=True)
        prediction = predict(image)
        st.image(prediction, caption='Processed Image', use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")

# Запуск Streamlit
if __name__ == '__main__':
    st.run()

