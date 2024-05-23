import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO

models = {
    'Axial': '/best_brain_axial_40epoch.pt',
    'Coronal': '/best_brain_coronal_50+50.pt',
    'Sagittal': '/best_brain_sagittal_70_epoch_yolov5l.pt'
}

def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


st.title('Определение опухолей с помощью Йолы')


option = st.selectbox('Выбери тип среза:', ('Axial', 'Coronal', 'Sagittal'))
model_path = models[option]
model = load_model(model_path)


