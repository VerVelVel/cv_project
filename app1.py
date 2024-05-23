import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO
import sys
from pathlib import Path

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

    model_ax = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
    model_ax.load_state_dict(torch.load(weights['Axial'], map_location='cpu'))
    model_ax.eval()
    models['Axial'] = model_ax

    model_cor = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=False)
    model_cor.load_state_dict(torch.load(weights['Coronal'], map_location='cpu'))
    model_cor.eval()
    models['Coronal'] = model_cor

    model_sag = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=False)
    model_sag.load_state_dict(torch.load(weights['Sagittal'], map_location='cpu'))
    model_sag.eval()
    models['Sagittal'] = model_sag

    return models

models = load_all_models()

st.title('Определение опухолей с помощью моделей YOLA')


option = st.selectbox('Выбери тип среза:', ('Axial', 'Coronal', 'Sagittal'))
model = models[option]




