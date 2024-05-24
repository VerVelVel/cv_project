import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img = img_to_array(img)
    img /= 255.0
    return np.expand_dims(img, axis=0)

def preprocess_input(x):
    return x / 255.0