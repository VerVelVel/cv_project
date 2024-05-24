# Импорт необходимых библиотек
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2DTranspose


# Функция для загрузки модели с кастомизированными настройками
def load_custom_unet_model(model_path):
    """
    Загружает модель U-Net, исправляя конфликты с неизвестными аргументами в слоях.
    
    Args:
    model_path (str): Путь к файлу модели .h5
    
    Returns:
    tf.keras.Model: Загруженная модель U-Net.
    """
    # Определение custom_objects для исправления ошибок при загрузке
    custom_objects = {
        'Conv2DTranspose': lambda **kwargs: Conv2DTranspose(**{key: arg for key, arg in kwargs.items() if key != 'groups'})
    }
    
    # Загрузка модели с custom_objects
    model = load_model(model_path, custom_objects=custom_objects)
    return model

# Тестирование кода, если файл запускается как основной скрипт
if __name__ == "__main__":
    model_path = 'my_unet_model.h5'  # Укажите правильный путь к файлу модели
    model = load_custom_unet_model(model_path)
    print("Модель успешно загружена.")
    model.summary()  # Вывод информации о модели
