import streamlit as st
import sys
from pathlib import Path
import pandas as pd

# Загрузка данных


# Определение корневого каталога проекта и добавление его в sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
df = pd.read_csv(str(project_root /'images/metrics_1.csv'))

# Информация о первой модели
st.write("# Localization INFO:")
st.write("Использовалась  модель - ResNet18 с добавлением блоков классификации и регрессии")
st.write("Модель обучалась на предсказание 3 классов")
st.write("Размер тренировочного датасета - 148 картинок")
st.write("Размер валидационного датасета - 38 картинок")
st.write("Применяемые аугментации:")
st.write("- T.RandomHorizontalFlip()")
st.write("- T.RandomRotation(10)")
st.write("- T.GaussianBlur(kernel_size=(5, 9), sigma=(0.3, 5.))")
st.write("- T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)")
st.image(str(project_root / 'images/training_history_2.png'), width=1000)
st.write("Время обучения модели - 100 эпох")
st.write("Метрики:")
st.write(df)
# st.write('Confusion matrix')
st.image(str(project_root / 'images/cf_1.png'))

# Информация о второй модели
st.write("# Blood Cells Classification INFO:")
st.write("Использовалась модель - ResNet18 и обучалась с нуля")
st.write("Модель обучалась на предсказание 4 классов")
st.write("Размер тренировочного датасета - 9957 картинок")
st.image(str(project_root / 'images/image3.png'))
st.write("Время обучения модели - 15 эпох = 40 минут")
st.write("Значения метрики f1 на последней эпохе: 0.915-train и 0.849-valid")
st.write('Confusion matrix')
st.image(str(project_root / 'images/image4.png'))

