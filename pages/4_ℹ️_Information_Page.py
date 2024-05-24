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
st.write("Использовалась  модель  ResNet18 с добавлением блоков классификации и регрессии")
st.write("Модель обучалась на предсказание 3 классов")
st.write("Размер тренировочного датасета - 148 картинок")
st.write("Размер валидационного датасета - 38 картинок")
st.write("Применяемые аугментации:")
st.write("- T.RandomHorizontalFlip()")
st.write("- T.RandomRotation(10)")
st.write("- T.GaussianBlur(kernel_size=(5, 9), sigma=(0.3, 5.))")
st.write("- T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)")
st.image(str(project_root / 'images/t_h1.png'), width=1000)
st.write("Время обучения модели - 100 эпох")
st.write("Метрики:")
st.write(df)
# st.write('Confusion matrix')
st.image(str(project_root / 'images/cf_1.png'))

# Информация о второй модели
st.write("# Brain tumor object detection INFO:")
st.write("Использовалась модель -YOLOv5")
st.write("Модель обучалась на задачу бинрной детекции")
st.write("Размер тренировочного датасета:")
st.write("Для разреза Axial - 310 картинок и txt-файлов к ним")
st.write("Для разреза Coronal - 319 картинок и txt-файлов к ним")
st.write("Для разреза Sagittal - 264 картинок и txt-файлов к ним")

st.write("## Axial - yolov5s")
st.write("Время обучения модели - 40 эпох")
st.image(str(project_root / 'images/res_2-1.png'))
st.image(str(project_root / 'images/PR_2-1.png'))
# st.write("Значения метрики f1 на последней эпохе: 0.915-train и 0.849-valid")
st.write('Confusion matrix')
st.image(str(project_root / 'images/cf_2-1.png'))

st.write("## Coronal - yolov5m")
st.write("Время обучения модели - 50 эпох")
st.image(str(project_root / 'images/res_2-2.png'))
st.image(str(project_root / 'images/PR_2-2.png'))
# st.write("Значения метрики f1 на последней эпохе: 0.915-train и 0.849-valid")
st.write('Confusion matrix')
st.image(str(project_root / 'images/cf_2-2.png'))

st.write("## Sagittal - yolov5l")
st.write("Время обучения модели - 70 эпох")
st.image(str(project_root / 'images/res_2-3.png'))
st.image(str(project_root / 'images/PR_2-3.png'))
# st.write("Значения метрики f1 на последней эпохе: 0.915-train и 0.849-valid")
st.write('Confusion matrix')
st.image(str(project_root / 'images/cf_2-3.png'))