# COMPUTER VISION

## Задачи

* Локализация объектов на изображении
* Бинарная детекция снимков опухолей головного мозга по трем срезам
* Бинарная сегментация лесов по аэро-снимкам

## Реализация и результаты
[Приложение](https://computer-vision-proj.streamlit.app/) развернуто на платформе Streamlit

### Локализация объектов на изображении
* Использовалась модель ResNet18 с добавлением блоков классификации и регрессии
* Модель обучалась на предсказание 3 классов
* Применялась аугментация
* Время обучения модели - 100 эпох
#### Метрики:
<img src="https://github.com/VerVelVel/cv_project/assets/156528877/029641e0-b179-4f49-b4c2-c1651b54e486" width="500">

### Бинарная детекция снимков опухолей головного мозга
#### Разрезы Axial 
* Использовалась модель YOLOv5s
* Время обучения модели - 40 эпох
##### Метрики:
<img src="images/res_2-1.png" width="500">
<img src="images/PR_2-1.png" width="500">

#### Разрезы Coronal 
* Использовалась модель YOLOv5m
* Время обучения модели - 70 эпох
##### Метрики:
<img src="images/res_2-2.png" width="500">
<img src="images/PR_2-2.png" width="500">

#### Разрезы Sagittal 
* Использовалась модель YOLOv5l
* Время обучения модели - 50 эпох
##### Метрики:
<img src="images/res_2-3.png" width="500">
<img src="images/PR_2-3.png" width="500">
<img src="https://github.com/VerVelVel/cv_project/assets/156528877/b189c6a5-7cb1-4b5d-9bcf-1a9b522de7e2" width="500">

### Бинарная сегментация лесов по аэро-снимкам
* Использовалась модель Unet
* Время обучения модели - 30 эпох
##### Метрики:
<img src="images/unet_loss.png" width="500">
<img src="images/unet_acc.png" width="500">
<img src="https://github.com/VerVelVel/cv_project/assets/156528877/e9674f0e-4bef-4884-8d96-3e69c0e7b18f" width="500">
