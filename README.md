# COMPUTER VISION

## Задачи

* Локализация объектов на изображении
* Бинарная детекция снимков опухолей головного мозга по трем срезам
* Бинарная сегментация лесов по аэро-снимкам

## Реализация и результаты

### Локализация объектов на изображении
* Использовалась модель ResNet18 с добавлением блоков классификации и регрессии
* Модель обучалась на предсказание 3 классов
* Применяелась аугментация
* Время обучения модели - 100 эпох
#### Метрики:
![image](https://github.com/VerVelVel/cv_project/assets/156528877/029641e0-b179-4f49-b4c2-c1651b54e486)

### Бинарная детекция снимков опухолей головного мозга
#### Разрезы Axial 
* Использовалась модель YOLOv5s
* Время обучения модели - 40 эпох
##### Метрики:
![Основные показатели](images/res_2-1.png)
![Precision-Recall Curve](images/PR_2-1.png)

#### Разрезы Coronal 
* Использовалась модель YOLOv5m
* Время обучения модели - 70 эпох
##### Метрики:
![Основные показатели](images/res_2-2.png)
![Precision-Recall Curve](images/PR_2-2.png)

#### Разрезы Sagittal 
* Использовалась модель YOLOv5l
* Время обучения модели - 50 эпох
##### Метрики:
![Основные показатели](images/res_2-3.png)
![Precision-Recall Curve](images/PR_2-3.png)
![Пример работы модели](https://github.com/VerVelVel/cv_project/assets/156528877/b189c6a5-7cb1-4b5d-9bcf-1a9b522de7e2)


### Бинарная сегментация лесов по аэро-снимкам
* Использовалась модель Unet
* Время обучения модели - 30 эпох
##### Метрики:
![Loss-функция](images/unet_loss.png)
![Accuracy](images/unet_acc.png) 
![Пример работы модели](https://github.com/VerVelVel/cv_project/assets/156528877/e9674f0e-4bef-4884-8d96-3e69c0e7b18f)