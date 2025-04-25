# 👗 Fashion Similarity App

Это рекомендательная система, которая по изображению товара находит визуально похожие товары. Приложение построено на базе извлечения признаков с изображений и поиска ближайших соседей.

## 📊 Что внутри

- EDA и предобработка данных
- Извлечение признаков с помощью моделей:
  - ResNet50
  - EfficientNet-B0
  - DenseNet121
- Поиск похожих изображений с использованием:
  - Косинусного расстояния
  - FAISS
  - KMeans кластеризации
- Веб-интерфейс на Streamlit

## 🗂️ Структура проекта

```
fashion-similarity-app/
│
├── app.py                  # Streamlit-приложение
├── лаба4.ipynb             # Исследование данных и построение модели
├── features.npy            # Извлечённые признаки изображений
├── filtered_products.csv   # Основной датасет товаров
├── images.csv              # Ссылки на изображения
└── README.md               # Этот файл
```

## 💡 Используемые технологии

	•	Python
	•	Streamlit
	•	NumPy, Pandas
	•	FAISS
	•	torchvision
	•	scikit-learn

## Как запустить?

streamlit run app.py

## Примеры использования

<img width="1323" alt="image" src="https://github.com/user-attachments/assets/a9330dfe-180b-4b77-92ef-1bc499c2a147" />
[url=https://online-letters.ru/][img]https://x-lines.ru/letters/i/cyrillicscript/0034/000000/20/0/4gnpbqgozxea3wcn4gypbcgto5emtwcxrdemoegoz9em7wf44napbp3y4gbpbxsosmembwcy4n9pbco.png[/img][/url]


<img width="1323" alt="image" src="https://github.com/user-attachments/assets/fee3f2fb-e8bd-4b20-b49a-9939b8b8dc2d" />
__рекомендации по id товара__


<img width="1323" alt="image" src="https://github.com/user-attachments/assets/d6ea47a4-6300-4b3b-aeed-195e0ca9aad2" />
*история поиска*


