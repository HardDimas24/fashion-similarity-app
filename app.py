import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import pickle
from streamlit.components.v1 import html

# === Загрузка и подготовка данных ===
CSV_PATH = "filtered_products.csv"
IMAGES_CSV = "images.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    images_df = pd.read_csv(IMAGES_CSV)
    df = df.merge(images_df, on='id', how='inner')  # Присоединяем ссылки
    df = df.head(5000)  # Только те, у кого есть фичи
    return df

df = load_data()

# === Загрузка признаков и FAISS индекса ===
features = np.load('features.npy')

def build_faiss_index(features):
    dim = features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(features)
    return index

faiss_index = build_faiss_index(features)

# === Фильтрация ===
def filter_data(df, gender=None, article_type=None, season=None, usage=None):
    if gender: df = df[df['gender'] == gender]
    if article_type: df = df[df['articleType'] == article_type]
    if season: df = df[df['season'] == season]
    if usage: df = df[df['usage'] == usage]
    return df

# === Рекомендации ===
def recommend_faiss(query_idx, faiss_index, top_k=10):
    D, I = faiss_index.search(features[query_idx].reshape(1, -1), top_k + 1)  # Запрашиваем на 1 больше, чтобы исключить сам товар
    I = I[0].tolist()  # Получаем список индексов
    
    # Убираем сам товар (если он в списке рекомендаций)
    I = [idx for idx in I if idx != query_idx]
    
    # Если после исключения осталось меньше top_k рекомендаций, добавляем недостающие
    while len(I) < top_k:
        I.append(I[-1])  # Добавляем последний элемент для заполнения (по логике всегда похожий)
    
    return I[:top_k]  # Обрезаем до top_k рекомендаций

def show_recommendations(indices, df):
    recommended_df = df.reset_index(drop=True)
    
    # Разбиваем на группы по 5 изображений в ряд
    rows = [indices[i:i + 5] for i in range(0, len(indices), 5)]
    
    # Отображаем товары по 5 в ряд
    for row in rows:
        cols = st.columns(5)  # Создаем 5 колонок для отображения
        for i, index in enumerate(row):
            if index < len(recommended_df):  # Проверяем, что индекс не выходит за пределы
                link = recommended_df.iloc[index]['link']
                cols[i].image(link, use_container_width=True)  # Отображаем изображение
                cols[i].write(f"Товар: {recommended_df.iloc[index]['id']}")  # Пишем ID товара

# === Функция для копирования ID ===
def copy_button(text):
    button_key = f"copy_button_{text}"
    
    js_code = f"""
    <script>
    function copyToClipboard_{text}() {{
        navigator.clipboard.writeText('{text}').then(() => {{
            const button = document.getElementById("{button_key}");
            const originalText = button.textContent;
            button.textContent = 'Скопировано!';
            
            setTimeout(() => {{
                button.textContent = originalText;
            }}, 1000);
        }});
    }}
    </script>
    <button id="{button_key}" 
            onclick="copyToClipboard_{text}()" 
            style="width:95%;padding:2px;margin:2px;cursor:pointer;">
        {'Копировать ID: ' + str(text)}
    </button>
    """
    html(js_code, height=40)

# === Интерфейс ===
st.title('Система рекомендаций товаров')

# Сохраняем состояние фильтров и товара в session_state
if 'gender' not in st.session_state:
    st.session_state.gender = "Все"
if 'article_type' not in st.session_state:
    st.session_state.article_type = "Все"
if 'season' not in st.session_state:
    st.session_state.season = "Все"
if 'usage' not in st.session_state:
    st.session_state.usage = "Все"
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Навигация по страницам
page = st.sidebar.radio("Выберите страницу", ("Фильтрация товаров", "Рекомендации", "История поиска"))

if page == "Фильтрация товаров":
    # Фильтрация товаров
    gender_options = ["Все"] + df['gender'].unique().tolist()
    st.session_state.gender = st.selectbox("Выберите пол", options=gender_options, index=gender_options.index(st.session_state.gender))

    article_type_options = ["Все"] + df['articleType'].unique().tolist()
    st.session_state.article_type = st.selectbox("Выберите тип товара", options=article_type_options, index=article_type_options.index(st.session_state.article_type))

    season_options = ["Все"] + df['season'].unique().tolist()
    st.session_state.season = st.selectbox("Выберите сезон", options=season_options, index=season_options.index(st.session_state.season))

    usage_options = ["Все"] + df['usage'].unique().tolist()
    st.session_state.usage = st.selectbox("Выберите использование", options=usage_options, index=usage_options.index(st.session_state.usage))

    apply_filters = st.button("Применить фильтры")
    if apply_filters:
        filtered_df = filter_data(df,
                                   st.session_state.gender if st.session_state.gender != "Все" else None,
                                   st.session_state.article_type if st.session_state.article_type != "Все" else None,
                                   st.session_state.season if st.session_state.season != "Все" else None,
                                   st.session_state.usage if st.session_state.usage != "Все" else None)
    else:
        filtered_df = df

    # Случайные товары
    st.subheader("Случайные товары")
    num_products = len(filtered_df)

    sample_size = min(num_products, 50)

    if sample_size == 0:
        st.warning("Нет товаров для отображения. Пожалуйста, примените другие фильтры.")

    random_df = filtered_df.sample(n=sample_size, random_state=42)
    for i in range(0, len(random_df), 5):
        cols = st.columns(5)
        for j in range(5):
            index = i + j
            if index < len(random_df):
                link = random_df.iloc[index]['link']
                product_id = random_df.iloc[index]['id']
                
                # Отображаем изображение
                cols[j].image(link, use_container_width=True)
                
                # Добавляем кнопку для копирования ID через HTML компонент
                with cols[j]:
                    copy_button(str(product_id))

elif page == "Рекомендации":
    # Рекомендации
    st.session_state.selected_product = st.selectbox("Выберите товар для рекомендаций", options=df['id'].tolist(), index=df['id'].tolist().index(st.session_state.selected_product) if st.session_state.selected_product is not None else 0)
    
    if st.session_state.selected_product:
        selected_idx = df[df['id'] == st.session_state.selected_product].index[0]
        recommendations = recommend_faiss(selected_idx, faiss_index)
        st.subheader(f"Рекомендации для товара {st.session_state.selected_product}")
        show_recommendations(recommendations, df)
        
        # Добавляем товар в историю поиска сразу
        if st.session_state.selected_product not in st.session_state.search_history:
            st.session_state.search_history.insert(0, st.session_state.selected_product)  # Добавляем в начало списка

elif page == "История поиска":
    # История поиска с фото по 7 в ряд
    st.subheader("История поиска")
    
    if st.session_state.search_history:
        # Разбиваем историю поиска на группы по 7 товаров
        rows = [st.session_state.search_history[i:i + 7] for i in range(0, len(st.session_state.search_history), 7)]
        
        # Отображаем товары по 7 в ряд
        for row in rows:
            cols = st.columns(7)  # Создаем 7 колонок для отображения
            for i, product_id in enumerate(row):
                product_row = df[df['id'] == product_id].iloc[0]
                link = product_row['link']
                cols[i].image(link, caption=f"Товар: {product_id}", use_container_width=True)
    else:
        st.write("История поиска пуста.")