import streamlit as st
st.set_page_config(page_title="Анализ недвижимости СПб", layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Загрузка данных
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/EgoVed/Research-of-ads-for-the-sale-of-apartments/refs/heads/master/real_estate_data.csv"
    data = pd.read_csv(url)
    
    # Преобразование даты
    data['date'] = pd.to_datetime(data['first_day_exposition'], errors='coerce')
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    
    # Расчет цены за кв.м
    data['price_per_sqm'] = data['last_price'] / data['total_area']
    
    # Очистка данных
    data = data[data['last_price'] < data['last_price'].quantile(0.99)]
    data = data[data['total_area'] < data['total_area'].quantile(0.99)]
    data = data[data['total_area'] > 10]  # Минимальная площадь
    
    # Заполнение пропусков
    data['rooms'] = data['rooms'].fillna(0).astype(int)
    data['floor'] = data['floor'].fillna(1).astype(int)
    data['floors_total'] = data['floors_total'].fillna(1).astype(int)
    
    return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {str(e)}")
    st.stop()

# Настройка страницы
st.title("📊 Анализ продаж недвижимости в Санкт-Петербурге")

# Сайдбар с фильтрами
st.sidebar.header("Фильтры")
selected_years = st.sidebar.multiselect(
    "Год публикации",
    options=sorted(data['year'].unique()),
    default=sorted(data['year'].unique())
)

min_area, max_area = st.sidebar.slider(
    "Площадь (кв.м)",
    min_value=float(data['total_area'].min()),
    max_value=float(data['total_area'].max()),
    value=(float(data['total_area'].min()), float(data['total_area'].max()))
)

min_price, max_price = st.sidebar.slider(
    "Цена (руб)",
    min_value=float(data['last_price'].min()),
    max_value=float(data['last_price'].max()),
    value=(float(data['last_price'].min()), float(data['last_price'].max()))
)

rooms_options = ['Все'] + sorted(data['rooms'].unique().tolist())
selected_rooms = st.sidebar.multiselect(
    "Количество комнат",
    options=rooms_options,
    default=['Все']
)

# Применение фильтров
filtered_data = data[
    (data['year'].isin(selected_years)) &
    (data['total_area'] >= min_area) &
    (data['total_area'] <= max_area) &
    (data['last_price'] >= min_price) &
    (data['last_price'] <= max_price)
]

if 'Все' not in selected_rooms:
    filtered_data = filtered_data[filtered_data['rooms'].isin(selected_rooms)]

# Вкладки
tab1, tab2, tab3, tab4 = st.tabs(["📊 Общая статистика", "🏢 Характеристики", "📈 Динамика", "🔮 Прогнозирование"])

with tab1:
    st.header("Основные метрики")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_price = np.mean(filtered_data['last_price'])
        st.metric("Средняя цена", f"{avg_price:,.0f} руб")
    with col2:
        avg_area = np.mean(filtered_data['total_area'])
        st.metric("Средняя площадь", f"{avg_area:.1f} кв.м")
    with col3:
        avg_ppsqm = np.mean(filtered_data['price_per_sqm'])
        st.metric("Цена за кв.м", f"{avg_ppsqm:,.0f} руб")
    
    st.subheader("Распределение цен и площадей")
    fig = make_subplots(rows=1, cols=2)
    fig.add_trace(go.Histogram(x=filtered_data['last_price'], name="Цены"), row=1, col=1)
    fig.add_trace(go.Histogram(x=filtered_data['total_area'], name="Площади"), row=1, col=2)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Зависимость цены от площади")
    fig = px.scatter(
        filtered_data,
        x='total_area',
        y='last_price',
        color='rooms',
        hover_data=['locality_name', 'floor', 'floors_total']
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Анализ характеристик")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Распределение по этажам")
        floor_counts = filtered_data['floor'].value_counts().sort_index()
        fig = px.bar(floor_counts, labels={'value': 'Количество', 'index': 'Этаж'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Типы квартир")
        type_counts = filtered_data[['studio', 'is_apartment', 'open_plan']].sum()
        fig = px.pie(values=type_counts, names=type_counts.index)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Наличие балконов")
    balcony_counts = filtered_data['balcony'].value_counts()
    fig = px.pie(values=balcony_counts, names=balcony_counts.index)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Динамика цен")
    
    # Группировка по времени
    time_data = filtered_data.groupby(['year', 'month']).agg({
        'last_price': 'mean',
        'price_per_sqm': 'mean',
        'total_area': 'mean'
    }).reset_index()
    time_data['date'] = pd.to_datetime(time_data['year'].astype(str) + '-' + time_data['month'].astype(str) + '-01')
    
    fig = px.line(
        time_data,
        x='date',
        y=['last_price', 'price_per_sqm'],
        labels={'value': 'Цена', 'variable': 'Тип цены'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Время экспозиции")
    fig = px.box(filtered_data, y='days_exposition')
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Прогнозирование цен")
    
    # Подготовка данных
    model_data = filtered_data.copy()
    features = ['total_area', 'rooms', 'floor', 'floors_total', 'kitchen_area', 'living_area']
    
    # Кодирование категориальных признаков
    le = LabelEncoder()
    if 'locality_name' in model_data.columns:
        model_data['locality_name'] = le.fit_transform(model_data['locality_name'])
        features.append('locality_name')
    
    # Удаление пропусков
    model_data = model_data.dropna(subset=features + ['last_price'])
    
    if len(model_data) > 100:  # Минимальное количество данных для модели
        X = model_data[features]
        y = model_data['last_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.metric("Средняя ошибка (MAE)", f"{mae:,.0f} руб")
        st.metric("Точность модели (R²)", f"{r2:.2f}")
        
        # Интерфейс для прогноза
        st.subheader("Онлайн-прогноз")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                area = st.number_input("Общая площадь (кв.м)", min_value=10.0, max_value=500.0, value=50.0)
                rooms = st.number_input("Количество комнат", min_value=0, max_value=10, value=1)
                floor = st.number_input("Этаж", min_value=1, max_value=100, value=5)
            with col2:
                floors_total = st.number_input("Всего этажей", min_value=1, max_value=100, value=10)
                kitchen_area = st.number_input("Площадь кухни (кв.м)", min_value=1.0, max_value=100.0, value=10.0)
                living_area = st.number_input("Жилая площадь (кв.м)", min_value=10.0, max_value=500.0, value=30.0)
            
            submitted = st.form_submit_button("Рассчитать")
            
            if submitted:
                input_data = pd.DataFrame({
                    'total_area': [area],
                    'rooms': [rooms],
                    'floor': [floor],
                    'floors_total': [floors_total],
                    'kitchen_area': [kitchen_area],
                    'living_area': [living_area]
                })
                
                if 'locality_name' in features:
                    # Используем наиболее частый район по умолчанию
                    common_locality = model_data['locality_name'].mode()[0]
                    input_data['locality_name'] = common_locality
                
                prediction = model.predict(input_data[features])
                price_per_sqm = prediction[0] / area
                
                st.success(f"Прогнозируемая цена: {prediction[0]:,.0f} руб")
                st.info(f"Цена за кв.м: {price_per_sqm:,.0f} руб")
    else:
        st.warning("Недостаточно данных для построения модели")

# Подвал
st.markdown("---")
st.markdown("Аналитическое приложение для анализа недвижимости СПб | Данные: GitHub")