import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from datetime import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Анализ недвижимости СПб", layout="wide")

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
    
    # Преобразование типов и очистка данных
    data['first_day_exposition'] = pd.to_datetime(data['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')
    data['year'] = data['first_day_exposition'].dt.year
    data['month'] = data['first_day_exposition'].dt.month
    data['price_per_sq_m'] = data['last_price'] / data['total_area']
    
    # Удаление выбросов
    data = data[(data['price_per_sq_m'] > 50000) & (data['price_per_sq_m'] < 500000)]
    data = data[(data['total_area'] > 20) & (data['total_area'] < 200)]
    
    # Заполнение пропусков
    data['ceiling_height'].fillna(data['ceiling_height'].median(), inplace=True)
    data['kitchen_area'].fillna(data['kitchen_area'].median(), inplace=True)
    data['living_area'].fillna(data['total_area'] * 0.5, inplace=True)
    return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Ошибка при загрузке данных: {str(e)}")
    st.stop()

# Сайдбар с фильтрами
st.sidebar.header("Фильтры")

# Исправленный выбор годов - используем slider для диапазона
year_range = st.sidebar.slider(
    "Диапазон годов",
    min_value=int(data['year'].min()),
    max_value=int(data['year'].max()),
    value=(int(data['year'].min()), int(data['year'].max()))
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

# Фильтрация данных - исправленная версия
filtered_data = data[
    (data['year'] >= year_range[0]) & 
    (data['year'] <= year_range[1]) &
    (data['total_area'] >= min_area) &
    (data['total_area'] <= max_area) &
    (data['last_price'] >= min_price) &
    (data['last_price'] <= max_price)
]

if 'Все' not in selected_rooms:
    filtered_data = filtered_data[filtered_data['rooms'].isin([int(x) for x in selected_rooms if x != 'Все'])]

# Настройка страницы
st.title("📊 Анализ продаж недвижимости в Санкт-Петербурге")

# Вкладки
tab1, tab2, tab3, tab4 = st.tabs(["📈 Общая статистика", "🏠 Анализ цен", "🌍 Геоанализ", "🔮 Прогнозирование"])

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
    
    # Динамика цен по годам
    st.subheader('Динамика цен за квадратный метр по годам')
    price_dynamics = filtered_data.groupby('year')['price_per_sq_m'].agg(['mean', 'median', 'count'])
    
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    ax1.plot(price_dynamics.index, price_dynamics['mean'], label='Средняя цена', marker='o')
    ax1.plot(price_dynamics.index, price_dynamics['median'], label='Медианная цена', marker='o')
    ax1.set_xlabel('Год')
    ax1.set_ylabel('Цена за м², руб')
    ax1.set_title('Динамика цен за квадратный метр')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    
        # Распределение по типам квартир
    st.subheader("Распределение по типам квартир")
    col1, col2 = st.columns(2)
    with col2:
        st.markdown("**Типы квартир**")
        type_counts = filtered_data[['studio', 'is_apartment', 'open_plan']].sum()
        russian_names = {
        'studio': 'Студия',
        'is_apartment': 'Апартаменты',
        'open_plan': 'Квартира'
    }
    
        fig_types = px.pie(values=type_counts, 
            names=type_counts.index.map(russian_names),
            height=400)
    
        # Положение
        fig_types.update_layout(
        legend=dict(
            orientation="v",  # вертикальная ориентация
            yanchor="top",   # привязка к верху
            y=1,             # позиция по Y
            xanchor="right", # привязка к правому краю
            x=1.1            # сдвиг вправо от графика
        )
    )
    
    st.plotly_chart(fig_types, use_container_width=True)

with tab2:
    st.header("Анализ характеристик")

        # Влияние характеристик на цену - в две колонки
    st.subheader('Влияние характеристик на цену')
    col1, col2 = st.columns(2)
    
    with col1:
        # Влияние количества комнат
            fig4, ax4 = plt.subplots(figsize=(6, 4)) 
            sns.boxplot(data=filtered_data, x='rooms', y='price_per_sq_m', ax=ax4)
            ax4.set_title('Цена по количеству комнат')
            ax4.set_xlabel('Количество комнат')
            ax4.set_ylabel('Цена за м², руб')
            st.pyplot(fig4)
    
            st.subheader(" Влияние параметров на цену за м²")
            corr_data = filtered_data[['price_per_sq_m', 'total_area', 'rooms', 'floor', 'kitchen_area']].corr()
            fig4 = px.line_polar(
            corr_data.drop('price_per_sq_m'),
            r=corr_data.loc['price_per_sq_m'].drop('price_per_sq_m'),
            theta=corr_data.columns.drop('price_per_sq_m'),
            line_close=True,
            height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("Анализ цен и площадей")
    col1, col2 = st.columns(2)
    
    with col1:
        try:
                st.markdown("**Распределение цен**")
                fig1 = plt.figure(figsize=(8,4))
                sns.boxplot(x=filtered_data['last_price'], color='skyblue')
                plt.xlabel('Цена (руб)')
                st.pyplot(fig1)
        except Exception as e:
                st.error(f"Ошибка при построении графика цен: {str(e)}")
    
    with col2:
        try:
                st.markdown("**Распределение площадей**")
                fig2 = plt.figure(figsize=(8,4))
                sns.boxplot(x=filtered_data['total_area'], color='lightgreen')
                plt.xlabel('Площадь (кв.м)')
                st.pyplot(fig2)
        except Exception as e:
                st.error(f"Ошибка при построении графика площадей: {str(e)}")

with tab3:
    st.header("Географический анализ")
    
    # Топ-10 самых дорогих/дешевых районов
    st.subheader(" Топ-10 районов по цене за м²")
        
    locality_prices = filtered_data.groupby('locality_name')['price_per_sq_m'].agg(['mean', 'count']).sort_values('mean')
    top10_expensive = locality_prices.nlargest(10, 'mean')
    top10_cheapest = locality_prices.nsmallest(10, 'mean')
        
    tab_exp, tab_cheap = st.tabs(["Самые дорогие", "Самые дешевые"])
        
with tab_exp:
    fig4a = px.bar(
    top10_expensive,
    x='mean',
    y=top10_expensive.index,
    orientation='h',
    labels={'mean': 'Средняя цена за м²', 'locality_name': 'Район'},
    height=400
    )
    st.plotly_chart(fig4a, use_container_width=True)
        
with tab_cheap:
    fig4b = px.bar(
    top10_cheapest,
    x='mean',
    y=top10_cheapest.index,
    orientation='h',
    labels={'mean': 'Средняя цена за м²', 'locality_name': 'Район'},
    height=400
    )
    st.plotly_chart(fig4b, use_container_width=True)


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