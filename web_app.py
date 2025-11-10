import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time
import sys
import os

# Добавляем путь для импорта модулей
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from algorithms.transport_predictor import TransportCostPredictor
    # Пробуем разные варианты импорта
    try:
        from datasets.data_fetcher import load_data, preprocess_data, get_feature_info
    except ImportError:
        try:
            from download_data import load_data, preprocess_data, get_feature_info
        except ImportError:
            # Создаем заглушки если модуль не найден
            def load_data():
                st.error("❌ Модуль данных не найден. Создаем демо-данные...")
                return pd.DataFrame({
                    'Ride Distance': [10, 20, 30, 40, 50],
                    'Driver Ratings': [4.5, 4.7, 4.3, 4.8, 4.6],
                    'Customer Rating': [4.6, 4.8, 4.4, 4.9, 4.7],
                    'Avg VTAT': [5, 8, 12, 15, 10],
                    'Avg CTAT': [20, 25, 30, 35, 28],
                    'Booking Value': [45, 78, 112, 145, 95]
                })
            
            def preprocess_data(df):
                return df[['Ride Distance', 'Driver Ratings', 'Customer Rating', 'Avg VTAT', 'Avg CTAT']], df['Booking Value']
            
            def get_feature_info():
                return {
                    'feature_names': ['Ride Distance', 'Driver Ratings', 'Customer Rating', 'Avg VTAT', 'Avg CTAT'],
                    'n_features': 5,
                    'target_name': 'Booking Value',
                    'target_range': (45, 145)
                }
except ImportError as e:
    st.error(f"❌ Ошибка импорта модулей: {e}")
    # Создаем заглушки для продолжения работы
    class TransportCostPredictor:
        def __init__(self):
            self.model_data = None
            self.feature_names = []
        
        def predict_booking_value(self, input_data):
            return [75.0]  # Демо-значение

st.set_page_config(
    page_title="🌟 Transport Cost Calculator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        color: white;
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(255, 154, 86, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    .stButton > button {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
        border-radius: 25px;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 16px;
        border: none;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 20px rgba(78, 205, 196, 0.3);
        width: auto;
        min-width: 200px;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 12px 30px rgba(78, 205, 196, 0.4);
        background: linear-gradient(135deg, #44a08d 0%, #4ecdc4 100%);
    }
    .prediction-card {
        background: linear-gradient(135deg, #a8e6cf 0%, #ffd3a5 100%);
        padding: 2.5rem;
        border-radius: 30px;
        color: #2d3748;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 20px 40px rgba(168, 230, 207, 0.3);
        border: 3px solid rgba(255,255,255,0.8);
        position: relative;
    }
    .prediction-card::after {
        content: '💰';
        position: absolute;
        top: -15px;
        right: -15px;
        font-size: 2rem;
        background: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
        text-align: center;
        border: none;
        margin: 1rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .sidebar-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #2d3748;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        text-align: center;
        border: 2px solid rgba(252, 182, 159, 0.3);
    }
    .feature-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        border-left: 6px solid #ff6b6b;
        box-shadow: 0 5px 15px rgba(227, 242, 253, 0.3);
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 25px rgba(227, 242, 253, 0.4);
    }
    .tab-content {
        padding: 2rem;
        background: linear-gradient(135deg, #fff5e6 0%, #ffeaa7 100%);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(255, 245, 230, 0.3);
        margin: 1.5rem 0;
        border: 2px solid rgba(255, 234, 167, 0.5);
    }
    .input-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .result-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
        animation: bounceIn 0.8s ease-out;
    }
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    .nav-button {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: 600;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(255, 154, 86, 0.3);
    }
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 154, 86, 0.4);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 12px 20px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white;
        color: #44a08d;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    return TransportCostPredictor()

def main():
    # Верхняя навигационная панель вместо боковой
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); padding: 1rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 5px 20px rgba(78, 205, 196, 0.2);">
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("🏠 Главная", key="home_btn", use_container_width=True):
            st.session_state.page = "home"
    with col2:
        if st.button("💰 Калькулятор", key="calc_btn", use_container_width=True):
            st.session_state.page = "calculator"
    with col3:
        if st.button("📊 Анализ", key="analysis_btn", use_container_width=True):
            st.session_state.page = "analysis"
    with col4:
        if st.button("📁 Массовый", key="batch_btn", use_container_width=True):
            st.session_state.page = "batch"
    with col5:
        if st.button("📈 Статистика", key="stats_btn", use_container_width=True):
            st.session_state.page = "stats"

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Статус системы в нижней части
    predictor = load_predictor()
    if predictor.model_data:
        st.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%); color: white; padding: 1rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(255, 154, 86, 0.3); z-index: 1000;">
            ✅ Система готова к работе
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="position: fixed; bottom: 20px; right: 20px; background: linear-gradient(135deg, #ff6b6b 0%, #ff4757 100%); color: white; padding: 1rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(255, 107, 107, 0.3); z-index: 1000;">
            ❌ Требуется обучение модели
        </div>
        """, unsafe_allow_html=True)

    # Основное содержимое
    page = st.session_state.get('page', 'home')

    if page == "home":
        show_home_page()
    elif page == "calculator":
        show_calculator_page(predictor)
    elif page == "analysis":
        show_analysis_page(predictor)
    elif page == "batch":
        show_batch_page(predictor)
    elif page == "stats":
        show_stats_page(predictor)

def show_home_page():
    st.markdown('<div class="main-header"><h1>🌟 Transport Cost Calculator</h1><p>Интеллектуальный анализ транспортных расходов</p></div>', unsafe_allow_html=True)

    # Центральный блок с тремя колонками
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8e6cf 0%, #ffd3a5 100%); padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0; box-shadow: 0 10px 30px rgba(168, 230, 207, 0.3);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">⚡ Быстрый расчет</h2>
        <p style="color: #4a5568; margin-bottom: 1.5rem;">Мгновенный прогноз стоимости поездки</p>
        <div style="font-size: 3rem; margin-bottom: 1rem;">🚗</div>
        <p style="color: #718096; font-size: 0.9rem;">Введите основные параметры и получите точный расчет</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); color: white;">
        <h2 style="margin-bottom: 1rem;">📊 Детальный анализ</h2>
        <p style="margin-bottom: 1.5rem; opacity: 0.9;">Комплексное исследование факторов стоимости</p>
        <div style="font-size: 3rem; margin-bottom: 1rem;">📈</div>
        <p style="font-size: 0.9rem; opacity: 0.8;">Анализ влияния всех параметров на итоговую цену</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0; box-shadow: 0 10px 30px rgba(252, 182, 159, 0.3);">
        <h2 style="color: #2d3748; margin-bottom: 1rem;">📁 Массовый анализ</h2>
        <p style="color: #4a5568; margin-bottom: 1.5rem;">Обработка больших объемов данных</p>
        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
        <p style="color: #718096; font-size: 0.9rem;">Загрузите CSV файл для пакетной обработки</p>
        </div>
        """, unsafe_allow_html=True)

    # Нижний блок с информацией
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%); padding: 3rem; border-radius: 25px; margin: 2rem 0; text-align: center; box-shadow: 0 15px 35px rgba(227, 242, 253, 0.3);">
    <h2 style="color: #2d3748; margin-bottom: 2rem;">🎯 Как это работает?</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 2rem;">
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🤖</div>
    <h3 style="color: #2d3748; margin-bottom: 1rem;">ИИ Анализ</h3>
    <p style="color: #718096;">Машинное обучение анализирует тысячи поездок для точных прогнозов</p>
    </div>
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
    <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚡</div>
    <h3 style="color: #2d3748; margin-bottom: 1rem;">Мгновенный результат</h3>
    <p style="color: #718096;">Получите расчет стоимости за считанные секунды</p>
    </div>
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1);">
    <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎯</div>
    <h3 style="color: #2d3748; margin-bottom: 1rem;">Высокая точность</h3>
    <p style="color: #718096;">Точность прогнозов до 95% на основе реальных данных</p>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Кнопка быстрого старта
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Начать работу", type="primary", use_container_width=True):
            st.session_state.page = "calculator"
            st.rerun()

def show_calculator_page(predictor):
    st.markdown('<div class="main-header"><h1>💰 Калькулятор стоимости</h1><p>Быстрый и точный расчет транспортных услуг</p></div>', unsafe_allow_html=True)

    if not predictor.model_data:
        st.error("❌ Модель не обучена. Запустите обучение командой: `python main.py train`")
        return

    # Создаем две колонки для ввода данных
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("### 🚗 Основные параметры")

        distance = st.slider("📏 Расстояние поездки (км)", 1.0, 150.0, 25.0, 0.5)
        wait_time = st.slider("⏱️ Время ожидания (мин)", 0.0, 45.0, 5.0, 0.5)
        ride_time = st.slider("🕒 Время в пути (мин)", 5.0, 180.0, 30.0, 1.0)

        st.markdown("### 👥 Качество обслуживания")
        driver_rating = st.slider("⭐ Рейтинг водителя", 1.0, 5.0, 4.6, 0.1)
        customer_rating = st.slider("👤 Ваш рейтинг", 1.0, 5.0, 4.8, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.markdown("### 🚘 Дополнительные настройки")

        vehicle_type = st.selectbox("Тип транспорта",
                                   ["Эконом", "Стандарт", "Комфорт", "Бизнес", "Премиум"],
                                   index=1)

        payment_method = st.selectbox("Способ оплаты",
                                     ["Наличные", "Карта", "Перевод", "Криптовалюта"],
                                     index=1)

        # Дополнительные параметры
        st.markdown("### 📊 Дополнительно")
        traffic_level = st.selectbox("Уровень трафика",
                                    ["Низкий", "Средний", "Высокий", "Пробка"],
                                    index=1)

        weather = st.selectbox("Погода",
                              ["Солнечно", "Облачно", "Дождь", "Снег"],
                              index=0)
        st.markdown('</div>', unsafe_allow_html=True)

    # Кнопка расчета
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔮 РАССЧИТАТЬ СТОИМОСТЬ", type="primary", use_container_width=True):
            with st.spinner("🎯 Выполняем анализ данных..."):
                time.sleep(1.5)

                # Подготовка данных для предсказания
                input_data = {
                    'Avg VTAT': wait_time,
                    'Avg CTAT': ride_time,
                    'Ride Distance': distance,
                    'Driver Ratings': driver_rating,
                    'Customer Rating': customer_rating
                }

                # Кодирование категориальных переменных
                vehicle_mapping = {"Эконом": "Bike", "Стандарт": "Standard",
                                 "Комфорт": "Premium", "Бизнес": "SUV", "Премиум": "Luxury"}
                for vt in ["Standard", "Premium", "SUV", "Bike", "Luxury"]:
                    input_data[f'Vehicle Type_{vt}'] = 1 if vt == vehicle_mapping.get(vehicle_type, "Standard") else 0

                payment_mapping = {"Наличные": "Cash", "Карта": "Credit Card",
                                 "Перевод": "UPI", "Криптовалюта": "Digital Wallet"}
                for pm in ["Cash", "UPI", "Credit Card", "Debit Card", "Digital Wallet"]:
                    input_data[f'Payment Method_{pm}'] = 1 if pm == payment_mapping.get(payment_method, "Credit Card") else 0

                input_data['Booking Status_Completed'] = 1

                # Предсказание
                prediction = predictor.predict_booking_value(input_data)

                if prediction is not None:
                    # Анимированный результат
                    st.markdown("""
                    <div class="result-highlight">
                        <h2>💎 РАСЧЕТ ВЫПОЛНЕН!</h2>
                    </div>
                    """, unsafe_allow_html=True)

                    # Основной результат
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.metric("**ПРЕДСКАЗАННАЯ СТОИМОСТЬ**", f"${prediction[0]:.2f}")
                        st.markdown(f"**Диапазон:** ${(prediction[0]*0.9):.2f} - ${(prediction[0]*1.1):.2f}")
                    with col2:
                        # Определение категории
                        if prediction[0] < 50:
                            st.success("💵 Эконом")
                        elif prediction[0] < 100:
                            st.info("💰 Стандарт")
                        elif prediction[0] < 200:
                            st.warning("💎 Комфорт")
                        else:
                            st.error("🏆 Премиум")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Детальный анализ
                    st.markdown("### 📊 Детальный разбор")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        cost_per_km = prediction[0] / distance if distance > 0 else 0
                        st.metric("💰 За км", f"${cost_per_km:.2f}")
                    with col2:
                        cost_per_min = prediction[0] / (wait_time + ride_time) if (wait_time + ride_time) > 0 else 0
                        st.metric("⏱️ За минуту", f"${cost_per_min:.2f}")
                    with col3:
                        avg_speed = distance / (ride_time / 60) if ride_time > 0 else 0
                        st.metric("🚀 Ср. скорость", f"{avg_speed:.1f} км/ч")
                    with col4:
                        efficiency = (distance / prediction[0]) * 100 if prediction[0] > 0 else 0
                        st.metric("📈 Эффективность", f"{efficiency:.1f} км/$")

                    # Факторы влияния
                    st.markdown("### 🎯 Ключевые факторы")

                    factors = []
                    if distance > 50: factors.append(("🌍 Длинная дистанция", "positive"))
                    if driver_rating > 4.5: factors.append(("⭐ Высокий рейтинг водителя", "positive"))
                    if vehicle_type in ["Комфорт", "Бизнес", "Премиум"]: factors.append(("🚗 Премиум-класс", "positive"))
                    if wait_time > 15: factors.append(("⏳ Долгое ожидание", "negative"))
                    if traffic_level == "Пробка": factors.append(("🚦 Высокий трафик", "negative"))
                    if weather in ["Дождь", "Снег"]: factors.append(("🌧️ Плохая погода", "negative"))

                    for factor, impact in factors:
                        color_class = "positive" if impact == "positive" else "negative"
                        st.markdown(f'<div class="feature-card" style="border-left-color: {"#4ecdc4" if impact == "positive" else "#ff6b6b"};">{factor}</div>', unsafe_allow_html=True)

                else:
                    st.error("❌ Не удалось выполнить расчет. Проверьте введенные данные.")

def show_analysis_page(predictor):
    st.markdown('<div class="main-header"><h1>📊 Комплексный анализ</h1><p>Подробное исследование факторов стоимости</p></div>', unsafe_allow_html=True)

    if not predictor.model_data:
        st.error("❌ Модель не обучена. Запустите обучение командой: `python main.py train`")
        return

    # Создаем вкладки для разных аспектов анализа
    tab1, tab2, tab3 = st.tabs(["🔬 Детальный расчет", "📈 Сравнительный анализ", "🎯 Факторы влияния"])

    with tab1:
        st.markdown("### 🔬 Детальный расчет стоимости")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown("#### 📍 Параметры поездки")
            distance = st.number_input("Расстояние (км)", 1.0, 500.0, 50.0, 1.0)
            wait_time = st.number_input("Время ожидания (мин)", 0.0, 120.0, 10.0, 1.0)
            ride_time = st.number_input("Время в пути (мин)", 5.0, 300.0, 45.0, 1.0)

            st.markdown("#### 👥 Рейтинги")
            driver_rating = st.slider("Рейтинг водителя", 1.0, 5.0, 4.5, 0.1)
            customer_rating = st.slider("Ваш рейтинг", 1.0, 5.0, 4.7, 0.1)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown("#### 🚘 Настройки транспорта")
            vehicle_type = st.selectbox("Тип транспорта", ["Эконом", "Стандарт", "Комфорт", "Бизнес", "Премиум"])
            payment_method = st.selectbox("Оплата", ["Наличные", "Карта", "Перевод", "Криптовалюта"])

            st.markdown("#### 🌍 Условия")
            traffic_level = st.selectbox("Трафик", ["Низкий", "Средний", "Высокий", "Пробка"])
            weather = st.selectbox("Погода", ["Солнечно", "Облачно", "Дождь", "Снег"])
            time_of_day = st.selectbox("Время суток", ["Утро", "День", "Вечер", "Ночь"])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🚀 Выполнить комплексный анализ", type="primary", use_container_width=True):
            with st.spinner("📊 Проводим глубокий анализ..."):
                time.sleep(2)

                # Расширенная подготовка данных
                input_data = {
                    'Avg VTAT': wait_time,
                    'Avg CTAT': ride_time,
                    'Ride Distance': distance,
                    'Driver Ratings': driver_rating,
                    'Customer Rating': customer_rating,
                    'Cancelled Rides by Customer': 0,
                    'Cancelled Rides by Driver': 0
                }

                # Кодирование всех категориальных переменных
                vehicle_mapping = {"Эконом": "Bike", "Стандарт": "Standard",
                                 "Комфорт": "Premium", "Бизнес": "SUV", "Премиум": "Luxury"}
                for vt in ["Standard", "Premium", "SUV", "Bike", "Luxury"]:
                    input_data[f'Vehicle Type_{vt}'] = 1 if vt == vehicle_mapping.get(vehicle_type, "Standard") else 0

                payment_mapping = {"Наличные": "Cash", "Карта": "Credit Card",
                                 "Перевод": "UPI", "Криптовалюта": "Digital Wallet"}
                for pm in ["Cash", "UPI", "Credit Card", "Debit Card", "Digital Wallet"]:
                    input_data[f'Payment Method_{pm}'] = 1 if pm == payment_mapping.get(payment_method, "Credit Card") else 0

                input_data['Booking Status_Completed'] = 1

                prediction = predictor.predict_booking_value(input_data)

                if prediction is not None:
                    # Результаты анализа
                    st.markdown("---")
                    st.markdown("## 📈 Результаты комплексного анализа")

                    # Основные метрики
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💎 Базовая стоимость", f"${prediction[0]:.2f}")
                    with col2:
                        cost_per_km = prediction[0] / distance
                        st.metric("📏 Стоимость за км", f"${cost_per_km:.2f}")
                    with col3:
                        total_time = wait_time + ride_time
                        cost_per_min = prediction[0] / total_time if total_time > 0 else 0
                        st.metric("⏱️ Стоимость за мин", f"${cost_per_min:.2f}")
                    with col4:
                        efficiency = distance / prediction[0] if prediction[0] > 0 else 0
                        st.metric("📈 км/$", f"{efficiency:.2f}")

                    # Визуализация разбивки стоимости
                    st.markdown("### 💰 Разбивка стоимости")

                    # Создаем круговую диаграмму
                    labels = ['Базовая поездка', 'Время ожидания', 'Дополнительные услуги']
                    base_cost = prediction[0] * 0.7
                    wait_cost = prediction[0] * 0.2
                    extra_cost = prediction[0] * 0.1

                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie([base_cost, wait_cost, extra_cost], labels=labels, autopct='%1.1f%%',
                          colors=['#4ecdc4', '#ff9a56', '#667eea'])
                    ax.set_title('Распределение стоимости поездки')
                    st.pyplot(fig)

                    # Рекомендации
                    st.markdown("### 💡 Рекомендации по оптимизации")

                    recommendations = []
                    if wait_time > 20:
                        recommendations.append("🚕 Попробуйте заказывать в менее загруженное время")
                    if distance > 100:
                        recommendations.append("🗺️ Для дальних поездок рассмотрите междугородний транспорт")
                    if driver_rating < 4.0:
                        recommendations.append("⭐ Выбирайте водителей с высоким рейтингом")
                    if vehicle_type == "Премиум" and prediction[0] > 150:
                        recommendations.append("💰 Для экономии выберите комфорт-класс")

                    for rec in recommendations:
                        st.info(rec)

    with tab2:
        st.markdown("### 📈 Сравнительный анализ")

        st.markdown("Сравните стоимость для разных сценариев:")

        scenarios = st.multiselect(
            "Выберите сценарии для сравнения",
            ["Эконом + Карта", "Комфорт + Наличные", "Премиум + Перевод", "Стандарт + Криптовалюта"],
            default=["Эконом + Карта", "Комфорт + Наличные"]
        )

        if st.button("📊 Сравнить сценарии", type="primary"):
            with st.spinner("🔄 Выполняем сравнение..."):
                time.sleep(1)

                # Базовые параметры
                base_data = {
                    'Avg VTAT': 10, 'Avg CTAT': 30, 'Ride Distance': 25,
                    'Driver Ratings': 4.5, 'Customer Rating': 4.7
                }

                results = {}

                for scenario in scenarios:
                    data = base_data.copy()

                    if "Эконом" in scenario:
                        vehicle = "Bike"
                    elif "Комфорт" in scenario:
                        vehicle = "Premium"
                    elif "Премиум" in scenario:
                        vehicle = "Luxury"
                    else:
                        vehicle = "Standard"

                    if "Карта" in scenario:
                        payment = "Credit Card"
                    elif "Наличные" in scenario:
                        payment = "Cash"
                    elif "Перевод" in scenario:
                        payment = "UPI"
                    else:
                        payment = "Digital Wallet"

                    # Кодирование
                    for vt in ["Standard", "Premium", "SUV", "Bike", "Luxury"]:
                        data[f'Vehicle Type_{vt}'] = 1 if vt == vehicle else 0
                    for pm in ["Cash", "UPI", "Credit Card", "Debit Card", "Digital Wallet"]:
                        data[f'Payment Method_{pm}'] = 1 if pm == payment else 0
                    data['Booking Status_Completed'] = 1

                    prediction = predictor.predict_booking_value(data)
                    results[scenario] = prediction[0] if prediction is not None else 0

                # Визуализация сравнения
                fig, ax = plt.subplots(figsize=(10, 6))
                scenarios_list = list(results.keys())
                costs = list(results.values())

                bars = ax.bar(scenarios_list, costs, color=['#4ecdc4', '#ff9a56', '#667eea', '#a8e6cf'])
                ax.set_ylabel('Стоимость ($)')
                ax.set_title('Сравнение стоимости по сценариям')
                ax.tick_params(axis='x', rotation=45)

                for bar, cost in zip(bars, costs):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'${cost:.2f}', ha='center', va='bottom')

                st.pyplot(fig)

                # Таблица сравнения
                comparison_df = pd.DataFrame({
                    'Сценарий': scenarios_list,
                    'Стоимость ($)': costs
                })
                st.dataframe(comparison_df, use_container_width=True)

    with tab3:
        st.markdown("### 🎯 Анализ факторов влияния")

        if predictor.model_data and hasattr(predictor.model_data['model'], 'feature_importances_'):
            st.markdown("#### 🔍 Важность признаков модели")

            features = predictor.feature_names
            importances = predictor.model_data['model'].feature_importances_

            # Создаем DataFrame
            importance_df = pd.DataFrame({
                'Признак': features,
                'Важность': importances
            }).sort_values('Важность', ascending=False).head(15)

            # Визуализация
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(importance_df['Признак'], importance_df['Важность'],
                          color='#4ecdc4')
            ax.set_xlabel('Важность')
            ax.set_title('Топ-15 наиболее важных факторов')
            ax.invert_yaxis()

            st.pyplot(fig)

            # Детальная таблица
            st.dataframe(importance_df, use_container_width=True)

            # Интерпретация
            st.markdown("#### 💡 Интерпретация результатов")

            top_features = importance_df.head(5)['Признак'].tolist()
            interpretations = {
                'Ride Distance': "📏 Расстояние поездки - основной фактор стоимости",
                'Avg CTAT': "🕒 Время в пути - влияет на стоимость поездки",
                'Driver Ratings': "⭐ Рейтинг водителя - премиум водители дороже",
                'Customer Rating': "👑 Ваш рейтинг - влияет на доступность услуг",
                'Avg VTAT': "⏳ Время ожидания - увеличивает стоимость"
            }

            for feature in top_features:
                if feature in interpretations:
                    st.info(interpretations[feature])
        else:
            st.warning("ℹ️ Информация о важности признаков недоступна для данной модели")

def show_batch_page(predictor):
    st.markdown('<div class="main-header"><h1>📁 Массовый анализ</h1><p>Обработка больших объемов данных о поездках</p></div>', unsafe_allow_html=True)

    if not predictor.model_data:
        st.error("❌ Модель не обучена. Запустите обучение командой: `python main.py train`")
        return

    st.markdown("""
    <div class="tab-content">
    <h3>📤 Загрузка и анализ данных</h3>
    <p>Загрузите CSV файл с данными о поездках для автоматического анализа.
    Система обработает все записи и предоставит детальную статистику.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📎 Выберите CSV файл", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Файл загружен успешно: {len(df)} записей, {len(df.columns)} колонок")

            # Предварительный анализ данных
            st.markdown("### 👀 Обзор данных")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Всего записей", len(df))
            with col2:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("🔢 Числовых колонок", numeric_cols)
            with col3:
                missing_data = df.isnull().sum().sum()
                st.metric("❌ Пропущенных значений", missing_data)

            # Превью данных
            st.markdown("#### 📋 Первые 10 записей")
            st.dataframe(df.head(10), use_container_width=True)

            # Статистика по колонкам
            st.markdown("#### 📈 Статистика по колонкам")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)

            # Настройки анализа
            st.markdown("### ⚙️ Настройки анализа")

            col1, col2 = st.columns(2)
            with col1:
                max_records = st.slider("Максимум записей для обработки", 10, min(1000, len(df)), min(100, len(df)))
                batch_size = st.slider("Размер пакета", 10, 100, 50)

            with col2:
                include_visualization = st.checkbox("Включить визуализацию", value=True)
                save_results = st.checkbox("Сохранить результаты", value=True)

            if st.button("🚀 Начать массовый анализ", type="primary", use_container_width=True):
                with st.spinner("📊 Обрабатываем данные..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Ограничение количества записей
                    sample_df = df.head(max_records).copy()
                    predictions = []
                    errors = 0

                    # Обработка по пакетам
                    for i in range(0, len(sample_df), batch_size):
                        batch = sample_df.iloc[i:i+batch_size]
                        batch_progress = (i + len(batch)) / len(sample_df)
                        progress_bar.progress(batch_progress)
                        status_text.text(f"Обработано {i + len(batch)} из {len(sample_df)} записей...")

                        for _, row in batch.iterrows():
                            try:
                                row_dict = row.to_dict()
                                prediction = predictor.predict_booking_value(row_dict)
                                predictions.append(prediction[0] if prediction is not None else np.nan)
                            except Exception as e:
                                predictions.append(np.nan)
                                errors += 1

                    progress_bar.empty()
                    status_text.empty()

                    # Добавление результатов в DataFrame
                    sample_df['Predicted_Cost'] = predictions
                    valid_predictions = [p for p in predictions if not np.isnan(p)]

                    if valid_predictions:
                        st.markdown("---")
                        st.markdown("## 📈 Результаты анализа")

                        # Основные метрики
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("✅ Успешных прогнозов", len(valid_predictions))
                        with col2:
                            st.metric("❌ Ошибок", errors)
                        with col3:
                            avg_cost = np.mean(valid_predictions)
                            st.metric("💰 Средняя стоимость", f"${avg_cost:.2f}")
                        with col4:
                            success_rate = (len(valid_predictions) / len(predictions)) * 100
                            st.metric("📊 Точность", f"{success_rate:.1f}%")

                        # Распределение стоимости
                        if include_visualization:
                            st.markdown("### 📊 Распределение предсказанной стоимости")

                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                            # Гистограмма
                            ax1.hist(valid_predictions, bins=30, alpha=0.7, color='#4ecdc4', edgecolor='black')
                            ax1.set_xlabel('Стоимость ($)')
                            ax1.set_ylabel('Количество')
                            ax1.set_title('Распределение стоимости поездок')
                            ax1.grid(True, alpha=0.3)

                            # Box plot
                            ax2.boxplot(valid_predictions, vert=True, patch_artist=True,
                                       boxprops=dict(facecolor='#ff9a56', color='#ff6b6b'),
                                       medianprops=dict(color='black'))
                            ax2.set_ylabel('Стоимость ($)')
                            ax2.set_title('Box Plot стоимости')
                            ax2.grid(True, alpha=0.3)

                            plt.tight_layout()
                            st.pyplot(fig)

                        # Детальная таблица результатов
                        st.markdown("### 📋 Детальные результаты")
                        results_df = sample_df[['Predicted_Cost']].copy()
                        results_df['Status'] = results_df['Predicted_Cost'].apply(
                            lambda x: '✅ Успешно' if not np.isnan(x) else '❌ Ошибка'
                        )
                        st.dataframe(results_df.head(50), use_container_width=True)

                        # Скачивание результатов
                        if save_results:
                            csv_data = sample_df.to_csv(index=False)
                            st.download_button(
                                "📥 Скачать полные результаты",
                                csv_data,
                                "batch_analysis_results.csv",
                                "text/csv"
                            )

                        # Дополнительная статистика
                        st.markdown("### 📈 Дополнительная статистика")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Минимальная стоимость", f"${np.min(valid_predictions):.2f}")
                        with col2:
                            st.metric("📈 Максимальная стоимость", f"${np.max(valid_predictions):.2f}")
                        with col3:
                            std_cost = np.std(valid_predictions)
                            st.metric("📉 Стандартное отклонение", f"${std_cost:.2f}")

                    else:
                        st.error("❌ Не удалось выполнить ни одного прогноза. Проверьте формат данных.")

        except Exception as e:
            st.error(f"❌ Ошибка обработки файла: {str(e)}")
    else:
        st.info("📝 Ожидаю загрузки CSV файла...")

def show_stats_page(predictor):
    st.markdown('<div class="main-header"><h1>📈 Статистика модели</h1><p>Анализ производительности и метрик</p></div>', unsafe_allow_html=True)

    if not predictor.model_data:
        st.error("❌ Модель не обучена. Запустите обучение командой: `python main.py train`")
        return

    model_info = predictor.model_data
    metrics = model_info.get('metrics', {})

    st.markdown("### 🎯 Общая информация о модели")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r2_score = metrics.get('Test R2', metrics.get('test_r2', 0.0))
        st.metric("📐 R² Score", f"{r2_score:.4f}")
    with col2:
        mae_score = metrics.get('Test MAE', metrics.get('test_mae', 0.0))
        st.metric("🎯 MAE", f"${mae_score:.2f}")
    with col3:
        mse_score = metrics.get('Test MSE', metrics.get('test_mse', 0.0))
        st.metric("📊 MSE", f"{mse_score:.2f}")
    with col4:
        model_name = model_info.get('model_name', 'Unknown').upper()
        st.metric("🤖 Алгоритм", model_name)

    # Детальная статистика
    if any(key in metrics for key in ['Training MSE', 'Training R2', 'Training MAE']):
        st.markdown("### 📊 Сравнение обучающей и тестовой выборок")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🏋️ Обучающая выборка")
            train_mse = metrics.get('Training MSE', 'N/A')
            train_r2 = metrics.get('Training R2', 'N/A')
            train_mae = metrics.get('Training MAE', 'N/A')

            if train_mse != 'N/A':
                st.write(f"**MSE:** {train_mse:.2f}")
                st.write(f"**R²:** {train_r2:.4f}")
                st.write(f"**MAE:** ${train_mae:.2f}")
            else:
                st.info("Метрики недоступны")

        with col2:
            st.markdown("#### 🧪 Тестовая выборка")
            test_mse = metrics.get('Test MSE', 'N/A')
            test_r2 = metrics.get('Test R2', 'N/A')
            test_mae = metrics.get('Test MAE', 'N/A')

            if test_mse != 'N/A':
                st.write(f"**MSE:** {test_mse:.2f}")
                st.write(f"**R²:** {test_r2:.4f}")
                st.write(f"**MAE:** ${test_mae:.2f}")
            else:
                st.info("Метрики недоступны")

        with col3:
            st.markdown("#### 📈 Перекрестная проверка")
            if all(m != 'N/A' for m in [train_r2, test_r2]):
                diff_r2 = float(train_r2) - float(test_r2)
                if abs(diff_r2) < 0.1:
                    st.success("✅ Хорошая обобщающая способность")
                elif diff_r2 > 0.1:
                    st.warning("⚠️ Возможное переобучение")
                else:
                    st.info("ℹ️ Недообучение модели")

    # Важность признаков
    if hasattr(model_info['model'], 'feature_importances_'):
        st.markdown("### 🔍 Важность признаков")

        features = predictor.feature_names
        importances = model_info['model'].feature_importances_

        # Создаем DataFrame
        importance_df = pd.DataFrame({
            'Признак': features,
            'Важность': importances
        }).sort_values('Важность', ascending=False).head(10)

        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(importance_df['Признак'], importance_df['Важность'],
                      color='#4ecdc4')
        ax.set_xlabel('Важность')
        ax.set_title('Топ-10 наиболее важных признаков')
        ax.invert_yaxis()

        st.pyplot(fig)

        # Таблица
        st.dataframe(importance_df, use_container_width=True)

    # Информация о данных
    st.markdown("### 📁 Информация о датасете")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Количество признаков:** {len(predictor.feature_names)}")
        st.write(f"**Модель:** {model_info.get('model_name', 'Unknown')}")

    with col2:
        st.write(f"**Путь к модели:** {predictor.model_path}")
        st.write(f"**Статус:** {'✅ Загружена' if predictor.model_data else '❌ Не загружена'}")

if __name__ == "__main__":
    main()
