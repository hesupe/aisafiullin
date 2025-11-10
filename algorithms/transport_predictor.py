import joblib
import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к datasets для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from datasets.data_fetcher import create_features, USEFUL_FEATURES
except ImportError:
    # Альтернативный импорт если структура папок отличается
    try:
        from download_data import create_features, USEFUL_FEATURES
    except ImportError as e:
        print(f"❌ Ошибка импорта модулей: {e}")
        # Создаем базовые константы если импорт не удался
        USEFUL_FEATURES = ['Ride Distance', 'Driver Ratings', 'Customer Rating', 'Avg VTAT', 'Avg CTAT']
        
        def create_features(X):
            """Базовая функция создания признаков если основной модуль недоступен"""
            X = X.copy()
            
            # Базовые преобразования
            if 'Ride Distance' in X.columns:
                X['distance_category'] = pd.cut(X['Ride Distance'], 
                                               bins=[0, 10, 25, 50, float('inf')], 
                                               labels=['short', 'medium', 'long', 'very_long'])
                X['distance_category'] = X['distance_category'].astype(str)
            
            if 'Driver Ratings' in X.columns and 'Customer Rating' in X.columns:
                X['rating_diff'] = X['Driver Ratings'] - X['Customer Rating']
                X['avg_rating'] = (X['Driver Ratings'] + X['Customer Rating']) / 2
            
            if 'Avg VTAT' in X.columns and 'Avg CTAT' in X.columns:
                X['total_time'] = X['Avg VTAT'] + X['Avg CTAT']
                if 'Ride Distance' in X.columns:
                    X['time_per_distance'] = X['total_time'] / (X['Ride Distance'] + 1e-8)
            
            return X

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'transport_model.joblib')

class TransportCostPredictor:
    """Класс для предсказания стоимости поездок с улучшенными признаками"""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model_data = None
        self.feature_names = None
        self.load_model()

    def load_model(self):
        """Загрузка модели и обновление списка признаков"""
        try:
            if not os.path.exists(self.model_path):
                print(f"🚨 Модель не найдена по пути: {self.model_path}")
                print("💡 Выполните обучение модели: python main.py train")
                return None
                
            self.model_data = joblib.load(self.model_path)
            self.feature_names = self.model_data.get('feature_names', USEFUL_FEATURES)
            print(f"✅ Модель успешно загружена: {self.model_data.get('model_name', 'Unknown')}")
            print(f"📊 Используется {len(self.feature_names)} признаков для прогнозирования")
            return self.model_data['model']
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return None

    def predict_booking_value(self, input_data):
        """Предсказание с применением feature engineering"""
        if self.model_data is None:
            print("⚠️ Модель не загружена. Предсказание невозможно.")
            return None

        try:
            # Создаем DataFrame из входных данных
            if isinstance(input_data, dict):
                df_input = pd.DataFrame([input_data])
            else:
                df_input = input_data

            # Проверяем наличие необходимых признаков
            missing_features = set(USEFUL_FEATURES) - set(df_input.columns)
            if missing_features:
                print(f"⚠️ Отсутствуют признаки: {missing_features}")
                # Добавляем недостающие признаки со значениями по умолчанию
                for feature in missing_features:
                    df_input[feature] = 0.0

            # Берем только основные признаки и применяем feature engineering
            X = df_input[USEFUL_FEATURES].copy()
            X = create_features(X)  # Применяем те же преобразования что и при обучении

            # Убеждаемся, что признаки совпадают с теми, на которых обучалась модель
            X = X.reindex(columns=self.feature_names, fill_value=0)

            # Предсказание
            prediction = self.model_data['model'].predict(X)
            return prediction

        except Exception as e:
            print(f"❌ Ошибка при предсказании: {str(e)}")
            return None
    
    def predict_interactive(self):
        """Интерактивный ввод данных для предсказания"""
        if self.model_data is None:
            print("❌ Модель не загружена. Запустите обучение модели.")
            return
        
        print("\n" + "🎯" + "="*60 + "🎯")
        print("           ИНТЕРАКТИВНЫЙ РЕЖИМ ПРОГНОЗИРОВАНИЯ")
        print("🎯" + "="*60 + "🎯")
        
        print("\n📋 Основные параметры для анализа:")
        for i, feature in enumerate(USEFUL_FEATURES, 1):
            print(f"   {i}. {feature}")
        
        print("\n✨ Дополнительно создаваемые признаки:")
        additional_features = [
            "distance_category", "rating_diff", "avg_rating", 
            "total_time", "time_per_distance", "driver_rating_category", 
            "customer_rating_category"
        ]
        for i, feature in enumerate(additional_features, 1):
            print(f"   • {feature}")
        
        # Пример входных данных
        example_data = {
            'Ride Distance': 20.0,
            'Driver Ratings': 4.5,
            'Customer Rating': 4.7,
            'Avg VTAT': 15.0,
            'Avg CTAT': 10.0
        }
        
        print("\n📝 Пример параметров поездки:")
        for key, value in example_data.items():
            print(f"   🚗 {key}: {value:.1f}")
        
        # Быстрое предсказание на примере
        print("\n" + "🔮" + "-"*58 + "🔮")
        prediction = self.predict_booking_value(example_data)
        if prediction is not None:
            print(f"💎 Тестовый прогноз: ${prediction[0]:.2f}")
        print("🔮" + "-"*58 + "🔮")
        
        # Интерактивный ввод
        print("\n💫 Введите параметры вашей поездки:")
        print("   (нажмите Enter для использования значений по умолчанию)")
        user_data = {}
        
        for feature in USEFUL_FEATURES:
            default_value = example_data.get(feature, 0.0)
            prompt = f"   📍 {feature} [по умолчанию: {default_value}]: "
            value_str = input(prompt)
            
            if value_str.strip() == "":
                user_data[feature] = default_value
            else:
                try:
                    user_data[feature] = float(value_str)
                except ValueError:
                    print(f"❌ Ошибка: '{value_str}' не является числом")
                    return
        
        # Предсказание
        print("\n🔄 Выполняем анализ...")
        prediction = self.predict_booking_value(user_data)
        
        if prediction is not None:
            print("\n" + "💰" + "="*60 + "💰")
            print(f"           ПРОГНОЗ СТОИМОСТИ: ${prediction[0]:.2f}")
            print("💰" + "="*60 + "💰")
            
            # Детальная интерпретация
            print("\n📈 Анализ результата:")
            cost = prediction[0]
            if cost > 200:
                print("   💎 Премиум-уровень: Дальняя поездка или транспорт высокого класса")
                print("   ✅ Отличное качество обслуживания")
            elif cost > 100:
                print("   💰 Стандартный уровень: Оптимальное соотношение цены и качества") 
                print("   ⭐ Комфортные условия поездки")
            elif cost > 50:
                print("   💵 Эконом-вариант: Короткая или стандартная поездка")
                print("   🎯 Бюджетное решение")
            else:
                print("   🎪 Базовый уровень: Минимальная стоимость")
                print("   📍 Короткое расстояние")
                
            print(f"\n🎯 Модель: {self.model_data.get('model_name', 'Unknown').upper()}")
            print(f"📊 Точность модели: R² = {self.model_data.get('metrics', {}).get('Test R2', 'N/A'):.3f}")
            
        else:
            print("❌ Не удалось выполнить прогноз. Проверьте входные данные.")

def main():
    """Основная функция для предсказаний"""
    print("\n" + "🚀" + "="*60 + "🚀")
    print("           AI RIDE PRICE PREDICTION SYSTEM")
    print("🚀" + "="*60 + "🚀")
    
    predictor = TransportCostPredictor()
    
    if predictor.model_data is None:
        print("\n💡 Рекомендации:")
        print("   1. Проверьте наличие файла модели")
        print("   2. Запустите обучение: python main.py train")
        print("   3. Убедитесь в правильности структуры проекта")
        return
    
    # Интерактивный режим
    predictor.predict_interactive()

if __name__ == "__main__":
    main()
