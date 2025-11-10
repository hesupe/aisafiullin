import pandas as pd
import numpy as np
import os

# Конфигурация системы
DATA_PATH = "transport_data.csv"
TARGET_COLUMN = 'Booking Value'
KEY_FEATURES = ['Ride Distance', 'Driver Ratings', 'Customer Rating', 'Avg VTAT', 'Avg CTAT']
USEFUL_FEATURES = KEY_FEATURES

def load_data():
    """Загрузка и валидация исходных данных"""
    print("📁 Загрузка данных о поездках...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"🚨 Файл данных не обнаружен: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Данные успешно загружены: {len(df)} записей")
    return df

def preprocess_data(df):
    """Интеллектуальная предобработка данных для ML"""
    print("\n🔧 Запуск процесса предобработки...")

    # Создаем рабочую копию данных
    df = df.copy()

    # Очистка целевой переменной
    initial_count = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    cleaned_count = len(df)
    
    if initial_count > cleaned_count:
        print(f"🧹 Удалено некорректных записей: {initial_count - cleaned_count}")

    print(f"📊 После очистки: {cleaned_count} валидных записей")

    # Разделение на признаки и целевую переменную
    y = df[TARGET_COLUMN]
    X = df[USEFUL_FEATURES].copy()

    # Умное заполнение пропущенных значений
    print("🎯 Заполнение пропущенных данных...")
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        missing_count = X[col].isnull().sum()
        if missing_count > 0:
            X[col] = X[col].fillna(X[col].median())
            print(f"   📈 {col}: заполнено {missing_count} пропусков (медиана)")

    # Обработка целевой переменной
    y_missing = y.isnull().sum()
    if y_missing > 0:
        y = y.fillna(y.median())
        print(f"   🎯 Целевая переменная: заполнено {y_missing} пропусков")

    # Статистика признаков
    print(f"\n📋 Используемые признаки ({len(X.columns)}):")
    for feature in X.columns:
        print(f"   • {feature}")

    print(f"💰 Диапазон стоимости поездок: ${y.min():.0f} - ${y.max():.0f}")
    print(f"📊 Средняя стоимость: ${y.mean():.2f}")

    return X, y

def create_features(X):
    """Создание расширенных признаков для улучшения прогнозирования"""
    print("🎨 Генерация дополнительных признаков...")
    X = X.copy()
    
    features_created = 0
    
    if 'Ride Distance' in X.columns:
        # Категоризация расстояния
        X['distance_category'] = pd.cut(X['Ride Distance'], 
                                       bins=[0, 10, 25, 50, float('inf')], 
                                       labels=['short', 'medium', 'long', 'very_long'])
        X['distance_category'] = X['distance_category'].astype(str)
        features_created += 1
        print("   📏 Добавлена категоризация расстояния")
    
    if 'Driver Ratings' in X.columns and 'Customer Rating' in X.columns:
        # Анализ рейтингов
        X['rating_diff'] = X['Driver Ratings'] - X['Customer Rating']
        X['avg_rating'] = (X['Driver Ratings'] + X['Customer Rating']) / 2
        features_created += 2
        print("   ⭐ Добавлены метрики рейтингов")
    
    if 'Avg VTAT' in X.columns and 'Avg CTAT' in X.columns:
        # Временные метрики
        X['total_time'] = X['Avg VTAT'] + X['Avg CTAT']
        if 'Ride Distance' in X.columns:
            X['time_per_distance'] = X['total_time'] / (X['Ride Distance'] + 1e-8)
            features_created += 1
            print("   ⏱️  Добавлены временные характеристики")
    
    if 'Driver Ratings' in X.columns:
        # Категоризация рейтинга водителя
        X['driver_rating_category'] = pd.cut(X['Driver Ratings'], 
                                            bins=[0, 3.0, 4.0, 4.5, 5.0], 
                                            labels=['low', 'medium', 'high', 'excellent'])
        X['driver_rating_category'] = X['driver_rating_category'].astype(str)
        features_created += 1
        print("   🚗 Добавлена категоризация водителей")
    
    if 'Customer Rating' in X.columns:
        # Категоризация рейтинга клиента
        X['customer_rating_category'] = pd.cut(X['Customer Rating'], 
                                              bins=[0, 3.0, 4.0, 4.5, 5.0], 
                                              labels=['low', 'medium', 'high', 'excellent'])
        X['customer_rating_category'] = X['customer_rating_category'].astype(str)
        features_created += 1
        print("   👑 Добавлена категоризация клиентов")
    
    print(f"🎯 Всего создано дополнительных признаков: {features_created}")
    print(f"📊 Общее количество признаков: {len(X.columns)}")
    
    return X

def get_feature_info():
    """Получение детальной информации о признаках для анализа"""
    print("\n🔍 Сбор информации о признаках...")
    
    df = load_data()
    X, y = preprocess_data(df)
    
    feature_info = {
        'feature_names': list(X.columns),
        'n_features': X.shape[1],
        'target_name': TARGET_COLUMN,
        'target_range': (y.min(), y.max()),
        'target_mean': y.mean(),
        'data_shape': X.shape,
        'feature_types': X.dtypes.to_dict()
    }
    
    print("✅ Информация о признаках собрана:")
    print(f"   📊 Признаков: {feature_info['n_features']}")
    print(f"   🎯 Целевая переменная: {feature_info['target_name']}")
    print(f"   💰 Диапазон значений: ${feature_info['target_range'][0]:.0f} - ${feature_info['target_range'][1]:.0f}")
    
    return feature_info

if __name__ == "__main__":
    print("\n" + "🚀" + "="*60 + "🚀")
    print("           ТЕСТИРОВАНИЕ МОДУЛЯ ДАННЫХ")
    print("🚀" + "="*60 + "🚀")
    
    try:
        df = load_data()
        X, y = preprocess_data(df)
        
        print("\n" + "📋" + "="*60 + "📋")
        print("           ПРЕВЬЮ ОБРАБОТАННЫХ ДАННЫХ")
        print("📋" + "="*60 + "📋")
        
        print("\n🎯 ПРИЗНАКИ (первые 5 записей):")
        print(X.head())
        
        print(f"\n💰 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (первые 10 значений):")
        print(y.head(10).to_string())
        
        print(f"\n📊 СТАТИСТИКА ДАННЫХ:")
        print(f"   Размерность признаков: {X.shape}")
        print(f"   Размерность целевой: {y.shape}")
        print(f"   Типы данных: {X.dtypes.unique()}")
        
        # Тестирование создания признаков
        print("\n" + "🎨" + "="*60 + "🎨")
        print("           ТЕСТ СОЗДАНИЯ ПРИЗНАКОВ")
        print("🎨" + "="*60 + "🎨")
        
        X_extended = create_features(X)
        print(f"\n✅ Расширенные признаки созданы:")
        print(f"   Исходные признаки: {len(X.columns)}")
        print(f"   После расширения: {len(X_extended.columns)}")
        print(f"   Новые признаки: {list(set(X_extended.columns) - set(X.columns))}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}")
    
    print("\n" + "✅" + "="*60 + "✅")
    print("           ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("✅" + "="*60 + "✅")
