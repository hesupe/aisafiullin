import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from configuration.settings import TEST_SIZE, RANDOM_STATE, RF_PARAMS, GB_PARAMS, MODEL_PATH
from datasets.data_fetcher import load_data, preprocess_data
from tools.helpers import evaluate_model, plot_predictions, plot_feature_importance, create_comparison_table

class TransportModelTrainer:
    """Класс для обучения и управления моделями предсказания стоимости поездок"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_data(self):
        """Подготовка и разделение данных"""
        print("="*60)
        print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        print("="*60)
        
        # Загружаем и предобрабатываем данные
        df = load_data()
        X, y = preprocess_data(df)
        
        self.feature_names = X.columns.tolist()
        
        # Разделяем на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        print(f"\nОбучающая выборка: {self.X_train.shape}")
        print(f"Тестовая выборка: {self.X_test.shape}")
        print(f"Среднее значение Booking Value (train): {self.y_train.mean():.2f}")
        print(f"Среднее значение Booking Value (test): {self.y_test.mean():.2f}")
        
    def train_linear_regression(self):
        """Обучение линейной регрессии"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ LINEAR REGRESSION")
        print("="*60)

        # Финальная проверка данных перед обучением
        print("Проверка данных перед обучением...")
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_train содержит NaN: {self.X_train.isnull().any().any()}")
        print(f"y_train содержит NaN: {self.y_train.isnull().any()}")

        # Последняя проверка и очистка
        if self.X_train.isnull().any().any():
            print("Удаляю строки с NaN в X_train...")
            self.X_train = self.X_train.fillna(0)
        if self.y_train.isnull().any():
            print("Удаляю строки с NaN в y_train...")
            valid_indices = ~self.y_train.isnull()
            self.X_train = self.X_train[valid_indices]
            self.y_train = self.y_train[valid_indices]

        print(f"После очистки - X_train: {self.X_train.shape}, y_train: {self.y_train.shape}")

        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        
        # Предсказания
        y_train_pred = lr.predict(self.X_train)
        y_test_pred = lr.predict(self.X_test)
        
        # Оценка
        train_mse, train_r2 = evaluate_model(self.y_train, y_train_pred, "Linear Regression Train")
        test_mse, test_r2 = evaluate_model(self.y_test, y_test_pred, "Linear Regression Test")
        
        # Дополнительные метрики
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        
        # Сохраняем результаты
        self.models['linear_regression'] = lr
        self.results['linear_regression'] = {
            'model': lr,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'metrics': {
                'Training MSE': train_mse,
                'Training R2': train_r2,
                'Training MAE': train_mae,
                'Test MSE': test_mse,
                'Test R2': test_r2,
                'Test MAE': test_mae
            }
        }
        
        # Визуализация
        plot_predictions(self.y_train, y_train_pred, self.y_test, y_test_pred, 
                        "Linear Regression - Booking Value Prediction")
        
        return lr
    
    def train_random_forest(self):
        """Обучение случайного леса"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ RANDOM FOREST")
        print("="*60)
        
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(self.X_train, self.y_train)
        
        # Предсказания
        y_train_pred = rf.predict(self.X_train)
        y_test_pred = rf.predict(self.X_test)
        
        # Оценка
        train_mse, train_r2 = evaluate_model(self.y_train, y_train_pred, "Random Forest Train")
        test_mse, test_r2 = evaluate_model(self.y_test, y_test_pred, "Random Forest Test")
        
        # Дополнительные метрики
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        
        # Сохраняем результаты
        self.models['random_forest'] = rf
        self.results['random_forest'] = {
            'model': rf,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'metrics': {
                'Training MSE': train_mse,
                'Training R2': train_r2,
                'Training MAE': train_mae,
                'Test MSE': test_mse,
                'Test R2': test_r2,
                'Test MAE': test_mae
            }
        }
        
        # Визуализация
        plot_predictions(self.y_train, y_train_pred, self.y_test, y_test_pred, 
                        "Random Forest - Booking Value Prediction")
        
        # Важность признаков
        plot_feature_importance(rf, self.feature_names, "Random Forest")
        
        return rf
    
    def train_gradient_boosting(self):
        """Обучение градиентного бустинга"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ GRADIENT BOOSTING")
        print("="*60)
        
        gb = GradientBoostingRegressor(**GB_PARAMS)
        gb.fit(self.X_train, self.y_train)
        
        # Предсказания
        y_train_pred = gb.predict(self.X_train)
        y_test_pred = gb.predict(self.X_test)
        
        # Оценка
        train_mse, train_r2 = evaluate_model(self.y_train, y_train_pred, "Gradient Boosting Train")
        test_mse, test_r2 = evaluate_model(self.y_test, y_test_pred, "Gradient Boosting Test")
        
        # Дополнительные метрики
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print(f"Training MAE: {train_mae:.2f}")
        print(f"Test MAE: {test_mae:.2f}")
        
        # Сохраняем результаты
        self.models['gradient_boosting'] = gb
        self.results['gradient_boosting'] = {
            'model': gb,
            'train_pred': y_train_pred,
            'test_pred': y_test_pred,
            'metrics': {
                'Training MSE': train_mse,
                'Training R2': train_r2,
                'Training MAE': train_mae,
                'Test MSE': test_mse,
                'Test R2': test_r2,
                'Test MAE': test_mae
            }
        }
        
        # Визуализация
        plot_predictions(self.y_train, y_train_pred, self.y_test, y_test_pred, 
                        "Gradient Boosting - Booking Value Prediction")
        
        # Важность признаков
        plot_feature_importance(gb, self.feature_names, "Gradient Boosting")
        
        return gb
    
    def compare_models(self):
        """Сравнение всех обученных моделей"""
        if not self.results:
            print("Нет обученных моделей для сравнения")
            return
        
        print("\n" + "="*60)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        metrics_dict = {}
        for model_name, result in self.results.items():
            metrics_dict[model_name] = [
                result['metrics']['Training MSE'],
                result['metrics']['Training R2'],
                result['metrics']['Training MAE'],
                result['metrics']['Test MSE'],
                result['metrics']['Test R2'],
                result['metrics']['Test MAE']
            ]
        
        comparison_df = pd.DataFrame(
            metrics_dict,
            index=['Train MSE', 'Train R²', 'Train MAE', 'Test MSE', 'Test R²', 'Test MAE']
        ).T
        
        print("\n", comparison_df.to_string())
        
        # Определяем лучшую модель по Test R²
        best_model_name = comparison_df['Test R²'].idxmax()
        print(f"\n🏆 Лучшая модель: {best_model_name.upper()}")
        print(f"   Test R²: {comparison_df.loc[best_model_name, 'Test R²']:.4f}")
        print(f"   Test MAE: {comparison_df.loc[best_model_name, 'Test MAE']:.2f}")
        
        return comparison_df
    
    def save_best_model(self):
        """Сохранение лучшей модели"""
        if not self.results:
            print("Нет обученных моделей для сохранения")
            return
        
        # Находим модель с лучшим R² на тестовой выборке
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['metrics']['Test R2'])
        best_model = self.results[best_model_name]['model']
        
        # Создаем папку models, если её нет
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Сохраняем модель с метаданными
        model_data = {
            'model': best_model,
            'feature_names': self.feature_names,
            'model_name': best_model_name,
            'metrics': self.results[best_model_name]['metrics']
        }
        
        joblib.dump(model_data, MODEL_PATH)
        
        print("\n" + "="*60)
        print(f"✓ Лучшая модель ({best_model_name}) сохранена в: {MODEL_PATH}")
        print(f"  Метрики модели:")
        print(f"  - Test R²: {self.results[best_model_name]['metrics']['Test R2']:.4f}")
        print(f"  - Test MAE: {self.results[best_model_name]['metrics']['Test MAE']:.2f}")
        print(f"  - Test MSE: {self.results[best_model_name]['metrics']['Test MSE']:.2f}")
        print("="*60)
    
    def train_all_models(self):
        """Обучение всех моделей"""
        self.prepare_data()
        self.train_linear_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.compare_models()
        self.save_best_model()

def main():
    """Основная функция для обучения моделей"""
    print("\n" + "="*60)
    print("CITY TRANSPORT ANALYTICS - ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ СТОИМОСТИ")
    print("="*60 + "\n")
    
    trainer = TransportModelTrainer()
    trainer.train_all_models()
    
    print("\n✓ Обучение завершено успешно!")

if __name__ == "__main__":
    main()
