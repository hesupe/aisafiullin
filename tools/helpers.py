import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred, model_name=""):
    """Оценка производительности модели"""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Результаты:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print("-" * 40)
    
    return mse, r2

def plot_predictions(y_train_true, y_train_pred, y_test_true, y_test_pred, model_name):
    """Визуализация предсказаний модели"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Обучающая выборка
    ax1.scatter(y_train_true, y_train_pred, alpha=0.7, color='blue', label='Train')
    ax1.plot([y_train_true.min(), y_train_true.max()], 
             [y_train_true.min(), y_train_true.max()], 'k--', lw=2)
    ax1.set_xlabel('Истинные значения')
    ax1.set_ylabel('Предсказанные значения')
    ax1.set_title(f'{model_name} - Обучающая выборка')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Тестовая выборка
    ax2.scatter(y_test_true, y_test_pred, alpha=0.7, color='red', label='Test')
    ax2.plot([y_test_true.min(), y_test_true.max()], 
             [y_test_true.min(), y_test_true.max()], 'k--', lw=2)
    ax2.set_xlabel('Истинные значения')
    ax2.set_ylabel('Предсказанные значения')
    ax2.set_title(f'{model_name} - Тестовая выборка')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Визуализация важности признаков"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_imp['feature'], feature_imp['importance'])
        plt.xlabel('Важность признака')
        plt.title(f'Важность признаков - {model_name}')
        plt.tight_layout()
        plt.show()

def create_comparison_table(results_dict):
    """Создание таблицы сравнения моделей"""
    comparison_df = pd.DataFrame.from_dict(results_dict, orient='index',
                                         columns=['Training MSE', 'Training R2', 
                                                 'Test MSE', 'Test R2'])
    return comparison_df
