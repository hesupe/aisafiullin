import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.train_model import main as train_main
from algorithms.transport_predictor import TransportCostPredictor

def launch_web_app():
    """Запуск интерактивного веб-приложения"""
    print("\n" + "✨" + "="*68 + "✨")
    print("           🚀 ЗАПУСК CITY TRANSPORT ANALYTICS SYSTEM")
    print("✨" + "="*68 + "✨")

    try:
        import streamlit
        import subprocess

        web_app_path = os.path.join(os.path.dirname(__file__), 'web_app.py')
        if not os.path.exists(web_app_path):
            print("❌ Основной файл приложения не обнаружен")
            print(f"   📁 Ожидаемый путь: {web_app_path}")
            return

        print("🎯 Инициализация веб-интерфейса...")
        print("🌍 После запуска откройте браузер по адресу:")
        print("   🔗 http://localhost:8501")
        print("\n" + "🔄" + "="*66 + "🔄")
        print("   ⏹️  Для остановки сервера: Ctrl+C")
        print("   📊 Доступные режимы: Прогноз, Анализ, Визуализация")
        print("🔄" + "="*66 + "🔄" + "\n")

        # Запуск с оптимизированными параметрами
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            web_app_path,
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.primaryColor=#667eea"
        ], check=True)

    except ImportError:
        print("🚨 Streamlit не установлен в системе")
        print("\n💡 Для установки выполните:")
        print("   pip install streamlit")
        print("   или")
        print("   conda install -c conda-forge streamlit")
    except KeyboardInterrupt:
        print("\n\n🛑 Работа веб-сервера завершена")
        print("👋 Возвращайтесь для новых анализов!")
    except Exception as e:
        print(f"💥 Критическая ошибка при запуске: {str(e)}")
        print("\n🔧 Рекомендуемые действия:")
        print("   1. Проверьте установку Streamlit")
        print("   2. Убедитесь что порт 8501 свободен")
        print("   3. Проверьте наличие файла web_app.py")

def main():
    """Главный контроллер City Transport Analytics System"""
    parser = argparse.ArgumentParser(
        description="🚀 City Transport Analytics System - Интеллектуальный анализ стоимости транспортных услуг",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📋 Примеры использования:
  python main.py train          🏋️  Обучение модели машинного обучения
  python main.py predict        🔮 Интерактивный режим прогнозирования  
  python main.py predict --batch data.csv  📊 Пакетная обработка файла
  python main.py web            🌐 Запуск веб-интерфейса

🎯 Возможности системы:
  • Мгновенные прогнозы стоимости транспортных услуг
  • Подробная аналитика и визуализация
  • Пакетная обработка данных
  • Современный веб-интерфейс
        """
    )
    
    parser.add_argument(
        'action', 
        choices=['train', 'predict', 'web'], 
        help='Режим работы системы'
    )
    parser.add_argument(
        '--batch', 
        help='Путь к CSV файлу для массового анализа'
    )

    args = parser.parse_args()

    print("\n" + "🌟" + "="*68 + "🌟")
    print("           🤖 CITY TRANSPORT ANALYTICS SYSTEM")
    print("🌟" + "="*68 + "🌟")

    if args.action == 'train':
        print("\n🏋️  АКТИВАЦИЯ РЕЖИМА ОБУЧЕНИЯ МОДЕЛИ")
        print("📊 Загрузка данных и подготовка функций...")
        print("⚙️  Оптимизация гиперпараметров...")
        train_main()

    elif args.action == 'predict':
        print("\n🔮 АКТИВАЦИЯ РЕЖИМА ПРОГНОЗИРОВАНИЯ")

        predictor = TransportCostPredictor()

        if predictor.model_data is None:
            print("❌ Модель искусственного интеллекта не обнаружена")
            print("\n💡 Для инициализации выполните:")
            print("   python main.py train")
            print("\n📚 Это создаст оптимизированную модель для точных прогнозов")
            return

        if args.batch:
            print(f"📁 Обработка файла: {args.batch}")
            print("📈 Массовый анализ данных...")
            predictor.predict_batch(args.batch)
        else:
            print("🎮 Запуск интерактивного режима")
            print("💬 Введите параметры поездки для мгновенного прогноза")
            predictor.predict_interactive()

    elif args.action == 'web':
        launch_web_app()

    print("\n" + "✅" + "="*68 + "✅")
    print("           🎉 ОПЕРАЦИЯ УСПЕШНО ВЫПОЛНЕНА!")
    print("✅" + "="*68 + "✅")
    print("\n🚀 Система готова к новым задачам!")
    print("💫 Для продолжения работы выберите следующий режим\n")

if __name__ == "__main__":
    main()
