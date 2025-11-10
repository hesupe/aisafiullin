import sys
import subprocess
import os

def check_requirements():
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn')
    ]
    missing_packages = []

    for name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(name)

    if missing_packages:
        print("🚨 Обнаружены отсутствующие зависимости:")
        for pkg in missing_packages:
            print(f"   📦 {pkg}")
        print("\n💡 Для установки выполните:")
        print("   pip install -r requirements.txt")
        return False

    print("✅ Все зависимости успешно загружены")
    return True

def check_model():
    model_path = os.path.join(os.path.dirname(__file__), 'algorithms', 'transport_model.joblib')
    if not os.path.exists(model_path):
        print("⚠️  Модель машинного обучения не обнаружена!")
        print(f"   📁 Ожидаемый путь: {model_path}")
        print("\n💡 Для создания модели выполните:")
        print("   python main.py train")
        print("   🔮 (можно продолжить без модели, но прогнозирование будет недоступно)")
        input("\n↵ Нажмите Enter для продолжения...")
    else:
        print("✅ Модель ИИ готова к работе")

def system_diagnostics():
    print("\n🔍 Диагностика системы:")
    print(f"   📂 Рабочая директория: {os.getcwd()}")
    print(f"   🐍 Версия Python: {sys.version.split()[0]}")
    print(f"   💻 Платформа: {sys.platform}")
    
    # Проверка доступности основных модулей
    try:
        import streamlit
        print(f"   🌐 Streamlit: {streamlit.__version__}")
    except:
        print("   🌐 Streamlit: ❌ Не доступен")
    
    try:
        import pandas
        print(f"   📊 Pandas: {pandas.__version__}")
    except:
        print("   📊 Pandas: ❌ Не доступен")

def main():
    print("\n" + "✨" + "="*58 + "✨")
    print("           🚀 CITY TRANSPORT ANALYTICS SYSTEM 🚀")
    print("✨" + "="*58 + "✨")

    print("\n🔄 Запуск системы диагностики...")
    
    if not check_requirements():
        sys.exit(1)

    check_model()
    system_diagnostics()

    try:
        web_app_path = os.path.join(os.path.dirname(__file__), 'web_app.py')
        if not os.path.exists(web_app_path):
            print(f"❌ Основной файл приложения не найден: {web_app_path}")
            sys.exit(1)

        print("\n🎯 Инициализация веб-приложения...")
        print("🌍 После запуска откройте браузер по адресу:")
        print("   🔗 http://localhost:8501")
        print("\n" + "🔄" + "="*56 + "🔄")
        print("   ⏹️  Для остановки: Ctrl+C")
        print("   📊 Интерфейс: Streamlit Dashboard")
        print("🔄" + "="*56 + "🔄" + "\n")

        # Запуск с дополнительными параметрами для улучшения производительности
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            web_app_path,
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)

    except subprocess.CalledProcessError as e:
        print(f"🚨 Ошибка запуска веб-сервера: {e}")
        print("\n💡 Альтернативные варианты запуска:")
        print("   1. streamlit run web_app.py")
        print("   2. python -m streamlit run web_app.py")
        print("   3. Убедитесь что порт 8501 свободен")
    except KeyboardInterrupt:
        print("\n\n🛑 Работа приложения завершена по запросу пользователя")
        print("👋 Благодарим за использование City Transport Analytics!")
        print("🎯 До новых встреч!")
    except Exception as e:
        print(f"💥 Критическая ошибка системы: {e}")
        print("\n🔧 Рекомендуемые действия:")
        print("   1. Проверьте установку Python и пакетов")
        print("   2. Убедитесь в наличии файла web_app.py")
        print("   3. Проверьте права доступа к файлам")
        sys.exit(1)

if __name__ == "__main__":
    main()
