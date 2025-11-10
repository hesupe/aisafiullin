# Конфигурационные параметры проекта City Transport Analytics

# Пути к данным
DATA_PATH = "transport_data.csv"
MODEL_PATH = "algorithms/transport_model.joblib"

# Параметры модели
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Параметры Random Forest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

# Параметры Gradient Boosting
GB_PARAMS = {
    'n_estimators': 150,
    'max_depth': 10,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

# Целевая переменная
TARGET_COLUMN = 'Booking Value'

# Колонки для исключения из признаков
EXCLUDE_COLUMNS = [
    'Booking ID',
    'Customer ID',
    'Date',
    'Time',
    'Booking Value',  # целевая переменная
    'Reason for cancelling by Customer',
    'Driver Cancellation Reason',
    'Incomplete Rides Reason'
]

# Категориальные признаки для кодирования
CATEGORICAL_FEATURES = [
    'Booking Status',
    'Vehicle Type',
    'Pickup Location',
    'Drop Location',
    'Payment Method'
]

# Параметры визуализации
PLOT_STYLE = "seaborn-v0_8"
FIGURE_SIZE = (12, 6)
