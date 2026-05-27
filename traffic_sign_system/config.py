from pathlib import Path  # Робота з шляхами

ROOT_DIR = Path(__file__).resolve().parent.parent  # Кореневий каталог проекту
MODEL_DIR = ROOT_DIR / "artifacts"  # Папка для збереження моделей
MODEL_PATH = MODEL_DIR / "traffic_sign_models.joblib"  # Шлях до файлу моделей

# Розміри зображень
DEFAULT_IMAGE_SIZE = (64, 64)  # Стандартний розмір
FAST_IMAGE_SIZE = (48, 48)  # Швидкий режим (менше)

# Ваги ансамблю (HOG краще, тому більша вага)
HOG_WEIGHT = 0.62  # 62% вага для HOG (точніше)
PIXEL_WEIGHT = 0.38  # 38% вага для пікселів

# Автоматично створюємо папку для моделей
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Якщо немає - створюємо
