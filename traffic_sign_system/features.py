import cv2  # Обробка зображень
import numpy as np  # Нумеричні операції

# Розмір зображення для HOG
HOG_SIZE = (64, 64)  # Розмір для HOG

# Створення HOG-дескриптора
_HOG_DESCRIPTOR = cv2.HOGDescriptor(
    _winSize=HOG_SIZE,      # Розмір вікна
    _blockSize=(16, 16),    # Розмір блоку
    _blockStride=(8, 8),    # Крок зміщення блоку
    _cellSize=(8, 8),       # Розмір комірки
    _nbins=9,               # Кількість бінів гістограми
)

def preprocess_image(image: np.ndarray, image_size: tuple) -> np.ndarray:  # Предобробка
    """Знаходить область знака та масштабує до потрібного розміру."""
    region = _find_sign_region(image)  # Знаходимо область знака
    return cv2.resize(region, image_size)  # Масштабуємо до цільового розміру


def _find_sign_region(image: np.ndarray) -> np.ndarray:  # Шукаємо робочу область
    """Шукає область дорожнього знака за кольором (червоний, синій, білий, жовтий)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR → HSV

    # Маски кольорів
    red_low = cv2.inRange(hsv, (0, 70, 40), (15, 255, 255))  # червоний (низ)
    red_high = cv2.inRange(hsv, (160, 70, 40), (180, 255, 255))  # червоний (верх)
    blue = cv2.inRange(hsv, (85, 40, 40), (135, 255, 255))  # синій
    white = cv2.inRange(hsv, (0, 0, 130), (180, 70, 255))  # білий
    yellow = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))  # жовтий

    mask = red_low | red_high | blue | white | yellow  # комбінована маска

    # Морфологічне очищення
    kernel = np.ones((5, 5), dtype=np.uint8)  # ядро морфології
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # закриваємо дірки
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # прибираємо шум

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # шукаємо контури
    if not contours:
        return image  # якщо нічого не знайдено — повертаємо оригінал

    image_area = image.shape[0] * image.shape[1]  # площа зображення
    best_box = None
    best_area = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)  # обмежувальний прямокутник
        area = w * h  # площа контуру
        if area < image_area * 0.03:  # фільтр за площею
            continue
        aspect = w / max(h, 1)  # співвідношення сторін
        if aspect < 0.55 or aspect > 1.8:  # фільтр за формою
            continue
        if area > best_area:  # зберігаємо найбільший релевантний бокс
            best_area = area
            best_box = (x, y, w, h)

    if best_box is None:
        return image  # якщо нічого підходящого — повертаємо оригінал

    x, y, w, h = best_box  # координати кращого боксу
    pad_x = max(int(w * 0.10), 4)  # паддінг по X (~10%)
    pad_y = max(int(h * 0.10), 4)  # паддінг по Y (~10%)
    return image[
        max(y - pad_y, 0): min(y + h + pad_y, image.shape[0]),
        max(x - pad_x, 0): min(x + w + pad_x, image.shape[1]),
    ]  # ріже і повертає регіон


def _to_gray_equalized(image: np.ndarray) -> np.ndarray:  # Сіре + вирівнювання
    """Переводить у сірий і вирівнює гістограму для кращого контрасту."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # в сірих відтінках
    return cv2.equalizeHist(gray)  # вирівнювання гістограми


def extract_hog_features(image: np.ndarray) -> np.ndarray:  # HOG-ознаки
    """Отримання HOG-ознак із зображення 64x64."""
    resized = cv2.resize(image, HOG_SIZE)  # до 64x64
    gray = _to_gray_equalized(resized)  # сірий + вирівнювання
    return _HOG_DESCRIPTOR.compute(gray).flatten()  # вектор HOG


def extract_pixel_features(image: np.ndarray) -> np.ndarray:  # Пікселі-ознаки
    """Піксельні ознаки — стислий сірий варіант 24x24."""
    gray = _to_gray_equalized(image)  # сірий + вирівнювання
    small = cv2.resize(gray, (24, 24))  # 24x24 стиснення
    return (small.astype("float32") / 255.0).flatten()  # нормалізований вектор


def get_prediction_views(image: np.ndarray, image_size: tuple) -> list:  # Варіанти перегляду
    """
    Генерує кілька варіантів перегляду одного зображення:
    - знайдена область знака
    - центральний кроп
    - оригінал масштабований
    Дублікати відфільтровуються.
    """
    h, w = image.shape[:2]  # розміри
    crop_h, crop_w = int(h * 0.82), int(w * 0.82)  # 82% центр кроп
    y0, x0 = (h - crop_h) // 2, (w - crop_w) // 2  # початкові координати
    center_crop = image[y0:y0 + crop_h, x0:x0 + crop_w]  # центральний кроп

    candidates = [
        preprocess_image(image, image_size),  # знайдена область
        cv2.resize(center_crop, image_size),  # центр кроп
        cv2.resize(image, image_size),  # весь кадр масштабуєм
    ]

    # Фільтруємо однакові зображення за сигнатурою
    views = []
    seen = set()
    for view in candidates:
        sig = view[:6, :6].tobytes()  # невеликий відбиток
        if sig not in seen:
            seen.add(sig)
            views.append(view)  # додаємо унікальний варіант
    return views


def build_feature_sets(images: list) -> tuple:  # Будує ознаки для батчу
    """Будує HOG та піксельні ознаки для списку зображень."""
    hog = [extract_hog_features(img) for img in images]  # HOG-матриця
    pixels = [extract_pixel_features(img) for img in images]  # Піксельна матриця
    return np.array(hog), np.array(pixels)  # повертаємо numpy-маси