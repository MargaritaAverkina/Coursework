from dataclasses import dataclass, field  # Автоматичні класи
from pathlib import Path  # Робота з шляхами

import cv2  # Обробка зображень
import numpy as np  # Числові масиви
import pandas as pd  # Робота з CSV
from sklearn.model_selection import train_test_split  # Розділ на тренування/тест

from traffic_sign_system.features import preprocess_image  # Нормалізація зображень

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}  # Підтримувані формати


@dataclass  # Автоматичний __init__
class DatasetSplit:  # Розділ датасету (тренування/тест)
    images: list = field(default_factory=list)  # Зображення
    labels: list = field(default_factory=list)  # ID класів
    label_names: list = field(default_factory=list)  # Назви класів


def load_label_map(dataset_dir: Path) -> dict:  # Завантажуємо словник класів
    """Завантажує словник {id: назва} з labels.csv."""
    candidates = [  # Можливі шляхи до CSV
        dataset_dir / "labels.csv",
        dataset_dir / "signnames.csv",
        dataset_dir.parent / "labels.csv",
        dataset_dir.parent / "signnames.csv",
    ]
    for csv_path in candidates:  # Перевіряємо кожен шлях
        if not csv_path.exists():  # Якщо не існує
            continue
        df = pd.read_csv(csv_path)  # Читаємо CSV
        cols = {c.lower(): c for c in df.columns}  # Чувствительность до регістру

        id_col = next((cols[k] for k in ["classid", "class id", "class", "id"] if k in cols), None)  # Шукаємо ID колонку
        name_col = next((cols[k] for k in ["name", "label", "signname", "description"] if k in cols), None)  # Шукаємо назву

        if id_col and name_col:  # Якщо знайшли обидві
            return {str(row[id_col]): str(row[name_col]) for _, row in df[[id_col, name_col]].dropna().iterrows()}  # Повертаємо словник
    return {}  # Порожній словник якщо не знайдено


def read_images_from_dir(class_dir: Path, image_size: tuple) -> list:  # Читаємо зображення з папки
    """Читає всі зображення з папки класу."""
    images = []  # Список для зображень
    for path in sorted(class_dir.rglob("*")):  # Шукаємо всі файли рекурсивно
        if path.suffix.lower() not in IMAGE_EXTENSIONS:  # Якщо не зображення - пропускаємо
            continue
        img = cv2.imread(str(path))  # Читаємо зображення
        if img is not None:  # Якщо успішно прочитали
            images.append(preprocess_image(img, image_size))  # Нормалізуємо та добавляємо
    return images  # Повертаємо список


def load_split_from_folders(split_dir: Path, class_names: list, image_size: tuple) -> DatasetSplit:  # Завантажуємо дані з папок
    """Завантажує дані де кожен клас — окрема папка."""
    split = DatasetSplit()  # Створюємо розбиття
    for idx, class_name in enumerate(class_names):  # Для кожного класу
        class_dir = split_dir / class_name  # Папка класу
        if not class_dir.exists():  # Якщо немає папки
            continue
        imgs = read_images_from_dir(class_dir, image_size)  # Читаємо зображення
        split.images.extend(imgs)  # Додаємо до списку
        split.labels.extend([idx] * len(imgs))  # Додаємо ID класу
        split.label_names.extend([class_name] * len(imgs))  # Додаємо назву класу
    return split  # Повертаємо розбиття


def load_flat_test(test_dir: Path, class_names: list, image_size: tuple) -> DatasetSplit:  # Завантажуємо плоскі тестові дані
    """Завантажує тестові дані де файли назвами 000_001.png (префікс = клас)."""
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}  # Словник назва → індекс
    split = DatasetSplit()  # Створюємо розбиття

    for path in sorted(test_dir.iterdir()):  # Перебираємо файли
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:  # Якщо не файл зображення
            continue
        prefix = path.stem.split("_")[0]  # Отримуємо префікс (ID класу)
        if not prefix.isdigit():  # Якщо не цифра
            continue
        class_name = str(int(prefix))  # Конвертуємо в ID
        if class_name not in name_to_idx:  # Якщо класу немає
            continue
        img = cv2.imread(str(path))  # Читаємо зображення
        if img is None:  # Якщо помилка
            continue
        split.images.append(preprocess_image(img, image_size))  # Додаємо нормалізоване
        split.labels.append(name_to_idx[class_name])  # Додаємо ID класу
        split.label_names.append(class_name)  # Додаємо назву класу
    return split  # Повертаємо розбиття


def load_dataset(dataset_dir: str, image_size: tuple) -> tuple:  # Головна функція завантаження
    """
    Основна функція завантаження датасету.
    Повертає (train_data, test_data, display_class_names).
    """
    root = Path(dataset_dir)  # Конвертуємо в Path
    if not root.exists():  # Перевіряємо існування
        raise FileNotFoundError(f"Датасет не знайдено: {root}")

    # Знаходимо папку з тренувальними даними
    train_dir = None  # Поки немає
    for name in ["Train", "train", "DATA", "data"]:  # Варіанти назв
        candidate = root / name  # Перевіряємо
        if candidate.exists():  # Якщо знайшли
            train_dir = candidate
            break
    if train_dir is None:  # Якщо не знайшли
        train_dir = root  # Використовуємо кореневу папку

    # Знаходимо папку з тестовими даними
    test_dir = None  # Поки немає
    for name in ["Test", "test", "valid", "val"]:  # Варіанти назв
        candidate = root / name  # Перевіряємо
        if candidate.exists():  # Якщо знайшли
            test_dir = candidate
            break

    # Класи — це підпапки тренувальної вибірки
    class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])  # Отримуємо класи
    if not class_names:  # Якщо немає папок
        raise RuntimeError("Папки класів не знайдено.")

    label_map = load_label_map(root)  # Завантажуємо словник назв

    train_data = load_split_from_folders(train_dir, class_names, image_size)  # Завантажуємо трен. дані

    # Завантаження тестової вибірки
    if test_dir is not None:  # Якщо є папка тестів
        has_subfolders = any(p.is_dir() for p in test_dir.iterdir())  # Перевіряємо структуру
        if has_subfolders:  # Якщо з підпапками
            test_data = load_split_from_folders(test_dir, class_names, image_size)  # Завантажуємо з папок
        else:  # Якщо плоска структура
            test_data = load_flat_test(test_dir, class_names, image_size)  # Завантажуємо плоскі
    else:  # Якщо немає папки тестів
        test_data = None  # Буде розділено з трену

    # Якщо тестових даних немає — відокремлюємо 20% від тренувальних
    if test_data is None or not test_data.images:  # Якщо тестів немає
        train_idx, test_idx = train_test_split(  # Розділяємо індекси
            np.arange(len(train_data.images)),  # Всі індекси
            test_size=0.2,  # 20% на тести
            random_state=42,  # Для відтворюваності
            stratify=train_data.labels,  # Зберігаємо пропорції класів
        )
        test_data = DatasetSplit(  # Створюємо тестові дані
            images=[train_data.images[i] for i in test_idx],  # Тестові зображення
            labels=[train_data.labels[i] for i in test_idx],  # Тестові мітки
            label_names=[train_data.label_names[i] for i in test_idx],  # Тестові назви
        )
        train_data = DatasetSplit(  # Оновлюємо тренувальні
            images=[train_data.images[i] for i in train_idx],  # Тренувальні зображення
            labels=[train_data.labels[i] for i in train_idx],  # Тренувальні мітки
            label_names=[train_data.label_names[i] for i in train_idx],  # Тренувальні назви
        )

    # Переводимо числові назви класів у читабельні
    display_names = [label_map.get(name, name) for name in class_names]  # Красиві назви
    train_data.label_names = [label_map.get(n, n) for n in train_data.label_names]  # Оновлюємо трен
    test_data.label_names = [label_map.get(n, n) for n in test_data.label_names]  # Оновлюємо тести

    return train_data, test_data, display_names  # Повертаємо результат
