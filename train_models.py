# Імпортуємо необхідні бібліотеки для обробки аргументів, роботи з JSON, часом та числовими операціями
import argparse
import json
import time
import numpy as np

# Імпортуємо алгоритми машинного навчання зі scikit-learn
from sklearn.decomposition import PCA  # Метод зменшення розмірності
from sklearn.linear_model import LogisticRegression  # Логістична регресія для класифікації
from sklearn.metrics import accuracy_score, classification_report  # Метрики оцінки моделей
from sklearn.neighbors import KNeighborsClassifier  # Алгоритм K найближчих сусідів
from sklearn.pipeline import make_pipeline  # Для створення конвеєра обробки даних
from sklearn.preprocessing import StandardScaler  # Нормалізація ознак

# Імпортуємо наші модулі системи розпізнавання знаків
from traffic_sign_system.config import DEFAULT_IMAGE_SIZE, FAST_IMAGE_SIZE, MODEL_PATH
from traffic_sign_system.data_utils import load_dataset
from traffic_sign_system.features import build_feature_sets
from traffic_sign_system.predictor import TrafficSignPredictor


def limit_per_class(images, labels, label_names, max_per_class):
    """Обмежує кількість зображень на клас для швидкого режиму навчання."""
    counter = {}  # Лічильник для отслідження кількості зображень кожного класу
    sel_images, sel_labels, sel_names = [], [], []  # Списки для вибраних зображень
    
    # Проходимо по всім зображенням та їхніх позначками
    for img, lbl, name in zip(images, labels, label_names):
        # Якщо вже набрали потрібну кількість зображень для цього класу, пропускаємо
        if counter.get(lbl, 0) >= max_per_class:
            continue
        
        # Збільшуємо лічильник для цього класу
        counter[lbl] = counter.get(lbl, 0) + 1
        # Додаємо зображення до вибірки
        sel_images.append(img)
        sel_labels.append(lbl)
        sel_names.append(name)
    
    return sel_images, sel_labels, sel_names


def train(dataset_dir: str, fast_mode: bool = False) -> None:
    """Основна функція для навчання двох моделей розпізнавання дорожних знаків."""
    
    # Вибираємо розмір зображень в залежності від режиму (швидкий або стандартний)
    image_size = FAST_IMAGE_SIZE if fast_mode else DEFAULT_IMAGE_SIZE
    mode = "fast" if fast_mode else "standard"

    print(f"Режим навчання: {mode}, розмір зображень: {image_size}")
    # Починаємо відлік часу виконання
    t_start = time.perf_counter()

    # Завантажуємо набір даних з розділенням на тренувальну та тестову частини
    print("Завантажуємо датасет...")
    train_data, test_data, class_names = load_dataset(dataset_dir, image_size=image_size)

    # У швидкому режимі зменшуємо обсяг даних для більш швидкого навчання (для демонстрації)
    if fast_mode:
        print("Обмежуємо вибірку для швидкого режиму...")
        train_data.images, train_data.labels, train_data.label_names = limit_per_class(
            train_data.images, train_data.labels, train_data.label_names, max_per_class=60
        )
        test_data.images, test_data.labels, test_data.label_names = limit_per_class(
            test_data.images, test_data.labels, test_data.label_names, max_per_class=20
        )

    # Виводимо статистику завантажених даних
    print(f"Тренувальних зображень: {len(train_data.images)}")
    print(f"Тестових зображень: {len(test_data.images)}")
    print(f"Кількість класів: {len(class_names)}")

    # Будуємо ознаки (HOG та пікселі) з зображень для машинного навчання
    print("Будуємо ознаки...")
    t = time.perf_counter()
    x_train_hog, x_train_pix = build_feature_sets(train_data.images)  # HOG та пікселі для тренування
    x_test_hog, x_test_pix = build_feature_sets(test_data.images)      # HOG та пікселі для тестування
    print(f"Ознаки побудовано за {time.perf_counter() - t:.1f}с")

    # Конвертуємо позначки класів у numpy масиви для зручності роботи з моделями
    y_train = np.array(train_data.labels)
    y_test = np.array(test_data.labels)

    # МОДЕЛЬ 1: HOG + Логістична регресія 
    # Це перша модель - використовує ознаки HOG (Histogram of Oriented Gradients)
    # та логістичну регресію для класифікації. Параметри обрані для кращої точності
    print("\nНаш перший варіант моделі - HOG + LogisticRegression:")
    hog_model = make_pipeline(
        StandardScaler(),  # Нормалізуємо ознаки (приводимо до однакового масштабу)
        LogisticRegression(
            max_iter=2000,           # Максимальна кількість ітерацій для збіжності
            C=3.0,                   # Параметр регуляризації (менше = більша регуляризація)
            solver="lbfgs",          # Метод оптимізації
            class_weight="balanced", # Враховуємо дисбаланс класів
        ),
    )

    # МОДЕЛЬ 2: Пікселі + PCA + KNN 
    # Друга модель - використовує пікселі зображень, зменшує розмірність через PCA
    # та класифікує за допомогою K найближчих сусідів
    print("Наш другий варіант моделі - Pixels + PCA + KNN:")
    pixel_model = make_pipeline(
        StandardScaler(),  # Нормалізуємо пікселі
        # PCA - зменшуємо кількість ознак з тисяч до 100-150 для швидкості та уникнення переобучення
        PCA(n_components=100 if fast_mode else 150, random_state=42),
        # KNN - класифікуємо, обираючи найчастіший клас з K найближчих сусідів
        KNeighborsClassifier(
            n_neighbors=5,      # Дивимось на 5 найближчих сусідів
            weights="distance", # Більш близькі сусіди мають більшу вагу
            metric="euclidean", # Використовуємо евклідову відстань
        ),
    )

    # Навчаємо першу модель на тренувальних ознаках HOG
    print("\nНавчаємо HOG + LogisticRegression...")
    t = time.perf_counter()
    hog_model.fit(x_train_hog, y_train)
    print(f"Навчено за {time.perf_counter() - t:.1f}с")

    # Навчаємо другу модель на тренувальних пікселях
    print("Навчаємо Pixels + PCA + KNN...")
    t = time.perf_counter()
    pixel_model.fit(x_train_pix, y_train)
    print(f"Навчено за {time.perf_counter() - t:.1f}с")

    # Робимо передбачення на тестових даних обома моделями
    print("\nПередбачаємо результати на тестовому наборі...")
    hog_pred = hog_model.predict(x_test_hog)      # Передбачення першої моделі
    pixel_pred = pixel_model.predict(x_test_pix)  # Передбачення другої моделі

    # Обчислюємо та виводимо точність кожної моделі окремо
    hog_acc = accuracy_score(y_test, hog_pred)
    pixel_acc = accuracy_score(y_test, pixel_pred)
    print(f"\nТочність HOG моделі: {hog_acc:.4f} ({hog_acc*100:.1f}%)")
    print(f"Точність Pixel моделі: {pixel_acc:.4f} ({pixel_acc*100:.1f}%)")

    # Використовуємо обидві моделі разом (ансамбль) для кращого результату
    # Ансамбль комбінує передбачення обох моделей для більш надійного розпізнавання
    print("\nВикористовуємо ансамбль обох моделей для кращого результату...")
    predictor = TrafficSignPredictor(
        hog_model=hog_model,
        pixel_model=pixel_model,
        class_names=class_names,
        image_size=image_size,
    )
    # Отримуємо передбачення ансамблю для кожного тестового зображення
    ensemble_pred = [predictor.predict_array(img)["predicted_class"] for img in test_data.images]
    ensemble_acc = accuracy_score(test_data.label_names, ensemble_pred)
    print(f"Точність ансамблю: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")

    # Виводимо детальний звіт про точність для кожного класу
    print("\nДетальний звіт точності для кожного дорожного знака:")
    print(classification_report(test_data.label_names, ensemble_pred, zero_division=0, digits=3))

    # Збережуємо натреновану модель на диск для подальшого використання
    print("\nЗбережуємо модель...")
    predictor.save(MODEL_PATH)
    print(f"Модель успішно збережена за адресою: {MODEL_PATH}")

    # Записуємо метрики точності у JSON файл для документування результатів
    metrics = {
        "hog_accuracy": round(hog_acc, 4),           # Точність HOG моделі
        "pixel_accuracy": round(pixel_acc, 4),       # Точність Pixel моделі
        "ensemble_accuracy": round(ensemble_acc, 4), # Точність ансамблю
        "classes": class_names,                      # Список усіх дорожних знаків
        "image_size": list(image_size),              # Розмір зображень, на яких навчали
        "mode": mode,                                # Режим навчання (fast чи standard)
        "models": ["HOG + LogisticRegression", "Pixels + PCA + KNN"],  # Назви моделей
    }
    metrics_path = MODEL_PATH.parent / "traffic_sign_models_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Метрики результатів збережено: {metrics_path}")
    
    # Виводимо загальний час виконання всього процесу навчання
    print(f"Загальний час навчання: {time.perf_counter() - t_start:.1f}с")


if __name__ == "__main__":
    # Цей блок виконується лише коли скрипт запущено безпосередньо (не як імпорт)
    
    # Створюємо парсер для обробки аргументів командного рядка
    parser = argparse.ArgumentParser(description="Навчання моделей розпізнавання дорожних знаків")
    
    # Додаємо аргумент для шляху до датасету (за замовчуванням "dataset")
    parser.add_argument("--dataset", default="dataset", help="Шлях до датасету з навчальними зображеннями")
    
    # Додаємо прапорець для швидкого режиму 
    parser.add_argument("--fast", action="store_true", help="Швидкий режим для швидкого тестування (скорочена вибірка)")
    
    # Парсимо передані аргументи
    args = parser.parse_args()
    
    # Запускаємо навчання з переданими параметрами
    train(args.dataset, fast_mode=args.fast)