import random  # Для випадкового вибору
from contextlib import asynccontextmanager  # Для управління ресурсами при старті/зупинці
from pathlib import Path  # Робота з шляхами файлів

from fastapi import FastAPI, File, HTTPException, UploadFile  # FastAPI фреймворк
from fastapi.responses import FileResponse, JSONResponse  # Типи відповідей
from fastapi.staticfiles import StaticFiles  # Обслуговування статичних файлів

from traffic_sign_system.config import MODEL_PATH, ROOT_DIR  # Шляхи до моделі
from traffic_sign_system.predictor import TrafficSignPredictor  # Клас розпізнавання
from traffic_sign_system.recommendations import confidence_level, translate_label  # Допоміжні функції


# Глобальні змінні - будуть заповнені при старті сервера
predictor: TrafficSignPredictor | None = None  # Натренована модель для розпізнавання
startup_error: str | None = None  # Помилка при завантаженні моделі
test_samples: list[dict] = []  # Список тестових зображень для демонстрації


def load_test_samples() -> list[dict]:  # Завантажує тестові зображення
    """Завантажує тестові зображення з папки dataset/Test."""
    test_dir = ROOT_DIR / "dataset" / "Test"  # Папка з тестовими зображеннями
    labels_path = ROOT_DIR / "labels.csv"  # Файл з назвами класів

    # Читаємо CSV файл з назвами дорожних знаків
    label_map: dict[str, str] = {}  # Словник: ID класу -> назва знака
    if labels_path.exists():
        # Проходимо по кожному рядку CSV (пропускаємо перший - заголовок)
        for row in labels_path.read_text(encoding="utf-8").splitlines()[1:]:
            if "," not in row:  # Пропускаємо рядки без коми
                continue
            class_id, label = row.split(",", 1)  # Розділяємо ID та назву
            label_map[str(int(class_id))] = label.strip()  # Зберігаємо у словник

    if not test_dir.exists():  # Якщо папки немає, повертаємо порожній список
        return []

    samples = []  # Список для зібраних зображень
    # Проходимо по всім файлам у папці та підпапках
    for img_path in sorted(test_dir.rglob("*")):
        # Перевіряємо, що це зображення (jpg, png тощо)
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        prefix = img_path.stem.split("_")[0]  # Отримуємо ID класу з назви файлу
        if not prefix.isdigit():  # Пропускаємо, якщо перша частина не цифра
            continue
        class_id = str(int(prefix))  # Нормалізуємо ID (видаляємо нулі на початку)
        samples.append({  # Додаємо до списку
            "path": img_path,  # Шлях до файлу
            "true_label": label_map.get(class_id, class_id),  # Справжня назва знака
        })

    random.shuffle(samples)  # Перемішуємо для випадкового порядку
    return samples


def build_payload(result: dict) -> dict:  # Форматує результат для відповіді
    """Формує відповідь з українськими назвами."""
    return {
        **result,  # Копіюємо всі оригінальні результати
        "predicted_class_ua": translate_label(result["predicted_class"]),  # Переводимо класи на українську мову
        "true_label_ua": None,  # Буде заповнено пізніше для тестів
        "model_votes_ua": {  # Голоси обох моделей українською
            "hog": translate_label(result["model_votes"]["hog"]),  # Голос HOG моделі
            "pixel": translate_label(result["model_votes"]["pixel"]),  # Голос Pixel моделі
        },
        # Топ кандидати з перекладом
        "top_candidates_ua": [
            {
                "label": c["label"],  # ID класу
                "label_ua": translate_label(c["label"]),  # Назва українською
                "probability": c["probability"],  # Вірогідність
            }
            for c in result["top_candidates"]  # Для кожного кандидата
        ],
        "confidence_level": confidence_level(result["confidence"]),  # Рівень впевненості
    }


@asynccontextmanager  # Спеціальний декоратор для керування життєвим циклом
async def lifespan(app: FastAPI):  # Запускається при старті та зупинці сервера
    """Завантажує модель при старті сервера."""
    global predictor, startup_error, test_samples  # Маємо доступ до глобальних змінних
    try:
        # Завантажуємо натреновану модель
        predictor = TrafficSignPredictor.load(MODEL_PATH)
        print(f"Модель завантажено: {MODEL_PATH}")
    except Exception as e:  # Якщо сталася помилка
        startup_error = str(e)  # Зберігаємо текст помилки
        print(f"Помилка завантаження моделі: {e}")
    test_samples = load_test_samples()  # Завантажуємо тестові зображення
    print(f"Тестових зображень: {len(test_samples)}")
    yield  # Розпочинаємо роботу сервера


app = FastAPI(title="Розпізнавання дорожніх знаків", lifespan=lifespan)  # Створюємо FastAPI застосунок
app.mount("/static", StaticFiles(directory=str(ROOT_DIR / "static")), name="static")  # Шляху до CSS/JS


@app.get("/")  # Головна сторінка
def index():  # Повертає HTML сторінку
    return FileResponse(ROOT_DIR / "templates" / "index.html")  # Головна HTML сторінка


@app.get("/health")  # Перевірка стану сервера
def health():  # Повертає статус розпізнавача
    return JSONResponse({  # JSON відповідь
        "status": "ok" if predictor else "model_not_loaded",  # Статус моделі
        "model_path": str(MODEL_PATH),  # Шлях до моделі
        "error": startup_error,  # Помилка (якщо була)
    })


@app.post("/api/predict")  # Точка для передачі зображення
async def predict(file: UploadFile = File(...)):  # Отримує файл від користувача
    # Перевіряємо чи модель завантажена
    if predictor is None:
        raise HTTPException(503, detail=startup_error or "Модель не завантажена. Спочатку запустіть train_models.py")

    # Перевіряємо розширення файлу
    suffix = Path(file.filename or "img.jpg").suffix.lower()  # Отримуємо розширення
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:  # Перевіряємо формат
        raise HTTPException(400, detail="Непідтримуваний формат. Використовуйте JPG або PNG.")

    # Розпізнаємо зображення
    image_bytes = await file.read()  # Читаємо байти файлу
    result = predictor.predict_bytes(image_bytes)  # Розпізнаємо знак
    return JSONResponse(build_payload(result))  # Повертаємо результат


@app.get("/api/test/random")  # Випадкове тестове зображення
def test_random():  # Розпізнає випадкове зображення з тесту
    if predictor is None:  # Перевіряємо модель
        raise HTTPException(503, detail="Модель не завантажена.")
    if not test_samples:  # Перевіряємо тестові зображення
        raise HTTPException(404, detail="Тестові зображення відсутні.")
    idx = random.randrange(len(test_samples))  # Вибираємо випадковий індекс
    return JSONResponse(_run_sample(idx))  # Розпізнаємо та повертаємо результат


@app.get("/api/test/next/{current_idx}")  # Наступне тестове зображення
def test_next(current_idx: int):  # Розпізнає наступне зображення
    if predictor is None:  # Перевіряємо модель
        raise HTTPException(503, detail="Модель не завантажена.")
    if not test_samples:  # Перевіряємо тестові зображення
        raise HTTPException(404, detail="Тестові зображення відсутні.")

    # Вибираємо індекс, відмінний від поточного
    if len(test_samples) == 1:  # Якщо одне зображення
        next_idx = 0
    else:  # Якщо більше одного
        next_idx = current_idx  # Починаємо з поточного
        # Генеруємо новий індекс доки він не змінится
        while next_idx == current_idx:
            next_idx = random.randrange(len(test_samples))
    return JSONResponse(_run_sample(next_idx))  # Розпізнаємо та повертаємо


@app.get("/api/test-image/{idx}")  # Отримання тестового зображення
def get_test_image(idx: int):  # Повертає зображення за індексом
    if idx < 0 or idx >= len(test_samples):  # Перевіряємо індекс
        raise HTTPException(404, detail="Зображення не знайдено.")
    return FileResponse(test_samples[idx]["path"])  # Повертаємо файл зображення


def _run_sample(idx: int) -> dict:  # Допоміжна функція для розпізнавання
    """Запускає розпізнавання на тестовому зображенні."""
    sample = test_samples[idx]  # Отримуємо тестовий зразок
    # Розпізнаємо зображення
    result = predictor.predict_bytes(sample["path"].read_bytes())  # Розпізнаємо
    payload = build_payload(result)  # Форматуємо результат
    payload["sample_index"] = idx  # Додаємо індекс
    payload["true_label"] = sample["true_label"]  # Справжня назва
    payload["true_label_ua"] = translate_label(sample["true_label"])  # Українською
    payload["image_url"] = f"/api/test-image/{idx}"  # URL для отримання зображення
    return payload  # Повертаємо результат
