from dataclasses import dataclass  # Клас з автоматичними методами
from pathlib import Path  # Робота з шляхами

import cv2  # Обробка зображень
import joblib  # Збереження моделей
import numpy as np  # Числові операції

from traffic_sign_system.config import HOG_WEIGHT, PIXEL_WEIGHT  # Ваги моделей
from traffic_sign_system.features import extract_hog_features, extract_pixel_features, get_prediction_views  # Ознаки
from traffic_sign_system.recommendations import build_recommendation  # Поради


@dataclass  # Автоматичний __init__ та __repr__
class TrafficSignPredictor:  # Клас для розпізнавання знаків
    hog_model: object  # HOG + LogisticRegression
    pixel_model: object  # Pixels + PCA + KNN
    class_names: list  # Назви дорожних знаків
    image_size: tuple = (64, 64)  # Розмір зображень

    def save(self, path: str | Path) -> None:  # Збереження моделей
        path = Path(path)  # Конвертуємо в Path
        path.parent.mkdir(parents=True, exist_ok=True)  # Створюємо папку
        joblib.dump({  # Зберігаємо словник
            "hog_model": self.hog_model,
            "pixel_model": self.pixel_model,
            "class_names": self.class_names,
            "image_size": self.image_size,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "TrafficSignPredictor":  # Завантаження моделей
        bundle = joblib.load(path)  # Завантажуємо
        return cls(  # Повертаємо об'єкт
            hog_model=bundle["hog_model"],
            pixel_model=bundle["pixel_model"],
            class_names=bundle["class_names"],
            image_size=tuple(bundle.get("image_size", (64, 64))),
        )

    def predict_bytes(self, image_bytes: bytes) -> dict:  # Розпізнаємо з байтів
        arr = np.frombuffer(image_bytes, dtype=np.uint8)  # Конвертуємо в масив
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # Декодуємо зображення
        if image is None:  # Якщо не вдалося декодувати
            raise ValueError("Не вдалося розпізнати зображення.")
        return self.predict_array(image)  # Розпізнаємо

    def predict_array(self, image: np.ndarray) -> dict:  # Розпізнаємо з масиву
        views = get_prediction_views(image, self.image_size)  # Отримуємо варіанти зображення

        # Збираємо ймовірності з усіх варіантів зображення
        hog_probs = []  # Ймовірності HOG
        pixel_probs = []  # Ймовірності Pixel
        for view in views:
            hog_probs.append(self.hog_model.predict_proba(extract_hog_features(view).reshape(1, -1))[0])
            pixel_probs.append(self.pixel_model.predict_proba(extract_pixel_features(view).reshape(1, -1))[0])

        # Усереднюємо та комбінуємо
        mean_hog = np.mean(hog_probs, axis=0)  # Середнє HOG
        mean_pixel = np.mean(pixel_probs, axis=0)  # Середнє Pixel
        combined = HOG_WEIGHT * mean_hog + PIXEL_WEIGHT * mean_pixel  # Комбінована ймовірність

        best_idx = int(np.argmax(combined))  # Найкращий клас
        hog_idx = int(np.argmax(mean_hog))  # Клас HOG
        pixel_idx = int(np.argmax(mean_pixel))  # Клас Pixel

        # Топ-3 кандидати
        top_indices = np.argsort(combined)[::-1][:3]  # Три найкращих
        top_candidates = [
            {"label": self.class_names[i], "probability": round(float(combined[i]), 4)}
            for i in top_indices
        ]

        confidence = self._calc_confidence(combined, hog_idx == best_idx, pixel_idx == best_idx)  # Обчислюємо впевненість

        return {  # Повертаємо результати
            "predicted_class": self.class_names[best_idx],  # Розпізнаний знак
            "confidence": round(float(confidence), 4),  # Впевненість (0-1)
            "model_votes": {  # Голоси моделей
                "hog": self.class_names[hog_idx],  # Вибір HOG
                "pixel": self.class_names[pixel_idx],  # Вибір Pixel
            },
            "model_confidence": {  # Впевненість кожної моделі
                "hog": round(float(np.max(mean_hog)), 4),  # HOG впевненість
                "pixel": round(float(np.max(mean_pixel)), 4),  # Pixel впевненість
            },
            "top_candidates": top_candidates,  # Топ-3 варіанти
            "recommendation": build_recommendation(self.class_names[best_idx], confidence),  # Порада водієві
        }

    def _calc_confidence(self, probs: np.ndarray, hog_agrees: bool, pixel_agrees: bool) -> float:  # Обчислюємо впевненість
        """Обчислює впевненість на основі ймовірності, відриву та згоди моделей."""
        best = float(np.max(probs))  # Найвища ймовірність
        sorted_p = np.sort(probs)[::-1]  # Сортуємо в порядку спадання
        margin = best - float(sorted_p[1]) if len(sorted_p) > 1 else 0.0  # Відрив до другого місця

        # Бонус якщо обидві моделі згодні, штраф якщо розходяться
        if hog_agrees and pixel_agrees:  # Обидві згідні
            agreement = 0.08  # Бонус
        elif hog_agrees or pixel_agrees:  # Одна згідна
            agreement = 0.0  # Без змін
        else:  # Жодна не згідна
            agreement = -0.10  # Штраф

        # Формула впевненості: 55% ймовірність + 35% відрив + 10% базова + бонус/штраф
        confidence = 0.55 * best + 0.35 * margin + 0.10 + agreement
        return max(min(confidence, 0.97), 0.05)  # Обмежуємо від 0.05 до 0.97
