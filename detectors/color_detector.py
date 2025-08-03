import cv2
import numpy as np
from collections import Counter


def extract(roi: np.ndarray, obj: dict) -> str:
    """
    Определяет доминирующий цвет объекта на картинке.
    Возвращает название цвета ("red", "blue", etc).
    """
    # TODO: Функция работает отстойно. Надо менять на CNN
    # TODO: Убрал обрезание по bbox отсюда. Не вижу смысла это делать в детекторах фичи. Надо перед передачей.

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    color_ranges = {
        "red": ((0, 50, 50), (10, 255, 255)),
        "blue": ((100, 50, 50), (130, 255, 255)),
        "green": ((40, 50, 50), (80, 255, 255)),
        "yellow": ((20, 50, 50), (40, 255, 255))
    }

    max_pixels = 0
    dominant_color = "unknown"

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        pixel_count = cv2.countNonZero(mask)
        if pixel_count > max_pixels:
            max_pixels = pixel_count
            dominant_color = color

    return dominant_color

