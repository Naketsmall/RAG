import cv2

from detectors.color_detector import extract_feature


image = cv2.imread("../images/yellow_cup.jpg")
#bbox = [50, 100, 200, 300]  # Пример bbox
bbox = [0, 0, 1279, 1279]
print(extract_feature(image, bbox))  # → "blue"