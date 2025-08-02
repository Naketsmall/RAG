from ultralytics import YOLO
import cv2

from gigachat import GigaChat
#from config import API_KEY


class YOLOLoader:
    @staticmethod
    def load_model(path_to_model):
        return YOLO(path_to_model)

class LLMLoader:
    @staticmethod
    def load_model(api_key):
        return GigaChat(
            credentials=api_key,
            verify_ssl_certs=False,
            model="GigaChat-2"
        )



