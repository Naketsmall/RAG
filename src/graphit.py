import os

import src.model_loaders
from neo4j import GraphDatabase

from config.configuration import YOLO_PATH, API_KEY
from src.model_loaders import YOLOLoader, LLMLoader


class Graphit:
    def __init__(self):
        self.yolo = YOLOLoader.load_model('../' + YOLO_PATH)
        self.llm = LLMLoader.load_model(API_KEY)

    def find_objects(self, images_dir, debug=0):
        image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        results = self.yolo.predict(image_paths)
        if not debug:
            return results

        print(os.listdir('./'))
        for i, result in enumerate(results):
            print(f"Результат для {image_paths[i]}:")
            objects = {}
            for obj in result.boxes:
                obj_name = self.yolo.names[obj.cls[0].item()]
                if obj_name in objects.keys():
                    objects[obj_name] += 1
                else:
                    objects[obj_name] = 1
            print("Обнаружены объекты:", objects)
            result.save(filename=f"./../images/results/result_{i}.jpg")
        return results

    #def check_scene(self, images_dir):
     #   object = self.find_objects()
