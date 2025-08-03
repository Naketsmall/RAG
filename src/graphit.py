import importlib
import os

import numpy as np

import src.model_loaders
from neo4j import GraphDatabase

from config.configuration import YOLO_PATH, API_KEY
from src.model_loaders import YOLOLoader, LLMLoader

import json

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SceneObject:
    id: str
    class_name: str
    bbox: List[float]
    features: Dict[str, any] = field(default_factory=dict)
    neighbours: Dict[str, List[str]] = field(default_factory=dict)  # {"on": ["table_1"], "near": [...]}

    def add_neighbour(self, relation: str, obj_id: str):
        if relation not in self.neighbours:
            self.neighbours[relation] = []
        if obj_id not in self.neighbours[relation]:
            self.neighbours[relation].append(obj_id)

    def to_graph_node(self) -> Dict:
        return {
            "id": self.id,
            "class": self.class_name,
            "features": self.features,
            "neighbours": self.neighbours
        }

class Graphit:
    def __init__(self, classes_config: str = "configs/classes.json", relations_config: str = "configs/relations.json"):
        self.yolo = YOLOLoader.load_model('../' + YOLO_PATH)
        self.llm = LLMLoader.load_model(API_KEY)

        with open(classes_config) as f:
            self.classes_config = json.load(f)

        with open(relations_config) as f:
            self.relations_config = json.load(f)

    def find_objects(self, images_dir):
        image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
        results = self.yolo.predict(image_paths)
        for (i, result) in enumerate(results):
            result.save(filename=f"./../images/results/result_{i}.jpg")
        return results


    def build_from_detection(self, img: np.ndarray, yolo_results: List[dict]) -> List[SceneObject]:
        # TODO: сейчас функция пока для одной картинки по сути (одномерный размер массива)
        objects = []

        for obj in yolo_results:
            class_cfg = self.classes_config.get(obj["class"], {})
            x1, y1, x2, y2 = map(int, obj['bbox'])
            roi = img[y1:y2, x1:x2]
            obj_features = self._extract_features(roi, obj, class_cfg)

            objects.append(SceneObject(
                id=f"{obj['class']}_{len(objects)}",
                class_name=obj["class"],
                bbox=obj["bbox"],
                features=obj_features
            ))


        #self._detect_relations(objects)
        return objects

    def _extract_features(self, roi: np.ndarray, obj: dict, class_cfg: dict) -> dict:
        features = {}
        for feat_name, feat_cfg in class_cfg.get("features", {}).items():
            # Если метод определен - вызываем соответствующий детектор
            if "method" in feat_cfg:
                detector = importlib.import_module(f"detectors.{feat_cfg['method']}")
                features[feat_name] = detector.extract(roi, obj)
            elif "default" in feat_cfg:
                features[feat_name] = feat_cfg["default"]
        return features

    def _detect_relations(self, objects: List[SceneObject]):
        # Реализация алгоритма из предыдущего ответа (IoU + координаты)
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1:]:
                relation = detect_relation(obj_a.bbox, obj_b.bbox)
                if relation:
                    # Добавляем двунаправленные связи для "near"
                    if relation == "near":
                        obj_a.add_neighbour("near", obj_b.id)
                        obj_b.add_neighbour("near", obj_a.id)
                    else:
                        # Для "on" и "inside" - направленные
                        obj_a.add_neighbour(relation, obj_b.id)

