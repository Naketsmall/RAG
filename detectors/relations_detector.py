from typing import Optional
from math import dist

from src.scene_object import SceneObject


def detect_relation(
        obj_a: SceneObject,
        obj_b: SceneObject,
        rules: list,
        img_diagonal: float
) -> Optional[str]:
    """
    Определяет отношение между obj_a и obj_b на основе правил.
    Правила проверяются в порядке их объявления.
    """
    for rule in rules:
        # Проверка классов (с учётом wildcard '*')
        class_a_ok = (rule["class_a"] == "*") or (obj_a.class_name == rule["class_a"])
        class_b_ok = (rule["class_b"] == "*") or (obj_b.class_name == rule["class_b"])
        if not (class_a_ok and class_b_ok):
            continue

        # Проверка условий
        cond = rule["conditions"]
        if cond["type"] == "iou_above":
            iou = calculate_iou(obj_a.bbox, obj_b.bbox)
            if iou >= cond["threshold"] and is_above(obj_a.bbox, obj_b.bbox):
                return rule["relation"]  # "on"

        elif cond["type"] == "iom_above":
            iom = calculate_iomin(obj_a.bbox, obj_b.bbox)
            if iom >= cond["threshold"] and is_above(obj_a.bbox, obj_b.bbox):
                return rule["relation"]  # "on"

        elif cond["type"] == "distance_below":
            center_a = bbox_center(obj_a.bbox)
            center_b = bbox_center(obj_b.bbox)
            distance = dist(center_a, center_b)
            if distance <= cond["threshold_ratio"] * img_diagonal:
                return rule["relation"]  # "near"

    return None


def calculate_iou(bbox_a, bbox_b):
    inter_x1 = max(bbox_a[0], bbox_b[0])
    inter_y1 = max(bbox_a[1], bbox_b[1])
    inter_x2 = min(bbox_a[2], bbox_b[2])
    inter_y2 = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (bbox_a[2]-bbox_a[0]) * (bbox_a[3]-bbox_a[1])
    area_b = (bbox_b[2]-bbox_b[0]) * (bbox_b[3]-bbox_b[1])
    return inter_area / (area_a + area_b - inter_area)


def calculate_iomin(bbox_a, bbox_b):
    inter_x1 = max(bbox_a[0], bbox_b[0])
    inter_y1 = max(bbox_a[1], bbox_b[1])
    inter_x2 = min(bbox_a[2], bbox_b[2])
    inter_y2 = min(bbox_a[3], bbox_b[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area_obj = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_surface = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    return float(inter_area / min(area_obj, area_surface))

def bbox_center(bbox):
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def is_above(bbox_a, bbox_b):
    return bbox_a[1] >= bbox_b[1]