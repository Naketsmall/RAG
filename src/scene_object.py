from dataclasses import dataclass, field
from typing import List, Dict


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

    def get_semantic_repr(self) -> str:
        parts = [f"{self.class_name}"]
        for feature_name, feature_value in self.features.items():
            parts.append(f"{feature_name} {feature_value}")
        return ". ".join(parts)
