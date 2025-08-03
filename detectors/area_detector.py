import numpy as np


def extract(roi: np.ndarray, obj: dict) -> float:
    # TODO: пока заглушка
    if obj['class'] == 'dining table':
        return 345
    else:
        return .308