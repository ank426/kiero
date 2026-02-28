from abc import ABC, abstractmethod

import numpy as np


class WatermarkDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray: ...

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray] | np.ndarray:
        return [self.detect(img) for img in images]
