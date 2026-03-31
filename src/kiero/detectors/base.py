from abc import ABC, abstractmethod

import numpy as np


class WatermarkDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray: ...

    def detect_batch(self, images: list[np.ndarray]) -> np.ndarray:
        # Default: average per-image masks into a single HxW float mask.
        if not images:
            return np.array([])
        masks = np.stack([self.detect(img) for img in images], axis=0)
        return masks.astype(np.float32).mean(axis=0)
