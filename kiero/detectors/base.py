"""Base class for watermark detectors."""

from abc import ABC, abstractmethod

import numpy as np


class WatermarkDetector(ABC):
    """Abstract base class for watermark detectors.

    Mask format convention:
        - dtype: uint8
        - shape: (H, W) — same spatial dimensions as the input image
        - values: 0 (clean pixel) or 255 (watermark pixel)
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray: ...

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        return [self.detect(img) for img in images]
