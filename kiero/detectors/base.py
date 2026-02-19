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

    default_batch_size: int = 1

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect watermark regions in an image.

        Args:
            image: Input image, shape (H, W, 3), dtype uint8, BGR.

        Returns:
            Binary mask, shape (H, W), dtype uint8, values 0 or 255.
        """
        ...

    def detect_batch(
        self, images: list[np.ndarray], batch_size: int | None = None
    ) -> list[np.ndarray]:
        """Detect watermark regions in multiple images.

        Default loops over ``detect()``. Override for GPU-batched inference.
        """
        return [self.detect(img) for img in images]
