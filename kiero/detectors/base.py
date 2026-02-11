"""Base class for watermark detectors."""

from abc import ABC, abstractmethod

import numpy as np


class WatermarkDetector(ABC):
    """Abstract base class for watermark detectors.

    All detectors must implement the `detect` method, which takes an image
    and returns a binary mask where 255 = watermark, 0 = clean.

    Mask format convention:
        - dtype: uint8
        - shape: (H, W) — same spatial dimensions as the input image
        - values: 0 (clean pixel) or 255 (watermark pixel)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this detector."""
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect watermark regions in an image.

        Args:
            image: Input image as a numpy array, shape (H, W, 3), dtype uint8, BGR format.

        Returns:
            Binary mask, shape (H, W), dtype uint8, values 0 or 255.
        """
        ...
