"""Base class for inpainters."""

from abc import ABC, abstractmethod

import numpy as np


class Inpainter(ABC):
    """Abstract base class for inpainters.

    All inpainters must implement the `inpaint` method, which takes an image
    and a binary mask, and returns a clean image with the masked regions filled.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this inpainter."""
        ...

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint masked regions of an image.

        Args:
            image: Input image, shape (H, W, 3), dtype uint8, BGR format.
            mask: Binary mask, shape (H, W), dtype uint8, 255 = region to inpaint.

        Returns:
            Inpainted image, shape (H, W, 3), dtype uint8, BGR format.
        """
        ...
