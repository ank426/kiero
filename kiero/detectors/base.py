"""Base class for watermark detectors."""

from abc import ABC, abstractmethod

import numpy as np


class WatermarkDetector(ABC):
    """Abstract base class for watermark detectors.

    All detectors must implement the `detect` method, which takes an image
    and returns a binary mask where 255 = watermark, 0 = clean.

    Detectors that support GPU batching should override `detect_batch` for
    improved throughput when processing multiple images (e.g. for shared mask
    averaging across a directory or CBZ).

    Mask format convention:
        - dtype: uint8
        - shape: (H, W) — same spatial dimensions as the input image
        - values: 0 (clean pixel) or 255 (watermark pixel)
    """

    # Default batch size hint. Subclasses override with their own sensible default.
    default_batch_size: int = 1

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

    def detect_batch(
        self, images: list[np.ndarray], batch_size: int | None = None
    ) -> list[np.ndarray]:
        """Detect watermark regions in multiple images.

        Default implementation loops over `detect()`. Subclasses with native
        GPU batch support (YOLO, CLIPSeg) should override this for vectorized
        inference.

        Args:
            images: List of input images, each shape (H, W, 3), dtype uint8, BGR.
            batch_size: Max images per GPU forward pass. If None, uses
                ``self.default_batch_size``.

        Returns:
            List of binary masks, one per input image.
        """
        return [self.detect(img) for img in images]
