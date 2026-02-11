"""OpenCV classical inpainting.

Provides two algorithms:
- Telea (2004): Fast marching method. Good for small regions.
- Navier-Stokes (NS): PDE-based, slightly better quality for larger regions.

These are purely classical (no ML) and very fast, but produce visibly worse
results than neural inpainting on large masked regions. Useful as a baseline
and for quick iteration.
"""

import cv2
import numpy as np

from kiero.inpainters.base import Inpainter


class OpenCVInpainter(Inpainter):
    """OpenCV classical inpainting."""

    def __init__(self, method: str = "telea", radius: int = 3):
        """Initialize the OpenCV inpainter.

        Args:
            method: Algorithm to use. "telea" or "ns" (Navier-Stokes).
            radius: Radius of the neighborhood for inpainting. Larger values
                consider more surrounding pixels but are slower.
        """
        if method not in ("telea", "ns"):
            raise ValueError(f"Unknown method '{method}'. Use 'telea' or 'ns'.")
        self._method = method
        self._radius = radius

    @property
    def name(self) -> str:
        return f"opencv({self._method})"

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using OpenCV classical methods."""
        flag = cv2.INPAINT_TELEA if self._method == "telea" else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, self._radius, flag)
