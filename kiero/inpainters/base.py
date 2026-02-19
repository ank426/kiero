from abc import ABC, abstractmethod

import cv2
import numpy as np


class Inpainter(ABC):
    @abstractmethod
    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = self._conform_mask(mask, image)
        return self._inpaint(image, mask)

    @staticmethod
    def _conform_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        if (mask_h, mask_w) == (img_h, img_w):
            return mask
        print(
            f"  Warning: mask size ({mask_w}x{mask_h}) differs from image "
            f"({img_w}x{img_h}), resizing mask to match."
        )
        return cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
