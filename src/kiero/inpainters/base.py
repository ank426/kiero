from abc import ABC, abstractmethod

import cv2
import numpy as np


class Inpainter(ABC):
    @abstractmethod
    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img_h, img_w = image.shape[:2]
        if mask.shape[:2] != (img_h, img_w):
            print(f"  Warning: mask {mask.shape[1]}x{mask.shape[0]} != image {img_w}x{img_h}, resizing")
            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        return self._inpaint(image, mask)
