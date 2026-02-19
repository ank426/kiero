import time
from pathlib import Path

import numpy as np

from kiero.detectors.yolo import YoloDetector
from kiero.inpainters.lama import LamaInpainter
from kiero.utils import load_image, load_mask, mask_stats, save_image


class Pipeline:
    def __init__(self, confidence: float = 0.25, padding: int = 10, device: str | None = None):
        self._detector = YoloDetector(confidence=confidence, padding=padding, device=device)
        self._inpainter = LamaInpainter(device=device)

    def _detect(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.time()
        mask = self._detector.detect(image)
        elapsed = time.time() - t0
        print(f"  Detection done in {elapsed:.1f}s — {mask_stats(mask)[2]:.1f}% masked")
        return mask, elapsed

    def detect(self, image_path: str | Path, output_path: str | Path) -> np.ndarray:
        mask, _ = self._detect(load_image(image_path))
        save_image(mask, output_path)
        print(f"  Mask saved to {output_path}")
        return mask

    def inpaint(self, image_path: str | Path, mask_path: str | Path, output_path: str | Path) -> np.ndarray:
        image, mask = load_image(image_path), load_mask(mask_path)
        t0 = time.time()
        result = self._inpainter.inpaint(image, mask)
        print(f"  Inpainting done in {time.time() - t0:.1f}s")
        save_image(result, output_path)
        print(f"  Result saved to {output_path}")
        return result

    def run(self, image_path: str | Path, output_path: str | Path, mask_path: str | Path | None = None) -> np.ndarray:
        image = load_image(image_path)
        mask, det_time = self._detect(image)
        if mask_path:
            save_image(mask, mask_path)
            print(f"  Mask saved to {mask_path}")
        if np.count_nonzero(mask) == 0:
            print("  No watermark detected, skipping inpainting.")
            save_image(image, output_path)
            return image
        t0 = time.time()
        result = self._inpainter.inpaint(image, mask)
        inp_time = time.time() - t0
        print(f"  Inpainting done in {inp_time:.1f}s")
        print(f"  Total: {det_time + inp_time:.1f}s")
        save_image(result, output_path)
        print(f"  Result saved to {output_path}")
        return result
