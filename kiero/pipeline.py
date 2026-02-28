import time
from pathlib import Path

import numpy as np

from kiero.detectors.yolo import YoloDetector
from kiero.inpainters.lama import LamaInpainter
from kiero.utils import binarize_mask, load_image, load_mask, mask_ratio, save_image


class Pipeline:
    def __init__(self, confidence: float = 0.25, padding: int = 10, device: str | None = None) -> None:
        self._detector = YoloDetector(confidence=confidence, padding=padding, device=device)
        self._inpainter = LamaInpainter(device=device)

    def _detect(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.time()
        mask = self._detector.detect(image)
        elapsed = time.time() - t0
        print(f"  Detection done in {elapsed:.1f}s — {mask_ratio(binarize_mask(mask)):.1%} masked")
        return mask, elapsed

    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.time()
        result = self._inpainter.inpaint(image, mask)
        elapsed = time.time() - t0
        print(f"  Inpainting done in {elapsed:.1f}s")
        return result, elapsed

    def detect(self, image_path: Path, output_path: Path) -> np.ndarray:
        mask_raw, _ = self._detect(load_image(image_path))
        mask = binarize_mask(mask_raw)
        save_image(mask, output_path)
        print(f"  Mask saved to {output_path}")
        return mask

    def inpaint(self, image_path: Path, output_path: Path, mask_path: Path) -> np.ndarray:
        result, _ = self._inpaint(load_image(image_path), load_mask(mask_path))
        save_image(result, output_path)
        print(f"  Result saved to {output_path}")
        return result

    def run(self, image_path: Path, output_path: Path, mask_path: Path | None = None) -> np.ndarray:
        image = load_image(image_path)
        mask_raw, det_time = self._detect(image)
        mask = binarize_mask(mask_raw)
        if mask_path:
            save_image(mask, mask_path)
            print(f"  Mask saved to {mask_path}")
        if mask_ratio(mask) == 0:
            print("  No watermark detected, skipping inpainting.")
            save_image(image, output_path)
            print(f"  Result saved to {output_path}")
            return image
        result, inp_time = self._inpaint(image, mask)
        print(f"  Total: {det_time + inp_time:.1f}s")
        save_image(result, output_path)
        print(f"  Result saved to {output_path}")
        return result
