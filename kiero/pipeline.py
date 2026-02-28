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

    def detect(self, image_path: Path, output_path: Path | None = None) -> np.ndarray:
        image = load_image(image_path)
        t0 = time.time()
        mask_raw = self._detector.detect(image)
        elapsed = time.time() - t0
        print(f"  Detection done in {elapsed:.1f}s — {mask_ratio(binarize_mask(mask_raw)):.1%} masked")
        mask = binarize_mask(mask_raw)
        if output_path is not None:
            save_image(mask, output_path)
            print(f"  Mask saved to {output_path}")
        return mask

    def inpaint(self, image_path: Path, output_path: Path, mask: np.ndarray | Path) -> None:
        image = load_image(image_path)
        mask = load_mask(mask) if isinstance(mask, Path) else mask
        t0 = time.time()
        result = self._inpainter.inpaint(image, mask)
        elapsed = time.time() - t0
        print(f"  Inpainting done in {elapsed:.1f}s")
        save_image(result, output_path)
        print(f"  Result saved to {output_path}")

    def run(self, image_path: Path, output_path: Path, mask_path: Path | None = None) -> None:
        t0 = time.time()
        mask = self.detect(image_path, output_path=mask_path)
        det_time = time.time() - t0
        if mask_ratio(mask) == 0:
            print("  No watermark detected, skipping inpainting.")
            image = load_image(image_path)
            save_image(image, output_path)
            print(f"  Result saved to {output_path}")
        t1 = time.time()
        self.inpaint(image_path, output_path, mask)
        inp_time = time.time() - t1
        print(f"  Total: {det_time + inp_time:.1f}s")
