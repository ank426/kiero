import time
from pathlib import Path

import numpy as np

from kiero.detectors.yolo import YoloDetector
from kiero.inpainters.lama import LamaInpainter
from kiero.utils import load_image, save_image, mask_stats


class Pipeline:
    def __init__(
        self,
        confidence: float = 0.25,
        padding: int = 10,
        device: str | None = None,
    ):
        self._detector = YoloDetector(
            confidence=confidence,
            padding=padding,
            device=device,
        )
        self._inpainter = LamaInpainter(device=device)

    def detect(self, image_path: str | Path) -> np.ndarray:
        image = load_image(image_path)
        t0 = time.time()
        mask = self._detector.detect(image)
        elapsed = time.time() - t0
        _, _, pct = mask_stats(mask)
        print(f"  Detection done in {elapsed:.1f}s — {pct:.1f}% masked")
        return mask

    def run(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        mask_path: str | Path | None = None,
    ) -> np.ndarray:
        image = load_image(image_path)

        t0 = time.time()
        mask = self._detector.detect(image)
        det_time = time.time() - t0
        n_masked, _, pct = mask_stats(mask)
        print(f"  Detection done in {det_time:.1f}s — {pct:.1f}% masked")

        if mask_path:
            save_image(mask, mask_path)
            print(f"  Mask saved to {mask_path}")

        if n_masked == 0:
            print("  No watermark detected, skipping inpainting.")
            if output_path:
                save_image(image, output_path)
            return image

        t0 = time.time()
        result = self._inpainter.inpaint(image, mask)
        inp_time = time.time() - t0
        print(f"  Inpainting done in {inp_time:.1f}s")
        print(f"  Total: {det_time + inp_time:.1f}s")

        if output_path:
            save_image(result, output_path)
            print(f"  Result saved to {output_path}")

        return result
