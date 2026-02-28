import time
from pathlib import Path

import numpy as np

from kiero.detectors.base import WatermarkDetector
from kiero.inpainters.base import Inpainter
from kiero.utils import binarize_mask, load_image, load_mask, make_detector, make_inpainter, mask_ratio, save_image


def detect(
    image_path: Path,
    output_path: Path | None = None,
    *,
    confidence: float = 0.25,
    padding: int = 10,
    device: str | None = None,
    detector: WatermarkDetector | None = None,
) -> np.ndarray:
    image = load_image(image_path)
    detector = detector or make_detector(confidence, padding, device)
    t0 = time.time()
    mask_raw = detector.detect(image)
    elapsed = time.time() - t0
    print(f"  Detection done in {elapsed:.1f}s — {mask_ratio(binarize_mask(mask_raw)):.1%} masked")
    mask = binarize_mask(mask_raw)
    if output_path is not None:
        save_image(mask, output_path)
        print(f"  Mask saved to {output_path}")
    return mask


def inpaint(
    image_path: Path,
    output_path: Path,
    mask: np.ndarray | Path,
    *,
    device: str | None = None,
    inpainter: Inpainter | None = None,
) -> None:
    image = load_image(image_path)
    mask = load_mask(mask) if isinstance(mask, Path) else mask
    inpainter = inpainter or make_inpainter(device)
    t0 = time.time()
    result = inpainter.inpaint(image, mask)
    elapsed = time.time() - t0
    print(f"  Inpainting done in {elapsed:.1f}s")
    save_image(result, output_path)
    print(f"  Result saved to {output_path}")


def run(
    image_path: Path,
    output_path: Path,
    mask_path: Path | None = None,
    *,
    confidence: float = 0.25,
    padding: int = 10,
    device: str | None = None,
    detector: WatermarkDetector | None = None,
    inpainter: Inpainter | None = None,
) -> None:
    t0 = time.time()
    mask = detect(
        image_path,
        output_path=mask_path,
        confidence=confidence,
        padding=padding,
        device=device,
        detector=detector,
    )
    det_time = time.time() - t0
    if mask_ratio(mask) == 0:
        print("  No watermark detected, skipping inpainting.")
        image = load_image(image_path)
        save_image(image, output_path)
        print(f"  Result saved to {output_path}")
        return
    t1 = time.time()
    inpaint(image_path, output_path, mask, device=device, inpainter=inpainter)
    inp_time = time.time() - t1
    print(f"  Total: {det_time + inp_time:.1f}s")
