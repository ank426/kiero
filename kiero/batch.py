import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from kiero.detectors.base import WatermarkDetector
from kiero.inpainters.base import Inpainter
from kiero.utils import is_image, load_image, mask_ratio, save_image


def _get_image_paths(input_dir: Path) -> list[Path]:
    images = sorted(
        (p for p in input_dir.rglob("*") if p.is_file() and is_image(p)),
        key=lambda p: p.name,
    )
    if not images:
        raise FileNotFoundError(f"No image files found in {input_dir}")
    return images


def _process_images(
    image_paths: list[Path], out_dir: Path, input_dir: Path, fn: Callable[[np.ndarray], tuple[np.ndarray, str]]
) -> None:
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        t0 = time.time()
        result, extra = fn(load_image(p))
        out_p = out_dir / p.relative_to(input_dir)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_image(result, out_p)
        print(f"  [{i + 1}/{n}] {p.name} ({time.time() - t0:.1f}s{f', {extra}' if extra else ''})")


def _collect_shared_mask(
    image_paths: list[Path],
    detector: WatermarkDetector,
    sample: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = 1024,
) -> np.ndarray:
    if sample is not None and sample < len(image_paths):
        sampled = sorted(random.sample(image_paths, sample), key=lambda p: p.name)
    else:
        sampled, sample = image_paths, len(image_paths)

    n = len(sampled)
    chunk = max(1, (memory_mb * 1024 * 1024) // load_image(sampled[0]).nbytes)
    print(f"  Computing shared mask from {n} images ({chunk} per batch)...")

    mask_sum: np.ndarray | None = None
    ref_shape: tuple[int, int] | None = load_image(sampled[0].shape[:2])
    det_t0 = time.time()

    for i in range(0, n, chunk):
        images = [load_image(p) for p in sampled[i : i + chunk]]
        if images.shape[:2] != ref_shape:
            raise ValueError(f"Cannot compute shared mask: mixed dimensions {ref_shape, images.shape[:2]}. Use --per-image mode instead.")  # noqa: E501
        for m in detector.detect_batch(images):
            m_f = m.astype(np.float32) / 255.0
            mask_sum = m_f if mask_sum is None else mask_sum + m_f
        print(f"  Processed {min(i + chunk, n)}/{n} images...")

    det_time = time.time() - det_t0
    print(f"  Detection done in {det_time:.1f}s ({det_time / n:.2f}s/image)")
    if mask_sum is None:
        raise ValueError("No images to process")

    shared_mask = ((mask_sum / n) >= confidence).astype(np.uint8) * 255
    print(f"  Shared mask: {mask_ratio(shared_mask):.1%} of image masked")
    return shared_mask


def run_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    inpainter: Inpainter,
    per_image: bool = False,
    sample_n: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = 1024,
    mask_output: Path | None = None,
) -> None:
    if per_image:
        t0 = time.time()
        image_paths = _get_image_paths(input_path)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  Source: directory ({len(image_paths)} images)")
        print("\n  Per-image mode: detecting and inpainting each image...")

        def _detect_and_inpaint(image: np.ndarray) -> tuple[np.ndarray, str]:
            pct = mask_ratio(mask := detector.detect(image))
            return (inpainter.inpaint(image, mask) if pct > 0 else image), f"{pct:.1%} masked"

        _process_images(image_paths, output_path, input_path, _detect_and_inpaint)

        n = len(image_paths)
        print(f"\n  Batch complete: {n} images in {time.time() - t0:.1f}s ({(time.time() - t0) / n:.1f}s/image avg)")
    else:
        if mask_output is not None:
            mask_path = mask_output
            cleanup = False
        else:
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.close()
            mask_path = Path(f.name)
            cleanup = True

        try:
            detect_batch(
                input_path=input_path,
                output_path=mask_path,
                detector=detector,
                sample=sample_n,
                confidence=confidence,
                memory_mb=memory_mb,
            )
            mask = load_image(mask_path)
            inpaint_batch(
                input_path=input_path,
                output_path=output_path,
                mask=mask,
                inpainter=inpainter,
            )
        finally:
            if cleanup and mask_path.exists():
                os.remove(mask_path)


def detect_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    sample: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = 1024,
) -> None:
    image_paths = _get_image_paths(input_path)
    print(f"  Source: directory ({len(image_paths)} images)")
    print(f"  Sample: {sample or 'all'}, confidence: {confidence}")
    mask = _collect_shared_mask(
        image_paths, detector, sample=sample, confidence=confidence, memory_mb=memory_mb
    )
    save_image(mask, output_path)
    print(f"Shared mask saved to {output_path}")


def inpaint_batch(input_path: Path, output_path: Path, mask: np.ndarray, inpainter: Inpainter) -> None:
    t0 = time.time()
    image_paths = _get_image_paths(input_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  Source: directory ({len(image_paths)} images)")

    empty = mask_ratio(mask) == 0
    if empty:
        print("  Mask is empty — nothing to inpaint.")

    _process_images(
        image_paths, output_path, input_path, lambda img: (img if empty else inpainter.inpaint(img, mask), "")
    )

    n = len(image_paths)
    print(f"\n  Batch complete: {n} images in {time.time() - t0:.1f}s ({(time.time() - t0) / n:.1f}s/image avg)")
