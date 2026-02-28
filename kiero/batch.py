import random
import sys
import time
from pathlib import Path

import numpy as np

from kiero.detectors.base import WatermarkDetector
from kiero.inpainters.base import Inpainter
from kiero.utils import is_image, load_image, make_pipeline, mask_ratio, save_image


def _get_image_paths(input_dir: Path) -> list[Path]:
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and is_image(p))
    if not images:
        sys.exit(f"Error: No image files found in {input_dir}")
    return images


def run_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    inpainter: Inpainter,
    per_image: bool = False,
    confidence: float = 0.25,
    padding: int = 10,
    memory_mb: int = 1024,
    device: str | None = None,
    mask_output: Path | None = None,
) -> None:
    if per_image:
        t0 = time.time()
        image_paths = _get_image_paths(input_path)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"  Source: directory ({len(image_paths)} images)")
        print("\n  Per-image mode: detecting and inpainting each image...")

        pipeline = make_pipeline(confidence, padding=padding, device=device)
        n = len(image_paths)
        for i, p in enumerate(image_paths):
            t1 = time.time()
            out_p = output_path / p.relative_to(input_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            pipeline.run(p, out_p)
            print(f"  [{i + 1}/{n}] {p.name} ({time.time() - t1:.1f}s)")

        print(f"\n  Batch complete: {n} images in {time.time() - t0:.1f}s ({(time.time() - t0) / n:.1f}s/image avg)")

    else:
        mask = detect_batch(
            input_path=input_path,
            output_path=None,
            detector=detector,
            confidence=confidence,
            memory_mb=memory_mb,
        )
        if mask_output is not None:
            save_image(mask, mask_output)
            print(f"  Shared mask saved to {mask_output}")
        inpaint_batch(
            input_path=input_path,
            output_path=output_path,
            mask=mask,
            inpainter=inpainter,
        )


def detect_batch(
    input_path: Path,
    output_path: Path | None,
    detector: WatermarkDetector,
    sample: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = 1024,
) -> np.ndarray:
    image_paths = _get_image_paths(input_path)
    print(f"  Source: directory ({len(image_paths)} images)")
    print(f"  Sample: {sample or 'all'}, confidence: {confidence}")

    sampled = sorted(random.sample(image_paths, sample) if sample and sample < len(image_paths) else image_paths)

    memory_limit = memory_mb * 1024 * 1024
    n = len(sampled)

    print(f"  Computing shared mask from {n} images (~{memory_mb}MB per batch)...")

    mask_sum: np.ndarray | None = None
    ref_shape: tuple[int, int] | None = None

    batch: list[np.ndarray] = []
    batch_bytes = 0
    processed = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal mask_sum, processed, batch, batch_bytes
        if not batch:
            return
        k = len(batch)
        m_raw = detector.detect_batch(batch)
        # Expect a single averaged HxW mask in uint8 or float.
        m_arr = np.asarray(m_raw, dtype=np.float32)
        m = m_arr if (m_arr.size and m_arr.max() <= 1.0) else (m_arr / 255.0)
        mask_sum = m * k if mask_sum is None else mask_sum + m * k
        processed += k
        print(f"  Processed {processed}/{n} images...")
        batch, batch_bytes = [], 0

    for p in sampled:
        img = load_image(p)

        if ref_shape is None:
            ref_shape = img.shape[:2]
        elif img.shape[:2] != ref_shape:
            sys.exit(f"Error: {p} has shape {img.shape[:2]}, expected {ref_shape}")

        if img.nbytes > memory_limit:
            sys.exit(f"Error: {p} requires {img.nbytes / (1024**2):.1f}MB, exceeds limit of {memory_mb}MB")

        if batch and batch_bytes + img.nbytes > memory_limit:
            flush()

        batch.append(img)
        batch_bytes += img.nbytes

    flush()

    dt = time.time() - t0
    print(f"  Detection done in {dt:.1f}s ({dt / n:.2f}s/image)")

    shared_mask = ((mask_sum / n) >= confidence).astype(np.uint8) * 255
    print(f"  Shared mask: {mask_ratio(shared_mask):.1%} of image masked")

    if output_path is not None:
        save_image(shared_mask, output_path)
        print(f"  Shared mask saved to {output_path}")

    return shared_mask


def inpaint_batch(input_path: Path, output_path: Path, mask: np.ndarray, inpainter: Inpainter) -> None:
    t0 = time.time()
    image_paths = _get_image_paths(input_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  Source: directory ({len(image_paths)} images)")

    empty = mask_ratio(mask) == 0
    if empty:
        print("  Mask is empty — nothing to inpaint.")

    n = len(image_paths)
    for i, p in enumerate(image_paths):
        t1 = time.time()
        img = load_image(p)
        result = img if empty else inpainter.inpaint(img, mask)
        out_p = output_path / p.relative_to(input_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        save_image(result, out_p)
        print(f"  [{i + 1}/{n}] {p.name} ({time.time() - t1:.1f}s)")

    n = len(image_paths)
    print(f"\n  Batch complete: {n} images in {time.time() - t0:.1f}s ({(time.time() - t0) / n:.1f}s/image avg)")
