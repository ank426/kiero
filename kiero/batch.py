import random
import shutil
import time
from collections.abc import Callable
from contextlib import contextmanager
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


@contextmanager
def _timed_batch(input_path: Path, output_path: Path):
    t0 = time.time()
    image_paths = _get_image_paths(input_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  Source: directory ({len(image_paths)} images)")
    yield image_paths, output_path
    n = len(image_paths)
    print(f"\n  Batch complete: {n} images in {time.time() - t0:.1f}s ({(time.time() - t0) / n:.1f}s/image avg)")


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


def _validate_shapes(images: list[np.ndarray], ref_shape: tuple[int, int] | None) -> tuple[int, int]:
    shapes = {img.shape[:2] for img in images}
    if ref_shape is not None:
        shapes.add(ref_shape)
    if len(shapes) > 1:
        raise ValueError(f"Cannot compute shared mask: mixed dimensions {shapes}. Use --per-image mode instead.")
    return next(iter(shapes))


def _chunk_size(sample_path: Path, memory_mb: int) -> int:
    return max(1, (memory_mb * 1024 * 1024) // load_image(sample_path).nbytes)


def _collect_shared_mask(
    image_paths: list[Path],
    detector: WatermarkDetector,
    sample_n: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = 1024,
) -> np.ndarray:
    if sample_n is not None and sample_n < len(image_paths):
        sampled = sorted(random.sample(image_paths, sample_n), key=lambda p: p.name)
    else:
        sampled, sample_n = image_paths, len(image_paths)

    n, chunk = len(sampled), _chunk_size(sampled[0], memory_mb)
    print(f"  Computing shared mask from {n} images ({chunk} per batch)...")

    mask_sum: np.ndarray | None = None
    ref_shape: tuple[int, int] | None = None
    det_t0 = time.time()

    for i in range(0, n, chunk):
        images = [load_image(p) for p in sampled[i : i + chunk]]
        ref_shape = _validate_shapes(images, ref_shape)
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


def detect_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    sample_n: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = 1024,
) -> None:
    image_paths = _get_image_paths(input_path)
    print(f"  Source: directory ({len(image_paths)} images)")
    print(f"  Sample: {sample_n or 'all'}, confidence: {confidence}")
    mask = _collect_shared_mask(
        image_paths, detector, sample_n=sample_n, confidence=confidence, memory_mb=memory_mb
    )
    save_image(mask, output_path)
    print(f"Shared mask saved to {output_path}")


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
    with _timed_batch(input_path, output_path) as (image_paths, out_dir):
        if per_image:
            print("\n  Per-image mode: detecting and inpainting each image...")

            def _detect_and_inpaint(image: np.ndarray) -> tuple[np.ndarray, str]:
                pct = mask_ratio(mask := detector.detect(image))
                return (inpainter.inpaint(image, mask) if pct > 0 else image), f"{pct:.1%} masked"

            _process_images(image_paths, out_dir, input_path, _detect_and_inpaint)
            return

        shared_mask = _collect_shared_mask(
            image_paths, detector, sample_n=sample_n, confidence=confidence, memory_mb=memory_mb
        )
        if mask_output:
            save_image(shared_mask, mask_output)
            print(f"  Shared mask saved to {mask_output}")

        if mask_ratio(shared_mask) == 0:
            print("  No watermark detected in shared mask. Copying originals.")
            for i, p in enumerate(image_paths):
                out_p = out_dir / p.relative_to(input_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, out_p)
                print(f"  [{i + 1}/{len(image_paths)}] {p.name} (copied)")
        else:
            print(f"\n  Inpainting {len(image_paths)} images with shared mask...")
            _process_images(image_paths, out_dir, input_path, lambda img: (inpainter.inpaint(img, shared_mask), ""))


def inpaint_batch(input_path: Path, output_path: Path, mask: np.ndarray, inpainter: Inpainter) -> None:
    empty = mask_ratio(mask) == 0
    if empty:
        print("  Mask is empty — nothing to inpaint.")
    with _timed_batch(input_path, output_path) as (image_paths, out_dir):
        _process_images(image_paths, out_dir, input_path, lambda img: (img if empty else inpainter.inpaint(img, mask), ""))
