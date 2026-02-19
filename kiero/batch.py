import random
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from kiero.detectors.base import WatermarkDetector
from kiero.inpainters.base import Inpainter
from kiero.utils import (
    IMAGE_EXTENSIONS,
    list_images,
    load_image,
    save_image,
    mask_stats,
)

_DEFAULT_MEMORY_MB = 1024


def _chunk_size(sample_path: Path, memory_mb: int) -> int:
    per_image = load_image(sample_path).nbytes
    return max(1, (memory_mb * 1024 * 1024) // per_image)


def resolve_inputs(input_path: Path) -> tuple[list[Path], str, Path | None]:
    if input_path.is_dir():
        images = list_images(input_path)
        if not images:
            raise FileNotFoundError(f"No image files found in {input_path}")
        return images, "directory", None

    if input_path.suffix.lower() == ".cbz":
        return _extract_cbz(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    return [input_path], "file", None


def _extract_cbz(cbz_path: Path) -> tuple[list[Path], str, Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix="kiero_cbz_"))
    with zipfile.ZipFile(cbz_path, "r") as zf:
        # zip-slip prevention
        for member in zf.namelist():
            dest = (temp_dir / member).resolve()
            if not str(dest).startswith(str(temp_dir.resolve())):
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(
                    f"Zip slip detected in {cbz_path}: member '{member}' "
                    f"escapes extraction directory"
                )
        zf.extractall(temp_dir)

    images = sorted(
        (
            p
            for p in temp_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )
    if not images:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError(f"No image files found in CBZ: {cbz_path}")

    return images, "cbz", temp_dir


def _write_cbz(image_paths: list[Path], output_path: Path, root_dir: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in image_paths:
            zf.write(p, arcname=str(p.relative_to(root_dir)))


@contextmanager
def _batch_io(input_path: Path, output_path: Path):
    image_paths, source_type, temp_input_dir = resolve_inputs(input_path)
    print(f"  Source: {source_type} ({len(image_paths)} images)")

    out_dir: Path | None = None
    try:
        if source_type == "cbz":
            out_dir = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        else:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)

        yield image_paths, source_type, out_dir

        if source_type == "cbz":
            output_images = sorted(p for p in out_dir.rglob("*") if p.is_file())
            _write_cbz(output_images, Path(output_path), root_dir=out_dir)
            print(f"\n  CBZ written to {output_path}")
    finally:
        if temp_input_dir is not None:
            shutil.rmtree(temp_input_dir, ignore_errors=True)
        if source_type == "cbz" and out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)


def _process_images(
    image_paths: list[Path],
    out_dir: Path,
    process_fn: Callable[[np.ndarray], tuple[np.ndarray, str]],
) -> None:
    n = len(image_paths)
    for i, p in enumerate(image_paths):
        t0 = time.time()
        image = load_image(p)
        result, extra = process_fn(image)
        save_image(result, out_dir / p.name)
        elapsed = time.time() - t0
        suffix = f", {extra}" if extra else ""
        print(f"  [{i + 1}/{n}] {p.name} ({elapsed:.1f}s{suffix})")


def _validate_shapes(
    images: list[np.ndarray], ref_shape: tuple[int, int] | None
) -> tuple[int, int]:
    shapes = {img.shape[:2] for img in images}
    if ref_shape is not None:
        shapes.add(ref_shape)
    if len(shapes) > 1:
        raise ValueError(
            f"Cannot compute shared mask: mixed dimensions {shapes}. "
            f"Use --per-image mode instead."
        )
    return next(iter(shapes))


def collect_shared_mask(
    image_paths: list[Path],
    detector: WatermarkDetector,
    sample_n: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = _DEFAULT_MEMORY_MB,
) -> np.ndarray:
    if sample_n is not None and sample_n < len(image_paths):
        sampled = random.sample(image_paths, sample_n)
        sampled.sort(key=lambda p: p.name)
    else:
        sampled = image_paths
        sample_n = len(sampled)

    n = len(sampled)
    chunk = _chunk_size(sampled[0], memory_mb)
    print(f"  Computing shared mask from {n} images ({chunk} per batch)...")

    mask_sum: np.ndarray | None = None
    ref_shape: tuple[int, int] | None = None
    det_t0 = time.time()

    for chunk_start in range(0, n, chunk):
        chunk_paths = sampled[chunk_start : chunk_start + chunk]
        images = [load_image(p) for p in chunk_paths]

        ref_shape = _validate_shapes(images, ref_shape)
        masks = detector.detect_batch(images)

        for m in masks:
            m_f = m.astype(np.float32) / 255.0
            if mask_sum is None:
                mask_sum = m_f
            else:
                mask_sum += m_f

        print(f"  Processed {min(chunk_start + chunk, n)}/{n} images...")

    det_time = time.time() - det_t0
    print(f"  Detection done in {det_time:.1f}s ({det_time / n:.2f}s/image)")

    if mask_sum is None:
        raise ValueError("No images to process")

    shared_mask = ((mask_sum / n) >= confidence).astype(np.uint8) * 255

    _, _, pct = mask_stats(shared_mask)
    print(f"  Shared mask: {pct:.1f}% of image masked")
    return shared_mask


def detect_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    sample_n: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = _DEFAULT_MEMORY_MB,
) -> None:
    image_paths, source_type, temp_dir = resolve_inputs(input_path)
    print(f"  Source: {source_type} ({len(image_paths)} images)")

    try:
        sample_str = str(sample_n) if sample_n else "all"
        print(f"  Sample: {sample_str}, confidence: {confidence}")

        mask = collect_shared_mask(
            image_paths,
            detector,
            sample_n=sample_n,
            confidence=confidence,
            memory_mb=memory_mb,
        )
        save_image(mask, output_path)
        print(f"Shared mask saved to {output_path}")
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def _timed_batch(input_path: Path, output_path: Path):
    t0 = time.time()
    with _batch_io(input_path, output_path) as (image_paths, source_type, out_dir):
        yield image_paths, source_type, out_dir
        n = len(image_paths)
        elapsed = time.time() - t0
        print(
            f"\n  Batch complete: {n} images in {elapsed:.1f}s ({elapsed / n:.1f}s/image avg)"
        )


def run_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    inpainter: Inpainter,
    per_image: bool = False,
    sample_n: int | None = None,
    confidence: float = 0.25,
    memory_mb: int = _DEFAULT_MEMORY_MB,
    mask_output: Path | None = None,
) -> None:
    with _timed_batch(input_path, output_path) as (image_paths, _, out_dir):
        n = len(image_paths)

        if per_image:
            print("\n  Per-image mode: detecting and inpainting each image...")

            def _detect_and_inpaint(image: np.ndarray) -> tuple[np.ndarray, str]:
                mask = detector.detect(image)
                n_m, _, pct = mask_stats(mask)
                result = inpainter.inpaint(image, mask) if n_m > 0 else image
                return result, f"{pct:.1f}% masked"

            _process_images(image_paths, out_dir, _detect_and_inpaint)
            return

        shared_mask = collect_shared_mask(
            image_paths,
            detector,
            sample_n=sample_n,
            confidence=confidence,
            memory_mb=memory_mb,
        )
        if mask_output:
            save_image(shared_mask, mask_output)
            print(f"  Shared mask saved to {mask_output}")

        if mask_stats(shared_mask)[0] == 0:
            print("  No watermark detected in shared mask. Copying originals.")
            for i, p in enumerate(image_paths):
                shutil.copy2(p, out_dir / p.name)
                print(f"  [{i + 1}/{n}] {p.name} (copied)")
        else:
            print(f"\n  Inpainting {n} images with shared mask...")
            _process_images(
                image_paths,
                out_dir,
                lambda img: (inpainter.inpaint(img, shared_mask), ""),
            )


def inpaint_batch(
    input_path: Path,
    output_path: Path,
    mask: np.ndarray,
    inpainter: Inpainter,
) -> None:
    n_masked = mask_stats(mask)[0]
    if n_masked == 0:
        print("  Mask is empty — nothing to inpaint.")

    with _timed_batch(input_path, output_path) as (image_paths, _, out_dir):
        _process_images(
            image_paths,
            out_dir,
            lambda img: (
                (inpainter.inpaint(img, mask), "") if n_masked > 0 else (img, "")
            ),
        )
