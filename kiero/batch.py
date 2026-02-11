"""Batch processing for directories and CBZ archives.

Supports two modes:
- Shared mask (default): detect on a sample of images, average the masks,
  threshold to binary, then apply that single mask to all images for inpainting.
  This is appropriate when all images share the same watermark pattern.
- Per-image: detect and inpaint each image independently.

CBZ files are ZIP archives of images. Input CBZ -> output CBZ. Input dir -> output dir.
"""

import random
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from kiero.detectors.base import WatermarkDetector
from kiero.inpainters.base import Inpainter
from kiero.utils import IMAGE_EXTENSIONS, list_images, load_image, save_image


def resolve_inputs(input_path: Path) -> tuple[list[Path], str, Path | None]:
    """Determine input type and return a list of image paths.

    Args:
        input_path: Path to an image file, directory, or .cbz file.

    Returns:
        Tuple of (image_paths, source_type, temp_dir).
        - image_paths: sorted list of image file paths to process.
        - source_type: "file", "directory", or "cbz".
        - temp_dir: path to temp directory if CBZ was extracted (caller must
          clean up), or None otherwise.
    """
    if input_path.is_dir():
        images = list_images(input_path)
        if not images:
            raise FileNotFoundError(f"No image files found in {input_path}")
        return images, "directory", None

    if input_path.suffix.lower() == ".cbz":
        return _extract_cbz(input_path)

    # Single file
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    return [input_path], "file", None


def _extract_cbz(cbz_path: Path) -> tuple[list[Path], str, Path]:
    """Extract a CBZ archive to a temp directory.

    Returns:
        Tuple of (image_paths, "cbz", temp_dir_path).
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="kiero_cbz_"))
    with zipfile.ZipFile(cbz_path, "r") as zf:
        zf.extractall(temp_dir)

    # Collect image files (may be in subdirectories)
    images = []
    for p in sorted(temp_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(p)

    if not images:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError(f"No image files found in CBZ: {cbz_path}")

    return images, "cbz", temp_dir


def write_cbz(image_paths: list[Path], output_path: Path) -> None:
    """Create a CBZ archive from a list of image files.

    Uses ZIP_STORED (no compression) since images are already compressed.
    Files are stored with just their filename (no directory prefix).

    Args:
        image_paths: List of image file paths to include.
        output_path: Path for the output .cbz file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in image_paths:
            zf.write(p, arcname=p.name)


def collect_shared_mask(
    image_paths: list[Path],
    detector: WatermarkDetector,
    sample_n: int | None = None,
    threshold: float = 0.5,
    batch_size: int | None = None,
) -> np.ndarray:
    """Build a shared mask by averaging detection across multiple images.

    Strategy:
    1. Optionally sample a subset of images.
    2. Run detector on all sampled images (batched if detector supports it).
    3. Accumulate masks as float32, average, threshold to binary.

    A pixel must be detected as watermark in >= (threshold * 100)% of sampled
    images to appear in the final mask. This filters out false positives that
    are inconsistent across images.

    Args:
        image_paths: All image paths (will be sampled from).
        detector: Detector instance to use.
        sample_n: Number of images to sample for averaging. None = use all.
        threshold: Fraction of samples that must agree for a pixel to be masked.
            Default 0.5 = detected in at least half the samples.
        batch_size: Batch size for batched detection. None = use detector default.

    Returns:
        Binary mask, shape (H, W), dtype uint8, values 0 or 255.
    """
    # Sample images
    if sample_n is not None and sample_n < len(image_paths):
        sampled = random.sample(image_paths, sample_n)
        sampled.sort(key=lambda p: p.name)  # deterministic order
    else:
        sampled = image_paths
        sample_n = len(sampled)

    n = len(sampled)
    print(f"  Computing shared mask from {n} images...")

    # Load all sampled images into memory
    t0 = time.time()
    images = [load_image(p) for p in sampled]
    load_time = time.time() - t0
    print(f"  Loaded {n} images in {load_time:.1f}s")

    # Verify all images have the same dimensions
    shapes = {img.shape[:2] for img in images}
    if len(shapes) > 1:
        raise ValueError(
            f"Cannot compute shared mask: images have different dimensions: "
            f"{shapes}. Use --per-image mode instead."
        )

    # Run batched detection
    t0 = time.time()
    masks = detector.detect_batch(images, batch_size=batch_size)
    det_time = time.time() - t0
    print(f"  Batched detection done in {det_time:.1f}s ({det_time / n:.2f}s/image)")

    # Average masks using numpy vectorization
    # Stack all masks into a single (N, H, W) float32 array and compute mean
    # along axis 0 — this is a single vectorized operation, no Python loops.
    t0 = time.time()
    mask_stack = np.stack(masks, axis=0).astype(np.float32) / 255.0  # (N, H, W)
    mask_avg = mask_stack.mean(axis=0)  # (H, W), values 0.0-1.0
    shared_mask = (mask_avg >= threshold).astype(np.uint8) * 255
    avg_time = time.time() - t0

    n_pixels = np.count_nonzero(shared_mask)
    total_pixels = shared_mask.shape[0] * shared_mask.shape[1]
    pct = n_pixels / total_pixels * 100
    print(f"  Mask averaging done in {avg_time:.2f}s — {pct:.1f}% of image masked")

    return shared_mask


def run_batch(
    input_path: Path,
    output_path: Path,
    detector: WatermarkDetector,
    inpainter: Inpainter,
    per_image: bool = False,
    sample_n: int | None = None,
    mask_threshold: float = 0.5,
    batch_size: int | None = None,
    mask_output: Path | None = None,
) -> None:
    """Run the full batch pipeline on a directory or CBZ.

    Args:
        input_path: Path to input directory or CBZ file.
        output_path: Path for output directory or CBZ file.
        detector: Detector instance.
        inpainter: Inpainter instance.
        per_image: If True, detect independently per image instead of averaging.
        sample_n: Number of images to sample for shared mask averaging.
        mask_threshold: Threshold for shared mask averaging (0-1).
        batch_size: Batch size for batched detection.
        mask_output: If set, save the shared mask to this path.
    """
    total_t0 = time.time()

    # Resolve inputs
    image_paths, source_type, temp_dir = resolve_inputs(input_path)
    n = len(image_paths)
    print(f"  Source: {source_type} ({n} images)")

    try:
        # Determine output directory for cleaned images
        if source_type == "cbz":
            # Work in a temp output dir, then pack into CBZ
            out_dir = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        else:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)

        # --- Shared mask mode ---
        if not per_image:
            shared_mask = collect_shared_mask(
                image_paths,
                detector,
                sample_n=sample_n,
                threshold=mask_threshold,
                batch_size=batch_size,
            )

            if mask_output:
                save_image(shared_mask, mask_output)
                print(f"  Shared mask saved to {mask_output}")

            n_masked = np.count_nonzero(shared_mask)
            if n_masked == 0:
                print("  No watermark detected in shared mask. Copying originals.")
                for i, p in enumerate(image_paths):
                    out_p = out_dir / p.name
                    if source_type == "directory":
                        shutil.copy2(p, out_p)
                    else:
                        shutil.copy2(p, out_p)
                    print(f"  [{i + 1}/{n}] {p.name} (copied)")
            else:
                # Inpaint each image with the shared mask
                print(f"\n  Inpainting {n} images with shared mask...")
                for i, p in enumerate(image_paths):
                    t0 = time.time()
                    image = load_image(p)
                    result = inpainter.inpaint(image, shared_mask)
                    out_p = out_dir / p.name
                    save_image(result, out_p)
                    elapsed = time.time() - t0
                    print(f"  [{i + 1}/{n}] {p.name} ({elapsed:.1f}s)")

        # --- Per-image mode ---
        else:
            print(f"\n  Per-image mode: detecting and inpainting each image...")
            for i, p in enumerate(image_paths):
                t0 = time.time()
                image = load_image(p)

                mask = detector.detect(image)
                n_masked = np.count_nonzero(mask)

                if n_masked > 0:
                    result = inpainter.inpaint(image, mask)
                else:
                    result = image

                out_p = out_dir / p.name
                save_image(result, out_p)
                elapsed = time.time() - t0
                pct = n_masked / (mask.shape[0] * mask.shape[1]) * 100
                print(f"  [{i + 1}/{n}] {p.name} ({elapsed:.1f}s, {pct:.1f}% masked)")

        # --- Write output ---
        if source_type == "cbz":
            output_images = sorted(out_dir.iterdir(), key=lambda p: p.name)
            write_cbz(output_images, Path(output_path))
            print(f"\n  CBZ written to {output_path}")
            # Clean up temp output dir
            shutil.rmtree(out_dir, ignore_errors=True)

        total_elapsed = time.time() - total_t0
        print(
            f"\n  Batch complete: {n} images in {total_elapsed:.1f}s "
            f"({total_elapsed / n:.1f}s/image avg)"
        )

    finally:
        # Clean up temp CBZ extraction dir
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
