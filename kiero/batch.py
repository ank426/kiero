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

    Raises:
        ValueError: If the archive contains entries with path traversal.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="kiero_cbz_"))
    with zipfile.ZipFile(cbz_path, "r") as zf:
        # Validate paths to prevent zip-slip attacks
        for member in zf.namelist():
            dest = (temp_dir / member).resolve()
            if not str(dest).startswith(str(temp_dir.resolve())):
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(
                    f"Zip slip detected in {cbz_path}: member '{member}' "
                    f"escapes extraction directory"
                )
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


def write_cbz(
    image_paths: list[Path], output_path: Path, root_dir: Path | None = None
) -> None:
    """Create a CBZ archive from a list of image files.

    Uses ZIP_STORED (no compression) since images are already compressed.

    Args:
        image_paths: List of image file paths to include.
        output_path: Path for the output .cbz file.
        root_dir: If provided, archive paths are relative to this directory,
            preserving subdirectory structure. Otherwise files are stored flat
            using just their filename.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in image_paths:
            if root_dir is not None:
                arcname = str(p.relative_to(root_dir))
            else:
                arcname = p.name
            zf.write(p, arcname=arcname)


_DEFAULT_LOAD_CHUNK = 20  # images loaded into RAM at a time


def collect_shared_mask(
    image_paths: list[Path],
    detector: WatermarkDetector,
    sample_n: int | None = None,
    threshold: float = 0.5,
    batch_size: int | None = None,
    load_chunk: int = _DEFAULT_LOAD_CHUNK,
) -> np.ndarray:
    """Build a shared mask by averaging detection across multiple images.

    Strategy:
    1. Optionally sample a subset of images.
    2. Load *load_chunk* images into RAM at a time, run ``detect_batch``
       (which internally chunks by the detector's GPU batch size), accumulate
       only the float32 mask sum.  Images are discarded after each chunk.
    3. Average the accumulated mask and threshold to binary.

    A pixel must be detected as watermark in >= (threshold * 100)% of sampled
    images to appear in the final mask. This filters out false positives that
    are inconsistent across images.

    Args:
        image_paths: All image paths (will be sampled from).
        detector: Detector instance to use.
        sample_n: Number of images to sample for averaging. None = use all.
        threshold: Fraction of samples that must agree for a pixel to be masked.
            Default 0.5 = detected in at least half the samples.
        batch_size: GPU batch size for detection. None = use detector default.
        load_chunk: Number of images to load into RAM at once (default 20).

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

    mask_sum: np.ndarray | None = None  # float32 (H, W) accumulator
    ref_shape: tuple[int, int] | None = None
    det_t0 = time.time()

    for chunk_start in range(0, n, load_chunk):
        chunk_paths = sampled[chunk_start : chunk_start + load_chunk]

        # Load this chunk into memory
        images = [load_image(p) for p in chunk_paths]

        # Verify all images in this chunk have the same dimensions
        shapes = {img.shape[:2] for img in images}
        if ref_shape is None:
            if len(shapes) > 1:
                raise ValueError(
                    f"Cannot compute shared mask: images have different "
                    f"dimensions: {shapes}. Use --per-image mode instead."
                )
            ref_shape = next(iter(shapes))
        else:
            for s in shapes:
                if s != ref_shape:
                    raise ValueError(
                        f"Cannot compute shared mask: images have different "
                        f"dimensions: {ref_shape} vs {s}. Use --per-image "
                        f"mode instead."
                    )

        # Run batched detection — detect_batch handles GPU chunking internally
        masks = detector.detect_batch(images, batch_size=batch_size)

        # Accumulate mask values
        for m in masks:
            m_f = m.astype(np.float32) / 255.0
            if mask_sum is None:
                mask_sum = m_f
            else:
                mask_sum += m_f

        chunk_end = min(chunk_start + load_chunk, n)
        print(f"  Processed {chunk_end}/{n} images...")

    det_time = time.time() - det_t0
    print(f"  Detection done in {det_time:.1f}s ({det_time / n:.2f}s/image)")

    # Average and threshold
    if mask_sum is None:
        raise ValueError("No images to process")
    mask_avg = mask_sum / n  # (H, W), values 0.0-1.0
    shared_mask = (mask_avg >= threshold).astype(np.uint8) * 255

    n_masked, _, pct = mask_stats(shared_mask)
    print(f"  Shared mask: {pct:.1f}% of image masked")

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

    out_dir: Path | None = None
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

            n_masked, _, _ = mask_stats(shared_mask)
            if n_masked == 0:
                print("  No watermark detected in shared mask. Copying originals.")
                for i, p in enumerate(image_paths):
                    out_p = out_dir / p.name
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
                n_masked, _, pct = mask_stats(mask)

                if n_masked > 0:
                    result = inpainter.inpaint(image, mask)
                else:
                    result = image

                out_p = out_dir / p.name
                save_image(result, out_p)
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{n}] {p.name} ({elapsed:.1f}s, {pct:.1f}% masked)")

        # --- Write output ---
        if source_type == "cbz":
            output_images = sorted(p for p in out_dir.rglob("*") if p.is_file())
            write_cbz(output_images, Path(output_path), root_dir=out_dir)
            print(f"\n  CBZ written to {output_path}")

        total_elapsed = time.time() - total_t0
        print(
            f"\n  Batch complete: {n} images in {total_elapsed:.1f}s "
            f"({total_elapsed / n:.1f}s/image avg)"
        )

    finally:
        # Clean up temp CBZ extraction dir
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        # Clean up temp output dir (only if CBZ — out_dir is a temp dir)
        if source_type == "cbz" and out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)
