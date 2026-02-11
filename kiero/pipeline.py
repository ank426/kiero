"""Pipeline orchestration — connects detectors to inpainters."""

import time
from pathlib import Path

import numpy as np

from kiero.detectors import get_detector, DETECTORS
from kiero.detectors.base import WatermarkDetector
from kiero.inpainters import get_inpainter
from kiero.inpainters.base import Inpainter
from kiero.utils import load_image, save_image, make_comparison, mask_stats


class Pipeline:
    """Orchestrates watermark detection and inpainting.

    Usage:
        pipeline = Pipeline(detector="yolo11x", inpainter="lama")
        result = pipeline.run("input.png", "output.png")

        # Accept pre-built instances:
        pipeline = Pipeline(detector=my_detector, inpainter=my_inpainter)

        # Detect-only (no inpainter needed):
        pipeline = Pipeline(detector="yolo11x", inpainter=None)
        mask = pipeline.detect("input.png")
    """

    def __init__(
        self,
        detector: str | WatermarkDetector = "yolo11x",
        inpainter: str | Inpainter | None = "lama",
        detector_kwargs: dict | None = None,
        inpainter_kwargs: dict | None = None,
    ):
        if isinstance(detector, WatermarkDetector):
            self._detector = detector
        else:
            self._detector = get_detector(detector, **(detector_kwargs or {}))

        if inpainter is None:
            self._inpainter = None
        elif isinstance(inpainter, Inpainter):
            self._inpainter = inpainter
        else:
            self._inpainter = get_inpainter(inpainter, **(inpainter_kwargs or {}))

    # --- Internal helpers that operate on loaded arrays ---

    def _run_detect(self, image: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        """Run detection on a loaded image.

        Returns:
            Tuple of (mask, n_masked, pct, elapsed_seconds).
        """
        print(f"  Detecting with {self._detector.name}...")
        t0 = time.time()
        mask = self._detector.detect(image)
        elapsed = time.time() - t0
        n_masked, _, pct = mask_stats(mask)
        print(f"  Detection done in {elapsed:.1f}s — {pct:.1f}% of image masked")
        return mask, n_masked, pct, elapsed

    def _run_inpaint(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Run inpainting on a loaded image with a mask.

        Returns:
            Tuple of (result, elapsed_seconds).

        Raises:
            RuntimeError: If no inpainter is configured.
        """
        if self._inpainter is None:
            raise RuntimeError(
                "No inpainter configured. Pass an inpainter to Pipeline() "
                "or use detect() instead."
            )
        print(f"  Inpainting with {self._inpainter.name}...")
        t0 = time.time()
        result = self._inpainter.inpaint(image, mask)
        elapsed = time.time() - t0
        print(f"  Inpainting done in {elapsed:.1f}s")
        return result, elapsed

    # --- Public API ---

    def detect(self, image_path: str | Path) -> np.ndarray:
        """Run detection only, return the mask.

        Args:
            image_path: Path to input image.

        Returns:
            Binary mask array.
        """
        image = load_image(image_path)
        mask, _, _, _ = self._run_detect(image)
        return mask

    def inpaint(self, image_path: str | Path, mask: np.ndarray) -> np.ndarray:
        """Run inpainting with a provided mask.

        Args:
            image_path: Path to input image.
            mask: Binary mask array.

        Returns:
            Inpainted image array.
        """
        image = load_image(image_path)
        result, _ = self._run_inpaint(image, mask)
        return result

    def run(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
        mask_path: str | Path | None = None,
    ) -> np.ndarray:
        """Run the full pipeline: detect then inpaint.

        Args:
            image_path: Path to input image.
            output_path: Path to save the result. If None, result is not saved.
            mask_path: Path to save the mask. If None, mask is not saved.

        Returns:
            Inpainted image array.
        """
        image = load_image(image_path)

        # Detect
        mask, n_masked, _, det_time = self._run_detect(image)

        if mask_path:
            save_image(mask, mask_path)
            print(f"  Mask saved to {mask_path}")

        # Skip inpainting if no watermark detected
        if n_masked == 0:
            print("  No watermark detected, skipping inpainting.")
            if output_path:
                save_image(image, output_path)
            return image

        # Inpaint
        result, inp_time = self._run_inpaint(image, mask)
        print(f"  Total: {det_time + inp_time:.1f}s")

        if output_path:
            save_image(result, output_path)
            print(f"  Result saved to {output_path}")

        return result


def compare(
    image_path: str | Path,
    output_dir: str | Path,
    inpainter: str = "lama",
    inpainter_kwargs: dict | None = None,
    detector_kwargs: dict[str, dict] | None = None,
) -> None:
    """Run all detectors on an image and produce a comparison.

    Args:
        image_path: Path to input image.
        output_dir: Directory to save results.
        inpainter: Inpainter name to use for all results.
        inpainter_kwargs: Extra kwargs for the inpainter.
        detector_kwargs: Per-detector kwargs, keyed by detector name.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_kwargs = detector_kwargs or {}
    image = load_image(image_path)
    inp = get_inpainter(inpainter, **(inpainter_kwargs or {}))

    results = []

    for det_name in DETECTORS:
        print(f"\n--- {det_name} ---")
        try:
            kwargs = detector_kwargs.get(det_name, {})
            detector = get_detector(det_name, **kwargs)

            # Detect
            t0 = time.time()
            mask = detector.detect(image)
            det_time = time.time() - t0
            n_masked, _, pct = mask_stats(mask)
            print(f"  Detection: {det_time:.1f}s — {pct:.1f}% masked")

            # Save individual mask
            save_image(mask, output_dir / f"mask_{det_name}.png")

            # Inpaint
            if n_masked > 0:
                t0 = time.time()
                result = inp.inpaint(image, mask)
                inp_time = time.time() - t0
                print(f"  Inpainting: {inp_time:.1f}s")
            else:
                result = image.copy()
                print("  No watermark detected, skipping inpainting.")

            # Save individual result
            save_image(result, output_dir / f"result_{det_name}.png")

            results.append((det_name, mask, result))
        except Exception as exc:
            print(f"  ERROR: {det_name} failed: {exc}")
            print(f"  Skipping {det_name}.")

    # Save comparison grid
    print("\nGenerating comparison grid...")
    comparison = make_comparison(image, results)
    save_image(comparison, output_dir / "comparison.png")
    print(f"Comparison saved to {output_dir / 'comparison.png'}")
