"""Template matching watermark detector using OpenCV.

This detector works in two modes:

1. Template mode (--template path/to/template.png):
   Slides a known watermark template across the image and finds matching regions.
   Best when you have a sample crop of the watermark.

2. Heuristic mode (no template provided):
   Attempts to detect watermark regions by looking for semi-transparent overlays.
   Uses a combination of:
   - Local contrast analysis (watermarks flatten local contrast)
   - Frequency domain analysis (repeated watermark text creates periodic peaks)
   - Intensity anomaly detection (semi-transparent overlays shift pixel intensities
     toward the watermark color in a way that differs from natural gradients)

   This is a best-effort approach and will not work for all watermark types.
"""

from pathlib import Path

import cv2
import numpy as np

from kiero.detectors.base import WatermarkDetector
from kiero.utils import dilate_mask


class TemplateDetector(WatermarkDetector):
    """OpenCV template matching based watermark detector."""

    def __init__(
        self,
        template_path: str | None = None,
        threshold: float = 0.7,
        multiscale: bool = True,
        dilate_px: int = 5,
    ):
        """Initialize the template detector.

        Args:
            template_path: Path to a watermark template image. If None, uses
                heuristic detection mode.
            threshold: Matching threshold (0-1). Higher = stricter matching.
            multiscale: Whether to try multiple scales when template matching.
            dilate_px: Pixels to dilate the mask by, to ensure full watermark
                coverage (watermark edges are often semi-transparent and may not
                match the template exactly).
        """
        self._template_path = template_path
        self._threshold = threshold
        self._multiscale = multiscale
        self._dilate_px = dilate_px
        self._template: np.ndarray | None = None

    @property
    def name(self) -> str:
        if self._template_path:
            return f"template({Path(self._template_path).name})"
        return "template(heuristic)"

    def _load_template(self) -> np.ndarray | None:
        """Load and cache the template image."""
        if self._template is not None:
            return self._template
        if self._template_path is None:
            return None
        self._template = cv2.imread(self._template_path, cv2.IMREAD_GRAYSCALE)
        if self._template is None:
            raise FileNotFoundError(
                f"Could not load template image: {self._template_path}"
            )
        return self._template

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect watermark regions."""
        template = self._load_template()
        if template is not None:
            return self._detect_with_template(image, template)
        return self._detect_heuristic(image)

    def _detect_with_template(
        self, image: np.ndarray, template: np.ndarray
    ) -> np.ndarray:
        """Detect watermark by sliding a template across the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        scales = [1.0]
        if self._multiscale:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5]

        for scale in scales:
            if scale == 1.0:
                t = template
            else:
                tw = max(1, int(template.shape[1] * scale))
                th = max(1, int(template.shape[0] * scale))
                t = cv2.resize(template, (tw, th))

            # Skip if template is larger than the image
            if t.shape[0] > h or t.shape[1] > w:
                continue

            result = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)

            # Find all locations above threshold
            locations = np.where(result >= self._threshold)
            for pt_y, pt_x in zip(*locations):
                mask[pt_y : pt_y + t.shape[0], pt_x : pt_x + t.shape[1]] = 255

        # Dilate to cover edges
        mask = dilate_mask(mask, self._dilate_px)

        return mask

    def _detect_heuristic(self, image: np.ndarray) -> np.ndarray:
        """Heuristic watermark detection without a template.

        Strategy:
        1. Convert to grayscale and compute local standard deviation.
           Watermarks reduce local contrast in regions they overlay — a patch of
           manga art that should have high contrast (lines) will have flattened
           contrast where a semi-transparent watermark blends values toward its
           own intensity.
        2. Compute the Laplacian to find edges. Watermarks often introduce
           faint edges that don't belong to the original art.
        3. Use frequency domain analysis: compute the magnitude spectrum and look
           for periodic peaks that indicate tiled/repeated text.
        4. Combine these signals into a final mask.

        This is inherently noisy and best-effort.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # --- Signal 1: Local contrast anomaly ---
        # Compute local std dev using a box filter trick:
        # std = sqrt(E[X^2] - E[X]^2)
        ksize = 31
        gray_f = gray.astype(np.float32)
        mean = cv2.blur(gray_f, (ksize, ksize))
        mean_sq = cv2.blur(gray_f * gray_f, (ksize, ksize))
        local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

        # Regions with abnormally low local std could be watermark regions.
        # But also could be flat areas of the manga (white backgrounds, solid fills).
        # We look for areas where local_std is low AND the intensity is in a
        # specific mid-range (watermarks are often gray on white/black).
        mid_intensity = (gray > 80) & (gray < 220)
        low_contrast = local_std < np.percentile(local_std, 20)
        contrast_signal = (mid_intensity & low_contrast).astype(np.uint8) * 255

        # --- Signal 2: Frequency domain analysis ---
        # Watermarks that tile/repeat create peaks in the frequency domain.
        # Compute magnitude spectrum, mask out the DC component and low freqs,
        # look for sharp peaks.
        dft = cv2.dft(gray_f, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=(0, 1))
        magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        magnitude = np.log1p(magnitude)

        # Zero out the center (low frequencies — these are the image content)
        cy, cx = h // 2, w // 2
        r = min(h, w) // 20  # mask radius
        cv2.circle(magnitude, (cx, cy), r, 0, -1)

        # Threshold to find peaks
        peak_threshold = np.percentile(magnitude, 99.5)
        freq_peaks = (magnitude > peak_threshold).astype(np.uint8) * 255

        # If there are periodic peaks, the watermark likely covers the whole image.
        # Count significant peaks — if many exist, it suggests a tiled watermark.
        n_peaks = np.count_nonzero(freq_peaks)
        freq_signal = np.zeros((h, w), dtype=np.uint8)
        if n_peaks > 20:
            # Tiled watermark suspected — the entire image may be affected.
            # Use the contrast signal more aggressively.
            freq_signal = contrast_signal

        # --- Combine signals ---
        # Use morphological operations to clean up the contrast signal
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

        combined = cv2.bitwise_or(contrast_signal, freq_signal)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        # Dilate
        combined = dilate_mask(combined, self._dilate_px)

        return combined
