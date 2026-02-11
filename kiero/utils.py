"""Utility functions for image I/O, mask visualization, and comparison output."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def mask_stats(mask: np.ndarray) -> tuple[int, int, float]:
    """Compute basic statistics about a binary mask.

    Args:
        mask: Binary mask, shape (H, W), dtype uint8, values 0 or 255.

    Returns:
        Tuple of (n_masked, total, percentage).
        - n_masked: number of non-zero pixels.
        - total: total number of pixels (H * W).
        - percentage: n_masked / total * 100.
    """
    n_masked = int(np.count_nonzero(mask))
    total = mask.shape[0] * mask.shape[1]
    pct = n_masked / total * 100 if total > 0 else 0.0
    return n_masked, total, pct


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a binary mask by a given number of pixels.

    Uses an elliptical structuring element for smooth expansion.

    Args:
        mask: Binary mask, shape (H, W), dtype uint8.
        px: Number of pixels to dilate by. If <= 0, mask is returned unchanged.

    Returns:
        Dilated mask, same shape and dtype.
    """
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to an RGB PIL Image.

    Args:
        image: Image array, shape (H, W, 3), dtype uint8, BGR channel order.

    Returns:
        PIL Image in RGB mode.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def list_images(directory: str | Path) -> list[Path]:
    """List image files in a directory, sorted by name.

    Args:
        directory: Path to directory.

    Returns:
        Sorted list of image file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    images = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=lambda p: p.name)
    return images


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Path to the image file.

    Returns:
        Image as numpy array, shape (H, W, 3), dtype uint8, BGR format.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as an image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an image to disk.

    Args:
        image: Image array (BGR or grayscale).
        path: Output path. Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask from disk.

    Args:
        path: Path to the mask image (should be grayscale or will be converted).

    Returns:
        Mask as numpy array, shape (H, W), dtype uint8.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask


def conform_mask(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Ensure a mask has the same spatial dimensions as an image.

    If the mask already matches, it is returned as-is (no copy). If dimensions
    differ, the mask is resized to match the image using nearest-neighbor
    interpolation (to keep it binary) and a warning is printed.

    Args:
        mask: Binary mask, shape (H_m, W_m), dtype uint8.
        image: Target image, shape (H, W, 3), dtype uint8.

    Returns:
        Mask with shape (H, W), dtype uint8.
    """
    img_h, img_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    if (mask_h, mask_w) == (img_h, img_w):
        return mask
    print(
        f"  Warning: mask size ({mask_w}x{mask_h}) differs from image "
        f"({img_w}x{img_h}), resizing mask to match."
    )
    return cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.4
) -> np.ndarray:
    """Overlay a mask on an image for visualization.

    Watermark regions (mask=255) are tinted with the given color.

    Args:
        image: Input image, shape (H, W, 3), BGR.
        mask: Binary mask, shape (H, W), uint8.
        color: BGR color to tint the masked regions.
        alpha: Opacity of the tint overlay.

    Returns:
        Visualization image with mask overlay.
    """
    mask = conform_mask(mask, image)
    vis = image.copy()
    mask_bool = mask > 127
    overlay = np.full_like(image, color, dtype=np.uint8)
    vis[mask_bool] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask_bool]
    return vis


def make_comparison(
    original: np.ndarray,
    results: list[tuple[str, np.ndarray, np.ndarray]],
    max_width: int = 800,
) -> np.ndarray:
    """Create a side-by-side comparison image.

    Produces a grid: [original | mask1 | result1 | mask2 | result2 | ...]

    Args:
        original: Original input image.
        results: List of (detector_name, mask, inpainted_result) tuples.
        max_width: Maximum width for each panel. Images are scaled down if wider.

    Returns:
        Comparison image as numpy array.
    """
    h, w = original.shape[:2]

    # Compute scale factor to fit panels in max_width
    scale = min(1.0, max_width / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    panels = []

    # Original panel with label
    orig_resized = cv2.resize(original, (new_w, new_h))
    orig_labeled = _add_label(orig_resized, "Original")
    panels.append(orig_labeled)

    for det_name, mask, inpainted in results:
        # Mask panel — convert to 3-channel for display
        mask_vis = overlay_mask(original, mask)
        mask_resized = cv2.resize(mask_vis, (new_w, new_h))
        mask_labeled = _add_label(mask_resized, f"Mask: {det_name}")
        panels.append(mask_labeled)

        # Inpainted result panel
        result_resized = cv2.resize(inpainted, (new_w, new_h))
        result_labeled = _add_label(result_resized, f"Result: {det_name}")
        panels.append(result_labeled)

    # Stack horizontally if 3 or fewer panels, otherwise in a grid
    if len(panels) <= 3:
        return np.hstack(panels)
    else:
        # Arrange in rows of 3
        rows = []
        for i in range(0, len(panels), 3):
            row_panels = panels[i : i + 3]
            # Pad the last row if needed
            while len(row_panels) < 3:
                row_panels.append(np.zeros_like(panels[0]))
            rows.append(np.hstack(row_panels))
        return np.vstack(rows)


def _add_label(image: np.ndarray, label: str) -> np.ndarray:
    """Add a text label to the top of an image."""
    h, w = image.shape[:2]
    label_height = 30
    labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    labeled[label_height:, :] = image

    # Draw label background
    labeled[:label_height, :] = (40, 40, 40)

    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (label_height + text_size[1]) // 2
    cv2.putText(
        labeled, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness
    )
    return labeled
