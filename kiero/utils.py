"""Utility functions for image I/O and mask statistics."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def mask_stats(mask: np.ndarray) -> tuple[int, int, float]:
    """Return (n_masked, total, percentage) for a binary mask."""
    n_masked = int(np.count_nonzero(mask))
    total = mask.shape[0] * mask.shape[1]
    pct = n_masked / total * 100 if total > 0 else 0.0
    return n_masked, total, pct


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to an RGB PIL Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def list_images(directory: str | Path) -> list[Path]:
    """List image files in a directory, sorted by name."""
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
    """Load a BGR image from disk. Raises on missing/unreadable files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save an image to disk, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")


def load_mask(path: str | Path) -> np.ndarray:
    """Load a grayscale mask from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask
