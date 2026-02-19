from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def mask_ratio(mask: np.ndarray) -> float:
    return np.count_nonzero(mask) / mask.size


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name
    )


def _read(path: Path, flags: int, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if (data := cv2.imread(str(path), flags)) is None:
        raise ValueError(f"Could not read {label.lower()}: {path}")
    return data


def load_image(path: Path) -> np.ndarray:
    return _read(path, cv2.IMREAD_COLOR, "Image")


def load_mask(path: Path) -> np.ndarray:
    return _read(path, cv2.IMREAD_GRAYSCALE, "Mask")


def save_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")
