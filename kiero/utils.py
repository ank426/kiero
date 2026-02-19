from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def mask_stats(mask: np.ndarray) -> tuple[int, int, float]:
    n_masked = int(np.count_nonzero(mask))
    total = mask.shape[0] * mask.shape[1]
    return n_masked, total, (n_masked / total * 100 if total > 0 else 0.0)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    return sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )


def _read(path: str | Path, flags: int, label: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    data = cv2.imread(str(path), flags)
    if data is None:
        raise ValueError(f"Could not read {label.lower()}: {path}")
    return data


def load_image(path: str | Path) -> np.ndarray:
    return _read(path, cv2.IMREAD_COLOR, "Image")


def load_mask(path: str | Path) -> np.ndarray:
    return _read(path, cv2.IMREAD_GRAYSCALE, "Mask")


def save_image(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write image: {path}")
