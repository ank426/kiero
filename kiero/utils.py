import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np


# --- Image I/O & Mask Utilities ---


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(path.suffix, img)
    if not success:
        raise ValueError(f"Failed to encode image: {path}")
    buffer.tofile(str(path))


def load_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def mask_ratio(mask: np.ndarray) -> float:
    return np.count_nonzero(mask) / mask.size


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0:
        return mask.astype(np.uint8)
    m = np.asarray(mask, dtype=np.float32)
    mx = float(m.max())
    thr = 0.5 if mx <= 1.0 else 128.0
    return (m >= thr).astype(np.uint8) * 255


# --- Archive Handling (CBZ/ZIP) ---


def extract_cbz(input_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="kiero_cbz_"))
    with zipfile.ZipFile(input_path, "r") as zf:
        for member in zf.namelist():
            if not str((tmp / member).resolve()).startswith(str(tmp.resolve())):
                shutil.rmtree(tmp, ignore_errors=True)
                raise ValueError(f"Zip slip detected in {input_path}: member '{member}' escapes extraction directory")
        zf.extractall(tmp)
    return tmp


def write_cbz(input_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and is_image(p))
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in image_paths:
            zf.write(p, arcname=str(p.relative_to(input_dir)))


# --- Path & Type Validation ---


def is_cbz(path: Path) -> bool:
    return path.suffix.lower() in {".cbz", ".zip"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        raise ValueError(f"{label} not found: {path}")


def validate(
    input_path: Path,
    output_path: Path | None = None,
    mask_input: Path | None = None,
    mask_output: Path | None = None,
):
    require_exists(input_path, "Input")
    if not (is_cbz(input_path) or input_path.is_dir() or is_image(input_path)):
        raise ValueError(f"Input must be a directory, .cbz/.zip or an image file: {input_path}")

    if output_path:
        if is_cbz(input_path) or input_path.is_dir():
            if is_image(output_path):
                raise ValueError(f"Output must be a directory or .cbz/.zip when input is a batch: {output_path}")
        else:
            if not is_image(output_path):
                raise ValueError(f"Output must be an image file when input is an image: {output_path}")

    if mask_input:
        require_exists(mask_input, "Mask file")
        if not is_image(mask_input):
            raise ValueError(f"Mask must be an image file: {mask_input}")

    if mask_output and not is_image(mask_output):
        raise ValueError(f"Mask output must be an image file: {mask_output}")


# --- Lazy Component Factories ---


def make_detector(confidence: float, padding: int, device: str | None):
    from kiero.detectors.yolo import YoloDetector

    return YoloDetector(confidence=confidence, padding=padding, device=device)


def make_inpainter(device: str | None):
    from kiero.inpainters.lama import LamaInpainter

    return LamaInpainter(device=device)


def make_pipeline(confidence: float, padding: int, device: str | None):
    from kiero.pipeline import Pipeline

    return Pipeline(confidence=confidence, padding=padding, device=device)
