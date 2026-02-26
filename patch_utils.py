import sys
from pathlib import Path

def is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in {".cbz", ".zip"}

def is_cbz(path: Path) -> bool:
    return path.suffix.lower() in {".cbz", ".zip"}

def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

def require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        sys.exit(f"Error: {label} not found: {path}")

def validate_run(input_path: Path, output_path: Path, mask_output: Path | None):
    require_exists(input_path)
    if not (is_cbz(input_path) or input_path.is_dir() or is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")
    if is_cbz(input_path) or input_path.is_dir():
        if is_image(output_path):
            sys.exit(f"Error: Output must be a directory or .cbz/.zip when input is a batch: {output_path}")
    else:
        if not is_image(output_path):
            sys.exit(f"Error: Output must be an image file when input is an image: {output_path}")
    if mask_output and not is_image(mask_output):
        sys.exit(f"Error: Mask output must be an image file: {mask_output}")

def validate_detect(input_path: Path, output_path: Path):
    require_exists(input_path)
    if not (is_cbz(input_path) or input_path.is_dir() or is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")
    if not is_image(output_path):
        sys.exit(f"Error: Mask output must be an image file: {output_path}")

def validate_inpaint(input_path: Path, output_path: Path, mask: Path):
    require_exists(input_path, "Input")
    require_exists(mask, "Mask file")
    if not (is_cbz(input_path) or input_path.is_dir() or is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")
    if not is_image(mask):
        sys.exit(f"Error: Mask must be an image file: {mask}")
    if is_cbz(input_path) or input_path.is_dir():
        if is_image(output_path):
            sys.exit(f"Error: Output must be a directory or .cbz/.zip when input is a batch: {output_path}")
    else:
        if not is_image(output_path):
            sys.exit(f"Error: Output must be an image file when input is an image: {output_path}")

def make_detector(confidence: float, padding: int, device: str | None):
    from kiero.detectors.yolo import YoloDetector
    return YoloDetector(confidence=confidence, padding=padding, device=device)

def make_inpainter(device: str | None):
    from kiero.inpainters.lama import LamaInpainter
    return LamaInpainter(device=device)

def make_pipeline(confidence: float, padding: int, device: str | None):
    from kiero.pipeline import Pipeline
    return Pipeline(confidence=confidence, padding=padding, device=device)
