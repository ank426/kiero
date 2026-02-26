def _is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in {".cbz", ".zip"}

def _is_cbz(path: Path) -> bool:
    return path.suffix.lower() in {".cbz", ".zip"}

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

def _require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        sys.exit(f"Error: {label} not found: {path}")

def _validate_run(input_path: Path, output_path: Path, mask_output: Path | None):
    if not (_is_cbz(input_path) or input_path.is_dir() or _is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")
    if _is_cbz(input_path) or input_path.is_dir():
        if _is_image(output_path):
            sys.exit(f"Error: Output must be a directory or .cbz/.zip when input is a batch: {output_path}")
    else:
        if not _is_image(output_path):
            sys.exit(f"Error: Output must be an image file when input is an image: {output_path}")
    if mask_output and not _is_image(mask_output):
        sys.exit(f"Error: Mask output must be an image file: {mask_output}")

def _validate_detect(input_path: Path, output_path: Path):
    if not (_is_cbz(input_path) or input_path.is_dir() or _is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")
    if not _is_image(output_path):
        sys.exit(f"Error: Mask output must be an image file: {output_path}")

def _validate_inpaint(input_path: Path, output_path: Path, mask: Path):
    if not (_is_cbz(input_path) or input_path.is_dir() or _is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")
    if not _is_image(mask):
        sys.exit(f"Error: Mask must be an image file: {mask}")
    if _is_cbz(input_path) or input_path.is_dir():
        if _is_image(output_path):
            sys.exit(f"Error: Output must be a directory or .cbz/.zip when input is a batch: {output_path}")
    else:
        if not _is_image(output_path):
            sys.exit(f"Error: Output must be an image file when input is an image: {output_path}")
