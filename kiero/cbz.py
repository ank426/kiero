import shutil
import tempfile
import zipfile
from pathlib import Path


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
    image_paths = sorted(p for p in input_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in image_paths:
            zf.write(p, arcname=str(p.relative_to(input_dir)))
