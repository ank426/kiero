def _run(
    input_path: Path,
    output_path: Path,
    per_image: bool,
    sample: int | None,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
    mask_output: Path | None,
):
    _require_exists(input_path)
    _validate_run(input_path, output_path, mask_output)

    if _is_cbz(input_path):
