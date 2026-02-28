def validate_command(
    command: str,
    *,
    input_path: Path,
    output_path: Path,
    mask: Path | None = None,
    mask_output: Path | None = None,
):
    require_exists(input_path, "Input")

    if command == "inpaint":
        if mask is None:
            raise ValueError("Mask must be provided for inpaint command")
        require_exists(mask, "Mask file")

    if not (is_cbz(input_path) or input_path.is_dir() or is_image(input_path)):
        raise ValueError(f"Input must be a directory, .cbz/.zip, or an image file: {input_path}")

    # Output validations
    if is_batch(input_path):
        if is_image(output_path):
            raise ValueError(f"Output must be a directory or .cbz/.zip when input is a batch: {output_path}")
    else:  # input is image
        if command == "detect":
            pass # Anything is fine for output, it's just the mask
        elif not is_image(output_path):
            raise ValueError(f"Output must be an image file when input is an image: {output_path}")

    # Optional paths
    if mask is not None and not is_image(mask):
        raise ValueError(f"Mask must be an image file: {mask}")

    if mask_output is not None and not is_image(mask_output):
        raise ValueError(f"Mask output must be an image file: {mask_output}")
