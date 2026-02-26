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

    if _is_cbz(input_path):
        import shutil

        from kiero.cbz import extract_cbz

        print(f"Input Archive: {input_path}")
        tmp_in = extract_cbz(input_path)
        try:
            _run(
                input_path=tmp_in,
                output_path=output_path,
                per_image=per_image,
                sample=sample,
                confidence=confidence,
                padding=padding,
                memory=memory,
                device=device,
                mask_output=mask_output,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
        return

    if input_path.is_dir() and _is_cbz(output_path):
        import shutil
        import tempfile

        from kiero.cbz import write_cbz

        print(f"Output Archive: {output_path}")
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        try:
            _run(
                input_path=input_path,
                output_path=tmp_out,
                per_image=per_image,
                sample=sample,
                confidence=confidence,
                padding=padding,
                memory=memory,
                device=device,
                mask_output=mask_output,
            )
            write_cbz(tmp_out, output_path)
            print(f"\n  Archive written to {output_path}")
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
        return

    if input_path.is_dir():
        from kiero.batch import run_batch

        print(f"Input Dir:  {input_path}\nOutput Dir: {output_path}")
        print(f"Mode: {'per-image' if per_image else 'shared mask'}")
        if not per_image:
            print(f"Sample: {sample or 'all'}, confidence: {confidence}")
        run_batch(
            input_path=input_path,
            output_path=output_path,
            detector=_make_detector(confidence, padding, device),
            inpainter=_make_inpainter(device),
            per_image=per_image,
            sample_n=sample,
            confidence=confidence,
            memory_mb=memory,
            mask_output=mask_output,
        )

    else:
        print(f"Input Image:  {input_path}\nOutput Image: {output_path}")
        _make_pipeline(confidence, padding, device).run(input_path, output_path, mask_path=mask_output)
        print("Done.")


def _detect(
    input_path: Path,
    output_path: Path,
    sample: int | None,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
):
    _require_exists(input_path)

    if _is_cbz(input_path):
        import shutil

        from kiero.cbz import extract_cbz

        print(f"Input Archive:  {input_path}\nMask Output: {output_path}")
        tmp_in = extract_cbz(input_path)
        try:
            _detect(
                input_path=tmp_in,
                output_path=output_path,
                sample=sample,
                confidence=confidence,
                padding=padding,
                memory=memory,
                device=device,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
        return

    if input_path.is_dir():
        from kiero.batch import detect_batch

        print(f"Input Dir:  {input_path}\nMask Output: {output_path}")
        detect_batch(
            input_path=input_path,
            output_path=output_path,
            detector=_make_detector(confidence, padding, device),
            sample_n=sample,
            confidence=confidence,
            memory_mb=memory,
        )

    else:
        print(f"Input Image:  {input_path}\nMask Output: {output_path}")
        _make_pipeline(confidence, padding, device).detect(input_path, output_path)
        print("Done.")


def _inpaint(input_path: Path, output_path: Path, mask: Path, device: str | None):
    _require_exists(input_path, "Input")
    _require_exists(mask, "Mask file")

    if _is_cbz(input_path):
        import shutil

        from kiero.cbz import extract_cbz

        print(f"Input Archive: {input_path}")
        tmp_in = extract_cbz(input_path)
        try:
            _inpaint(
                input_path=tmp_in,
                output_path=output_path,
                mask=mask,
                device=device,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
        return

    if input_path.is_dir() and _is_cbz(output_path):
        import shutil
        import tempfile

        from kiero.cbz import write_cbz

        print(f"Output Archive: {output_path}")
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        try:
            _inpaint(
                input_path=input_path,
                output_path=tmp_out,
                mask=mask,
                device=device,
            )
            write_cbz(tmp_out, output_path)
            print(f"\n  Archive written to {output_path}")
        finally:
            shutil.rmtree(tmp_out, ignore_errors=True)
        return

    if input_path.is_dir():
        from kiero.batch import inpaint_batch
        from kiero.utils import load_mask

        print(f"Input Dir: {input_path}\nMask:  {mask}\nOutput Dir: {output_path}")
        inpaint_batch(
            input_path=input_path,
            output_path=output_path,
            mask=load_mask(mask),
            inpainter=_make_inpainter(device),
        )

    else:
        print(f"Input Image: {input_path}\nMask:  {mask}\nOutput Image: {output_path}")
        _make_pipeline(0.25, 10, device).inpaint(input_path, output_path, mask)
        print("Done.")

