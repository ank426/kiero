def main():
    parser = argparse.ArgumentParser(
        prog="kiero", description="Manga watermark detector and remover", formatter_class=_Formatter, add_help=False
    )
    _add_help(parser)
    parser._positionals.title, parser._optionals.title = "Commands", "Options"
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = _subparser(sub, "run", desc="Detect and inpaint (full pipeline)", usage="%(prog)s [OPTIONS] input output")
    p_run.add_argument("--mask-output", help="Save detection mask here")
    _add_options(p_run)

    _add_options(
        _subparser(sub, "detect", desc="Detect watermarks, save mask only", usage="%(prog)s [OPTIONS] input output")
    )

    p_inp = _subparser(
        sub, "inpaint", desc="Inpaint with a provided mask", usage="%(prog)s [OPTIONS] -m MASK input output"
    )
    p_inp.add_argument("-m", "--mask", required=True, help="Binary mask image")
    p_inp.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or auto (default: auto)")

    args = parser.parse_args()

    match args.command:
        case "run":
            _run(
                input_path=Path(args.input),
                output_path=Path(args.output),
                per_image=args.per_image,
                sample=args.sample,
                confidence=args.confidence,
                padding=args.padding,
                memory=args.memory,
                device=args.device,
                mask_output=Path(args.mask_output) if getattr(args, "mask_output", None) else None,
            )
        case "detect":
            _detect(
                input_path=Path(args.input),
                output_path=Path(args.output),
                sample=args.sample,
                confidence=args.confidence,
                padding=args.padding,
                memory=args.memory,
                device=args.device,
            )
        case "inpaint":
            _inpaint(
                input_path=Path(args.input),
                output_path=Path(args.output),
                mask=Path(args.mask),
                device=args.device,
            )
        case _:
            parser.print_help()


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
    print(f"Input:  {input_path}\nOutput: {output_path}")

    if _is_cbz(input_path):
        import shutil
        import tempfile

        from kiero.batch import run_batch
        from kiero.cbz import extract_cbz, write_cbz

        tmp_in = extract_cbz(input_path)
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))

        try:
            print(f"Mode: {'per-image' if per_image else 'shared mask'} (Archive)")
            if not per_image:
                print(f"Sample: {sample or 'all'}, confidence: {confidence}")
            run_batch(
                input_path=tmp_in,
                output_path=tmp_out,
                detector=_make_detector(confidence, padding, device),
                inpainter=_make_inpainter(device),
                per_image=per_image,
                sample_n=sample,
                confidence=confidence,
                memory_mb=memory,
                mask_output=mask_output,
            )
            write_cbz(tmp_out, output_path)
            print(f"\n  Archive written to {output_path}")
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)

    elif input_path.is_dir():
        from kiero.batch import run_batch

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
    print(f"Input:  {input_path}\nOutput: {output_path}")

    if _is_cbz(input_path):
        import shutil

        from kiero.batch import detect_batch
        from kiero.cbz import extract_cbz

        tmp_in = extract_cbz(input_path)
        try:
            detect_batch(
                input_path=tmp_in,
                output_path=output_path,
                detector=_make_detector(confidence, padding, device),
                sample_n=sample,
                confidence=confidence,
                memory_mb=memory,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)

    elif input_path.is_dir():
        from kiero.batch import detect_batch

        detect_batch(
            input_path=input_path,
            output_path=output_path,
            detector=_make_detector(confidence, padding, device),
            sample_n=sample,
            confidence=confidence,
            memory_mb=memory,
        )
    else:
        _make_pipeline(confidence, padding, device).detect(input_path, output_path)
    print("Done.")


def _inpaint(input_path: Path, output_path: Path, mask: Path, device: str | None):
    _require_exists(input_path, "Input")
    _require_exists(mask, "Mask file")
    print(f"Input: {input_path}\nMask:  {mask}\nOutput: {output_path}")

    if _is_cbz(input_path):
        import shutil
        import tempfile

        from kiero.batch import inpaint_batch
        from kiero.cbz import extract_cbz, write_cbz
        from kiero.utils import load_mask

        tmp_in = extract_cbz(input_path)
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))

        try:
            inpaint_batch(
                input_path=tmp_in,
                output_path=tmp_out,
                mask=load_mask(mask),
                inpainter=_make_inpainter(device),
            )
            write_cbz(tmp_out, output_path)
            print(f"\n  Archive written to {output_path}")
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)

    elif input_path.is_dir():
        from kiero.batch import inpaint_batch
        from kiero.utils import load_mask

        inpaint_batch(
            input_path=input_path,
            output_path=output_path,
            mask=load_mask(mask),
            inpainter=_make_inpainter(device),
        )
    else:
        _make_pipeline(0.25, 10, device).inpaint(input_path, output_path, mask)
    print("Done.")

