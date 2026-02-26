def _is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in {".cbz", ".zip"}

def _is_cbz(path: Path) -> bool:
    return path.suffix.lower() in {".cbz", ".zip"}

def _require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        sys.exit(f"Error: {label} not found: {path}")

def _add_help(p: argparse.ArgumentParser) -> None:
    p.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit")

def _subparser(sub, name: str, *, desc: str, usage: str) -> argparse.ArgumentParser:
    p = sub.add_parser(name, help=desc, usage=usage, formatter_class=_Formatter, add_help=False)
    _add_help(p)
    p.add_argument("input", help="Image, directory, or .cbz file")
    p.add_argument("output", help="Output path")
    p._positionals.title, p._optionals.title = "Arguments", "Options"
    return p

def _add_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--confidence", type=float, default=0.25, help="YOLO detection confidence threshold (default: 0.25)")
    p.add_argument("--padding", type=int, default=10, help="Extra pixels around each detected box (default: 10)")
    p.add_argument("--device", default=None, help="Device: 'cuda', 'cpu', or auto (default: auto)")
    p.add_argument("--per-image", action="store_true", help="Detect independently per image instead of shared mask")
    p.add_argument(
        "--sample", type=int, default=None, metavar="N", help="Sample N images for mask averaging (default: all)"
    )
    p.add_argument(
        "--memory", type=int, default=1024, metavar="MB", help="Memory budget in MB for batch loading (default: 1024)"
    )

def _make_detector(confidence: float, padding: int, device: str | None):
    from kiero.detectors.yolo import YoloDetector
    return YoloDetector(confidence=confidence, padding=padding, device=device)

def _make_inpainter(device: str | None):
    from kiero.inpainters.lama import LamaInpainter
    return LamaInpainter(device=device)

def _make_pipeline(confidence: float, padding: int, device: str | None):
    from kiero.pipeline import Pipeline
    return Pipeline(confidence=confidence, padding=padding, device=device)

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
                input_path=args.input,
                output_path=args.output,
                per_image=args.per_image,
                sample=args.sample,
                confidence=args.confidence,
                padding=args.padding,
                memory=args.memory,
                device=args.device,
                mask_output=args.mask_output,
            )
        case "detect":
            _detect(
                input_path=args.input,
                output_path=args.output,
                sample=args.sample,
                confidence=args.confidence,
                padding=args.padding,
                memory=args.memory,
                device=args.device,
            )
        case "inpaint":
            _inpaint(
                input_path=args.input,
                output_path=args.output,
                mask=args.mask,
                device=args.device,
            )
        case _:
            parser.print_help()


def _run(
    input_path: str,
    output_path: str,
    per_image: bool,
    sample: int | None,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
    mask_output: str | None,
):
    inp, out = Path(input_path), Path(output_path)
    _require_exists(inp)
    print(f"Input:  {inp}\nOutput: {out}")

    if _is_cbz(inp):
        import shutil
        import tempfile
        from kiero.cbz import extract_cbz, write_cbz
        from kiero.batch import run_batch
        
        tmp_in = extract_cbz(inp)
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
                mask_output=Path(mask_output) if mask_output else None,
            )
            write_cbz(tmp_out, out)
            print(f"\n  Archive written to {out}")
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            
    elif inp.is_dir():
        from kiero.batch import run_batch

        print(f"Mode: {'per-image' if per_image else 'shared mask'}")
        if not per_image:
            print(f"Sample: {sample or 'all'}, confidence: {confidence}")
        run_batch(
            input_path=inp,
            output_path=out,
            detector=_make_detector(confidence, padding, device),
            inpainter=_make_inpainter(device),
            per_image=per_image,
            sample_n=sample,
            confidence=confidence,
            memory_mb=memory,
            mask_output=Path(mask_output) if mask_output else None,
        )
    else:
        _make_pipeline(confidence, padding, device).run(inp, out, mask_path=mask_output)
    print("Done.")


def _detect(
    input_path: str,
    output_path: str,
    sample: int | None,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
):
    inp, out = Path(input_path), Path(output_path)
    _require_exists(inp)
    print(f"Input:  {inp}\nOutput: {out}")

    if _is_cbz(inp):
        import shutil
        from kiero.cbz import extract_cbz
        from kiero.batch import detect_batch
        
        tmp_in = extract_cbz(inp)
        try:
            detect_batch(
                input_path=tmp_in,
                output_path=out,
                detector=_make_detector(confidence, padding, device),
                sample_n=sample,
                confidence=confidence,
                memory_mb=memory,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            
    elif inp.is_dir():
        from kiero.batch import detect_batch

        detect_batch(
            input_path=inp,
            output_path=out,
            detector=_make_detector(confidence, padding, device),
            sample_n=sample,
            confidence=confidence,
            memory_mb=memory,
        )
    else:
        _make_pipeline(confidence, padding, device).detect(inp, out)
    print("Done.")


def _inpaint(input_path: str, output_path: str, mask: str, device: str | None):
    inp, out, mask_path = Path(input_path), Path(output_path), Path(mask)
    _require_exists(inp, "Input")
    _require_exists(mask_path, "Mask file")
    print(f"Input: {inp}\nMask:  {mask_path}\nOutput: {out}")

    if _is_cbz(inp):
        import shutil
        import tempfile
        from kiero.cbz import extract_cbz, write_cbz
        from kiero.batch import inpaint_batch
        from kiero.utils import load_mask
        
        tmp_in = extract_cbz(inp)
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        
        try:
            inpaint_batch(
                input_path=tmp_in,
                output_path=tmp_out,
                mask=load_mask(mask_path),
                inpainter=_make_inpainter(device),
            )
            write_cbz(tmp_out, out)
            print(f"\n  Archive written to {out}")
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)
            shutil.rmtree(tmp_out, ignore_errors=True)
            
    elif inp.is_dir():
        from kiero.batch import inpaint_batch
        from kiero.utils import load_mask

        inpaint_batch(
            input_path=inp,
            output_path=out,
            mask=load_mask(mask_path),
            inpainter=_make_inpainter(device),
        )
    else:
        _make_pipeline(0.25, 10, device).inpaint(inp, out, mask_path)
    print("Done.")

if __name__ == "__main__":
    main()
