import argparse
import shutil
import sys
from pathlib import Path


class _Formatter(argparse.HelpFormatter):
    def __init__(self, prog: str):
        super().__init__(prog, max_help_position=40, width=shutil.get_terminal_size().columns)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "Usage: "
        return super()._format_usage(usage, actions, groups, prefix)

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            return self._join_parts([self._format_action(a) for a in action._get_subactions()])
        return super()._format_action(action)


def _is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() in {".cbz", ".zip"}


def _is_cbz(path: Path) -> bool:
    return path.suffix.lower() in {".cbz", ".zip"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _validate_run(input_path: Path, output_path: Path, mask_output: Path | None):
    _require_exists(input_path)
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
    _require_exists(input_path)
    if not (_is_cbz(input_path) or input_path.is_dir() or _is_image(input_path)):
        sys.exit(f"Error: Input must be a directory, .cbz/.zip, or an image file: {input_path}")

    if not _is_image(output_path):
        sys.exit(f"Error: Mask output must be an image file: {output_path}")


def _validate_inpaint(input_path: Path, output_path: Path, mask: Path):
    _require_exists(input_path, "Input")
    _require_exists(mask, "Mask file")
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
    _validate_run(input_path, output_path, mask_output)

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

    elif _is_cbz(output_path):
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

    elif input_path.is_dir():
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
    _validate_detect(input_path, output_path)

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

    elif input_path.is_dir():
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
    _validate_inpaint(input_path, output_path, mask)

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

    elif _is_cbz(output_path):
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

    elif input_path.is_dir():
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


if __name__ == "__main__":
    main()
