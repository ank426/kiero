import argparse
import shutil
import sys
import time
from pathlib import Path


def _formatter(prog: str) -> argparse.HelpFormatter:
    width = shutil.get_terminal_size().columns
    return argparse.HelpFormatter(prog, max_help_position=40, width=width)


def _is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() == ".cbz"


def _require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Extra pixels around each detected box (default: 10).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: 'cuda', 'cpu', or auto (default: auto).",
    )


def _add_batch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Detect independently per image instead of shared mask.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N images for mask averaging (default: all).",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=1024,
        metavar="MB",
        help="Memory budget in MB for batch loading (default: 1024).",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="kiero",
        description="Manga watermark detector and remover.",
        formatter_class=_formatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run",
        help="Detect and inpaint (full pipeline).",
        usage="%(prog)s [OPTIONS] input output",
        formatter_class=_formatter,
    )
    p_run.add_argument("input", help="Image, directory, or .cbz file.")
    p_run.add_argument("output", help="Output path.")
    p_run.add_argument("--mask-output", help="Save detection mask here.")
    _add_common_options(p_run)
    _add_batch_options(p_run)
    p_run._positionals.title = "arguments"

    p_det = sub.add_parser(
        "detect",
        help="Detect watermarks, save mask only.",
        usage="%(prog)s [OPTIONS] input output",
        formatter_class=_formatter,
    )
    p_det.add_argument("input", help="Image, directory, or .cbz file.")
    p_det.add_argument("output", help="Mask output path.")
    _add_common_options(p_det)
    _add_batch_options(p_det)
    p_det._positionals.title = "arguments"

    p_inp = sub.add_parser(
        "inpaint",
        help="Inpaint with a provided mask.",
        usage="%(prog)s [OPTIONS] -m MASK input output",
        formatter_class=_formatter,
    )
    p_inp.add_argument("input", help="Image, directory, or .cbz file.")
    p_inp.add_argument("output", help="Result output path.")
    p_inp.add_argument("-m", "--mask", required=True, help="Binary mask image.")
    p_inp.add_argument(
        "--device", default=None, help="Device: 'cuda', 'cpu', or auto (default: auto)."
    )
    p_inp._positionals.title = "arguments"

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "detect":
        _cmd_detect(args)
    elif args.command == "inpaint":
        _cmd_inpaint(args)


def _make_detector(args):
    from kiero.detectors.yolo import YoloDetector

    return YoloDetector(
        confidence=args.confidence,
        padding=args.padding,
        device=args.device,
    )


def _make_inpainter(args):
    from kiero.inpainters.lama import LamaInpainter

    return LamaInpainter(device=args.device)


def _cmd_run(args):
    input_path = Path(args.input)
    _require_exists(input_path)
    output_path = Path(args.output)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    if _is_batch(input_path):
        from kiero.batch import run_batch

        print(f"Mode: {'per-image' if args.per_image else 'shared mask'}")
        if not args.per_image:
            sample_str = str(args.sample) if args.sample else "all"
            print(f"Sample: {sample_str}, confidence: {args.confidence}")

        run_batch(
            input_path=input_path,
            output_path=output_path,
            detector=_make_detector(args),
            inpainter=_make_inpainter(args),
            per_image=args.per_image,
            sample_n=args.sample,
            confidence=args.confidence,
            memory_mb=args.memory,
            mask_output=Path(args.mask_output) if args.mask_output else None,
        )
    else:
        from kiero.pipeline import Pipeline

        pipeline = Pipeline(
            confidence=args.confidence,
            padding=args.padding,
            device=args.device,
        )
        pipeline.run(input_path, output_path, mask_path=args.mask_output)

    print("Done.")


def _cmd_detect(args):
    input_path = Path(args.input)
    _require_exists(input_path)

    print(f"Input: {input_path}")

    if _is_batch(input_path):
        import shutil
        from kiero.batch import resolve_inputs, collect_shared_mask
        from kiero.utils import save_image

        image_paths, source_type, temp_dir = resolve_inputs(input_path)
        print(f"  Source: {source_type} ({len(image_paths)} images)")

        try:
            sample_str = str(args.sample) if args.sample else "all"
            print(f"  Sample: {sample_str}, confidence: {args.confidence}")

            mask = collect_shared_mask(
                image_paths,
                _make_detector(args),
                sample_n=args.sample,
                confidence=args.confidence,
                memory_mb=args.memory,
            )
            save_image(mask, args.output)
            print(f"Shared mask saved to {args.output}")
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        from kiero.pipeline import Pipeline
        from kiero.utils import save_image

        pipeline = Pipeline(
            confidence=args.confidence,
            padding=args.padding,
            device=args.device,
        )
        mask = pipeline.detect(input_path)
        save_image(mask, args.output)
        print(f"Mask saved to {args.output}")


def _cmd_inpaint(args):
    from kiero.inpainters.lama import LamaInpainter
    from kiero.utils import load_image, load_mask, save_image

    input_path = Path(args.input)
    mask_path = Path(args.mask)

    _require_exists(input_path, "Input")
    _require_exists(mask_path, "Mask file")

    print(f"Input: {input_path}")
    print(f"Mask:  {mask_path}")

    inpainter = LamaInpainter(device=args.device)
    mask = load_mask(mask_path)

    if _is_batch(input_path):
        from kiero.batch import inpaint_batch

        print(f"Output: {args.output}")
        inpaint_batch(
            input_path=input_path,
            output_path=Path(args.output),
            mask=mask,
            inpainter=inpainter,
        )
    else:
        image = load_image(input_path)

        t0 = time.time()
        result = inpainter.inpaint(image, mask)
        elapsed = time.time() - t0
        print(f"  Inpainting done in {elapsed:.1f}s")

        save_image(result, args.output)
        print(f"Result saved to {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
