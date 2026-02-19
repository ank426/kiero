import argparse
import sys
import time
from pathlib import Path


def _is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() == ".cbz"


def _require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def _default_output(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_clean"
    return input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"


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
    group = parser.add_argument_group("batch options")
    group.add_argument(
        "--per-image",
        action="store_true",
        help="Detect independently per image instead of shared mask.",
    )
    group.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N images for mask averaging (default: all).",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="GPU batch size for detection (default: 4).",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="kiero",
        description="Manga watermark detector and remover.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Detect and inpaint (full pipeline).")
    p_run.add_argument("input", help="Image, directory, or .cbz file.")
    p_run.add_argument("-o", "--output", help="Output path.")
    p_run.add_argument("--mask-output", help="Save detection mask here.")
    _add_common_options(p_run)
    _add_batch_options(p_run)

    p_det = sub.add_parser("detect", help="Detect watermarks, save mask only.")
    p_det.add_argument("input", help="Image, directory, or .cbz file.")
    p_det.add_argument("-o", "--output", required=True, help="Mask output path.")
    _add_common_options(p_det)
    _add_batch_options(p_det)

    p_inp = sub.add_parser("inpaint", help="Inpaint with a provided mask.")
    p_inp.add_argument("input", help="Image, directory, or .cbz file.")
    p_inp.add_argument("-o", "--output", required=True, help="Result output path.")
    p_inp.add_argument("-m", "--mask", required=True, help="Binary mask image.")
    p_inp.add_argument("--device", default=None, help="Device (default: auto).")

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
    output_path = Path(args.output) if args.output else _default_output(input_path)

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
            batch_size=args.batch_size,
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
                batch_size=args.batch_size,
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
