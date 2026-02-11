"""CLI entry point for kiero."""

import argparse
import sys
from pathlib import Path

from kiero.detectors import DETECTORS
from kiero.inpainters import INPAINTERS


def _is_batch_input(path: Path) -> bool:
    """Check if the input path is a directory or CBZ file (batch mode)."""
    return path.is_dir() or path.suffix.lower() == ".cbz"


def main():
    parser = argparse.ArgumentParser(
        prog="kiero",
        description="Manga watermark detector and remover.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run: full pipeline ---
    run_parser = subparsers.add_parser(
        "run",
        help="Detect watermarks and inpaint them (full pipeline).",
    )
    run_parser.add_argument(
        "input",
        help="Path to input image, directory of images, or .cbz file.",
    )
    run_parser.add_argument(
        "-o", "--output", help="Path to save the result (file, directory, or .cbz)."
    )
    run_parser.add_argument("--mask-output", help="Path to save the detection mask.")
    run_parser.add_argument(
        "-d",
        "--detector",
        default="yolo11x",
        choices=list(DETECTORS.keys()),
        help="Detector to use (default: yolo11x).",
    )
    run_parser.add_argument(
        "-i",
        "--inpainter",
        default="lama",
        choices=list(INPAINTERS.keys()),
        help="Inpainter to use (default: lama).",
    )
    _add_detector_options(run_parser)
    _add_inpainter_options(run_parser)
    _add_batch_options(run_parser)

    # --- detect: detection only ---
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect watermarks and save the mask (no inpainting).",
    )
    detect_parser.add_argument(
        "input",
        help="Path to input image, directory of images, or .cbz file.",
    )
    detect_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the mask (single PNG). For batch input, saves the averaged shared mask.",
    )
    detect_parser.add_argument(
        "-d",
        "--detector",
        default="yolo11x",
        choices=list(DETECTORS.keys()),
        help="Detector to use (default: yolo11x).",
    )
    _add_detector_options(detect_parser)
    _add_batch_options(detect_parser)

    # --- inpaint: inpainting only ---
    inpaint_parser = subparsers.add_parser(
        "inpaint",
        help="Inpaint masked regions using a provided mask.",
    )
    inpaint_parser.add_argument("input", help="Path to input image.")
    inpaint_parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to save the result.",
    )
    inpaint_parser.add_argument(
        "-m",
        "--mask",
        required=True,
        help="Path to the binary mask image.",
    )
    inpaint_parser.add_argument(
        "-i",
        "--inpainter",
        default="lama",
        choices=list(INPAINTERS.keys()),
        help="Inpainter to use (default: lama).",
    )
    _add_inpainter_options(inpaint_parser)

    # --- compare: run all detectors ---
    compare_parser = subparsers.add_parser(
        "compare",
        help="Run all detectors on a single image and produce a side-by-side comparison.",
    )
    compare_parser.add_argument("input", help="Path to input image (single file only).")
    compare_parser.add_argument(
        "-o",
        "--output-dir",
        default="./comparison_output",
        help="Directory to save comparison results (default: ./comparison_output).",
    )
    compare_parser.add_argument(
        "-i",
        "--inpainter",
        default="lama",
        choices=list(INPAINTERS.keys()),
        help="Inpainter to use for all results (default: lama).",
    )
    _add_inpainter_options(compare_parser)

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "detect":
        _cmd_detect(args)
    elif args.command == "inpaint":
        _cmd_inpaint(args)
    elif args.command == "compare":
        _cmd_compare(args)


def _add_detector_options(parser: argparse.ArgumentParser):
    """Add detector-specific options to a parser."""
    group = parser.add_argument_group("detector options")
    group.add_argument(
        "--template",
        help="Path to watermark template image (for template detector).",
    )
    group.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Detection confidence threshold (for YOLO detector, default: 0.25).",
    )
    group.add_argument(
        "--prompt",
        default="watermark",
        help='Text prompt (for CLIPSeg detector, default: "watermark").',
    )
    group.add_argument(
        "--det-threshold",
        type=float,
        default=None,
        help="Detection threshold. For template detector: matching threshold (default: 0.7). "
        "For CLIPSeg: heatmap threshold (default: 0.3).",
    )
    group.add_argument(
        "--dilate",
        type=int,
        default=None,
        help="Pixels to dilate the mask by (default varies by detector).",
    )
    group.add_argument(
        "--device",
        default=None,
        help="Device for ML detectors: 'cuda', 'cpu', or auto (default: auto).",
    )


def _add_inpainter_options(parser: argparse.ArgumentParser):
    """Add inpainter-specific options to a parser."""
    group = parser.add_argument_group("inpainter options")
    group.add_argument(
        "--inp-method",
        default="telea",
        choices=["telea", "ns"],
        help="OpenCV inpainting method (for opencv inpainter, default: telea).",
    )
    group.add_argument(
        "--inp-radius",
        type=int,
        default=3,
        help="OpenCV inpainting radius (for opencv inpainter, default: 3).",
    )


def _add_batch_options(parser: argparse.ArgumentParser):
    """Add batch-mode options to a parser."""
    group = parser.add_argument_group("batch options")
    group.add_argument(
        "--per-image",
        action="store_true",
        help="Detect independently per image instead of computing a shared mask. "
        "Only relevant for directory/CBZ input.",
    )
    group.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Number of images to sample for shared mask averaging. "
        "Default: use all images. Ignored with --per-image.",
    )
    group.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Fraction of sampled images that must agree for a pixel to be "
        "in the shared mask (default: 0.5 = detected in >=50%% of samples).",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Batch size for batched GPU detection. Default: auto per detector "
        "(4 for YOLO, 16 for CLIPSeg).",
    )


def _build_detector_kwargs(args) -> dict:
    """Build kwargs for a detector from CLI args."""
    kwargs = {}
    detector = getattr(args, "detector", "yolo11x")

    if detector == "template":
        if args.template:
            kwargs["template_path"] = args.template
        if args.det_threshold is not None:
            kwargs["threshold"] = args.det_threshold
        if args.dilate is not None:
            kwargs["dilate_px"] = args.dilate

    elif detector == "yolo11x":
        kwargs["confidence"] = args.confidence
        if args.device:
            kwargs["device"] = args.device

    elif detector == "clipseg":
        kwargs["prompt"] = args.prompt
        if args.det_threshold is not None:
            kwargs["threshold"] = args.det_threshold
        if args.dilate is not None:
            kwargs["dilate_px"] = args.dilate
        if args.device:
            kwargs["device"] = args.device

    return kwargs


def _build_inpainter_kwargs(args) -> dict:
    """Build kwargs for an inpainter from CLI args."""
    kwargs = {}
    inpainter = getattr(args, "inpainter", "lama")

    if inpainter == "opencv":
        kwargs["method"] = args.inp_method
        kwargs["radius"] = args.inp_radius

    elif inpainter == "lama":
        device = getattr(args, "device", None)
        if device:
            kwargs["device"] = device

    return kwargs


def _cmd_run(args):
    """Execute the 'run' command."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    detector_kwargs = _build_detector_kwargs(args)
    inpainter_kwargs = _build_inpainter_kwargs(args)

    # --- Batch mode (directory or CBZ) ---
    if _is_batch_input(input_path):
        from kiero.batch import run_batch
        from kiero.detectors import get_detector
        from kiero.inpainters import get_inpainter

        # Default output path for batch
        output_path = args.output
        if output_path is None:
            if input_path.is_dir():
                output_path = str(input_path) + "_clean"
            else:  # CBZ
                output_path = (
                    input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"
                )

        print(f"Input: {input_path}")
        print(f"Detector: {args.detector}")
        print(f"Inpainter: {args.inpainter}")
        print(f"Output: {output_path}")
        print(f"Mode: {'per-image' if args.per_image else 'shared mask'}")
        if not args.per_image:
            sample_str = str(args.sample) if args.sample else "all"
            print(f"Sample: {sample_str}, mask threshold: {args.mask_threshold}")

        detector = get_detector(args.detector, **detector_kwargs)
        inpainter = get_inpainter(args.inpainter, **inpainter_kwargs)

        run_batch(
            input_path=input_path,
            output_path=Path(output_path),
            detector=detector,
            inpainter=inpainter,
            per_image=args.per_image,
            sample_n=args.sample,
            mask_threshold=args.mask_threshold,
            batch_size=args.batch_size,
            mask_output=Path(args.mask_output) if args.mask_output else None,
        )
        print("Done.")
        return

    # --- Single image mode ---
    from kiero.pipeline import Pipeline

    output_path = args.output
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"

    print(f"Input: {input_path}")
    print(f"Detector: {args.detector}")
    print(f"Inpainter: {args.inpainter}")
    print(f"Output: {output_path}")

    pipeline = Pipeline(
        detector=args.detector,
        inpainter=args.inpainter,
        detector_kwargs=detector_kwargs,
        inpainter_kwargs=inpainter_kwargs,
    )
    pipeline.run(input_path, output_path, mask_path=args.mask_output)
    print("Done.")


def _cmd_detect(args):
    """Execute the 'detect' command."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    detector_kwargs = _build_detector_kwargs(args)

    # --- Batch mode: compute shared mask ---
    if _is_batch_input(input_path):
        from kiero.batch import resolve_inputs, collect_shared_mask
        from kiero.detectors import get_detector
        from kiero.utils import save_image
        import shutil

        print(f"Input: {input_path}")
        print(f"Detector: {args.detector}")

        detector = get_detector(args.detector, **detector_kwargs)
        image_paths, source_type, temp_dir = resolve_inputs(input_path)
        print(f"  Source: {source_type} ({len(image_paths)} images)")

        try:
            sample_str = str(args.sample) if args.sample else "all"
            print(f"  Sample: {sample_str}, mask threshold: {args.mask_threshold}")

            mask = collect_shared_mask(
                image_paths,
                detector,
                sample_n=args.sample,
                threshold=args.mask_threshold,
                batch_size=args.batch_size,
            )
            save_image(mask, args.output)
            print(f"Shared mask saved to {args.output}")
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        return

    # --- Single image mode ---
    from kiero.pipeline import Pipeline
    from kiero.utils import save_image

    print(f"Input: {input_path}")
    print(f"Detector: {args.detector}")

    pipeline = Pipeline(
        detector=args.detector,
        inpainter="opencv",  # dummy, won't be used
        detector_kwargs=detector_kwargs,
    )

    mask = pipeline.detect(input_path)
    save_image(mask, args.output)
    print(f"Mask saved to {args.output}")


def _cmd_inpaint(args):
    """Execute the 'inpaint' command."""
    from kiero.utils import load_image, load_mask, save_image
    from kiero.inpainters import get_inpainter

    input_path = Path(args.input)
    mask_path = Path(args.mask)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not mask_path.exists():
        print(f"Error: Mask file not found: {mask_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {input_path}")
    print(f"Mask: {mask_path}")
    print(f"Inpainter: {args.inpainter}")

    inpainter_kwargs = _build_inpainter_kwargs(args)
    inpainter = get_inpainter(args.inpainter, **inpainter_kwargs)

    image = load_image(input_path)
    mask = load_mask(mask_path)

    import time

    t0 = time.time()
    result = inpainter.inpaint(image, mask)
    elapsed = time.time() - t0
    print(f"  Inpainting done in {elapsed:.1f}s")

    save_image(result, args.output)
    print(f"Result saved to {args.output}")


def _cmd_compare(args):
    """Execute the 'compare' command."""
    from kiero.pipeline import compare

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {input_path}")
    print(f"Inpainter: {args.inpainter}")
    print(f"Output dir: {args.output_dir}")
    print(f"Running all detectors: {', '.join(DETECTORS.keys())}")

    inpainter_kwargs = _build_inpainter_kwargs(args)

    compare(
        image_path=input_path,
        output_dir=args.output_dir,
        inpainter=args.inpainter,
        inpainter_kwargs=inpainter_kwargs,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
