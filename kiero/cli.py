import argparse
import shutil
import sys
import time
from pathlib import Path


class _Formatter(argparse.HelpFormatter):
    def __init__(self, prog: str):
        width = shutil.get_terminal_size().columns
        super().__init__(prog, max_help_position=40, width=width)

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "Usage: "
        return super()._format_usage(usage, actions, groups, prefix)

    def _format_action(self, action):
        if isinstance(action, argparse._SubParsersAction):
            parts = []
            for choice_action in action._get_subactions():
                parts.append(self._format_action(choice_action))
            return self._join_parts(parts)
        return super()._format_action(action)


def _is_batch(path: Path) -> bool:
    return path.is_dir() or path.suffix.lower() == ".cbz"


def _require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        print(f"Error: {label} not found: {path}", file=sys.stderr)
        sys.exit(1)


def _subparser(sub, name: str, *, desc: str, usage: str) -> argparse.ArgumentParser:
    p = sub.add_parser(
        name, help=desc, usage=usage, formatter_class=_Formatter, add_help=False
    )
    p.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    p.add_argument("input", help="Image, directory, or .cbz file")
    p.add_argument("output", help="Output path")
    p._positionals.title = "Arguments"
    p._optionals.title = "Options"
    return p


def _add_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Extra pixels around each detected box (default: 10)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: 'cuda', 'cpu', or auto (default: auto)",
    )
    parser.add_argument(
        "--per-image",
        action="store_true",
        help="Detect independently per image instead of shared mask",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Sample N images for mask averaging (default: all)",
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=1024,
        metavar="MB",
        help="Memory budget in MB for batch loading (default: 1024)",
    )


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


def main():
    parser = argparse.ArgumentParser(
        prog="kiero",
        description="Manga watermark detector and remover.",
        formatter_class=_Formatter,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit",
    )
    parser._positionals.title = "Commands"
    parser._optionals.title = "Options"
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = _subparser(
        sub,
        "run",
        desc="Detect and inpaint (full pipeline)",
        usage="%(prog)s [OPTIONS] input output",
    )
    p_run.add_argument("--mask-output", help="Save detection mask here")
    _add_options(p_run)

    p_det = _subparser(
        sub,
        "detect",
        desc="Detect watermarks, save mask only",
        usage="%(prog)s [OPTIONS] input output",
    )
    _add_options(p_det)

    p_inp = _subparser(
        sub,
        "inpaint",
        desc="Inpaint with a provided mask",
        usage="%(prog)s [OPTIONS] -m MASK input output",
    )
    p_inp.add_argument("-m", "--mask", required=True, help="Binary mask image")
    p_inp.add_argument(
        "--device",
        default=None,
        help="Device: 'cuda', 'cpu', or auto (default: auto)",
    )

    args = parser.parse_args()
    {"run": _cmd_run, "detect": _cmd_detect, "inpaint": _cmd_inpaint}[args.command](
        args
    )


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

        Pipeline(
            confidence=args.confidence,
            padding=args.padding,
            device=args.device,
        ).run(input_path, output_path, mask_path=args.mask_output)

    print("Done.")


def _cmd_detect(args):
    input_path = Path(args.input)
    _require_exists(input_path)
    output_path = Path(args.output)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    if _is_batch(input_path):
        from kiero.batch import detect_batch

        detect_batch(
            input_path=input_path,
            output_path=output_path,
            detector=_make_detector(args),
            sample_n=args.sample,
            confidence=args.confidence,
            memory_mb=args.memory,
        )
    else:
        from kiero.pipeline import Pipeline
        from kiero.utils import save_image

        mask = Pipeline(
            confidence=args.confidence,
            padding=args.padding,
            device=args.device,
        ).detect(input_path)
        save_image(mask, output_path)
        print(f"Mask saved to {output_path}")


def _cmd_inpaint(args):
    from kiero.utils import load_image, load_mask, save_image

    input_path = Path(args.input)
    mask_path = Path(args.mask)

    _require_exists(input_path, "Input")
    _require_exists(mask_path, "Mask file")

    print(f"Input: {input_path}")
    print(f"Mask:  {mask_path}")

    inpainter = _make_inpainter(args)
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
        print(f"  Inpainting done in {time.time() - t0:.1f}s")
        save_image(result, args.output)
        print(f"Result saved to {args.output}")

    print("Done.")


if __name__ == "__main__":
    main()
