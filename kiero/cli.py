import argparse
import shutil
import sys
import time
from pathlib import Path

_HELP = "Show this help message and exit"
_DEVICE_HELP = "Device: 'cuda', 'cpu', or auto (default: auto)"
_USAGE = "%(prog)s [OPTIONS] input output"


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
    return path.is_dir() or path.suffix.lower() == ".cbz"


def _require_exists(path: Path, label: str = "Input") -> None:
    if not path.exists():
        sys.exit(f"Error: {label} not found: {path}")


def _add_help(p: argparse.ArgumentParser) -> None:
    p.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=_HELP)


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
    p.add_argument("--device", default=None, help=_DEVICE_HELP)
    p.add_argument("--per-image", action="store_true", help="Detect independently per image instead of shared mask")
    p.add_argument(
        "--sample", type=int, default=None, metavar="N", help="Sample N images for mask averaging (default: all)"
    )
    p.add_argument(
        "--memory", type=int, default=1024, metavar="MB", help="Memory budget in MB for batch loading (default: 1024)"
    )


def _make_detector(args):
    from kiero.detectors.yolo import YoloDetector

    return YoloDetector(confidence=args.confidence, padding=args.padding, device=args.device)


def _make_inpainter(args):
    from kiero.inpainters.lama import LamaInpainter

    return LamaInpainter(device=args.device)


def _make_pipeline(args):
    from kiero.pipeline import Pipeline

    return Pipeline(confidence=args.confidence, padding=args.padding, device=args.device)


def main():
    parser = argparse.ArgumentParser(
        prog="kiero", description="Manga watermark detector and remover", formatter_class=_Formatter, add_help=False
    )
    _add_help(parser)
    parser._positionals.title, parser._optionals.title = "Commands", "Options"
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = _subparser(sub, "run", desc="Detect and inpaint (full pipeline)", usage=_USAGE)
    p_run.add_argument("--mask-output", help="Save detection mask here")
    _add_options(p_run)

    _add_options(_subparser(sub, "detect", desc="Detect watermarks, save mask only", usage=_USAGE))

    p_inp = _subparser(
        sub, "inpaint", desc="Inpaint with a provided mask", usage="%(prog)s [OPTIONS] -m MASK input output"
    )
    p_inp.add_argument("-m", "--mask", required=True, help="Binary mask image")
    p_inp.add_argument("--device", default=None, help=_DEVICE_HELP)

    args = parser.parse_args()
    {"run": _cmd_run, "detect": _cmd_detect, "inpaint": _cmd_inpaint}[args.command](args)


def _cmd_run(args):
    inp, out = Path(args.input), Path(args.output)
    _require_exists(inp)
    print(f"Input:  {inp}\nOutput: {out}")

    if _is_batch(inp):
        from kiero.batch import run_batch

        print(f"Mode: {'per-image' if args.per_image else 'shared mask'}")
        if not args.per_image:
            print(f"Sample: {args.sample or 'all'}, confidence: {args.confidence}")
        run_batch(
            input_path=inp,
            output_path=out,
            detector=_make_detector(args),
            inpainter=_make_inpainter(args),
            per_image=args.per_image,
            sample_n=args.sample,
            confidence=args.confidence,
            memory_mb=args.memory,
            mask_output=Path(args.mask_output) if args.mask_output else None,
        )
    else:
        _make_pipeline(args).run(inp, out, mask_path=args.mask_output)
    print("Done.")


def _cmd_detect(args):
    inp, out = Path(args.input), Path(args.output)
    _require_exists(inp)
    print(f"Input:  {inp}\nOutput: {out}")

    if _is_batch(inp):
        from kiero.batch import detect_batch

        detect_batch(
            input_path=inp,
            output_path=out,
            detector=_make_detector(args),
            sample_n=args.sample,
            confidence=args.confidence,
            memory_mb=args.memory,
        )
    else:
        from kiero.utils import save_image

        save_image(_make_pipeline(args).detect(inp), out)
        print(f"Mask saved to {out}")


def _cmd_inpaint(args):
    from kiero.utils import load_image, load_mask, save_image

    inp, mask_path = Path(args.input), Path(args.mask)
    _require_exists(inp, "Input")
    _require_exists(mask_path, "Mask file")
    print(f"Input: {inp}\nMask:  {mask_path}")

    inpainter, mask = _make_inpainter(args), load_mask(mask_path)
    if _is_batch(inp):
        from kiero.batch import inpaint_batch

        print(f"Output: {args.output}")
        inpaint_batch(input_path=inp, output_path=Path(args.output), mask=mask, inpainter=inpainter)
    else:
        t0 = time.time()
        result = inpainter.inpaint(load_image(inp), mask)
        print(f"  Inpainting done in {time.time() - t0:.1f}s")
        save_image(result, args.output)
        print(f"Result saved to {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
