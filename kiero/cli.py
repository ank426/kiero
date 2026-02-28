import argparse
import shutil
from pathlib import Path

from kiero import commands


class _Formatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=40, width=shutil.get_terminal_size().columns)

    def _format_usage(
        self,
        usage: str | None,
        actions: list[argparse.Action],
        groups: list[argparse._ArgumentGroup | argparse._MutuallyExclusiveGroup],
        prefix: str | None,
    ) -> str:
        if prefix is None:
            prefix = "Usage: "
        return super()._format_usage(usage, actions, groups, prefix)

    def _format_action(self, action: argparse.Action) -> str:
        if isinstance(action, argparse._SubParsersAction):
            return self._join_parts([self._format_action(a) for a in action._get_subactions()])
        return super()._format_action(action)


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


def main() -> None:
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
            commands.run(
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
            commands.detect(
                input_path=Path(args.input),
                output_path=Path(args.output),
                sample=args.sample,
                confidence=args.confidence,
                padding=args.padding,
                memory=args.memory,
                device=args.device,
            )
        case "inpaint":
            commands.inpaint(
                input_path=Path(args.input),
                output_path=Path(args.output),
                mask=Path(args.mask),
                device=args.device,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
