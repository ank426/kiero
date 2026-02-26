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

    if _is_batch(inp):
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

    if _is_batch(inp):
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

    if _is_batch(inp):
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
