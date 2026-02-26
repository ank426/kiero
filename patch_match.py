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
