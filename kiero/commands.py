import shutil
import sys
import tempfile
from pathlib import Path

from kiero.batch import detect_batch, inpaint_batch, run_batch
from kiero.single import detect as detect_image
from kiero.single import inpaint as inpaint_image
from kiero.single import run as run_image
from kiero.utils import (
    extract_cbz,
    is_cbz,
    make_detector,
    make_inpainter,
    validate,
    write_cbz,
)


def detect(
    input_path: Path,
    output_path: Path,
    sample: int | None,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
) -> None:
    validate(input_path, mask_output=output_path)

    if is_cbz(input_path):
        print(f"Input Archive:  {input_path}\nMask Output: {output_path}")
        tmp_in = extract_cbz(input_path)
        try:
            detect(
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
        print(f"Input Dir:  {input_path}\nMask Output: {output_path}")
        detect_batch(
            input_path=input_path,
            output_path=output_path,
            detector=make_detector(confidence, padding, device),
            sample=sample,
            confidence=confidence,
            memory_mb=memory,
        )

    else:
        print(f"Input Image:  {input_path}\nMask Output: {output_path}")
        detect_image(input_path, output_path, confidence=confidence, padding=padding, device=device)
        print("Done.")


def inpaint(input_path: Path, output_path: Path, mask: Path, device: str | None) -> None:
    validate(input_path, output_path, mask)

    if is_cbz(input_path):
        print(f"Input Archive: {input_path}")
        tmp_in = extract_cbz(input_path)
        try:
            inpaint(
                input_path=tmp_in,
                output_path=output_path,
                mask=mask,
                device=device,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)

    elif is_cbz(output_path):
        print(f"Output Archive: {output_path}")
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        try:
            inpaint(
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
        print(f"Input Dir: {input_path}\nMask:  {mask}\nOutput Dir: {output_path}")
        inpaint_batch(
            input_path=input_path,
            output_path=output_path,
            mask=mask,
            inpainter=make_inpainter(device),
        )

    else:
        print(f"Input Image: {input_path}\nMask:  {mask}\nOutput Image: {output_path}")
        inpaint_image(input_path, output_path, mask, device=device)
        print("Done.")


def run(
    input_path: Path,
    output_path: Path,
    per_image: bool,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
    mask_output: Path | None,
) -> None:
    validate(input_path, output_path, mask_output=mask_output)

    if is_cbz(input_path):
        print(f"Input Archive: {input_path}")
        tmp_in = extract_cbz(input_path)
        try:
            run(
                input_path=tmp_in,
                output_path=output_path,
                per_image=per_image,
                confidence=confidence,
                padding=padding,
                memory=memory,
                device=device,
                mask_output=mask_output,
            )
        finally:
            shutil.rmtree(tmp_in, ignore_errors=True)

    elif is_cbz(output_path):
        print(f"Output Archive: {output_path}")
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        try:
            run(
                input_path=input_path,
                output_path=tmp_out,
                per_image=per_image,
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
        if per_image and mask_output is not None:
            sys.exit("Error: Mask output is not supported in per-image mode.")
        print(f"Input Dir:  {input_path}\nOutput Dir: {output_path}")
        print(f"Mode: {'per-image' if per_image else 'shared mask'}")
        run_batch(
            input_path=input_path,
            output_path=output_path,
            detector=make_detector(confidence, padding, device),
            inpainter=make_inpainter(device),
            per_image=per_image,
            confidence=confidence,
            padding=padding,
            memory_mb=memory,
            device=device,
            mask_output=mask_output,
        )

    else:
        print(f"Input Image:  {input_path}\nOutput Image: {output_path}")
        run_image(input_path, output_path, mask_path=mask_output, confidence=confidence, padding=padding, device=device)
        print("Done.")
