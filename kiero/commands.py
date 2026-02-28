import shutil
import tempfile
from pathlib import Path

from kiero.batch import detect_batch, inpaint_batch, run_batch
from kiero.utils import (
    extract_cbz,
    is_cbz,
    load_mask,
    make_detector,
    make_inpainter,
    make_pipeline,
    validate,
    write_cbz,
)


def run(
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
    validate(input_path, output_path, mask_output=mask_output)

    if is_cbz(input_path):
        print(f"Input Archive: {input_path}")
        tmp_in = extract_cbz(input_path)
        try:
            run(
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

    elif is_cbz(output_path):
        print(f"Output Archive: {output_path}")
        tmp_out = Path(tempfile.mkdtemp(prefix="kiero_out_"))
        try:
            run(
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
        print(f"Input Dir:  {input_path}\nOutput Dir: {output_path}")
        print(f"Mode: {'per-image' if per_image else 'shared mask'}")
        if not per_image:
            print(f"Sample: {sample or 'all'}, confidence: {confidence}")
        run_batch(
            input_path=input_path,
            output_path=output_path,
            detector=make_detector(confidence, padding, device),
            inpainter=make_inpainter(device),
            per_image=per_image,
            sample_n=sample,
            confidence=confidence,
            padding=padding,
            memory_mb=memory,
            device=device,
            mask_output=mask_output,
        )

    else:
        print(f"Input Image:  {input_path}\nOutput Image: {output_path}")
        make_pipeline(confidence, padding, device).run(input_path, output_path, mask_path=mask_output)
        print("Done.")


def detect(
    input_path: Path,
    output_path: Path,
    sample: int | None,
    confidence: float,
    padding: int,
    memory: int,
    device: str | None,
):
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
        make_pipeline(confidence, padding, device).detect(input_path, output_path)
        print("Done.")


def inpaint(input_path: Path, output_path: Path, mask: Path, device: str | None):
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
            mask=load_mask(mask),
            inpainter=make_inpainter(device),
        )

    else:
        print(f"Input Image: {input_path}\nMask:  {mask}\nOutput Image: {output_path}")
        make_pipeline(0.25, 10, device).inpaint(input_path, output_path, mask)
        print("Done.")
