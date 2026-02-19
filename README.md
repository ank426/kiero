# kiero

Manga watermark detector and remover. Uses YOLO11x for detection and LaMa for inpainting.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Models are downloaded automatically on first use (~200 MB for YOLO, ~200 MB for LaMa).
Cached in `~/.cache/huggingface` and `~/.cache/torch`.

## Usage

Input can be a **single image**, a **directory of images**, or a **`.cbz` archive**.
Output format mirrors the input.

### Full pipeline (detect + inpaint)

```bash
kiero run input.png -o clean.png
kiero run imgs/ -o imgs_clean/
kiero run manga.cbz -o manga_clean.cbz
```

Output path is optional — defaults to `<input_stem>_clean.<ext>`.

### Batch mode

When processing a directory or CBZ, kiero assumes all images share the same
watermark. It samples images, averages the masks, and applies the shared mask
to every image.

```bash
# Sample 10 images for mask averaging
kiero run imgs/ -o clean/ --sample 10

# Save the shared mask for inspection
kiero run imgs/ -o clean/ --sample 10 --mask-output shared_mask.png

# Per-image mode: detect independently per image (no averaging)
kiero run imgs/ -o clean/ --per-image
```

Batch size for GPU inference is configurable:

```bash
kiero run imgs/ -o clean/ --sample 20 --batch-size 8
```

### Detection only

```bash
kiero detect input.png -o mask.png
kiero detect imgs/ -o shared_mask.png --sample 10
```

### Inpainting only

Bring your own mask (white = watermark, black = clean):

```bash
kiero inpaint -m mask.png input.png -o result.png
```

### Options

```
--confidence 0.25    # Detection threshold, also used for mask averaging (0-1)
--padding 0          # Pixels to pad detection boxes
--device cuda        # Force device (default: auto)
--per-image          # Detect per image instead of shared mask
--sample N           # Sample N images for mask averaging (default: all)
--batch-size N       # GPU batch size for detection (default: 4)
```

## Project structure

```
kiero/
  __init__.py
  cli.py              # argparse CLI
  pipeline.py         # Single-image orchestration
  batch.py            # Batch processing (directory/CBZ), shared mask averaging
  utils.py            # Image I/O, mask stats
  detectors/
    __init__.py
    base.py            # ABC
    yolo.py            # YOLO11x detector
  inpainters/
    __init__.py
    base.py            # ABC with conform_mask template method
    lama.py            # LaMa neural inpainting
```
