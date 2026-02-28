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
kiero run input.png clean.png
kiero run imgs/ imgs_clean/
kiero run manga.cbz manga_clean.cbz
```

### Batch mode

When processing a directory or CBZ, kiero assumes all images share the same
watermark. It averages the masks and applies the shared mask to every image.

```bash
# Save the shared mask for inspection
kiero run imgs/ clean/ --mask-output shared_mask.png

# Per-image mode: detect independently per image (no averaging)
kiero run imgs/ clean/ --per-image
```

Memory budget controls how many images are loaded and sent to the GPU at once:

```bash
kiero run imgs/ clean/ --memory 2048
```

### Detection only

```bash
kiero detect input.png mask.png
kiero detect imgs/ shared_mask.png --sample 10
```

### Inpainting only

Bring your own mask (white = watermark, black = clean):

```bash
kiero inpaint -m mask.png input.png result.png
```

### Options

```
--confidence 0.25    # Detection threshold, also used for mask averaging (0-1)
--padding 0          # Pixels to pad detection boxes
--device cuda        # Force device (default: auto)
--per-image          # Detect per image instead of shared mask (run only)
--sample N           # Sample N images for mask averaging (detect only; default: all)
--memory MB          # Memory budget for batch loading in MB (run/detect; default: 1024)
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
