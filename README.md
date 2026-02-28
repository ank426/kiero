# Kiero

Manga watermark detector and remover. Uses YOLO11x for detection and LaMa for inpainting.

## Install

### Option A: Direct Installation (Recommended)

Install directly from GitHub (no clone required):

```bash
uv tool install git+https://github.com/ank426/kiero
```

### Option B: Manual Clone and Install

```bash
git clone https://github.com/ank426/kiero
cd kiero
uv tool install --editable .
```

Models are downloaded automatically on first use (~200 MB for YOLO, ~200 MB for LaMa).
Cached in `~/.cache/huggingface` and `~/.cache/torch`.

## Usage

Input can be a **single image**, a **directory of images**, or a **`.cbz` archive**.
Output format mirrors the input.

### Run (detect + inpaint)

```bash
kiero run input.png clean.png
kiero run imgs/ imgs_clean/
kiero run manga.cbz manga_clean.cbz
```

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

### Detect only

Generates a mask (white = watermark, black = clean).

> [!NOTE]
> Detection on a single image is less reliable; accuracy improves significantly
> when masks are averaged across many pages (e.g., a full chapter or volume).

```bash
kiero detect input.png mask.png
kiero detect imgs/ shared_mask.png --sample 10
```

### Inpaint only

Inpaints using your own mask (white = watermark, black = clean):

```bash
kiero inpaint -m mask.png input.png result.png
```

### Options

```
--confidence 0.25    # Detection threshold, also used for mask averaging (0-1)
--padding 0          # Pixels to pad detection boxes
--device cuda        # Force device (default: auto)
--per-image          # Detect per image instead of shared mask (run only)
--mask-output PATH   # Save shared mask (run only)
--sample N           # Sample N images for mask averaging (detect only; default: all)
--memory MB          # Memory budget for batch loading in MB (run/detect; default: 1024)
```
