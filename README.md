# kiero

Manga watermark detector and remover. Modular pipeline that pairs swappable
**detectors** (locate the watermark) with swappable **inpainters** (fill it in).

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Models are downloaded automatically on first use (~200 MB for YOLO, ~400 MB for
CLIPSeg, ~200 MB for LaMa). They are cached in `~/.cache/huggingface` and
`~/.cache/torch`.

## Usage

Input can be a **single image**, a **directory of images**, or a **`.cbz` archive**.
Output format mirrors the input: directory in -> directory out, CBZ in -> CBZ out.

### Full pipeline (detect + inpaint)

```bash
# Single image
kiero run -d yolo11x -i lama input.png -o clean.png

# Directory of images
kiero run imgs/ -o imgs_clean/

# CBZ archive
kiero run manga.cbz -o manga_clean.cbz
```

Default detector is `yolo11x`, default inpainter is `lama`. Output path is
optional and defaults to `<input_stem>_clean.<ext>` (or `<dir>_clean/`).

### Batch mode: shared mask vs per-image

When processing a directory or CBZ, kiero assumes all images share the same
watermark. It runs detection on a sample of images, averages the masks, and
applies the resulting shared mask to every image. This is faster and filters
out false positives (a pixel must be flagged in multiple images to survive
averaging).

```bash
# Default: shared mask from all images
kiero run imgs/ -o clean/

# Sample only 10 images for mask averaging (much faster)
kiero run imgs/ -o clean/ --sample 10

# Require 30% agreement instead of default 50%
kiero run imgs/ -o clean/ --sample 10 --mask-threshold 0.3

# Save the shared mask for inspection
kiero run imgs/ -o clean/ --sample 10 --mask-output shared_mask.png

# Per-image mode: detect independently per image (no averaging)
kiero run imgs/ -o clean/ --per-image
```

YOLO and CLIPSeg detectors use **batched GPU inference** during shared mask
computation — multiple images are processed in a single forward pass for
higher throughput. Batch size is configurable:

```bash
kiero run imgs/ -o clean/ --sample 20 --batch-size 8
```

### Detection only

Save the binary mask for inspection or manual editing:

```bash
# Single image
kiero detect -d clipseg input.png -o mask.png

# Batch: outputs the averaged shared mask
kiero detect imgs/ -o shared_mask.png --sample 10
```

### Inpainting only

Bring your own mask (white = watermark, black = clean):

```bash
kiero inpaint -i lama -m mask.png input.png -o result.png
```

### Compare all detectors

Run every registered detector on the same image, inpaint each result, and
produce a side-by-side comparison grid:

```bash
kiero compare input.png -o comparison_output/
```

This saves per-detector masks (`mask_<name>.png`), per-detector results
(`result_<name>.png`), and a `comparison.png` grid image.

## Detectors

| Name | Type | Description |
|---|---|---|
| `yolo11x` | Bounding box | [corzent/yolo11x_watermark_detection](https://huggingface.co/corzent/yolo11x_watermark_detection). Best published metrics (mAP@50 = 0.90). Outputs bounding boxes converted to masks. |
| `clipseg` | Heatmap | [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined). Zero-shot text-prompted segmentation. No watermark-specific training. Configurable prompt via `--prompt`. |
| `template` | Template / Heuristic | OpenCV `matchTemplate` when given `--template path.png`. Falls back to a heuristic mode (local contrast + frequency analysis) when no template is provided. |

### Detector options

```
--confidence 0.25    # YOLO detection threshold (0-1)
--prompt "watermark"  # CLIPSeg text prompt
--template crop.png  # Template image for template detector
--det-threshold 0.3  # Heatmap/matching threshold
--dilate 10          # Pixels to dilate the mask by
--device cuda        # Force device (default: auto)
```

## Inpainters

| Name | Type | Description |
|---|---|---|
| `lama` | Neural | [LaMa](https://github.com/advimman/lama) via `simple-lama-inpainting`. Uses Fast Fourier Convolutions for a global receptive field — good at reconstructing periodic patterns like screentone. Auto-downscales images larger than 2048px. |
| `opencv` | Classical | `cv2.inpaint()` with Telea or Navier-Stokes method. Very fast, no GPU needed. Useful as a baseline. |

### Inpainter options

```
--inp-method telea   # OpenCV method: "telea" or "ns"
--inp-radius 3       # OpenCV neighborhood radius
```

### Batch options

```
--per-image          # Detect per image instead of shared mask
--sample N           # Sample N images for mask averaging (default: all)
--mask-threshold 0.5 # Agreement fraction for shared mask (default: 0.5)
--batch-size N       # GPU batch size (default: 4 for YOLO, 16 for CLIPSeg)
```

## Adding a new detector or inpainter

1. Create a new file in `kiero/detectors/` (or `kiero/inpainters/`).
2. Implement the `WatermarkDetector` (or `Inpainter`) abstract base class.
3. Add one entry to the `DETECTORS` (or `INPAINTERS`) dict in the package
   `__init__.py`.

The new model is immediately available via `--detector`/`--inpainter` flags and
is included in `compare` mode.

### Detector interface

```python
class WatermarkDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> np.ndarray:
        """BGR image in, (H, W) uint8 mask out. 255 = watermark, 0 = clean."""
        ...

    def detect_batch(self, images: list[np.ndarray], batch_size: int | None = None) -> list[np.ndarray]:
        """Optional: override for GPU-batched detection."""
        return [self.detect(img) for img in images]
```

### Inpainter interface

```python
class Inpainter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """BGR image + uint8 mask in, BGR image out."""
        ...
```

## Project structure

```
kiero/
  __init__.py
  cli.py              # argparse CLI
  pipeline.py         # Single-image orchestration + compare
  batch.py            # Batch processing (directory/CBZ), shared mask averaging
  utils.py            # Image I/O, mask overlay, comparison grid
  detectors/
    __init__.py        # Registry
    base.py            # ABC (detect + detect_batch)
    template.py        # OpenCV template matching / heuristic
    yolo.py            # YOLO11x (batched GPU inference)
    clipseg.py         # CLIPSeg (batched GPU inference)
  inpainters/
    __init__.py        # Registry
    base.py            # ABC
    opencv.py          # Classical inpainting
    lama.py            # LaMa neural inpainting
```
