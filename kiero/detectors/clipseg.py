"""CLIPSeg zero-shot watermark detector.

Uses the pre-trained CLIPSeg model (`CIDAS/clipseg-rd64-refined`) which can segment
image regions based on a text prompt. We prompt it with "watermark" (or a
user-specified prompt) and threshold the resulting heatmap into a binary mask.

CLIPSeg outputs at a fixed low resolution (~352x352) which is then upscaled to the
original image dimensions. This means fine watermark details may be lost, but it
provides reasonable region-level localization without any watermark-specific training.

Supports batched inference — multiple images are stacked into a single tensor and
processed in one GPU forward pass for higher throughput.
"""

import numpy as np

from kiero.detectors.base import WatermarkDetector


class ClipSegDetector(WatermarkDetector):
    """CLIPSeg text-prompted watermark detector."""

    MODEL_NAME = "CIDAS/clipseg-rd64-refined"
    default_batch_size = 16  # CLIPSeg processes at 352x352, very light per image

    def __init__(
        self,
        prompt: str = "watermark",
        threshold: float = 0.3,
        dilate_px: int = 10,
        device: str | None = None,
    ):
        """Initialize the CLIPSeg detector.

        Args:
            prompt: Text prompt describing what to detect. Default "watermark".
                Can be customized, e.g. "text overlay", "semi-transparent logo",
                "watermark text".
            threshold: Threshold for converting the heatmap to a binary mask.
                The heatmap values are sigmoid outputs (0-1). Pixels above this
                threshold are marked as watermark. Lower = more aggressive
                detection (more false positives).
            dilate_px: Pixels to dilate the mask by after thresholding.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        self._prompt = prompt
        self._threshold = threshold
        self._dilate_px = dilate_px
        self._device = device
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return f'clipseg("{self._prompt}")'

    def _load_model(self):
        """Lazy-load the CLIPSeg model on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

        self._processor = CLIPSegProcessor.from_pretrained(self.MODEL_NAME)
        self._model = CLIPSegForImageSegmentation.from_pretrained(self.MODEL_NAME)

        # Determine device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

    def _heatmap_to_mask(self, heatmap: np.ndarray, h: int, w: int) -> np.ndarray:
        """Convert a single CLIPSeg heatmap to a binary mask at target resolution."""
        import cv2

        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (heatmap_resized > self._threshold).astype(np.uint8) * 255

        if self._dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._dilate_px * 2 + 1, self._dilate_px * 2 + 1),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect watermark regions using CLIPSeg."""
        import cv2
        import torch
        from PIL import Image

        self._load_model()

        h, w = image.shape[:2]

        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Run CLIPSeg
        inputs = self._processor(
            text=[self._prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        heatmap = torch.sigmoid(outputs.logits[0]).cpu().numpy()
        return self._heatmap_to_mask(heatmap, h, w)

    def detect_batch(
        self, images: list[np.ndarray], batch_size: int | None = None
    ) -> list[np.ndarray]:
        """Batched detection using CLIPSeg.

        Stacks multiple images into a single tensor with repeated text prompts
        and runs one forward pass per chunk.
        """
        if not images:
            return []

        import cv2
        import torch
        from PIL import Image

        self._load_model()
        bs = batch_size or self.default_batch_size
        masks = []

        for chunk_start in range(0, len(images), bs):
            chunk = images[chunk_start : chunk_start + bs]
            n = len(chunk)

            # Convert BGR numpy arrays to PIL RGB
            pil_images = []
            sizes = []  # (h, w) per image for mask resizing
            for img in chunk:
                sizes.append(img.shape[:2])
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))

            # Repeat prompt for each image in the batch
            inputs = self._processor(
                text=[self._prompt] * n,
                images=pil_images,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # outputs.logits shape: (n, 352, 352)
            logits = outputs.logits
            heatmaps = torch.sigmoid(logits).cpu().numpy()

            for i in range(n):
                h, w = sizes[i]
                masks.append(self._heatmap_to_mask(heatmaps[i], h, w))

        return masks
