"""CLIPSeg zero-shot watermark detector.

Uses the pre-trained CLIPSeg model (`CIDAS/clipseg-rd64-refined`) which can segment
image regions based on a text prompt. We prompt it with "watermark" (or a
user-specified prompt) and threshold the resulting heatmap into a binary mask.

CLIPSeg outputs at a fixed low resolution (~352x352) which is then upscaled to the
original image dimensions. This means fine watermark details may be lost, but it
provides reasonable region-level localization without any watermark-specific training.
"""

import numpy as np

from kiero.detectors.base import WatermarkDetector


class ClipSegDetector(WatermarkDetector):
    """CLIPSeg text-prompted watermark detector."""

    MODEL_NAME = "CIDAS/clipseg-rd64-refined"

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

        # outputs.logits shape: (1, 352, 352) — fixed resolution
        logits = outputs.logits[0]
        heatmap = torch.sigmoid(logits).cpu().numpy()

        # Resize heatmap to original image dimensions
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Threshold to binary mask
        mask = (heatmap_resized > self._threshold).astype(np.uint8) * 255

        # Dilate to cover edges
        if self._dilate_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._dilate_px * 2 + 1, self._dilate_px * 2 + 1),
            )
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask
