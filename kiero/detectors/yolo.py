"""YOLO11x watermark detector.

Uses the pre-trained model `corzent/yolo11x_watermark_detection` from HuggingFace.
This model outputs bounding boxes around detected watermarks. We convert those
boxes into a binary mask for the inpainting stage.

Bounding boxes are inherently imprecise for this task — they include clean artwork
around the watermark. We provide a `padding` parameter to control how much extra
area around each detection is included in the mask.

Supports batched inference via Ultralytics' native list input — a single GPU
forward pass processes multiple images simultaneously for higher throughput.
"""

import numpy as np

from kiero.detectors.base import WatermarkDetector


class YoloDetector(WatermarkDetector):
    """YOLO11x based watermark detector."""

    MODEL_REPO = "corzent/yolo11x_watermark_detection"
    default_batch_size = 4  # conservative for laptop GPUs (6-8 GB VRAM)

    def __init__(
        self,
        confidence: float = 0.25,
        padding: int = 10,
        device: str | None = None,
    ):
        """Initialize the YOLO detector.

        Args:
            confidence: Minimum confidence threshold for detections (0-1).
            padding: Extra pixels to add around each bounding box in the mask.
                Helps ensure full watermark coverage since bounding boxes may
                be tight.
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
        """
        self._confidence = confidence
        self._padding = padding
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return "yolo11x"

    def _load_model(self):
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        # Download the model weights from HuggingFace
        model_path = hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename="best.pt",
        )
        self._model = YOLO(model_path)

    def _results_to_mask(self, result, h: int, w: int) -> np.ndarray:
        """Convert a single YOLO result to a binary mask."""
        mask = np.zeros((h, w), dtype=np.uint8)
        if result.boxes is None:
            return mask
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            x1 = max(0, x1 - self._padding)
            y1 = max(0, y1 - self._padding)
            x2 = min(w, x2 + self._padding)
            y2 = min(h, y2 + self._padding)
            mask[y1:y2, x1:x2] = 255
        return mask

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect watermark regions using YOLO11x."""
        self._load_model()

        h, w = image.shape[:2]
        results = self._model(
            image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        return self._results_to_mask(results[0], h, w)

    def detect_batch(
        self, images: list[np.ndarray], batch_size: int | None = None
    ) -> list[np.ndarray]:
        """Batched detection using Ultralytics' native list input.

        Ultralytics YOLO accepts a list of images and processes them in a
        single batched GPU forward pass, which is significantly faster than
        calling detect() in a loop.
        """
        if not images:
            return []

        self._load_model()
        bs = batch_size or self.default_batch_size
        masks = []

        for chunk_start in range(0, len(images), bs):
            chunk = images[chunk_start : chunk_start + bs]

            # Ultralytics accepts a list of numpy arrays directly
            results = self._model(
                chunk,
                conf=self._confidence,
                device=self._device,
                verbose=False,
            )

            for img, result in zip(chunk, results):
                h, w = img.shape[:2]
                masks.append(self._results_to_mask(result, h, w))

        return masks
