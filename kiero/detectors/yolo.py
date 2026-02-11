"""YOLO11x watermark detector.

Uses the pre-trained model `corzent/yolo11x_watermark_detection` from HuggingFace.
This model outputs bounding boxes around detected watermarks. We convert those
boxes into a binary mask for the inpainting stage.

Bounding boxes are inherently imprecise for this task — they include clean artwork
around the watermark. We provide a `padding` parameter to control how much extra
area around each detection is included in the mask.
"""

import numpy as np

from kiero.detectors.base import WatermarkDetector


class YoloDetector(WatermarkDetector):
    """YOLO11x based watermark detector."""

    MODEL_REPO = "corzent/yolo11x_watermark_detection"

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

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect watermark regions using YOLO11x."""
        self._load_model()

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Run inference
        results = self._model(
            image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        # Convert bounding boxes to mask
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Apply padding
                x1 = max(0, x1 - self._padding)
                y1 = max(0, y1 - self._padding)
                x2 = min(w, x2 + self._padding)
                y2 = min(h, y2 + self._padding)

                mask[y1:y2, x1:x2] = 255

        return mask
