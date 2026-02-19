"""YOLO11x watermark detector using ``corzent/yolo11x_watermark_detection``."""

import numpy as np

from kiero.detectors.base import WatermarkDetector

_MODEL_REPO = "corzent/yolo11x_watermark_detection"


class YoloDetector(WatermarkDetector):
    def __init__(
        self,
        confidence: float = 0.25,
        padding: int = 10,
        device: str | None = None,
    ):
        self._confidence = confidence
        self._padding = padding
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=_MODEL_REPO, filename="best.pt")
        self._model = YOLO(model_path)

    def _results_to_mask(self, result, h: int, w: int) -> np.ndarray:
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
        self._load_model()
        h, w = image.shape[:2]
        results = self._model(
            image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        return self._results_to_mask(results[0], h, w)

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not images:
            return []

        self._load_model()
        results = self._model(
            images,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        return [
            self._results_to_mask(result, img.shape[0], img.shape[1])
            for img, result in zip(images, results)
        ]
