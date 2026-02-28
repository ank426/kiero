import numpy as np
from typing import TYPE_CHECKING

from kiero.detectors.base import WatermarkDetector

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

class YoloDetector(WatermarkDetector):
    _MODEL_REPO = "corzent/yolo11x_watermark_detection"

    def __init__(self, confidence: float = 0.25, padding: int = 10, device: str | None = None) -> None:
        self._confidence, self._padding, self._device = confidence, padding, device
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
        from huggingface_hub import hf_hub_download

        self._model = YOLO(hf_hub_download(repo_id=self._MODEL_REPO, filename="best.pt"))

    def _run(self, source: np.ndarray | list[np.ndarray]) -> list["Results"]:
        self._load_model()
        assert self._model is not None
        return self._model(source, conf=self._confidence, device=self._device, verbose=False)

    def _to_mask(self, results: list["Results"], h: int, w: int, n: int) -> np.ndarray:
        masks = np.zeros((n, h, w), dtype=np.uint8)
        pad = self._padding

        for i, result in enumerate(results):
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                masks[i, max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)] = 255

        return masks.astype(np.float32).mean(axis=0)

    def detect(self, image: np.ndarray) -> np.ndarray:
        return self._to_mask(self._run(image), *image.shape[:2], 1)

    def detect_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.array([])
        return self._to_mask(self._run(images), *images[0].shape[:2], len(images))
