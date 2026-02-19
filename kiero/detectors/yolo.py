import numpy as np

from kiero.detectors.base import WatermarkDetector

_MODEL_REPO = "corzent/yolo11x_watermark_detection"


class YoloDetector(WatermarkDetector):
    def __init__(self, confidence: float = 0.25, padding: int = 10, device: str | None = None):
        self._confidence, self._padding, self._device = confidence, padding, device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
        from huggingface_hub import hf_hub_download

        self._model = YOLO(hf_hub_download(repo_id=_MODEL_REPO, filename="best.pt"))

    def _run(self, source):
        self._load_model()
        assert self._model is not None
        return self._model(source, conf=self._confidence, device=self._device, verbose=False)

    def _to_mask(self, result, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if result.boxes is None:
            return mask
        pad = self._padding
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask[max(0, y1 - pad) : min(h, y2 + pad), max(0, x1 - pad) : min(w, x2 + pad)] = 255
        return mask

    def detect(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return self._to_mask(self._run(image)[0], h, w)

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not images:
            return []
        return [self._to_mask(r, img.shape[0], img.shape[1]) for img, r in zip(images, self._run(images))]
