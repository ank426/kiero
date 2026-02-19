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
        from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id=_MODEL_REPO, filename="best.pt")
        self._model = YOLO(model_path)

    def _run(self, source):
        self._load_model()
        assert self._model is not None
        return self._model(
            source,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

    def _to_mask(self, result, h: int, w: int) -> np.ndarray:
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
        h, w = image.shape[:2]
        return self._to_mask(self._run(image)[0], h, w)

    def detect_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not images:
            return []
        return [
            self._to_mask(r, img.shape[0], img.shape[1])
            for img, r in zip(images, self._run(images))
        ]
