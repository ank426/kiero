"""Watermark detector registry."""

from kiero.detectors.base import WatermarkDetector
from kiero.detectors.template import TemplateDetector
from kiero.detectors.yolo import YoloDetector
from kiero.detectors.clipseg import ClipSegDetector

DETECTORS: dict[str, type[WatermarkDetector]] = {
    "template": TemplateDetector,
    "yolo11x": YoloDetector,
    "clipseg": ClipSegDetector,
}


def get_detector(name: str, **kwargs) -> WatermarkDetector:
    """Instantiate a detector by name."""
    if name not in DETECTORS:
        available = ", ".join(DETECTORS.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")
    return DETECTORS[name](**kwargs)


__all__ = [
    "WatermarkDetector",
    "DETECTORS",
    "get_detector",
    "TemplateDetector",
    "YoloDetector",
    "ClipSegDetector",
]
