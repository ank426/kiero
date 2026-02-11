"""Inpainter registry."""

from kiero.inpainters.base import Inpainter
from kiero.inpainters.opencv import OpenCVInpainter
from kiero.inpainters.lama import LamaInpainter

INPAINTERS: dict[str, type[Inpainter]] = {
    "opencv": OpenCVInpainter,
    "lama": LamaInpainter,
}


def get_inpainter(name: str, **kwargs) -> Inpainter:
    """Instantiate an inpainter by name."""
    if name not in INPAINTERS:
        available = ", ".join(INPAINTERS.keys())
        raise ValueError(f"Unknown inpainter '{name}'. Available: {available}")
    return INPAINTERS[name](**kwargs)


__all__ = [
    "Inpainter",
    "INPAINTERS",
    "get_inpainter",
    "OpenCVInpainter",
    "LamaInpainter",
]
