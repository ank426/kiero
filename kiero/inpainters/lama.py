import cv2
import numpy as np
from PIL import Image

from kiero.inpainters.base import Inpainter
from kiero.utils import bgr_to_pil

_MAX_DIM = 2048


class LamaInpainter(Inpainter):
    def __init__(self, device: str | None = None, max_dim: int = _MAX_DIM):
        self._device, self._max_dim, self._model = device, max_dim, None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from simple_lama_inpainting import SimpleLama

        dev = torch.device(self._device if self._device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = SimpleLama(device=dev)

    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load_model()
        assert self._model is not None
        h, w = image.shape[:2]
        img_in, mask_in = image, mask
        if needs_resize := max(h, w) > self._max_dim:
            sz = (int(w * (s := self._max_dim / max(h, w))), int(h * s))
            img_in = cv2.resize(image, sz, interpolation=cv2.INTER_AREA)
            mask_in = cv2.resize(mask, sz, interpolation=cv2.INTER_NEAREST)
        result_bgr = cv2.cvtColor(
            np.array(self._model(bgr_to_pil(img_in), Image.fromarray(mask_in))), cv2.COLOR_RGB2BGR
        )
        if not needs_resize:
            return result_bgr
        full = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        out = image.copy()
        out[mask > 127] = full[mask > 127]
        return out
