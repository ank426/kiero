import cv2
import numpy as np
from PIL import Image

from kiero.inpainters.base import Inpainter
from kiero.utils import bgr_to_pil

_MAX_DIM = 2048


class LamaInpainter(Inpainter):
    def __init__(self, device: str | None = None, max_dim: int = _MAX_DIM):
        self._device = device
        self._max_dim = max_dim
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from simple_lama_inpainting import SimpleLama

        if self._device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self._device)
        self._model = SimpleLama(device=device)

    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load_model()
        assert self._model is not None

        h, w = image.shape[:2]
        needs_resize = max(h, w) > self._max_dim

        if needs_resize:
            scale = self._max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_small = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            mask_small = cv2.resize(
                mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            image_small = image
            mask_small = mask

        pil_image = bgr_to_pil(image_small)
        pil_mask = Image.fromarray(mask_small)
        result_pil = self._model(pil_image, pil_mask)

        result_bgr = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        if needs_resize:
            result_full = cv2.resize(
                result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4
            )
            output = image.copy()
            mask_bool = mask > 127
            output[mask_bool] = result_full[mask_bool]
            return output

        return result_bgr
