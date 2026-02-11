"""LaMa (Large Mask Inpainting) neural inpainter.

Uses the `simple-lama-inpainting` package which wraps the pre-trained LaMa model.
LaMa uses Fast Fourier Convolutions (FFC) for a global receptive field, making it
excellent at reconstructing periodic patterns like manga screentone.

The model was trained on natural images (Places2, CelebA-HQ) at 256x256 but
generalizes well to higher resolutions (~2k).

For very large images (>2048 on either dimension), we downscale before inpainting
and then composite the result back at original resolution to avoid OOM on laptop GPUs.
"""

import numpy as np

from kiero.inpainters.base import Inpainter

# Max dimension for LaMa inference. Images larger than this are downscaled.
_MAX_DIM = 2048


class LamaInpainter(Inpainter):
    """LaMa neural inpainter via simple-lama-inpainting."""

    def __init__(self, device: str | None = None, max_dim: int = _MAX_DIM):
        """Initialize the LaMa inpainter.

        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto).
            max_dim: Maximum dimension (height or width) for inference. Images
                larger than this are downscaled, inpainted, then the inpainted
                regions are upscaled and composited back.
        """
        self._device = device
        self._max_dim = max_dim
        self._model = None

    @property
    def name(self) -> str:
        return "lama"

    def _load_model(self):
        """Lazy-load the LaMa model on first use."""
        if self._model is not None:
            return
        import torch
        from simple_lama_inpainting import SimpleLama

        if self._device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self._device)
        self._model = SimpleLama(device=device)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint using LaMa.

        For large images, we:
        1. Downscale both image and mask to fit within max_dim
        2. Run LaMa on the downscaled version
        3. Upscale the inpainted result
        4. Composite: only replace pixels that were masked in the original
        """
        import cv2
        from PIL import Image

        self._load_model()

        h, w = image.shape[:2]
        needs_resize = max(h, w) > self._max_dim

        if needs_resize:
            scale = self._max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_small = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_AREA
            )
            mask_small = cv2.resize(
                mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
        else:
            image_small = image
            mask_small = mask

        # simple-lama-inpainting expects PIL images in RGB
        image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_mask = Image.fromarray(mask_small)

        # Run LaMa
        result_pil = self._model(pil_image, pil_mask)

        # Convert back to BGR numpy array
        result_np = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        if needs_resize:
            # Upscale the result back to original resolution
            result_full = cv2.resize(
                result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4
            )
            # Composite: only replace masked pixels
            output = image.copy()
            mask_bool = mask > 127
            output[mask_bool] = result_full[mask_bool]
            return output
        else:
            return result_bgr
