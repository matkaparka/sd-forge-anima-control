import base64
import hashlib
import io
import math

import cv2
import numpy as np
import torch
from PIL import Image

from lib_anima_control import ControlError

CROP_AND_RESIZE = "Crop and Resize"
JUST_RESIZE = "Just Resize"
RESIZE_AND_FILL = "Resize and Fill"
RESIZE_MODES = (CROP_AND_RESIZE, JUST_RESIZE, RESIZE_AND_FILL)

LATENT_F = 8  # Wan VAE spatial compression


def to_rgb_array(image) -> np.ndarray | None:
    """Gradio numpy / PIL / base64 (API) -> HxWx3 uint8."""
    if image is None:
        return None
    if isinstance(image, dict):  # gradio editor-style payloads
        image = image.get("image") or image.get("background") or image.get("composite")
        if image is None:
            return None
    if isinstance(image, str):
        if not image:
            return None
        if image.startswith("data:"):
            image = image.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(image)))
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def image_digest(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()[:12] + f"{image.shape}"


def resize_short_side(image: np.ndarray, short: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = short / min(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if (nw, nh) == (w, h):
        return image
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (nw, nh), interpolation=interp)


def canny(image: np.ndarray, low: int, high: int, detect_resolution: int) -> np.ndarray:
    """RGB -> gray -> cv2.Canny(low, high), white lines on black, 3 channels.

    The levzzz canny LoRA's training recipe (thresholds, resolution, polarity) is NOT published,
    so these are the most likely values, not facts. See README "Unconfirmed".
    """
    image = resize_short_side(image, int(detect_resolution))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, int(low), int(high))
    return np.stack([edges, edges, edges], axis=-1)


def fit_to(image: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    """Bring a control map to the exact target size (same semantics as ControlNet's resize modes)."""
    h, w = image.shape[:2]
    if (w, h) == (width, height):
        return image

    def _resize(img, nw, nh):
        interp = cv2.INTER_AREA if nw * nh < img.shape[0] * img.shape[1] else cv2.INTER_LINEAR
        return cv2.resize(img, (max(1, nw), max(1, nh)), interpolation=interp)

    if mode == JUST_RESIZE:
        return _resize(image, width, height)

    if mode == RESIZE_AND_FILL:
        scale = min(width / w, height / h)
        nw, nh = round(w * scale), round(h * scale)
        resized = _resize(image, nw, nh)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)  # black = "no signal" for canny / pose
        y0, x0 = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        return canvas

    # Crop and Resize
    scale = max(width / w, height / h)
    nw, nh = math.ceil(w * scale), math.ceil(h * scale)
    resized = _resize(image, nw, nh)
    y0, x0 = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
    return np.ascontiguousarray(resized[y0 : y0 + height, x0 : x0 + width])


@torch.inference_mode()
def encode_ref_latent(vae, control_map: np.ndarray) -> torch.Tensor:
    """Control map (H, W multiples of 8, uint8 RGB) -> reference latent in *sampler space*,
    shape (1, 16, 1, H/8, W/8), CPU float32.

    Same path as Forge's native Anima Edit (ForgeDiffusionEngine.encode_first_stage): VAE encode, then
    `process_in`. Deliberately does not call the engine's `encode_first_stage`, which would also stash
    the result in `ini_latent` / `ref_latents` when "[Anima] Enable Reference" is on.
    """
    h, w = control_map.shape[:2]
    if h % LATENT_F or w % LATENT_F:
        raise ValueError(f"control map {w}x{h} is not a multiple of {LATENT_F}")
    s = torch.from_numpy(control_map.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W, 3) in [0, 1]
    sample = vae.encode(s)  # batch of 1 -> (1, C, 1, h, w); a larger batch would be read as video frames
    latent = vae.first_stage_model.process_in(sample)
    if latent.ndim == 4:
        latent = latent.unsqueeze(2)
    if latent.shape[-2:] != (h // LATENT_F, w // LATENT_F):
        raise RuntimeError(f"VAE returned a {tuple(latent.shape[-2:])} latent for a {w}x{h} control map (expected {(h // LATENT_F, w // LATENT_F)})")
    return latent[:1].float().cpu()


@torch.inference_mode()
def encode_raw_latent(vae, control_map: np.ndarray) -> torch.Tensor:
    """Skeleton image (H, W multiples of 8) -> RAW Wan-VAE latent, shape (1, 16, 1, H/8, W/8), CPU float32.

    Deliberately NOT passed through `process_in`: the reference ComfyUI workflow of the Pose adapter feeds the
    plain VAEEncode output to the control embedder, so that is the space it was trained in. (Contrast with
    `encode_ref_latent`, which is sampler space.)
    """
    h, w = control_map.shape[:2]
    if h % LATENT_F or w % LATENT_F:
        raise ValueError(f"control map {w}x{h} is not a multiple of {LATENT_F}")
    s = torch.from_numpy(control_map.astype(np.float32) / 255.0).unsqueeze(0)  # (1, H, W, 3) in [0, 1]
    sample = vae.encode(s)
    if sample.ndim == 4:
        sample = sample.unsqueeze(2)
    return sample[:1].float().cpu()


def run_forge_preprocessor(name: str, image: np.ndarray, detect_resolution: int) -> np.ndarray:
    """Call one of Forge's registered ControlNet preprocessors (e.g. dw_openpose_full)."""
    from modules_forge.shared import supported_preprocessors

    pre = supported_preprocessors.get(name)
    if pre is None:
        raise ControlError(f'Preprocessor "{name}" is not available in this Forge install')
    out = pre(input_image=image, resolution=int(detect_resolution), slider_1=None, slider_2=None, slider_3=None)
    if isinstance(out, torch.Tensor):
        out = out.detach().cpu().numpy()
        if out.ndim == 4:
            out = out[0]
        if out.shape[0] in (1, 3) and out.shape[-1] not in (1, 3):
            out = np.moveaxis(out, 0, -1)
        if out.max() <= 1.0:
            out = out * 255.0
    return to_rgb_array(out)
