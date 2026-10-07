"""Control embedder of Anima Control Pose (OpenPose mode). Pure functions; the hook that applies it is in patches.py.

Adapted from sd-forge-anima-pose (lib_anima_pose/control.py).

The Pose adapter is a channel-concat control-LoRA. Two parts live in one safetensors file:
  * LoRA deltas (`diffusion_model.*.lora_A/B.weight`)  -> loaded by Forge's own LoRA system
  * `diffusion_model.control_embedder.proj.{weight,bias}` -> handled here

The control embedder is a Linear(64 -> model_channels) over 2x2 patches of the 16-channel VAE latent of
the skeleton image. Its output is added to the output of the DiT's `x_embedder` (patch embed), i.e. before
the first transformer block. At strength 0 the model is unchanged.
"""
import os

import torch
import torch.nn.functional as F
from einops import rearrange

from lib_anima_control import ControlError

PATCH = 2
EMBEDDER_KEY = "control_embedder."

_embedder_cache: dict = {}


def load_embedder(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read control_embedder.proj.{weight,bias} (float32, CPU) from the adapter file."""
    from safetensors import safe_open

    stamp = (path, os.path.getmtime(path))
    hit = _embedder_cache.get(stamp)
    if hit is not None:
        return hit

    found = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if EMBEDDER_KEY in key:
                found[key.split(EMBEDDER_KEY, 1)[1]] = f.get_tensor(key).float()

    if "proj.weight" not in found or "proj.bias" not in found:
        raise ControlError(f"{os.path.basename(path)} has no control_embedder.proj.* keys. This file is a plain LoRA, not an Anima Control Pose adapter (expected e.g. anima_pose_preview2.safetensors). Without the embedder the control would silently do nothing.")

    weight, bias = found["proj.weight"], found["proj.bias"]
    if weight.shape[1] != 16 * PATCH * PATCH:
        raise ControlError(f"Unexpected control embedder shape {tuple(weight.shape)} (expected [model_channels, 64])")

    _embedder_cache.clear()
    _embedder_cache[stamp] = (weight, bias)
    return weight, bias


def embed(control_latent: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, strength: float) -> torch.Tensor:
    """(1, 16, T, H, W) raw VAE latent -> control tokens (1, T, H/2, W/2, D), float32."""
    if control_latent.ndim == 4:
        control_latent = control_latent.unsqueeze(2)
    x = control_latent.float()
    _, _, _, h, w = x.shape
    ph, pw = (-h) % PATCH, (-w) % PATCH
    if ph or pw:  # same right/bottom circular padding the DiT applies to its own input
        x = F.pad(x, (0, pw, 0, ph, 0, 0), mode="circular")
    patches = rearrange(x, "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=PATCH, n=PATCH)
    return strength * F.linear(patches, weight, bias)
