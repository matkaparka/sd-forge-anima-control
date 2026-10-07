"""Every runtime patch this extension applies to Forge Neo lives in this file. Nothing under backend/ is edited.

What is patched (see README "What is patched"):

1. ``diffusion_model.forward`` (instance attribute, once per model object; inert unless armed).
   While the wrapped call runs, ``dynamic_args.ref_latents`` is set to the control latent so that the
   *native* reference code in ``backend/nn/anima.py`` concatenates it along T. The previous value is
   restored in ``finally``, so ``ref_latents`` is empty outside a forward even if sampling raises.
   The wrapper also sizes the reference to the batch: the native code only duplicates a reference when
   ``x.shape[0] == 2`` (one cond + one uncond), so batch_size >= 2 would otherwise fail in ``torch.cat``.
2. ``KModel.memory_required`` (instance attribute): multiplied while a reference is active, because the
   sequence is longer. Only affects whether Forge batches cond and uncond together.
3. Online LoRA weights (``OnlineLoRAPatch.patch[0][0]``): zeroed after Control End and restored afterwards.
   Only possible when Forge patches LoRAs on the fly ("Diffusion in Low Bits" = "... (fp16 LoRA)").
4. A ``cfg_denoiser`` callback records ``CFGDenoiser.step / total_steps`` for Control End.
5. ``diffusion_model.x_embedder`` forward hook (OpenPose mode only): adds the control tokens to the patch
   embedding. One persistent hook per embedder, a no-op unless armed.
"""
import logging

import torch

from backend.args import dynamic_args

logger = logging.getLogger("Anima Control")

LATENT_DIMS = 5  # Anima latents are (B, C, T, H, W)


# ---------------------------------------------------------------------------
# sampling progress
# ---------------------------------------------------------------------------


def on_cfg_denoiser(params) -> None:
    """Registered with ``script_callbacks.on_cfg_denoiser`` by the script. Runs right before each model call."""
    if not STATE.active:
        return
    d = getattr(params, "denoiser", None)
    steps = getattr(d, "total_steps", None)
    if d is not None and steps:
        STATE.progress = d.step / steps
    elif params.total_sampling_steps:
        STATE.progress = params.sampling_step / params.total_sampling_steps
    else:
        return
    STATE.progress_seen = True


# ---------------------------------------------------------------------------
# online LoRA fade
# ---------------------------------------------------------------------------


class LoraFade:
    """Switch one LoRA off mid-sampling by zeroing its online patch strength, and put it back afterwards."""

    def __init__(self, unet, filename: str):
        self.unet = unet
        self.filename = filename
        self._saved: dict[int, tuple[object, float]] = {}

    def _patches(self):
        wrappers = getattr(self.unet, "weight_wrapper_patches", None) or {}
        for lora_list in wrappers.values():
            for lora in lora_list:
                if getattr(lora, "name", None) == self.filename:
                    yield lora

    def available(self) -> bool:
        return any(True for _ in self._patches())

    def drop(self) -> int:
        n = 0
        for lora in self._patches():
            if id(lora) not in self._saved:
                self._saved[id(lora)] = (lora, lora.patch[0][0])
            lora.patch[0][0] = 0.0
            n += 1
        return n

    def restore(self) -> None:
        for lora, strength in self._saved.values():
            lora.patch[0][0] = strength
        self._saved.clear()


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------


class _State:
    def __init__(self):
        self.fade: LoraFade | None = None
        self.reset()

    def reset(self):
        if self.fade is not None:
            self.fade.restore()
        self.fade = None
        self.active = False
        self.ref: torch.Tensor | None = None  # (1, C, 1, h, w), CPU float32
        self.end = 1.0  # fraction of the sampling steps during which the control is applied
        self.stage = "base"
        self.calls = 0  # forwards that received the reference
        self.skipped = 0  # forwards after Control End
        self.ended = False
        self.progress = 0.0
        self.progress_seen = False
        self.warned_native = False
        self._placed: dict = {}

    def arm(self, ref: torch.Tensor, end: float = 1.0, fade: LoraFade | None = None, stage: str = "base"):
        self.reset()
        if ref.ndim != LATENT_DIMS or ref.shape[0] != 1 or ref.shape[2] != 1:
            raise ValueError(f"reference latent must be (1, C, 1, h, w), got {tuple(ref.shape)}")
        self.active = True
        self.ref = ref
        self.end = float(end)
        self.fade = fade
        self.stage = stage

    def in_range(self) -> bool:
        if self.end >= 1.0:
            return True
        if self.end <= 0.0:
            return False
        return self.progress < self.end

    def ref_for(self, x: torch.Tensor) -> torch.Tensor:
        """The reference as `Anima.forward` wants it for this input batch."""
        if tuple(x.shape[-2:]) != tuple(self.ref.shape[-2:]):
            raise RuntimeError(f"Anima Control: the control latent is {tuple(self.ref.shape[-2:])} but the sampling latent is {tuple(x.shape[-2:])}. The control must be encoded at exactly the generation size.")
        key = (x.device, x.dtype)
        placed = self._placed.get(key)
        if placed is None:
            placed = self.ref.to(device=x.device, dtype=x.dtype)
            self._placed = {key: placed}
        n = x.shape[0]
        if n <= 2:
            # n == 2 is cond + uncond: the model duplicates a batch-1 reference itself, do not do it twice
            return placed
        return placed.expand(n, -1, -1, -1, -1)


STATE = _State()


# ---------------------------------------------------------------------------
# patch 1: diffusion_model.forward
# ---------------------------------------------------------------------------

_FWD_ATTR = "_anima_control_original_forward"


def install_forward_wrapper(dm) -> None:
    if _FWD_ATTR in dm.__dict__:
        return
    original = dm.forward

    def forward(x, *args, **kwargs):
        st = STATE
        if not st.active or st.ref is None:
            return original(x, *args, **kwargs)

        if st.end < 1.0 and not st.progress_seen:
            logger.error("Anima Control: Control End needs the sampler progress, but it was not reported -- the control stays on for ALL steps")
            st.progress_seen = True  # say it once

        if not st.in_range():
            st.skipped += 1
            if not st.ended:
                st.ended = True
                note = ""
                if st.fade is not None:
                    note = f", LoRA switched off ({st.fade.drop()} patches)"
                logger.info(f"Anima Control [{st.stage}]: Control End reached at {st.progress:.0%} of the steps -- reference dropped{note}")
            return original(x, *args, **kwargs)

        ref = st.ref_for(x)
        previous = dynamic_args.ref_latents
        if previous and not st.warned_native:
            logger.warning("Anima Control: dynamic_args.ref_latents already holds Forge's own reference (is '[Anima] Enable Reference' on, or ImageStitch?). It is replaced by the control latent for this generation.")
            st.warned_native = True
        dynamic_args.ref_latents = [ref]
        st.calls += 1
        try:
            return original(x, *args, **kwargs)
        finally:
            dynamic_args.ref_latents = previous

    dm.__dict__[_FWD_ATTR] = original
    dm.forward = forward


# ---------------------------------------------------------------------------
# patch 2: KModel.memory_required
# ---------------------------------------------------------------------------

_MEM_ATTR = "_anima_control_original_memory_required"


def install_memory_patch(kmodel) -> None:
    if _MEM_ATTR in kmodel.__dict__ or not hasattr(kmodel, "memory_required"):
        return
    original = kmodel.memory_required

    def memory_required(input_shape):
        need = original(input_shape)
        if STATE.active and STATE.ref is not None and STATE.in_range():
            return need * (1 + STATE.ref.shape[2])  # one extra frame of tokens per reference
        return need

    kmodel.__dict__[_MEM_ATTR] = original
    kmodel.memory_required = memory_required


# ---------------------------------------------------------------------------
# patch 5: x_embedder forward hook (OpenPose / control embedder)
# ---------------------------------------------------------------------------


class _EmbedState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.tokens: torch.Tensor | None = None
        self.calls = 0
        self.stage = "base"
        self.end = 1.0  # no Control End for this mode; kept so both states can be inspected alike
        self._placed: dict = {}

    def arm(self, tokens: torch.Tensor, stage: str = "base"):
        self.reset()
        self.active = True
        self.tokens = tokens
        self.stage = stage

    def placed(self, device, dtype) -> torch.Tensor:
        key = (device, dtype)
        t = self._placed.get(key)
        if t is None:
            t = self.tokens.to(device=device, dtype=dtype)
            self._placed = {key: t}
        return t


EMBED = _EmbedState()


def _embed_hook(module, inputs, output):
    st = EMBED
    if not st.active or st.tokens is None:
        return output

    tokens = st.placed(output.device, output.dtype)
    if tokens.shape[2:4] != output.shape[2:4]:
        raise RuntimeError(f"Anima Control: control grid {tuple(tokens.shape[2:4])} does not match the latent token grid {tuple(output.shape[2:4])}")

    st.calls += 1
    t = tokens.shape[1]
    if output.shape[1] == t:
        return output + tokens
    # extra frames in the sequence (ImageStitch / reference latents): the control belongs to the target frame only
    output = output.clone()
    output[:, :t] += tokens
    return output


def install_embedder_hook(dm) -> None:
    embedder = dm.x_embedder
    if not getattr(embedder, "_anima_control_hooked", False):
        embedder.register_forward_hook(_embed_hook)
        embedder._anima_control_hooked = True


def reset_all() -> None:
    """Disarm every patch (the reference one also puts a faded LoRA back)."""
    STATE.reset()
    EMBED.reset()
