"""CPU checks of the Canny (reference-latent) path against Forge Neo's own ``backend/nn/anima.py``.

Run with the Forge venv (random small weights, no GPU, no model files needed):

    <Forge Neo>\\venv\\Scripts\\python.exe tests\\test_reference.py

``FORGE_DIR`` overrides the Forge Neo checkout (default: three levels above this file).
``ANIMA_CANNY_FILE`` points at the real anima-preview-canny-v0.2.safetensors to enable the real-file checks.
"""
import logging
import os
import sys
from pathlib import Path

sys.argv = ["x", "--cpu"]  # backend.args parses sys.argv on import
EXT = Path(__file__).resolve().parents[1]
FORGE = Path(os.environ.get("FORGE_DIR") or EXT.parents[1])
assert (FORGE / "backend" / "nn" / "anima.py").exists(), f"set FORGE_DIR to your Forge Neo checkout (looked in {FORGE})"
sys.path.insert(0, str(FORGE))
sys.path.insert(0, str(FORGE / "modules_forge" / "packages"))  # vendored gguf / huggingface_guess, as modules_forge/initialization.py does
sys.path.insert(0, str(EXT))

import base64
import io

import numpy as np
import torch
from PIL import Image

import backend.operations as ops
from backend.args import dynamic_args
from backend.nn.anima import Anima
from lib_anima_control import imaging as IM
from lib_anima_control import loras as L
from lib_anima_control import patches as P

torch.manual_seed(0)
ok = 0


def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("ok  ", msg)


def raises(exc, fn, needle=""):
    try:
        fn()
    except exc as e:
        return needle in str(e)
    return False


def make_dit(blocks=2, dim=96, heads=2):
    with ops.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        dm = Anima(in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1, model_channels=dim, num_heads=heads, num_blocks=blocks, crossattn_emb_channels=32, adaln_lora_dim=16)
    dm.eval()
    with torch.no_grad():
        for p in dm.parameters():
            torch.nn.init.normal_(p, std=0.05)
    return dm


dm = make_dit()
ctx1 = torch.randn(1, 5, 32)


def run(x, t=0.5, ctx=None):
    n = x.shape[0]
    ctx = ctx1.repeat(n, 1, 1) if ctx is None else ctx
    with torch.no_grad():
        return dm(x.clone(), torch.full((n,), float(t)), ctx)


def ref_latent(h, w):
    return torch.randn(1, 16, 1, h, w)


x = torch.randn(1, 16, 1, 8, 12)
ref = ref_latent(8, 12)
base = run(x)
check(dynamic_args.ref_latents == [], "ref_latents starts empty")

# --- 1. wrapper: idle is a no-op, install is idempotent -----------------------------------
P.STATE.reset()
P.install_forward_wrapper(dm)
P.install_forward_wrapper(dm)
check(P._FWD_ATTR in dm.__dict__, "wrapper installed once on the instance")
check(torch.equal(run(x), base), "idle wrapper does not change the output")

# --- 2. armed == Forge's own native reference path -----------------------------------------
dynamic_args.ref_latents = [ref]
native = dm.__dict__[P._FWD_ATTR](x.clone(), torch.tensor([0.5]), ctx1.clone())  # the original forward, native refs
dynamic_args.ref_latents = []
check(not torch.allclose(native, base, atol=1e-4), "a reference changes the output (sanity)")

P.STATE.arm(ref)
armed = run(x)
check(torch.equal(armed, native), "armed wrapper reproduces the native ref_latents path exactly (batch 1)")
check(P.STATE.calls == 1 and dynamic_args.ref_latents == [], "one call counted; ref_latents empty again after the forward")

# --- 3. output shape: reference part is cropped away -----------------------------------------
check(armed.shape == base.shape == x.shape, "output shape equals the no-reference shape")
for h, w in ((7, 11), (9, 9), (5, 14)):
    xo = torch.randn(1, 16, 1, h, w)
    P.STATE.arm(ref_latent(h, w))
    yo = run(xo)
    P.STATE.reset()
    check(yo.shape == xo.shape == run(xo).shape, f"odd latent {h}x{w}: output shape unchanged by the reference")

# --- 4. batch 1 / 2 / >2 -------------------------------------------------------------------------
xb2 = torch.randn(2, 16, 1, 8, 12)
dynamic_args.ref_latents = [ref]
native_b2 = dm.__dict__[P._FWD_ATTR](xb2.clone(), torch.full((2,), 0.5), ctx1.repeat(2, 1, 1))
dynamic_args.ref_latents = []
P.STATE.arm(ref)
b2 = run(xb2)
check(torch.equal(b2, native_b2), "batch 2 (cond+uncond): wrapper == native, the reference is not duplicated twice")
single = [run(xb2[i : i + 1]) for i in range(2)]
check(all(torch.allclose(b2[i : i + 1], single[i], atol=1e-5) for i in range(2)), "batch 2 rows equal the batch-1 runs (same reference for every row)")

xb4 = torch.randn(4, 16, 1, 8, 12)
dynamic_args.ref_latents = [ref]
native_fails = raises(RuntimeError, lambda: dm.__dict__[P._FWD_ATTR](xb4.clone(), torch.full((4,), 0.5), ctx1.repeat(4, 1, 1)), "Sizes of tensors must match")
dynamic_args.ref_latents = []
check(native_fails, "NATIVE path fails for x.shape[0] == 4 (batch_size 2 with cond+uncond batched) -- the reason the wrapper sizes the reference")
b4 = run(xb4)
check(b4.shape == xb4.shape, "batch 4: wrapper runs")
single4 = torch.cat([run(xb4[i : i + 1]) for i in range(4)])
check(torch.allclose(b4, single4, atol=1e-5), "batch 4 rows equal the batch-1 runs")
xb3 = torch.randn(3, 16, 1, 8, 12)
b3 = run(xb3)
check(b3.shape == xb3.shape and torch.allclose(b3, torch.cat([run(xb3[i : i + 1]) for i in range(3)]), atol=1e-5), "batch 3 works too")
check(dynamic_args.ref_latents == [], "ref_latents empty after every batch size")
P.STATE.reset()

# --- 5. the reference shares the target's timestep ------------------------------------------------
seen = {}
h1 = dm.t_embedder[0].register_forward_hook(lambda m, i, o: seen.__setitem__("t_in", tuple(i[0].shape)))
h2 = dm.blocks[0].register_forward_pre_hook(lambda m, a, kw: seen.__setitem__("block", (tuple(a[0].shape), tuple(a[1].shape))), with_kwargs=True)
P.STATE.arm(ref)
run(x)
h1.remove()
h2.remove()
P.STATE.reset()
check(seen["t_in"] == (1, 1), "the timestep tensor is (B, 1): one value, not one per frame")
(xs, es) = seen["block"]
check(xs[1] == 2 and es[1] == 1, "block sees T=2 frames (target + reference) but a single (B, 1, D) timestep embedding -> broadcast to both = 'current t'")

# --- 6. Control End: dropping the reference mid-sampling == no reference --------------------------
P.STATE.arm(ref, end=0.5)
P.STATE.progress, P.STATE.progress_seen = 0.0, True
first = run(x)
P.STATE.progress = 0.49
last_on = run(x)
P.STATE.progress = 0.5
after = run(x)
P.STATE.progress = 0.99
later = run(x)
check(torch.equal(first, armed) and torch.equal(last_on, armed), "before Control End the reference is applied (steps < end)")
check(torch.equal(after, base) and torch.equal(later, base), "from Control End on, output is identical to a run without any reference")
check(P.STATE.calls == 2 and P.STATE.skipped == 2 and P.STATE.ended, "counters: 2 with reference, 2 without")
check(dynamic_args.ref_latents == [], "ref_latents empty throughout")
P.STATE.reset()
P.STATE.arm(ref, end=0.0)
P.STATE.progress_seen = True
check(torch.equal(run(x), base) and P.STATE.calls == 0, "end = 0 never applies the reference")
P.STATE.reset()

# progress reaches the state through the cfg_denoiser callback
class _Den:
    step, total_steps = 7, 20


class _Params:
    denoiser = _Den()
    sampling_step, total_sampling_steps = 3, 20


P.STATE.arm(ref, end=0.5)
P.on_cfg_denoiser(_Params())
check(abs(P.STATE.progress - 0.35) < 1e-9 and P.STATE.progress_seen, "progress comes from CFGDenoiser.step / total_steps (not the lagging state.sampling_step)")
P.STATE.reset()
P.on_cfg_denoiser(_Params())
check(P.STATE.progress == 0.0, "callback is inert while disarmed")

# end < 1 but no progress ever reported -> loud
records = []


class _Cap(logging.Handler):
    def emit(self, record):
        records.append(record)


cap = _Cap()
logging.getLogger("Anima Control").addHandler(cap)
P.STATE.arm(ref, end=0.5)
run(x)
check(any("Control End needs the sampler progress" in r.getMessage() for r in records), "missing progress is reported as an error instead of silently ignoring Control End")
P.STATE.reset()

# --- 7. failure paths ------------------------------------------------------------------------------
boom = dm.blocks[1].register_forward_pre_hook(lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")))
P.STATE.arm(ref)
check(raises(RuntimeError, lambda: run(x), "boom"), "an exception inside the model propagates")
check(dynamic_args.ref_latents == [], "ref_latents is empty after an exception inside the forward")
boom.remove()
P.STATE.reset()

P.STATE.arm(ref_latent(6, 6))
check(raises(RuntimeError, lambda: run(x), "must be encoded at exactly the generation size"), "a control latent of the wrong size fails loudly")
check(dynamic_args.ref_latents == [], "ref_latents empty after the size error")
P.STATE.reset()
check(raises(ValueError, lambda: P.STATE.arm(torch.randn(1, 16, 8, 12))), "arm() rejects a 4D reference")
check(raises(ValueError, lambda: P.STATE.arm(torch.randn(2, 16, 1, 8, 12))), "arm() rejects a batched reference")

# native reference already present: replaced for the call, restored after, and warned about
records.clear()
native_ref = [torch.randn(1, 16, 1, 8, 12)]
dynamic_args.ref_latents = native_ref
P.STATE.arm(ref)
via = run(x)
check(dynamic_args.ref_latents is native_ref and len(native_ref) == 1, "a pre-existing native reference is restored untouched afterwards")
check(torch.equal(via, armed), "the control latent wins during the call")
check(any("already holds Forge's own reference" in r.getMessage() for r in records), "the conflict is logged, not silent")
dynamic_args.ref_latents = []
P.STATE.reset()
logging.getLogger("Anima Control").removeHandler(cap)

# --- 8. memory estimate --------------------------------------------------------------------------------
class _KModel:
    def memory_required(self, shape):
        return 100.0


km = _KModel()
P.install_memory_patch(km)
P.install_memory_patch(km)
check(km.memory_required([1, 16, 1, 8, 8]) == 100.0, "memory estimate unchanged while idle")
P.STATE.arm(ref)
check(km.memory_required([1, 16, 1, 8, 8]) == 200.0, "memory estimate doubled while a reference is applied")
P.STATE.end, P.STATE.progress = 0.5, 0.9
check(km.memory_required([1, 16, 1, 8, 8]) == 100.0, "...and back to normal after Control End")
P.STATE.reset()

# --- 9. LoRA fade (online LoRA only) ---------------------------------------------------------------------
class _Lora:
    def __init__(self, name, w):
        self.name, self.patch = name, [[w, "data", 1.0, None, None]]


mine = [_Lora("C:/loras/canny.safetensors", 0.7) for _ in range(3)]
other = _Lora("C:/loras/style.safetensors", 0.5)


class _Unet:
    weight_wrapper_patches = {"a": [mine[0], other], "b": [mine[1], mine[0]], "c": [mine[2]]}


fade = P.LoraFade(_Unet(), "C:/loras/canny.safetensors")
check(fade.available(), "fade finds the control LoRA among the online patches")
check(not P.LoraFade(_Unet(), "C:/loras/missing.safetensors").available(), "fade reports a LoRA that is not patched in")
P.STATE.arm(ref, end=0.5, fade=fade)
P.STATE.progress, P.STATE.progress_seen = 0.1, True
run(x)
check(all(m.patch[0][0] == 0.7 for m in mine), "LoRA untouched while the control is on")
P.STATE.progress = 0.6
run(x)
check(all(m.patch[0][0] == 0.0 for m in mine), "LoRA strength is 0 after Control End")
check(other.patch[0][0] == 0.5, "other LoRAs are not touched")
P.STATE.reset()
check(all(m.patch[0][0] == 0.7 for m in mine), "reset() restores the original strength (a duplicated patch object is restored to its real value, not to 0)")
fade2 = P.LoraFade(_Unet(), "C:/loras/canny.safetensors")
fade2.drop()
fade2.restore()
fade2.restore()
check(all(m.patch[0][0] == 0.7 for m in mine), "restore() is idempotent")

# --- 10. imaging ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
photo = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
photo[20:70, 30:90] = 200  # a bright block: strong edges
e = IM.canny(photo, 100, 200, 64)
check(e.ndim == 3 and e.shape[-1] == 3 and e.dtype == np.uint8, "canny returns HxWx3 uint8")
check(min(e.shape[:2]) == 64, "canny works at the requested short side")
check(set(np.unique(e)) <= {0, 255} and (e == 255).any(), "canny is binary: white lines on black")
check(np.array_equal(e[..., 0], e[..., 1]) and np.array_equal(e[..., 1], e[..., 2]), "all three channels are identical")
flat = np.full((64, 64, 3), 127, np.uint8)
check(IM.canny(flat, 100, 200, 64).max() == 0, "a flat image has no edges")

for mode, (w, h) in [(IM.CROP_AND_RESIZE, (96, 64)), (IM.JUST_RESIZE, (96, 64)), (IM.RESIZE_AND_FILL, (96, 64)), (IM.CROP_AND_RESIZE, (64, 96))]:
    check(IM.fit_to(photo, w, h, mode).shape == (h, w, 3), f"fit_to {mode} -> {w}x{h}")
filled = IM.fit_to(np.full((40, 80, 3), 255, np.uint8), 80, 80, IM.RESIZE_AND_FILL)
check(filled[0].max() == 0 and filled[-1].max() == 0 and filled[40].min() == 255, "Resize and Fill pads with black (no signal), image centred")
check(np.array_equal(IM.fit_to(photo, 120, 90, IM.CROP_AND_RESIZE), photo), "fit_to is a no-op at the same size")

buf = io.BytesIO()
Image.fromarray(photo).save(buf, "PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
check(np.array_equal(IM.to_rgb_array(b64), photo) and np.array_equal(IM.to_rgb_array("data:image/png;base64," + b64), photo), "to_rgb_array reads API base64 (with and without data: prefix)")
check(IM.to_rgb_array(Image.fromarray(photo)).shape == photo.shape and IM.to_rgb_array(None) is None and IM.to_rgb_array("") is None, "to_rgb_array: PIL / None / empty")
check(IM.to_rgb_array(photo[..., 0]).shape == photo.shape and IM.to_rgb_array(np.dstack([photo, photo[..., :1]])).shape == photo.shape, "to_rgb_array: gray and RGBA")


class _FSM:
    @staticmethod
    def process_in(s):
        return s * 0.5 + 1.0


class _VAE:
    first_stage_model = _FSM()

    def encode(self, s):
        assert s.ndim == 4 and s.shape[0] == 1 and s.shape[-1] == 3 and 0.0 <= float(s.min()) and float(s.max()) <= 1.0
        return torch.full((1, 16, 1, s.shape[1] // 8, s.shape[2] // 8), float(s.mean()))


lat = IM.encode_ref_latent(_VAE(), np.full((64, 96, 3), 255, np.uint8))
check(tuple(lat.shape) == (1, 16, 1, 8, 12) and lat.dtype == torch.float32 and lat.device.type == "cpu", "encode_ref_latent -> (1, 16, 1, H/8, W/8), CPU float32")
check(torch.allclose(lat, torch.full_like(lat, 1.5)), "process_in is applied (sampler space, like native Anima Edit)")
check(raises(ValueError, lambda: IM.encode_ref_latent(_VAE(), np.zeros((63, 96, 3), np.uint8)), "multiple of 8"), "a size that is not a multiple of 8 is refused")


class _BadVAE(_VAE):
    def encode(self, s):
        return torch.zeros(1, 16, 1, 3, 3)


check(raises(RuntimeError, lambda: IM.encode_ref_latent(_BadVAE(), np.zeros((64, 96, 3), np.uint8)), "expected"), "a VAE that returns the wrong grid is caught")

# --- 11. LoRA tag injection ----------------------------------------------------------------------------------
check(L.append_tag("1girl", "canny", 0.7) == "1girl <lora:canny:0.7>", "tag appended")
check(L.append_tag("", "canny", 0.7) == "<lora:canny:0.7>", "tag on an empty prompt")
check(L.append_tag("1girl <lora:canny:0.3>", "canny", 0.7) == "1girl <lora:canny:0.3>", "user's own tag and weight kept")
check(L.append_tag("1girl <lora:CANNY>", "canny", 0.7) == "1girl <lora:CANNY>", "user's tag without weight / different case kept")
check(L.append_tag("1girl <lora:canny2:0.3>", "canny", 0.7).endswith("<lora:canny:0.7>"), "a LoRA whose name merely starts with this one does not count")
check(L.append_tag("1girl", "my lora (v2)", 1.0) == "1girl <lora:my lora (v2):1>", "names with spaces/brackets")


class _P:
    all_prompts = ["a", "b <lora:canny:0.2>"]
    all_hr_prompts = ["a hr", "b hr"]
    enable_hr = True


pp = _P()
L.inject_lora(pp, "canny", 0.7)
check(pp.all_prompts == ["a <lora:canny:0.7>", "b <lora:canny:0.2>"] and pp.all_hr_prompts == ["a hr <lora:canny:0.7>", "b hr <lora:canny:0.7>"], "inject_lora: prompts and Hires prompts")
pp = _P()
pp.enable_hr = False
L.inject_lora(pp, "canny", 0.7)
check(pp.all_hr_prompts == ["a hr", "b hr"], "Hires prompts untouched when Hires. fix is off")
check(L.guess_lora(["style", "Anima-Canny-v2", "pose"], ("anima", "canny")) == "Anima-Canny-v2", "guess_lora finds a name containing every word")
check(L.guess_lora(["NK2E-canny-v0.1", "style"], ("anima", "canny")) == L.NONE, "guess_lora does not pre-select Krea 2's canny LoRA for Anima")

# --- 12. the real Canny LoRA file (optional) ----------------------------------------------------------------
CANNY_FILE = os.environ.get("ANIMA_CANNY_FILE")
if CANNY_FILE and os.path.exists(CANNY_FILE):
    import json
    import re
    import struct

    with open(CANNY_FILE, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    keys = list(header)
    check(set(meta) <= {"format"}, f"the file's __metadata__ carries no training information ({meta})")
    check(not any("control_embedder" in k for k in keys), "a plain LoRA: no control_embedder (a reference-latent adapter, not a Pose-type one)")
    check({int(m.group(1)) for k in keys if (m := re.match(r"diffusion_model\.blocks\.(\d+)\.", k))} == set(range(28)), "targets all 28 DiT blocks")
    check({header[k]["shape"][0] for k in keys if k.endswith("lora_A.weight")} == {32}, "LoRA rank is 32")
    sd28 = make_dit(blocks=28, dim=96).state_dict()
    bases = {k[: -len(".lora_A.weight")] for k in keys if k.endswith("lora_A.weight")}
    missing = [b for b in bases if b[len("diffusion_model.") :] + ".weight" not in sd28]
    check(not missing and len(bases) == 448, f"all {len(bases)} LoRA targets exist in Forge's Anima ({missing[:3]})")
    check(not [k for k in keys if not k.startswith("diffusion_model.blocks.")], "no keys outside the DiT blocks (no llm_adapter, no embedder)")
else:
    print("skip real Canny LoRA checks (set ANIMA_CANNY_FILE)")

print(f"\nall {ok} checks passed")
