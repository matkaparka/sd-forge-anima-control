"""CPU checks of the OpenPose (control embedder) path against Forge Neo's own Anima DiT, and optionally the real adapter file.

Ported from sd-forge-anima-pose (tests/test_control.py) onto lib_anima_control.

    <Forge Neo>\\venv\\Scripts\\python.exe tests\\test_pose.py

``FORGE_DIR`` overrides the Forge Neo checkout (default: three levels above this file).
``ANIMA_POSE_FILE`` points at the real adapter (e.g. anima_pose_preview2.safetensors) to enable the real-file checks.
"""
import json
import os
import re
import struct
import sys
from pathlib import Path

sys.argv = ["x", "--cpu"]
EXT = Path(__file__).resolve().parents[1]
FORGE = Path(os.environ.get("FORGE_DIR") or EXT.parents[1])
assert (FORGE / "backend" / "nn" / "anima.py").exists(), f"set FORGE_DIR to your Forge Neo checkout (looked in {FORGE})"
sys.path.insert(0, str(FORGE))
sys.path.insert(0, str(FORGE / "modules_forge" / "packages"))  # vendored gguf, as modules_forge/initialization.py does
sys.path.insert(0, str(EXT))

import numpy as np
import torch

import backend.operations as ops
from backend.nn.anima import Anima, LLMAdapter
from lib_anima_control import embedder as C
from lib_anima_control import imaging as IM
from lib_anima_control import patches as PT
from lib_anima_control import pose as P

ADAPTER = os.environ.get("ANIMA_POSE_FILE")
torch.manual_seed(0)
ok = 0


def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("ok  ", msg)


def make_dit(blocks=2, dim=96, heads=2):
    with ops.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        dm = Anima(in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1, model_channels=dim, num_heads=heads, num_blocks=blocks, crossattn_emb_channels=32, adaln_lora_dim=16)
    dm.eval()
    with torch.no_grad():
        for p in dm.parameters():
            torch.nn.init.normal_(p, std=0.05)
    return dm


dm = make_dit()
D = 96
W_ = torch.randn(D, 64) * 0.1
B_ = torch.randn(D) * 0.1
ctx = torch.randn(1, 5, 32)
t = torch.tensor([0.5])


def run(x):
    with torch.no_grad():
        return dm(x.clone(), t, ctx.clone())


# --- 1. embedder ordering == Forge's own patch-embed ordering --------------------------
ctrl = torch.randn(1, 16, 1, 8, 12)
forge_patches = dm.x_embedder.proj[0](torch.cat([ctrl, torch.zeros(1, 1, 1, 8, 12)], 1))  # (b t h w (c r m n)) incl. mask ch
tok = C.embed(ctrl, W_, B_, 1.0)
manual = torch.zeros(1, 1, 4, 6, D)
for i in range(4):
    for j in range(6):
        vec = ctrl[0, :, 0, 2 * i : 2 * i + 2, 2 * j : 2 * j + 2].flatten()  # (c, m, n) order
        manual[0, 0, i, j] = W_ @ vec + B_
check(torch.allclose(tok, manual, atol=1e-5), "embed() equals a hand-rolled per-patch Linear in (c, m, n) feature order")
check(torch.allclose(forge_patches[..., :64], C.rearrange(ctrl, "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=2, n=2), atol=0), "channel/patch ordering matches Forge PatchEmbed's Rearrange for the 16 latent channels")

# --- 2. hook: no-op when idle, exact at strength 0, plumbing when armed ------------------
x = torch.randn(1, 16, 1, 8, 12)
base = run(x)
PT.install_embedder_hook(dm)
PT.install_embedder_hook(dm)  # idempotent
check(len(dm.x_embedder._forward_hooks) == 1, "install_hook is idempotent")
check(torch.equal(run(x), base), "idle hook does not change the output")

PT.EMBED.arm(C.embed(ctrl, W_, B_, 0.0))
check(torch.equal(run(x), base), "strength 0 reproduces the base model exactly")
check(PT.EMBED.calls == 1, "hook was called once")

PT.EMBED.arm(C.embed(ctrl, W_, B_, 1.0))
armed = run(x)
check(not torch.allclose(armed, base, atol=1e-4), "armed control changes the output")

# reference: patch x_embedder.forward by hand and compare
orig_forward = dm.x_embedder.forward
PT.EMBED.reset()
dm.x_embedder.forward = lambda z: orig_forward(z) + C.embed(ctrl, W_, B_, 1.0)
ref = run(x)
dm.x_embedder.forward = orig_forward
check(torch.allclose(armed, ref, atol=1e-5), "hook output equals manually adding the tokens after x_embedder")

# --- 3. batch (cond+uncond) broadcast, odd latent size padding, reference frames --------
PT.EMBED.arm(C.embed(ctrl, W_, B_, 1.0))
xb = torch.randn(2, 16, 1, 8, 12)
with torch.no_grad():
    yb = dm(xb, t.repeat(2), ctx.repeat(2, 1, 1))
check(yb.shape == xb.shape, "batch of 2 with a batch-1 control broadcasts")

ctrl_odd = torch.randn(1, 16, 1, 7, 11)
tok_odd = C.embed(ctrl_odd, W_, B_, 1.0)
check(tok_odd.shape == (1, 1, 4, 6, D), "odd latent dims are padded to the token grid the DiT uses")
PT.EMBED.arm(tok_odd)
y_odd = run(torch.randn(1, 16, 1, 7, 11))
check(y_odd.shape == (1, 16, 1, 7, 11), "odd-size forward runs with control")

PT.EMBED.arm(C.embed(ctrl, W_, B_, 1.0))
x2 = torch.randn(1, 16, 2, 8, 12)  # target frame + one reference frame (ImageStitch)
y2 = run(x2)
check(y2.shape == x2.shape, "extra reference frame in the sequence is accepted")

try:
    PT.EMBED.arm(C.embed(torch.randn(1, 16, 1, 6, 6), W_, B_, 1.0))
    run(x)
    check(False, "mismatched grid should raise")
except RuntimeError as e:
    check("does not match" in str(e), "mismatched grid fails loudly instead of silently misapplying")
PT.EMBED.reset()

# --- 4. renderer / imaging --------------------------------------------------------------
kp = np.zeros((1, 133, 2))
sc = np.zeros((1, 133))
for i, (x_, y_) in enumerate([(50, 20), (45, 18), (55, 18), (40, 20), (60, 20), (35, 50), (65, 50), (30, 80), (70, 80), (28, 100), (72, 100), (40, 100), (60, 100), (40, 130), (60, 130), (40, 160), (60, 160)]):
    kp[0, i] = (x_, y_)
    sc[0, i] = 0.9
img = P.render(kp, sc, 100, 170)
check(img.shape == (170, 100, 3) and img.dtype == np.uint8, "render output shape/dtype")
check(tuple(img[20, 50]) == (255, 255, 255), "body joint dot is white")
check(img.max() == 255 and (img.sum(-1) > 0).sum() > 100, "limbs drawn")
sc_low = sc.copy()
sc_low[:] = 0.1
check(P.render(kp, sc_low, 100, 170).sum() == 0, "keypoints below the score threshold are not drawn")
kp2 = np.concatenate([kp, kp + 5])
sc2 = np.concatenate([sc, sc * 0.9])
big, _ = P.pick_largest(np.concatenate([kp * 0.5, kp]), np.concatenate([sc, sc]))
check(np.allclose(big[0], kp[0]), "pick_largest chooses the person with the larger span")

src = (np.random.rand(40, 60, 3) * 255).astype(np.uint8)
check(IM.fit_to(src, 64, 64, IM.CROP_AND_RESIZE).shape == (64, 64, 3), "fit_to crop")
check(IM.fit_to(src, 64, 96, IM.RESIZE_AND_FILL).shape == (96, 64, 3), "fit_to fill")
check(IM.fit_to(src, 64, 64, IM.JUST_RESIZE).shape == (64, 64, 3), "fit_to resize")

class _Fake:
    def __init__(self, kp, sc):
        self.kp, self.sc = kp, sc

    def __call__(self, bgr):
        return self.kp, self.sc


try:
    P.detect(np.zeros((64, 64, 3), np.uint8), estimator=_Fake(np.zeros((1, 133, 2)), np.full((1, 133), 0.1)))
    check(False, "low-confidence junk skeleton should be rejected")
except P.NoPersonError:
    check(True, "junk low-confidence skeleton (rtmlib's blank-image fallback) is rejected as 'no person'")
kp_ok, sc_ok = P.detect(np.zeros((64, 64, 3), np.uint8), estimator=_Fake(kp, sc))
check(kp_ok.shape == (1, 133, 2), "a real skeleton passes detect()")

# --- 5. real adapter file ------------------------------------------------------------------
if ADAPTER and os.path.exists(ADAPTER):
    w, b = C.load_embedder(ADAPTER)
    check(tuple(w.shape) == (2048, 64) and tuple(b.shape) == (2048,), "real embedder is Linear(64 -> 2048)")
    check(w.abs().max() > 0 and b.abs().max() > 0, "real embedder is non-zero (trained)")

    # the LoRA part must be loadable by Forge's native loader: compare names against Forge's own modules
    with open(ADAPTER, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    keys = list(header)
    block_ids = {int(m.group(1)) for k in keys if (m := re.match(r"diffusion_model\.blocks\.(\d+)\.", k))}
    check(max(block_ids) + 1 == 28, "adapter targets 28 DiT blocks (Forge's Anima 2B has 28)")
    ranks = {header[k]["shape"][0] for k in keys if k.endswith("lora_A.weight")}
    check(ranks == {16}, "LoRA rank is 16")

    dm28 = make_dit(blocks=28, dim=96)
    sd = dm28.state_dict()
    dit_bases = {k[: -len(".lora_A.weight")] for k in keys if k.startswith("diffusion_model.blocks.") and k.endswith("lora_A.weight")}
    missing = [b_ for b_ in dit_bases if b_[len("diffusion_model.") :] + ".weight" not in sd]
    check(not missing, f"all {len(dit_bases)} DiT LoRA targets exist in Forge's Anima ({missing[:3]})")

    with ops.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
        ad = LLMAdapter()
    asd = ad.state_dict()
    ad_bases = {k[len("diffusion_model.llm_adapter.") : -len(".lora_A.weight")] for k in keys if k.startswith("diffusion_model.llm_adapter.") and k.endswith("lora_A.weight")}
    missing_ad = [b_ for b_ in ad_bases if b_ + ".weight" not in asd]
    check(not missing_ad, f"all {len(ad_bases)} llm_adapter LoRA targets exist in Forge's LLMAdapter ({missing_ad[:3]})")
    other = [k for k in keys if not k.startswith(("diffusion_model.blocks.", "diffusion_model.llm_adapter.", "diffusion_model.control_embedder."))]
    check(not other, f"no unexpected key groups in the adapter file ({other[:3]})")
else:
    print("skip real-file checks (set ANIMA_POSE_FILE)")

print(f"\nall {ok} checks passed")
