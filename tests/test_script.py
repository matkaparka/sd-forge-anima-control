"""Integration checks of scripts/anima_control.py on Forge Neo's real ``modules`` (UI construction + the
hook sequence of a txt2img / img2img / Hires. fix run), with a small random Anima DiT, a fake VAE and a fake
processing object. CPU only, no model files.

    <Forge Neo>\\venv\\Scripts\\python.exe tests\\test_script.py

``FORGE_DIR`` overrides the Forge Neo checkout (default: three levels above this file).
"""
import atexit
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

EXT = Path(__file__).resolve().parents[1]
FORGE = Path(os.environ.get("FORGE_DIR") or EXT.parents[1])
assert (FORGE / "backend" / "nn" / "anima.py").exists(), f"set FORGE_DIR to your Forge Neo checkout (looked in {FORGE})"
DATA = tempfile.mkdtemp(prefix="anima_control_test_")  # keeps config.json / ui-config.json of the real install untouched
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
sys.argv = ["x", "--cpu", "--data-dir", DATA]
sys.path.insert(0, str(FORGE))
sys.path.insert(0, str(FORGE / "modules_forge" / "packages"))
sys.path.insert(0, str(EXT))

import gradio as gr
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from modules import initialize

initialize.imports()  # what webui.py does first: builds shared.opts / shared.state

import backend.operations as ops
from backend.args import dynamic_args
from backend.nn.anima import Anima
from lib_anima_control import ControlError
from lib_anima_control import embedder as EM
from lib_anima_control import imaging as IM
from lib_anima_control import loras as LR
from lib_anima_control import patches as PT
from lib_anima_control import pose as PO
from modules import scripts as forge_scripts
from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img

spec = importlib.util.spec_from_file_location("anima_control_script", EXT / "scripts" / "anima_control.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

torch.manual_seed(0)
ok = 0
records: list[logging.LogRecord] = []


class _Cap(logging.Handler):
    def emit(self, record):
        records.append(record)


logging.getLogger("Anima Control").addHandler(_Cap())


def logged(needle, level=None):
    return any(needle in r.getMessage() and (level is None or r.levelno == level) for r in records)


def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("ok  ", msg)


# --- fakes -------------------------------------------------------------------------------------------------
D = 96
with ops.using_forge_operations(device=torch.device("cpu"), dtype=torch.float32):
    DM = Anima(in_channels=16, out_channels=16, patch_spatial=2, patch_temporal=1, model_channels=D, num_heads=2, num_blocks=2, crossattn_emb_channels=32, adaln_lora_dim=16)
DM.eval()
with torch.no_grad():
    for prm in DM.parameters():
        torch.nn.init.normal_(prm, std=0.05)
CTX = torch.randn(1, 5, 32)


def forward(x):
    n = x.shape[0]
    with torch.no_grad():
        return DM(x.clone(), torch.full((n,), 0.5), CTX.repeat(n, 1, 1))


class FakeFSM:
    @staticmethod
    def process_in(s):
        return s * 0.5 + 1.0


class FakeVAE:
    first_stage_model = FakeFSM()
    fail = False

    def encode(self, s):
        if FakeVAE.fail:
            raise RuntimeError("vae exploded")
        g = torch.Generator().manual_seed(int(s.shape[1] * 1000 + s.shape[2]))
        return torch.randn(1, 16, 1, s.shape[1] // 8, s.shape[2] // 8, generator=g) + s.mean()


class FakeKModel:
    diffusion_model = DM

    def memory_required(self, shape):
        return 1.0


class FakeUnet:
    model = FakeKModel()

    def __init__(self, patches=None):
        self.weight_wrapper_patches = patches or {}


class FakeSD:
    def __init__(self, patches=None):
        self.forge_objects = types.SimpleNamespace(vae=FakeVAE(), unet=FakeUnet(patches))


def make_p(w=96, h=64, hr=False, img2img=False, patches=None):
    p = object.__new__(StableDiffusionProcessingImg2Img) if img2img else types.SimpleNamespace()
    p.width, p.height, p.enable_hr = w, h, hr
    p.all_prompts = ["1girl", "2girls <lora:canny:0.2>"]
    p.all_hr_prompts = ["hr one", "hr two"] if hr else []
    p.extra_generation_params = {}
    p.is_hr_pass = False
    p.sd_model = FakeSD(patches)
    p.init_images = []
    return p


rng = np.random.default_rng(1)
PHOTO = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
PHOTO[20:70, 30:90] = 200

# adapter files: a Pose adapter (LoRA keys + control embedder), a plain LoRA, and a stand-in Canny LoRA
TMP = Path(DATA)
W_, B_ = torch.randn(D, 64) * 0.1, torch.randn(D) * 0.1
lora_keys = {"diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.zeros(4, D), "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.zeros(D, 4)}
POSE_FILE = str(TMP / "anima_pose_test.safetensors")
save_file({**lora_keys, "diffusion_model.control_embedder.proj.weight": W_, "diffusion_model.control_embedder.proj.bias": B_}, POSE_FILE)
PLAIN_FILE = str(TMP / "plain_lora.safetensors")
save_file(lora_keys, PLAIN_FILE)
CANNY_FILE = "C:/loras/anima-canny.safetensors"
FILES = {"canny": CANNY_FILE, "pose": POSE_FILE, "plain": PLAIN_FILE}
LR.lora_file = lambda name: FILES[name] if name in FILES else (_ for _ in ()).throw(ControlError(f'Control LoRA "{name}" was not found'))

# a skeleton "image" (already drawn)
kp = np.zeros((1, 133, 2))
sc = np.zeros((1, 133))
for i, (x_, y_) in enumerate([(50, 20), (45, 18), (55, 18), (40, 20), (60, 20), (35, 50), (65, 50), (30, 80), (70, 80), (28, 100), (72, 100), (40, 100), (60, 100), (40, 130), (60, 130), (40, 160), (60, 160)]):
    kp[0, i] = (x_, y_)
    sc[0, i] = 0.9
SKELETON = PO.render(kp, sc, 100, 170)

script = S.AnimaControl()
CANNY_DEFAULT = dict(enable=True, image=PHOTO, pre=S.PRE_CANNY, detect_res=64, low=100, high=200, invert=False, resize_mode=IM.CROP_AND_RESIZE, lora="canny", weight=0.7, end=100, fade=True, apply_hr=False)
POSE_DEFAULT = dict(enable=True, image=SKELETON, pre=S.PRE_POSE_NONE, detect_res=512, people=S.PEOPLE_LARGEST, hands=True, face=True, feet=True, resize_mode=IM.CROP_AND_RESIZE, lora="pose", weight=1.0, strength=0.8, apply_hr=False)


def proc(p, canny=None, pose=False, image=None, enable=True, types=None):
    """canny / pose: None or False = that Control Type is not ticked ({} = ticked with defaults); a dict = overrides
    for its settings (its own optional image is `image`). `image` here is the shared Control Image; `types`
    overrides the ticked list."""
    c = {**CANNY_DEFAULT, **(canny or {})}
    q = {**POSE_DEFAULT, **(pose or {})}
    ticked = types if types is not None else ([S.POSE] if pose not in (False, None) else []) + ([S.CANNY] if canny not in (False, None) else [])
    a = dict(enable=enable, types=ticked, image=image, c_image=c["image"], p_image=q["image"])
    short = {"detect_res": "res", "resize_mode": "resize", "apply_hr": "hr"}  # the panel's key names
    a.update({"c_" + short.get(k, k): v for k, v in c.items() if k not in ("image", "enable")})
    a.update({"p_" + short.get(k, k): v for k, v in q.items() if k not in ("image", "enable")})
    script.process(p, *[a[k] for k in S.KEYS])


def reset_world():
    S._cleanup()
    shared.state.interrupted = False
    FakeVAE.fail = False
    dynamic_args.anima = True
    dynamic_args.online_lora = False
    dynamic_args.ref_latents.clear()
    records.clear()


def expected_ref(w, h, **over):
    a = {**CANNY_DEFAULT, **over}
    det = S._canny_map(a["image"], a["pre"], a["detect_res"], a["low"], a["high"], a["invert"])
    return IM.encode_ref_latent(FakeVAE(), IM.fit_to(det, w, h, a["resize_mode"]))


def expected_tokens(w, h, **over):
    a = {**POSE_DEFAULT, **over}
    return EM.embed(IM.encode_raw_latent(FakeVAE(), IM.fit_to(a["image"], w, h, a["resize_mode"])), W_, B_, a["strength"])


def native(x, ref):
    """Forge's own reference path (the ORIGINAL forward with dynamic_args.ref_latents set by hand)."""
    dynamic_args.ref_latents = [ref]
    try:
        return DM.__dict__[PT._FWD_ATTR](x.clone(), torch.full((x.shape[0],), 0.5), CTX.repeat(x.shape[0], 1, 1))
    finally:
        dynamic_args.ref_latents = []


def with_tokens(fn, tokens, first_frame_only=True):
    """Run fn() with `tokens` added after x_embedder by hand (idle hooks), on the target frame only."""
    orig = DM.x_embedder.forward

    def emb(z):
        o = orig(z)
        if first_frame_only and o.shape[1] != tokens.shape[1]:
            o = o.clone()
            o[:, : tokens.shape[1]] += tokens
            return o
        return o + tokens

    DM.x_embedder.forward = emb
    try:
        return fn()
    finally:
        DM.x_embedder.forward = orig


def sample(x, steps=1, total=1, start=0):
    """Fake sampler: the cfg_denoiser callback fires right before each model call."""
    outs = []
    for i in range(start, start + steps):
        PT.on_cfg_denoiser(types.SimpleNamespace(denoiser=types.SimpleNamespace(step=i, total_steps=total), sampling_step=i, total_sampling_steps=total))
        outs.append(forward(x))
    return outs


def sampling(p, x):
    script.process_before_every_sampling(p, x=x, noise=x)


def finish(p):
    script.postprocess_batch(p)
    script.postprocess(p, None)


# =============================================================================================================
# 1. UI
# =============================================================================================================
for img2img in (False, True):
    with gr.Blocks() as blocks:
        comps = script.ui(img2img)
    name = "img2img" if img2img else "txt2img"
    check(len(comps) == len(S.KEYS), f"ui({name}) returns {len(comps)} components == len(KEYS)")

    # what Forge's ui-config.json will key on: "<script>/<label>" (buttons: their value). Duplicates would make the
    # second component inherit the first one's default.
    seen, dups = {}, []

    def walk(x):
        if hasattr(x, "children"):
            for c in x.children:
                walk(c)
        else:
            key = x.label if getattr(x, "label", None) is not None else (x.value if isinstance(x, gr.Button) else None)
            if key is not None:
                if key in seen:
                    dups.append((key, seen[key], getattr(x, "value", None)))
                seen[key] = getattr(x, "value", None)

    walk(blocks)
    check(all(k == "\U0001f504" for k, _, _ in dups), f"ui({name}): no two controls share a label (only the two refresh buttons share their icon): {ascii([d[0] for d in dups])}")

labels = [getattr(c, "label", None) or "" for c in comps]
check(len([1 for l in labels if l.startswith("Anima Control")]) == 1, "ONE panel: a single 'Anima Control' enable box")
for want in ("Control Type", "Control Image", "Canny Image (optional)", "OpenPose Image (optional)", "Canny Preprocessor", "Canny Low Threshold", "Canny High Threshold", "Invert Canny Map", "Canny LoRA", "Control End (%)", "Pose Detector", "Pose Adapter (LoRA)", "Control Strength", "People", "Hands"):
    check(any(l.startswith(want) for l in labels), f'UI has "{want}"')
check(script.title() == "Anima Control" and script.show(False) is forge_scripts.AlwaysVisible, "script is 'Anima Control' and always visible")
defaults = {l: c.value for l, c in zip(labels, comps)}
check(defaults["Canny LoRA Weight"] == 0.7 and defaults["Pose LoRA Weight"] == 1.0, "the two LoRA weight sliders keep their own defaults (0.7 / 1.0)")
check(defaults["Anima Control"] is False and defaults["Control Type"] == [S.POSE], "panel starts disabled with OpenPose ticked (like Krea2 Control)")
type_box = comps[S.KEYS.index("types")]
check(type(type_box).__name__ == "CheckboxGroup" and list(S.TYPES) == [c[0] if isinstance(c, tuple) else c for c in type_box.choices], "Control Type is a CheckboxGroup (non-exclusive), not a Radio")

for ticked, want in (([S.POSE], [True, False, True, False]), ([S.CANNY], [False, True, False, True]), ([S.POSE, S.CANNY], [True, True, True, True]), ([], [False, False, False, False])):
    check([u["visible"] for u in S._on_types(ticked)] == want, f"ticking {ticked or 'nothing'} shows exactly those settings and previews")

PV = (S.PRE_CANNY, 64, 100, 200, False)
check(S._canny_preview(None, None, *PV) is None, "Canny Preview with no image warns and returns nothing")
check(min(S._canny_preview(PHOTO, None, *PV).shape[:2]) == 64, "Canny Preview returns the map at the preprocessing resolution")
check(np.array_equal(S._canny_preview(PHOTO, None, S.PRE_EDGE_NONE, 64, 100, 200, False), PHOTO), "preprocessor None passes the image through")
check(np.array_equal(S._canny_preview(PHOTO, None, S.PRE_EDGE_NONE, 64, 100, 200, True), 255 - PHOTO), "Invert flips the map")
check(np.array_equal(S._canny_preview(SKELETON, PHOTO, S.PRE_EDGE_NONE, 64, 100, 200, False), PHOTO), "Preview: the type's own image wins over the shared Control Image")
check(S._pose_preview(SKELETON, None, S.PRE_POSE_NONE, IM.CROP_AND_RESIZE, 512, S.PEOPLE_LARGEST, True, True, True).shape[0] % 8 == 0, "OpenPose Preview of a ready skeleton: multiple-of-8 size")
check(all(u["visible"] for u in S._on_canny_pre(S.PRE_CANNY)) and not any(u["visible"] for u in S._on_canny_pre(S.PRE_EDGE_NONE)), "Canny thresholds / resolution only show for the Canny preprocessor")
check(S._on_pose_pre(S.PRE_FORGE)["visible"] and not S._on_pose_pre(S.PRE_RTM)["visible"], "Pose resolution slider only shows for the Forge detector")

# =============================================================================================================
# 2. Canny alone (only Canny ticked)
# =============================================================================================================
reset_world()
p = make_p(hr=True)
proc(p, canny={})
check(set(S.JOBS) == {S.CANNY} and not shared.state.interrupted, "Canny only: setup succeeds, one job")
check(p.all_prompts == ["1girl <lora:canny:0.7>", "2girls <lora:canny:0.2>"], "LoRA tag appended to the prompt; the user's own tag and weight kept")
check(p.all_hr_prompts == ["hr one <lora:canny:0.7>", "hr two <lora:canny:0.7>"], "LoRA tag appended to the Hires prompts too")
info = p.extra_generation_params.get("Anima Control", "")
check(all(s_ in info for s_ in ("Canny", "64", "100/200", "Crop and Resize", "canny:0.7", "end 100%", "hires off")) and "OpenPose" not in info, f"PNG info line: {info!r}")

x = torch.randn(1, 16, 1, 8, 12)
base = forward(x)
sampling(p, x)
check(PT.STATE.active and not PT.EMBED.active and tuple(PT.STATE.ref.shape) == (1, 16, 1, 8, 12), "armed with a (1, 16, 1, 8, 12) reference for a 96x64 image; the embedder stays idle")
(out,) = sample(x)
canny_only = native(x, expected_ref(96, 64))
check(torch.equal(out, canny_only), "output equals Forge's native reference path fed with the same edge map")
check(not torch.equal(out, base) and dynamic_args.ref_latents == [], "the control changes the output, ref_latents empty after the forward")
(o2,) = sample(torch.randn(2, 16, 1, 8, 12))
check(o2.shape == (2, 16, 1, 8, 12), "batch of 2 through the armed wrapper")
script.postprocess_batch(p)
check(not PT.STATE.active and torch.equal(forward(x), base), "postprocess_batch disarms: the model behaves as without control again")
script.postprocess(p, None)
check(not S.JOBS, "postprocess drops the jobs")

# Invert (line drawing: black on white) with the preprocessor on None
LINEART = np.full((64, 96, 3), 255, np.uint8)
LINEART[10:54, 46:50] = 0
reset_world()
p = make_p()
proc(p, canny={"image": LINEART, "pre": S.PRE_EDGE_NONE, "invert": True})
m = S.JOBS[S.CANNY].at(96, 64)
check(m[30, 48].min() == 255 and m[30, 10].max() == 0, "line drawing + None + Invert -> white line on black (what Canny output looks like)")
check("inverted" in p.extra_generation_params["Anima Control"], "PNG info says the map was inverted")
sampling(p, torch.randn(1, 16, 1, 8, 12))
check(logged("map: preprocessor None + inverted, 96x64, 2.9% white pixels"), "the console log says what was fed: preprocessor, inverted, and 2.9% white pixels (a thin white line on black)")
finish(p)

reset_world()
p = make_p()
proc(p, canny={"image": LINEART, "pre": S.PRE_EDGE_NONE, "invert": False})
sampling(p, torch.randn(1, 16, 1, 8, 12))
check(logged("map: preprocessor None, 96x64, 97.1% white pixels") and not logged("inverted"), "...and without Invert the same drawing is reported as 97.1% white (so a forgotten Invert is visible in the log)")
finish(p)

# Hires. fix
for apply_hr in (False, True):
    reset_world()
    p = make_p(hr=True)
    proc(p, canny={"apply_hr": apply_hr})
    xb = torch.randn(1, 16, 1, 8, 12)
    sampling(p, xb)
    sample(xb)
    p.is_hr_pass = True
    xh = torch.randn(1, 16, 1, 16, 24)
    sampling(p, xh)
    if apply_hr:
        check(PT.STATE.active and tuple(PT.STATE.ref.shape[-2:]) == (16, 24) and PT.STATE.stage == "hires", "Hires pass ON: re-armed with a reference encoded at the hires size")
        check(sorted(S.JOBS[S.CANNY].cache) == [(96, 64), (192, 128)], "references cached per size")
        (oh,) = sample(xh)
        check(torch.equal(oh, native(xh, expected_ref(192, 128))), "hires output equals native with the edge map fitted to 192x128")
    else:
        check(not PT.STATE.active and logged("Canny skipped"), "Hires pass OFF (default): control disarmed, and it says so in the log")
        (oh,) = sample(xh)
        check(torch.equal(oh, forward(xh)) and dynamic_args.ref_latents == [], "hires output has no reference in it")
        check(not logged("NOT applied"), "no false 'never applied' warning for the skipped stage")
    finish(p)


# Control End + online LoRA fade
class _L:
    def __init__(self, name, w):
        self.name, self.patch = name, [[w, "data", 1.0, None, None]]


for online in (False, True):
    reset_world()
    dynamic_args.online_lora = online
    mine, other = [_L(CANNY_FILE, 0.7), _L(CANNY_FILE, 0.7)], _L("C:/loras/style.safetensors", 0.5)
    p = make_p(patches={"k1": [mine[0], other], "k2": [mine[1]]})
    proc(p, canny={"end": 50})
    xs = torch.randn(1, 16, 1, 8, 12)
    base = forward(xs)
    sampling(p, xs)
    outs = sample(xs, steps=10, total=10)
    armed = native(xs, expected_ref(96, 64))
    check(all(torch.equal(o, armed) for o in outs[:5]), f"[online LoRA {online}] steps 0-4 (< 50%) have the reference")
    check(all(torch.equal(o, base) for o in outs[5:]), f"[online LoRA {online}] steps 5-9 have none")
    check(PT.STATE.calls == 5 and PT.STATE.skipped == 5, f"[online LoRA {online}] 5 steps with / 5 without")
    check(sum("Control End reached" in r.getMessage() for r in records) == 1, f"[online LoRA {online}] Control End is logged once")
    if online:
        check(all(m.patch[0][0] == 0.0 for m in mine) and other.patch[0][0] == 0.5, "[online LoRA True] control LoRA at 0 after Control End, other LoRAs untouched")
        script.postprocess_batch(p)
        check(all(m.patch[0][0] == 0.7 for m in mine), "[online LoRA True] LoRA strength restored at the end of the batch (no residue for the next generation)")
    else:
        check(all(m.patch[0][0] == 0.7 for m in mine) and logged("LoRA stays merged", logging.WARNING), "[online LoRA False] LoRA cannot be faded: left alone, and the log says so")
        script.postprocess_batch(p)
    script.postprocess(p, None)

reset_world()
dynamic_args.online_lora = True
p = make_p(patches={"k": [_L(CANNY_FILE, 0.7)]})
proc(p, canny={"end": 100})
sampling(p, torch.randn(1, 16, 1, 8, 12))
check(PT.STATE.fade is None, "Control End 100% never touches the LoRA")
finish(p)

# failures are visible, never a silent uncontrolled run
reset_world()
p = make_p()
proc(p, canny={"image": None})
check(shared.state.interrupted and not S.JOBS and logged("no control image", logging.ERROR) and not any(r.exc_info for r in records), "Canny without a control image: one clear error line (no traceback) + interrupt")
check(p.all_prompts[0] == "1girl", "...and nothing was injected into the prompt")

reset_world()
proc(make_p(), canny={"lora": "nope"})
check(shared.state.interrupted and not S.JOBS and logged("was not found", logging.ERROR) and not any(r.exc_info for r in records), "unknown LoRA: one clear error line (no traceback) + interrupt")

reset_world()
p = make_p()
proc(p, canny={})
FakeVAE.fail = True
xs = torch.randn(1, 16, 1, 8, 12)
sampling(p, xs)
check(shared.state.interrupted and not PT.STATE.active and logged("failed to prepare the control"), "VAE failure at sampling time: error + interrupt, nothing armed")
script.postprocess(p, None)

reset_world()
dynamic_args.anima = False
p = make_p()
proc(p, canny={})
check(not S.JOBS and not shared.state.interrupted and logged("not Anima", logging.WARNING) and p.all_prompts[0] == "1girl", "non-Anima checkpoint: warns and does nothing (no LoRA tag, no interrupt)")

reset_world()
p = make_p()
proc(p, canny={}, pose={}, enable=False)
check(not S.JOBS and not records and p.all_prompts[0] == "1girl", "panel switched off: completely inert, whatever is ticked")

reset_world()
p = make_p()
proc(p, types=[])
check(shared.state.interrupted and not S.JOBS and logged("no Control Type is ticked", logging.ERROR) and not any(r.exc_info for r in records) and p.all_prompts[0] == "1girl", "panel on but no Control Type ticked: one clear error + interrupt, nothing injected")

reset_world()
p = make_p()
proc(p, canny={"lora": "None"})
check(S.JOBS and logged('"Canny LoRA" is None', logging.WARNING) and p.all_prompts[0] == "1girl", 'LoRA "None": allowed (user may write the tag), warned, nothing appended')
S._cleanup()

reset_world()
p = make_p()
proc(p, canny={"end": 0})
check(logged("Control End is 0%", logging.WARNING), "Control End 0% is flagged")
sampling(p, torch.randn(1, 16, 1, 8, 12))
script.postprocess_batch(p)
check(not logged("NOT applied"), "...without a false 'never applied' warning")
script.postprocess(p, None)

reset_world()
p = make_p()
proc(p, canny={})
sampling(p, torch.randn(1, 16, 1, 8, 12))
script.postprocess_batch(p)
check(logged("NOT applied", logging.WARNING) and not PT.STATE.active, "a control that never reached the model is reported at the end of the batch")
script.postprocess(p, None)

# img2img fallback, native Enable Reference, leftovers, exceptions
reset_world()
p = make_p(img2img=True)
p.init_images = [Image.fromarray(PHOTO)]
proc(p, canny={"image": None})
check(S.CANNY in S.JOBS and not shared.state.interrupted, "img2img with an empty control image uses the img2img input")
check(np.array_equal(S.JOBS[S.CANNY].at(96, 64), IM.fit_to(IM.canny(PHOTO, 100, 200, 64), 96, 64, IM.CROP_AND_RESIZE)), "...preprocessed from that input image")
script.postprocess(p, None)

reset_world()
shared.opts.data["anima_do_reference"] = True
proc(make_p(), canny={})
check(logged("Enable Reference", logging.WARNING), "[Anima] Enable Reference ON is warned about (Canny)")
shared.opts.data["anima_do_reference"] = False
script.postprocess(make_p(), None)

reset_world()
PT.STATE.arm(torch.randn(1, 16, 1, 8, 12))
PT.EMBED.arm(torch.randn(1, 1, 4, 6, D))
S.JOBS[S.CANNY] = S._Job(S.CANNY, lambda w, h: PHOTO, 1.0, False, None, 0.7)
script.before_process(make_p())
check(not PT.STATE.active and not PT.EMBED.active and not S.JOBS, "before_process clears the leftovers of an interrupted job (both controls)")

reset_world()
p = make_p()
proc(p, canny={})
xs = torch.randn(1, 16, 1, 8, 12)
sampling(p, xs)
boom = DM.blocks[1].register_forward_pre_hook(lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")))
try:
    forward(xs)
    raised = False
except RuntimeError:
    raised = True
boom.remove()
check(raised and dynamic_args.ref_latents == [], "exception during sampling: ref_latents empty afterwards")
script.postprocess(p, None)
check(not PT.STATE.active and not S.JOBS, "postprocess cleans up after the failed run")

# =============================================================================================================
# 3. OpenPose alone (only OpenPose ticked)
# =============================================================================================================
reset_world()
p = make_p(w=96, h=64, hr=True)
proc(p, pose={})
check(set(S.JOBS) == {S.POSE} and not shared.state.interrupted, "OpenPose only: setup succeeds with an adapter that has a control embedder")
check(p.all_prompts[0] == "1girl <lora:pose:1>" and p.all_hr_prompts[0] == "hr one <lora:pose:1>", "adapter added to the prompt and the Hires prompt as a LoRA")
info = p.extra_generation_params.get("Anima Control", "")
check(all(s_ in info for s_ in ("OpenPose", "strength 0.8", "pose:1", "hires off")) and "Canny" not in info, f"PNG info line: {info!r}")

x = torch.randn(1, 16, 1, 8, 12)
base = forward(x)
sampling(p, x)
check(PT.EMBED.active and not PT.STATE.active and tuple(PT.EMBED.tokens.shape) == (1, 1, 4, 6, D), "OpenPose arms the embedder (not the reference path): tokens (1, 1, 4, 6, 96)")
out = forward(x)
tokens = expected_tokens(96, 64)
PT.EMBED.reset()
pose_only = with_tokens(lambda: forward(x), tokens)
check(torch.allclose(out, pose_only, atol=1e-5) and not torch.allclose(out, base, atol=1e-4), "output equals adding the embedded skeleton after x_embedder by hand")
check(dynamic_args.ref_latents == [], "the reference path is not used (ref_latents untouched)")
finish(p)
check(not PT.EMBED.active and torch.equal(forward(x), base), "postprocess_batch disarms the hook: base behaviour again")
fit = IM.fit_to(SKELETON, 96, 64, IM.CROP_AND_RESIZE)
check(torch.allclose(IM.encode_raw_latent(FakeVAE(), fit) * 0.5 + 1.0, IM.encode_ref_latent(FakeVAE(), fit)), "OpenPose feeds the RAW latent, Canny the process_in'd one")

for apply_hr in (False, True):
    reset_world()
    p = make_p(w=96, h=64, hr=True)
    proc(p, pose={"apply_hr": apply_hr})
    xb = torch.randn(1, 16, 1, 8, 12)
    sampling(p, xb)
    forward(xb)
    p.is_hr_pass = True
    xh = torch.randn(1, 16, 1, 16, 24)
    sampling(p, xh)
    if apply_hr:
        check(PT.EMBED.active and PT.EMBED.stage == "hires" and tuple(PT.EMBED.tokens.shape[2:4]) == (8, 12), "OpenPose Hires pass ON: tokens re-made at the hires size")
    else:
        check(not PT.EMBED.active and logged("OpenPose skipped"), "OpenPose Hires pass OFF: disarmed and logged")
    finish(p)

reset_world()
proc(make_p(), pose={"lora": "None"})
check(shared.state.interrupted and not S.JOBS and logged("pick the adapter file", logging.ERROR), "OpenPose without an adapter: error + interrupt (the embedder comes from it)")

reset_world()
proc(make_p(), pose={"lora": "plain"})
check(shared.state.interrupted and not S.JOBS and logged("plain LoRA", logging.ERROR) and not any(r.exc_info for r in records), "a plain LoRA instead of the adapter: one clear error + interrupt, not a silent no-op")

reset_world()
proc(make_p(), pose={"image": None})
check(shared.state.interrupted and logged("no control image", logging.ERROR), "OpenPose without an image: error + interrupt")

reset_world()
proc(make_p(), pose={"lora": "nope"})
check(shared.state.interrupted and logged("was not found", logging.ERROR), "unknown adapter: error + interrupt")

real_detect = PO.detect
PO.detect = lambda rgb, estimator=None: (kp * np.array([rgb.shape[1] / 100, rgb.shape[0] / 170]), sc)
reset_world()
p = make_p(w=96, h=64)
proc(p, pose={"pre": S.PRE_RTM, "image": PHOTO})
check(S.POSE in S.JOBS and not shared.state.interrupted, "DWPose path: setup succeeds (fake detector)")
skel = S.JOBS[S.POSE].at(192, 128)
check(skel.shape == (128, 192, 3) and skel.max() == 255 and (skel.sum(-1) > 0).sum() > 100, "the skeleton is rendered at whatever size is asked for (here the hires size)")
sampling(p, torch.randn(1, 16, 1, 8, 12))
check(PT.EMBED.active, "DWPose path arms the embedder")
finish(p)

PO.detect = lambda rgb, estimator=None: (_ for _ in ()).throw(PO.NoPersonError("no person detected in the control image"))
reset_world()
proc(make_p(), pose={"pre": S.PRE_RTM, "image": PHOTO})
check(shared.state.interrupted and logged("no person detected", logging.ERROR) and not any(r.exc_info for r in records), "no person found: one clear error + interrupt")
PO.detect = real_detect

# =============================================================================================================
# 3b. Shared Control Image vs the optional per-type images
# =============================================================================================================
PHOTO2 = np.ascontiguousarray(PHOTO[::-1, ::-1])
reset_world()
p = make_p()
proc(p, canny={"image": None}, pose={"image": None}, image=PHOTO)
check(set(S.JOBS) == {S.POSE, S.CANNY}, "only the shared Control Image given: both ticked types use it")
check(np.array_equal(S.JOBS[S.CANNY].at(96, 64), IM.fit_to(IM.canny(PHOTO, 100, 200, 64), 96, 64, IM.CROP_AND_RESIZE)) and np.array_equal(S.JOBS[S.POSE].at(96, 64), IM.fit_to(PHOTO, 96, 64, IM.CROP_AND_RESIZE)), "...Canny edges and the pose map both come from that one image")
S._cleanup()

reset_world()
p = make_p()
proc(p, canny={"image": PHOTO2}, pose={"image": None}, image=PHOTO)
check(np.array_equal(S.JOBS[S.CANNY].at(96, 64), IM.fit_to(IM.canny(PHOTO2, 100, 200, 64), 96, 64, IM.CROP_AND_RESIZE)) and np.array_equal(S.JOBS[S.POSE].at(96, 64), IM.fit_to(PHOTO, 96, 64, IM.CROP_AND_RESIZE)), "a type's own image wins; the other type still uses the shared one")
S._cleanup()

reset_world()
p = make_p()
proc(p, canny={"image": None}, pose={"image": None})
check(shared.state.interrupted and not S.JOBS and logged("no control image", logging.ERROR), "no shared image and no own image: error + interrupt (txt2img)")

reset_world()
p = make_p(img2img=True)
p.init_images = [Image.fromarray(PHOTO)]
proc(p, canny={"image": None}, pose={"image": None})
check(set(S.JOBS) == {S.POSE, S.CANNY} and not shared.state.interrupted, "img2img: with no image anywhere both types use the img2img input")
S._cleanup()

reset_world()
p = make_p()
proc(p, canny={}, pose=False)
check(set(S.JOBS) == {S.CANNY}, "only Canny ticked: no OpenPose job, even though the OpenPose settings hold values")
S._cleanup()

# =============================================================================================================
# 4. Canny and OpenPose ticked together
# =============================================================================================================
reset_world()
p = make_p(w=96, h=64, hr=True)
proc(p, canny={}, pose={})
check(set(S.JOBS) == {S.CANNY, S.POSE} and not shared.state.interrupted, "both ticked: two jobs, setup succeeds")
check(p.all_prompts == ["1girl <lora:pose:1> <lora:canny:0.7>", "2girls <lora:canny:0.2> <lora:pose:1>"], "both adapters are added to the prompt, OpenPose first (the user's own canny tag still wins)")
check(p.all_hr_prompts[0] == "hr one <lora:pose:1> <lora:canny:0.7>", "...and to the Hires prompt")
info = p.extra_generation_params["Anima Control"]
check(info.count(" + ") == 1 and "Canny" in info and "OpenPose" in info, f"one PNG info line carries both: {info!r}")

x = torch.randn(1, 16, 1, 8, 12)
base = forward(x)
sampling(p, x)
check(PT.STATE.active and PT.EMBED.active, "both controls are armed at the same time")
both = forward(x)
ref, tokens = expected_ref(96, 64), expected_tokens(96, 64)
PT.reset_all()
expected = with_tokens(lambda: native(x, ref), tokens)  # reference frame appended, control tokens on the TARGET frame only
check(torch.allclose(both, expected, atol=1e-5), "output == Forge's native reference path + the embedder tokens on the target frame only")
check(not torch.allclose(both, canny_only, atol=1e-4) and not torch.allclose(both, with_tokens(lambda: forward(x), tokens), atol=1e-4), "...and it differs from using either control alone")
check(dynamic_args.ref_latents == [], "ref_latents empty after the forward")

# batch sizes
sampling(p, x)
for n in (2, 3, 4):
    xn = torch.randn(n, 16, 1, 8, 12)
    got = forward(xn)
    sampling_rows = torch.cat([forward(xn[i : i + 1]) for i in range(n)])
    check(torch.allclose(got, sampling_rows, atol=1e-5), f"batch {n}: both controls apply to every row, same as batch-1 runs")
finish(p)
check(not PT.STATE.active and not PT.EMBED.active and torch.equal(forward(x), base), "postprocess_batch disarms both")

# Hires: each control follows its own switch
for hr_canny, hr_pose in ((False, True), (True, False), (True, True), (False, False)):
    reset_world()
    p = make_p(w=96, h=64, hr=True)
    proc(p, canny={"apply_hr": hr_canny}, pose={"apply_hr": hr_pose})
    xb = torch.randn(1, 16, 1, 8, 12)
    sampling(p, xb)
    forward(xb)
    p.is_hr_pass = True
    xh = torch.randn(1, 16, 1, 16, 24)
    sampling(p, xh)
    check(PT.STATE.active == hr_canny and PT.EMBED.active == hr_pose, f"Hires pass: Canny {'on' if hr_canny else 'off'}, OpenPose {'on' if hr_pose else 'off'} -> armed exactly that")
    if hr_canny or hr_pose:
        forward(xh)
    finish(p)
    check(not logged("NOT applied"), "...no false 'never applied' warning")

# Control End belongs to Canny only; OpenPose keeps going
for online in (False, True):
    reset_world()
    dynamic_args.online_lora = online
    canny_lora = [_L(CANNY_FILE, 0.7), _L(CANNY_FILE, 0.7)]
    pose_lora = [_L(POSE_FILE, 1.0)]
    p = make_p(patches={"a": [canny_lora[0], pose_lora[0]], "b": [canny_lora[1]]})
    proc(p, canny={"end": 50}, pose={})
    xs = torch.randn(1, 16, 1, 8, 12)
    sampling(p, xs)
    outs = sample(xs, steps=10, total=10)
    counters = (PT.STATE.calls, PT.STATE.skipped, PT.EMBED.calls)
    faded = ([m.patch[0][0] for m in canny_lora], pose_lora[0].patch[0][0])
    # restore the LoRA strengths exactly as the end of the batch would, then compute the expectations with idle hooks
    script.postprocess_batch(p)
    restored = ([m.patch[0][0] for m in canny_lora], pose_lora[0].patch[0][0])
    ref, tokens = expected_ref(96, 64), expected_tokens(96, 64)
    full = with_tokens(lambda: native(xs, ref), tokens)
    pose_alone = with_tokens(lambda: forward(xs), tokens)
    check(all(torch.allclose(o, full, atol=1e-5) for o in outs[:5]), f"[online {online}] steps 0-4: reference + skeleton")
    check(all(torch.allclose(o, pose_alone, atol=1e-5) for o in outs[5:]), f"[online {online}] steps 5-9: Canny has ended, the OpenPose control keeps going")
    check(counters == (5, 5, 10), f"[online {online}] counters (Canny with, Canny without, OpenPose): {counters}")
    if online:
        check(faded == ([0.0, 0.0], 1.0), f"[online] only the Canny LoRA is switched off after Control End; the pose adapter stays at its weight: {faded}")
    else:
        check(faded == ([0.7, 0.7], 1.0), "[offline LoRA] nothing could be switched off, nothing was touched")
    check(restored == ([0.7, 0.7], 1.0), "both LoRAs are at their own strength again once the batch ends")
    script.postprocess(p, None)

# failures stay atomic
reset_world()
p = make_p()
proc(p, canny={}, pose={"image": None})
check(shared.state.interrupted and not S.JOBS and p.all_prompts[0] == "1girl" and logged("OpenPose: no control image", logging.ERROR), "OpenPose fails while Canny is fine: job interrupted, and the Canny tag was NOT left in the prompt")

reset_world()
p = make_p()
proc(p, canny={"image": None}, pose={})
check(shared.state.interrupted and not S.JOBS and p.all_prompts[0] == "1girl" and logged("Canny: no control image", logging.ERROR), "Canny fails while OpenPose is fine: same, and the message says which control")

reset_world()
p = make_p()
proc(p, canny={"lora": "pose"}, pose={})
check(shared.state.interrupted and not S.JOBS and p.all_prompts[0] == "1girl" and logged("two different adapter files", logging.ERROR), "the same LoRA for both types is refused with a clear message")

reset_world()
p = make_p()
proc(p, canny={}, pose={})
FakeVAE.fail = True
sampling(p, torch.randn(1, 16, 1, 8, 12))
check(shared.state.interrupted and not PT.STATE.active and not PT.EMBED.active, "VAE failure with both enabled: interrupt, nothing left armed")
script.postprocess(p, None)

reset_world()
p = make_p()
proc(p, canny={}, pose={})
xs = torch.randn(1, 16, 1, 8, 12)
sampling(p, xs)
boom = DM.blocks[1].register_forward_pre_hook(lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")))
try:
    forward(xs)
    raised = False
except RuntimeError:
    raised = True
boom.remove()
check(raised and dynamic_args.ref_latents == [], "exception during sampling with both: ref_latents empty afterwards")
script.postprocess(p, None)
check(not PT.STATE.active and not PT.EMBED.active and not S.JOBS, "postprocess cleans up both")

print(f"\nall {ok} checks passed")
