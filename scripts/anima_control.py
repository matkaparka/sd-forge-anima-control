import logging

import gradio as gr
import numpy as np

from backend.args import dynamic_args
from lib_anima_control import ControlError
from lib_anima_control import embedder as EM
from lib_anima_control import imaging as IM
from lib_anima_control import loras as LR
from lib_anima_control import patches as PT
from lib_anima_control import pose as PO
from modules import script_callbacks, scripts, shared
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.ui_components import FormRow, InputAccordion, ToolButton

try:
    from backend.logging import setup_logger

    logger = logging.getLogger("Anima Control")
    setup_logger(logger)
except Exception:
    logger = logging.getLogger("Anima Control")

POSE = "OpenPose"
CANNY = "Canny"
TYPES = (POSE, CANNY)  # the "Control Type" choices; any subset can be ticked

PRE_CANNY = "Canny"
PRE_EDGE_NONE = "None (image is already an edge map)"
PRE_RTM = "DWPose (rtmlib) - matches training"
PRE_FORGE = "Forge dw_openpose_full - different render, weaker"
PRE_POSE_NONE = "None (image is already a skeleton)"
FORGE_DW_NAME = "dw_openpose_full"

PEOPLE_LARGEST = "Largest person"
PEOPLE_ALL = "Everyone"

DEFAULT_WEIGHT = {CANNY: 0.7, POSE: 1.0}
LORA_WORDS = {CANNY: ("anima", "canny"), POSE: ("anima", "pose")}  # every word must be in the file name to be pre-selected
PREVIEW_MAX = 1024

# One panel, any combination of control types. `ui()` returns the components in this order and `process()` names the
# flat argument list again with it. Labels must be unique inside the panel: Forge's ui-config.json is keyed by
# "<script>/<label>", and a repeated label would hand the second component the first one's default.
KEYS = (
    "enable", "types", "image", "c_image", "p_image",
    "c_pre", "c_res", "c_low", "c_high", "c_invert", "c_resize", "c_lora", "c_weight", "c_end", "c_fade", "c_hr",
    "p_pre", "p_res", "p_people", "p_hands", "p_face", "p_feet", "p_resize", "p_lora", "p_weight", "p_strength", "p_hr",
)  # fmt: skip


script_callbacks.on_cfg_denoiser(PT.on_cfg_denoiser)


# ---------------------------------------------------------------------------
# module-level job state (shared by the txt2img and img2img instances)
# ---------------------------------------------------------------------------


class _Job:
    """One ticked control type (OpenPose and Canny can both be ticked)."""

    def __init__(self, kind, at, end, apply_hr, lora_name, lora_weight, fade=False, embedder=None, strength=1.0):
        self.kind = kind
        self.at = at  # (width, height) -> control map, RGB uint8, exactly that size
        self.end = end  # 0..1, Canny only
        self.apply_hr = apply_hr
        self.lora_name = lora_name
        self.lora_weight = lora_weight
        self.fade = fade  # Canny: also switch the LoRA off after Control End (online LoRA only)
        self.embedder = embedder  # OpenPose: (weight, bias)
        self.strength = strength  # OpenPose
        self.cache: dict[tuple[int, int], object] = {}  # size -> reference latent (Canny) / control tokens (OpenPose)
        self.fade_warned = False
        self.note = ""  # what the control map is, for the console log


JOBS: dict[str, _Job] = {}


def _disarm(where: str):
    """Disarm the patches. A control that was armed but never reached the model is reported, not hidden."""
    if JOBS:
        for st in (PT.STATE, PT.EMBED):
            if st.active and st.calls == 0 and st.end > 0:
                logger.warning(f"Anima Control [{st.stage}]: the model was never called with the control before {where} -- the control was NOT applied")
    PT.reset_all()


def _cleanup():
    JOBS.clear()
    PT.reset_all()


def _toast(msg: str):
    try:
        gr.Warning(msg)
    except Exception:
        pass  # outside a Gradio request (API / tests): the log line is enough


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


def _canny_map(src, pre, detect_res, low, high, invert):
    """Source image -> edge map at the preprocessing resolution. Invert flips white-on-black to black-on-white;
    with the preprocessor set to None it turns a black-on-white line drawing into white lines on black."""
    m = src if pre == PRE_EDGE_NONE else IM.canny(src, low, high, detect_res)
    return (255 - m) if invert else m


def _build_skeleton(src, pre, resize_mode, width, height, people, hands, face, feet, detect_res):
    """OpenPose -> callable (w, h) -> skeleton RGB. Detection runs once; rendering is cheap and size-agnostic."""
    if pre == PRE_POSE_NONE:
        return lambda w, h: IM.fit_to(src, w, h, resize_mode)

    if pre == PRE_FORGE:
        detected = IM.run_forge_preprocessor(FORGE_DW_NAME, src, detect_res)
        return lambda w, h: IM.fit_to(detected, w, h, resize_mode)

    target = IM.fit_to(src, width, height, resize_mode)
    kp, sc = PO.detect(target)
    if people == PEOPLE_LARGEST:
        kp, sc = PO.pick_largest(kp, sc)
    norm = kp / np.array([width, height], float)  # keypoints as fractions of the canvas, so any size can be rendered
    return lambda w, h: PO.render(norm * np.array([w, h], float), sc, w, h, hands, face, feet)


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------


def _on_types(types):
    types = types or []
    pose, canny = POSE in types, CANNY in types
    return [gr.update(visible=pose), gr.update(visible=canny), gr.update(visible=pose), gr.update(visible=canny)]  # previews, then settings


def _on_canny_pre(pre):
    show = pre == PRE_CANNY
    return [gr.update(visible=show), gr.update(visible=show)]  # detect_res, thresholds row


def _on_pose_pre(pre):
    return gr.update(visible=pre == PRE_FORGE)  # detect_res


def _on_refresh(current_lora):
    return gr.update(choices=LR.list_loras(refresh=True), value=current_lora)


def _canny_preview(image, own_image, pre, detect_res, low, high, invert):
    src = IM.to_rgb_array(own_image) if own_image is not None else IM.to_rgb_array(image)
    if src is None:
        gr.Warning("Upload a control image first")
        return None
    try:
        return _canny_map(src, pre, detect_res, low, high, invert)
    except Exception as e:
        gr.Warning(f"Preprocessing failed: {e}")
        return None


def _pose_preview(image, own_image, pre, resize_mode, detect_res, people, hands, face, feet):
    src = IM.to_rgb_array(own_image) if own_image is not None else IM.to_rgb_array(image)
    if src is None:
        gr.Warning("Upload a control image first")
        return None
    try:
        h, w = src.shape[:2]
        scale = min(1.0, PREVIEW_MAX / max(h, w))
        w, h = max(8, round(w * scale) // 8 * 8), max(8, round(h * scale) // 8 * 8)
        return _build_skeleton(src, pre, resize_mode, w, h, people, hands, face, feet, int(detect_res))(w, h)
    except Exception as e:
        gr.Warning(f"Preview failed: {e}")
        return None


INFO = """
Needs a <b>control LoRA</b> in your <b>LoRA</b> folder (not included) for each ticked Control Type; it is added to your prompt for you.<br>
<b>OpenPose</b>: <a href="https://huggingface.co/Claquasse/Anima-Control-Pose" target="_blank">Claquasse/Anima-Control-Pose</a>
(skeleton &rarr; control embedder; the same file is the LoRA and the embedder). Non-commercial weights; the author calls it an experimental preview.<br>
<b>Canny</b>: levzzz's <a href="https://civitai.com/models/2443202" target="_blank">anima-preview-canny-v0.2</a>
(edge map &rarr; reference latent). Keep <b>[Anima] Enable Reference</b> (Settings) OFF and ImageStitch disabled.
Edges come from pictures: a black-and-white line drawing usually works badly with Canny.<br>
<b>Tick one or both.</b> Stacking two separately trained adapters is not covered by either author: start with lower LoRA weights.
"""


class AnimaControl(scripts.Script):
    def title(self):
        return "Anima Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        loras = LR.list_loras()
        suffix = " (empty = use img2img input)" if is_img2img else ""

        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(INFO)
            types = gr.CheckboxGroup(list(TYPES), value=[POSE], label="Control Type", info="tick one or both; each ticked type shows its own settings below")
            with gr.Row():
                image = gr.Image(label="Control Image" + suffix, sources=["upload", "clipboard"], type="numpy", image_mode="RGB", height=320)
                with gr.Column():
                    p_preview = gr.Image(label="Skeleton", type="numpy", interactive=False, height=320)
                    c_preview = gr.Image(label="Canny Map", type="numpy", interactive=False, height=320, visible=False)

            # ---------------- OpenPose ----------------
            with gr.Column() as pose_box:
                gr.HTML("<b>OpenPose</b>")
                with FormRow():
                    p_pre = gr.Dropdown([PRE_RTM, PRE_FORGE, PRE_POSE_NONE], value=PRE_RTM, label="Pose Detector")
                    p_res = gr.Slider(128, 2048, value=512, step=8, label="Forge DWPose Resolution", info="only for the Forge detector", visible=False)
                    p_btn = gr.Button("Preview Skeleton", size="sm")
                with FormRow():
                    p_people = gr.Radio([PEOPLE_LARGEST, PEOPLE_ALL], value=PEOPLE_LARGEST, label="People")
                    p_hands = gr.Checkbox(True, label="Hands")
                    p_face = gr.Checkbox(True, label="Face")
                    p_feet = gr.Checkbox(True, label="Feet")
                p_resize = gr.Radio(list(IM.RESIZE_MODES), value=IM.CROP_AND_RESIZE, label="Pose Resize Mode")
                with FormRow():
                    p_lora = gr.Dropdown(loras, value=LR.guess_lora(loras, LORA_WORDS[POSE]), label="Pose Adapter (LoRA)", allow_custom_value=True)
                    p_refresh = ToolButton("🔄", elem_id=self.elem_id("pose_refresh"))
                    p_weight = gr.Slider(0.0, 2.0, value=DEFAULT_WEIGHT[POSE], step=0.05, label="Pose LoRA Weight")
                p_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Control Strength", info="0 = no control, 1 = follow the skeleton; higher follows closer but can cost image quality")
                p_hr = gr.Checkbox(False, label="OpenPose: also apply during Hires. fix", info="off by default: the adapter was trained at 512-1024 px")

            # ---------------- Canny ----------------
            with gr.Column(visible=False) as canny_box:
                gr.HTML("<b>Canny</b>")
                with FormRow():
                    c_pre = gr.Dropdown([PRE_CANNY, PRE_EDGE_NONE], value=PRE_CANNY, label="Canny Preprocessor")
                    c_res = gr.Slider(128, 2048, value=512, step=8, label="Canny Preprocessor Resolution", info="short side the edges are detected at")
                    c_btn = gr.Button("Preview Canny", size="sm")
                with FormRow() as c_thr_row:
                    c_low = gr.Slider(1, 255, value=100, step=1, label="Canny Low Threshold")
                    c_high = gr.Slider(1, 255, value=200, step=1, label="Canny High Threshold")
                c_invert = gr.Checkbox(False, label="Invert Canny Map", info="flip the map; with the preprocessor on None this turns a black-on-white line drawing into white lines on black (Canny already makes white on black). Resize and Fill pads with black after inverting")
                c_resize = gr.Radio(list(IM.RESIZE_MODES), value=IM.CROP_AND_RESIZE, label="Canny Resize Mode")
                with FormRow():
                    c_lora = gr.Dropdown(loras, value=LR.guess_lora(loras, LORA_WORDS[CANNY]), label="Canny LoRA", allow_custom_value=True)
                    c_refresh = ToolButton("🔄", elem_id=self.elem_id("canny_refresh"))
                    c_weight = gr.Slider(0.0, 2.0, value=DEFAULT_WEIGHT[CANNY], step=0.05, label="Canny LoRA Weight")
                with FormRow():
                    c_end = gr.Slider(0, 100, value=100, step=1, label="Control End (%)", info="share of the steps that get the Canny control; 100 = every step. Worth trying 40-60 when exact line-following is not needed")
                    c_fade = gr.Checkbox(True, label="Also switch the Canny LoRA off after Control End", info="only works with Settings 'Diffusion in Low Bits' = '... (fp16 LoRA)'; otherwise the LoRA stays on")
                c_hr = gr.Checkbox(False, label="Canny: also apply during Hires. fix", info="off by default: the edge map gets drawn into the image as light, jagged ghost lines")

            with gr.Accordion("Separate images (optional)", open=False):
                gr.HTML("Each ticked type uses the Control Image above unless you give it its own image here.")
                with gr.Row():
                    p_image = gr.Image(label="OpenPose Image (optional)", sources=["upload", "clipboard"], type="numpy", image_mode="RGB", height=240)
                    c_image = gr.Image(label="Canny Image (optional)", sources=["upload", "clipboard"], type="numpy", image_mode="RGB", height=240)

        types.change(_on_types, inputs=[types], outputs=[p_preview, c_preview, pose_box, canny_box], queue=False, show_progress=False)
        c_pre.change(_on_canny_pre, inputs=[c_pre], outputs=[c_res, c_thr_row], queue=False, show_progress=False)
        p_pre.change(_on_pose_pre, inputs=[p_pre], outputs=[p_res], queue=False, show_progress=False)
        c_refresh.click(_on_refresh, inputs=[c_lora], outputs=[c_lora], queue=False, show_progress=False)
        p_refresh.click(_on_refresh, inputs=[p_lora], outputs=[p_lora], queue=False, show_progress=False)
        c_btn.click(_canny_preview, inputs=[image, c_image, c_pre, c_res, c_low, c_high, c_invert], outputs=[c_preview])
        p_btn.click(_pose_preview, inputs=[image, p_image, p_pre, p_resize, p_res, p_people, p_hands, p_face, p_feet], outputs=[p_preview])

        return [
            enable, types, image, c_image, p_image,
            c_pre, c_res, c_low, c_high, c_invert, c_resize, c_lora, c_weight, c_end, c_fade, c_hr,
            p_pre, p_res, p_people, p_hands, p_face, p_feet, p_resize, p_lora, p_weight, p_strength, p_hr,
        ]  # fmt: skip

    # ------------------------------------------------------------------ hooks

    def before_process(self, p, *args, **kwargs):
        _cleanup()  # leftovers from an interrupted / crashed job

    def process(self, p, *args, **kwargs):
        _cleanup()
        try:
            a = dict(zip(KEYS, args))
            self._setup(p, a)
        except ControlError as e:
            logger.error(f"Anima Control: {e} -- interrupting instead of generating without the control")
            _cleanup()
            _toast(f"Anima Control: {e}")
            shared.state.interrupt()
        except Exception as e:
            logger.exception("Anima Control setup failed -- interrupting instead of generating without the control")
            _cleanup()
            _toast(f"Anima Control: {e}")
            shared.state.interrupt()

    # ---- setup of one control ----

    @staticmethod
    def _source(p, own_image, shared_image):
        """The type's own image, else the shared Control Image, else (img2img) the img2img input."""
        src = IM.to_rgb_array(own_image)
        if src is None:
            src = IM.to_rgb_array(shared_image)
        if src is None and isinstance(p, StableDiffusionProcessingImg2Img) and p.init_images:
            src = IM.to_rgb_array(p.init_images[0])
        if src is None:
            raise ControlError("no control image: upload one" + (" (an empty control image falls back to the img2img input, which is empty too)" if isinstance(p, StableDiffusionProcessingImg2Img) else ""))
        return src

    @classmethod
    def _build_canny(cls, p, a):
        src = cls._source(p, a["c_image"], a["image"])

        lora = a["c_lora"]
        use_lora = bool(lora) and lora != LR.NONE
        if use_lora:
            LR.lora_file(lora)  # raises if Forge does not know this LoRA
        else:
            logger.warning('Anima Control: "Canny LoRA" is None -- make sure the control LoRA is in your prompt, otherwise the reference does nothing useful')

        end = max(0.0, min(100.0, float(a["c_end"]))) / 100.0
        if end == 0.0:
            logger.warning("Anima Control: Control End is 0% -- the Canny control is never applied")

        detected = _canny_map(src, a["c_pre"], a["c_res"], a["c_low"], a["c_high"], bool(a["c_invert"]))
        resize_mode = a["c_resize"]
        job = _Job(CANNY, lambda w, h: IM.fit_to(detected, w, h, resize_mode), end, bool(a["c_hr"]), lora if use_lora else None, float(a["c_weight"]), fade=bool(a["c_fade"]))
        white = float((detected.max(axis=-1) > 127).mean())
        job.note = f"preprocessor {a['c_pre'].split(' ')[0]}{' + inverted' if a['c_invert'] else ''}, {detected.shape[1]}x{detected.shape[0]}, {white:.1%} white pixels"

        info = f"{CANNY}, {a['c_pre'].split(' ')[0]}"
        if a["c_pre"] == PRE_CANNY:
            info += f" {int(a['c_res'])} {int(a['c_low'])}/{int(a['c_high'])}"
        if a["c_invert"]:
            info += ", inverted"
        return job, f"{info}, {resize_mode}, end {end:.0%}"

    @classmethod
    def _build_pose(cls, p, a):
        src = cls._source(p, a["p_image"], a["image"])

        lora = a["p_lora"]
        if not lora or lora == LR.NONE:
            raise ControlError('pick the adapter file under "Pose Adapter (LoRA)": the control embedder is read from it')
        embedder = EM.load_embedder(LR.lora_file(lora))  # fails loudly if this is a plain LoRA

        resize_mode = a["p_resize"]
        strength = float(a["p_strength"]) if a["p_strength"] is not None else 1.0
        at = _build_skeleton(src, a["p_pre"], resize_mode, p.width, p.height, a["p_people"], a["p_hands"], a["p_face"], a["p_feet"], int(a["p_res"]))
        job = _Job(POSE, at, 1.0, bool(a["p_hr"]), lora, float(a["p_weight"]), embedder=embedder, strength=strength)
        return job, f"{POSE}, {a['p_pre'].split(' ')[0]}, {resize_mode}, strength {strength:g}"

    def _setup(self, p, a):
        if not a.get("enable"):
            return

        if not getattr(dynamic_args, "anima", False):
            msg = "Anima Control is enabled, but the loaded checkpoint is not Anima -- skipped"
            logger.warning(msg)
            _toast(msg)
            return

        ticked = [t for t in TYPES if t in (a.get("types") or [])]
        if not ticked:
            raise ControlError('no Control Type is ticked -- tick OpenPose and/or Canny, or switch "Anima Control" off')

        if CANNY in ticked and getattr(shared.opts, "anima_do_reference", False):
            logger.warning("Anima Control: '[Anima] Enable Reference' (Settings > Stable Diffusion) is ON. Forge then also feeds the img2img input / ImageStitch images as references; turn it off, this extension supplies its own reference and replaces Forge's during sampling.")

        # build and validate everything first, so a failing control never leaves the other one's tag in the prompt
        builders = {POSE: self._build_pose, CANNY: self._build_canny}
        jobs: dict[str, _Job] = {}
        infos = []
        for kind in ticked:
            try:
                job, head = builders[kind](p, a)
            except ControlError as e:
                raise ControlError(f"{kind}: {e}") from e
            jobs[kind] = job
            infos.append(f"{head}, LoRA {job.lora_name}:{job.lora_weight:g}, hires {'on' if job.apply_hr else 'off'}")

        names = [j.lora_name for j in jobs.values() if j.lora_name]
        if len(set(names)) != len(names):
            raise ControlError(f'OpenPose and Canny are both set to the LoRA "{names[0]}". They need two different adapter files')

        for job in jobs.values():
            if job.lora_name:
                LR.inject_lora(p, job.lora_name, job.lora_weight)

        JOBS.update(jobs)
        p.extra_generation_params["Anima Control"] = " + ".join(infos)

    # ---- sampling ----

    def process_before_every_sampling(self, p, *args, **kwargs):
        if not JOBS:
            return
        try:
            _disarm("the next sampling run")  # every run starts clean (also puts a faded LoRA back)
            stage = "hires" if getattr(p, "is_hr_pass", False) else "base"

            x = kwargs["x"]
            size = (int(x.shape[-1]) * IM.LATENT_F, int(x.shape[-2]) * IM.LATENT_F)
            vae = p.sd_model.forge_objects.vae
            unet = p.sd_model.forge_objects.unet
            dm = unet.model.diffusion_model

            for kind, job in JOBS.items():
                if stage == "hires" and not job.apply_hr:
                    logger.info(f"Anima Control [hires] {kind} skipped (Hires. fix control is off)")
                    continue
                if kind == POSE:
                    self._arm_pose(job, vae, dm, size, stage)
                else:
                    self._arm_canny(job, vae, unet, dm, size, stage)
        except Exception as e:
            logger.exception("Anima Control failed to prepare the control -- interrupting instead of sampling with the LoRA but no control")
            PT.reset_all()
            _toast(f"Anima Control: {e}")
            shared.state.interrupt()

    @staticmethod
    def _arm_pose(job, vae, dm, size, stage):
        tokens = job.cache.get(size)
        if tokens is None:
            latent = IM.encode_raw_latent(vae, job.at(*size))
            tokens = EM.embed(latent, *job.embedder, job.strength)
            job.cache[size] = tokens
        PT.install_embedder_hook(dm)
        PT.EMBED.arm(tokens, stage=stage)
        logger.info(f"Anima Control [{stage}] OpenPose: {size[0]}x{size[1]}, control tokens {tuple(tokens.shape)}, strength {job.strength:g}")

    def _arm_canny(self, job, vae, unet, dm, size, stage):
        ref = job.cache.get(size)
        if ref is None:
            ref = IM.encode_ref_latent(vae, job.at(*size))
            job.cache[size] = ref
        PT.install_forward_wrapper(dm)
        PT.install_memory_patch(unet.model)

        fade = self._lora_fade(job, unet) if (job.fade and job.lora_name and 0.0 < job.end < 1.0) else None
        PT.STATE.arm(ref, end=job.end, fade=fade, stage=stage)
        logger.info(f"Anima Control [{stage}] Canny: {size[0]}x{size[1]}, reference latent {tuple(ref.shape)}, control end {job.end:.0%}, map: {job.note}" + (", LoRA fades out after it" if fade else ""))

    @staticmethod
    def _lora_fade(job, unet):
        if not dynamic_args.online_lora:
            if not job.fade_warned:
                job.fade_warned = True
                logger.warning("Anima Control: Control End < 100% drops the reference, but the LoRA stays merged into the weights (it cannot be switched off mid-sampling). Set Settings 'Diffusion in Low Bits' to '... (fp16 LoRA)' to switch it off too.")
            return None
        fade = PT.LoraFade(unet, LR.lora_file(job.lora_name))
        if not fade.available():
            if not job.fade_warned:
                job.fade_warned = True
                logger.warning(f'Anima Control: could not find "{job.lora_name}" among the on-the-fly LoRA patches -- the LoRA will not be switched off after Control End')
            return None
        return fade

    def postprocess_batch(self, p, *args, **kwargs):
        # Sampling (incl. hires) is done: disarm before postprocess_image hooks such as ADetailer run their own passes.
        _disarm("the end of the batch")

    def postprocess(self, p, processed, *args, **kwargs):
        _cleanup()
