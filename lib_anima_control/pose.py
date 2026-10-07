"""Skeleton detection + rendering in the style the Anima Control Pose adapter was trained on.

Adapted from sd-forge-anima-pose (lib_anima_pose/pose.py).

The adapter was trained on thin whole-body skeletons on a black background: DWPose-family
keypoints (COCO-WholeBody, 133 points), body limbs as coloured 2 px sticks, white joint
dots, and small dots for hands/face. The model card says this plain thin render is the only
one it saw in training, so we reproduce it instead of using Forge's own OpenPose renderer.

COCO-WholeBody-133 layout: body 0-16, feet 17-22, face 23-90, left hand 91-111, right hand 112-132.
"""
import os
import shutil

import cv2
import numpy as np

from lib_anima_control import ControlError

BODY = list(range(17))
FEET = list(range(17, 23))
FACE = list(range(23, 91))
HAND = list(range(91, 133))

LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12), (0, 5), (0, 6)]
_PALETTE = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (128, 255, 0), (0, 255, 0), (0, 255, 128), (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 128), (180, 180, 180), (220, 220, 220)]
LIMB_COLOR = {limb: _PALETTE[i] for i, limb in enumerate(LIMBS)}

STICK_PX, BODY_DOT_PX, FEET_DOT_PX, EXTRA_DOT_PX = 2, 3, 1, 2
HAND_COLOR, WHITE = (0, 255, 255), (255, 255, 255)
SCORE_THRESHOLD = 0.3
MIN_BODY_POINTS = 5  # confident body keypoints needed to count as a person


class NoPersonError(ControlError):
    pass


def render(people_kp, people_sc, width: int, height: int, hands=True, face=True, feet=True, thr=SCORE_THRESHOLD) -> np.ndarray:
    """people_kp: (N, 133, 2) pixel coords on a width x height canvas, people_sc: (N, 133). Returns RGB uint8."""
    canvas = np.zeros((height, width, 3), np.uint8)

    def pt(kp, i):
        return int(round(kp[i][0])), int(round(kp[i][1]))

    for kp, sc in zip(people_kp, people_sc):
        for a, b in LIMBS:
            if sc[a] >= thr and sc[b] >= thr:
                cv2.line(canvas, pt(kp, a), pt(kp, b), LIMB_COLOR[(a, b)], STICK_PX)
        for i in BODY:
            if sc[i] >= thr:
                cv2.circle(canvas, pt(kp, i), BODY_DOT_PX, WHITE, -1)
        if feet:
            for i in FEET:
                if sc[i] >= thr:
                    cv2.circle(canvas, pt(kp, i), FEET_DOT_PX, WHITE, -1)
        if hands:
            for i in HAND:
                if sc[i] >= thr:
                    cv2.circle(canvas, pt(kp, i), EXTRA_DOT_PX, HAND_COLOR, -1)
        if face:
            for i in FACE:
                if sc[i] >= thr:
                    cv2.circle(canvas, pt(kp, i), EXTRA_DOT_PX, WHITE, -1)
    return canvas


# ---------------------------------------------------------------------------
# detection (rtmlib RTMW whole-body, the same family the adapter's training skeletons came from)
# ---------------------------------------------------------------------------

_HF_REPO = "Claquasse/Anima-Control-Pose"
_HF_MIRROR = "https://hf-mirror.com"
_ONNX = (
    "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx",
    "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx",
)
_estimator = None


def _ckpt_dir() -> str:
    try:
        from rtmlib.tools.file import _get_rtmhub_dir

        return os.path.join(_get_rtmhub_dir(), "checkpoints")
    except Exception:
        base = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        return os.path.join(base, "rtmlib", "hub", "checkpoints")


def _plausible(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1_000_000  # an HTML error page is tiny


def _fetch(name: str, dst: str) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        shutil.copyfile(hf_hub_download(repo_id=_HF_REPO, filename="detector/" + name), dst)
        if _plausible(dst):
            return True
    except Exception:
        pass
    try:
        import urllib.request

        with urllib.request.urlopen(f"{_HF_MIRROR}/{_HF_REPO}/resolve/main/detector/{name}", timeout=120) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        if _plausible(dst):
            return True
    except Exception:
        pass
    if os.path.exists(dst) and not _plausible(dst):
        os.remove(dst)
    return False


def ensure_detector_files() -> None:
    """Seed rtmlib's cache from the adapter's Hugging Face repo (or a mirror) so it never has to
    reach download.openmmlab.com, which is slow or blocked on many networks."""
    ckpt = _ckpt_dir()
    os.makedirs(ckpt, exist_ok=True)
    for name in _ONNX:
        stale_zip = os.path.join(ckpt, os.path.splitext(name)[0] + ".zip")  # leftover of a failed rtmlib download breaks later runs
        if os.path.exists(stale_zip):
            os.remove(stale_zip)
        dst = os.path.join(ckpt, name)
        if not _plausible(dst) and not _fetch(name, dst):
            raise RuntimeError(f"Could not download {name}. Download detector/{name} from https://huggingface.co/{_HF_REPO} by hand and put it in {ckpt}")


def _get_estimator():
    global _estimator
    if _estimator is None:
        ensure_detector_files()
        from rtmlib import Wholebody

        _estimator = Wholebody(to_openpose=False, mode="balanced", backend="onnxruntime", device="cpu")
    return _estimator


def detect(rgb: np.ndarray, estimator=None):
    """-> (kp (N, 133, 2) in pixel coords of `rgb`, sc (N, 133)). Raises NoPersonError."""
    est = estimator or _get_estimator()
    kp, sc = est(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if kp is None or len(kp) == 0:
        raise NoPersonError("no person detected in the control image")
    kp, sc = np.asarray(kp, float), np.asarray(sc, float)
    # rtmlib falls back to a whole-image box when its person detector finds nobody, which yields a
    # junk low-confidence skeleton on e.g. a blank image. Keep only people with a believable body.
    real = [n for n in range(len(kp)) if int((sc[n][BODY] >= SCORE_THRESHOLD).sum()) >= MIN_BODY_POINTS]
    if not real:
        raise NoPersonError("no person detected in the control image")
    return kp[real], sc[real]


def pick_largest(kp: np.ndarray, sc: np.ndarray, thr=SCORE_THRESHOLD):
    """Keep only the person whose confident body/feet keypoints span the biggest box."""
    best, best_area = 0, -1.0
    for n in range(len(kp)):
        idx = [i for i in BODY + FEET if sc[n][i] >= thr]
        if len(idx) < 2:
            continue
        pts = kp[n][idx]
        area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
        if area > best_area:
            best, best_area = n, area
    return kp[best : best + 1], sc[best : best + 1]
