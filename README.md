# sd-forge-anima-control

Control-LoRA support for **Anima** in [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo): a ControlNet-style panel (control image, preprocessor, preview, resize mode, LoRA picker) for two control adapters, **OpenPose** and **Canny**. Tick one, or **both at once**.

[中文说明](README_zh.md)

> **Status: lightly tested on a GPU.** On an RTX 5070 Ti laptop with Forge Neo `neo 2.29` and the checkpoint `indigoFurryMixAnima_v10`, both controls armed correctly with no errors, and the Canny reference made each step about 2.2x slower until Control End dropped it, as designed. Both controls ticked at once has also been run on a GPU and works. Image quality has not been evaluated across checkpoints, LoRA weights and settings. Everything that can be checked without a GPU has been (see [Verification](#verification)); [docs/verification.md](docs/verification.md) lists the real-machine steps. Please report how real generations look.
>
> **Tip, Canny with line drawings:** like any Canny control, it expects edges extracted from a picture. A black-and-white line drawing as the reference usually goes badly: Canny turns every line into a hollow double outline, or the model copies the drawing and the result stays uncolored. Use a photo or a colored illustration as the reference, or see *Invert Canny Map* below.

This extension contains **no model weights**. It feeds the control images to the model and adds the `<lora:…>` tags to your prompt.

## What it does

Anima has no real ControlNet. Both adapters are "control LoRAs", and they put the control image into the model in different ways, so each gets its own injection:

| Control Type | Adapter | How the control enters the model |
| --- | --- | --- |
| **Canny** | levzzz's [anima-preview-canny-v0.2](https://civitai.com/models/2443202) | The edge map is VAE-encoded (`process_in`, sampler space) and appended as a **reference latent** along the T axis (like Flux.2 ReferenceLatent). The reference uses the same timestep as the target ("current t"). Forge's own reference code in `backend/nn/anima.py` does the concatenation; this extension supplies the latent. This is the same thing ComfyUI PR [#13392](https://github.com/Comfy-Org/ComfyUI/pull/13392) does (`torch.cat` on dim 2, `process_latent_in` on the references). |
| **OpenPose** | Claquasse's [Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose) Preview-2 | The skeleton is VAE-encoded (raw latent), run through the adapter's trained linear *control embedder* and **added to the output of the DiT's patch embedding**. The sequence length does not change. The same file is both the LoRA and the embedder. |

**Tick one, or both.** The two paths do not interfere with each other mechanically: the reference frame is appended after the target, and the control tokens are added to the target frame only. Both LoRAs are added to the prompt and need to be two different files. Neither adapter's author trained or tested the two together; running both at once has been tried on a GPU and works. If they fight each other, lower the LoRA weights.

OpenPose was merged in from [sd-forge-anima-pose](#credits); it was a separate extension before.

## Install

1. Put the adapters in your LoRA folder (not included). **Not** the ControlNet folder: the LoRA dropdown and `<lora:…>` only know the LoRA folder.
   - Canny: `anima-preview-canny-v0.2.safetensors` (132.24 MB). Download: [CivitAI](https://civitai.com/models/2443202) or a mirror listed on [CivArchive](https://civarchive.com/models/2443202).
   - OpenPose: `anima_pose_preview2.safetensors` from [Claquasse/Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose).
2. Put this folder in Forge Neo's `extensions/` and restart. `install.py` installs `rtmlib` (only OpenPose uses it).
3. OpenPose only: on first use the two detector ONNX files (~316 MB) are fetched from the adapter's Hugging Face repo, falling back to hf-mirror.com. If both fail, download `detector/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` and `detector/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx` by hand into `~/.cache/rtmlib/hub/checkpoints/`.
4. If you had **sd-forge-anima-pose** installed, remove or disable it. Both hook the patch embedding; with both installed and enabled the pose control is applied twice.

## Use

1. Load an Anima checkpoint. In txt2img / img2img open **Anima Control** and tick its enable box.
2. Under **Control Type** tick **OpenPose**, **Canny**, or both. Each ticked type shows its own settings.
3. Drop in the **Control Image**. Every ticked type processes it with its own preprocessor (the **Preview** buttons show the results). Want a different image for one type? Open *Separate images (optional)*. An edge map / skeleton you already have: set that type's preprocessor to *None*. In img2img an empty control image falls back to the img2img input.
4. Pick the adapter under **Pose Adapter (LoRA)** / **Canny LoRA**. The tag `<lora:name:weight>` is appended to the positive prompt and the Hires. fix prompt. If your prompt already has a tag for that LoRA, yours (and its weight) is kept.
5. Write your prompt and generate.

Failures are loud on purpose: a missing control image, no Control Type ticked, an unknown LoRA, a plain LoRA where the OpenPose adapter is needed, the same LoRA for both types, a failed VAE encode, or a person-less image **interrupts the job** with a message in the console (and a toast) rather than quietly generating without control. With both types ticked, everything is checked before anything is added to the prompt, so a failure in one never leaves the other's tag behind.

## Options

**Shared**

- **Control Type**: tick **OpenPose**, **Canny**, or both. OpenPose is ticked by default.
- **Control Image**: one image for every ticked type; each type applies its own preprocessor to it.
- **Separate images (optional)**: give OpenPose and/or Canny its own image. An empty slot falls back to the Control Image (and, in img2img, to the img2img input).

**Canny** (settings shown when ticked)

- **Canny Preprocessor**: *Canny* or *None* (the image is already an edge map).
- **Canny Preprocessor Resolution**: short side the edges are detected at, default 512.
- **Canny Low / High Threshold**: default 100 / 200 (`cv2.Canny` on the gray image; white lines on black). **These are the most likely values, not confirmed facts**, see [Unconfirmed](#unconfirmed).
- **Invert Canny Map**: flips the map. Mainly for line drawings (black lines on white): Canny turns every line into a hollow double outline, or loses thin lines when the picture is small; with the preprocessor on *None* plus Invert the drawing becomes white lines on black, which is what Canny output looks like. Resize and Fill pads with black after inverting. Also the quickest way to test the opposite polarity.
- **Canny Resize Mode**: Crop and Resize / Just Resize / Resize and Fill, as in ControlNet. Resize and Fill pads with black, which is "no signal" for edges.
- **Canny LoRA / Canny LoRA Weight**: weight 0.7 by default.
- **Control End (%)**: the share of sampling steps that get the control, default 100. In ComfyUI PR #13392 a tester (Kosinkadink) reported that using the reference latents + LoRA for only half of the steps or less "vastly improved" quality when not adhering strictly to individual control lines, and the adapter's author replied that a selected step range often works better, perhaps because the dataset and LoRA were made for an older preview version of Anima. So 40–60 is worth trying. Steps are counted like Forge counts them for prompt scheduling (`CFGDenoiser.step / total_steps`).
- **Also switch the Canny LoRA off after Control End**: that test applied the LoRA together with the reference for the same step range. Forge can only change a LoRA mid-sampling when it patches LoRAs on the fly (Settings → *Diffusion in Low Bits* = "… (fp16 LoRA)", slower). Otherwise the LoRA stays merged, only the reference is dropped, and the console says so once. Only the Canny LoRA is touched; the pose adapter keeps its weight.
- **Canny: also apply during Hires. fix**: off by default. Canny + Hires. fix draws the edge map into the image as light, jagged ghost lines. With it off, the second pass runs without the reference (the LoRA tag is still in the Hires prompt).

**OpenPose** (settings shown when ticked)

- **Pose Detector**: *DWPose (rtmlib)* is the default because it is the family the training skeletons came from, drawn in the thin style the model card says is the only one seen in training. *Forge dw_openpose_full* is a weaker fallback (its resolution slider appears only then). *None* = the image is already a skeleton.
- **People** (largest person / everyone), **Hands / Face / Feet**.
- **Pose Resize Mode**, **Pose Adapter (LoRA)**, **Pose LoRA Weight** (1.0 by default).
- **Control Strength**: 0 = off, 1 = follow the skeleton, up to 2.
- **OpenPose: also apply during Hires. fix**: off by default; the adapter was trained at 512–1024 px. There is no Control End here: nothing suggests it helps for this adapter.

PNG info gets one line, e.g. `Anima Control: "Canny, Canny 512 100/200, Crop and Resize, end 50%, LoRA anima-preview-canny-v0.2:0.7, hires off + OpenPose, None, Crop and Resize, strength 1, LoRA animaControlPose_preview2:1, hires off"` (the second part only when both types are ticked). Pasting parameters does not restore the panel.

## Works with / does not

- **Keep "[Anima] Enable Reference" (Settings → Stable Diffusion) OFF and ImageStitch disabled** when using Canny. That native path also puts the img2img input / stitched images into `dynamic_args.ref_latents`. If it is on, this extension logs a warning and its control latent replaces Forge's during sampling; the native references are restored afterwards.
- **Batch size > 1 works.** Forge batches cond and uncond together, so with batch size 2 the model sees 4 items, and Forge's own reference code only handles exactly 2 (it fails with a `torch.cat` size error). The extension sizes the reference to the batch.
- **CFG > 1**: the control is applied to both cond and uncond, as Forge's own code does.
- **VRAM**: with Canny the token sequence is twice as long. The extension raises Forge's memory estimate for the model while a reference is applied, so Forge is less likely to batch cond+uncond into one OOM-prone call.
- **ADetailer**: the controls are disarmed when the batch finishes sampling, before post-processing runs. If ADetailer reuses your main prompt it would also pick up the control LoRA tags but get no control (not tested); give it its own prompt.
- LoRA scheduling syntax (`<lora:name:[...]>`) needs on-the-fly LoRA and only reads the main prompt, so it does not see the tags this extension adds; set the LoRA dropdown to None and write the tag yourself if you want it.

## What is patched

Nothing under `backend/` is edited. All runtime patches are in one file, [`lib_anima_control/patches.py`](lib_anima_control/patches.py):

1. `diffusion_model.forward` (instance attribute, once per model object, inert unless armed): sets `dynamic_args.ref_latents` for the duration of the call and restores the previous value in `finally`. Sizes the reference to `x.shape[0]`. Implements Control End. (Canny)
2. `KModel.memory_required` (instance attribute): multiplied while a reference is applied. (Canny)
3. `OnlineLoRAPatch.patch[0][0]` of the Canny LoRA: zeroed after Control End and restored when the batch ends (online LoRA only).
4. A `cfg_denoiser` callback that records `CFGDenoiser.step / total_steps`.
5. A forward hook on `diffusion_model.x_embedder`: adds the control tokens; a no-op unless armed. When a reference frame is in the sequence the tokens go to the target frame only. (OpenPose)

The Canny and OpenPose states are separate objects (`patches.STATE`, `patches.EMBED`), which is what lets both be armed at once.

Hooks used: `process`, `process_before_every_sampling` (after conditioning is encoded, which matters because Forge's Anima engine clears `ref_latents` every time it encodes a positive prompt while "[Anima] Enable Reference" is off), `postprocess_batch`, `postprocess`.

## Verification

Tests run Forge Neo's real `backend/nn/anima.py` with small random weights on CPU, plus Forge's real `modules` for the panel:

```
<Forge Neo>\venv\Scripts\python.exe tests\test_reference.py   # Canny path: 74 checks (+6 on the real LoRA file when ANIMA_CANNY_FILE is set)
<Forge Neo>\venv\Scripts\python.exe tests\test_script.py      # panel + hook sequence (txt2img, img2img, Hires. fix), each control alone and both together: 145 checks
set ANIMA_POSE_FILE=path\to\anima_pose_preview2.safetensors
<Forge Neo>\venv\Scripts\python.exe tests\test_pose.py        # OpenPose path + the real adapter file: 30 checks
```

Covered, among other things:

- output shape equals the no-reference shape (the reference part is cropped), for batch 1 / 2 / 3 / 4 and odd latent sizes;
- the armed wrapper reproduces Forge's native `ref_latents` path bit for bit (batch 1 and 2) and `dynamic_args.ref_latents` is empty again after every forward, including after an exception;
- the reference and the target share one timestep embedding;
- after Control End the output is identical to a run without any reference; LoRA strength is restored afterwards;
- the native path really does fail at batch 4 (so the batch sizing is needed);
- **both controls at once**: the output equals Forge's native reference path plus the control tokens on the target frame only, differs from either control alone, holds for batch 2 / 3 / 4, Hires. fix with every on/off combination, Control End dropping only Canny while OpenPose continues, the LoRA fade touching only the Canny LoRA, atomic failure (nothing left in the prompt), and cleanup after an exception;
- the Control Type box is a non-exclusive CheckboxGroup; the shared Control Image / per-type images / img2img fallback resolve in that order; panel on with nothing ticked is an error;
- no two controls in the panel share a label (Forge keys `ui-config.json` by label, and a repeat would hand the second control the first one's default);
- Hires. fix on/off, references cached per size, img2img input fallback, every failure path above;
- `encode_ref_latent` against the real Anima Wan VAE weights on CPU: shape `(1, 16, 1, h, w)` and identical to Forge's own `encode_first_stage` result;
- the real `anima-preview-canny-v0.2.safetensors` (SHA256 matches the CivArchive page): a plain rank-32 LoRA with no control embedder; its 448 targets over 28 blocks all exist in Forge's Anima;
- OpenPose: embedder ordering equals Forge's `PatchEmbed`, exactness at strength 0, odd sizes, the real adapter file (embedder is `Linear(64 → 2048)`, all 448 DiT and 60 `llm_adapter` LoRA targets exist in Forge's Anima).

**Not covered by these tests** (CPU, random weights): image quality and real-weight GPU runs (those were done by hand, see the status note at the top), the panel in a real browser, the real DWPose detector (its ONNX files were not downloaded), LoRA fade on a real on-the-fly LoRA model, and the effect of the memory estimate on real VRAM use.

Built against Forge Neo `neo` branch, commit `534e6ecce1039dbe5647e2c391b84a75aaa61d35` ("meta device", 2026-09-13). It was not checked against newer upstream commits.

## Unconfirmed

- **How the Canny LoRA was trained**: Canny thresholds, edge-map resolution, line polarity (white on black or the reverse), and whether the reference had the target's resolution are not stated on its CivitAI/CivArchive page or in the text of ComfyUI PR #13392, and the file's own `__metadata__` holds only `format: pt`. levzzz5154/diffusion-pipe, named there as the training code, has only the branches `main`, `feature/anima-distillation` and `feature/lycoris-loha-lokr`, and a code search for "canny" in it finds nothing (the search index may be incomplete; no dataset-processing code was found or read). The PR's example workflow [`Anima-RefLatent.json`](https://github.com/user-attachments/files/26686944/Anima-RefLatent.json) probably shows the Canny node, resize and sampler settings, but it could not be fetched here: **open it in a text editor and compare**. Defaults here (`cv2.Canny(100, 200)`, white on black, detected at 512 px, reference resized to the generation size) are guesses. Thresholds and resolution are adjustable, and *Invert Canny Map* tests the opposite polarity. Note that ComfyUI's built-in Canny node (kornia, defaults 0.4 / 0.8 on a 0–1 scale) is not numerically the same as `cv2.Canny`, so if the workflow uses it, expect to tune the thresholds until the Preview looks like the CivitAI example edge maps.
- The LoRA was trained for an Anima **preview** version; results on Anima-Base v1.0 may be weaker.
- Its **license** is not stated on the page.
- A user on the CivArchive page shared a Forge Neo patch and talks about selecting "the correct control method". Its content was not seen, so whether the LoRA needs a reference method other than "append along T, shared timestep" is unknown.
- Whether switching the LoRA off after Control End leaves any residue on a real model (the strength is restored and tested on stand-ins, not on real weights).
- How Canny and OpenPose behave **together** beyond the maintainer's own GPU runs (where it works). Neither author trained or tested stacking, so how the two LoRAs interact at other weights and checkpoints is unknown. The code path is verified to equal "each control's own mechanism, side by side".

## Known issues

- The control embedder / reference latent are made at exactly the generation size (`latent × 8`); the Hires. fix pass re-encodes at the upscaled size when enabled.
- Edge maps are binary 1-px lines; resizing them to the generation size (when it differs from the preprocessing resolution) can soften or thin them. If the control looks weak, try a Preprocessor Resolution closer to the generation size.
- UI defaults saved by earlier local builds of this extension (two separate panels) are not carried over: the panel and its labels changed.

## Credits

- Canny adapter: levzzz ([CivitAI](https://civitai.com/models/2443202)); training code: [levzzz5154/diffusion-pipe](https://github.com/levzzz5154/diffusion-pipe). Reference-latent behaviour as in ComfyUI PR [#13392](https://github.com/Comfy-Org/ComfyUI/pull/13392).
- OpenPose adapter and ComfyUI reference implementation: [Claquasse/Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose). `lib_anima_control/embedder.py`, `pose.py` and the OpenPose tests were adapted from the `sd-forge-anima-pose` extension.
- Panel structure follows [sd-forge-krea2-control](https://github.com/matkaparka/sd-forge-krea2-control).
- Anima by CircleStone Labs (non-commercial license; the Pose adapter's card also cites the NVIDIA Open Model License), built on Cosmos-Predict2.

## License

Copyright (c) 2026 matkaparka. AGPL-3.0, like Forge Neo and sd-forge-krea2-control (see [LICENSE](LICENSE)). Model weights keep their own licenses: Anima is CircleStone Labs' non-commercial license; the Pose adapter's weights are non-commercial; the Canny LoRA's license is not stated. Images you generate are not restricted by this extension's license.
