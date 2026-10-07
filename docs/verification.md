# What to check on a GPU

The CPU tests (see the README) cover the wiring against Forge Neo's own model code. These are the checks that need a real model on a GPU. Every step lists what you should see in the console (grep for `Anima Control`); if something differs, please open an issue with those log lines, the PNG info of the image, and your sampler / steps / CFG / size / checkpoint.

## 0. Setup

1. Put the adapters in your **LoRA** folder (not the ControlNet folder: the LoRA dropdown and `<lora:…>` only know the LoRA folder).
   - Canny: `anima-preview-canny-v0.2.safetensors` (132.24 MB; SHA256 `37bb7d8a96908ef88dfc501cb002b2e63b07b1f30c1487ddb1ac164d129867de` as listed on CivArchive).
   - OpenPose: `anima_pose_preview2.safetensors` from [Claquasse/Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose).
2. Settings → Stable Diffusion: keep **[Anima] Enable Reference** OFF; do not enable ImageStitch.
3. Remove the old **sd-forge-anima-pose** extension if you had it.
4. Use sampler ER SDE + Simple, 30 steps, CFG 4, 1024×1024 as a starting point (the Pose adapter's reference ComfyUI settings; the Canny adapter's author does not publish any).

## 1. Canny

Tick **Anima Control**, **Control Type = Canny** only. Use a **photo or a colored illustration** as the Control Image; Canny Preprocessor = Canny, resolution 512, thresholds 100 / 200; Canny LoRA = the Canny adapter, weight 0.7, Control End 100. Press **Preview Canny** first: you should see white lines on black.

Expect:
- `Anima Control [base] Canny: 1024x1024, reference latent (1, 16, 1, 128, 128), control end 100%, map: preprocessor Canny, …`
- the composition follows the lines; PNG info has an `Anima Control: "Canny, …"` line; the prompt got `<lora:…:0.7>` appended;
- no `NOT applied` warning at the end.

Compare with the same seed with the panel off.

**Line drawings.** Like any Canny control it expects edges taken from a picture. With a black-and-white line drawing, Canny makes a hollow double outline of every line (or loses thin lines on a small image). To feed a drawing directly: preprocessor *None* plus **Invert Canny Map** (black-on-white becomes white-on-black), and check the `map: … N% white pixels` log line: a thin-line drawing should be a few percent white, not 90%.

## 2. Control End

Same seed, only **Control End (%)** changes: 100, 50, 30.

Expect `Control End reached at 50% of the steps -- reference dropped` once per image. Step time should drop after that point (the reference doubles the token sequence). Unless Settings → *Diffusion in Low Bits* is set to a `… (fp16 LoRA)` option, the LoRA stays merged and a warning says so once.

Optional, with *Diffusion in Low Bits* = `Automatic (fp16 LoRA)`: Control End 50 with "Also switch the Canny LoRA off after Control End" should log `LoRA switched off (N patches)`. Then render the same seed with Control End 100 again: it must be identical to the first Control End 100 image. A difference means the LoRA strength was not restored.

## 3. Hires. fix

Enable Hires. fix. With **Canny: also apply during Hires. fix** off (default) expect `[hires] Canny skipped`. Turn it on and expect `[hires] Canny: …`; look for light, jagged ghost outlines.

## 4. Other cases

- **Batch size 2**: renders two images (Forge's own reference code fails here with a `torch.cat` size error; the extension sizes the reference to the batch).
- **img2img** with an empty Control Image: the img2img input is used.
- **Errors are loud**: panel on without any control image, with no Control Type ticked, with an unknown LoRA, or with a plain LoRA where the OpenPose adapter is needed → the job is interrupted with one error line.
- **Interrupt** mid-generation, then switch the panel off and render: the result must match a run without the extension.

## 5. OpenPose

**Control Type = OpenPose** only. Pose Adapter = the Pose adapter, weight 1.0, Control Strength 1.0, detector *DWPose (rtmlib)*. The first use downloads about 316 MB of detector models. A photo of a person; press **Preview Skeleton** (thin lines on black), then generate.

Expect `Anima Control [base] OpenPose: 1024x1024, control tokens (1, 1, 64, 64, 2048), strength 1`.

## 6. Both at once

Tick **both** Control Types. By default both use the one Control Image; *Separate images (optional)* gives either its own. The two LoRAs must be two different files. Start with lower weights (Canny 0.5, Pose 0.7).

Neither adapter's author trained or tested stacking, so this is the combination with the least outside evidence. The wiring is verified by the CPU tests (the reference frame and the control tokens go in side by side without interfering), and the maintainer has run it on a GPU, where it works. Render three images with the same seed: Canny only, OpenPose only, both. Expect both log lines (`OpenPose` and `Canny`) and a PNG info line with the two parts joined by ` + `. Setting Canny's Control End to 50 should drop only Canny at half way while OpenPose keeps applying.
