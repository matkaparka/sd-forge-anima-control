# sd-forge-anima-control

在 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 里给 **Anima** 加控制：ControlNet 风格的面板（控制图、预处理器、预览、Resize Mode、LoRA 下拉框），支持 **OpenPose** 和 **Canny** 两种控制 LoRA。勾选一个，或者**两个同时勾选**。

[English](README.md)

> **状态：在 GPU 上做过初步测试。** 在 RTX 5070 Ti 笔记本、Forge Neo `neo 2.29`、底模 `indigoFurryMixAnima_v10` 上，两种控制都正确布防，没有报错，Canny 的参考让每一步慢了约 2.2 倍，直到 Control End 按设计把它撤掉。出图质量还没有在不同配置下评估过。不需要 GPU 就能验证的部分都验证过了（见下方"验证情况"），真机验证步骤在 [docs/verification.md](docs/verification.md)。欢迎反馈实际出图效果。
>
> **提示，Canny 配线稿：** 和所有 Canny 控制一样，它期望的是从图片里提取出来的边缘。用黑白线稿做参考通常效果不好：Canny 会把每条线变成空心的双线轮廓，或者模型直接照抄线稿，结果没有上色。请用照片或带颜色的插画做参考，或者看下面的 *Invert Canny Map*。

本扩展**不含任何模型权重**。它只负责把控制图喂给模型，并在提示词里自动追加 `<lora:…>`。

## 原理

Anima 没有真正的 ControlNet，两个适配器都是"控制 LoRA"，而且把控制图送进模型的方式不一样，所以各走各的注入：

| Control Type | 适配器 | 控制图怎么进模型 |
| --- | --- | --- |
| **Canny** | levzzz 的 [anima-preview-canny-v0.2](https://civitai.com/models/2443202) | 边缘图经 VAE 编码（`process_in`，采样器空间），作为**参考 latent** 沿 T 维拼在目标后面（和 Flux.2 的 ReferenceLatent 一样），参考和目标共用同一个 timestep（"current t"）。拼接由 Forge 自带的 `backend/nn/anima.py` 完成，本扩展负责提供 latent。ComfyUI PR [#13392](https://github.com/Comfy-Org/ComfyUI/pull/13392) 做的是同一件事（`dim=2` 上 `torch.cat`，参考过 `process_latent_in`）。 |
| **OpenPose** | Claquasse 的 [Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose) Preview-2 | 骨架图经 VAE 编码（原始 latent），再过适配器里训练好的线性 *control embedder*，**加到 DiT patch embedding 的输出上**，序列长度不变。同一个文件既是 LoRA 也是 embedder。 |

**可以只勾一个，也可以两个同时勾。** 两条路径在机制上互不干扰：参考帧拼在目标帧后面，控制 token 只加到目标帧上。两个 LoRA 都会加进提示词，必须是两个不同的文件。两个分别训练的适配器**叠在一起**效果好不好，两位作者都没有说过，也还没有在 GPU 上测过；建议先把两个 LoRA 权重都调低一些。

OpenPose 是从 [sd-forge-anima-pose](#致谢) 合并进来的，之前是独立扩展。

## 安装

1. 把适配器放进 LoRA 目录（不含在本扩展里）。**不是** ControlNet 目录：LoRA 下拉框和 `<lora:…>` 只认 LoRA 目录。
   - Canny：`anima-preview-canny-v0.2.safetensors`（132.24 MB），下载见 [CivitAI](https://civitai.com/models/2443202)，或 [CivArchive](https://civarchive.com/models/2443202) 上列的镜像。
   - OpenPose：[Claquasse/Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose) 里的 `anima_pose_preview2.safetensors`。
2. 把本文件夹放进 Forge Neo 的 `extensions/`，重启。`install.py` 会装 `rtmlib`（只有 OpenPose 用）。
3. 只有 OpenPose：首次使用会从适配器的 Hugging Face 仓库下载两个检测器 ONNX（约 316 MB），失败再走 hf-mirror.com。都失败的话手动下载 `detector/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx` 和 `detector/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx`，放到 `~/.cache/rtmlib/hub/checkpoints/`（Windows 为用户目录下同名路径）。
4. 如果装过 **sd-forge-anima-pose**，请删掉或停用。两者都挂在 patch embedding 上，同时安装并启用的话姿势控制会被加两遍。

## 使用

1. 加载 Anima 底模，在 txt2img / img2img 里展开 **Anima Control** 并勾选启用。
2. 在 **Control Type** 里勾选 **OpenPose**、**Canny**，或者两个都勾。每个被勾选的类型会显示它自己的设置。
3. 放入 **Control Image**。每个被勾选的类型都用自己的预处理器处理它（**Preview** 按钮可以先看效果）。想给某个类型单独用另一张图，展开 *Separate images (optional)*。已经做好的边缘图 / 骨架图，把该类型的预处理器设为 *None*。img2img 里控制图为空时，用 img2img 的输入图。
4. 在 **Pose Adapter (LoRA)** / **Canny LoRA** 里选适配器。`<lora:名字:权重>` 会追加到正向提示词和高清修复提示词；如果提示词里已经写了同名 LoRA，就保留你自己的标签和权重。
5. 正常写提示词，生成。

出错时会**明确中断任务**，而不是悄悄不带控制出图：没有勾选任何 Control Type、没有控制图、LoRA 找不到、OpenPose 选了普通 LoRA、两个类型选了同一个 LoRA、VAE 编码失败、图里没有人，都会在控制台报错（并弹出提示）。两个类型同时勾选时，会先把全部检查做完，再往提示词里加标签，所以其中一个出错不会把另一个的标签留在提示词里。

## 参数

**通用**

- **Control Type**：勾选 **OpenPose**、**Canny**，或者两个都勾。默认勾选 OpenPose。
- **Control Image**：所有被勾选的类型共用的一张图，每个类型用自己的预处理器处理它。
- **Separate images (optional)**：给 OpenPose 和 / 或 Canny 单独指定图。留空就退回到 Control Image（img2img 里再退回到 img2img 的输入图）。

**Canny**（勾选后显示）

- **Canny Preprocessor**：*Canny*，或 *None*（图本身就是边缘图）。
- **Canny Preprocessor Resolution**：Canny 边缘检测时的短边尺寸，默认 512。
- **Canny Low / High Threshold**：默认 100 / 200（灰度图上跑 `cv2.Canny`，白线黑底）。**这是最可能的取值，不是已确认的事实**，见"未确认事项"。
- **Invert Canny Map**：把边缘图反相。主要给线稿（白底黑线）用：对线稿跑 Canny，每条线会变成空心的双线轮廓，图小的时候细线还会丢；预处理器选 *None* 再勾 Invert，线稿就变成黑底白线，和 Canny 的输出一致。Resize and Fill 在反相之后用黑色填充。也是测试相反极性最快的办法。
- **Canny Resize Mode**：Crop and Resize / Just Resize / Resize and Fill，含义和 ControlNet 一样。Resize and Fill 用黑色填充，对边缘图来说黑色就是"无信号"。
- **Canny LoRA / Canny LoRA Weight**：权重默认 0.7。
- **Control End (%)**：只在前面这部分步数里加控制，默认 100。ComfyUI PR #13392 里 Kosinkadink 实测：只在一半步数或更少的范围内使用参考 latent + LoRA，在不要求逐线贴合时质量"大幅提升"；适配器作者回复说，限定步数范围通常效果更好，可能是因为数据集和 LoRA 是针对旧的 preview 版 Anima 做的。所以值得试 40–60。步数按 Forge 给提示词调度用的方式计算（`CFGDenoiser.step / total_steps`）。
- **Also switch the Canny LoRA off after Control End**：上面那次测试是 LoRA 和参考在同一段步数里一起用。Forge 只有在"即时打 LoRA"时才能在采样中途改 LoRA 权重（Settings 里 *Diffusion in Low Bits* 选 "… (fp16 LoRA)"，会慢一些）。否则 LoRA 一直合并在权重里，只会撤掉参考，控制台会提示一次。只会动 Canny 的 LoRA，姿势适配器权重不变。
- **Canny: also apply during Hires. fix**：默认关。Canny + Hires. fix 会把边缘图"画"进成图，出现浅色锯齿鬼影。关闭时第二阶段不带参考（LoRA 标签仍在高清修复提示词里）。

**OpenPose**（勾选后显示）

- **Pose Detector**：默认 *DWPose (rtmlib)*，和训练骨架同源，按模型卡说的训练时唯一见过的细线风格绘制；*Forge dw_openpose_full* 是较弱的备选（只有选它时才会出现分辨率滑块）；*None* 表示图本身就是骨架。
- **People**（最大的人 / 所有人）、**Hands / Face / Feet**。
- **Pose Resize Mode**、**Pose Adapter (LoRA)**、**Pose LoRA Weight**（默认 1.0）。
- **Control Strength**：0 关闭，1 跟随骨架，最高 2。
- **OpenPose: also apply during Hires. fix**：默认关；适配器只训练到 512–1024 px。这里没有 Control End：没有迹象表明它对这个适配器有帮助。

PNG info 里会写一行，例如 `Anima Control: "Canny, Canny 512 100/200, Crop and Resize, end 50%, LoRA anima-preview-canny-v0.2:0.7, hires off + OpenPose, None, Crop and Resize, strength 1, LoRA animaControlPose_preview2:1, hires off"`（第二段只在两个类型都勾选时才有）。粘贴参数时不会自动恢复面板设置。

## 和 Forge 其它功能一起用

- 用 Canny 时，**Settings → Stable Diffusion 里的 "[Anima] Enable Reference" 要关掉，ImageStitch 也不要同时用**。那条原生路径同样会把 img2img 输入图 / 拼接图放进 `dynamic_args.ref_latents`。开着的话本扩展会在日志里警告，采样期间用自己的控制 latent 替换 Forge 的，用完再把原生的还回去。
- **Batch size > 1 可用。** Forge 会把 cond 和 uncond 拼成一个 batch，batch size 2 时模型看到的是 4 个样本，而 Forge 自带的参考代码只处理恰好 2 个的情况（会报 `torch.cat` 尺寸错误）。扩展会按 batch 补齐参考。
- **CFG > 1**：控制对 cond 和 uncond 都生效，和 Forge 自带代码一致。
- **显存**：Canny 下 token 序列长一倍。参考生效期间，扩展会调高 Forge 对该模型的显存估算，让 Forge 更不容易把 cond+uncond 拼进一次容易 OOM 的调用。
- **ADetailer**：这一批采样结束、进入后处理之前就会撤掉控制。ADetailer 如果沿用主提示词，会带上控制 LoRA 标签但没有控制（未测试），请给它单独写提示词。
- LoRA 调度语法（`<lora:名字:[…]>`）需要即时打 LoRA，而且只读主提示词，看不到本扩展追加的标签；想用的话，把 LoRA 下拉框设为 None，自己在提示词里写。

## patch 了什么

不改 `backend/` 下任何文件。所有运行时 patch 集中在一个文件：[`lib_anima_control/patches.py`](lib_anima_control/patches.py)：

1. `diffusion_model.forward`（实例属性，每个模型对象一次，未布防时不起作用）：调用期间设置 `dynamic_args.ref_latents`，`finally` 里还原；按 `x.shape[0]` 补齐参考；实现 Control End。（Canny）
2. `KModel.memory_required`（实例属性）：参考生效期间乘以倍数。（Canny）
3. Canny LoRA 的 `OnlineLoRAPatch.patch[0][0]`：Control End 之后置 0，这一批结束时还原（仅即时打 LoRA 时）。
4. 一个 `cfg_denoiser` 回调，记录 `CFGDenoiser.step / total_steps`。
5. `diffusion_model.x_embedder` 上的 forward hook：加上控制 token；未布防时不起作用。序列里有参考帧时，token 只加到目标帧。（OpenPose）

Canny 和 OpenPose 的状态是两个独立的对象（`patches.STATE`、`patches.EMBED`），所以能同时布防。

用到的钩子：`process`、`process_before_every_sampling`（在条件编码之后，这点很重要：Forge 的 Anima 引擎在 "[Anima] Enable Reference" 关闭时，每次编码正向提示词都会清空 `ref_latents`）、`postprocess_batch`、`postprocess`。

## 验证情况

测试用 Forge Neo 真实的 `backend/nn/anima.py`，缩小尺寸、随机权重，在 CPU 上跑；面板部分用 Forge 真实的 `modules`：

```
<Forge Neo>\venv\Scripts\python.exe tests\test_reference.py   # Canny 路径：74 项（设置 ANIMA_CANNY_FILE 时再加 6 项真实 LoRA 文件检查）
<Forge Neo>\venv\Scripts\python.exe tests\test_script.py      # 面板 + 钩子流程（txt2img、img2img、Hires. fix），两个控制单独用和一起用：145 项
set ANIMA_POSE_FILE=path\to\anima_pose_preview2.safetensors
<Forge Neo>\venv\Scripts\python.exe tests\test_pose.py        # OpenPose 路径 + 真实适配器文件：30 项
```

覆盖了：

- 输出形状等于不注入参考时的形状（参考部分被裁掉），batch 1 / 2 / 3 / 4、奇数 latent 尺寸；
- 布防后的包装器和 Forge 原生 `ref_latents` 路径逐位一致（batch 1 和 2），每次 forward 之后 `dynamic_args.ref_latents` 都是空的，包括异常之后；
- 参考和目标共用同一个 timestep embedding；
- Control End 之后输出和完全不带参考的运行完全一致；之后 LoRA 强度被还原；
- 原生路径在 batch 4 时确实会失败（所以需要按 batch 补齐）；
- **两个控制同时用**：输出等于 Forge 原生参考路径加上"只加在目标帧上的控制 token"，和任何一个单独用都不同；batch 2 / 3 / 4 成立；Hires. fix 的每一种开关组合；Control End 只撤掉 Canny、OpenPose 继续；LoRA 淡出只动 Canny 的 LoRA；失败是原子的（提示词里不会留下标签）；异常之后两个都被清理；
- Control Type 是可多选的 CheckboxGroup；共用的 Control Image / 各类型自己的图 / img2img 回退按这个顺序生效；面板打开但什么都没勾是报错；
- 面板里没有两个控件共用同一个标签（Forge 按标签给 `ui-config.json` 建键，重复的话第二个控件会继承第一个的默认值）；
- Hires. fix 开 / 关、按尺寸缓存参考、img2img 输入图回退、上面提到的各种失败路径；
- `encode_ref_latent` 用真实的 Anima Wan VAE 权重在 CPU 上跑过：形状 `(1, 16, 1, h, w)`，和 Forge 自己的 `encode_first_stage` 结果完全相同；
- 真实的 `anima-preview-canny-v0.2.safetensors`（SHA256 与 CivArchive 页面一致）：是不带 control embedder 的 rank-32 纯 LoRA，覆盖 28 个 block 的 448 个目标在 Forge 的 Anima 里全部存在；
- OpenPose：embedder 的排布和 Forge 的 `PatchEmbed` 一致、strength 0 时精确还原、奇数尺寸、真实适配器文件（embedder 是 `Linear(64 → 2048)`，448 个 DiT 和 60 个 `llm_adapter` 的 LoRA 目标在 Forge 的 Anima 里都存在）。

**没覆盖到的：** 带真实权重的 GPU 运行、真实浏览器里的面板、真实的 DWPose 检测器（ONNX 文件没下载）、真实即时 LoRA 模型上的 LoRA 淡出、显存估算在真实显存占用上的效果，以及**两个适配器一起用时的出图质量**。

适配版本：Forge Neo `neo` 分支，提交 `534e6ecce1039dbe5647e2c391b84a75aaa61d35`（"meta device"，2026-09-13）。没有针对更新的上游提交检查过。

## 未确认事项

- **Canny LoRA 是怎么训练的**：Canny 阈值、边缘图分辨率、线条极性（白线黑底还是反过来）、参考图是否和目标同分辨率，在它的 CivitAI / CivArchive 页面和 ComfyUI PR #13392 的正文里都没写，文件自己的 `__metadata__` 里也只有 `format: pt`。PR 里说是训练代码的 levzzz5154/diffusion-pipe 只有 `main`、`feature/anima-distillation`、`feature/lycoris-loha-lokr` 三个分支，在里面搜索 "canny" 没有结果（搜索索引可能不全；没有找到也没有读到数据集处理代码）。PR 附带的示例工作流 [`Anima-RefLatent.json`](https://github.com/user-attachments/files/26686944/Anima-RefLatent.json) 大概率能看到 Canny 节点、缩放和采样器的设置，但这边取不到：**请用文本编辑器打开对照**。这里的默认值（`cv2.Canny(100, 200)`、白线黑底、512 px 检测、参考图缩放到出图尺寸）都是猜的。阈值和分辨率可调，*Invert Canny Map* 可以试相反极性。另外 ComfyUI 内置的 Canny 节点（kornia，默认 0.4 / 0.8，0–1 尺度）和 `cv2.Canny` 数值上不是一回事，如果工作流用的是它，就需要调阈值，直到 Preview 看起来和 CivitAI 上的示例边缘图差不多。
- LoRA 是针对 Anima **preview** 版本训练的，在 Anima-Base v1.0 上效果可能打折。
- 它的**许可证**页面上没写。
- CivArchive 页面上有人分享了 Forge Neo 的补丁，并提到要选"正确的 control method"。没看到补丁内容，所以不知道这个 LoRA 是否需要"沿 T 维拼接、共用 timestep"之外的参考方式。
- Control End 之后关掉 LoRA，在真实模型上是否会留下残余影响（强度的还原在替身对象上测过，没在真实权重上测过）。
- Canny 和 OpenPose **一起用**能不能出好图。代码路径已验证等于"两个控制各自的机制并排生效"，但两个 LoRA 之间会怎么相互影响，目前一无所知。

## 已知问题

- 控制 embedder / 参考 latent 严格按出图尺寸（`latent × 8`）生成；启用时，高清修复阶段会按放大后的尺寸重新编码。
- 边缘图是二值的 1 像素细线，缩放到出图尺寸（和预处理分辨率不一致时）可能会变糊或变细。控制偏弱的话，试试让 Preprocessor Resolution 更接近出图尺寸。
- 本扩展早期本地版本（两个分开的面板）保存的界面默认值不会沿用：面板和标签都变了。

## 致谢

- Canny 适配器：levzzz（[CivitAI](https://civitai.com/models/2443202)）；训练代码：[levzzz5154/diffusion-pipe](https://github.com/levzzz5154/diffusion-pipe)。参考 latent 的行为同 ComfyUI PR [#13392](https://github.com/Comfy-Org/ComfyUI/pull/13392)。
- OpenPose 适配器和 ComfyUI 参考实现：[Claquasse/Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose)。`lib_anima_control/embedder.py`、`pose.py` 和 OpenPose 测试改自 `sd-forge-anima-pose` 扩展。
- 面板结构沿用 [sd-forge-krea2-control](https://github.com/matkaparka/sd-forge-krea2-control)。
- Anima 由 CircleStone Labs 发布（非商用许可；Pose 适配器的模型卡还引用了 NVIDIA Open Model License），基于 Cosmos-Predict2。

## 许可

AGPL-3.0，和 Forge Neo、sd-forge-krea2-control 一致（见 [LICENSE](LICENSE)）。模型权重遵循各自的许可：Anima 是 CircleStone Labs 的非商用许可；Pose 适配器权重为非商用；Canny LoRA 没有标明许可。你生成的图片不受本扩展许可限制。
