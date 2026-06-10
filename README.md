# ComfyUI-Bernini

Standalone ComfyUI plugin with a **complete Wan 2.2 Bernini pipeline** that does not depend on ComfyUI core Bernini nodes ([PR #14216](https://github.com/Comfy-Org/ComfyUI/pull/14216)).

**中文文档** → [README_ZH.md](README_ZH.md)

## Quick start

1. **Clone** into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/AIMixer/ComfyUI-Bernini.git
   ```

2. **Install Python dependencies** (use the same Python environment as your ComfyUI):
   ```bash
   cd ComfyUI-Bernini
   pip install -r requirements.txt
   ```
   On Windows portable builds, prefer ComfyUI's bundled Python, for example:
   ```bash
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```

3. **Restart ComfyUI**. Nodes appear under the **Bernini** category.

4. **Download models** — quantized GGUF / FP8 + workflows: **[comfyit.cn/article/489](https://comfyit.cn/article/489)**; original Kijai FP8 only: [HuggingFace Bernini](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/Bernini). Put weights under `ComfyUI/models/` (`diffusion_models/`, `vae/`, `text_encoders/`). See [Quantized models](#quantized-models).

5. **Run a workflow**: Model Loader → Context Embeds → Sampler → Decode. Example JSON workflows: [Comfyit article 489](https://comfyit.cn/article/489).

## Quantized models

**Download:** [Comfyit article 489 — Bernini models & workflows](https://comfyit.cn/article/489) (GGUF quantizations, scaled FP8, VAE, T5, and example JSON).

**Wan 2.2 Bernini** diffusion weights are available in **GGUF** and **FP8 safetensors** quantizations. Place files under `ComfyUI/models/diffusion_models/` and load both **HIGH** and **LOW** in **Bernini Model Loader** (GGUF is supported).

| Tier | GGUF (LOW / HIGH) | Min VRAM (Bernini Director) |
|------|-------------------|----------------------------|
| **Q4_K_M** (lowest) | `Wan22_Bernini_LOW-Q4_K_M.gguf` · `Wan22_Bernini_HIGH-Q4_K_M.gguf` | **8 GB** |
| Q5_K_M | `Wan22_Bernini_LOW-Q5_K_M.gguf` · `Wan22_Bernini_HIGH-Q5_K_M.gguf` | **10 GB** |
| Q6_K | `Wan22_Bernini_LOW-Q6_K.gguf` · `Wan22_Bernini_HIGH-Q6_K.gguf` | **12 GB** |
| Q8_0 | `Wan22_Bernini_LOW-Q8_0.gguf` · `Wan22_Bernini_HIGH-Q8_0.gguf` | **16 GB** |

**FP8 safetensors** (scaled, same naming as Kijai Bernini pack):

- `Wan22_Bernini_LOW_fp8_e4m3fn_scaled.safetensors`
- `Wan22_Bernini_HIGH_fp8_e4m3fn_scaled.safetensors`

**VRAM tips (Bernini Director, HIGH + LOW):** Q4 → 8 GB min; Q5 → 10 GB; Q6 → 12 GB; Q8 → 16 GB. Enable **Block Swap** on both loaders; use T5 disk cache or fp8 text encoder where possible.

> All quantized builds above are included in the resource pack at **[comfyit.cn/article/489](https://comfyit.cn/article/489)**. For the original non-GGUF FP8 pack from Kijai, see [HuggingFace](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/Bernini).

## Node chain

`BerniniModelLoader` · `BerniniVAELoader` · `BerniniTextEncodeCached` · `BerniniContextEmbeds` · `BerniniContextOptions` · `BerniniSamplerExtraArgs` · `BerniniScheduler` · `BerniniSampler` · `BerniniDecode` · **`BerniniDirector`** · **`BerniniDirectorOfficial`**

## Bernini Director (KJ backend)

All-in-one node with an embedded **timeline editor**: upload video and reference images inside the node, split segments, set per-segment prompts / `task_type`, then run the full Bernini HIGH/LOW pipeline in one queue. Uses **Bernini Model Loader** (GGUF + Block Swap) for lower VRAM.

![Bernini Director node UI](docs/assets/bernini_director_ui.png)

Example workflows: see [Example workflows](#example-workflows) below (all from [Comfyit article 489](https://comfyit.cn/article/489)).

## Bernini Director Official (ComfyUI core backend)

Same **timeline UI** as Bernini Director, but execution goes through ComfyUI native Bernini (`VAELoader` / `UNETLoader` ×2 / `CLIPLoader` → `BerniniConditioning` → dual-stage `KSamplerAdvanced`), aligned with [PR #14216](https://github.com/Comfy-Org/ComfyUI/pull/14216).

![Bernini Director Official node UI](docs/assets/bernini_director_official_ui.png)

| | KJ Director | Official Director |
|---|-------------|-------------------|
| Models | Bernini Model Loader (GGUF) | UNETLoader fp8_scaled |
| Text | Bernini Text Encode Cached | CLIPLoader (wan) |
| VRAM | ~8 GB with Q4 GGUF | Higher; use `--lowvram`, CLIP on CPU |

**Typical v2v workflow:** load `bernini_director_official_core_v2v_2.json` → connect VAE / UNET×2 / CLIP → upload source video → **Equal Split** into segments → global prompt → queue once, export all segments.

## Example workflows

Download **Bernini model weights + example JSON workflows** from [Comfyit: Bernini models & workflows (article 489)](https://comfyit.cn/article/489):

| Workflow | `task_type` | Download |
|----------|-------------|----------|
| `bernini_director_minimal_test (r2v) .json` | `r2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (t2i) .json` | `t2i` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (t2v) .json` | `t2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (r2i) .json` | `r2i` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (v2v).json` | `v2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (i2v) .json` | `i2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (i2i).json` | `i2i` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (rv2v).json` | `rv2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_official_core_v2v_1.json` | `v2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_official_core_v2v_2.json` | `v2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_official_core_rv2v.json` | `rv2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(r2v) .json` | `r2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(v2v).json` | `v2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(vi2v) .json` | `vi2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(rv2v) .json` | `rv2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |

After download: merge `models/` into `ComfyUI/models`, install plugins/deps, drag the JSON into ComfyUI. Details on the article page.

## Acknowledgements

The `engine/` layer is **adapted from** [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache-2.0). Deep respect and gratitude to kijai and all contributors to the WanVideo ecosystem.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

The `engine/` layer is **adapted from** [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (also Apache-2.0). See [Acknowledgements](#acknowledgements).

---

## Ecosystem · [Comfyit 搅拌站](https://comfyit.cn/)

[Comfyit](https://comfyit.cn/) is a one-stop ComfyUI tools & learning platform. For environment setup, models, workflows, and tutorials that complement this plugin, see the [**Product Center**](https://comfyit.cn/products) (ComfyUI Manager, LoRA Trainer, Prompt Master) and free resources: [packages](https://comfyit.cn/resources/packages) · [models](https://comfyit.cn/resources/models) · [workflows](https://comfyit.cn/workflows) · [learning center](https://comfyit.cn/lc/beginner).

Full details in [README_ZH.md](README_ZH.md#配套生态--comfyit-搅拌站).

## Contact

| | |
|---|---|
| **Maintainer** | [AIMixer](https://github.com/AIMixer) |
| **Author QQ** | **3697688140** |
| **Bilibili** | [space.bilibili.com/1997403556](https://space.bilibili.com/1997403556) |
| **QQ groups** | **551482703** · **425064221** · **559826331** |
| **Comfyit** | [comfyit.cn](https://comfyit.cn/) |
