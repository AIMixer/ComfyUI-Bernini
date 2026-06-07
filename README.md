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

4. **Download Bernini weights** from [Kijai/HuggingFace](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/Bernini) into `ComfyUI/models/` (`diffusion_models/`, `vae/`, `text_encoders/`).

5. **Run a workflow**: Model Loader → Context Embeds → Sampler → Decode. Example JSON workflows: [Comfyit article 489](https://comfyit.cn/article/489).

## Node chain

`BerniniModelLoader` · `BerniniVAELoader` · `BerniniTextEncodeCached` · `BerniniContextEmbeds` · `BerniniContextOptions` · `BerniniSamplerExtraArgs` · `BerniniScheduler` · `BerniniSampler` · `BerniniDecode` · **`BerniniDirector`**

## Bernini Director

All-in-one node with an embedded **timeline editor**: upload video and reference images inside the node, split segments, set per-segment prompts / `task_type`, then run the full Bernini HIGH/LOW pipeline in one queue.

![Bernini Director node UI](docs/assets/bernini_director_ui.png)

Example workflows: see [Example workflows](#example-workflows) below (all from [Comfyit article 489](https://comfyit.cn/article/489)).

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
| `bernini_video_edit(r2v) .json` | `r2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(v2v).json` | `v2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(vi2v) .json` | `vi2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(rv2v) .json` | `rv2v` | [comfyit.cn/article/489](https://comfyit.cn/article/489) |

After download: merge `models/` into `ComfyUI/models`, install plugins/deps, drag the JSON into ComfyUI. Details on the article page.

## Acknowledgements

The `engine/` layer is **adapted from** [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (Apache-2.0). Deep respect and gratitude to kijai and all contributors to the WanVideo ecosystem.

## License

- `bernini/` — MIT
- `engine/` — Apache-2.0

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
