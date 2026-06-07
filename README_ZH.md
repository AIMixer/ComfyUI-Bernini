# ComfyUI-Bernini

独立的 ComfyUI 插件，提供 **完整的 Wan 2.2 Bernini 视频生成/编辑链路**，不依赖 ComfyUI 核心 Bernini 节点（[PR #14216](https://github.com/Comfy-Org/ComfyUI/pull/14216)）。

基于 [Bernini](https://bernini-ai.github.io/) 论文框架：MLLM 语义规划 + DiT 渲染，通过 in-context `context_latents` 实现 v2v / rv2v / r2v / t2v。

**English** → [README.md](README.md)

## 特性

- **Bernini 专用推理引擎**（`engine/`）：仅保留 Bernini 视频编辑所需模块（已移除 MultiTalk / HuMo / WanMove 等）
- **Bernini 语义上下文节点**：VAE 编码源视频/参考图，写入 `context_latents`
- **完整 Bernini 节点链**：ModelLoader → TextEncode → Context Embeds → Sampler → Decode
- **可选原生 Conditioning 节点**：`BerniniConditioning`（需 ComfyUI 核心支持时才生效）

## 节点一览

| 节点 | 作用 |
|------|------|
| **Bernini Model Loader** | 加载 Bernini HIGH/LOW DiT 权重 |
| **Bernini VAE Loader** | 加载 Wan 2.1 VAE |
| **Bernini Text Encode Cached** | T5 文本编码（带磁盘缓存）；`task_type` 自动拼接 Bernini 系统提示词 |
| **Bernini Context Embeds** | 构建 Bernini in-context 条件（R2V 最多 5 张参考图，对应 prompt 中 image0–image4） |
| **Bernini Context Options** | 长视频 context window |
| **Bernini Sampler Extra Args** | 采样附加参数 |
| **Bernini Scheduler** | Flow-match 调度器（双阶段 HIGH/LOW） |
| **Bernini Sampler** | DiT 去噪采样 |
| **Bernini Decode** | VAE 解码输出帧 |
| **Bernini Block Swap** / **Set Block Swap** | 显存优化 |
| **Bernini LoRA Select Multi** / **Set LoRAs** | LoRA 加载 |
| **Bernini Director** | 节点内时间轴导演台：上传视频/参考图、分割片段、分段提示词，一键批量跑 Bernini 双阶段推理 |

## Bernini Director（导演台）

单个节点集成 **时间轴编辑 + 批量推理**：在节点内上传视频与参考图，分割/均分片段，全局或分段编辑提示词与 `task_type`，Queue 一次即可逐段执行 Bernini HIGH/LOW 并拼接输出。

![Bernini Director 节点界面](docs/assets/bernini_director_ui.png)

| 区域 | 功能 |
|------|------|
| 工具栏 | 上传/追加视频、分割、均分、删除片段、全局/分段模式 |
| 时间轴 | 帧缩略图、片段边界、播放预览、缩放 |
| 输出设置 | 最长边/固定分辨率、全部导出/分段导出、最大帧数 |
| 片段编辑 | 每段正向/反向提示词、参考图 img0–4 |
| 运行状态 | Queue 执行时显示片段与阶段进度 |

示例工作流见下方 [示例工作流下载](#示例工作流下载)（均从 [Comfyit 文章 489](https://comfyit.cn/article/489) 获取）。

## 快速开始

1. **克隆**到 `ComfyUI/custom_nodes/`：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/AIMixer/ComfyUI-Bernini.git
   ```

2. **安装 Python 依赖**（务必使用 ComfyUI 同一 Python 环境）：
   ```bash
   cd ComfyUI-Bernini
   pip install -r requirements.txt
   ```
   Windows 便携版示例：
   ```bash
   ..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```

3. **重启 ComfyUI**，在节点菜单 **Bernini** 分类下使用。

> **范围说明**：本插件专注 Bernini 论文的 in-context 视频编辑（rv2v / v2v / r2v 等）。

## 模型

| 文件 | 目录 |
|------|------|
| `Bernini_HIGH_fp8_*.safetensors` | `models/diffusion_models/` |
| `Bernini_LOW_fp8_*.safetensors` | `models/diffusion_models/` |
| `Wan2_1_VAE_bf16.safetensors` | `models/vae/` |
| `umt5-xxl-enc-bf16.safetensors` | `models/text_encoders/`（推荐，约 11 GB） |
| `umt5-xxl-enc-fp8_e4m3fn.safetensors` | `models/text_encoders/`（非 scaled fp8，约 6.7 GB） |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders/`（ComfyUI scaled fp8，加载时会反量化到 bf16） |

**Bernini Text Encode Cached** 推荐使用 `umt5-xxl-enc-bf16.safetensors`。若使用 scaled fp8 版本，节点会在加载时自动反量化（显存占用与 bf16 相近，不节省 VRAM）。

推荐：[Kijai/WanVideo_comfy_fp8_scaled/Bernini](https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/tree/main/Bernini)

## 示例工作流下载

完整资源包（**Bernini 模型权重** + **示例 JSON 工作流**）见 [Comfyit 搅拌站文章：视频编辑 Bernini 模型和工作流](https://comfyit.cn/article/489)。

| 工作流文件 | `task_type` | 说明 | 下载 |
|------------|-------------|------|------|
| `bernini_director_minimal_test (r2v) .json` | `r2v` | 导演台 · 参考图生视频 · 多组提示词 **更新** | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (t2i) .json` | `t2i` | 导演台 · 文生图 · 多组提示词 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (t2v) .json` | `t2v` | 导演台 · 文生视频 · 多组提示词 **更新** | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (r2i) .json` | `r2i` | 导演台 · 参考图生图 · 多组提示词 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (v2v).json` | `v2v` | 导演台 · 源视频 prompt 编辑 **更新** | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (i2v) .json` | `i2v` | 导演台 · 图生视频 · 实验性 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (i2i).json` | `i2i` | 导演台 · 图生图 · 多组提示词 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (rv2v).json` | `rv2v` | 导演台 · 参考图 + 源视频编辑 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_director_minimal_test (rv2v)).json` | `rv2v` | 导演台 · 参考图 + 源视频编辑（备用文件名） | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(r2v) .json` | `r2v` | 纯参考图生视频（`reference_image_0`–`4` 对应 prompt 中 `image0`–`image4`） | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(v2v).json` | `v2v` | 源视频 prompt 驱动编辑 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(vi2v) .json` | `vi2v` | 内容延展改视频 | [comfyit.cn/article/489](https://comfyit.cn/article/489) |
| `bernini_video_edit(rv2v) .json` | `rv2v` | 参考图 + 源视频编辑（双阶段 HIGH/LOW MoE） | [comfyit.cn/article/489](https://comfyit.cn/article/489) |

**使用步骤**（摘自 [文章说明](https://comfyit.cn/article/489)）：

1. 下载后，用包内 `models` 覆盖（或合并到）你的 `ComfyUI/models`
2. 安装工作流所需插件与 Python 依赖（含本仓库 `ComfyUI-Bernini`）
3. 启动 ComfyUI，将对应 JSON 工作流拖入画布即可运行

> 文章页需登录搅拌站账号；部分资源为积分下载。配套环境也可使用 [ComfyUI 管理大师](https://comfyit.cn/products#comfyui-master) 一键分析依赖。

## 工作流结构（rv2v 双阶段 MoE 示例）

```
VHS_LoadVideo → ImageResize → Bernini Context Embeds ← LoadImage (refs)
                                    ↓
Bernini ModelLoader (HIGH) → SetBlockSwap → SetLoRAs → Bernini Sampler (HIGH, step 0–10)
Bernini ModelLoader (LOW)  → SetBlockSwap → SetLoRAs → Bernini Sampler (LOW,  step 10–end)
Bernini TextEncodeCached ─────────────────────────────→ 两个 Sampler
Bernini Scheduler ×2 ────────────────────────────────→ 两个 Sampler
Bernini Context Options → Sampler Extra Args ────────→ 两个 Sampler
Bernini Sampler (LOW) → Bernini Decode → VHS_VideoCombine
```

工作流使用 **Bernini \*** 节点链；socket 类型包括 `WANVIDEOMODEL`、`WANVIDIMAGE_EMBEDS` 等。

## 外部辅助节点（可选）

视频 I/O 与 resize 仍可使用社区节点：

- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) — 加载/合成视频
- [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) — INTConstant、ImageResizeKJv2

## 目录结构

```
ComfyUI-Bernini/
├── __init__.py              # 节点注册
├── bernini/                 # 原创：语义上下文 + 节点封装
│   ├── context_pipeline.py
│   ├── encoders.py
│   └── nodes/
├── engine/                  # Bernini 专用推理引擎（Apache-2.0，改编自 WanVideoWrapper）
│   ├── bernini_core_nodes.py
│   ├── wanvideo/
│   ├── nodes_sampler.py
│   └── nodes_model_loading.py
└── requirements.txt
```

## 来源说明

本项目的 `engine/` 推理引擎**改编自** [kijai/ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)（Apache-2.0）。向 kijai 及 WanVideoWrapper / WanVideo 生态的所有贡献者**致以崇高的敬意**，感谢他们为 ComfyUI WanVideo 社区做出的巨大贡献。

`engine/` 中的 LoRA 线性算子与 SageAttention 自定义算子**参考自** [wuwukaka/ComfyUI-WanAnimatePlus](https://github.com/wuwukaka/ComfyUI-WanAnimatePlus)（Apache-2.0）。感谢 wuwukaka 的 Wan Animate Plus 优化实现。

## 致谢

- [Bernini](https://bernini-ai.github.io/) — ByteDance 论文与模型框架

## 许可证

- `bernini/` — MIT
- `engine/` — Apache-2.0

---

## 配套生态 · [Comfyit 搅拌站](https://comfyit.cn/)

[Comfyit 搅拌站](https://comfyit.cn/) 是一站式 **ComfyUI 工具与学习平台**，助力本地工作流高效运行。跑通本插件所需的 ComfyUI 环境、模型下载、工作流与教程，都可以在搅拌站找到配套资源。

> 产品详情与购买：[产品中心](https://comfyit.cn/products)

### 三款桌面工具（产品中心主推）

| 产品 | 一句话 | 与 Bernini 的关系 |
|------|--------|-------------------|
| [**ComfyUI 管理大师**](https://comfyit.cn/products#comfyui-master) | 本地 ComfyUI 一站式管家：启动 · 插件 · 依赖 · 工作流 · 资源下载 | 一键启动整合包、自动补齐缺失插件/模型，适合部署 **ComfyUI-Bernini** 双阶段 rv2v 工作流 |
| [**LoRA 训练大师**](https://comfyit.cn/products#lora-master) | 零环境配置 · 图形化训练 · 自动打标 · 实时监控 | 训练角色/风格 LoRA，配合 Bernini **rv2v / r2v** 参考图与提示词做个性化视频 |
| [**提示词大师**](https://comfyit.cn/products#prompt-master) | 反推 · 扩写 · 词库 · 工程模板，本地提示词工作台 | 辅助撰写含 `image0`–`image4` 的多参考图文案，以及 Bernini `task_type` 正向提示词 |

**ComfyUI 管理大师** 核心能力：一键启动 ComfyUI、整合包多开管理、插件安装与升级（国内镜像加速）、依赖与环境检测、工作流智能分析（自动补齐缺失节点/模型）、资源下载中心、日志与进度监控、环境快照与回滚。

**LoRA 训练大师** 核心能力：开箱即训、智能参数分配、显存友好优化、数据集与自动打标（Qwen / JoyCaption / ACE-Step 等）、实时采样与 Loss 曲线、素材预处理工具集、LoRA 格式转换、训练任务状态清晰展示。

**提示词大师** 核心能力：图片/视频反推（智谱 API、Ollama、本地 Qwen3-VL 等）、短句智能扩写、我的提示词词库（可拖拽到 ComfyUI）、提示词工程模板、中英互译、数据本地保存、多后端自由切换。

三款工具均支持 [**微信扫码购买**](https://comfyit.cn/products) 与搅拌站同账号登录；未充电也可体验部分核心功能。

### 免费在线资源

| 栏目 | 链接 | 说明 |
|------|------|------|
| 整合包 | [comfyit.cn/resources/packages](https://comfyit.cn/resources/packages) | 预配置 ComfyUI 便携环境，减少手工装依赖 |
| 模型广场 | [comfyit.cn/resources/models](https://comfyit.cn/resources/models) | Bernini / Wan 等模型与工作流相关权重 |
| 工作流广场 | [comfyit.cn/workflows](https://comfyit.cn/workflows) | 社区工作流下载，可一键分析依赖 |
| 资源下载 | [comfyit.cn/resources](https://comfyit.cn/resources) | 整合包、模型、工作流、软件工具一站获取 |
| 学习中心 | [comfyit.cn/lc/beginner](https://comfyit.cn/lc/beginner) | 从小白入门到插件进阶，教程与工具衔接 |
| 插件学习 | [comfyit.cn/lc/customnodes](https://comfyit.cn/lc/customnodes) | 自定义节点安装与使用说明 |
| 问答社区 | [comfyit.cn](https://comfyit.cn/) | 技术交流与 Bernini 使用问题讨论 |

### 推荐路径（Bernini 用户）

1. 在 [整合包](https://comfyit.cn/resources/packages) 或 **ComfyUI 管理大师** 中准备本地 ComfyUI 环境  
2. 安装本插件，从 [模型广场](https://comfyit.cn/resources/models) 下载 Bernini HIGH/LOW、VAE、T5 权重  
3. 从 [示例工作流下载](https://comfyit.cn/article/489) 获取 JSON，用 **Bernini Context Embeds** 连接 `reference_image_0`–`4`  
4. 用 **提示词大师** 打磨 R2V / RV2V 提示词，需要 LoRA 时用 **LoRA 训练大师** 训练素材  

访问 [**Comfyit 搅拌站 → 产品中心**](https://comfyit.cn/products) 了解价格方案与下载方式。

## 作者与交流

| | |
|---|---|
| **维护者** | [AI搅拌手 / AIMixer](https://github.com/AIMixer) |
| **作者 QQ** | **3697688140** |
| **B 站** | [space.bilibili.com/1997403556](https://space.bilibili.com/1997403556) |
| **QQ 交流群** | **551482703** · **425064221** · **559826331**（ComfyUI / Bernini 使用交流） |
| **Comfyit 搅拌站** | [comfyit.cn](https://comfyit.cn/) — 教程、工作流与工具下载 |
