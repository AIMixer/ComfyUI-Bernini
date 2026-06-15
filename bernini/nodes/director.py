"""Bernini Director — in-node timeline editor + batch Bernini execution."""

from __future__ import annotations

from ..director.executor import execute_director_plan
from .director_common import (
    director_perf_inputs,
    finalize_director_outputs,
    prepare_director_plan,
    validate_decode_tiles,
)
from .t5_config import resolve_t5_config
from ..task_prompts import task_type_combo_options

_CATEGORY = "Bernini"


class BerniniDirector:
    """Upload video + refs in-node; connect VAE / models / schedulers / T5 config from outside."""

    @classmethod
    def INPUT_TYPES(cls):
        combo_options, combo_meta = task_type_combo_options()
        return {
            "required": {
                "vae": ("WANVAE", {"tooltip": "Bernini VAE — connect from BerniniVAELoader."}),
                "model_high": ("WANVIDEOMODEL", {"tooltip": "High-noise Bernini / Wan video model."}),
                "model_low": ("WANVIDEOMODEL", {"tooltip": "Low-noise Bernini / Wan video model."}),
                "scheduler_high": ("WANVIDEOSCHEDULER",),
                "scheduler_low": ("WANVIDEOSCHEDULER",),
                "t5_config": (
                    "BERNINIT5CONFIG",
                    {"tooltip": "Connect Bernini T5 Config (model / precision / cache / device)."},
                ),
                "task_type": (combo_options, combo_meta),
                "global_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Synced from in-node UI (global mode).",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "bad video",
                        "multiline": True,
                        "tooltip": "Synced from in-node UI — shared negative prompt for all segments.",
                    },
                ),
                "bd_grp_high": ("BDGROUP", {"default": "高噪采样设置"}),
                "high_noise_cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for high-noise sampler pass."},
                ),
                "high_noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed for high-noise sampler pass.",
                    },
                ),
                "high_noise_force_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Offload model after high-noise sampling."},
                ),
                "high_noise_add_noise_to_samples": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Add noise before high-noise pass (v2v / rv2v)."},
                ),
                "bd_grp_low": ("BDGROUP", {"default": "低噪采样设置"}),
                "low_noise_cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01, "tooltip": "CFG for low-noise sampler pass."},
                ),
                "low_noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Seed for low-noise sampler pass.",
                    },
                ),
                "low_noise_force_offload": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Offload model after low-noise sampling."},
                ),
                "low_noise_add_noise_to_samples": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Add noise before low-noise pass."},
                ),
                "bd_grp_decode": ("BDGROUP", {"default": "Decode 解码设置"}),
                "enable_vae_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Bernini Decode: tiled VAE decode (reduces VRAM, may show seams).",
                    },
                ),
                "tile_x": (
                    "INT",
                    {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Decode tile width (px)."},
                ),
                "tile_y": (
                    "INT",
                    {"default": 272, "min": 40, "max": 2048, "step": 8, "tooltip": "Decode tile height (px)."},
                ),
                "tile_stride_x": (
                    "INT",
                    {"default": 144, "min": 32, "max": 2040, "step": 8, "tooltip": "Decode tile stride X (px)."},
                ),
                "tile_stride_y": (
                    "INT",
                    {"default": 128, "min": 32, "max": 2040, "step": 8, "tooltip": "Decode tile stride Y (px)."},
                ),
                "normalization": (
                    ["default", "minmax", "none"],
                    {"default": "default", "tooltip": "Bernini Decode output normalization."},
                ),
                "frame_rate": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Timeline / output FPS."},
                ),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "ref_max_size": ("INT", {"default": 848, "min": 16, "max": 8192, "step": 16}),
                "total_frames": (
                    "INT",
                    {"default": 81, "min": 1, "max": 8192, "tooltip": "Synced from uploaded video / timeline UI."},
                ),
                "timeline_data": (
                    "STRING",
                    {"default": "", "multiline": True, "tooltip": "Internal — video, segments, refs (populated by UI)."},
                ),
            },
            "optional": {
                "bd_grp_context": ("BDGROUP", {"default": "Context 编码设置"}),
                "high_noise_extra_args": ("WANVIDSAMPLEREXTRAARGS",),
                "low_noise_extra_args": ("WANVIDSAMPLEREXTRAARGS",),
                "tiled_vae": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Tiled VAE during context encode (not decode)."},
                ),
                "vae_force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offload VAE after context encode."}),
                **director_perf_inputs(),
                "enable_teacache": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "TeaCache 加速双阶段采样（开 = 更快，画质可能略降）。默认关 = 全精度。若已连接 Extra Args 的 Cache 节点，以 Cache 为准。",
                    },
                ),
                "high_noise_only": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Only run the high-noise sampler; skip low-noise sampling and decode high-noise samples as final images.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    VALIDATE_INPUTS = validate_decode_tiles

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING", "IMAGE", "FLOAT", "IMAGE")
    RETURN_NAMES = ("images", "audio", "frame_count", "report", "source_images", "fps", "images_high_noise")
    OUTPUT_IS_LIST = (True, True, False, False, True, False, True)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Bernini video director: upload video/refs in-node, split timeline, global or per-segment prompts. "
        "images output (list): one merged clip when export=all; one clip per segment when export=segments "
        "or prompt batch — connect to Video Combine and PreviewImage. "
        "source_images: timeline source frames aligned with images (enable export in Performance). "
        "fps: timeline frame rate for Video Combine / CreateVideo. "
        "audio output (v2v / rv2v): source video audio aligned to the export timeline when available. "
        "Separate high-noise / low-noise sampler settings (cfg, seed, force_offload, add_noise, extra_args)."
    )

    def execute(
        self,
        vae,
        model_high,
        model_low,
        scheduler_high,
        scheduler_low,
        t5_config,
        task_type,
        global_prompt,
        negative_prompt,
        high_noise_cfg,
        high_noise_seed,
        high_noise_force_offload,
        high_noise_add_noise_to_samples,
        low_noise_cfg,
        low_noise_seed,
        low_noise_force_offload,
        low_noise_add_noise_to_samples,
        enable_vae_tiling,
        tile_x,
        tile_y,
        tile_stride_x,
        tile_stride_y,
        normalization,
        frame_rate,
        width,
        height,
        ref_max_size,
        total_frames,
        timeline_data,
        unique_id=None,
        high_noise_extra_args=None,
        low_noise_extra_args=None,
        tiled_vae=False,
        vae_force_offload=True,
        clear_vram_between_segments=True,
        export_source_images=False,
        enable_teacache=False,
        high_noise_only=False,
        **kwargs,
    ):
        del kwargs  # bd_grp_* section headers — UI only
        high_noise_only = high_noise_only is True or str(high_noise_only).lower() in ("1", "true", "yes", "on")
        t5 = resolve_t5_config(t5_config)

        plan = prepare_director_plan(
            timeline_data=timeline_data,
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            unique_id=unique_id,
        )

        combined, segment_outputs, high_noise_combined, high_noise_segment_outputs, report = execute_director_plan(
            plan,
            node_id=unique_id,
            vae=vae,
            model_high=model_high,
            model_low=model_low,
            scheduler_high=scheduler_high,
            scheduler_low=scheduler_low,
            t5_model_name=t5["model_name"],
            t5_precision=t5["precision"],
            negative_prompt=negative_prompt,
            t5_quantization=t5["quantization"],
            use_disk_cache=t5["use_disk_cache"],
            t5_device=t5["device"],
            high_noise_cfg=high_noise_cfg,
            high_noise_seed=high_noise_seed,
            high_noise_force_offload=high_noise_force_offload,
            high_noise_add_noise_to_samples=high_noise_add_noise_to_samples,
            high_noise_only=high_noise_only,
            low_noise_cfg=low_noise_cfg,
            low_noise_seed=low_noise_seed,
            low_noise_force_offload=low_noise_force_offload,
            low_noise_add_noise_to_samples=low_noise_add_noise_to_samples,
            enable_teacache=enable_teacache,
            high_noise_extra_args=high_noise_extra_args,
            low_noise_extra_args=low_noise_extra_args,
            enable_vae_tiling=enable_vae_tiling,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_stride_x=tile_stride_x,
            tile_stride_y=tile_stride_y,
            normalization=normalization,
            tiled_vae=tiled_vae,
            vae_force_offload=vae_force_offload,
            clear_vram_between_segments=clear_vram_between_segments,
        )

        return finalize_director_outputs(
            plan,
            combined,
            segment_outputs,
            high_noise_combined,
            high_noise_segment_outputs,
            report,
            export_source_images=export_source_images,
        )


BerniniDirectorExecute = BerniniDirector
