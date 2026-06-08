"""Bernini Director — in-node timeline editor + batch Bernini execution."""

from __future__ import annotations

import json
import logging

from ..director.audio_export import build_director_audio_outputs, task_passes_source_audio
from ..director.executor import execute_director_plan
from ..director.gen_timeline import is_prompt_batch_timeline, is_video_batch_task_key
from ..director.plan import build_director_plan, count_all_timeline_segments, count_timeline_segments, plan_summary
from ..director.progress import report_director_planning
from ..task_prompts import task_type_combo_options
from .t5_config import resolve_t5_config

log = logging.getLogger("ComfyUI-Bernini")

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
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def VALIDATE_INPUTS(cls, tile_x, tile_y, tile_stride_x, tile_stride_y, **_kwargs):
        if tile_x <= tile_stride_x:
            return "Decode tile_x must be larger than tile_stride_x."
        if tile_y <= tile_stride_y:
            return "Decode tile_y must be larger than tile_stride_y."
        return True

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("images", "audio", "frame_count", "report")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "execute"
    CATEGORY = _CATEGORY
    DESCRIPTION = (
        "Bernini video director: upload video/refs in-node, split timeline, global or per-segment prompts. "
        "images output (list): one merged clip when export=all; one clip per segment when export=segments "
        "or prompt batch — connect to Video Combine and PreviewImage. "
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
        **kwargs,
    ):
        del kwargs  # bd_grp_* section headers — UI only
        t5 = resolve_t5_config(t5_config)

        if not timeline_data or not timeline_data.strip():
            timeline_data = json.dumps(
                {
                    "version": 4,
                    "editMode": "global",
                    "totalFrames": total_frames,
                    "frameRate": frame_rate,
                    "width": width,
                    "height": height,
                    "refMaxSize": ref_max_size,
                    "output": {
                        "mode": "long_edge",
                        "longEdge": ref_max_size,
                        "width": width,
                        "height": height,
                        "maxExportFrames": 0,
                        "exportMode": "all",
                    },
                    "videoClips": [],
                    "video": {
                        "fileName": "",
                        "videoFile": "",
                        "subfolder": "",
                        "type": "input",
                        "frames": [],
                        "frameMap": [],
                    },
                    "global": {"taskType": task_type, "prompt": global_prompt, "refs": []},
                    "segments": [
                        {
                            "id": "s0",
                            "start": 0,
                            "length": total_frames,
                            "prompt": "",
                            "taskType": "",
                            "refs": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )

        report_director_planning(
            unique_id,
            count_timeline_segments(timeline_data),
            timeline_segment_total=count_all_timeline_segments(timeline_data),
        )

        plan = build_director_plan(
            timeline_data,
            global_task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

        log.info(plan_summary(plan).replace("\n", " | "))

        combined, segment_outputs, report = execute_director_plan(
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
            low_noise_cfg=low_noise_cfg,
            low_noise_seed=low_noise_seed,
            low_noise_force_offload=low_noise_force_offload,
            low_noise_add_noise_to_samples=low_noise_add_noise_to_samples,
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
        )

        is_batch = is_prompt_batch_timeline(plan.raw, plan.global_task_key)
        export_segments = plan.export_mode == "segments"
        video_batch = is_video_batch_task_key(plan.global_task_key)

        if export_segments or (is_batch and not video_batch):
            images_out = segment_outputs
            frame_count = sum(int(s.shape[0]) for s in segment_outputs)
            if export_segments and len(segment_outputs) > 1:
                report = (
                    report
                    + f"\n\nExport mode: segments — {len(segment_outputs)} clip(s) on images output "
                    "(one MP4 per segment when connected to Video Combine / PreviewImage)."
                )
            if plan.run_indices is not None:
                report = (
                    report
                    + f"\n\nPartial run: output contains {len(segment_outputs)} re-generated "
                    f"{'group(s)' if is_batch else 'segment clip(s)'} only."
                )
        else:
            images_out = [combined]
            frame_count = int(combined.shape[0])
            if video_batch and is_batch and len(segment_outputs) > 1:
                report = (
                    report
                    + f"\n\nExport mode: all — merged {frame_count} frame(s) on images output "
                    "(single clip when connected to Video Combine / PreviewImage)."
                )
            if plan.run_indices is not None and video_batch:
                report = (
                    report
                    + f"\n\nPartial run: re-generated {len(segment_outputs)} video group(s); "
                    "skipped groups merged from cache or source when available."
                )
        audio_out = build_director_audio_outputs(
            plan,
            images_out,
            export_segments=export_segments or (is_batch and not video_batch),
            output_frame_end=frame_count if not (export_segments or (is_batch and not video_batch)) else None,
        )
        if task_passes_source_audio(plan.global_task_key):
            has_audio = any(a is not None for a in audio_out)
            if has_audio:
                report = report + "\n\nSource audio: extracted from input video (connect audio → VHS Video Combine)."
            else:
                report = report + "\n\nSource audio: none (input video has no audio track or ffmpeg unavailable)."
        return (images_out, audio_out, frame_count, report)


BerniniDirectorExecute = BerniniDirector
