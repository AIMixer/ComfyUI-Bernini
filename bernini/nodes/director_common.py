"""Shared helpers for Bernini Director timeline nodes (KJ + Official)."""

from __future__ import annotations

import json
import logging

import torch

from ..director.audio_export import build_director_audio_outputs, task_passes_source_audio
from ..director.gen_timeline import is_prompt_batch_timeline, is_video_batch_task_key
from ..director.plan import build_director_plan, count_all_timeline_segments, count_timeline_segments, plan_summary
from ..director.progress import report_director_planning
from ..task_prompts import task_type_combo_options

log = logging.getLogger("ComfyUI-Bernini")


def timeline_required_inputs() -> dict:
    """Timeline + prompt widgets shared by KJ and Official director nodes."""
    combo_options, combo_meta = task_type_combo_options()
    return {
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
    }


def director_perf_inputs() -> dict:
    """Performance widgets shared by Bernini Director nodes."""
    return {
        "bd_grp_perf": ("BDGROUP", {"default": "性能 Performance"}),
        "clear_vram_between_segments": (
            "BOOLEAN",
            {
                "default": True,
                "tooltip": (
                    "段间清理显存：每段结束后卸载已加载模型并清空 CUDA 缓存，"
                    "降低多段峰值显存（段间略慢），从而降低爆显存风险"
                ),
            },
        ),
    }


def validate_decode_tiles(tile_x, tile_y, tile_stride_x, tile_stride_y, **_kwargs):
    if tile_x <= tile_stride_x:
        return "Decode tile_x must be larger than tile_stride_x."
    if tile_y <= tile_stride_y:
        return "Decode tile_y must be larger than tile_stride_y."
    return True


def default_timeline_json(
    *,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> str:
    return json.dumps(
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


def prepare_director_plan(
    *,
    timeline_data: str,
    task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
    unique_id: str | None,
):
    if not timeline_data or not timeline_data.strip():
        timeline_data = default_timeline_json(
            task_type=task_type,
            global_prompt=global_prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
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
    return plan


def finalize_director_outputs(plan, combined, segment_outputs, report):
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
        has_audio = any(
            isinstance(a, dict)
            and isinstance(a.get("waveform"), torch.Tensor)
            and int(a["waveform"].numel()) > 0
            for a in audio_out
        )
        if has_audio:
            report = report + "\n\nSource audio: extracted from input video (connect audio → VHS Video Combine)."
        else:
            report = report + "\n\nSource audio: none (input video has no audio track or ffmpeg unavailable)."
    return images_out, audio_out, frame_count, report
