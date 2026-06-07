"""Run Bernini pipeline for each Director segment and concatenate output."""

from __future__ import annotations

import logging

import torch

from ..image_prep import fit_canvas, fit_video_long_edge, cat_frames_variable_size
from ..nodes.text_encode import BerniniTextEncodeCached
from ..nodes.wan import BerniniWanContextEmbeds
from ...engine.bernini_core_nodes import WanVideoDecode
from ...engine.nodes_sampler import WanVideoSamplerv2

from .plan import (
    DirectorPlan,
    plan_summary,
    prepare_segment_clip,
    refs_to_kwargs_for_context,
    slice_video_frames,
)
from .progress import report_director_finish, report_director_progress, report_director_segment_preview

log = logging.getLogger("ComfyUI-Bernini.director")


def _needs_source_video(task_key: str) -> bool:
    return task_key in {"v2v", "rv2v", "vi2v", "vrc2v", "mv2v", "ads2v", "i2v", "i2i"}


def _tensor_frame_to_jpeg_b64(frame: torch.Tensor) -> str:
    import base64
    import io

    from PIL import Image

    arr = (frame.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frames_label(seg) -> str:
    return f"帧 {seg.start_frame}–{seg.end_frame} ({seg.frame_count}f)"


def execute_director_plan(
    plan: DirectorPlan,
    *,
    node_id: str | None = None,
    vae,
    model_high,
    model_low,
    scheduler_high,
    scheduler_low,
    t5_model_name: str,
    t5_precision: str,
    negative_prompt: str,
    t5_quantization: str = "disabled",
    use_disk_cache: bool = True,
    t5_device: str = "gpu",
    high_noise_cfg: float = 1.0,
    high_noise_seed: int = 0,
    high_noise_force_offload: bool = True,
    high_noise_add_noise_to_samples: bool = True,
    low_noise_cfg: float = 1.0,
    low_noise_seed: int = 0,
    low_noise_force_offload: bool = True,
    low_noise_add_noise_to_samples: bool = False,
    high_noise_extra_args=None,
    low_noise_extra_args=None,
    enable_vae_tiling: bool = False,
    tile_x: int = 272,
    tile_y: int = 272,
    tile_stride_x: int = 144,
    tile_stride_y: int = 128,
    normalization: str = "default",
    tiled_vae: bool = False,
    vae_force_offload: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], str]:
    """Process every segment; return combined frames, per-segment frames, and report."""
    source_video = plan.source_video
    if source_video.ndim != 4:
        raise ValueError("source_video must be [F, H, W, C]")

    text_encoder = BerniniTextEncodeCached()
    context_node = BerniniWanContextEmbeds()
    sampler = WanVideoSamplerv2()
    decoder = WanVideoDecode()

    output_chunks: list[torch.Tensor] = []
    segment_outputs: list[torch.Tensor] = []
    reports: list[str] = [plan_summary(plan), ""]
    seg_total = plan.segment_count

    for seg in plan.segments:
        meta = {
            "frames_label": _frames_label(seg),
            "task_key": seg.task_key,
        }

        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="prepare",
            phase_value=0,
            phase_max=1,
            **meta,
        )

        raw_clip = seg.source_clip.clone() if seg.source_clip is not None else slice_video_frames(
            source_video, seg.start_frame, seg.end_frame
        )
        target_len = raw_clip.shape[0]
        if seg.source_clip is not None:
            # Already scaled in plan build — keep each i2i/i2v group on its own canvas.
            clip = raw_clip
        elif plan.output_mode == "fixed":
            clip = fit_canvas(raw_clip, plan.width, plan.height)
        else:
            clip = fit_video_long_edge(raw_clip, plan.ref_max_size)
        clip, num_frames = prepare_segment_clip(clip, target_len)

        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="prepare",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        positive = seg.prompt
        seg_negative = (seg.negative_prompt or "").strip() or negative_prompt
        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="text_encode",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        text_embeds, _, _ = text_encoder.process(
            model_name=t5_model_name,
            precision=t5_precision,
            task_type=seg.task_type,
            positive_prompt=positive,
            negative_prompt=seg_negative,
            quantization=t5_quantization,
            use_disk_cache=use_disk_cache,
            device=t5_device,
        )
        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="text_encode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        ref_kwargs = refs_to_kwargs_for_context(seg.task_key, seg.refs)
        source_arg = clip if _needs_source_video(seg.task_key) else None

        if seg.task_key in ("i2i", "i2v") and clip is not None and clip.shape[0] > 0:
            ctx_h, ctx_w = int(clip.shape[1]), int(clip.shape[2])
        else:
            ctx_w, ctx_h = plan.width, plan.height

        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="context_encode",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        image_embeds, task_hint = context_node.build(
            vae=vae,
            width=ctx_w,
            height=ctx_h,
            num_frames=num_frames,
            source_video=source_arg,
            ref_max_size=plan.ref_max_size,
            tiled_vae=tiled_vae,
            force_offload=vae_force_offload,
            **ref_kwargs,
        )
        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="context_encode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="high_noise",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        samples_high, _ = sampler.process(
            model=model_high,
            image_embeds=image_embeds,
            scheduler=scheduler_high,
            text_embeds=text_embeds,
            cfg=high_noise_cfg,
            seed=high_noise_seed,
            force_offload=high_noise_force_offload,
            add_noise_to_samples=high_noise_add_noise_to_samples,
            extra_args=high_noise_extra_args,
        )
        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="high_noise",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="low_noise",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        samples_low, _ = sampler.process(
            model=model_low,
            image_embeds=image_embeds,
            scheduler=scheduler_low,
            text_embeds=text_embeds,
            samples=samples_high,
            cfg=low_noise_cfg,
            seed=low_noise_seed,
            force_offload=low_noise_force_offload,
            add_noise_to_samples=low_noise_add_noise_to_samples,
            extra_args=low_noise_extra_args,
        )
        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="low_noise",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="decode",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        decoded, = decoder.decode(
            vae=vae,
            samples=samples_low,
            enable_vae_tiling=enable_vae_tiling,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_stride_x=tile_stride_x,
            tile_stride_y=tile_stride_y,
            normalization=normalization,
        )
        report_director_progress(
            node_id,
            segment_index=seg.index,
            segment_total=seg_total,
            phase="decode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        if decoded.shape[0] > target_len:
            decoded = decoded[:target_len]
        elif decoded.shape[0] < target_len and decoded.shape[0] > 0:
            pad = decoded[-1:].repeat(target_len - decoded.shape[0], 1, 1, 1)
            decoded = torch.cat([decoded, pad], dim=0)

        chunk = decoded.cpu().float()
        output_chunks.append(chunk)
        segment_outputs.append(chunk)

        if plan.global_task_key in {"t2i", "i2i", "r2i"} and decoded.shape[0] >= 1:
            try:
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                report_director_segment_preview(
                    node_id,
                    segment_index=seg.index,
                    image_b64=_tensor_frame_to_jpeg_b64(decoded[0]),
                    width=w,
                    height=h,
                )
            except Exception as exc:
                log.debug("Segment preview skipped: %s", exc)
        elif plan.global_task_key in {"t2v", "i2v", "r2v"} and decoded.shape[0] >= 1:
            try:
                frames_b64 = [
                    _tensor_frame_to_jpeg_b64(decoded[i])
                    for i in range(int(decoded.shape[0]))
                ]
                h, w = int(decoded.shape[1]), int(decoded.shape[2])
                report_director_segment_preview(
                    node_id,
                    segment_index=seg.index,
                    image_b64=frames_b64[0],
                    width=w,
                    height=h,
                    frames=frames_b64,
                    fps=float(plan.frame_rate or 24),
                )
            except Exception as exc:
                log.debug("Segment video preview skipped: %s", exc)

        reports.append(
            f"Segment {seg.index + 1}/{plan.segment_count}: {task_hint} "
            f"({target_len} frames, high_seed={high_noise_seed}, low_seed={low_noise_seed})"
        )
        log.info(
            "Bernini Director segment %d/%d done (%d frames, task=%s)",
            seg.index + 1,
            plan.segment_count,
            target_len,
            seg.task_key,
        )

    if not output_chunks:
        raise ValueError("Director plan produced no segments.")

    report_director_finish(node_id, seg_total)

    combined = cat_frames_variable_size(output_chunks)
    return combined, segment_outputs, "\n".join(reports)
