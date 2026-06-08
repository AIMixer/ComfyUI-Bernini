"""Run Bernini pipeline for each Director segment and concatenate output."""

from __future__ import annotations

import logging

import torch

from ..image_prep import fit_canvas, fit_video_long_edge, cat_frames_variable_size
from ..nodes.text_encode import BerniniTextEncodeCached
from ..nodes.wan import BerniniWanContextEmbeds
from ...engine.bernini_core_nodes import WanVideoDecode
from ...engine.nodes_sampler import WanVideoSamplerv2

from ..video_io import load_timeline_segment
from .plan import (
    DirectorPlan,
    plan_summary,
    prepare_segment_clip,
    refs_to_kwargs_for_context,
)
from .segment_cache import load_segment_cache, save_segment_cache
from .progress import report_director_finish, report_director_progress, report_director_segment_preview

log = logging.getLogger("ComfyUI-Bernini.director")


def _needs_source_video(task_key: str) -> bool:
    return task_key in {"v2v", "rv2v", "vi2v", "vrc2v", "mv2v", "ads2v", "i2v", "i2i"}


def _source_passthrough_chunk(plan: DirectorPlan, seg) -> torch.Tensor:
    """Scaled source frames for skipped v2v segments with no generation cache yet."""
    raw_clip = load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)
    target_len = raw_clip.shape[0]
    if plan.output_mode == "fixed":
        clip = fit_canvas(raw_clip, plan.width, plan.height)
    else:
        clip = fit_video_long_edge(raw_clip, plan.ref_max_size)
    clip, _ = prepare_segment_clip(clip, target_len)
    return clip.cpu().float()


def _segment_passthrough_chunk(plan: DirectorPlan, seg) -> torch.Tensor | None:
    """Best-effort fill for skipped segments (gen source clip, then timeline video)."""
    if seg.source_clip is not None and seg.source_clip.shape[0] > 0:
        target_len = max(1, seg.frame_count or int(seg.source_clip.shape[0]))
        clip = seg.source_clip.clone()
        if clip.shape[0] > target_len:
            clip = clip[:target_len]
        elif clip.shape[0] < target_len:
            pad = clip[-1:].repeat(target_len - clip.shape[0], 1, 1, 1)
            clip = torch.cat([clip, pad], dim=0)
        return clip.cpu().float()
    if _needs_source_video(seg.task_key):
        try:
            return _source_passthrough_chunk(plan, seg)
        except Exception:
            return None
    return None


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
    text_encoder = BerniniTextEncodeCached()
    context_node = BerniniWanContextEmbeds()
    sampler = WanVideoSamplerv2()
    decoder = WanVideoDecode()

    all_segments = plan.segments
    run_indices = plan.run_indices if plan.run_indices is not None else frozenset(range(len(all_segments)))
    run_list = sorted(run_indices)
    seg_total = len(run_list)
    progress_pos = {idx: pos for pos, idx in enumerate(run_list)}

    output_chunks: list[torch.Tensor] = []
    segment_outputs: list[torch.Tensor] = []
    reports: list[str] = [plan_summary(plan), ""]
    if plan.run_indices is not None:
        skipped = [i + 1 for i in range(len(all_segments)) if i not in run_indices]
        reports.append(
            f"Run selection: {len(run_list)}/{len(all_segments)} segment(s) "
            f"(indices {[i + 1 for i in run_list]}; skipped {skipped or 'none'})"
        )
        if plan.export_mode == "all" and skipped:
            reports.append(
                "Export mode all: skipped segment(s) use cache when available; "
                "v2v/i2v segments without cache fall back to source video for merge."
            )

    def _run_one_segment(seg, *, progress_index: int) -> torch.Tensor:
        meta = {
            "frames_label": _frames_label(seg),
            "task_key": seg.task_key,
            "timeline_segment_index": seg.index,
            "timeline_segment_total": len(all_segments),
        }

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="prepare",
            phase_value=0,
            phase_max=1,
            **meta,
        )

        raw_clip = seg.source_clip.clone() if seg.source_clip is not None else load_timeline_segment(
            plan.raw, seg.start_frame, seg.end_frame
        )
        target_len = raw_clip.shape[0]
        if seg.source_clip is not None:
            clip = raw_clip
        elif plan.output_mode == "fixed":
            clip = fit_canvas(raw_clip, plan.width, plan.height)
        else:
            clip = fit_video_long_edge(raw_clip, plan.ref_max_size)
        clip, num_frames = prepare_segment_clip(clip, target_len)

        report_director_progress(
            node_id,
            segment_index=progress_index,
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
            segment_index=progress_index,
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
            segment_index=progress_index,
            segment_total=seg_total,
            phase="text_encode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        ref_kwargs = refs_to_kwargs_for_context(seg.task_key, seg.refs)
        source_arg = clip if _needs_source_video(seg.task_key) else None

        if clip is not None and clip.shape[0] > 0:
            ctx_h, ctx_w = int(clip.shape[1]), int(clip.shape[2])
        else:
            ctx_w, ctx_h = plan.width, plan.height

        report_director_progress(
            node_id,
            segment_index=progress_index,
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
            segment_index=progress_index,
            segment_total=seg_total,
            phase="context_encode",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        report_director_progress(
            node_id,
            segment_index=progress_index,
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
            segment_index=progress_index,
            segment_total=seg_total,
            phase="high_noise",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        report_director_progress(
            node_id,
            segment_index=progress_index,
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
            segment_index=progress_index,
            segment_total=seg_total,
            phase="low_noise",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        report_director_progress(
            node_id,
            segment_index=progress_index,
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
            segment_index=progress_index,
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
        save_segment_cache(node_id, seg, plan, chunk)

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
            f"Segment {seg.index + 1}/{len(all_segments)}: {task_hint} "
            f"({target_len} frames, high_seed={high_noise_seed}, low_seed={low_noise_seed})"
        )
        log.info(
            "Bernini Director segment %d/%d done (%d frames, task=%s)",
            seg.index + 1,
            len(all_segments),
            target_len,
            seg.task_key,
        )
        return chunk

    for seg in all_segments:
        if seg.index in run_indices:
            chunk = _run_one_segment(seg, progress_index=progress_pos[seg.index])
            segment_outputs.append(chunk)
            if plan.export_mode == "all":
                output_chunks.append(chunk)
            continue

        if plan.export_mode != "all":
            continue

        cached = load_segment_cache(node_id, seg, plan)
        if cached is not None:
            reports.append(
                f"Segment {seg.index + 1}/{len(all_segments)}: "
                f"loaded from cache ({cached.shape[0]} frames)"
            )
        elif _needs_source_video(seg.task_key) or seg.source_clip is not None:
            try:
                cached = _segment_passthrough_chunk(plan, seg)
                if cached is not None:
                    log.info(
                        "Segment %d: no generation cache — using source/passthrough for merge.",
                        seg.index + 1,
                    )
                    save_segment_cache(node_id, seg, plan, cached)
                    reports.append(
                        f"Segment {seg.index + 1}/{len(all_segments)}: "
                        f"source passthrough ({cached.shape[0]} frames, no prior cache)"
                    )
            except Exception as exc:
                log.warning(
                    "Segment %d source passthrough failed: %s",
                    seg.index + 1,
                    exc,
                )
                cached = None
        if cached is None:
            raise ValueError(
                f"Segment {seg.index + 1} is not selected and has no valid cache. "
                "Run all segments once (全部运行), or include this segment in your run selection. "
                "Partial re-run with 「全部导出」 requires cached results for skipped segments "
                "(v2v/i2v may use source video or uploaded image when available)."
            )
        output_chunks.append(cached.float())

    if not output_chunks and not segment_outputs:
        raise ValueError("Director plan produced no segments.")

    report_director_finish(node_id, seg_total)

    combined = cat_frames_variable_size(output_chunks) if output_chunks else cat_frames_variable_size(segment_outputs)
    return combined, segment_outputs, "\n".join(reports)
