"""Run Bernini Director segments through the official ComfyUI core pipeline."""

from __future__ import annotations

import logging

import torch

from ..image_prep import fit_canvas, fit_video_long_edge, cat_frames_variable_size
from ..nodes.conditioning import _run_conditioning
from .core_sampling import apply_apg, apply_model_sampling_shift, sample_dual_stage
from .core_text_encode import encode_core_conditioning
from .executor import (
    _frames_label,
    _is_gen_timeline_plan,
    _needs_source_video,
    _resolve_segment_raw_clip,
    _segment_passthrough_chunk,
    _source_passthrough_chunk,
    _tensor_frame_to_jpeg_b64,
)
from .plan import (
    DirectorPlan,
    plan_summary,
    prepare_segment_clip,
    reference_video_for_segment,
    refs_to_kwargs_for_context,
    wan_align_frame_count,
)
from .progress import report_director_finish, report_director_progress, report_director_segment_preview
from .segment_cache import load_segment_cache, save_segment_cache
from .vram_cleanup import cleanup_segment_vram

log = logging.getLogger("ComfyUI-Bernini.director.core")


def execute_director_plan_core(
    plan: DirectorPlan,
    *,
    node_id: str | None = None,
    vae,
    model_high,
    model_low,
    clip,
    negative_prompt: str,
    high_noise_cfg: float = 1.0,
    high_noise_seed: int = 0,
    low_noise_cfg: float = 1.0,
    low_noise_seed: int = 0,
    steps: int = 6,
    split_step: int = 3,
    sampler: str = "res_multistep",
    scheduler: str = "simple",
    model_shift: float = 0.0,
    apg_eta: float = 1.0,
    apg_momentum: float = 0.0,
    apg_norm_threshold: float = 0.0,
    clear_vram_between_segments: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor], str]:
    """Process every segment with ComfyUI core Bernini conditioning + KSampler."""
    from nodes import VAEDecode

    decoder = VAEDecode()

    model_high = apply_apg(
        apply_model_sampling_shift(model_high, model_shift),
        eta=apg_eta,
        momentum=apg_momentum,
        norm_threshold=apg_norm_threshold,
    )
    model_low = apply_apg(
        apply_model_sampling_shift(model_low, model_shift),
        eta=apg_eta,
        momentum=apg_momentum,
        norm_threshold=apg_norm_threshold,
    )

    all_segments = plan.segments
    run_indices = plan.run_indices if plan.run_indices is not None else frozenset(range(len(all_segments)))
    run_list = sorted(run_indices)
    seg_total = len(run_list)
    progress_pos = {idx: pos for pos, idx in enumerate(run_list)}

    output_chunks: list[torch.Tensor] = []
    segment_outputs: list[torch.Tensor] = []
    reports: list[str] = [plan_summary(plan), "", "Execution backend: 官方流 (ComfyUI core)"]
    if clear_vram_between_segments:
        reports.append(
            "VRAM: 段间清理显存已开启（多段时在 context 编码后卸载模型；"
            "单段时跳过采样前 unload，避免重载叠峰）"
        )
    if plan.run_indices is not None:
        skipped = [i + 1 for i in range(len(all_segments)) if i not in run_indices]
        reports.append(
            f"Run selection: {len(run_list)}/{len(all_segments)} segment(s) "
            f"(indices {[i + 1 for i in run_list]}; skipped {skipped or 'none'})"
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

        raw_clip = _resolve_segment_raw_clip(plan, seg)
        is_one_frame_i2v = seg.task_key == "i2v" and seg.source_clip is not None
        target_len = max(1, seg.frame_count or raw_clip.shape[0]) if is_one_frame_i2v else raw_clip.shape[0]
        if seg.source_clip is not None:
            clip_frames = raw_clip
        elif plan.output_mode == "fixed":
            clip_frames = fit_canvas(raw_clip, plan.width, plan.height)
        else:
            clip_frames = fit_video_long_edge(raw_clip, plan.ref_max_size)
        if is_one_frame_i2v:
            num_frames = wan_align_frame_count(target_len)
        else:
            clip_frames, num_frames = prepare_segment_clip(clip_frames, target_len)

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="prepare",
            phase_value=1,
            phase_max=1,
            **meta,
        )

        positive_prompt = seg.prompt
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
        positive, negative = encode_core_conditioning(
            clip,
            task_type=seg.task_type,
            positive_prompt=positive_prompt,
            negative_prompt=seg_negative,
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

        if clear_vram_between_segments:
            cleanup_segment_vram(enabled=True)

        ref_kwargs = refs_to_kwargs_for_context(seg.task_key, seg.refs)
        source_arg = clip_frames if _needs_source_video(seg.task_key) else None
        ref_video_arg = reference_video_for_segment(plan, seg, num_frames)

        if clip_frames is not None and clip_frames.shape[0] > 0:
            ctx_h, ctx_w = int(clip_frames.shape[1]), int(clip_frames.shape[2])
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
        positive, negative, latent, task_hint = _run_conditioning(
            positive,
            negative,
            vae,
            ctx_w,
            ctx_h,
            num_frames,
            1,
            source_video=source_arg,
            reference_video=ref_video_arg,
            ref_max_size=plan.ref_max_size,
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

        if clear_vram_between_segments:
            cleanup_segment_vram(enabled=True, unload_models=seg_total > 1)

        report_director_progress(
            node_id,
            segment_index=progress_index,
            segment_total=seg_total,
            phase="high_noise",
            phase_value=0,
            phase_max=1,
            **meta,
        )
        samples = sample_dual_stage(
            model_high=model_high,
            model_low=model_low,
            positive=positive,
            negative=negative,
            latent=latent,
            high_seed=high_noise_seed,
            low_seed=low_noise_seed,
            high_cfg=high_noise_cfg,
            low_cfg=low_noise_cfg,
            steps=steps,
            split_step=split_step,
            sampler_name=sampler,
            scheduler=scheduler,
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
        decoded, = decoder.decode(vae, samples)
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

        if clear_vram_between_segments:
            del positive, negative, latent, samples, decoded, clip_frames, source_arg, raw_clip
            cleanup_segment_vram(enabled=True)

        reports.append(
            f"Segment {seg.index + 1}/{len(all_segments)}: {task_hint} "
            f"({target_len} frames, high_seed={high_noise_seed}, low_seed={low_noise_seed})"
        )
        log.info(
            "Bernini Director [core] segment %d/%d done (%d frames, task=%s)",
            seg.index + 1,
            len(all_segments),
            target_len,
            seg.task_key,
        )
        return chunk

    for seg in all_segments:
        if seg.index in run_indices:
            if clear_vram_between_segments and segment_outputs:
                cleanup_segment_vram(enabled=True)
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
                    save_segment_cache(node_id, seg, plan, cached)
                    reports.append(
                        f"Segment {seg.index + 1}/{len(all_segments)}: "
                        f"source passthrough ({cached.shape[0]} frames, no prior cache)"
                    )
            except Exception as exc:
                log.warning("Segment %d source passthrough failed: %s", seg.index + 1, exc)
                cached = None
        if cached is None:
            raise ValueError(
                f"Segment {seg.index + 1} is not selected and has no valid cache. "
                "Run all segments once (全部运行), or include this segment in your run selection."
            )
        output_chunks.append(cached.float())

    if not output_chunks and not segment_outputs:
        raise ValueError("Director plan produced no segments.")

    report_director_finish(node_id, seg_total)
    combined = cat_frames_variable_size(output_chunks) if output_chunks else cat_frames_variable_size(segment_outputs)
    return combined, segment_outputs, "\n".join(reports)
