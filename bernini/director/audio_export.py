"""Build ComfyUI AUDIO outputs for Bernini Director v2v / rv2v runs."""

from __future__ import annotations

from typing import Any

from ..audio_io import extract_timeline_audio


def task_passes_source_audio(task_key: str) -> bool:
    return task_key in {"v2v", "rv2v"}


def build_director_audio_outputs(
    plan,
    images_out: list,
    *,
    export_segments: bool,
    output_frame_end: int | None = None,
) -> list[Any]:
    """Return one AUDIO dict (or None) per images_out entry."""
    if not task_passes_source_audio(plan.global_task_key):
        return [None] * len(images_out)

    timeline = plan.raw or {}
    fps = float(plan.frame_rate or 24)

    if export_segments:
        if plan.run_indices is not None:
            seg_indices = sorted(plan.run_indices)
        else:
            seg_indices = list(range(len(plan.segments)))
        outputs: list[Any] = []
        for i, _tensor in enumerate(images_out):
            if i >= len(seg_indices):
                outputs.append(None)
                continue
            seg = plan.segments[seg_indices[i]]
            outputs.append(extract_timeline_audio(timeline, seg.start_frame, seg.end_frame, fps))
        return outputs

    end = max(0, int(output_frame_end if output_frame_end is not None else plan.total_frames))
    audio = extract_timeline_audio(timeline, 0, end, fps) if end > 0 else None
    return [audio] if len(images_out) == 1 else [None] * len(images_out)
