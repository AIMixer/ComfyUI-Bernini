"""Parse Bernini Director timeline JSON and prepare per-segment edit plans."""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

import folder_paths

from ..ref_images import MAX_REFERENCE_IMAGES, REF_IMAGE_KEY_PREFIX
from ..image_prep import resolve_output_dimensions
from ..task_prompts import get_task_prompt_spec, resolve_task_key
from ..video_io import (
    logical_frame_count,
    logical_frame_map,
    load_multi_clip_timeline,
    load_video_resampled,
    parse_frame_map_entry,
    resolve_video_path,
    video_clips_from_timeline,
)
from .gen_timeline import (
    build_gen_director_plan,
    is_gen_timeline,
)

log = logging.getLogger("ComfyUI-Bernini.director")

MIN_SEGMENT_FRAMES = 4


@dataclass
class SegmentRef:
    index: int
    tensor: torch.Tensor


@dataclass
class SegmentPlan:
    index: int
    start_frame: int
    end_frame: int
    prompt: str
    task_type: str
    task_key: str
    use_global: bool
    refs: list[SegmentRef] = field(default_factory=list)
    negative_prompt: str = ""
    source_clip: torch.Tensor | None = None

    @property
    def frame_count(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class DirectorPlan:
    frame_rate: float
    total_frames: int
    width: int
    height: int
    ref_max_size: int
    output_mode: str
    source_width: int
    source_height: int
    global_task_type: str
    global_task_key: str
    global_prompt: str
    global_refs: list[SegmentRef]
    segments: list[SegmentPlan]
    source_video: torch.Tensor
    edit_mode: str
    raw: dict
    source_total_frames: int = 0
    export_max_frames: int = 0
    export_mode: str = "all"  # "all" | "segments"

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def wan_align_frame_count(frame_count: int) -> int:
    if frame_count <= 1:
        return 1
    return ((frame_count - 1) // 4) * 4 + 1


def _decode_image_b64(b64_str: str) -> torch.Tensor:
    if not b64_str:
        raise ValueError("Empty image data.")
    if b64_str.startswith("/view?"):
        raise ValueError("Remote view URLs are not supported; upload images in the Director node.")
    payload = b64_str.split(",", 1)[1] if "," in b64_str else b64_str
    img_bytes = base64.b64decode(payload)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def load_reference_tensor(ref: dict) -> torch.Tensor | None:
    if ref.get("imageFile"):
        rel = str(ref["imageFile"]).replace("\\", "/")
        file_path = os.path.join(folder_paths.get_input_directory(), rel.replace("/", os.sep))
        if os.path.exists(file_path):
            img = Image.open(file_path).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)

    b64_str = ref.get("imageB64", "")
    if not b64_str:
        return None
    try:
        return _decode_image_b64(b64_str)
    except Exception as exc:
        log.warning("Failed to decode reference image: %s", exc)
        return None


def load_source_video_from_timeline(timeline: dict) -> torch.Tensor:
    video = timeline.get("video") or {}
    frames_b64 = video.get("frames") or []
    if frames_b64:
        chunks: list[torch.Tensor] = []
        for frame_b64 in frames_b64:
            chunks.append(_decode_image_b64(frame_b64))
        if not chunks:
            raise ValueError("Uploaded video has no decodable frames.")
        return torch.cat(chunks, dim=0)

    clips = video_clips_from_timeline(timeline)
    if not clips:
        raise ValueError(
            "No source video in Bernini Director. Upload a video inside the node timeline UI before running."
        )

    frame_map = logical_frame_map(timeline)
    if not frame_map:
        raise ValueError("No frames in Bernini Director timeline frameMap.")

    frame_rate = float(timeline.get("frameRate") or 24)
    output_block = timeline.get("output") or {}
    default_long_edge = int(
        output_block.get("longEdge")
        or output_block.get("long_edge")
        or timeline.get("refMaxSize")
        or 848
    )

    entries = [parse_frame_map_entry(e) for e in frame_map]
    single_clip = len(clips) == 1 and all(c == 0 for c, _ in entries)

    if single_clip:
        clip = clips[0]
        path = resolve_video_path(clip)
        indices = [f for _, f in entries]
        return load_video_resampled(
            path,
            frame_rate,
            indices,
            storage_width=clip.get("storageWidth"),
            storage_height=clip.get("storageHeight"),
            long_edge=default_long_edge,
        )

    return load_multi_clip_timeline(
        timeline,
        frame_map,
        frame_rate=frame_rate,
        default_long_edge=default_long_edge,
    )


def _load_refs(ref_list: list[dict]) -> list[SegmentRef]:
    refs: list[SegmentRef] = []
    for item in ref_list or []:
        index = int(item.get("index", item.get("slot", len(refs))))
        if index < 0 or index >= MAX_REFERENCE_IMAGES:
            continue
        tensor = load_reference_tensor(item)
        if tensor is not None:
            refs.append(SegmentRef(index=index, tensor=tensor))
    return sorted(refs, key=lambda r: r.index)


def _segment_ranges_from_timeline(timeline: dict, total: int) -> list[tuple[int, int, dict]]:
    segments = timeline.get("segments") or []
    if segments and ("length" in segments[0] or "end" in segments[0]):
        ranges: list[tuple[int, int, dict]] = []
        for raw in sorted(segments, key=lambda s: int(s.get("start", 0))):
            start = int(raw.get("start", 0))
            if "end" in raw:
                end = int(raw["end"])
            else:
                end = start + int(raw.get("length", 0))
            start = max(0, min(start, total))
            end = max(start, min(end, total))
            if end - start >= MIN_SEGMENT_FRAMES or not ranges:
                ranges.append((start, end, raw))
        if ranges:
            return ranges

    split_points = timeline.get("splitPoints") or timeline.get("split_points") or []
    auto_count = int(timeline.get("autoSegmentCount") or timeline.get("auto_segment_count") or 0)
    if auto_count > 1:
        points = [int(round(total * i / auto_count)) for i in range(1, auto_count)]
    else:
        points = sorted({int(p) for p in split_points if 0 < int(p) < total})

    edges = [0] + points + [total]
    ranges = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if end <= start:
            continue
        raw = segments[i] if i < len(segments) else {}
        ranges.append((start, end, raw))
    return ranges or [(0, total, {})]


def _resolve_export_total(timeline: dict, source_total: int) -> int:
    output_block = timeline.get("output") or {}
    max_export = int(output_block.get("maxExportFrames") or output_block.get("max_export_frames") or 0)
    if max_export <= 0 or source_total <= 0:
        return source_total
    return min(source_total, max_export)


def _resolve_export_mode(output_block: dict) -> str:
    mode = str(output_block.get("exportMode") or output_block.get("export_mode") or "all").lower()
    if mode in ("segments", "segment", "per_segment", "by_segment"):
        return "segments"
    return "all"


def _clip_segment_ranges(
    ranges: list[tuple[int, int, dict]], export_total: int
) -> list[tuple[int, int, dict]]:
    if export_total <= 0:
        return ranges
    clipped: list[tuple[int, int, dict]] = []
    for start, end, data in ranges:
        if start >= export_total:
            break
        end = min(end, export_total)
        if end <= start:
            continue
        if end - start < MIN_SEGMENT_FRAMES and clipped:
            ps, _, pd = clipped[-1]
            clipped[-1] = (ps, end, pd)
        else:
            clipped.append((start, end, data))
    if not clipped and export_total > 0:
        data = ranges[0][2] if ranges else {}
        clipped.append((0, export_total, data))
    return clipped


def _trim_timeline_for_export(timeline: dict, export_total: int) -> dict:
    t = copy.deepcopy(timeline)
    video = dict(t.get("video") or {})
    frames_b64 = video.get("frames") or []
    if frames_b64 and export_total < len(frames_b64):
        video["frames"] = frames_b64[:export_total]
    fm = logical_frame_map(timeline)
    if fm and export_total < len(fm):
        video["frameMap"] = fm[:export_total]
    elif fm:
        video["frameMap"] = fm
    t["video"] = video
    t["totalFrames"] = export_total
    return t


def count_timeline_segments(timeline_data: str) -> int:
    """Estimate segment count from timeline JSON without loading video."""
    if not timeline_data or not str(timeline_data).strip():
        return 1
    try:
        timeline = json.loads(timeline_data)
    except json.JSONDecodeError:
        return 1

    segments = timeline.get("segments") or []
    global_task = (timeline.get("global") or {}).get("taskType") or ""
    task_key = resolve_task_key(global_task) if global_task else ""
    if is_gen_timeline(timeline, task_key):
        return max(1, len(segments) or 1)

    source_total = logical_frame_count(timeline) or int(timeline.get("totalFrames") or 0)
    export_total = _resolve_export_total(timeline, source_total)
    plan_total = export_total or source_total or 1
    ranges = _segment_ranges_from_timeline(timeline, source_total or plan_total)
    return max(1, len(_clip_segment_ranges(ranges, plan_total)))


def build_director_plan(
    timeline_data: str,
    *,
    global_task_type: str,
    global_prompt: str,
    total_frames: int,
    frame_rate: float,
    width: int,
    height: int,
    ref_max_size: int,
) -> DirectorPlan:
    timeline: dict = {}
    if timeline_data and timeline_data.strip():
        try:
            timeline = json.loads(timeline_data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid timeline_data JSON: {exc}") from exc

    global_block = timeline.get("global") or {}
    edit_mode = timeline.get("editMode") or timeline.get("edit_mode") or "global"
    if edit_mode not in ("global", "segment"):
        edit_mode = "global"

    task_type = global_block.get("taskType") or global_task_type or "rv2v — 参考素材改视频"
    prompt = global_block.get("prompt") or global_prompt or ""
    global_refs = _load_refs(global_block.get("refs") or [])

    task_key_early = resolve_task_key(task_type)
    if is_gen_timeline(timeline, task_key_early):
        return build_gen_director_plan(
            timeline,
            global_task_type=task_type,
            global_prompt=prompt,
            total_frames=total_frames,
            frame_rate=frame_rate,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
        )

    frame_map = logical_frame_map(timeline)
    source_total = len(frame_map) or int(timeline.get("totalFrames") or total_frames or 0)
    export_max = int(
        (timeline.get("output") or {}).get("maxExportFrames")
        or (timeline.get("output") or {}).get("max_export_frames")
        or 0
    )
    export_total = _resolve_export_total(timeline, source_total)

    load_timeline = _trim_timeline_for_export(timeline, export_total) if export_total < source_total else timeline
    source_video = load_source_video_from_timeline(load_timeline)
    source_h = int(source_video.shape[1])
    source_w = int(source_video.shape[2])
    video_meta = timeline.get("video") or {}
    source_w = int(video_meta.get("width") or source_w or width)
    source_h = int(video_meta.get("height") or source_h or height)

    output_block = timeline.get("output") or {}
    export_mode = _resolve_export_mode(output_block)
    out_w, out_h, ref_max, output_mode = resolve_output_dimensions(
        int(video_meta.get("width") or source_w or width),
        int(video_meta.get("height") or source_h or height),
        mode=str(output_block.get("mode") or "long_edge"),
        long_edge=int(output_block.get("longEdge") or output_block.get("long_edge") or ref_max_size or 848),
        fixed_width=int(output_block.get("width") or timeline.get("width") or width),
        fixed_height=int(output_block.get("height") or timeline.get("height") or height),
    )

    total = int(load_timeline.get("totalFrames") or export_total or total_frames or source_video.shape[0] or 0)
    if total <= 0:
        total = int(source_video.shape[0])
    if source_video.shape[0] < total:
        total = int(source_video.shape[0])

    segment_ranges = _segment_ranges_from_timeline(timeline, source_total or total)
    segment_ranges = _clip_segment_ranges(segment_ranges, total)
    segments: list[SegmentPlan] = []

    for idx, (start, end, seg_data) in enumerate(segment_ranges):
        if edit_mode == "global":
            seg_prompt = prompt
            seg_task = task_type
            seg_refs = list(global_refs)
            use_global = True
        else:
            use_global = False
            seg_prompt = (seg_data.get("prompt") or "").strip() or prompt
            seg_task = seg_data.get("taskType") or seg_data.get("task_type") or task_type
            seg_refs = _load_refs(seg_data.get("refs") or [])

        seg_task_key = resolve_task_key(seg_task)
        seg_refs = segment_refs_for_context(seg_task_key, seg_refs)

        segments.append(
            SegmentPlan(
                index=idx,
                start_frame=start,
                end_frame=end,
                prompt=seg_prompt,
                task_type=seg_task,
                task_key=seg_task_key,
                use_global=use_global,
                refs=seg_refs,
            )
        )

    return DirectorPlan(
        frame_rate=float(timeline.get("frameRate") or frame_rate or 24),
        total_frames=total,
        width=out_w,
        height=out_h,
        ref_max_size=ref_max,
        output_mode=output_mode,
        source_width=int(video_meta.get("width") or source_w),
        source_height=int(video_meta.get("height") or source_h),
        global_task_type=task_type,
        global_task_key=resolve_task_key(task_type),
        global_prompt=prompt,
        global_refs=global_refs,
        segments=segments,
        source_video=source_video,
        edit_mode=edit_mode,
        raw=timeline,
        source_total_frames=source_total or total,
        export_max_frames=export_max,
        export_mode=export_mode,
    )


def slice_video_frames(source: torch.Tensor, start: int, end: int) -> torch.Tensor:
    end = min(end, source.shape[0])
    start = max(0, min(start, end))
    return source[start:end].clone()


def prepare_segment_clip(clip: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, int]:
    actual = clip.shape[0]
    if actual <= 0:
        raise ValueError("Segment has no frames.")
    num_frames = wan_align_frame_count(max(actual, target_frames))
    if actual < num_frames:
        pad = clip[-1:].repeat(num_frames - actual, 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)
    elif actual > num_frames:
        clip = clip[:num_frames]
    return clip, num_frames


# i2v uses source video context (frame0 image + gray tail); img0–img4 must not join context_latents.
CONTEXT_REFERENCE_EXCLUDED_KEYS = frozenset({"i2v"})


def segment_refs_for_context(task_key: str, refs: list[SegmentRef]) -> list[SegmentRef]:
    if task_key in CONTEXT_REFERENCE_EXCLUDED_KEYS:
        return []
    return refs


def refs_to_kwargs(refs: list[SegmentRef]) -> dict[str, torch.Tensor]:
    return {f"{REF_IMAGE_KEY_PREFIX}{ref.index}": ref.tensor for ref in refs}


def refs_to_kwargs_for_context(task_key: str, refs: list[SegmentRef]) -> dict[str, torch.Tensor]:
    return refs_to_kwargs(segment_refs_for_context(task_key, refs))


def plan_summary(plan: DirectorPlan) -> str:
    mode = str(plan.raw.get("timelineMode") or "")
    if mode in ("gen_blank", "gen_image", "prompt_batch", "image_batch"):
        if mode in ("prompt_batch", "image_batch"):
            mode_label = f"批量生成 ({plan.global_task_key})"
        else:
            mode_label = "空白画布" if mode == "gen_blank" else "图片生成"
        lines = [
            f"Bernini Director [{mode_label}] ({plan.edit_mode}): "
            f"{plan.segment_count} segment(s), {plan.total_frames} frames @ {plan.frame_rate:.2f} fps",
            f"Output: {plan.width}×{plan.height} ({plan.output_mode})",
            f"Global task: {get_task_prompt_spec(plan.global_task_type).label}",
        ]
        for seg in plan.segments:
            lines.append(
                f"  #{seg.index + 1} [{seg.start_frame}:{seg.end_frame}] "
                f"{seg.frame_count}f — {seg.task_key} — {seg.prompt[:60]}{'…' if len(seg.prompt) > 60 else ''}"
            )
        return "\n".join(lines)

    lines = [
        f"Bernini Director ({plan.edit_mode}): {plan.segment_count} segment(s), "
        f"{plan.total_frames} frames @ {plan.frame_rate:.2f} fps",
    ]
    if plan.export_max_frames > 0 and plan.source_total_frames > plan.total_frames:
        lines.append(
            f"Export cap: {plan.total_frames}/{plan.source_total_frames} frames "
            f"(max {plan.export_max_frames})"
        )
    export_label = "分段导出" if plan.export_mode == "segments" else "全部导出"
    lines.append(f"Export mode: {export_label}")
    lines.append(f"Global task: {get_task_prompt_spec(plan.global_task_type).label}")
    for seg in plan.segments:
        lines.append(
            f"  #{seg.index + 1} [{seg.start_frame}:{seg.end_frame}] "
            f"{seg.frame_count}f — {seg.task_key} — {seg.prompt[:60]}{'…' if len(seg.prompt) > 60 else ''}"
        )
    return "\n".join(lines)
