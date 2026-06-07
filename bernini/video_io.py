"""Load source videos from ComfyUI input folder (VHS-style file references)."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch

import folder_paths

from .image_prep import resolve_output_dimensions

log = logging.getLogger("ComfyUI-Bernini.video_io")

MAX_SOURCE_FRAMES = 512


def _require_cv2():
    try:
        import cv2

        return cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for Bernini Director video loading. "
            "Install: pip install opencv-python-headless"
        ) from exc


def resolve_video_path(video: dict) -> str:
    """Resolve timeline video metadata to an absolute path under ComfyUI input."""
    video_file = (video.get("videoFile") or video.get("fileName") or "").strip()
    if not video_file:
        raise ValueError("No video file in Bernini Director timeline.")

    base = folder_paths.get_input_directory()
    subfolder = (video.get("subfolder") or "").strip().replace("\\", "/")

    candidates = []
    if subfolder and not video_file.startswith(subfolder):
        candidates.append(os.path.join(base, subfolder, os.path.basename(video_file)))
    candidates.append(os.path.join(base, video_file.replace("/", os.sep)))
    candidates.append(os.path.join(base, os.path.basename(video_file)))

    for path in candidates:
        if os.path.isfile(path):
            return path

    raise ValueError(f"Video file not found in ComfyUI input: {video_file}")


def load_video_resampled(
    path: str,
    frame_rate: float,
    frame_indices: Sequence[int],
    *,
    storage_width: int | None = None,
    storage_height: int | None = None,
    long_edge: int = 848,
) -> torch.Tensor:
    """Decode selected resampled frame indices from a video file."""
    if not frame_indices:
        raise ValueError("No frames requested from video.")

    cv2 = _require_cv2()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = float(frame_rate or 24.0)

    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_w, out_h, rotate_90_cw = _resolve_load_dimensions(
        source_w,
        source_h,
        storage_width=storage_width,
        storage_height=storage_height,
        long_edge=long_edge,
    )

    unique = sorted({int(i) for i in frame_indices})
    decoded: dict[int, np.ndarray] = {}
    fallback: np.ndarray | None = None

    for src_idx in unique:
        t_sec = max(0.0, src_idx / float(frame_rate or 24.0))
        native_frame = int(round(t_sec * native_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, native_frame)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            log.warning("Failed to read frame %d (t=%.3fs) from %s", native_frame, t_sec, path)
            if fallback is not None:
                decoded[src_idx] = fallback
            continue

        if rotate_90_cw:
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        if (bgr.shape[1], bgr.shape[0]) != (out_w, out_h):
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        decoded[src_idx] = rgb
        fallback = rgb

    cap.release()

    if not decoded:
        raise ValueError(f"No frames decoded from video: {path}")

    rows = []
    last = next(iter(decoded.values()))
    for idx in frame_indices:
        rows.append(decoded.get(int(idx), last))
        last = rows[-1]

    return torch.from_numpy(np.stack(rows, axis=0))


def _aspect_ratio(w: int, h: int) -> float:
    return w / h if h > 0 else 0.0


def _aspect_close(a: float, b: float, *, tol: float = 0.04) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / max(a, b) <= tol


def _resolve_load_dimensions(
    source_w: int,
    source_h: int,
    *,
    storage_width: int | None,
    storage_height: int | None,
    long_edge: int,
) -> tuple[int, int, bool]:
    """Return (out_w, out_h, rotate_90_cw) for proportional long-edge loading."""
    if storage_width and storage_height and source_w > 0 and source_h > 0:
        sw, sh = int(storage_width), int(storage_height)
        native_ar = _aspect_ratio(source_w, source_h)
        storage_ar = _aspect_ratio(sw, sh)
        if _aspect_close(native_ar, storage_ar):
            return sw, sh, False
        # Browser/UI metadata often includes rotation; OpenCV raw frames may be transposed.
        if _aspect_close(_aspect_ratio(source_h, source_w), storage_ar):
            log.info(
                "Video %dx%d decoded transposed vs storage %dx%d; applying 90° rotation before scale",
                source_w,
                source_h,
                sw,
                sh,
            )
            return sw, sh, True
        log.warning(
            "storage %dx%d aspect mismatch vs native %dx%d; using proportional long_edge=%d",
            sw,
            sh,
            source_w,
            source_h,
            long_edge,
        )

    out_w, out_h, _, _ = resolve_output_dimensions(
        source_w,
        source_h,
        mode="long_edge",
        long_edge=long_edge,
    )
    return out_w, out_h, False


def parse_frame_map_entry(entry: Any, default_clip: int = 0) -> tuple[int, int]:
    """Parse a frameMap entry to (clip_index, source_frame_index)."""
    if isinstance(entry, dict):
        clip = int(entry.get("clip", entry.get("videoClip", default_clip)))
        frame = int(entry.get("frame", 0))
        return clip, frame
    return default_clip, int(entry)


def video_clips_from_timeline(timeline: dict) -> list[dict]:
    """Return ordered video clip metadata; falls back to legacy single ``video`` block."""
    clips = timeline.get("videoClips") or timeline.get("video_clips")
    if clips:
        return list(clips)
    video = timeline.get("video") or {}
    if (video.get("videoFile") or video.get("fileName") or "").strip():
        return [video]
    return []


def logical_frame_map(timeline: dict) -> list[Any]:
    """Ordered logical timeline frame map (ints or {clip, frame} objects)."""
    video = timeline.get("video") or {}
    frame_map = video.get("frameMap")
    if frame_map:
        return list(frame_map)

    total = int(timeline.get("totalFrames") or 0)
    if total > 0:
        return list(range(total))

    source_count = int(video.get("sourceFrameCount") or 0)
    if source_count > 0:
        return list(range(source_count))

    return []


def logical_frame_count(timeline: dict) -> int:
    fm = logical_frame_map(timeline)
    if fm:
        return len(fm)
    return int(timeline.get("totalFrames") or 0)


def frame_indices_from_timeline(timeline: dict) -> list[int]:
    """Legacy helper: source-frame indices for single-clip timelines."""
    entries = [parse_frame_map_entry(e) for e in logical_frame_map(timeline)]
    if entries and all(c == 0 for c, _ in entries):
        return [f for _, f in entries]
    return list(range(logical_frame_count(timeline)))


def load_multi_clip_timeline(
    timeline: dict,
    frame_map: list[Any],
    *,
    frame_rate: float,
    default_long_edge: int,
) -> torch.Tensor:
    """Decode a logical timeline that may reference multiple source videos."""
    clips = video_clips_from_timeline(timeline)
    if not clips:
        raise ValueError("No video clips in Bernini Director timeline.")

    entries = [parse_frame_map_entry(e) for e in frame_map]
    by_clip: dict[int, set[int]] = defaultdict(set)
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        by_clip[clip_idx].add(frame_idx)

    frame_tensors: dict[tuple[int, int], torch.Tensor] = {}
    for clip_idx, frame_set in sorted(by_clip.items()):
        clip = clips[clip_idx]
        path = resolve_video_path(clip)
        sorted_idx = sorted(frame_set)
        tensor = load_video_resampled(
            path,
            frame_rate,
            sorted_idx,
            storage_width=clip.get("storageWidth"),
            storage_height=clip.get("storageHeight"),
            long_edge=int(clip.get("longEdge") or default_long_edge),
        )
        for row, fi in enumerate(sorted_idx):
            frame_tensors[(clip_idx, fi)] = tensor[row]

    rows: list[torch.Tensor] = []
    fallback: torch.Tensor | None = None
    for clip_idx, frame_idx in entries:
        if clip_idx < 0 or clip_idx >= len(clips):
            clip_idx = 0
        key = (clip_idx, frame_idx)
        tensor = frame_tensors.get(key, fallback)
        if tensor is None:
            raise ValueError(f"Missing decoded frame for clip {clip_idx} frame {frame_idx}")
        fallback = tensor
        rows.append(tensor)

    return torch.stack(rows, dim=0)
