"""Cross-segment continuity: light context guidance + seam crossfade (full timeline export)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..encoders import ComfyCoreEncoder, WanVaeEncoder
from ..image_prep import cat_frames_variable_size, fit_canvas
from .plan import DirectorPlan, SegmentPlan, wan_align_frame_count
from .segment_cache import load_segment_cache

log = logging.getLogger("ComfyUI-Bernini.director.continuity")

CONTINUITY_TASK_KEYS = frozenset({"v2v", "rv2v", "vi2v", "vrc2v", "mv2v", "ads2v", "i2v"})

DEFAULT_CONTINUITY_OVERLAP = 9
MIN_CONTINUITY_OVERLAP = 1
MAX_CONTINUITY_OVERLAP = 81
# Generation caps — long tails / many ref streams can cause temporal flicker.
MAX_CONTINUITY_CONTEXT_FRAMES = 1
MAX_CONTINUITY_REF_FRAMES = 1
# No merge-time pixel crossfade (causes ghosting/flicker at seams).
MAX_CONTINUITY_SEAM_BLEND = 0
MIN_CONTINUITY_SEAM_BLEND = 0
# Post-decode pixel blends / appearance anchors cause ghosting — gen-side motion+ref only.
MAX_OPENING_BLEND_FRAMES = 0
OPENING_BLEND_PULL = 0.0
APPEARANCE_ANCHOR_DUPLICATES = 0
APPEARANCE_COLOR_MATCH_STRENGTH = 0.0
MAX_APPEARANCE_COLOR_MATCH_FRAMES = 0


def resolve_continuity_settings(timeline: dict, *, segment_count: int) -> tuple[bool, int]:
    """Read segment continuity flags from timeline JSON (output only; default off)."""
    if segment_count < 2:
        return False, 0
    output = timeline.get("output") or {}
    enabled = bool(
        output.get("continuityEnabled") is True
        or output.get("continuity_enabled") is True
    )
    if not enabled:
        return False, 0
    raw = (
        output.get("continuityOverlapFrames")
        or output.get("continuity_overlap_frames")
        or DEFAULT_CONTINUITY_OVERLAP
    )
    overlap = max(MIN_CONTINUITY_OVERLAP, min(MAX_CONTINUITY_OVERLAP, int(raw)))
    return True, overlap


def resolve_continuity_lock_pixels(overlap_frames: int) -> int:
    """Wan-aligned pixel prefix locked via SCAIL noise_mask (uses full UI overlap)."""
    ov = max(MIN_CONTINUITY_OVERLAP, min(MAX_CONTINUITY_OVERLAP, int(overlap_frames)))
    return wan_align_frame_count(ov)


def resolve_continuity_guide_frames(overlap_frames: int) -> tuple[int, int, int, int, int]:
    """Map UI overlap → (context_px, tail_refs, seam_blend, opening_blend, color_match)."""
    ov = max(MIN_CONTINUITY_OVERLAP, int(overlap_frames))
    ctx = wan_align_frame_count(min(ov, MAX_CONTINUITY_CONTEXT_FRAMES))
    refs = min(MAX_CONTINUITY_REF_FRAMES, max(1, min(ov, MAX_CONTINUITY_REF_FRAMES)))
    return ctx, refs, 0, 0, 0


def resolve_segment_generation_frames(
    *,
    segment_frame_count: int,
    segment_index: int,
    continuity_enabled: bool,
    continuity_overlap: int,
) -> tuple[int, int]:
    """Return (wan_aligned_gen_frames, prefix_trim_after_decode) for SCAIL handoff."""
    base = wan_align_frame_count(max(1, int(segment_frame_count)))
    if not continuity_enabled or segment_index <= 0:
        return base, 0
    lock_px = resolve_continuity_lock_pixels(continuity_overlap)
    return wan_align_frame_count(base + lock_px), lock_px


# Legacy helper default (no longer used on export path).
CONTINUITY_TRANSITION_FRAMES = 0


def _continuity_active(plan: DirectorPlan, seg: SegmentPlan) -> bool:
    return (
        plan.continuity_enabled
        and plan.segment_count >= 2
        and seg.index > 0
        and seg.task_key in CONTINUITY_TASK_KEYS
    )


def resolve_prev_segment_output(
    plan: DirectorPlan,
    all_segments: list[SegmentPlan],
    seg_index: int,
    completed: dict[int, torch.Tensor],
    node_id: str | None,
) -> torch.Tensor | None:
    prev_idx = seg_index - 1
    if prev_idx < 0:
        return None
    if prev_idx in completed:
        return completed[prev_idx]
    prev_seg = all_segments[prev_idx]
    cached = load_segment_cache(node_id, prev_seg, plan)
    if cached is not None:
        return cached
    if not plan.continuity_enabled:
        return None
    raise ValueError(
        f"段间连贯：片段 #{seg_index + 1} 需要上一段 #{prev_idx + 1} 的生成结果。"
        "请先运行上一段，或开启「全部运行」以生成完整序列；"
        "若使用「选择运行」，请确保上一段已有有效缓存。"
    )


def _normalize_context_latent_4d(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim == 5:
        if int(latent.shape[0]) != 1:
            raise ValueError(f"Context latent batch must be 1, got {tuple(latent.shape)}")
        latent = latent.squeeze(0)
    elif latent.ndim == 3:
        latent = latent.unsqueeze(1)
    if latent.ndim != 4:
        raise ValueError(f"Context latent must be 4D [C,F,H,W], got {tuple(latent.shape)}")
    return latent


def _latent_frame_count(pixel_frames: int) -> int:
    return max(1, (max(1, int(pixel_frames)) - 1) // 4 + 1)


def build_scail_continuity_init(
    target_shape: tuple[int, int, int, int],
    tail_latent: torch.Tensor,
    overlap_pixel_frames: int,
) -> dict[str, torch.Tensor] | None:
    """SCAIL-style init: prefix latents from prev tail + noise_mask=0 on prefix (locked).

    Matches WanSCAILToVideo / ComfyUI differential diffusion semantics.
    Requires add_noise_to_samples on the high-noise stage.
    """
    tail_latent = _normalize_context_latent_4d(tail_latent)
    c, t_total, h, w = (int(x) for x in target_shape)
    aligned_pixels = wan_align_frame_count(int(overlap_pixel_frames))
    t_tail = min(
        _latent_frame_count(aligned_pixels),
        int(tail_latent.shape[1]),
        t_total,
    )
    if t_tail <= 0:
        return None

    full = torch.zeros(c, t_total, h, w, dtype=tail_latent.dtype)
    full[:, :t_tail] = tail_latent[:, :t_tail].to(full.dtype)
    clean = full.cpu().float()
    samples = clean.unsqueeze(0)

    noise_mask = torch.ones((1, 1, t_total, h, w), dtype=clean.dtype)
    noise_mask[:, :, :t_tail] = 0.0

    return {
        "samples": samples,
        "noise_mask": noise_mask,
        "original_image": clean,
    }


def apply_scail_prefix_to_latent(
    latent: dict[str, Any],
    tail_latent: torch.Tensor,
    overlap_pixel_frames: int,
) -> dict[str, Any]:
    """Official core path: write prev-tail prefix into latent dict (KSampler noise_mask)."""
    tail_latent = _normalize_context_latent_4d(tail_latent)
    samples = latent["samples"]
    if samples.ndim == 4:
        samples = samples.unsqueeze(0)
    _, c, t_total, h, w = samples.shape
    aligned_pixels = wan_align_frame_count(int(overlap_pixel_frames))
    t_tail = min(
        int(tail_latent.shape[1]),
        _latent_frame_count(aligned_pixels),
        t_total,
    )
    if t_tail <= 0:
        return latent

    out = dict(latent)
    patched = samples.clone()
    patched[0, :, :t_tail] = tail_latent[:, :t_tail].to(patched.dtype)
    noise_mask = torch.ones((1, 1, t_total, h, w), dtype=patched.dtype)
    noise_mask[:, :, :t_tail] = 0.0
    out["samples"] = patched
    out["noise_mask"] = noise_mask
    return out


def _context_latents_from_conditioning(conditioning) -> list[torch.Tensor]:
    for _tensor, payload in conditioning or []:
        if isinstance(payload, dict):
            streams = payload.get("context_latents")
            if streams:
                return list(streams)
    return []


def apply_continuity_to_core_conditioning(
    positive,
    negative,
    *,
    tail_latent: torch.Tensor,
    prev_output: torch.Tensor,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    n_frames: int = 1,
    tiled_vae: bool = False,
    force_offload: bool = True,
):
    """Insert tail context + ref streams into ComfyUI core conditioning."""
    import node_helpers

    streams = _context_latents_from_conditioning(positive)
    fake = {"context_latents": streams}
    fake = insert_tail_context_after_source(fake, tail_latent)
    fake = append_tail_reference_latents(
        fake,
        prev_output,
        vae=vae,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        n_frames=n_frames,
        tiled_vae=tiled_vae,
        force_offload=force_offload,
        comfy_core=True,
    )
    payload = {"context_latents": fake["context_latents"]}
    positive = node_helpers.conditioning_set_values(positive, payload)
    negative = node_helpers.conditioning_set_values(negative, payload)
    return positive, negative


def merge_continuity_context_streams(
    image_embeds: dict[str, Any],
    *,
    prev_output: torch.Tensor,
    tail_latent: torch.Tensor,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    ref_frames: int,
    tiled_vae: bool = False,
    force_offload: bool = True,
    comfy_core: bool = False,
) -> dict[str, Any]:
    """source → 1× appearance anchor → motion tail → tail ref(s); gen-only, no pixel blend."""
    anchor_latents: list[torch.Tensor] = []
    if APPEARANCE_ANCHOR_DUPLICATES > 0:
        anchor_latents = encode_appearance_anchor_latents(
            prev_output,
            vae=vae,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            duplicates=APPEARANCE_ANCHOR_DUPLICATES,
            tiled_vae=tiled_vae,
            force_offload=force_offload,
            comfy_core=comfy_core,
        )
    if anchor_latents:
        merged = insert_continuity_streams_after_source(
            image_embeds,
            anchor_latents=anchor_latents,
            motion_tail_latent=tail_latent,
        )
    else:
        merged = insert_tail_context_after_source(image_embeds, tail_latent)
    return append_tail_reference_latents(
        merged,
        prev_output,
        vae=vae,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        n_frames=ref_frames,
        tiled_vae=tiled_vae,
        force_offload=force_offload,
        comfy_core=comfy_core,
    )


def insert_tail_context_after_source(
    image_embeds: dict[str, Any],
    tail_latent: torch.Tensor,
) -> dict[str, Any]:
    """Insert prev-tail latent stream immediately after source (better rv2v handoff than prepend)."""
    merged = dict(image_embeds)
    tail = _normalize_context_latent_4d(tail_latent)
    streams = list(merged.get("context_latents") or [])
    if not streams:
        merged["context_latents"] = [tail]
        return merged
    merged["context_latents"] = [streams[0], tail, *streams[1:]]
    return merged


def insert_continuity_streams_after_source(
    image_embeds: dict[str, Any],
    *,
    anchor_latents: list[torch.Tensor],
    motion_tail_latent: torch.Tensor,
) -> dict[str, Any]:
    """Order: source → appearance anchor ref(s) → motion tail (detail lock before motion hint)."""
    merged = dict(image_embeds)
    motion = _normalize_context_latent_4d(motion_tail_latent)
    anchors = [_normalize_context_latent_4d(lat) for lat in anchor_latents if lat is not None]
    streams = list(merged.get("context_latents") or [])
    if not streams:
        merged["context_latents"] = [*anchors, motion]
        return merged
    merged["context_latents"] = [streams[0], *anchors, motion, *streams[1:]]
    return merged


def _encode_reference_latent(
    frame: torch.Tensor,
    *,
    vae,
    ref_max_size: int,
    tiled_vae: bool = False,
    force_offload: bool = True,
    comfy_core: bool = False,
) -> torch.Tensor:
    if comfy_core:
        encoder = ComfyCoreEncoder(vae)
    else:
        encoder = WanVaeEncoder(vae, tiled=tiled_vae, force_offload=force_offload)
    lat = encoder.encode_reference_image(frame, ref_max_size)
    encoder.offload()
    if lat.ndim == 5:
        lat = lat.squeeze(0)
    if lat.ndim == 3:
        lat = lat.unsqueeze(1)
    return lat


def encode_appearance_anchor_latents(
    prev_output: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    duplicates: int = APPEARANCE_ANCHOR_DUPLICATES,
    tiled_vae: bool = False,
    force_offload: bool = True,
    comfy_core: bool = False,
) -> list[torch.Tensor]:
    """Last-frame reference latent(s) for clothes/hair appearance lock."""
    if int(prev_output.shape[0]) <= 0 or int(duplicates) <= 0:
        return []
    frame = fit_canvas(prev_output[-1:], width, height)
    latent = _encode_reference_latent(
        frame,
        vae=vae,
        ref_max_size=ref_max_size,
        tiled_vae=tiled_vae,
        force_offload=force_offload,
        comfy_core=comfy_core,
    )
    return [latent] * int(duplicates)


def _match_frame_color_statistics(
    frame: torch.Tensor,
    reference: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Per-channel mean/std match toward reference (texture-preserving tone cohesion)."""
    s = max(0.0, min(1.0, float(strength)))
    if s <= 0.0:
        return frame
    out = frame.clone().float()
    ref = reference.float()
    for c in range(int(out.shape[-1])):
        r_mean = ref[..., c].mean()
        r_std = ref[..., c].std().clamp(min=1e-6)
        f_mean = out[..., c].mean()
        f_std = out[..., c].std().clamp(min=1e-6)
        matched = (out[..., c] - f_mean) / f_std * r_std + r_mean
        out[..., c] = out[..., c] * (1.0 - s) + matched * s
    return out.clamp(0.0, 1.0).to(dtype=frame.dtype)


def match_opening_appearance_colors(
    decoded: torch.Tensor,
    prev_output: torch.Tensor,
    n_frames: int,
    *,
    width: int,
    height: int,
    strength: float = APPEARANCE_COLOR_MATCH_STRENGTH,
) -> torch.Tensor:
    """Align opening frames' color statistics to prev segment last frame."""
    n = min(int(n_frames), int(decoded.shape[0]))
    if n <= 0 or int(prev_output.shape[0]) <= 0:
        return decoded
    ref = fit_canvas(prev_output[-1:], width, height)[0].to(
        device=decoded.device, dtype=decoded.dtype
    )
    out = decoded.clone()
    for i in range(n):
        fade = strength * (1.0 - i / (n + 1))
        out[i] = _match_frame_color_statistics(out[i], ref, fade)
    log.info("Segment continuity: color-matched %d opening frame(s)", n)
    return out


def append_tail_reference_latents(
    image_embeds: dict[str, Any],
    prev_output: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
    ref_max_size: int,
    n_frames: int = 1,
    tiled_vae: bool = False,
    force_offload: bool = True,
    comfy_core: bool = False,
) -> dict[str, Any]:
    """Encode last N prev-tail pixels as reference-image latents (appearance lock)."""
    n = min(int(n_frames), int(prev_output.shape[0]), MAX_CONTINUITY_REF_FRAMES)
    if n <= 0:
        return image_embeds
    tail_clip = fit_canvas(prev_output[-n:], width, height)
    ref_latents: list[torch.Tensor] = []
    for i in range(n):
        lat = _encode_reference_latent(
            tail_clip[i : i + 1],
            vae=vae,
            ref_max_size=ref_max_size,
            tiled_vae=tiled_vae,
            force_offload=force_offload,
            comfy_core=comfy_core,
        )
        ref_latents.append(lat)
    merged = dict(image_embeds)
    streams = list(merged.get("context_latents") or [])
    merged["context_latents"] = streams + ref_latents
    log.info("Segment continuity: appended %d tail reference latent(s)", n)
    return merged


def blend_continuity_transition(
    decoded: torch.Tensor,
    overlap: int,
    transition_frames: int = CONTINUITY_TRANSITION_FRAMES,
) -> torch.Tensor:
    """Blend frames after the forced prefix toward the last forced frame (smooth appearance)."""
    overlap = int(overlap)
    if overlap <= 0 or overlap > int(decoded.shape[0]):
        return decoded
    n = min(int(transition_frames), int(decoded.shape[0]) - overlap)
    if n <= 0:
        return decoded
    out = decoded.clone()
    anchor = out[overlap - 1]
    for i in range(n):
        t = (i + 1) / (n + 1)
        out[overlap + i] = anchor * (1.0 - t) + out[overlap + i] * t
    log.info(
        "Segment continuity: blended %d transition frame(s) after %d forced prefix",
        n,
        overlap,
    )
    return out


def soft_blend_segment_opening(
    decoded: torch.Tensor,
    prev_output: torch.Tensor,
    blend_frames: int,
    *,
    width: int,
    height: int,
    pull: float = OPENING_BLEND_PULL,
) -> torch.Tensor:
    """Light post-decode pull on segment opening toward prev last frame (not a hard lock)."""
    n = min(int(blend_frames), int(decoded.shape[0]), int(prev_output.shape[0]))
    if n <= 0:
        return decoded
    anchor = fit_canvas(prev_output[-1:], width, height)[0].to(
        device=decoded.device, dtype=decoded.dtype
    )
    out = decoded.clone()
    for i in range(n):
        t = (n - i) / (n + 1)
        w = float(pull) * t
        out[i] = anchor * w + out[i] * (1.0 - w)
    log.info("Segment continuity: soft opening blend on %d frame(s)", n)
    return out


def _crossfade_merge_seam(
    merged: torch.Tensor,
    new_part: torch.Tensor,
    blend_frames: int,
    *,
    color_match_frames: int = 0,
    color_strength: float = APPEARANCE_COLOR_MATCH_STRENGTH,
) -> torch.Tensor:
    """Crossfade at segment boundary; optional color match on seam region."""
    if blend_frames <= 0 or new_part.shape[0] == 0:
        return cat_frames_variable_size([merged, new_part])
    combined = cat_frames_variable_size([merged, new_part])
    seam = int(merged.shape[0]) - 1
    n = min(int(blend_frames), int(new_part.shape[0]))
    if seam < 0:
        return combined
    for i in range(n):
        t = (i + 1) / (n + 1)
        idx = seam + 1 + i
        combined[idx] = combined[seam] * (1.0 - t) + combined[idx] * t
    if color_match_frames > 0 and n > 0:
        ref = combined[seam]
        cm = min(int(color_match_frames), n)
        for i in range(cm):
            idx = seam + 1 + i
            fade = color_strength * (1.0 - i / (cm + 1))
            combined[idx] = _match_frame_color_statistics(combined[idx], ref, fade)
    return combined


def align_segment_boundary_frame(
    segment: torch.Tensor,
    prev_segment: torch.Tensor,
) -> torch.Tensor:
    """Lock only the first frame of a segment to the previous segment's last frame (timeline stitch)."""
    if int(segment.shape[0]) <= 0 or int(prev_segment.shape[0]) <= 0:
        return segment
    out = segment.clone()
    out[0] = prev_segment[-1].to(device=out.device, dtype=out.dtype)
    return out


def force_prev_tail_pixels(
    decoded: torch.Tensor,
    prev_output: torch.Tensor,
    overlap_frames: int,
    *,
    width: int,
    height: int,
) -> tuple[torch.Tensor, int]:
    """Hard-splice prev segment tail pixels onto this segment prefix (guaranteed visual continuity)."""
    overlap = min(int(overlap_frames), int(decoded.shape[0]), int(prev_output.shape[0]))
    if overlap <= 0:
        return decoded, 0
    tail = fit_canvas(prev_output[-overlap:], width, height)
    tail = tail.to(device=decoded.device, dtype=decoded.dtype)
    out = decoded.clone()
    out[:overlap] = tail
    log.info(
        "Segment continuity: forced %d prefix frame(s) from prev output (%dx%d)",
        overlap,
        int(tail.shape[1]),
        int(tail.shape[2]),
    )
    return out, overlap


def continuity_merged_frame_count(plan: DirectorPlan) -> int:
    """Director segments are contiguous on the timeline — export keeps all segment frames."""
    return int(plan.total_frames)


def concat_continuous_chunks(
    chunks: list[torch.Tensor],
    segments: list[SegmentPlan],
    plan: DirectorPlan,
) -> torch.Tensor:
    """Concatenate full segments; plain join (no seam pixel crossfade)."""
    if not chunks:
        raise ValueError("concat_continuous_chunks: no chunks")
    if not plan.continuity_enabled or len(chunks) <= 1:
        return cat_frames_variable_size(chunks)

    merged = chunks[0]
    for seg, chunk in zip(segments[1:], chunks[1:]):
        if seg.index <= 0:
            merged = cat_frames_variable_size([merged, chunk])
            continue
        merged = cat_frames_variable_size([merged, chunk])
        log.info(
            "Segment continuity merge: seg #%d +%d frame(s), plain concat",
            seg.index + 1,
            int(chunk.shape[0]),
        )
    return merged


def apply_cached_segment_continuity(
    chunk: torch.Tensor,
    seg: SegmentPlan,
    plan: DirectorPlan,
    completed_outputs: dict[int, torch.Tensor],
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """No per-chunk rewrite when loading cache; seam alignment happens at merge."""
    del seg, plan, completed_outputs, width, height
    return chunk


def trim_chunks_for_continuity_export(
    chunks: list[torch.Tensor],
    segments: list[SegmentPlan],
    plan: DirectorPlan,
) -> list[torch.Tensor]:
    """Drop overlap prefix frames when merging (SCAIL GetImageRangeFromBatch start=N)."""
    if not plan.continuity_enabled or not chunks:
        return chunks
    overlap = int(plan.continuity_overlap_frames)
    if overlap <= 0:
        return chunks
    trimmed: list[torch.Tensor] = []
    for seg, chunk in zip(segments, chunks):
        if seg.index > 0 and int(chunk.shape[0]) > overlap:
            trimmed.append(chunk[overlap:])
        else:
            trimmed.append(chunk)
    return trimmed


def encode_tail_clip(
    tail_clip: torch.Tensor,
    *,
    vae,
    width: int,
    height: int,
    tiled_vae: bool = False,
    force_offload: bool = True,
    comfy_core: bool = False,
) -> torch.Tensor:
    if comfy_core:
        encoder = ComfyCoreEncoder(vae)
    else:
        encoder = WanVaeEncoder(vae, tiled=tiled_vae, force_offload=force_offload)
    aligned = wan_align_frame_count(int(tail_clip.shape[0]))
    clip = fit_canvas(tail_clip, width, height)
    if int(clip.shape[0]) < aligned:
        pad = clip[-1:].repeat(aligned - int(clip.shape[0]), 1, 1, 1)
        clip = torch.cat([clip, pad], dim=0)
    latent = encoder.encode_source_video(clip, width, height, aligned)
    encoder.offload()
    return _normalize_context_latent_4d(latent)


def apply_scail_continuity(
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    prev_output: torch.Tensor | None,
    vae,
    width: int,
    height: int,
    ref_max_size: int = 848,
    target_shape: tuple[int, int, int, int] | None = None,
    image_embeds: dict[str, Any] | None = None,
    latent: dict[str, Any] | None = None,
    tiled_vae: bool = False,
    force_offload: bool = True,
    comfy_core: bool = False,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """SCAIL latent prefix lock + light context/ref streams from prev tail."""
    del latent
    if not _continuity_active(plan, seg) or prev_output is None:
        return None, image_embeds, None, None

    lock_px = min(
        resolve_continuity_lock_pixels(plan.continuity_overlap_frames),
        int(prev_output.shape[0]),
    )
    if lock_px <= 0:
        return None, image_embeds, None, None

    _, ref_frames, seam_blend, _, _ = resolve_continuity_guide_frames(
        plan.continuity_overlap_frames
    )

    tail_clip = fit_canvas(prev_output[-lock_px:], width, height)
    tail_latent = encode_tail_clip(
        tail_clip,
        vae=vae,
        width=width,
        height=height,
        tiled_vae=tiled_vae,
        force_offload=force_offload,
        comfy_core=comfy_core,
    )

    scail_init = None
    if target_shape is not None:
        scail_init = build_scail_continuity_init(
            target_shape,
            tail_latent,
            lock_px,
        )
        if scail_init is not None:
            t_lock = _latent_frame_count(wan_align_frame_count(lock_px))
            log.info(
                "Segment continuity seg #%d: SCAIL lock %df (%d latent prefix)",
                seg.index + 1,
                lock_px,
                t_lock,
            )

    if image_embeds is not None:
        image_embeds = insert_tail_context_after_source(image_embeds, tail_latent)
        image_embeds = append_tail_reference_latents(
            image_embeds,
            prev_output,
            vae=vae,
            width=width,
            height=height,
            ref_max_size=ref_max_size,
            n_frames=ref_frames,
            tiled_vae=tiled_vae,
            force_offload=force_offload,
            comfy_core=comfy_core,
        )
        stream_count = len(image_embeds.get("context_latents") or [])
        log.info(
            "Segment continuity seg #%d: + %d ref, %d context stream(s), seam=%d",
            seg.index + 1,
            ref_frames,
            stream_count,
            seam_blend,
        )

    t_lock = _latent_frame_count(wan_align_frame_count(lock_px))
    note = (
        f"  continuity seg #{seg.index + 1}: SCAIL lock {lock_px}f ({t_lock} latent) "
        f"+ {ref_frames}f ref (overlap {plan.continuity_overlap_frames})"
    )
    return scail_init, image_embeds, None, note


def apply_scail_continuity_core(
    *,
    plan: DirectorPlan,
    seg: SegmentPlan,
    prev_output: torch.Tensor | None,
    positive,
    negative,
    vae,
    width: int,
    height: int,
    ref_max_size: int = 848,
    latent: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any] | None, str | None]:
    """Core executor: SCAIL latent prefix + conditioning context/ref streams."""
    if not _continuity_active(plan, seg) or prev_output is None:
        return positive, negative, latent, None

    lock_px = min(
        resolve_continuity_lock_pixels(plan.continuity_overlap_frames),
        int(prev_output.shape[0]),
    )
    if lock_px <= 0:
        return positive, negative, latent, None

    _, ref_frames, _seam, _opening, _color = resolve_continuity_guide_frames(
        plan.continuity_overlap_frames
    )

    tail_clip = fit_canvas(prev_output[-lock_px:], width, height)
    tail_latent = encode_tail_clip(
        tail_clip,
        vae=vae,
        width=width,
        height=height,
        comfy_core=True,
    )
    positive, negative = apply_continuity_to_core_conditioning(
        positive,
        negative,
        tail_latent=tail_latent,
        prev_output=prev_output,
        vae=vae,
        width=width,
        height=height,
        ref_max_size=ref_max_size,
        n_frames=ref_frames,
    )
    if latent is not None:
        latent = apply_scail_prefix_to_latent(latent, tail_latent, lock_px)
    t_lock = _latent_frame_count(wan_align_frame_count(lock_px))
    note = (
        f"  continuity seg #{seg.index + 1}: SCAIL lock {lock_px}f ({t_lock} latent) "
        f"+ {ref_frames}f ref (overlap {plan.continuity_overlap_frames})"
    )
    return positive, negative, latent, note
