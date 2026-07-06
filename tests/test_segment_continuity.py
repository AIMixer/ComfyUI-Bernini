"""Tests for cross-segment continuity (SCAIL-style latent anchor)."""

from __future__ import annotations

from bernini.director.plan import DirectorPlan, SegmentPlan
from bernini.director.segment_continuity import (
    blend_continuity_transition,
    build_scail_continuity_init,
    concat_continuous_chunks,
    continuity_merged_frame_count,
    force_prev_tail_pixels,
    match_opening_appearance_colors,
    resolve_continuity_guide_frames,
    resolve_continuity_lock_pixels,
    resolve_continuity_settings,
    resolve_segment_generation_frames,
    soft_blend_segment_opening,
)


def _plan(**kwargs) -> DirectorPlan:
    defaults = dict(
        frame_rate=24.0,
        total_frames=100,
        width=64,
        height=64,
        ref_max_size=64,
        output_mode="fixed",
        source_width=64,
        source_height=64,
        global_task_type="rv2v",
        global_task_key="rv2v",
        global_prompt="",
        global_refs=[],
        segments=[],
        source_video=None,
        edit_mode="segment",
        raw={},
        continuity_enabled=True,
        continuity_overlap_frames=5,
    )
    defaults.update(kwargs)
    return DirectorPlan(**defaults)


def _seg(index: int, *, frames: int = 20) -> SegmentPlan:
    return SegmentPlan(
        index=index,
        start_frame=index * frames,
        end_frame=(index + 1) * frames,
        task_key="rv2v",
        task_type="rv2v",
        prompt="test",
        use_global=True,
    )


def test_resolve_continuity_guide_frames_default_overlap():
    assert resolve_continuity_guide_frames(9) == (1, 1, 0, 0, 0)
    assert resolve_continuity_guide_frames(13) == (1, 1, 0, 0, 0)
    assert resolve_continuity_guide_frames(5) == (1, 1, 0, 0, 0)


def test_resolve_continuity_lock_pixels_uses_full_overlap():
    assert resolve_continuity_lock_pixels(13) == 13
    assert resolve_continuity_lock_pixels(9) == 9
    assert resolve_continuity_lock_pixels(10) == 13


def test_resolve_segment_generation_frames_adds_overlap_for_later_segments():
    frames, trim = resolve_segment_generation_frames(
        segment_frame_count=34,
        segment_index=0,
        continuity_enabled=True,
        continuity_overlap=13,
    )
    assert trim == 0
    assert frames == 37

    frames2, trim2 = resolve_segment_generation_frames(
        segment_frame_count=34,
        segment_index=1,
        continuity_enabled=True,
        continuity_overlap=13,
    )
    assert trim2 == 13
    assert frames2 == 49


def test_concat_continuous_chunks_plain_join():
    import torch

    plan = _plan(continuity_enabled=True, continuity_overlap_frames=9)
    segs = [_seg(0), _seg(1)]
    seg1 = torch.zeros(10, 4, 4, 3)
    seg1[-1] = 2.0
    seg2 = torch.ones(8, 4, 4, 3)
    merged = concat_continuous_chunks([seg1, seg2], segs, plan)
    assert merged.shape[0] == 18
    assert torch.all(merged[9] == 1.0)
    assert torch.all(merged[8] == 2.0)


def test_match_opening_appearance_colors_noop_when_disabled():
    import torch

    prev = torch.zeros(2, 4, 4, 3)
    decoded = torch.full((3, 4, 4, 3), 0.5)
    out = match_opening_appearance_colors(decoded, prev, 0, width=4, height=4)
    assert torch.all(out == decoded)


def test_soft_blend_segment_opening():
    import torch

    prev = torch.zeros(4, 2, 2, 3)
    prev[-1] = 1.0
    decoded = torch.full((4, 2, 2, 3), 0.5)
    out = soft_blend_segment_opening(decoded, prev, 2, width=2, height=2)
    assert out[0, 0, 0, 0] < 1.0
    assert out[0, 0, 0, 0] > 0.5
    assert torch.all(out[2:] == 0.5)


def test_resolve_continuity_default_off():
    timeline = {"output": {}}
    enabled, overlap = resolve_continuity_settings(timeline, segment_count=2)
    assert enabled is False
    assert overlap == 0


def test_resolve_continuity_ignores_timeline_root_flag():
    timeline = {"continuityEnabled": True, "output": {}}
    enabled, overlap = resolve_continuity_settings(timeline, segment_count=2)
    assert enabled is False
    assert overlap == 0


def test_resolve_continuity_disabled_for_single_segment_with_flag():
    enabled, overlap = resolve_continuity_settings(timeline, segment_count=1)
    assert enabled is False
    assert overlap == 0


def test_build_scail_continuity_init_prefix_and_mask():
    import torch

    tail = torch.ones(16, 2, 8, 8)
    init = build_scail_continuity_init((16, 5, 8, 8), tail, overlap_pixel_frames=5)
    assert init is not None
    assert init["samples"].shape == (1, 16, 5, 8, 8)
    assert init["noise_mask"].shape == (1, 1, 5, 8, 8)
    assert torch.all(init["noise_mask"][:, :, :2] == 0.0)
    assert torch.all(init["noise_mask"][:, :, 2:] == 1.0)
    assert torch.all(init["samples"][0, :, :2] == 1.0)
    assert torch.all(init["samples"][0, :, 2:] == 0.0)


def test_concat_continuous_chunks_keeps_all_frames():
    import torch

    plan = _plan(continuity_enabled=True, continuity_overlap_frames=9)
    segs = [_seg(0), _seg(1)]
    seg1 = torch.zeros(10, 4, 4, 3)
    seg1[-1] = 2.0
    seg2 = torch.ones(8, 4, 4, 3)
    merged = concat_continuous_chunks([seg1, seg2], segs, plan)
    assert merged.shape[0] == 18
    assert torch.all(merged[9] == 1.0)
    assert torch.all(merged[10] == 1.0)


def test_continuity_merged_frame_count():
    plan = _plan(
        continuity_enabled=True,
        continuity_overlap_frames=9,
        total_frames=69,
        segments=[_seg(0), _seg(1)],
    )
    assert continuity_merged_frame_count(plan) == 69


def test_blend_continuity_transition():
    import torch

    decoded = torch.zeros(10, 2, 2, 3)
    decoded[5:] = 1.0
    out = blend_continuity_transition(decoded, overlap=5, transition_frames=3)
    assert torch.all(out[:5] == 0.0)
    assert out[5, 0, 0, 0] < 1.0
    assert out[7, 0, 0, 0] > out[5, 0, 0, 0]


def test_force_prev_tail_pixels():
    import torch

    prev = torch.zeros(10, 4, 4, 3)
    prev[-3:] = 1.0
    decoded = torch.full((8, 4, 4, 3), 0.5)
    out, n = force_prev_tail_pixels(decoded, prev, 5, width=4, height=4)
    assert n == 3
    assert torch.all(out[:3] == 1.0)
    assert torch.all(out[3:] == 0.5)
