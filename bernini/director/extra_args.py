"""Merge Bernini Director sampler options into WANVIDSAMPLEREXTRAARGS dicts."""

from __future__ import annotations

from typing import Any, Mapping


def merge_sampler_extra_args(
    extra_args: Mapping[str, Any] | None,
    *,
    enable_teacache: bool,
) -> dict[str, Any]:
    """Apply Director TeaCache toggle unless extra_args already sets cache/teacache."""
    merged = dict(extra_args or {})
    if "cache_args" in merged or "teacache_args" in merged:
        return merged
    merged["enable_teacache"] = enable_teacache
    return merged
