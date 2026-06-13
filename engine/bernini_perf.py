"""Bernini runtime performance defaults (attention, cache, block swap, LoRA merge)."""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

log = logging.getLogger(__name__)


def resolve_attention_mode(requested: str) -> str:
    """Map ``auto`` to the best available attention backend for this GPU."""
    if requested not in ("auto", "bernini_auto"):
        return requested

    try:
        import importlib.util

        import torch

        if not torch.cuda.is_available():
            return "sdpa"
        if importlib.util.find_spec("sageattention") is None:
            return "sdpa"
        major, _minor = torch.cuda.get_device_capability()
        if major >= 10:
            return "sageattn_3"
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attn_2"
        return "sageattn"
    except Exception:
        return "sageattn"


def prefer_merged_loras(lora: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Prefer merged static LoRAs (distill adapters) unless low-mem load is requested."""
    if not lora:
        return lora
    merged: list[dict[str, Any]] = []
    for entry in lora:
        item = dict(entry)
        if not item.get("low_mem_load", False):
            item["merge_loras"] = True
        merged.append(item)
    return merged


def default_teacache_args(timesteps) -> dict[str, Any]:
    """TeaCache defaults when explicitly enabled (Director toggle or cache_args)."""
    return {
        "cache_type": "TeaCache",
        "cache_device": "cpu",
        "rel_l1_thresh": 0.15,
        "start_step": 0,
        "end_step": -1,
        "use_coefficients": True,
        "mode": "e",
    }


def auto_batched_cfg(
    text_embeds: Mapping[str, Any],
    cfg: list[float],
    image_embeds: Mapping[str, Any] | None,
) -> bool:
    """Enable batched cond/uncond when CFG>1 and NAG is not active."""
    if not image_embeds or not image_embeds.get("bernini_pipeline"):
        return False
    if text_embeds.get("nag_prompt_embeds"):
        return False
    prompt = text_embeds.get("prompt_embeds") or []
    negative = text_embeds.get("negative_prompt_embeds") or []
    if len(prompt) != 1 or len(negative) != 1:
        return False
    return any(not math.isclose(float(c), 1.0) for c in cfg)
