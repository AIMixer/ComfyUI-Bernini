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


def tune_block_swap_args(block_swap_args: dict[str, Any] | None) -> dict[str, Any] | None:
    """Enable non-blocking transfer + one-block prefetch when swap is active."""
    if not block_swap_args:
        return block_swap_args
    tuned = dict(block_swap_args)
    if int(tuned.get("blocks_to_swap", 0)) <= 0:
        return tuned
    if int(tuned.get("prefetch_blocks", 0)) == 0:
        tuned["prefetch_blocks"] = 1
    if not tuned.get("use_non_blocking", False):
        tuned["use_non_blocking"] = True
    return tuned


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
    """Conservative TeaCache defaults for Bernini Wan2.2 4+4 step schedules."""
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


def apply_block_swap_tuning(transformer_options: dict[str, Any] | None) -> None:
    if not transformer_options:
        return
    bsa = transformer_options.get("block_swap_args")
    if bsa is None:
        return
    tuned = tune_block_swap_args(bsa)
    if tuned != bsa:
        transformer_options["block_swap_args"] = tuned
        log.info(
            "Bernini perf: block swap tuned (prefetch_blocks=%s, use_non_blocking=%s)",
            tuned.get("prefetch_blocks"),
            tuned.get("use_non_blocking"),
        )
