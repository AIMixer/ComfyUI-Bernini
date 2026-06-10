"""Release GPU memory between Bernini Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-Bernini.director.vram")


def cleanup_segment_vram(*, enabled: bool = True) -> None:
    """Release segment GPU memory: gc, unload ComfyUI models, empty CUDA cache."""
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    log.debug("Bernini Director: segment VRAM cleanup (models unloaded, cache cleared)")
