"""Bernini-branded text encoding nodes."""

from __future__ import annotations

from ...engine.bernini_core_nodes import WanVideoTextEncodeCached
from ..task_prompts import apply_task_system_prompt, task_type_combo_options

_CATEGORY = "Bernini"


class BerniniTextEncodeCached(WanVideoTextEncodeCached):
    """T5 encode with Bernini task_type system prompt injection."""

    CATEGORY = _CATEGORY
    DESCRIPTION = """Encodes text prompts into T5 embeddings with Bernini task presets.

Select **task_type** to automatically prepend the matching Bernini system prompt
to **positive_prompt** — no need to copy system prompts manually.

Loads and unloads T5 after encoding; disk cache skips T5 reload when prompts match.
"""

    @classmethod
    def INPUT_TYPES(cls):
        base = WanVideoTextEncodeCached.INPUT_TYPES()
        required = dict(base["required"])
        ordered_required = {}

        for name, widget in required.items():
            if name == "positive_prompt":
                combo_options, combo_meta = task_type_combo_options()
                ordered_required["task_type"] = (combo_options, combo_meta)
            ordered_required[name] = widget

        return {
            "required": ordered_required,
            "optional": base.get("optional", {}),
        }

    def process(
        self,
        model_name,
        precision,
        task_type,
        positive_prompt,
        negative_prompt,
        quantization="disabled",
        use_disk_cache=True,
        device="gpu",
        extender_args=None,
        **kwargs,
    ):
        if extender_args is not None:
            from ...engine.utils import log
            log.warning(
                "BerniniTextEncodeCached: extender_args is ignored "
                "(Qwen prompt extender is not supported in ComfyUI-Bernini)."
            )
        positive_prompt = apply_task_system_prompt(task_type, positive_prompt)
        return super().process(
            model_name=model_name,
            precision=precision,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            quantization=quantization,
            use_disk_cache=use_disk_cache,
            device=device,
        )
