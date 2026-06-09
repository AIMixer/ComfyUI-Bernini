"""Detect when the Bernini runtime sampler can be used instead of legacy WanVideo."""

from __future__ import annotations

from typing import Any, Mapping

# image_embeds keys that imply non-Bernini WanVideo features
_IMAGE_EMBED_BLOCKERS = frozenset({
    "image_embeds",
    "vace_context",
    "control_embeds",
    "recammaster",
    "mocha_embeds",
    "phantom_latents",
    "wananim_pose_latents",
    "wananim_face_pixels",
    "multitalk_sampling",
    "story_mem_latents",
    "lynx_embeds",
    "standin_input",
    "add_cond_latents",
    "qwenvl_embeds_pos",
    "qwenvl_embeds_neg",
    "flashvsr_LQ_latent",
    "extra_latents",
    "framepack",
    "pusa_noisy_steps",
})

# extra_args / process kwargs that force the legacy sampler path
_SAMPLER_BLOCKERS = frozenset({
    "unianimate_poses",
    "fantasytalking_embeds",
    "uni3c_embeds",
    "multitalk_embeds",
    "loop_args",
    "experimental_args",
    "freeinit_args",
    "flowedit_args",
    "feta_args",
})


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    if isinstance(value, bool):
        return value
    return True


def bernini_runtime_eligible(
    image_embeds: Mapping[str, Any] | None,
    *,
    samples=None,
    context_options=None,
    transformer=None,
    **sampler_kwargs: Any,
) -> bool:
    """Return True when Bernini context + standard CFG/APG can use the slim denoise loop."""
    embeds = image_embeds or {}

    if embeds.get("bernini_pipeline") is False:
        return False

    if not embeds.get("target_shape") and not embeds.get("context_latents"):
        return False

    for key in _IMAGE_EMBED_BLOCKERS:
        if _truthy(embeds.get(key)):
            return False

    for key in _SAMPLER_BLOCKERS:
        if _truthy(sampler_kwargs.get(key)):
            return False

    if _truthy(sampler_kwargs.get("multitalk_sampling")):
        return False

    if transformer is not None:
        if getattr(transformer, "audio_model", None) is not None:
            return False
        if getattr(transformer, "multitalk_model_type", None):
            return False

    # Classic I2V image_cond path (not Bernini semantic context)
    if _truthy(embeds.get("image_embeds")):
        return False

    return True
